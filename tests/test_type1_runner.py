import importlib.util
import sys
import unittest
import subprocess
import tempfile
import json
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "scripts/run_type1.py"


class Type1RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.exists(), "Type 1 attack runner is missing")
        sys.path.insert(0, str(PATH.parent))
        spec = importlib.util.spec_from_file_location("run_type1", PATH)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_alignment_checks_text_label_and_source_hash(self):
        import hashlib
        rows = [{"text": "unchanged text", "label": 1}]
        metadata = [{"row": "0", "source_id": "9", "label_name": "positive",
            "text_sha256": hashlib.sha256(b"unchanged text").hexdigest()}]
        self.module.validate_alignment(rows, metadata, {"negative": 0, "positive": 1})
        with self.assertRaises(ValueError):
            self.module.validate_alignment([{"text": "edited text", "label": 1}], metadata,
                                           {"negative": 0, "positive": 1})
        with self.assertRaises(ValueError):
            self.module.validate_alignment([{"text": "unchanged text", "label": 0}], metadata,
                                           {"negative": 0, "positive": 1})

    def test_metrics_use_all_examples_and_eligible_attempts(self):
        rows = [
            {"category": "a", "label": 0, "original_prediction": 0, "perturbed_prediction": 1,
             "result_type": "Successful", "num_queries": 10, "original_word_count": 10, "changed_word_count": 2},
            {"category": "b", "label": 1, "original_prediction": 1, "perturbed_prediction": 1,
             "result_type": "Failed", "num_queries": 20, "original_word_count": 10, "changed_word_count": 3},
            {"category": "b", "label": 1, "original_prediction": 0, "perturbed_prediction": 0,
             "result_type": "Skipped", "num_queries": 1, "original_word_count": 10, "changed_word_count": 0}]
        metrics = self.module.attack_metrics(rows)
        self.assertEqual(metrics["examples"], 3)
        self.assertAlmostEqual(metrics["original_accuracy_percent"], 200 / 3)
        self.assertAlmostEqual(metrics["after_attack_accuracy_percent"], 100 / 3)
        self.assertEqual(metrics["attack_success_rate_percent"], 50)
        self.assertEqual(metrics["mean_queries_eligible"], 15)
        self.assertEqual(metrics["mean_perturbed_word_percent_successful"], 20)

    def test_all_skipped_and_no_success_have_unavailable_conditional_metrics(self):
        row = {"category": "x", "label": 0, "original_prediction": 1, "perturbed_prediction": 1,
               "result_type": "Skipped", "num_queries": 1, "original_word_count": 3, "changed_word_count": 0}
        metrics = self.module.attack_metrics([row])
        self.assertIsNone(metrics["attack_success_rate_percent"])
        self.assertIsNone(metrics["mean_perturbed_word_percent_successful"])

    def test_inconsistent_attack_outcome_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module.attack_metrics([{"category": "x", "label": 0, "original_prediction": 1,
                "perturbed_prediction": 0, "result_type": "Successful", "num_queries": 10,
                "original_word_count": 3, "changed_word_count": 1}])

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            run = subprocess.run([sys.executable, str(PATH), "--dataset-dir", "missing",
                                  "--output-dir", directory], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("already exists", run.stderr)

    def test_textattack_text_log_preserves_unicode(self):
        from type1_models import configure_cache
        configure_cache(PATH.parents[1])
        from textattack.loggers import AttackLogManager

        class Result:
            def __str__(self, color_method=None):
                return "\ufeffA film about café life — 电影"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attack_results.txt"
            manager = AttackLogManager(metrics=None)
            self.module.add_utf8_text_log(manager, path)
            manager.log_result(Result())
            manager.flush()
            manager.loggers[0].close()
            self.assertIn(str(Result()), path.read_text(encoding="utf-8"))

    def test_analysis_rejects_partial_outcomes_before_semantic_encoding(self):
        script = PATH.parent / "analyze_type1.py"
        self.assertTrue(script.exists(), "Type 1 analysis runner is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps({"status": "completed", "examples_planned": 2}), encoding="utf-8")
            (root / "outcomes.csv").write_text("row,category,label,original_prediction,perturbed_prediction,result_type,num_queries,original_word_count,changed_word_count\n0,a,0,0,1,Successful,10,3,1\n", encoding="utf-8")
            run = subprocess.run([sys.executable, str(script), "--run-dir", str(root)],
                                 capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("incomplete", run.stderr)


if __name__ == "__main__":
    unittest.main()
