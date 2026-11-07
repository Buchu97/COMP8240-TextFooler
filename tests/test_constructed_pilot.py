import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/run_constructed_pilot.py"


class ConstructedPilotTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE_PATH.is_file(), "Constructed-data pilot runner is missing")
        spec = importlib.util.spec_from_file_location("constructed_pilot", MODULE_PATH)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_baseline_keeps_incorrect_examples_and_separate_category_counts(self):
        rows = [
            {"category": "plain", "label": 0, "prediction": 0},
            {"category": "plain", "label": 1, "prediction": 0},
            {"category": "sarcasm", "label": 1, "prediction": 1},
        ]
        summary = self.module.summarise_baseline(rows)
        self.assertEqual(summary["examples"], 3)
        self.assertEqual(summary["correct"], 2)
        self.assertAlmostEqual(summary["accuracy_percent"], 200 / 3)
        self.assertEqual(summary["by_category"]["plain"]["accuracy_percent"], 50)
        self.assertEqual(summary["by_category"]["sarcasm"]["accuracy_percent"], 100)

    def test_attack_metrics_use_eligible_denominator_and_successful_perturbations(self):
        rows = [
            {"category": "plain", "label": 0, "original_prediction": 0,
             "perturbed_prediction": 1, "result_type": "Successful", "num_queries": 10,
             "original_word_count": 10, "changed_word_count": 2},
            {"category": "plain", "label": 1, "original_prediction": 1,
             "perturbed_prediction": 1, "result_type": "Failed", "num_queries": 20,
             "original_word_count": 10, "changed_word_count": 3},
            {"category": "sarcasm", "label": 1, "original_prediction": 0,
             "perturbed_prediction": 0, "result_type": "Skipped", "num_queries": 1,
             "original_word_count": 10, "changed_word_count": 0},
        ]
        summary = self.module.summarise_attacks(rows)
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["skipped"], 1)
        self.assertAlmostEqual(summary["original_accuracy_percent"], 200 / 3)
        self.assertAlmostEqual(summary["after_attack_accuracy_percent"], 100 / 3)
        self.assertEqual(summary["attack_success_rate_percent"], 50)
        self.assertEqual(summary["mean_queries_eligible"], 15)
        self.assertEqual(summary["mean_perturbed_word_percent_successful"], 20)
        self.assertIsNone(summary["by_category"]["sarcasm"]["attack_success_rate_percent"])

    def test_all_skipped_has_unavailable_not_zero_attack_success_rate(self):
        rows = [{"category": "negation", "label": 1, "original_prediction": 0,
                 "perturbed_prediction": 0, "result_type": "Skipped", "num_queries": 1,
                 "original_word_count": 10, "changed_word_count": 0}]
        summary = self.module.summarise_attacks(rows)
        self.assertEqual(summary["after_attack_accuracy_percent"], 0)
        self.assertIsNone(summary["attack_success_rate_percent"])
        self.assertIsNone(summary["mean_queries_eligible"])
        self.assertIsNone(summary["mean_perturbed_word_percent_successful"])

    def test_inconsistent_outcome_is_rejected(self):
        row = {"category": "plain", "label": 0, "original_prediction": 1,
               "perturbed_prediction": 0, "result_type": "Successful", "num_queries": 10,
               "original_word_count": 10, "changed_word_count": 2}
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            self.module.summarise_attacks([row])

    def test_arbitrary_checkpoint_override_is_rejected_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH), "--dataset", str(root / "missing.jsonl"),
                 "--output-dir", str(output), "--model-path", str(root / "other_checkpoint")],
                capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("unrecognized arguments", completed.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
