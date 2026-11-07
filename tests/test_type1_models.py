import importlib.util
import sys
import unittest
import subprocess
import tempfile
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "scripts/type1_models.py"


class Type1ModelTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.exists(), "Type 1 model checks are missing")
        sys.path.insert(0, str(PATH.parent))
        spec = importlib.util.spec_from_file_location("type1_models", PATH)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_label_id_swap_is_rejected_not_assumed_equivalent(self):
        self.module.validate_label_map({"negative": 0, "positive": 1},
                                       {0: "negative", 1: "positive"})
        with self.assertRaisesRegex(ValueError, "label"):
            self.module.validate_label_map({"negative": 0, "positive": 1},
                                           {0: "positive", 1: "negative"})

    def test_duplicate_label_name_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module.validate_label_map({"a": 0, "b": 1}, {0: "a", 1: "a"})

    def test_text_overlap_is_rejected_even_when_case_differs(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.module.validate_split_separation({"train": [{"text": "Same Text"}],
                "validation": [{"text": "different"}], "test": [{"text": "same  text"}]})
        self.module.validate_split_separation({"train": [{"text": "A"}],
            "validation": [{"text": "B"}], "test": [{"text": "C"}]})

    def test_multiclass_baseline_uses_all_examples_and_reports_per_class(self):
        rows = [{"label": 0, "prediction": 0}, {"label": 1, "prediction": 0},
                {"label": 2, "prediction": 2}]
        result = self.module.baseline_metrics(rows, {"negative": 0, "positive": 1, "neutral": 2})
        self.assertEqual(result["examples"], 3)
        self.assertEqual(result["correct"], 2)
        self.assertAlmostEqual(result["accuracy_percent"], 200 / 3)
        self.assertEqual(result["by_label"]["positive"]["correct"], 0)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [1, 0, 0], [0, 0, 1]])

    def test_empty_baseline_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module.baseline_metrics([], {"a": 0})

    def test_news_training_refuses_existing_checkpoint_directory(self):
        script = PATH.parent / "train_news.py"
        self.assertTrue(script.exists(), "News training runner is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            run = subprocess.run([sys.executable, str(script), "--dataset-dir", str(root / "missing"),
                "--model-dir", str(model), "--report-dir", str(root / "report")],
                capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("already exists", run.stderr)
            self.assertFalse((root / "report").exists())


if __name__ == "__main__":
    unittest.main()
