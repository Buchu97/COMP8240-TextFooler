import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "main_table_matrix.py"


def load_module():
    spec = importlib.util.spec_from_file_location("main_table_matrix", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MainTableMatrixTests(unittest.TestCase):
    def test_matrix_represents_all_fifteen_table_three_cells(self):
        module = load_module()
        matrix = module.build_matrix()

        self.assertEqual(len(matrix), 15)
        self.assertEqual(sum(cell.supported for cell in matrix), 12)
        self.assertEqual(
            {(cell.model_family, cell.dataset) for cell in matrix if not cell.supported},
            {("wordcnn", "fake"), ("wordlstm", "fake"), ("bert", "fake")},
        )

    def test_command_uses_paper_delta_attack_and_separate_output_files(self):
        module = load_module()
        cell = next(
            cell
            for cell in module.build_matrix()
            if cell.model_family == "bert" and cell.dataset == "imdb"
        )
        output_dir = PROJECT_ROOT / "outputs" / "test-cell"
        command = module.build_attack_command(
            python_executable=PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
            attack_file=PROJECT_ROOT / "scripts" / "textfooler_paper_delta_0_7.py",
            cell=cell,
            output_dir=output_dir,
            num_examples=1,
            random_seed=8240,
        )

        self.assertIn("bert-base-uncased-imdb", command)
        self.assertIn("--attack-from-file", command)
        self.assertIn("--log-summary-to-json", command)
        self.assertIn(str(output_dir / "attack_summary.json"), command)


if __name__ == "__main__":
    unittest.main()
