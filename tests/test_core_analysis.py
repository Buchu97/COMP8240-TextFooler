import ast
import hashlib
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_core_replication.py"


class CoreAnalysisTests(unittest.TestCase):
    def test_fingerprints_bind_analysis_to_csv_and_supporting_artifacts(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "source_fingerprints"]
        self.assertEqual(len(functions), 1, "Analysis fingerprint function is missing")
        module = ast.Module(body=functions, type_ignores=[])
        namespace = {"hashlib": hashlib, "Path": Path}
        exec(compile(module, str(SCRIPT), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "attack_results.csv").write_bytes(b"abc\n")
            (root / "attack_summary.json").write_bytes(b"{}")
            result = namespace["source_fingerprints"](root / "attack_results.csv")
            self.assertEqual(result["source_csv_sha256"], "edeaaff3f1774ad2888673770c6d64097e391bc362d7d6fb34982ddf0efd18cb")
            self.assertEqual(set(result["supporting_artifacts_sha256"]), {"attack_summary.json"})
            (root / "attack_results.csv").write_bytes(b"changed\n")
            changed = namespace["source_fingerprints"](root / "attack_results.csv")
            self.assertNotEqual(result["source_csv_sha256"], changed["source_csv_sha256"])


if __name__ == "__main__":
    unittest.main()
