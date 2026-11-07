import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import judge_protocol as protocol


def source_fixture(root):
    run = root / 'attack'
    run.mkdir()
    rows = []
    for i in range(9):
        outcome = 'Successful' if i < 7 else ('Failed' if i == 7 else 'Skipped')
        rows.append({'row': i, 'source_id': str(i), 'category': 'positive', 'label': 1,
                     'original_text': f'Good movie {i}.', 'perturbed_text': f'Fine movie {i}.',
                     'original_prediction': 0 if outcome == 'Skipped' else 1,
                     'perturbed_prediction': 1 if outcome == 'Failed' else 0,
                     'result_type': outcome, 'original_token_truncated': False})
    protocol.write_csv(run / 'outcomes.csv', rows, list(rows[0]))
    protocol.save_json(run / 'run_metadata.json', {'label2id': {'negative': 0, 'positive': 1}})
    protocol.save_json(run / 'summary.json', {'status': 'completed', 'examples_planned': 9,
                       'artifacts_sha256': {'outcomes.csv': protocol.file_digest(run / 'outcomes.csv')}})
    return {'dataset': 'reviews', 'task': 'sentiment', 'format': 'type1', 'run_dir': str(run)}


class JudgeDataTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/judge_data.py'
        self.assertTrue(path.exists(), 'Judge data preparation is missing')
        spec = importlib.util.spec_from_file_location('judge_data', path)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        spec.loader.exec_module(self.module)

    def test_preparation_is_reproducible_blind_and_separates_development_controls_and_final(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            source = source_fixture(root)
            config = {'sources': [source]}
            for name in ('one', 'two'):
                self.module.prepare(config, root / name, development_per_dataset=2, audit_size=3)
            first, second = self.module.load_prepared(root / 'one'), self.module.load_prepared(root / 'two')
            self.assertEqual(first, second)
            manifest, pairs, spaces = first
            phases = {phase: [p for p in pairs if p['phase'] == phase]
                      for phase in ('development', 'final', 'control')}
            self.assertEqual(len(phases['development']), 2)
            self.assertEqual(len(phases['final']), 5)
            self.assertFalse({p['pair_id'] for p in phases['development']} & set(manifest['audit_ids']))
            self.assertEqual(len(manifest['audit_ids']), 3)
            audit = self.module.read_csv(root / 'one/human_review.csv')
            self.assertTrue(all(r['meaning_preserved'] == '' and r['reviewer_id'] == '' for r in audit))
            self.assertNotIn('gold_label', audit[0])
            self.assertNotIn('original_prediction', audit[0])
            self.assertEqual(manifest['sources']['reviews']['eligible'], 8)
            self.assertEqual(manifest['sources']['reviews']['successful'], 7)
            self.assertEqual(spaces['reviews'], protocol.SENTIMENT)
            with self.assertRaises(FileExistsError):
                self.module.prepare(config, root / 'one')

    def test_modified_source_or_prepared_pairs_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            source = source_fixture(root)
            self.module.prepare({'sources': [source]}, root / 'prepared', development_per_dataset=1, audit_size=2)
            pairs_path = root / 'prepared/pairs.json'
            pairs = protocol.read_json(pairs_path)
            pairs[0]['original_text'] = 'A changed record.'
            protocol.save_json(pairs_path, pairs)
            with self.assertRaises(ValueError):
                self.module.load_prepared(root / 'prepared')
            (root / 'attack/outcomes.csv').write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                self.module.read_source(source)

    def test_textattack_markers_and_label_mapping_are_handled_without_binary_relabeling(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            protocol.write_csv(root / 'attack_results.csv', [{'original_text': 'A [[good]] film.',
                'perturbed_text': 'A [[fine]] film.', 'original_output': 1, 'perturbed_output': 0,
                'ground_truth_output': 1, 'result_type': 'Successful'}],
                ['original_text', 'perturbed_text', 'original_output', 'perturbed_output', 'ground_truth_output', 'result_type'])
            source = {'dataset': 'mr', 'task': 'sentiment', 'format': 'textattack', 'run_dir': str(root),
                      'label2id': {'negative': 0, 'positive': 1}}
            pairs, summary, labels = self.module.read_source(source)
            self.assertEqual(pairs[0]['original_text'], 'A good film.')
            self.assertEqual(pairs[0]['perturbed_text'], 'A fine film.')
            self.assertEqual(pairs[0]['gold_label'], 'positive')

    def test_predictions_outside_the_task_taxonomy_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            source = source_fixture(root)
            path = root / 'attack/outcomes.csv'
            rows = self.module.read_csv(path)
            rows[0]['perturbed_prediction'] = '999'
            protocol.write_csv(path, rows, list(rows[0]), exclusive=False)
            summary = protocol.read_json(root / 'attack/summary.json')
            summary['artifacts_sha256']['outcomes.csv'] = protocol.file_digest(path)
            protocol.save_json(root / 'attack/summary.json', summary)
            with self.assertRaises(ValueError):
                self.module.read_source(source)


if __name__ == '__main__':
    unittest.main()
