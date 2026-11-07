import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import judge_data
import judge_local
import judge_protocol as protocol
from test_judge_data import source_fixture


def task_source(root, name, task, labels, positive):
    root.mkdir()
    source = source_fixture(root)
    source.update(dataset=name, task=task)
    path = root / 'attack/outcomes.csv'
    rows = judge_data.read_csv(path)
    for row in rows:
        row['category'] = positive
    protocol.write_csv(path, rows, list(rows[0]), exclusive=False)
    protocol.save_json(root / 'attack/run_metadata.json', {'label2id': labels})
    summary = protocol.read_json(root / 'attack/summary.json')
    summary['artifacts_sha256']['outcomes.csv'] = protocol.file_digest(path)
    protocol.save_json(root / 'attack/summary.json', summary)
    return source


class Type1JudgeExtensionTests(unittest.TestCase):
    def test_citation_prompt_explains_function_and_hides_predictions(self):
        pair = {'pair_id': 'citation', 'task': 'citation_intent',
                'original_text': 'We used the method of Smith et al.',
                'perturbed_text': 'We applied the method of Smith et al.',
                'gold_label': 'method', 'original_prediction': 'SECRET_PREDICTION'}
        labels = {'background': 'Prior context.', 'method': 'An adopted method.', 'result': 'Results comparison.'}
        requests = protocol.build_requests(pair, labels, 8240)
        payload = json.loads(requests['quality']['messages'][1]['content'])
        self.assertIn('citation', payload['task'].lower())
        self.assertIn('function', payload['task'].lower())
        for request in requests.values():
            self.assertNotIn('SECRET_PREDICTION', json.dumps(request))
            self.assertNotIn('gold_label', json.dumps(request))

    def test_environmental_task_distinguishes_claim_presence_from_truth(self):
        pair = {'pair_id': 'claim', 'task': 'environmental_claim',
                'original_text': 'Our process reduces emissions.',
                'perturbed_text': 'Our process lowers emissions.'}
        requests = protocol.build_requests(pair, {'no': 'No claim.', 'yes': 'Claim present.'}, 8240)
        task = json.loads(requests['quality']['messages'][1]['content'])['task'].lower()
        self.assertIn('explicit', task)
        self.assertIn('truth', task)
        self.assertIn('implicit', task)

    def test_task_specific_label_definitions_are_exported(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            source = task_source(Path(folder) / 'claims', 'claims', 'environmental_claim', {'no': 0, 'yes': 1}, 'yes')
            _, info, labels = judge_data.read_source(source)
            self.assertIn('explicit', labels['yes'].lower())
            self.assertIn('environment', labels['yes'].lower())
            self.assertIn('no explicit', labels['no'].lower())
            self.assertIn('task-specific', info['label_descriptions'].lower())

    def test_new_task_rejects_an_incompatible_label_taxonomy(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            source = source_fixture(Path(folder))
            source['task'] = 'citation_intent'
            with self.assertRaisesRegex(ValueError, 'task label'):
                judge_data.read_source(source)

    def test_fixed_final_sample_is_reproducible_and_separate_from_development(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            root = Path(folder)
            source = source_fixture(root)
            manifests = []
            for name in ['one', 'two']:
                manifests.append(judge_data.prepare({'sources': [source]}, root / name,
                    development_per_dataset=2, audit_size=2, final_per_dataset=3))
            self.assertEqual(manifests[0], manifests[1])
            manifest, pairs, _ = judge_data.load_prepared(root / 'one')
            final = {p['pair_id'] for p in pairs if p['phase'] == 'final'}
            development = {p['pair_id'] for p in pairs if p['phase'] == 'development'}
            self.assertEqual(len(final), 3)
            self.assertEqual(len(development), 2)
            self.assertFalse(final & development)
            self.assertTrue(set(manifest['audit_ids']) <= final)
            self.assertEqual(manifest['sources']['reviews']['successful'], 7)
            self.assertEqual(manifest['sources']['reviews']['final_population_after_development'], 5)
            self.assertEqual(manifest['sources']['reviews']['final_selected'], 3)
            self.assertEqual(manifest['final_per_dataset'], 3)

    def test_zero_final_sample_cap_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                judge_data.prepare({'sources': [source_fixture(root)]}, root / 'prepared', final_per_dataset=0)
            self.assertFalse((root / 'prepared').exists())

    def test_new_tasks_have_known_meaning_and_label_reversal_controls(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            root = Path(folder)
            sources = [
                task_source(root / 'sci', 'scicite', 'citation_intent', {'background': 0, 'method': 1, 'result': 2}, 'method'),
                task_source(root / 'env', 'claims', 'environmental_claim', {'no': 0, 'yes': 1}, 'yes'),
            ]
            judge_data.prepare({'sources': sources}, root / 'prepared', development_per_dataset=2, audit_size=2)
            _, pairs, spaces = judge_data.load_prepared(root / 'prepared')
            for task in ['citation_intent', 'environmental_claim']:
                controls = [p for p in pairs if p['task'] == task and p['control'] == 'task_label_reversal']
                self.assertEqual(len(controls), 1)
                control = controls[0]
                self.assertEqual(control['expected_meaning'], 'no')
                self.assertEqual(control['expected_same_label'], 'no')
                self.assertIn(control['gold_label'], spaces[control['dataset']])
                payload = protocol.build_requests(control, spaces[control['dataset']], 8240)
                self.assertNotIn('expected_meaning', json.dumps(payload))

    def test_prepared_protocol_mismatch_is_rejected_explicitly(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            root = Path(folder)
            protocol.save_json(root / 'manifest.json', {'protocol': 'obsolete-protocol',
                                                      'immutable_sha256': {}, 'sources': {}})
            with self.assertRaisesRegex(ValueError, 'protocol'):
                judge_data.load_prepared(root)

    def test_complete_h100_export_includes_final_jobs_without_downloading_weights(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            root = Path(folder)
            judge_data.prepare({'sources': [source_fixture(root)]}, root / 'prepared',
                               development_per_dataset=2, audit_size=2)
            result = judge_local.export_h100(root / 'prepared', root / 'export', 'Qwen/Qwen3.8-27B',
                                            include_final=True, model_revision='a' * 40)
            pack = protocol.read_json(root / 'export/h100_pack.json')
            self.assertEqual(set(pack['phases']), {'development', 'control', 'final'})
            self.assertEqual(pack['model_revision'], 'a' * 40)
            final = protocol.read_json(root / 'export' / pack['phases']['final'] / 'bundle.json')
            self.assertEqual(len(final['jobs']), 15)
            self.assertEqual(result['live_judgements'], 0)
            self.assertFalse((root / 'export/model_cache').exists())
            self.assertFalse((root / 'export/results').exists())


if __name__ == '__main__':
    unittest.main()
