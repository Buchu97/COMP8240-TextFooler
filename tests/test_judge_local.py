import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import judge_protocol as protocol
import judge_data
from test_judge_data import source_fixture


class LocalJudgeTests(unittest.TestCase):
    def setUp(self):
        for name in ('judge_local', 'hf_judge_worker'):
            path = ROOT / 'scripts' / (name + '.py')
            self.assertTrue(path.exists(), f'{name} is missing')
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            setattr(self, name, module)

    def test_self_contained_pack_exports_runtime_and_jobs_without_a_model_cache(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared = root / 'prepared'
            judge_data.prepare({'sources': [source_fixture(root)]}, prepared, development_per_dataset=2, audit_size=3)
            exported = root / 'portable'
            self.assertTrue(hasattr(self.judge_local, 'export_h100'), 'Self-contained exporter is missing')
            self.judge_local.export_h100(prepared, exported, 'Qwen/Qwen3.8-27B')
            self.assertTrue((exported / 'h100_judge.py').exists())
            self.assertTrue((exported / 'requirements_h100_judge.txt').exists())
            self.assertTrue((exported / 'h100_development/bundle.json').exists())
            self.assertTrue((exported / 'h100_controls/bundle.json').exists())
            self.assertFalse((exported / 'model_cache').exists())
            self.assertFalse((exported / '.venv').exists())
            with self.assertRaises(FileExistsError):
                self.judge_local.export_h100(prepared, exported, 'Qwen/Qwen3.8-27B')

    def test_portable_worker_roundtrip_resumes_and_imports_without_network_or_gold(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared = root / 'prepared'
            judge_data.prepare({'sources': [source_fixture(root)]}, prepared, development_per_dataset=2, audit_size=3)
            exported = root / 'export'
            self.judge_local.export_local(prepared, exported, 'development', 'Qwen/Qwen3.8-27B')
            bundle = protocol.read_json(exported / 'bundle.json')
            self.assertEqual(len(bundle['jobs']), 6)
            self.assertNotIn('gold_label', json.dumps(bundle))
            self.assertTrue((exported / 'hf_judge_worker.py').exists())
            calls = []
            def generate(body):
                calls.append(body)
                if body['response_format']['json_schema']['name'].endswith('quality'):
                    answer = {'meaning_preserved': 'yes', 'same_task_label': 'yes',
                              'naturalness_a': 4, 'naturalness_b': 4, 'meaning_reason': 'Fixture meaning.',
                              'label_reason': 'Fixture label.', 'naturalness_reason': 'Fixture fluency.'}
                else:
                    answer = {'label': 'positive', 'reason': 'Fixture classification.'}
                return {'text': json.dumps(answer), 'prompt_tokens': 20, 'completion_tokens': 10, 'elapsed_seconds': 0.1}
            metadata = {'model': 'Qwen/Qwen3.8-27B', 'revision': 'a' * 40,
                        'runtime': 'test_fixture_not_live_inference'}
            results = root / 'local_results'
            self.hf_judge_worker.run_jobs(exported / 'bundle.json', results, generate, metadata, 2)
            self.hf_judge_worker.run_jobs(exported / 'bundle.json', results, generate, metadata, 10)
            self.assertEqual(len(calls), 6)
            self.hf_judge_worker.run_jobs(exported / 'bundle.json', results, generate, metadata, 10)
            self.assertEqual(len(calls), 6)
            evaluation = root / 'evaluation'
            stats = self.judge_local.import_local(prepared, exported / 'bundle.json', results, evaluation)
            self.assertEqual(stats['completed'], 6)
            self.assertEqual(stats['network_requests_made'], 0)
            import llm_judge
            report = llm_judge.summarize(prepared, evaluation, root / 'report', phase='development')
            self.assertEqual(report['datasets']['reviews']['accepted'], 2)
            raw = next((results / 'responses').glob('*.json'))
            record = protocol.read_json(raw)
            record['job_id'] = 'wrong'
            protocol.save_json(raw, record)
            with self.assertRaises(ValueError):
                self.judge_local.import_local(prepared, exported / 'bundle.json', results, root / 'bad')
            self.assertFalse((root / 'bad').exists())

    def test_worker_rejects_modified_job_ids_before_any_output_or_generation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared, exported = root / 'prepared', root / 'export'
            judge_data.prepare({'sources': [source_fixture(root)]}, prepared, development_per_dataset=2, audit_size=3)
            self.judge_local.export_local(prepared, exported, 'development', 'Qwen/Qwen3.8-27B')
            bundle = protocol.read_json(exported / 'bundle.json')
            bundle['jobs'][0]['job_id'] = '../../escape'
            protocol.save_json(exported / 'bundle.json', bundle)
            def forbidden(body):
                raise AssertionError('Invalid job reached the model')
            output = root / 'nested/output'
            with self.assertRaises(ValueError):
                self.hf_judge_worker.run_jobs(exported / 'bundle.json', output, forbidden,
                    {'model': 'Qwen/Qwen3.8-27B', 'revision': 'a' * 40}, 3)
            self.assertFalse(output.exists())

    def test_duplicate_text_jobs_keep_distinct_response_identity_and_runtime_provenance(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            source = source_fixture(root)
            csv_path = root / 'attack/outcomes.csv'
            rows = judge_data.read_csv(csv_path)
            for row in rows:
                row['original_text'] = 'A good shared movie review.'
            protocol.write_csv(csv_path, rows, list(rows[0]), exclusive=False)
            summary = protocol.read_json(root / 'attack/summary.json')
            summary['artifacts_sha256']['outcomes.csv'] = protocol.file_digest(csv_path)
            protocol.save_json(root / 'attack/summary.json', summary)
            prepared, exported = root / 'prepared', root / 'export'
            judge_data.prepare({'sources': [source]}, prepared, development_per_dataset=2, audit_size=3)
            self.judge_local.export_local(prepared, exported, 'development', 'Qwen/Qwen3.8-27B')
            counter = []
            def generate(body):
                counter.append(1)
                return {'text': json.dumps({'label': 'positive', 'reason': f'fixture generation {len(counter)}'}),
                        'prompt_tokens': 20, 'completion_tokens': 10, 'elapsed_seconds': 0.1}
            metadata = {'model': 'Qwen/Qwen3.8-27B', 'revision': 'a' * 40,
                        'runtime': 'fixture_runtime_one', 'dtype': 'bfloat16'}
            results, evaluation = root / 'results', root / 'evaluation'
            self.hf_judge_worker.run_jobs(exported / 'bundle.json', results, generate, metadata, 10)
            self.judge_local.import_local(prepared, exported / 'bundle.json', results, evaluation)
            records = [protocol.read_json(p) for p in (evaluation / 'requests').glob('*.json')]
            self.assertTrue(all(r['response']['id'] == 'local_' + r['job_id'] for r in records))
            settings = protocol.read_json(evaluation / 'evaluation.json')
            self.assertEqual(settings['backend']['local_model_metadata'], metadata)
            local_meta_path = results / 'worker_metadata.json'
            local_meta = protocol.read_json(local_meta_path)
            local_meta['model_metadata']['runtime'] = 'fixture_runtime_two'
            protocol.save_json(local_meta_path, local_meta)
            with self.assertRaises(ValueError):
                self.judge_local.import_local(prepared, exported / 'bundle.json', results, evaluation)

    def test_completed_pack_cli_can_verify_and_resume_without_loading_a_gpu_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared, exported = root / 'prepared', root / 'export'
            judge_data.prepare({'sources': [source_fixture(root)]}, prepared, development_per_dataset=2, audit_size=3)
            self.judge_local.export_local(prepared, exported, 'development', 'Qwen/Qwen3.8-27B')
            results = root / 'results'
            metadata = {'model': 'Qwen/Qwen3.8-27B', 'revision': 'a' * 40}
            self.hf_judge_worker.run_jobs(exported / 'bundle.json', results,
                lambda body: {'text': '{}', 'prompt_tokens': 20, 'completion_tokens': 2, 'elapsed_seconds': 0.1}, metadata, 10)
            arguments = ['worker', '--bundle', str(exported / 'bundle.json'), '--output-dir', str(results),
                         '--snapshot-dir', str(root / ('a' * 40)), '--max-jobs', '10']
            with patch.object(sys, 'argv', arguments), \
                 patch.object(self.hf_judge_worker, 'load_generator', side_effect=AssertionError('Completed pack tried to reload the GPU model')):
                self.hf_judge_worker.main()


if __name__ == '__main__':
    unittest.main()
