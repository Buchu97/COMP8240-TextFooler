import copy
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import judge_data
import judge_local
import judge_protocol as protocol
import hf_judge_worker
from test_judge_data import source_fixture


QUALITY = dict(meaning_preserved='no', same_task_label='yes', naturalness_a=4,
               naturalness_b=2, meaning_reason='A detail changed.',
               label_reason='Both are positive.', naturalness_reason='The change is awkward.')
META = dict(model='Qwen/Qwen3.8-27B', revision='a' * 40, dtype='bfloat16',
            config_sha256='b' * 64, gpu='NVIDIA H100 PCIe (test fixture)',
            packages={'torch': '2.7.1+cu126', 'transformers': '5.20.1', 'accelerate': '1.10.1'},
            attention='efficient_sdpa_explicit_gqa', thinking=False, do_sample=False,
            max_input_tokens=8192, schema_enforcement='post_generation_validation')


class JudgeRepairTests(unittest.TestCase):
    def setUp(self):
        for name in ('hf_judge_repair', 'judge_repair'):
            path = ROOT / 'scripts' / (name + '.py')
            self.assertTrue(path.exists(), f'{name} repair module is missing')
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            setattr(self, name, module)

    def label_job(self):
        body = {'model': META['model'], 'messages': [
            {'role': 'system', 'content': 'Classify sentiment.'},
            {'role': 'user', 'content': '{"text":"A fine movie.","labels":{"negative":"Bad","positive":"Good"}}'}],
            'max_tokens': 1024, 'temperature': 0, 'store': False,
            'response_format': {'type': 'json_schema', 'json_schema': {
                'schema': protocol.object_schema({'label': {'type': 'string', 'enum': ['negative', 'positive', '__uncertain__']},
                                                  'reason': {'type': 'string'}})}}}
        job = dict(pair_id='fixture_pair', kind='original_label', body=body, phase='final')
        job['job_id'] = protocol.digest({'protocol': protocol.VERSION, 'pair_id': job['pair_id'],
                                         'kind': job['kind'], 'body': body})
        return job

    def test_lossless_wrappers_preserve_all_decisions_without_coercion(self):
        job = self.label_job()
        wrapped = {'type': 'object', 'properties': {'label': 'negative', 'reason': 'The film is disliked.'}}
        answer, method = self.hf_judge_repair.parse_answer(job, json.dumps(wrapped))
        self.assertEqual(answer, {'label': 'negative', 'reason': 'The film is disliked.'})
        self.assertEqual(method, 'unwrap_properties')
        wrapped['properties'] = {k: {'const': v} for k, v in wrapped['properties'].items()}
        answer, method = self.hf_judge_repair.parse_answer(job, json.dumps(wrapped))
        self.assertEqual(answer['label'], 'negative')
        self.assertEqual(method, 'unwrap_constants')
        job['kind'] = 'quality'
        job['body']['response_format']['json_schema']['schema'] = protocol.quality_schema()
        answer, _ = self.hf_judge_repair.parse_answer(job, json.dumps({'type': 'object', 'properties': QUALITY}))
        self.assertEqual(answer, QUALITY)

    def test_schema_echo_missing_scores_and_duplicate_fields_are_not_guessed(self):
        job = self.label_job()
        for text in (json.dumps(job['body']['response_format']['json_schema']['schema']),
                     '{"label":"positive","label":"negative","reason":"Duplicated."}',
                     '{"label":"positive"}',
                     '{"type":"object","properties":{"label":"positive","reason":"Fine"},"extra":1}'):
            with self.subTest(text=text):
                self.assertEqual(self.hf_judge_repair.parse_answer(job, text), (None, None))
        job['kind'] = 'quality'
        job['body']['response_format']['json_schema']['schema'] = protocol.quality_schema()
        bad = dict(QUALITY, naturalness_a='4')
        self.assertEqual(self.hf_judge_repair.parse_answer(job, json.dumps(bad)), (None, None))

    def test_format_retry_retains_original_text_taxonomy_model_and_generation_settings(self):
        job = self.label_job()
        before = copy.deepcopy(job)
        for attempt in (1, 2, 3):
            body = self.hf_judge_repair.retry_body(job, attempt)
            self.assertEqual(body['messages'][:2], job['body']['messages'])
            for key in ('model', 'temperature', 'max_tokens', 'response_format', 'store'):
                self.assertEqual(body[key], job['body'][key])
            self.assertNotIn('gold_label', json.dumps(body))
        self.assertEqual(job, before)

    def fixture(self, root):
        prepared, pack, returned = root / 'prepared', root / 'pack', root / 'returned'
        judge_data.prepare({'sources': [source_fixture(root)]}, prepared, development_per_dataset=2, audit_size=3)
        judge_local.export_h100(prepared, pack, META['model'], include_final=True, model_revision=META['revision'])
        source_hashes = {}
        for phase, folder in (('control', 'h100_controls'), ('development', 'h100_development'), ('final', 'h100_final')):
            seen = []
            def generate(body):
                quality = set(body['response_format']['json_schema']['schema']['properties']) != {'label', 'reason'}
                answer = dict(QUALITY) if quality else {'label': 'positive', 'reason': 'A favourable review.'}
                if quality and not seen:
                    seen.append(True)
                    if phase == 'development':
                        answer = {'type': 'object', 'properties': answer}
                    elif phase == 'final':
                        answer = body['response_format']['json_schema']['schema']
                return {'text': json.dumps(answer), 'prompt_tokens': 30, 'completion_tokens': 20, 'elapsed_seconds': 0.1}
            hf_judge_worker.run_jobs(pack / folder / 'bundle.json', returned / phase, generate, META, 999)
        for path in returned.rglob('*.json'):
            source_hashes[path] = protocol.file_digest(path)
        return prepared, pack, returned, source_hashes

    def test_export_recovers_only_complete_wrappers_and_import_keeps_provenance_separate(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared, pack, returned, hashes = self.fixture(root)
            repair = root / 'repair'
            stats = self.judge_repair.prepare_repair(prepared, pack, returned, repair)
            self.assertEqual(stats['normalized'], 1)
            self.assertEqual(stats['needs_generation'], 1)
            plan_path = repair / 'judge_format_repair/repair_plan.json'
            plan = self.hf_judge_repair.load_plan(plan_path)
            self.assertEqual(len(plan['jobs']), 1)
            self.assertNotIn('gold_label', json.dumps(plan))
            retry_results = root / 'retry_results'
            valid = lambda body: dict(text=json.dumps(QUALITY), prompt_tokens=30, completion_tokens=20, elapsed_seconds=0.1)
            result = self.hf_judge_repair.run_repair(plan_path, retry_results, valid, META, 15)
            self.assertEqual(result['remaining'], 0)
            output = root / 'repaired_import'
            result = self.judge_repair.import_repaired(prepared, pack, returned, repair, output, retry_results)
            self.assertEqual(result['unresolved'], 0)
            self.assertEqual(result['normalized'], 1)
            self.assertEqual(result['regenerated'], 1)
            summary = protocol.read_json(output / 'reports/final/summary.json')
            self.assertEqual(summary['datasets']['reviews']['judged'], 5)
            self.assertEqual(summary['datasets']['reviews']['rejected'], 5)
            records = [protocol.read_json(p) for p in (output / 'evaluations/final/requests').glob('*.json')]
            retried = [r for r in records if r['import_provenance']['method'] == 'format_retry']
            self.assertEqual(len(retried), 1)
            self.assertEqual(len(retried[0]['import_provenance']['executed_request']['messages']), 3)
            for path, expected in hashes.items():
                self.assertEqual(protocol.file_digest(path), expected)
            with self.assertRaises(FileExistsError):
                self.judge_repair.import_repaired(prepared, pack, returned, repair, output, retry_results)

    def test_first_valid_response_is_retained_and_resume_never_selects_a_later_outcome(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared, pack, returned, _ = self.fixture(root)
            self.judge_repair.prepare_repair(prepared, pack, returned, root / 'repair')
            plan = root / 'repair/judge_format_repair/repair_plan.json'
            calls = []
            def generate(body):
                calls.append(body)
                answer = {} if len(calls) == 1 else QUALITY
                return dict(text=json.dumps(answer), prompt_tokens=30, completion_tokens=20, elapsed_seconds=0.1)
            output = root / 'attempts'
            result = self.hf_judge_repair.run_repair(plan, output, generate, META, 15)
            self.assertEqual((result['generated'], result['remaining']), (2, 0))
            self.assertEqual(len(list((output / 'attempts').rglob('*.json'))), 2)
            def forbidden(body):
                raise AssertionError('A completed repair generated another outcome')
            result = self.hf_judge_repair.run_repair(plan, output, forbidden, META, 15)
            self.assertEqual((result['generated'], result['remaining']), (0, 0))

    def test_invalid_output_has_a_bounded_retry_limit_and_records_every_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared, pack, returned, _ = self.fixture(root)
            self.judge_repair.prepare_repair(prepared, pack, returned, root / 'repair')
            plan = root / 'repair/judge_format_repair/repair_plan.json'
            bad = lambda body: dict(text='{}', prompt_tokens=30, completion_tokens=2, elapsed_seconds=0.1)
            output = root / 'attempts'
            result = self.hf_judge_repair.run_repair(plan, output, bad, META, 15)
            self.assertEqual((result['generated'], result['remaining']), (3, 1))
            result = self.hf_judge_repair.run_repair(plan, output, bad, META, 15)
            self.assertEqual((result['generated'], result['remaining']), (0, 1))
            self.assertEqual(len(list((output / 'attempts').rglob('*.json'))), 3)

    def test_saved_retry_response_must_keep_its_exact_request_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared, pack, returned, _ = self.fixture(root)
            self.judge_repair.prepare_repair(prepared, pack, returned, root / 'repair')
            plan = root / 'repair/judge_format_repair/repair_plan.json'
            valid = lambda body: dict(text=json.dumps(QUALITY), prompt_tokens=30, completion_tokens=20, elapsed_seconds=0.1)
            output = root / 'attempts'
            self.hf_judge_repair.run_repair(plan, output, valid, META, 15)
            path = next((output / 'attempts').rglob('*.json'))
            record = protocol.read_json(path)
            record['response']['id'] = 'repair_another_request'
            record['response_sha256'] = protocol.digest(record['response'])
            protocol.save_json(path, record)
            with self.assertRaises(ValueError):
                self.hf_judge_repair.run_repair(plan, output, valid, META, 15)

    def test_altered_source_or_retry_runtime_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared, pack, returned, _ = self.fixture(root)
            repair = root / 'repair'
            self.judge_repair.prepare_repair(prepared, pack, returned, repair)
            plan = repair / 'judge_format_repair/repair_plan.json'
            with self.assertRaises(ValueError):
                self.hf_judge_repair.run_repair(plan, root / 'retry', lambda body: {}, dict(META, revision='c' * 40), 15)
            path = next((returned / 'final/responses').glob('*.json'))
            record = protocol.read_json(path)
            record['response']['choices'][0]['message']['content'] = '{}'
            protocol.save_json(path, record)
            with self.assertRaises(ValueError):
                self.judge_repair.import_repaired(prepared, pack, returned, repair, root / 'bad_import')
            self.assertFalse((root / 'bad_import').exists())
            record['response_sha256'] = protocol.digest(record['response'])
            protocol.save_json(path, record)
            with self.assertRaises(ValueError):
                self.judge_repair.import_repaired(prepared, pack, returned, repair, root / 'bad_import_rehashed')
            self.assertFalse((root / 'bad_import_rehashed').exists())

    def test_generation_exception_or_incomplete_envelope_consumes_its_attempt(self):
        for mode in ('exception', 'incomplete'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
                root = Path(directory)
                prepared, pack, returned, _ = self.fixture(root)
                self.judge_repair.prepare_repair(prepared, pack, returned, root / 'repair')
                plan = root / 'repair/judge_format_repair/repair_plan.json'
                output = root / 'attempts'
                def bad(body):
                    if mode == 'exception':
                        raise RuntimeError('Fixture generation interrupted')
                    return {'text': '{}'}
                with self.assertRaises((RuntimeError, ValueError)):
                    self.hf_judge_repair.run_repair(plan, output, bad, META, 15)
                records = list((output / 'attempts').rglob('*.json'))
                self.assertEqual(len(records), 1)
                self.assertEqual(protocol.read_json(records[0])['state'], 'generation_error')
                valid = lambda body: dict(text=json.dumps(QUALITY), prompt_tokens=30, completion_tokens=20, elapsed_seconds=0.1)
                stats = self.hf_judge_repair.run_repair(plan, output, valid, META, 15)
                self.assertEqual((stats['generated'], stats['remaining']), (1, 0))
                self.assertEqual(sorted(p.name for p in (output / 'attempts').rglob('*.json')), ['01.json', '02.json'])

    def test_abrupt_process_exit_leaves_a_consumed_reservation_and_resumable_lock(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared, pack, returned, _ = self.fixture(root)
            self.judge_repair.prepare_repair(prepared, pack, returned, root / 'repair')
            plan = root / 'repair/judge_format_repair/repair_plan.json'
            output = root / 'attempts'
            program = ('import sys,os,json; from pathlib import Path; '
                       'sys.path.insert(0,sys.argv[1]); import hf_judge_repair as repair; '
                       'repair.run_repair(Path(sys.argv[2]),Path(sys.argv[3]),'
                       'lambda body: os._exit(23),json.loads(sys.argv[4]),15)')
            result = subprocess.run([sys.executable, '-c', program, str(ROOT / 'scripts'),
                                     str(plan), str(output), json.dumps(META)], capture_output=True)
            self.assertEqual(result.returncode, 23)
            records = list((output / 'attempts').rglob('*.json'))
            self.assertEqual(len(records), 1)
            self.assertEqual(protocol.read_json(records[0])['state'], 'reserved')
            valid = lambda body: dict(text=json.dumps(QUALITY), prompt_tokens=30, completion_tokens=20, elapsed_seconds=0.1)
            stats = self.hf_judge_repair.run_repair(plan, output, valid, META, 15)
            self.assertEqual((stats['generated'], stats['remaining']), (1, 0))
            self.assertEqual(sorted(p.name for p in (output / 'attempts').rglob('*.json')), ['01.json', '02.json'])

    def test_partial_retry_archive_without_attempts_imports_as_pending(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared, pack, returned, _ = self.fixture(root)
            repair = root / 'repair'
            self.judge_repair.prepare_repair(prepared, pack, returned, repair)
            plan = self.hf_judge_repair.load_plan(repair / 'judge_format_repair/repair_plan.json')
            retry_results = root / 'retry_results'
            retry_results.mkdir()
            protocol.save_json(retry_results / 'repair_metadata.json',
                               {'plan_signature': plan['signature'], 'model_metadata': META})
            output = root / 'partial_import'
            result = self.judge_repair.import_repaired(prepared, pack, returned, repair, output, retry_results)
            self.assertEqual((result['normalized'], result['unresolved']), (1, 1))
            report = protocol.read_json(output / 'reports/final/summary.json')
            self.assertEqual((report['datasets']['reviews']['judged'], report['datasets']['reviews']['unjudged']), (4, 1))
            self.assertEqual(result['phases']['final']['pending'], 1)

    @contextmanager
    def redirected_directory(self, link, target):
        # Exercise resolver containment without requiring OS junction privileges.
        link.mkdir()
        original = Path.resolve
        def redirected(path, *args, **kwargs):
            try:
                relative = path.absolute().relative_to(link.absolute())
            except ValueError:
                return original(path, *args, **kwargs)
            return original(target / relative, *args, **kwargs)
        with patch.object(Path, 'resolve', redirected):
            yield

    def test_redirected_default_results_cannot_target_first_attempt_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared, pack, returned, hashes = self.fixture(root)
            self.judge_repair.prepare_repair(prepared, pack, returned, root / 'repair')
            plan = root / 'repair/judge_format_repair/repair_plan.json'
            with self.redirected_directory(plan.parent / 'results', returned / 'final'):
                with self.assertRaises(ValueError):
                    self.hf_judge_repair.default_results(plan)
            for path, expected in hashes.items():
                self.assertEqual(protocol.file_digest(path), expected)

    def test_existing_foreign_output_and_redirected_attempt_paths_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared, pack, returned, hashes = self.fixture(root)
            self.judge_repair.prepare_repair(prepared, pack, returned, root / 'repair')
            plan_path = root / 'repair/judge_format_repair/repair_plan.json'
            plan = self.hf_judge_repair.load_plan(plan_path)
            valid = lambda body: dict(text=json.dumps(QUALITY), prompt_tokens=30, completion_tokens=20, elapsed_seconds=0.1)
            with self.assertRaises(ValueError):
                self.hf_judge_repair.run_repair(plan_path, returned / 'final', valid, META, 15)
            output, outside = root / 'retry', root / 'outside'
            output.mkdir()
            outside.mkdir()
            protocol.save_json(output / 'repair_metadata.json', {'plan_signature': plan['signature'], 'model_metadata': META})
            with self.redirected_directory(output / 'attempts', outside):
                with self.assertRaises(ValueError):
                    self.hf_judge_repair.run_repair(plan_path, output, valid, META, 15)
            self.assertEqual(list(outside.iterdir()), [])
            for path, expected in hashes.items():
                self.assertEqual(protocol.file_digest(path), expected)

    def test_live_runtime_requires_dedicated_interpreter_and_clears_inherited_caches(self):
        path = ROOT / 'scripts/h100_judge.py'
        spec = importlib.util.spec_from_file_location('repair_bootstrap_test', path)
        bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bootstrap)
        with self.assertRaises(RuntimeError):
            self.hf_judge_repair.isolate_runtime(ROOT, bootstrap)
        with (
            patch.object(sys, 'executable', str(bootstrap.python_path(ROOT))),
            patch.object(sys, 'flags', SimpleNamespace(isolated=1)),
            patch.dict(os.environ, {
                'PYTHONPATH': 'other_project', 'HF_HOME': 'other_project_cache'
            }),
        ):
            self.hf_judge_repair.isolate_runtime(ROOT, bootstrap)
            self.assertNotIn('PYTHONPATH', os.environ)
            self.assertEqual(os.environ['HF_HOME'], str(ROOT / 'hf_home'))
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')


if __name__ == '__main__':
    unittest.main()
