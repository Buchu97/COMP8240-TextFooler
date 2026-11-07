import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import judge_protocol as protocol


def configuration():
    return {'base_url': 'https://example.invalid/v1', 'model': 'test-model', 'billing': 'api',
            'key_env': 'JUDGE_TEST_KEY', 'max_completion_tokens': 100, 'token_limit_field': 'max_completion_tokens',
            'input_usd_per_million': 1, 'output_usd_per_million': 2, 'pricing_source': 'https://example.invalid/prices'}


def job():
    return {'pair_id': 'p1', 'kind': 'original_label', 'request': {
        'messages': [{'role': 'user', 'content': 'Classify this text.'}],
        'schema': protocol.object_schema({'label': {'type': 'string', 'enum': ['positive', 'negative', protocol.UNCERTAIN]},
                                          'reason': {'type': 'string'}})}, 'labels': ['positive', 'negative']}


def response(content=None):
    return {'id': 'fixture-response', 'model': 'test-model-v1',
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(content or {'label': 'positive', 'reason': 'Favourable.'})}}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 10}}


class JudgeRuntimeTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/judge_runtime.py'
        self.assertTrue(path.exists(), 'Judge runner is missing')
        spec = importlib.util.spec_from_file_location('judge_runtime', path)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        spec.loader.exec_module(self.module)

    def test_dry_run_never_reads_credentials_creates_files_or_calls_transport(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            output = Path(directory) / 'evaluation'
            def forbidden(*args):
                raise AssertionError('Dry run contacted a backend')
            stats = self.module.execute_jobs([job()], output, configuration(), 'prepared', transport=forbidden)
            self.assertEqual(stats['pending'], 1)
            self.assertFalse(output.exists())

    def test_completed_response_is_saved_and_reused_without_second_call(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            calls = []
            def transport(body, config):
                calls.append(body)
                return response()
            kwargs = dict(execute=True, budget_usd=1, max_calls=1, transport=transport)
            output = Path(directory) / 'evaluation'
            stats = self.module.execute_jobs([job()], output, configuration(), 'prepared', **kwargs)
            self.assertEqual(stats['completed'], 1)
            self.module.execute_jobs([job()], output, configuration(), 'prepared', **kwargs)
            self.assertEqual(len(calls), 1)
            record = protocol.read_json(next((output / 'requests').glob('*.json')))
            self.assertEqual(record['status'], 'completed')
            self.assertAlmostEqual(record['cost_usd'], 0.00004)
            self.assertEqual(record['response_model'], 'test-model-v1')

    def test_budget_reservation_precedes_network_and_unknown_calls_are_not_retried(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            output = Path(directory) / 'evaluation'
            calls = []
            def interrupted(body, config):
                calls.append(body)
                record = protocol.read_json(next((output / 'requests').glob('*.json')))
                self.assertEqual(record['status'], 'reserved')
                raise TimeoutError('simulated interruption')
            with self.assertRaises(RuntimeError):
                self.module.execute_jobs([job()], output, configuration(), 'prepared', execute=True,
                                         budget_usd=1, max_calls=1, transport=interrupted)
            with self.assertRaises(RuntimeError):
                self.module.execute_jobs([job()], output, configuration(), 'prepared', execute=True,
                                         budget_usd=1, max_calls=1, transport=interrupted)
            self.assertEqual(len(calls), 1)
            with self.assertRaises(ValueError):
                self.module.execute_jobs([job()], Path(directory) / 'tiny', configuration(), 'prepared',
                                         execute=True, budget_usd=0.000001, max_calls=1, transport=interrupted)
            self.assertEqual(len(calls), 1)

    def test_received_result_recovers_locally_and_invalid_model_answers_are_retained(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            output = Path(directory) / 'evaluation'
            self.module.execute_jobs([job()], output, configuration(), 'prepared', execute=True,
                                     budget_usd=1, max_calls=1, transport=lambda *args: response())
            path = next((output / 'requests').glob('*.json'))
            record = protocol.read_json(path)
            record['status'] = 'received'
            record.pop('answer')
            protocol.save_json(path, record)
            def forbidden(*args):
                raise AssertionError('Received response was charged twice')
            stats = self.module.execute_jobs([job()], output, configuration(), 'prepared', execute=True,
                                            budget_usd=1, max_calls=1, transport=forbidden)
            self.assertEqual(stats['completed'], 1)
            other = Path(directory) / 'invalid'
            stats = self.module.execute_jobs([job()], other, configuration(), 'prepared', execute=True,
                                            budget_usd=1, max_calls=1,
                                            transport=lambda *args: response({'label': 'unknown', 'reason': 'bad'}))
            self.assertEqual(stats['invalid'], 1)
            self.module.execute_jobs([job()], other, configuration(), 'prepared', execute=True,
                                     budget_usd=1, max_calls=1, transport=forbidden)

    def test_endpoint_secrets_and_changed_configuration_are_rejected(self):
        for bad in (dict(configuration(), base_url='http://example.invalid/v1'),
                    dict(configuration(), api_key='SECRET'),
                    dict(configuration(), billing='self_hosted')):
            with self.assertRaises(ValueError):
                self.module.validate_config(bad)
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            output = Path(directory) / 'evaluation'
            self.module.execute_jobs([job()], output, configuration(), 'prepared', execute=True,
                                     budget_usd=1, max_calls=1, transport=lambda *args: response())
            with self.assertRaises(ValueError):
                self.module.execute_jobs([job()], output, dict(configuration(), model='different'), 'prepared',
                                         execute=True, budget_usd=1, max_calls=1, transport=lambda *args: response())

    def test_active_lock_cannot_create_or_change_evaluation_settings(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            output = Path(directory)
            (output / 'judge.lock').write_text('another-process', encoding='ascii')
            with self.assertRaises(FileExistsError):
                self.module.execute_jobs([job()], output, configuration(), 'prepared', execute=True,
                                         budget_usd=1, max_calls=1, transport=lambda *args: response())
            self.assertFalse((output / 'evaluation.json').exists())

    def test_modified_request_payload_cannot_be_silently_reused(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            output = Path(directory) / 'evaluation'
            self.module.execute_jobs([job()], output, configuration(), 'prepared', execute=True,
                                     budget_usd=1, max_calls=1, transport=lambda *args: response())
            path = next((output / 'requests').glob('*.json'))
            record = protocol.read_json(path)
            record['request']['messages'][0]['content'] = 'Different request.'
            protocol.save_json(path, record)
            with self.assertRaises(ValueError):
                self.module.execute_jobs([job()], output, configuration(), 'prepared', execute=True,
                                         budget_usd=1, max_calls=1, transport=lambda *args: response())

    def test_http_adapter_sends_strict_json_and_keeps_authentication_out_of_body(self):
        body = self.module.request_body(job(), configuration())
        def endpoint(request, timeout):
            self.assertEqual(request.full_url, 'https://example.invalid/v1/chat/completions')
            self.assertEqual(request.get_method(), 'POST')
            self.assertEqual(request.get_header('Authorization'), 'Bearer fixture-secret')
            decoded = json.loads(request.data)
            self.assertEqual(decoded['response_format']['type'], 'json_schema')
            self.assertTrue(decoded['response_format']['json_schema']['strict'])
            self.assertEqual(decoded['max_completion_tokens'], 100)
            self.assertNotIn('fixture-secret', request.data.decode())
            return io.BytesIO(json.dumps(response()).encode())
        with patch.dict(os.environ, {'JUDGE_TEST_KEY': 'fixture-secret'}), \
             patch.object(self.module, 'open_http', side_effect=endpoint, create=True):
            actual = self.module.http_transport(body, configuration())
        self.assertEqual(json.loads(actual.body)['model'], 'test-model-v1')

    def test_accidental_credential_echo_is_redacted_before_raw_response_is_saved(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            output = Path(directory) / 'evaluation'
            echoed = dict(response(), debug='fixture-secret')
            with patch.dict(os.environ, {'JUDGE_TEST_KEY': 'fixture-secret'}):
                self.module.execute_jobs([job()], output, configuration(), 'prepared', execute=True,
                                         budget_usd=1, max_calls=1, transport=lambda *args: echoed)
            raw = next((output / 'requests').glob('*.json')).read_text(encoding='utf-8')
            self.assertNotIn('fixture-secret', raw)
            self.assertIn('[REDACTED]', raw)

    def test_redirects_are_refused_before_authentication_can_be_forwarded(self):
        from urllib.request import Request
        handler = getattr(self.module, 'RejectRedirects', None)
        self.assertIsNotNone(handler, 'Remote and local endpoint restrictions must apply to redirects too')
        request = Request('http://127.0.0.1:8000/v1/chat/completions', data=b'{}',
                          headers={'Authorization': 'Bearer fixture-secret'})
        self.assertIsNone(handler().redirect_request(request, None, 302, 'redirect', {}, 'https://other.invalid/'))

    def test_malformed_envelopes_are_retained_as_invalid_instead_of_crashing(self):
        for malformed in ([], None, 'wrong', dict(response(), usage=[])):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
                output = Path(directory) / 'evaluation'
                stats = self.module.execute_jobs([job()], output, configuration(), 'prepared', execute=True,
                                                budget_usd=1, max_calls=1, transport=lambda *args: malformed)
                self.assertEqual(stats['invalid'], 1)
                self.assertEqual(protocol.read_json(next((output / 'requests').glob('*.json')))['status'], 'invalid')

    def test_raw_non_json_http_response_is_saved_before_parsing(self):
        holder = getattr(self.module, 'HTTPResponseData', None)
        self.assertIsNotNone(holder, 'Raw HTTP bodies must survive JSON decoding failures')
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            output = Path(directory) / 'evaluation'
            stats = self.module.execute_jobs([job()], output, configuration(), 'prepared', execute=True,
                                            budget_usd=1, max_calls=1,
                                            transport=lambda *args: holder(200, b'{"broken":', 'fixture-request'))
            self.assertEqual(stats['invalid'], 1)
            record = protocol.read_json(next((output / 'requests').glob('*.json')))
            self.assertIn('raw_http', record)
            self.assertEqual(record['raw_http']['request_id'], 'fixture-request')


if __name__ == '__main__':
    unittest.main()
