"""Resumable structured judging with explicit execution and client-side spending caps."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import base64
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener
from urllib.error import HTTPError

from judge_protocol import (VERSION, digest, file_digest, frozen_json, read_json,
                            save_json, validate_answer)


def validate_config(config):
    allowed = {'base_url', 'model', 'billing', 'key_env', 'max_completion_tokens', 'token_limit_field',
               'input_usd_per_million', 'output_usd_per_million', 'pricing_source', 'timeout_seconds',
               'temperature', 'reasoning_effort', 'expected_response_model', 'execution_mode',
               'local_model_revision', 'local_worker_sha256', 'local_model_metadata'}
    if set(config) - allowed:
        raise ValueError('Unknown configuration fields; credentials belong in an environment variable')
    if not isinstance(config.get('model'), str) or not config['model'].strip():
        raise ValueError('Choose an explicit judge model before running')
    endpoint = urlsplit(config.get('base_url', ''))
    local = endpoint.hostname in ('localhost', '127.0.0.1', '::1')
    if (not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
            or endpoint.scheme not in ('http', 'https') or (endpoint.scheme == 'http' and not local)):
        raise ValueError('Use HTTPS for remote endpoints; URLs cannot contain credentials or query strings')
    if config.get('billing') not in ('api', 'self_hosted') or (config['billing'] == 'self_hosted' and not local):
        raise ValueError('Self-hosted mode requires a loopback model server; remote APIs require billing safeguards')
    limit = config.get('max_completion_tokens', 1024)
    if type(limit) is not int or not 1 <= limit <= 16384:
        raise ValueError('Invalid completion token limit')
    if config.get('token_limit_field', 'max_completion_tokens') not in ('max_completion_tokens', 'max_tokens'):
        raise ValueError('Invalid token-limit parameter')
    if config['billing'] == 'api':
        rates = [config.get(k) for k in ('input_usd_per_million', 'output_usd_per_million')]
        if (not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in rates)
                or not sum(rates) or not config.get('pricing_source')):
            raise ValueError('Supply current input/output USD rates and their source before paid requests')
    return config


def request_body(job, config):
    body = {'model': config['model'], 'messages': job['request']['messages'], 'store': False,
            config.get('token_limit_field', 'max_completion_tokens'): config.get('max_completion_tokens', 1024),
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'textfooler_' + job['kind'], 'strict': True, 'schema': job['request']['schema']}}}
    for field in ('temperature', 'reasoning_effort'):
        if config.get(field) is not None:
            body[field] = config[field]
    return body


def request_id(job, config):
    return digest({'protocol': VERSION, 'pair_id': job['pair_id'], 'kind': job['kind'],
                   'body': request_body(job, config)})


def estimated_reservation(body, config):
    if config['billing'] == 'self_hosted':
        return 0.0
    # UTF-8 byte count plus a conservative protocol allowance, not a tokenizer estimate.
    input_bound = len(json.dumps(body, ensure_ascii=False).encode('utf-8')) + 8192
    return (input_bound * config['input_usd_per_million']
            + config.get('max_completion_tokens', 1024) * config['output_usd_per_million']) / 1_000_000


@dataclass(frozen=True)
class HTTPResponseData:
    status: int
    body: bytes
    request_id: str = ''


class RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def open_http(request, timeout):
    return build_opener(RejectRedirects()).open(request, timeout=timeout)


def http_transport(body, config):
    key = os.environ.get(config.get('key_env', 'JUDGE_API_KEY'), '')
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = Request(config['base_url'].rstrip('/') + '/chat/completions',
                      data=json.dumps(body, ensure_ascii=False).encode('utf-8'), headers=headers, method='POST')
    try:
        with open_http(request, timeout=config.get('timeout_seconds', 120)) as response:
            headers = getattr(response, 'headers', {})
            return HTTPResponseData(getattr(response, 'status', 200), response.read(), headers.get('x-request-id', ''))
    except HTTPError as error:
        return HTTPResponseData(error.code, error.read(), (error.headers or {}).get('x-request-id', ''))


def redact(value, secret):
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, '[REDACTED]')
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    if isinstance(value, dict):
        return {k: redact(v, secret) for k, v in value.items()}
    return value


def finalize(record, job, config):
    response = record['response']
    if digest(response) != record['response_sha256']:
        raise ValueError('Saved provider response hash mismatch')
    record['status'] = 'invalid'
    record.pop('cost_usd', None)
    try:
        if 'raw_http' in record:
            if digest(record['raw_http']) != record['raw_http_sha256']:
                raise ValueError('Raw HTTP response hash mismatch')
            response = json.loads(base64.b64decode(record['raw_http']['body_base64'], validate=True).decode('utf-8'))
        if not isinstance(response, dict):
            raise ValueError('Provider response must be a JSON object')
        usage = response.get('usage')
        if usage is None and config['billing'] == 'self_hosted':
            usage = {}
        if not isinstance(usage, dict):
            raise ValueError('Provider token usage must be an object')
        if config['billing'] == 'api':
            if not all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
                raise ValueError('Paid response lacks valid token usage; reservation remains held')
            record['cost_usd'] = (usage['prompt_tokens'] * config['input_usd_per_million']
                                  + usage['completion_tokens'] * config['output_usd_per_million']) / 1_000_000
            if record['cost_usd'] > record['reserved_usd'] + 1e-12:
                raise ValueError('Provider usage exceeded the reservation; inspect rates and token limits')
        else:
            record['cost_usd'] = 0.0
        record['usage'] = usage
        record['response_model'] = response.get('model')
        if not isinstance(record['response_model'], str) or not record['response_model']:
            raise ValueError('Provider did not identify the responding model')
        if config.get('expected_response_model') and record['response_model'] != config['expected_response_model']:
            raise ValueError('Unexpected responding model snapshot')
        choices = response.get('choices')
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError('Provider choices must be a nonempty array of objects')
        choice = choices[0]
        if not isinstance(choice.get('message'), dict):
            raise ValueError('Provider message must be an object')
        if choice['finish_reason'] != 'stop' or choice['message'].get('refusal'):
            raise ValueError('Judge refused or exhausted its output limit')
        answer = json.loads(choice['message']['content'])
        validate_answer(job['kind'], answer, job['labels'])
        record.update(status='completed', answer=answer)
        record.pop('error', None)
    except (KeyError, IndexError, TypeError, ValueError, UnicodeError) as error:
        record.pop('answer', None)
        record['error'] = str(error) if isinstance(error, ValueError) else 'Malformed provider response'
    return record


@contextmanager
def evaluation_lock(directory):
    lock = directory / 'judge.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def evaluation_settings(config, prepared_signature):
    return {'protocol': VERSION, 'prepared_signature': prepared_signature, 'backend': config,
            'implementation_sha256': {name: file_digest(Path(__file__).with_name(name))
                                      for name in ('judge_protocol.py', 'judge_runtime.py')}}


def accounted(directory):
    total = 0.0
    for path in (directory / 'requests').glob('*.json'):
        record = read_json(path)
        if record['status'] in ('unknown', 'reserved'):
            raise RuntimeError('An earlier request has unknown completion status. Its reservation is retained; do not automatically resubmit it.')
        value = record.get('cost_usd', record['reserved_usd'])
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid saved spending record')
        total += value
    return total


def execute_jobs(jobs, directory, config, prepared_signature, *, execute=False, budget_usd=0,
                 max_calls=0, transport=None, local_responses=None):
    validate_config(config)
    directory = Path(directory)
    stats = {'planned': len(jobs), 'completed': 0, 'invalid': 0, 'pending': len(jobs), 'new_calls': 0}
    if not execute:
        return stats
    if config.get('execution_mode') == 'local_hf_import' and local_responses is None:
        raise ValueError('Use import-local for offline H100 results; no HTTP requests are sent in this mode')
    if local_responses is not None and (config.get('execution_mode') != 'local_hf_import' or transport is not None):
        raise ValueError('Offline responses cannot be mixed with an HTTP transport')
    if type(max_calls) is not int or max_calls < 1:
        raise ValueError('Live execution needs an explicit positive max-calls limit')
    if config['billing'] == 'api' and (not math.isfinite(budget_usd) or budget_usd <= 0):
        raise ValueError('Paid execution needs an explicit positive budget-usd cap')
    if transport is None and local_responses is None and config['billing'] == 'api' and not os.environ.get(config.get('key_env', 'JUDGE_API_KEY')):
        raise ValueError('Set the configured API key environment variable before live execution')
    sender = transport or http_transport
    directory.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(directory):
        frozen_json(directory / 'evaluation.json', evaluation_settings(config, prepared_signature))
        accounted(directory)
        for job in jobs:
            identifier = request_id(job, config)
            path = directory / 'requests' / (identifier + '.json')
            if path.exists():
                record = read_json(path)
                if (record['job_id'] != identifier or record['job_signature'] != digest(job)
                        or record['request'] != request_body(job, config)):
                    raise ValueError('Saved judge request identity mismatch')
                if local_responses is not None and record.get('import_provenance') != local_responses[identifier]['provenance']:
                    raise ValueError('Local source response provenance changed')
                if record['status'] not in ('received', 'completed', 'invalid'):
                    raise RuntimeError('Request completion is unknown; automatic paid retries are disabled')
                record = finalize(record, job, config)
                save_json(path, record)
            else:
                if stats['new_calls'] >= max_calls:
                    continue
                body = request_body(job, config)
                reservation = estimated_reservation(body, config)
                spent = accounted(directory)
                if config['billing'] == 'api' and spent + reservation > budget_usd + 1e-12:
                    raise ValueError(f'Client budget cap reached: accounted ${spent:.6f}, next reservation ${reservation:.6f}, cap ${budget_usd:.6f}')
                record = {'job_id': identifier, 'job_signature': digest(job), 'pair_id': job['pair_id'],
                          'kind': job['kind'], 'status': 'reserved', 'reserved_usd': reservation,
                          'request': body}
                save_json(path, record)
                try:
                    if local_responses is not None:
                        response = local_responses[identifier]['response']
                        record['import_provenance'] = local_responses[identifier]['provenance']
                    else:
                        response = sender(body, config)
                except Exception as error:
                    record.update(status='unknown', error_type=type(error).__name__)
                    if isinstance(error, HTTPError):
                        record['http_status'] = error.code
                    save_json(path, record)
                    raise RuntimeError('Judge request failed with unknown completion status; reservation retained and automatic retries disabled') from None
                secret = os.environ.get(config.get('key_env', 'JUDGE_API_KEY'), '')
                if isinstance(response, HTTPResponseData):
                    raw_body = response.body.replace(secret.encode(), b'[REDACTED]') if secret else response.body
                    raw = {'status': response.status, 'request_id': response.request_id,
                           'body_base64': base64.b64encode(raw_body).decode('ascii')}
                    record.update(raw_http=raw, raw_http_sha256=digest(raw))
                    response = None
                response = redact(response, secret)
                record.update(status='received', response=response, response_sha256=digest(response))
                save_json(path, record)
                if 'raw_http' in record and not 200 <= record['raw_http']['status'] < 300:
                    record.update(status='unknown', http_status=record['raw_http']['status'])
                    save_json(path, record)
                    raise RuntimeError('HTTP error or redirect response retained; batch stopped without automatic retries')
                stats['new_calls'] += 1
                record = finalize(record, job, config)
                save_json(path, record)
            stats['completed' if record['status'] == 'completed' else 'invalid'] += 1
            stats['pending'] -= 1
    stats['accounted_usd'] = accounted(directory)
    return stats
