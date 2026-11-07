"""Run a frozen, bounded response-format repair with the existing H100 judge."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_protocol as protocol


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result


def parse_answer(job, content):
    """Extract complete values only; schema descriptions and coercions are rejected."""
    try:
        answer = json.loads(content, object_pairs_hook=unique_object)
        method = 'unchanged'
        if isinstance(answer, dict) and set(answer) == {'type', 'properties'} and answer['type'] == 'object':
            answer = answer['properties']
            method = 'unwrap_properties'
            if isinstance(answer, dict) and answer and all(isinstance(v, dict) and set(v) == {'const'} for v in answer.values()):
                answer = {k: v['const'] for k, v in answer.items()}
                method = 'unwrap_constants'
        schema = job['body']['response_format']['json_schema']['schema']
        labels = [label for label in schema['properties'].get('label', {}).get('enum', [])
                  if label != protocol.UNCERTAIN]
        protocol.validate_answer(job['kind'], answer, labels)
        return answer, method
    except (ValueError, TypeError):
        return None, None


def retry_body(job, attempt):
    if type(attempt) is not int or attempt not in (1, 2, 3):
        raise ValueError('Formatting attempts are numbered 1 to 3')
    body = deepcopy(job['body'])
    schema = body['response_format']['json_schema']['schema']
    fields = []
    for name, rule in schema['properties'].items():
        detail = 'an integer from 1 to 5' if rule['type'] == 'integer' else 'a nonempty string'
        if 'enum' in rule:
            detail = 'one of ' + ', '.join(json.dumps(v) for v in rule['enum'])
        fields.append(name + ': ' + detail)
    lead = (
        'Return the completed judgement as a flat JSON object containing actual values.',
        'Fill in the requested judgement fields directly at the top level of one JSON object.',
        'Answer the evaluation questions, using a single JSON object whose field values are your decisions.',
    )[attempt - 1]
    contract = (lead + ' Required fields: ' + '; '.join(fields) + '. '
                'Do not return a JSON Schema, an object definition, a properties wrapper, const objects, '
                'Markdown, or additional keys. Ratings must be JSON integers, not descriptions or quoted numbers. '
                'Keep the original evaluation rubric and taxonomy; use uncertainty where the rubric permits it.')
    body['messages'].append({'role': 'user', 'content': contract})
    return body


def load_plan(path):
    plan = protocol.read_json(path)
    unsigned = {k: v for k, v in plan.items() if k != 'signature'}
    if (plan['signature'] != protocol.digest(unsigned)
            or plan['runner_sha256'] != protocol.file_digest(Path(__file__))
            or plan['protocol_sha256'] != protocol.file_digest(Path(protocol.__file__))
            or plan['max_attempts'] != 3):
        raise ValueError('Repair plan or implementation changed')
    seen = set()
    for job in plan['jobs']:
        expected = protocol.digest({'protocol': protocol.VERSION, 'pair_id': job['pair_id'],
                                    'kind': job['kind'], 'body': job['body']})
        if job['job_id'] != expected or expected in seen:
            raise ValueError('Invalid or duplicated repair request identity')
        seen.add(expected)
    if not re.fullmatch('[a-f0-9]{40}', plan['model_metadata']['revision']):
        raise ValueError('Repair requires the original immutable model revision')
    return plan


def validate_runtime(plan, metadata):
    baseline = plan['model_metadata']
    keys = ('model', 'revision', 'config_sha256', 'dtype', 'packages', 'attention',
            'thinking', 'do_sample', 'max_input_tokens', 'schema_enforcement')
    if any(metadata.get(k) != baseline.get(k) for k in keys):
        raise ValueError('Repair model or inference configuration differs from the original run')


def checked_path(directory, *parts):
    directory = Path(directory)
    path = directory.joinpath(*parts)
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError('A repair output path escapes its result folder')
    return path


def default_results(plan_path):
    folder = Path(plan_path).resolve().parent
    return checked_path(folder, 'results')


def validate_output(plan_path, output, plan):
    output = Path(output)
    if output.resolve() == Path(plan_path).resolve().parent or Path(plan_path).resolve().is_relative_to(output.resolve()):
        raise ValueError('Repair outputs must remain separate from input files')
    metadata_path = checked_path(output, 'repair_metadata.json')
    if output.exists() and any(p.name != 'repair.lock' for p in output.iterdir()) and not metadata_path.is_file():
        raise ValueError('Existing output is not a repair result folder; original files are preserved')
    for name in ('repair.lock', 'summary.json'):
        checked_path(output, name)
    for job in plan['jobs']:
        checked_path(output, 'attempts', job['job_id'])


def isolate_runtime(root, bootstrap):
    if Path(sys.executable).resolve() != bootstrap.python_path(root).resolve() or not sys.flags.isolated:
        raise RuntimeError('Use the standalone .venv\\Scripts\\python.exe with -I for this repair')
    environment = bootstrap.environment(root)
    os.environ.clear()
    os.environ.update(environment)


def scan_attempts(plan, job, directory, metadata):
    folder = checked_path(directory, 'attempts', job['job_id'])
    files = sorted(folder.glob('*.json'))
    if len(files) > 3 or [p.name for p in files] != [f'{i:02}.json' for i in range(1, len(files) + 1)]:
        raise ValueError('Repair attempts must be contiguous and limited to three')
    selected = None
    for attempt, path in enumerate(files, 1):
        if selected is not None:
            raise ValueError('A later attempt exists after a usable judgement')
        path = checked_path(directory, 'attempts', job['job_id'], path.name)
        record = protocol.read_json(path)
        body = retry_body(job, attempt)
        if (record['plan_signature'] != plan['signature'] or record['original_job_id'] != job['job_id']
                or record['attempt'] != attempt or record['executed_request'] != body
                or record['executed_request_sha256'] != protocol.digest(body)
                or record['model_metadata_sha256'] != protocol.digest(metadata)):
            raise ValueError('Saved repair request or runtime changed')
        if record.get('state') in ('reserved', 'generation_error'):
            continue
        if (record.get('state') != 'completed'
                or record['response_sha256'] != protocol.digest(record['response'])
                or record['response'].get('id') != 'repair_' + protocol.digest([job['job_id'], attempt, body])
                or record['response'].get('model') != metadata['model'] + '@' + metadata['revision']):
            raise ValueError('Saved repair request, response or runtime changed')
        choice = record['response']['choices'][0]
        answer, method = parse_answer(job, choice['message']['content'])
        if choice.get('finish_reason') == 'stop' and answer is not None:
            selected = {'answer': answer, 'method': method, 'record': record, 'path': path}
    return selected, len(files)


@contextmanager
def result_lock(directory):
    # OS locks release on interruption; the marker can remain safely on disk.
    with checked_path(directory, 'repair.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another repair worker is active') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_repair(plan_path, output, generate, metadata, max_jobs):
    plan, output = load_plan(plan_path), Path(output)
    validate_runtime(plan, metadata)
    if type(max_jobs) is not int or max_jobs < 1:
        raise ValueError('max-jobs must be a positive integer')
    validate_output(plan_path, output, plan)
    output.mkdir(parents=True, exist_ok=True)
    generated = touched = 0
    expected = {'plan_signature': plan['signature'], 'model_metadata': metadata}
    with result_lock(output):
        protocol.frozen_json(checked_path(output, 'repair_metadata.json'), expected)
        for job in plan['jobs']:
            selected, attempts = scan_attempts(plan, job, output, metadata)
            if selected or attempts == 3 or touched >= max_jobs:
                continue
            touched += 1
            for attempt in range(attempts + 1, 4):
                body = retry_body(job, attempt)
                path = checked_path(output, 'attempts', job['job_id'], f'{attempt:02}.json')
                if path.exists():
                    raise FileExistsError('Repair attempt already exists')
                record = {'plan_signature': plan['signature'], 'original_job_id': job['job_id'],
                          'attempt': attempt, 'executed_request': body,
                          'executed_request_sha256': protocol.digest(body),
                          'model_metadata_sha256': protocol.digest(metadata), 'state': 'reserved',
                          'started_at_utc': datetime.now(timezone.utc).isoformat()}
                protocol.save_json(path, record)
                generated += 1
                try:
                    result = generate(body)
                    if (not isinstance(result, dict) or not isinstance(result.get('text'), str)
                            or any(type(result.get(k)) is not int or result[k] < 0 for k in ('prompt_tokens', 'completion_tokens'))
                            or type(result.get('elapsed_seconds')) not in (float, int)
                            or not math.isfinite(result['elapsed_seconds']) or result['elapsed_seconds'] < 0):
                        record['generation_returned_repr'] = repr(result)
                        raise ValueError('Incomplete generation result')
                except Exception as error:
                    record.update(state='generation_error', error_type=type(error).__name__,
                                  error=str(error), finished_at_utc=datetime.now(timezone.utc).isoformat())
                    protocol.save_json(path, record)
                    raise
                response = {'id': 'repair_' + protocol.digest([job['job_id'], attempt, body]),
                            'model': metadata['model'] + '@' + metadata['revision'],
                            'choices': [{'finish_reason': result.get('finish_reason', 'stop'),
                                         'message': {'role': 'assistant', 'content': result['text']}}],
                            'usage': {k: result[k] for k in ('prompt_tokens', 'completion_tokens')}}
                record.update(state='completed', response=response,
                              response_sha256=protocol.digest(response), elapsed_seconds=result['elapsed_seconds'],
                              completed_at_utc=datetime.now(timezone.utc).isoformat())
                protocol.save_json(path, record)
                selected, _ = scan_attempts(plan, job, output, metadata)
                print(f'[repair] request {touched}/{min(max_jobs, len(plan["jobs"]))} | attempt {attempt} | '
                      + ('usable' if selected else 'invalid format'), flush=True)
                if selected:
                    break
        ready = sum(scan_attempts(plan, job, output, metadata)[0] is not None for job in plan['jobs'])
        summary = {'planned': len(plan['jobs']), 'generated': generated, 'usable': ready,
                   'remaining': len(plan['jobs']) - ready, 'plan_signature': plan['signature']}
        protocol.save_json(checked_path(output, 'summary.json'), summary)
    return summary


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--max-jobs', type=int, default=15)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan_path = Path(__file__).with_name('repair_plan.json')
    try:
        plan = load_plan(plan_path)
        root = args.job_root.resolve()
        pack = protocol.read_json(root / 'h100_pack.json')
        bootstrap, worker = root / 'h100_judge.py', root / 'h100_final/hf_judge_worker.py'
        if (protocol.digest(pack) != plan['original_pack_signature']
                or protocol.file_digest(bootstrap) != plan['bootstrap_sha256']
                or protocol.file_digest(worker) != plan['original_worker_sha256']):
            raise ValueError('Use the original standalone judge folder for this repair')
        if args.check_only:
            print(json.dumps({'planned': len(plan['jobs']), 'model': plan['model_metadata']['model'],
                              'max_attempts_per_request': 3, 'model_loaded': False}, indent=2))
            return
        output = default_results(plan_path)
        validate_output(plan_path, output, plan)
        if (output / 'repair_metadata.json').exists():
            metadata = protocol.read_json(output / 'repair_metadata.json')['model_metadata']
            if all(scan_attempts(plan, job, output, metadata)[0] for job in plan['jobs']):
                def forbidden(body):
                    raise RuntimeError('A completed repair must not regenerate judgements')
                print(json.dumps(run_repair(plan_path, output, forbidden, metadata, args.max_jobs), indent=2))
                return
        h100 = load_module(bootstrap, 'original_h100_bootstrap')
        isolate_runtime(root, h100)
        snapshot = h100.ready_snapshot(root)
        original = load_module(worker, 'original_h100_worker')
        contexts = {'config': plan['config'], 'jobs': [dict(body=retry_body(job, attempt))
                    for job in plan['jobs'] for attempt in (1, 2, 3)]}
        generate, metadata = original.load_generator(contexts, snapshot, 8192)
        summary = run_repair(plan_path, output, generate, metadata, args.max_jobs)
        print(json.dumps(summary, indent=2))
        if summary['remaining']:
            parser.exit(2, 'Some repair requests remain unresolved; saved attempts have been retained.\n')
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
