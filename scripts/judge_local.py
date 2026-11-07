"""Portable job export and verified result import for an offline H100 judge."""
from pathlib import Path
import re
import shutil

from judge_data import load_prepared
from judge_protocol import digest, file_digest, read_json, save_json
from judge_runtime import execute_jobs, request_body, request_id


def export_local(prepared, output_dir, phase, model):
    from llm_judge import jobs_for_phase, safe_output
    prepared, output_dir = Path(prepared), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Local export directory already exists')
    if not model or '/' not in model:
        raise ValueError('Supply the exact Hugging Face model ID')
    manifest, pairs, spaces = load_prepared(prepared)
    safe_output(manifest, prepared, output_dir)
    config = {'base_url': 'http://127.0.0.1:8000/v1', 'model': model, 'billing': 'self_hosted',
              'max_completion_tokens': 1024, 'token_limit_field': 'max_tokens', 'temperature': 0,
              'execution_mode': 'local_hf_import'}
    jobs = jobs_for_phase(manifest, pairs, spaces, phase)
    worker = Path(__file__).with_name('hf_judge_worker.py')
    bundle = {'schema_version': 1, 'protocol': manifest['protocol'], 'phase': phase,
              'prepared_signature': digest(manifest), 'config': config,
              'worker_sha256': file_digest(worker),
              'jobs': [{'job_id': request_id(job, config), 'pair_id': job['pair_id'], 'kind': job['kind'],
                        'body': request_body(job, config)} for job in jobs]}
    output_dir.mkdir(parents=True)
    save_json(output_dir / 'bundle.json', bundle)
    shutil.copyfile(worker, output_dir / 'hf_judge_worker.py')
    return {'phase': phase, 'pairs': len(jobs) // 3, 'requests': len(jobs), 'model': model,
            'bundle': str((output_dir / 'bundle.json').resolve()), 'network_requests_made': 0}


def export_h100(prepared, output_dir, model, include_final=False, model_revision=None):
    from llm_judge import safe_output
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Standalone H100 export directory already exists')
    if model_revision is not None and not re.fullmatch('[a-f0-9]{40}', model_revision):
        raise ValueError('Supply a full immutable model revision SHA')
    manifest, _, _ = load_prepared(prepared)
    safe_output(manifest, prepared, output_dir)
    output_dir.mkdir(parents=True)
    phases = {'development': 'h100_development', 'control': 'h100_controls'}
    if include_final:
        phases['final'] = 'h100_final'
    exported = [export_local(prepared, output_dir / folder, phase, model) for phase, folder in phases.items()]
    for filename in ('h100_judge.py', 'requirements_h100_judge.txt'):
        shutil.copyfile(Path(__file__).with_name(filename), output_dir / filename)
    save_json(output_dir / 'h100_pack.json', {'schema_version': 1, 'model': model, 'model_revision': model_revision, 'phases': phases,
              'prepared_signature': digest(manifest), 'runtime': 'Windows/Python 3.11/CUDA 12.6/BF16',
              'cache': 'model_cache', 'environment': '.venv', 'results': 'results'})
    return {'folder': str(output_dir.resolve()), 'model': model, 'phases': exported,
            'model_downloaded': False, 'live_judgements': 0}


def import_local(prepared, bundle_path, results_dir, evaluation_dir):
    from llm_judge import jobs_for_phase, safe_output
    manifest, pairs, spaces = load_prepared(prepared)
    bundle, results_dir = read_json(bundle_path), Path(results_dir)
    if bundle['prepared_signature'] != digest(manifest):
        raise ValueError('Local bundle belongs to different prepared inputs')
    metadata = read_json(results_dir / 'worker_metadata.json')
    if metadata['bundle_signature'] != digest(bundle) or metadata['worker_sha256'] != bundle['worker_sha256']:
        raise ValueError('Local worker or bundle provenance mismatch')
    model_meta = metadata['model_metadata']
    if model_meta['model'] != bundle['config']['model']:
        raise ValueError('Local worker used a different model')
    config = dict(bundle['config'], expected_response_model=model_meta['model'] + '@' + model_meta['revision'],
                  local_model_revision=model_meta['revision'], local_worker_sha256=bundle['worker_sha256'],
                  local_model_metadata=model_meta)
    jobs = jobs_for_phase(manifest, pairs, spaces, bundle['phase'])
    exported = {job['job_id']: job for job in bundle['jobs']}
    if len(exported) != len(jobs):
        raise ValueError('Local job pack is incomplete or duplicated')
    available, responses = [], {}
    for job in jobs:
        identifier = request_id(job, config)
        if identifier not in exported or exported[identifier]['body'] != request_body(job, config):
            raise ValueError('Local request payload changed')
        path = results_dir / 'responses' / (identifier + '.json')
        if not path.exists():
            continue
        record = read_json(path)
        if (record['job_id'] != identifier or record['request_sha256'] != digest(exported[identifier]['body'])
                or record['bundle_signature'] != digest(bundle) or record['response_sha256'] != digest(record['response'])
                or record['response'].get('model') != config['expected_response_model']
                or record['response'].get('id') != 'local_' + identifier):
            raise ValueError('Local response content, request identity or model changed')
        available.append(job)
        responses[identifier] = {'response': record['response'], 'provenance': {
            'bundle_signature': digest(bundle), 'worker_metadata_signature': digest(model_meta),
            'source_record_sha256': file_digest(path), 'source_response_sha256': record['response_sha256'],
            'phase': bundle['phase']}}
    safe_output(manifest, prepared, evaluation_dir)
    stats = execute_jobs(available, evaluation_dir, config, digest(manifest), execute=True,
                         max_calls=max(1, len(available)), local_responses=responses)
    stats['imported_responses'] = stats.pop('new_calls')
    stats['network_requests_made'] = 0
    return stats
