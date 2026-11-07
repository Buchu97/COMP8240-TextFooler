"""Prepare and import an auditable response-format repair of saved judgements."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from judge_data import load_prepared
from judge_protocol import VERSION, digest, file_digest, read_json, save_json
from judge_runtime import execute_jobs, request_body, request_id
from hf_judge_repair import load_plan, parse_answer, scan_attempts, validate_runtime


def separate_output(output, *sources):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output folder; existing results are preserved')
    for source in sources:
        source = Path(source).resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Repair output must be separate from all input folders')


def inspect_sources(prepared, pack, results):
    from llm_judge import jobs_for_phase
    prepared, pack, results = map(Path, (prepared, pack, results))
    manifest, pairs, spaces = load_prepared(prepared)
    catalog = read_json(pack / 'h100_pack.json')
    fingerprints = {}
    def capture(path):
        fingerprints[str(path.resolve())] = file_digest(path)
        return read_json(path)
    capture(pack / 'h100_pack.json')
    for name in ('manifest.json', 'pairs.json', 'label_spaces.json'):
        capture(prepared / name)
    entries, runtimes, bundles = [], [], {}
    for phase in ('control', 'development', 'final'):
        folder = catalog['phases'][phase]
        bundle_path = (pack / folder / 'bundle.json').resolve()
        if not bundle_path.is_relative_to(pack.resolve()):
            raise ValueError('Bundle path escapes the standalone folder')
        bundle = capture(bundle_path)
        if bundle['prepared_signature'] != digest(manifest) or bundle['protocol'] != VERSION:
            raise ValueError('Prepared inputs and source bundle differ')
        metadata = capture(results / phase / 'worker_metadata.json')
        if metadata['bundle_signature'] != digest(bundle) or metadata['worker_sha256'] != bundle['worker_sha256']:
            raise ValueError('Source worker or bundle identity changed')
        runtime = metadata['model_metadata']
        if (runtime['model'] != catalog['model'] or runtime['revision'] != catalog['model_revision']
                or runtime['dtype'] != 'bfloat16'):
            raise ValueError('Source model differs from the frozen BF16 configuration')
        runtimes.append(runtime)
        bundles[phase] = bundle
        jobs = jobs_for_phase(manifest, pairs, spaces, phase)
        exported = {job['job_id']: job for job in bundle['jobs']}
        expected_ids = {request_id(job, bundle['config']) for job in jobs}
        actual_ids = {p.stem for p in (results / phase / 'responses').glob('*.json')}
        if len(exported) != len(jobs) or set(exported) != expected_ids or actual_ids != expected_ids:
            raise ValueError('Source requests are missing, duplicated or unexpected')
        for job in jobs:
            identifier = request_id(job, bundle['config'])
            original = exported[identifier]
            if original['body'] != request_body(job, bundle['config']):
                raise ValueError('Original blinded request changed')
            path = results / phase / 'responses' / (identifier + '.json')
            record = capture(path)
            if (record['job_id'] != identifier or record['bundle_signature'] != digest(bundle)
                    or record['request_sha256'] != digest(original['body'])
                    or record['response_sha256'] != digest(record['response'])
                    or record['response'].get('id') != 'local_' + identifier
                    or record['response'].get('model') != runtime['model'] + '@' + runtime['revision']):
                raise ValueError('Source response identity or content changed')
            exported_job = dict(original, phase=phase)
            choice = record['response']['choices'][0]
            answer, method = parse_answer(exported_job, choice['message']['content'])
            if choice.get('finish_reason') != 'stop':
                answer, method = None, None
            entries.append(dict(job=job, exported=exported_job, record=record, path=path,
                                answer=answer, method=method, phase=phase))
    if len({digest(value) for value in runtimes}) != 1:
        raise ValueError('Original phases used different runtimes')
    return manifest, catalog, bundles, runtimes[0], entries, fingerprints


def prepare_repair(prepared, pack, results, output):
    prepared, pack, results, output = map(Path, (prepared, pack, results, output))
    separate_output(output, prepared, pack, results)
    manifest, catalog, bundles, metadata, entries, fingerprints = inspect_sources(prepared, pack, results)
    normalized = [entry for entry in entries if entry['method'] in ('unwrap_properties', 'unwrap_constants')]
    unresolved = [entry for entry in entries if entry['answer'] is None]
    script = Path(__file__).with_name('hf_judge_repair.py')
    validator = Path(__file__).with_name('judge_protocol.py')
    worker_hashes = {bundle['worker_sha256'] for bundle in bundles.values()}
    if len(worker_hashes) != 1:
        raise ValueError('Original phases used different worker implementations')
    plan = {'schema_version': 1, 'protocol': VERSION, 'prepared_signature': digest(manifest),
            'original_pack_signature': digest(catalog), 'model_metadata': metadata,
            'config': bundles['final']['config'], 'max_attempts': 3,
            'original_worker_sha256': next(iter(worker_hashes)),
            'bootstrap_sha256': file_digest(pack / 'h100_judge.py'),
            'runner_sha256': file_digest(script), 'protocol_sha256': file_digest(validator),
            'normalization_policy': 'lossless_properties_or_const_wrapper_v1',
            'retry_policy': 'format_only_first_usable_maximum_three_attempts',
            'jobs': [entry['exported'] for entry in unresolved]}
    plan['signature'] = digest(plan)
    audit = {'prepared_signature': digest(manifest), 'plan_signature': plan['signature'],
             'source_file_sha256': fingerprints, 'source_responses': len(entries),
             'retained': sum(entry['method'] == 'unchanged' for entry in entries),
             'normalized': len(normalized), 'needs_generation': len(unresolved),
             'recoveries': [{'job_id': entry['exported']['job_id'], 'phase': entry['phase'],
                            'method': entry['method'], 'answer': entry['answer'],
                            'source_record_sha256': file_digest(entry['path']),
                            'source_response_sha256': entry['record']['response_sha256']}
                           for entry in normalized]}
    output.mkdir(parents=True)
    portable = output / 'judge_format_repair'
    portable.mkdir()
    shutil.copyfile(script, portable / script.name)
    shutil.copyfile(validator, portable / validator.name)
    save_json(portable / 'repair_plan.json', plan)
    save_json(output / 'recovery_audit.json', audit)
    load_plan(portable / 'repair_plan.json')
    return {k: audit[k] for k in ('source_responses', 'retained', 'normalized', 'needs_generation')}


def import_repaired(prepared, pack, results, repair, output, retry_results=None):
    from llm_judge import summarize
    prepared, pack, results, repair, output = map(Path, (prepared, pack, results, repair, output))
    separate_output(output, prepared, pack, results, repair)
    manifest, catalog, bundles, baseline, entries, fingerprints = inspect_sources(prepared, pack, results)
    audit = read_json(repair / 'recovery_audit.json')
    plan = load_plan(repair / 'judge_format_repair/repair_plan.json')
    if (audit['source_file_sha256'] != fingerprints or audit['plan_signature'] != plan['signature']
            or plan['prepared_signature'] != digest(manifest)
            or plan['original_pack_signature'] != digest(catalog)):
        raise ValueError('Repair sources or plan changed')
    selected, retry_metadata = {}, None
    if retry_results is not None:
        retry_results = Path(retry_results)
        separate_output(output, retry_results)
        retry_metadata = read_json(retry_results / 'repair_metadata.json')
        if retry_metadata['plan_signature'] != plan['signature']:
            raise ValueError('Retry responses belong to another repair plan')
        validate_runtime(plan, retry_metadata['model_metadata'])
        expected_ids = {job['job_id'] for job in plan['jobs']}
        attempts = retry_results / 'attempts'
        if attempts.exists() and any(p.name not in expected_ids for p in attempts.iterdir()):
            raise ValueError('Unexpected request folder in retry results')
        for job in plan['jobs']:
            result, _ = scan_attempts(plan, job, retry_results, retry_metadata['model_metadata'])
            if result:
                selected[job['job_id']] = result
    methods = {'normalized': 0, 'regenerated': 0, 'unresolved': 0, 'retained': 0}
    available = {phase: [] for phase in bundles}
    responses = {phase: {} for phase in bundles}
    for entry in entries:
        identifier = entry['exported']['job_id']
        provenance = {'method': entry['method'], 'phase': entry['phase'], 'plan_signature': plan['signature'],
                      'original_source_record_sha256': file_digest(entry['path']),
                      'original_response_sha256': entry['record']['response_sha256'],
                      'original_request_sha256': digest(entry['exported']['body']),
                      'executed_request': entry['exported']['body']}
        response, answer = deepcopy(entry['record']['response']), entry['answer']
        if answer is not None:
            methods['retained' if entry['method'] == 'unchanged' else 'normalized'] += 1
        elif identifier in selected:
            retry = selected[identifier]
            answer, response = retry['answer'], deepcopy(retry['record']['response'])
            provenance.update(method='format_retry', retry_normalization=retry['method'],
                              executed_request=retry['record']['executed_request'],
                              executed_request_sha256=retry['record']['executed_request_sha256'],
                              retry_record_sha256=file_digest(retry['path']),
                              retry_response_sha256=retry['record']['response_sha256'],
                              retry_attempt=retry['record']['attempt'])
            methods['regenerated'] += 1
        else:
            methods['unresolved'] += 1
            continue
        if provenance['method'] != 'unchanged' and (provenance['method'] != 'format_retry'
                                                    or provenance['retry_normalization'] != 'unchanged'):
            response['choices'][0]['message']['content'] = json.dumps(answer, ensure_ascii=False)
        available[entry['phase']].append(entry['job'])
        responses[entry['phase']][identifier] = {'response': response, 'provenance': provenance}
    output.mkdir(parents=True)
    save_json(output / 'repair_import.json', dict(methods, plan_signature=plan['signature'],
              source_recovery_audit_sha256=file_digest(repair / 'recovery_audit.json'),
              original_runtime=baseline, retry_runtime=retry_metadata,
              interpretation='Post-hoc lossless wrapper extraction and explicitly logged format-only retries. First-attempt evidence is unchanged. Unresolved requests are pending, not inferred judgements.'))
    stats = {}
    for phase, bundle in bundles.items():
        metadata = dict(baseline, response_format_repair={
            'plan_signature': plan['signature'], 'normalization_policy': plan['normalization_policy'],
            'retry_policy': plan['retry_policy'], 'retry_metadata': retry_metadata,
            'importer_sha256': file_digest(Path(__file__)),
            'runner_sha256': plan['runner_sha256']})
        config = dict(bundle['config'], expected_response_model=baseline['model'] + '@' + baseline['revision'],
                      local_model_revision=baseline['revision'], local_worker_sha256=bundle['worker_sha256'],
                      local_model_metadata=metadata)
        stats[phase] = execute_jobs(available[phase], output / 'evaluations' / phase, config, digest(manifest),
                                   execute=True, max_calls=max(1, len(available[phase])),
                                   local_responses=responses[phase])
        if stats[phase]['invalid']:
            raise ValueError('A selected repair failed strict validation')
        stats[phase]['planned'] = len(bundle['jobs'])
        stats[phase]['pending'] = len(bundle['jobs']) - len(available[phase])
        stats[phase]['imported_responses'] = stats[phase].pop('new_calls')
        summarize(prepared, output / 'evaluations' / phase, output / 'reports' / phase, phase=phase)
    return dict(methods, phases=stats, network_requests_made=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'import'):
        command = commands.add_parser(name)
        for field in ('prepared', 'pack', 'results', 'output-dir'):
            command.add_argument('--' + field, type=Path, required=True)
        if name == 'import':
            command.add_argument('--repair-dir', type=Path, required=True)
            command.add_argument('--retry-results', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare_repair(args.prepared, args.pack, args.results, args.output_dir)
        else:
            result = import_repaired(args.prepared, args.pack, args.results, args.repair_dir,
                                     args.output_dir, args.retry_results)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
