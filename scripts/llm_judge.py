"""Prepare, run and analyse blinded semantic judgements of saved TextFooler attacks."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
import math
from pathlib import Path

from judge_data import load_prepared, prepare, read_csv
from judge_protocol import (UNCERTAIN, assess, build_requests, csv_safe, digest, file_digest,
                            read_json, save_json, write_csv)
from judge_runtime import (evaluation_settings, execute_jobs, finalize, request_body, request_id)


def jobs_for_phase(manifest, pairs, spaces, phase):
    jobs = []
    for pair in pairs:
        if pair['phase'] != phase:
            continue
        for kind, request in build_requests(pair, spaces[pair['dataset']], manifest['seed']).items():
            jobs.append({'pair_id': pair['pair_id'], 'kind': kind, 'request': request,
                         'labels': list(spaces[pair['dataset']])})
    return jobs


def agreement(decisions):
    resolved = [(a, b) for a, b in decisions if a not in (None, 'uncertain', UNCERTAIN)
                and b not in (None, 'uncertain', UNCERTAIN)]
    n = len(resolved)
    observed = sum(a == b for a, b in resolved) / n if n else None
    chance = None
    if n:
        left, right = Counter(a for a, _ in resolved), Counter(b for _, b in resolved)
        chance = sum(left[k] * right[k] for k in set(left) | set(right)) / n**2
    return {'n': n, 'unresolved': len(decisions) - n,
            'agreement_percent': observed * 100 if n else None,
            'cohens_kappa': (observed - chance) / (1 - chance) if n and chance < 1 else None}


def protocol_diagnostics(rows):
    paired_and_single = []
    for row in rows:
        original, edited = row['original_judge_label'], row['perturbed_judge_label']
        if UNCERTAIN in (original, edited):
            single = 'uncertain'
        else:
            single = 'yes' if original == edited else 'no'
        paired_and_single.append((row['same_task_label'], single))
    return {
        'meaning_yes': sum(row['meaning_preserved'] == 'yes' for row in rows),
        'original_label_disagreements': sum(not row['original_label_matches_gold'] for row in rows),
        'uncertain_from_original_label': sum(
            row['status'] == 'uncertain' and not row['original_label_matches_gold'] for row in rows
        ),
        'paired_vs_single_label_preservation': agreement(paired_and_single),
    }


def human_comparison(prepared, human_review, pairs_by_id, judged_by_id, spaces):
    key = read_json(Path(prepared) / 'audit_key.json')
    rows = read_csv(human_review)
    if len(rows) != len(key) or {r['pair_id'] for r in rows} != set(key):
        raise ValueError('Human review IDs do not match the fixed audit sample')
    if len({r['pair_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate human review row')
    meaning, label_preservation, classifications, naturalness = [], [], [], []
    complete = shared = partial = 0
    for row in rows:
        expected = key[row['pair_id']]
        if any(row[name] != expected[name] for name in ('dataset', 'text_a', 'text_b')):
            raise ValueError('Human review text or dataset identity changed')
        fields = ('meaning_preserved', 'label_a', 'label_b', 'naturalness_a', 'naturalness_b', 'reviewer_id', 'reviewed_at')
        if not all(row.get(name, '').strip() for name in fields):
            partial += int(any(row.get(name, '').strip() for name in fields))
            continue
        if row['meaning_preserved'].lower() not in ('yes', 'no', 'uncertain'):
            raise ValueError('Invalid human meaning decision')
        date.fromisoformat(row['reviewed_at'])
        labels = {name.casefold(): name for name in spaces[row['dataset']]}
        labels.update({UNCERTAIN: UNCERTAIN, 'uncertain': UNCERTAIN})
        try:
            a, b = [labels[row[name].strip().casefold()] for name in ('label_a', 'label_b')]
            na, nb = [int(row[name]) for name in ('naturalness_a', 'naturalness_b')]
        except (KeyError, ValueError):
            raise ValueError('Invalid human label or naturalness rating') from None
        if not 1 <= na <= 5 or not 1 <= nb <= 5:
            raise ValueError('Human naturalness must be from 1 to 5')
        complete += 1
        judged = judged_by_id.get(row['pair_id'])
        if not judged or judged['status'] not in ('accepted', 'rejected', 'uncertain'):
            continue
        shared += 1
        original, changed = (a, b) if expected['a_is_original'] else (b, a)
        meaning.append((row['meaning_preserved'].lower(), judged['meaning_preserved']))
        same = 'uncertain' if UNCERTAIN in (a, b) else ('yes' if a == b else 'no')
        label_preservation.append((same, judged['same_task_label']))
        classifications.extend([(original, judged['original_judge_label']), (changed, judged['perturbed_judge_label'])])
        naturalness.append((nb if expected['a_is_original'] else na, judged['perturbed_naturalness']))
    return {'completed_reviews': complete, 'shared_reviews': shared, 'pending_reviews': len(rows) - complete,
            'partially_filled_reviews': partial, 'human_source_sha256': file_digest(human_review),
            'sampling': 'Balanced fixed audit across datasets; pooled agreement describes this sample, not population-weighted accuracy.',
            'meaning': agreement(meaning), 'label_preservation': agreement(label_preservation),
            'independent_classification': agreement(classifications),
            'perturbed_naturalness': {'n': len(naturalness),
                'mean_absolute_difference': sum(abs(a-b) for a, b in naturalness) / len(naturalness) if naturalness else None}}


def safe_output(manifest, prepared, output):
    output, prepared = Path(output).resolve(), Path(prepared).resolve()
    if output == prepared or prepared.is_relative_to(output):
        raise ValueError('Output cannot replace or contain the prepared dataset')
    for source in manifest['sources'].values():
        original = Path(source['source_csv']).parent.resolve()
        if output == original or output.is_relative_to(original) or original.is_relative_to(output):
            raise ValueError('Output must remain separate from attack inputs')


def summarize(prepared, evaluation, report_dir, phase='final', human_review=None, use_threshold=0.8):
    prepared, evaluation, report_dir = Path(prepared), Path(evaluation), Path(report_dir)
    if report_dir.exists():
        raise FileExistsError('Report directory already exists; use a new report tag')
    if not math.isfinite(use_threshold) or not -1 <= use_threshold <= 1:
        raise ValueError('Invalid diagnostic USE cutoff')
    manifest, pairs, spaces = load_prepared(prepared)
    safe_output(manifest, prepared, report_dir)
    settings = read_json(evaluation / 'evaluation.json')
    config = settings['backend']
    if settings != evaluation_settings(config, digest(manifest)):
        raise ValueError('Evaluation provenance differs from these prepared inputs or implementation')
    selected = [p for p in pairs if p['phase'] == phase]
    jobs = jobs_for_phase(manifest, pairs, spaces, phase)
    answers, states, artifacts = {}, {}, {}
    for job in jobs:
        identifier = request_id(job, config)
        path = evaluation / 'requests' / (identifier + '.json')
        if not path.exists():
            states[(job['pair_id'], job['kind'])] = 'pending'
            continue
        record = read_json(path)
        if (record['job_id'] != identifier or record['job_signature'] != digest(job)
                or record['request'] != request_body(job, config)):
            raise ValueError('Saved request identity or payload changed')
        artifacts[path.name] = file_digest(path)
        if record['status'] in ('completed', 'invalid', 'received'):
            original_answer = record.get('answer')
            original_status = record['status']
            record = finalize(record, job, config)
            if original_status == 'completed' and (record['status'] != 'completed' or record['answer'] != original_answer):
                raise ValueError('Saved parsed judgement differs from its provider response')
            if record['status'] == 'completed':
                answers[(job['pair_id'], job['kind'])] = record['answer']
        states[(job['pair_id'], job['kind'])] = record['status']
    judged = []
    for pair in selected:
        row = dict(pair)
        kinds = ('quality', 'original_label', 'perturbed_label')
        pair_states = [states[(pair['pair_id'], kind)] for kind in kinds]
        row['status'] = ('invalid_judgement' if 'invalid' in pair_states else
                         'unknown_request' if any(s in ('unknown', 'reserved') for s in pair_states) else 'pending')
        if all(s == 'completed' for s in pair_states):
            q, original, changed = [answers[(pair['pair_id'], kind)] for kind in kinds]
            order = build_requests(pair, spaces[pair['dataset']], manifest['seed'])['quality']['a_is_original']
            row.update(assess(pair, q, original, changed, order))
            row.update({key: q[key] for key in ('meaning_reason', 'label_reason', 'naturalness_reason')})
            row['original_label_reason'], row['perturbed_label_reason'] = original['reason'], changed['reason']
        judged.append(row)
    by_id = {p['pair_id']: p for p in judged}
    human = human_comparison(prepared, human_review or prepared / 'human_review.csv',
                             {p['pair_id']: p for p in pairs}, by_id, spaces)
    datasets, diagnostics = {}, {}
    for name in sorted({p['dataset'] for p in selected if not p['control']}):
        rows = [p for p in judged if p['dataset'] == name and not p['control']]
        complete = [p for p in rows if p['status'] in ('accepted', 'rejected', 'uncertain')]
        diagnostics[name] = protocol_diagnostics(complete)
        counts = Counter(p['status'] for p in rows)
        high = [p for p in complete if p['use_cosine'] is not None and p['use_cosine'] >= use_threshold]
        source = manifest['sources'][name]
        datasets[name] = {'selected': len(rows), 'judged': len(complete),
            'accepted': counts['accepted'], 'rejected': counts['rejected'], 'uncertain': counts['uncertain'],
            'unjudged': len(rows) - len(complete), 'invalid_judgements': counts['invalid_judgement'],
            'raw_attack_successes': source['successful'], 'raw_eligible_attempts': source['eligible'],
            'raw_attack_asr_percent': source['raw_asr_percent'],
            'judge_acceptance_percent': 100 * counts['accepted'] / len(complete) if complete else None,
            'mean_perturbed_naturalness': sum(p['perturbed_naturalness'] for p in complete) / len(complete) if complete else None,
            'high_use_judged': len(high), 'high_use_rejected': sum(p['status'] == 'rejected' for p in high),
            'high_use_uncertain': sum(p['status'] == 'uncertain' for p in high),
            'full_population_judge_accepted_asr_percent': (100 * counts['accepted'] / source['eligible']
                if len(complete) == len(rows) == source['successful'] and source['eligible'] else None)}
    controls = [p for p in judged if p['control']]
    control_ready = [p for p in controls if p['status'] in ('accepted', 'rejected', 'uncertain')]
    report = {'phase': phase, 'prepared_manifest_sha256': file_digest(prepared / 'manifest.json'),
              'evaluation_manifest_sha256': file_digest(evaluation / 'evaluation.json'),
              'request_artifacts_sha256': artifacts, 'datasets': datasets, 'human_comparison': human,
              'protocol_diagnostics': diagnostics,
              'controls': {'selected': len(controls), 'judged': len(control_ready),
                           'passed_pair_checks': sum(p['meaning_preserved'] == p['expected_meaning']
                               and p['same_task_label'] == p['expected_same_label'] for p in control_ready)},
              'definitions': {'judge_acceptance': 'Accepted / all fully judged selected attacks, including uncertain cases in the denominator; not human-verified truth.',
                              'naturalness': 'Separate 1-5 genre-aware rating, not part of semantic-label acceptance.',
                              'high_use_cutoff': use_threshold, 'high_use_cutoff_role': 'Diagnostic comparison only; not the attack word threshold or a semantic validity guarantee.',
                              'sampling': manifest['selection']}}
    report_dir.mkdir(parents=True)
    fields = ['pair_id', 'dataset', 'phase', 'source_id', 'original_text', 'perturbed_text', 'gold_label',
              'status', 'meaning_preserved', 'same_task_label', 'original_judge_label', 'perturbed_judge_label',
              'original_label_matches_gold', 'original_naturalness', 'perturbed_naturalness', 'use_cosine',
              'original_token_truncated', 'control', 'meaning_reason', 'label_reason', 'naturalness_reason',
              'original_label_reason', 'perturbed_label_reason']
    display = [{k: csv_safe(row[k]) if row.get(k) is not None else '' for k in fields} for row in judged]
    write_csv(report_dir / 'judged_pairs.csv', display, fields)
    write_csv(report_dir / 'dataset_summary.csv', [dict(dataset=k, **v) for k, v in datasets.items()],
              ['dataset'] + list(next(iter(datasets.values())).keys()) if datasets else ['dataset'])
    write_csv(report_dir / 'high_use_disagreements.csv', [row for row, pair in zip(display, judged)
              if not pair['control'] and pair.get('use_cosine') is not None and pair['use_cosine'] >= use_threshold
              and pair['status'] in ('rejected', 'uncertain')], fields)
    save_json(report_dir / 'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--sources', type=Path, required=True)
    prep.add_argument('--output-dir', type=Path, required=True)
    prep.add_argument('--seed', type=int, default=8240)
    prep.add_argument('--development-per-dataset', type=int, default=5)
    prep.add_argument('--audit-size', type=int, default=50)
    prep.add_argument('--final-per-dataset', type=int, help='Fixed final sample cap per dataset; default retains every remaining success')
    for name in ('plan', 'run', 'summarize'):
        command = commands.add_parser(name)
        command.add_argument('--prepared', type=Path, required=True)
        command.add_argument('--phase', choices=['development', 'final', 'control'], default='development' if name != 'summarize' else 'final')
        if name != 'plan':
            command.add_argument('--evaluation', type=Path, required=True)
        if name == 'run':
            command.add_argument('--config', type=Path, required=True)
            command.add_argument('--execute', action='store_true')
            command.add_argument('--budget-usd', type=float, default=0)
            command.add_argument('--max-calls', type=int, default=0)
        if name == 'summarize':
            command.add_argument('--report-dir', type=Path, required=True)
            command.add_argument('--human-review', type=Path)
            command.add_argument('--use-threshold', type=float, default=0.8)
    local_export = commands.add_parser('export-local')
    local_export.add_argument('--prepared', type=Path, required=True)
    local_export.add_argument('--output-dir', type=Path, required=True)
    local_export.add_argument('--phase', choices=['development', 'final', 'control'], default='development')
    local_export.add_argument('--model', required=True)
    h100_export = commands.add_parser('export-h100')
    h100_export.add_argument('--prepared', type=Path, required=True)
    h100_export.add_argument('--output-dir', type=Path, required=True)
    h100_export.add_argument('--model', required=True)
    h100_export.add_argument('--include-final', action='store_true')
    h100_export.add_argument('--model-revision')
    local_import = commands.add_parser('import-local')
    local_import.add_argument('--prepared', type=Path, required=True)
    local_import.add_argument('--bundle', type=Path, required=True)
    local_import.add_argument('--results-dir', type=Path, required=True)
    local_import.add_argument('--evaluation', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'export-h100':
            from judge_local import export_h100
            print(json.dumps(export_h100(args.prepared, args.output_dir, args.model, args.include_final, args.model_revision), indent=2))
            return
        if args.command == 'export-local':
            from judge_local import export_local
            print(json.dumps(export_local(args.prepared, args.output_dir, args.phase, args.model), indent=2))
            return
        if args.command == 'import-local':
            from judge_local import import_local
            print(json.dumps(import_local(args.prepared, args.bundle, args.results_dir, args.evaluation), indent=2))
            return
        if args.command == 'prepare':
            manifest = prepare(read_json(args.sources), args.output_dir, args.seed,
                               args.development_per_dataset, args.audit_size, args.final_per_dataset)
            print(json.dumps({'prepared': str(args.output_dir.resolve()), 'datasets': list(manifest['sources']),
                              'human_audit_pairs': len(manifest['audit_ids']), 'live_judgements': 0}, indent=2))
            return
        manifest, pairs, spaces = load_prepared(args.prepared)
        jobs = jobs_for_phase(manifest, pairs, spaces, args.phase)
        if args.command == 'plan':
            print(json.dumps({'phase': args.phase, 'pairs': len(jobs) // 3, 'requests': len(jobs),
                              'by_dataset': dict(Counter(p['dataset'] for p in pairs if p['phase'] == args.phase)),
                              'live_requests_made': 0, 'human_audit_pairs': len(manifest['audit_ids'])}, indent=2))
        elif args.command == 'run':
            safe_output(manifest, args.prepared, args.evaluation)
            result = execute_jobs(jobs, args.evaluation, read_json(args.config), digest(manifest),
                                  execute=args.execute, budget_usd=args.budget_usd, max_calls=args.max_calls)
            print(json.dumps(result, indent=2))
        else:
            report = summarize(args.prepared, args.evaluation, args.report_dir, args.phase, args.human_review, args.use_threshold)
            print(json.dumps({'phase': args.phase, 'datasets': report['datasets'],
                              'human_reviews': report['human_comparison']['completed_reviews']}, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
