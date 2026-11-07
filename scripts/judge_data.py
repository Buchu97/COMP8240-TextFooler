"""Prepare immutable, blinded judging inputs from completed attack outputs."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import random
from collections import Counter

from judge_protocol import (VERSION, SENTIMENT, TASKS, TASK_LABELS, TASK_CONTROLS, build_requests, csv_safe, digest,
                            file_digest, read_json, save_json, write_csv)

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def integer(value):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError('Class ID is not an integer')
    return int(number)


def read_source(source):
    dataset, task = source['dataset'], source['task']
    if task not in TASKS or not dataset or not dataset.replace('_', '').isalnum():
        raise ValueError('Invalid dataset name or task')
    run = Path(source['run_dir'])
    run = (ROOT / run).resolve() if not run.is_absolute() else run.resolve()
    kind = source['format']
    if kind not in ('type1', 'textattack'):
        raise ValueError('Unknown saved attack format')
    path = run / ('outcomes.csv' if kind == 'type1' else 'attack_results.csv')
    source_hash = file_digest(path)
    rows = read_csv(path)
    supporting = {}
    if kind == 'type1':
        summary = read_json(run / 'summary.json')
        if summary['status'] != 'completed' or summary['examples_planned'] != len(rows):
            raise ValueError('Source attack is incomplete')
        for name, expected in summary['artifacts_sha256'].items():
            if Path(name).name != name or file_digest(run / name) != expected:
                raise ValueError('Source attack artifact hash mismatch')
        metadata = read_json(run / 'run_metadata.json')
        label2id = metadata['label2id']
    else:
        label2id = source['label2id']
    if len(set(label2id.values())) != len(label2id) or not label2id:
        raise ValueError('Invalid source label mapping')
    labels = {int(value): name for name, value in label2id.items()}
    spaces = dict(TASK_LABELS[task]) if task in TASK_LABELS else dict(SENTIMENT) if task == 'sentiment' else {
        name: ('Neutral or no clear emotional impact.' if name == 'no_impact' else name.replace('_', ' '))
        for _, name in sorted(labels.items())}
    if set(spaces) != set(label2id):
        raise ValueError('Source task labels differ from the declared task label definitions')
    for name in ('summary.json', 'run_metadata.json', 'attack_summary.json', 'EXPERIMENT_METADATA.json'):
        if (run / name).is_file():
            supporting[str(run / name)] = file_digest(run / name)
    similarities = {}
    if kind == 'type1' and (run / 'semantic_similarity.csv').exists():
        semantic = read_json(run / 'semantic_similarity.json')
        if semantic['source_outcomes_sha256'] != source_hash:
            raise ValueError('Per-pair USE measurements belong to another source')
        for row in read_csv(run / 'semantic_similarity.csv'):
            index = integer(row['row'])
            score = float(row['cosine_similarity'])
            if index in similarities or not math.isfinite(score) or not -1 <= score <= 1:
                raise ValueError('Invalid per-pair USE measurement')
            similarities[index] = (str(row['source_id']), score)
        supporting.update({str(run / name): file_digest(run / name)
                           for name in ('semantic_similarity.csv', 'semantic_similarity.json')})
    counts, pairs = Counter(), []
    for index, row in enumerate(rows):
        outcome = row['result_type']
        if outcome not in ('Successful', 'Failed', 'Skipped'):
            raise ValueError('Unknown attack result')
        gold = integer(row['label' if kind == 'type1' else 'ground_truth_output'])
        original = integer(row['original_prediction' if kind == 'type1' else 'original_output'])
        changed = integer(row['perturbed_prediction' if kind == 'type1' else 'perturbed_output'])
        if any(value not in labels for value in (gold, original, changed)):
            raise ValueError('Source prediction or gold label is outside the task taxonomy')
        if ((outcome == 'Successful' and not (original == gold and changed != gold))
                or (outcome == 'Failed' and not (original == gold and changed == gold))
                or (outcome == 'Skipped' and original == gold)):
            raise ValueError('Attack outcome contradicts the saved predictions')
        if kind == 'type1' and (integer(row['row']) != index or row['category'] != labels[gold]):
            raise ValueError('Source row or category alignment mismatch')
        counts[outcome] += 1
        if outcome != 'Successful':
            continue
        left, right = row['original_text'], row['perturbed_text']
        if kind == 'textattack':
            left, right = [text.replace('[[', '').replace(']]', '') for text in (left, right)]
        if not left.strip() or not right.strip() or left == right:
            raise ValueError('Successful attack has an empty or unchanged text')
        source_id = str(row.get('source_id', index))
        score = None
        if similarities:
            if index not in similarities or similarities[index][0] != source_id:
                raise ValueError('Per-pair USE/source identity mismatch')
            score = similarities[index][1]
        pairs.append({'pair_id': dataset + '_' + digest([source_hash, index, left, right])[:16],
                      'dataset': dataset, 'task': task, 'source_row': index, 'source_id': source_id,
                      'original_text': left, 'perturbed_text': right, 'gold_label': labels[gold],
                      'original_prediction': original, 'perturbed_prediction': changed,
                      'use_cosine': score, 'original_token_truncated': row.get('original_token_truncated', 'unknown'),
                      'control': ''})
    if similarities and len(similarities) != counts['Successful']:
        raise ValueError('Per-pair USE population mismatch')
    if kind == 'textattack' and (run / 'attack_summary.json').exists():
        saved = read_json(run / 'attack_summary.json')['Attack Results']
        for outcome in ('Successful', 'Failed', 'Skipped'):
            if int(saved[f'Number of {outcome.lower()} attacks:']) != counts[outcome]:
                raise ValueError('Saved attack summary does not cover this CSV')
    if file_digest(path) != source_hash:
        raise ValueError('Source changed during preparation')
    eligible = counts['Successful'] + counts['Failed']
    info = {'source_csv': str(path), 'source_csv_sha256': source_hash, 'supporting_sha256': supporting,
            'examples': len(rows), 'eligible': eligible, 'successful': counts['Successful'],
            'failed': counts['Failed'], 'skipped': counts['Skipped'],
            'raw_asr_percent': 100 * counts['Successful'] / eligible if eligible else None,
            'label2id': label2id, 'label_descriptions': ('Task-specific label definitions.' if task in TASK_LABELS
                else 'Source category names; sentiment rubric supplied explicitly.')}
    return pairs, info, spaces


def balanced_sample(groups, size, seed):
    rng = random.Random(seed)
    groups = {name: rng.sample(rows, len(rows)) for name, rows in sorted(groups.items())}
    chosen = []
    while len(chosen) < size and any(groups.values()):
        for rows in groups.values():
            if rows and len(chosen) < size:
                chosen.append(rows.pop())
    return chosen


def prepare(config, output_dir, seed=8240, development_per_dataset=5, audit_size=50, final_per_dataset=None):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Judging preparation directory already exists')
    if development_per_dataset < 1 or audit_size < 1 or not config.get('sources'):
        raise ValueError('Preparation needs sources and positive development/audit sizes')
    if final_per_dataset is not None and (type(final_per_dataset) is not int or final_per_dataset < 1):
        raise ValueError('Final sample cap must be a positive integer')
    pairs, spaces, sources, groups = [], {}, {}, {}
    for source in config['sources']:
        name = source['dataset']
        if name in sources:
            raise ValueError('Duplicate dataset source')
        rows, info, label_space = read_source(source)
        source_dir = Path(info['source_csv']).parent
        if output_dir.is_relative_to(source_dir) or source_dir.is_relative_to(output_dir):
            raise ValueError('Judging output must be separate from attack sources')
        if len(rows) <= development_per_dataset:
            raise ValueError('Not enough successful attacks for separate development/final sets')
        rows = sorted(rows, key=lambda p: digest([seed, p['pair_id']]))
        final_population = len(rows) - development_per_dataset
        if final_per_dataset is not None:
            rows = rows[:development_per_dataset + final_per_dataset]
        for index, row in enumerate(rows):
            row['phase'] = 'development' if index < development_per_dataset else 'final'
        pairs.extend(rows)
        groups[name] = [r for r in rows if r['phase'] == 'final']
        info.update(final_population_after_development=final_population, final_selected=len(groups[name]))
        sources[name], spaces[name] = info, label_space
        control = dict(rows[0], pair_id='unchanged_' + rows[0]['pair_id'], phase='control',
                       perturbed_text=rows[0]['original_text'], control='unchanged', use_cosine=None,
                       expected_meaning='yes', expected_same_label='yes')
        pairs.append(control)
        if source['task'] in TASK_CONTROLS:
            reversal = TASK_CONTROLS[source['task']]
            pairs.append({'pair_id': 'control_label_reversal_' + name, 'dataset': name,
                'task': source['task'], 'source_row': None, 'source_id': 'constructed_control',
                **reversal, 'phase': 'control', 'control': 'task_label_reversal',
                'expected_meaning': 'no', 'expected_same_label': 'no', 'use_cosine': None})
    spaces['control_sentiment'] = dict(SENTIMENT)
    pairs.append({'pair_id': 'control_sentiment_reversal', 'dataset': 'control_sentiment',
                  'task': 'sentiment', 'source_row': None, 'source_id': 'constructed_control',
                  'original_text': 'The film was excellent.', 'perturbed_text': 'The film was terrible.',
                  'gold_label': 'positive', 'phase': 'control', 'control': 'sentiment_reversal',
                  'expected_meaning': 'no', 'expected_same_label': 'no', 'use_cosine': None})
    audit = balanced_sample(groups, audit_size, seed)
    output_dir.mkdir(parents=True)
    save_json(output_dir / 'pairs.json', pairs)
    save_json(output_dir / 'label_spaces.json', spaces)
    fields = ['pair_id', 'dataset', 'phase', 'source_id', 'original_text', 'perturbed_text', 'gold_label', 'use_cosine', 'control']
    write_csv(output_dir / 'pairs.csv', [{k: csv_safe(p.get(k, '')) for k in fields} for p in pairs], fields)
    audit_rows, audit_key = [], {}
    for p in audit:
        order = build_requests(p, spaces[p['dataset']], seed)['quality']['a_is_original']
        a, b = (p['original_text'], p['perturbed_text']) if order else (p['perturbed_text'], p['original_text'])
        record = {'pair_id': p['pair_id'], 'dataset': p['dataset'], 'text_a': csv_safe(a), 'text_b': csv_safe(b)}
        audit_key[p['pair_id']] = dict(record, a_is_original=order)
        audit_rows.append(dict(record, meaning_preserved='', label_a='', label_b='',
                               naturalness_a='', naturalness_b='', reviewer_id='', reviewed_at='', notes=''))
    write_csv(output_dir / 'human_review.csv', audit_rows, list(audit_rows[0]))
    save_json(output_dir / 'audit_key.json', audit_key)
    options = [{'dataset': name, 'label': label, 'description': definition}
               for name, labels in spaces.items() if name in sources for label, definition in labels.items()]
    write_csv(output_dir / 'human_label_options.csv', options, ['dataset', 'label', 'description'])
    guide = [{'field': 'meaning_preserved', 'allowed': 'yes | no | uncertain', 'definition': 'Same assertions, sentiment, intent and relevant details; topic overlap alone is insufficient.'},
             {'field': 'label_a / label_b', 'allowed': 'A label from human_label_options.csv | __uncertain__', 'definition': 'Assign each text its task label; do not consult classifier or LLM outputs.'},
             {'field': 'naturalness_a / naturalness_b', 'allowed': '1 | 2 | 3 | 4 | 5', 'definition': '1 unreadable; 2 major errors; 3 understandable but awkward; 4 minor issues; 5 natural for the genre.'},
             {'field': 'reviewer_id / reviewed_at', 'allowed': 'Reviewer identifier / ISO date', 'definition': 'Record the actual reviewer and review date. Leave unreviewed decisions blank.'}]
    write_csv(output_dir / 'human_review_guide.csv', guide, ['field', 'allowed', 'definition'])
    manifest = {'schema_version': 1, 'protocol': VERSION, 'seed': seed, 'sources': sources,
                'source_config': config, 'development_per_dataset': development_per_dataset,
                'final_per_dataset': final_per_dataset,
                'audit_requested': audit_size, 'audit_ids': [p['pair_id'] for p in audit],
                'selection': ('All saved successes, with a deterministic separate development set; balanced random audit across datasets.'
                    if final_per_dataset is None else
                    'Deterministic uniform hash-ranked successful attacks within each dataset, separate development pairs followed by a capped final sample; balanced random human audit from the selected final pairs. Sample results are not a full-population accepted attack rate.'),
                'judging_status': 'not_started', 'human_review_status': 'blank_template',
                'immutable_sha256': {name: file_digest(output_dir / name) for name in
                                     ('pairs.json', 'label_spaces.json', 'audit_key.json', 'human_label_options.csv', 'human_review_guide.csv')}}
    save_json(output_dir / 'manifest.json', manifest)
    return manifest


def load_prepared(directory):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    if manifest.get('protocol') != VERSION:
        raise ValueError('Prepared protocol differs from this implementation; prepare a new tagged dataset')
    for name, expected in manifest['immutable_sha256'].items():
        if Path(name).name != name or file_digest(directory / name) != expected:
            raise ValueError('Prepared judging input hash mismatch')
    for source in manifest['sources'].values():
        files = dict(source['supporting_sha256'], **{source['source_csv']: source['source_csv_sha256']})
        for name, expected in files.items():
            if file_digest(name) != expected:
                raise ValueError('Attack source changed after judging preparation')
    return manifest, read_json(directory / 'pairs.json'), read_json(directory / 'label_spaces.json')
