"""Prepare a complete human-reviewed Type2 set for fixed-order evaluation only.

Original canonical/reviewer/adjudication files must remain available for replay.
The explicit --min-reviewers policy must match the import report. Hashes prove
file consistency; they cannot establish reviewer identity or independence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

from import_type2_reviews import import_reviews


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'notes/type2_execution_20261103.json'
LABELS = {'negative': 0, 'positive': 1}
IMPORT_ARTIFACTS = ('reviewed_samples.csv', 'reviewed_metadata.json',
    'unresolved_samples.json', 'human_review_audit.csv', 'unresolved_review.csv')
META_FIELDS = ('row', 'source_id', 'source_row', 'label_name', 'text_sha256', 'words')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f'nonfinite JSON value: {value}')

    return json.loads(Path(path).read_text(encoding='utf-8'),
                      object_pairs_hook=pairs, parse_constant=constant)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def verify_import(import_dir, min_reviewers, *, allow_partial=False):
    """Replay existing validation rather than deriving or resolving any label."""
    import_dir = Path(import_dir)
    if type(min_reviewers) is not int or min_reviewers < 1:
        raise ValueError('min_reviewers must be an explicit positive integer')
    report = read_json(import_dir / 'import_report.json')
    if report.get('schema_version') != 1 or report.get('min_reviewers') != min_reviewers:
        raise ValueError('Unsupported import schema or min_reviewers policy mismatch')
    if set(report.get('artifacts_sha256', {})) != set(IMPORT_ARTIFACTS):
        raise ValueError('Import artifact inventory mismatch')
    for name, expected in report['artifacts_sha256'].items():
        if sha256(import_dir / name) != expected:
            raise ValueError(f'Import artifact hash mismatch: {name}')
    canonical = report.get('canonical_inputs', [])
    reviews = report.get('review_inputs', [])
    adjudication = report.get('adjudication_input')
    if len(canonical) != 2 or not reviews:
        raise ValueError('Missing original canonical/reviewer source provenance')
    for source in canonical + reviews + ([adjudication] if adjudication else []):
        if sha256(source['path']) != source['sha256']:
            raise ValueError(f'Original source hash mismatch: {source["path"]}')
    with tempfile.TemporaryDirectory(prefix='type2_import_replay_') as directory:
        replay = import_reviews(canonical[0]['path'], canonical[1]['path'],
            [source['path'] for source in reviews], Path(directory) / 'replayed',
            min_reviewers=min_reviewers,
            adjudication=adjudication['path'] if adjudication else None)
    # Creation time is the only intentionally nondeterministic importer field.
    original_core = {k: v for k, v in report.items() if k != 'created_at_utc'}
    replay_core = {k: v for k, v in replay.items() if k != 'created_at_utc'}
    if original_core != replay_core:
        raise ValueError('Import provenance replay differs from the original report/artifacts')
    records = read_json(import_dir / 'reviewed_metadata.json')
    counts = report['counts']
    if not records or len(records) != counts['eligible_samples']:
        raise ValueError('Evaluation requires a nonempty resolved review population')
    if not allow_partial and (counts['unresolved_samples'] != 0
            or counts['eligible_samples'] != counts['canonical_samples']):
        raise ValueError('Evaluation requires a nonempty, fully resolved canonical review population')
    return records, report


def metadata_rows(records):
    names = {index: name for name, index in LABELS.items()}
    return [dict(row=str(index), source_id=record['sample_id'],
        source_row=str(record['source_metadata']['sample_row']),
        label_name=names[record['label']],
        text_sha256=record['source_metadata']['text_sha256'],
        words=str(len(record['text'].split()))) for index, record in enumerate(records)]


def prepare_evaluation(import_dir, output_dir, *, experiment_tag, min_reviewers, allow_partial=False):
    import_dir, output_dir = Path(import_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')
    if not isinstance(experiment_tag, str) or not re.fullmatch(r'TYPE2_[A-Z0-9_]+', experiment_tag):
        raise ValueError('experiment_tag must start TYPE2_ and contain only uppercase letters/digits/underscores')
    if allow_partial and not experiment_tag.startswith('TYPE2_PILOT_'):
        raise ValueError('Partial mode requires an explicit TYPE2_PILOT_ experiment tag')
    records, report = verify_import(import_dir, min_reviewers, allow_partial=allow_partial)
    config = read_json(CONFIG)
    metadata = metadata_rows(records)
    output_dir.mkdir(parents=True, exist_ok=False)
    saved_import = output_dir / 'review_import'
    saved_import.mkdir()
    for name in (*IMPORT_ARTIFACTS, 'import_report.json'):
        shutil.copyfile(import_dir / name, saved_import / name)
    shutil.copyfile(import_dir / 'reviewed_samples.csv', output_dir / 'evaluation.csv')
    with (output_dir / 'evaluation_metadata.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(metadata)
    manifest = dict(schema_version=1, dataset=experiment_tag, dataset_type='type2',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        label2id=LABELS, label_status='human_reviewed',
        train_test_split_created=False, partial_pilot=allow_partial,
        evaluation=dict(examples=len(records), sha256=sha256(output_dir / 'evaluation.csv'),
            label_counts=dict(sorted(Counter(r['label_name'] for r in metadata).items())),
            coverage_fraction=len(records) / report['counts']['canonical_samples'],
            selection=('All importer-eligible samples; unresolved samples explicitly excluded in a partial pilot' if allow_partial else
                'All canonical samples, fully resolved under the explicit importer policy') +
                '; canonical order; no correctness filtering, additional sampling or shuffle'),
        model=config['model'], attack=config['attack'],
        checkpoint_training_membership='Unknown; no claim of source independence from the published checkpoint training corpus',
        review_provenance=dict(import_report_sha256=sha256(import_dir / 'import_report.json'),
            min_reviewers=min_reviewers, reviewer_ids=report['reviewer_ids'],
            counts=report['counts'],
            agreement_before_adjudication=report['agreement_before_adjudication'],
            identity_limit=report['rules']['reviewer_independence'],
            validation='Original source hashes verified and existing importer replay matched all deterministic report fields and artifacts'),
        execution_config_sha256=sha256(CONFIG),
        human_or_llm_adversarial_semantic_judging_completed=False)
    manifest['artifacts_sha256'] = {
        path.relative_to(output_dir).as_posix(): sha256(path)
        for path in sorted(output_dir.rglob('*')) if path.is_file()}
    write_json(output_dir / 'manifest.json', manifest)
    return manifest


def validate_prepared(dataset_dir, *, allow_partial=False):
    """Verify both byte-level integrity and agreement with the replayed reviews."""
    dataset_dir = Path(dataset_dir)
    manifest = read_json(dataset_dir / 'manifest.json')
    config = read_json(CONFIG)
    if (manifest.get('schema_version') != 1 or manifest.get('dataset_type') != 'type2'
            or manifest.get('label2id') != LABELS or manifest.get('label_status') != 'human_reviewed'
            or manifest.get('train_test_split_created') is not False
            or not re.fullmatch(r'TYPE2_[A-Z0-9_]+', manifest.get('dataset', ''))
            or manifest.get('model') != config['model'] or manifest.get('attack') != config['attack']
            or manifest.get('execution_config_sha256') != sha256(CONFIG)):
        raise ValueError('Prepared manifest/configuration mismatch')
    partial = manifest.get('partial_pilot')
    if type(partial) is not bool or (partial and
            (not allow_partial or not manifest['dataset'].startswith('TYPE2_PILOT_'))):
        raise ValueError('Partial pilot requires explicit --allow-partial and a TYPE2_PILOT_ tag')
    expected_names = {'evaluation.csv', 'evaluation_metadata.csv'} | {
        f'review_import/{name}' for name in (*IMPORT_ARTIFACTS, 'import_report.json')}
    if set(manifest.get('artifacts_sha256', {})) != expected_names:
        raise ValueError('Prepared artifact inventory mismatch')
    for name, expected in manifest['artifacts_sha256'].items():
        if sha256(dataset_dir / name) != expected:
            raise ValueError(f'Prepared artifact hash mismatch: {name}')
    provenance = manifest['review_provenance']
    records, report = verify_import(dataset_dir / 'review_import', provenance['min_reviewers'],
                                    allow_partial=partial)
    if (sha256(dataset_dir / 'review_import/import_report.json') != provenance['import_report_sha256']
            or provenance['reviewer_ids'] != report['reviewer_ids']
            or provenance['counts'] != report['counts']
            or provenance['agreement_before_adjudication'] != report['agreement_before_adjudication']):
        raise ValueError('Prepared review provenance mismatch')
    if (dataset_dir / 'evaluation.csv').read_bytes() != (dataset_dir / 'review_import/reviewed_samples.csv').read_bytes():
        raise ValueError('Evaluation bytes differ from verified human labels/text')
    with (dataset_dir / 'evaluation_metadata.csv').open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != list(META_FIELDS) or list(reader) != metadata_rows(records):
            raise ValueError('Evaluation metadata alignment differs from verified sample IDs/provenance')
    if (manifest['evaluation']['examples'] != len(records)
            or manifest['evaluation']['sha256'] != sha256(dataset_dir / 'evaluation.csv')
            or manifest['evaluation']['coverage_fraction'] != len(records) / report['counts']['canonical_samples']
            or manifest['evaluation']['label_counts'] != dict(Counter(r['label_name'] for r in metadata_rows(records)))):
        raise ValueError('Evaluation population/count/hash mismatch')
    return manifest


def verify_local_assets():
    """Read and fingerprint existing assets only; no model imports or downloads."""
    config = read_json(CONFIG)
    for relative, expected in config['required_files_sha256'].items():
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Asset path escapes project root')
        if sha256(path) != expected:
            raise ValueError(f'Local asset hash mismatch: {relative}')
    model_config = read_json(ROOT / config['model']['snapshot_path'] / 'config.json')
    if (model_config.get('model_type') != 'bert'
            or model_config.get('architectures') != ['BertForSequenceClassification']
            or model_config.get('finetuning_task') != 'rotten_tomatoes'
            or 'id2label' in model_config or 'label2id' in model_config):
        raise ValueError('Cached sentiment classifier configuration differs from verified label contract')
    dataset_info = read_json(ROOT / config['label_mapping_evidence']['cached_dataset_info'])
    if dataset_info['features']['label']['names'] != ['neg', 'pos']:
        raise ValueError('Cached Rotten Tomatoes label order mismatch')
    anchor_record = read_json(ROOT / config['label_mapping_evidence']['previous_anchor_record'])
    if (anchor_record['model_revision'] != config['model']['revision']
            or anchor_record['label_map_anchor_predictions'] != [0, 1]):
        raise ValueError('Previous sentiment anchor evidence mismatch')
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--experiment-tag', required=True)
    parser.add_argument('--min-reviewers', type=int, required=True)
    parser.add_argument('--allow-partial', action='store_true',
        help='Explicit progress pilot containing only importer-eligible samples; requires TYPE2_PILOT_ tag')
    args = parser.parse_args(argv)
    try:
        manifest = prepare_evaluation(args.import_dir, args.output_dir,
            experiment_tag=args.experiment_tag, min_reviewers=args.min_reviewers, allow_partial=args.allow_partial)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Preparation rejected: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
