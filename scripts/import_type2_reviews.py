"""Validate human Type 2 annotations without deriving labels from source data.

Example: python scripts/import_type2_reviews.py --samples samples.csv
  --metadata sample_metadata.csv --reviews reviewer_a.csv reviewer_b.csv
  --output-dir outputs/import_name --min-reviewers 2

Review CSV columns match the original human_review.csv. Polarity names are
case-insensitive and surrounding whitespace is removed; numeric labels and
other synonyms are rejected. Statuses are exactly pending, reviewed, ambiguous.
Completed reviews require a reviewer ID and ISO date or timezone-aware datetime.
Ambiguous rows have no polarity label; pending rows have no label or review date.
Each export represents one reviewer (blank IDs on pending rows are allowed).

Optional adjudication columns: sample_id,text,human_label,adjudicator_id,
adjudicated_at,reason. Only conflicts/ambiguity with at least two completed
reviewers may be adjudicated, by a different reviewer with a nonempty reason.
Agreement is always calculated from original polarity reviews before adjudication.

Human-facing audit CSV cells are formula-escaped. They are display artifacts,
not round-trip input: raw values remain in JSON. Model CSV text is unchanged.
"""

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
import re
import sys


REVIEW_FIELDS = ('sample_id', 'text', 'human_label', 'review_status',
                 'reviewer_id', 'reviewed_at', 'notes')
ADJUDICATION_FIELDS = ('sample_id', 'text', 'human_label', 'adjudicator_id',
                       'adjudicated_at', 'reason')
LABELS = {'negative': 0, 'positive': 1}


def _read_csv(path, required):
    path = Path(path)
    contents = path.read_bytes()
    try:
        reader = csv.DictReader(io.StringIO(contents.decode('utf-8-sig'), newline=''), strict=True)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)) or any(not key for key in fields):
            raise ValueError('missing or duplicate column headers')
        if set(required) - set(fields):
            raise ValueError(f'missing columns: {sorted(set(required) - set(fields))}')
        rows = []
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'ragged CSV row {number}')
            rows.append(row)
    except (UnicodeError, csv.Error, ValueError) as exc:
        raise ValueError(f'{path}: {exc}') from exc
    return rows, {'path': str(path.resolve()), 'sha256': hashlib.sha256(contents).hexdigest(),
                  'rows': len(rows), 'columns': fields}


def _identity(value, field, required=True):
    if (required and not value) or value != value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError(f'invalid {field}: a nonempty, unpadded ID is required')


def _review_date(value, field):
    try:
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            return date.fromisoformat(value)
        elif re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)', value):
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if parsed.utcoffset() is None:
                raise ValueError('timezone required')
            return parsed
        else:
            raise ValueError('ISO date or timezone-aware datetime required')
    except ValueError as exc:
        raise ValueError(f'invalid {field}: {value!r}') from exc


def _label(value):
    normalized = value.strip().casefold()
    if normalized not in LABELS:
        raise ValueError(f'invalid human_label: {value!r}; expected Positive or Negative')
    return LABELS[normalized]


def _canonical(samples_path, metadata_path):
    samples, sample_input = _read_csv(samples_path, ('text', 'label'))
    metadata, metadata_input = _read_csv(metadata_path, ('sample_id', 'sample_row', 'text_sha256'))
    if not samples or len(samples) != len(metadata):
        raise ValueError('canonical samples/metadata must be nonempty and have equal row counts')
    by_row, ids, texts = {}, set(), set()
    for meta in metadata:
        _identity(meta['sample_id'], 'sample_id')
        raw_row = meta['sample_row']
        if not re.fullmatch(r'[1-9]\d*', raw_row):
            raise ValueError('invalid canonical sample_row')
        number = int(raw_row)
        if number > len(samples) or number in by_row or meta['sample_id'] in ids:
            raise ValueError('duplicate or out-of-range canonical sample_row/sample_id')
        text = samples[number - 1]['text']
        if not text.strip() or text in texts:
            raise ValueError('empty or duplicate canonical text')
        if hashlib.sha256(text.encode('utf-8')).hexdigest() != meta['text_sha256']:
            raise ValueError(f'text_sha256 mismatch for {meta["sample_id"]}')
        ids.add(meta['sample_id'])
        texts.add(text)
        by_row[number] = {'sample_id': meta['sample_id'], 'text': text,
                          'source_metadata': meta}
    records = [by_row[number] for number in range(1, len(samples) + 1)]
    return records, [sample_input, metadata_input]


def _matching_sample(row, canonical):
    sample_id = row['sample_id']
    if sample_id not in canonical:
        raise ValueError(f'unknown or changed sample_id: {sample_id!r}')
    if row['text'] != canonical[sample_id]['text']:
        raise ValueError(f'changed text for {sample_id}; exact canonical text required')


def _reviews(paths, canonical):
    by_sample = {key: [] for key in canonical}
    inputs, seen, reviewers = [], set(), set()
    for path in paths:
        rows, source = _read_csv(path, REVIEW_FIELDS)
        inputs.append(source)
        file_ids = {row['reviewer_id'] for row in rows if row['reviewer_id']}
        if len(file_ids) > 1:
            raise ValueError(f'{path}: inconsistent reviewer IDs within one export')
        file_reviewer = next(iter(file_ids), '')
        source['reviewer_id'] = file_reviewer
        sample_ids = set()
        for number, row in enumerate(rows, 2):
            try:
                _matching_sample(row, canonical)
                if row['sample_id'] in sample_ids:
                    raise ValueError('duplicate sample_id within reviewer export')
                sample_ids.add(row['sample_id'])
                reviewer = row['reviewer_id']
                status = row['review_status']
                if status not in ('pending', 'reviewed', 'ambiguous'):
                    raise ValueError(f'invalid review_status: {status!r}')
                _identity(reviewer, 'reviewer_id', required=status != 'pending')
                if reviewer:
                    reviewers.add(reviewer)
                # Blank pending cells are placeholders, not annotations by a shared
                # anonymous reviewer. Attribute duplicate checks to this export's
                # reviewer, or its path when the entire export is still unassigned.
                key = (file_reviewer or source['path'], row['sample_id'])
                if key in seen:
                    raise ValueError('duplicate reviewer/sample_id across exports')
                seen.add(key)
                label = None
                if status == 'reviewed':
                    label = _label(row['human_label'])
                    _review_date(row['reviewed_at'], 'reviewed_at')
                elif status == 'ambiguous':
                    if row['human_label']:
                        raise ValueError('ambiguous review must have a blank human_label')
                    _review_date(row['reviewed_at'], 'reviewed_at')
                elif row['human_label'] or row['reviewed_at']:
                    raise ValueError('pending review must have blank human_label and reviewed_at')
                by_sample[row['sample_id']].append(dict(row, normalized_label=label,
                    export_path=source['path'], export_row=number))
            except ValueError as exc:
                raise ValueError(f'{path}: row {number}: {exc}') from exc
    return by_sample, inputs, sorted(reviewers)


def _agreement(by_sample, reviewers):
    results = []
    for first, second in combinations(reviewers, 2):
        pairs = []
        for rows in by_sample.values():
            labels = {r['reviewer_id']: r['normalized_label'] for r in rows
                      if r['review_status'] == 'reviewed'}
            if first in labels and second in labels:
                pairs.append((labels[first], labels[second]))
        count = len(pairs)
        item = dict(reviewer_a=first, reviewer_b=second, n_overlap=count,
                    percent_agreement=None, cohen_kappa=None,
                    kappa_unavailable_reason='no_overlap')
        if count:
            observed = sum(a == b for a, b in pairs) / count
            first_counts, second_counts = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
            expected = sum(first_counts[label] * second_counts[label] for label in (0, 1)) / count ** 2
            item['percent_agreement'] = 100 * observed
            item['kappa_unavailable_reason'] = 'constant_label_marginals' if expected == 1 else None
            if expected != 1:
                item['cohen_kappa'] = (observed - expected) / (1 - expected)
        results.append(item)
    return results


def _resolve(rows, min_reviewers):
    completed = [r for r in rows if r['review_status'] != 'pending']
    labels = {r['normalized_label'] for r in completed if r['review_status'] == 'reviewed'}
    if not completed:
        return 'pending', None
    if len(labels) > 1:
        return 'disagreement', None
    if any(r['review_status'] == 'ambiguous' for r in completed):
        return 'ambiguous', None
    if len(completed) < min_reviewers:
        return 'insufficient_reviews', None
    return ('agreed' if len(completed) > 1 else 'single_reviewer'), next(iter(labels))


def _adjudications(path, records):
    rows, source = _read_csv(path, ADJUDICATION_FIELDS)
    canonical = {r['sample_id']: r for r in records}
    seen = set()
    for row in rows:
        _matching_sample(row, canonical)
        sample_id = row['sample_id']
        if sample_id in seen:
            raise ValueError('duplicate adjudication sample_id')
        seen.add(sample_id)
        record = canonical[sample_id]
        completed = {r['reviewer_id'] for r in record['reviews'] if r['review_status'] != 'pending'}
        if record['resolution'] not in ('disagreement', 'ambiguous') or len(completed) < 2:
            raise ValueError('adjudication requires disagreement/ambiguity from two completed reviewers')
        _identity(row['adjudicator_id'], 'adjudicator_id')
        if row['adjudicator_id'] in completed:
            raise ValueError('adjudicator_id must differ from the original reviewers')
        adjudicated_at = _review_date(row['adjudicated_at'], 'adjudicated_at')
        for review in record['reviews']:
            if review['review_status'] == 'pending':
                continue
            reviewed_at = _review_date(review['reviewed_at'], 'reviewed_at')
            if isinstance(adjudicated_at, datetime) and isinstance(reviewed_at, datetime):
                predates = adjudicated_at < reviewed_at
            else:
                # A date-only annotation has no time/zone precision; compare supplied calendar dates.
                predates = row['adjudicated_at'][:10] < review['reviewed_at'][:10]
            if predates:
                raise ValueError('adjudication must not predate its underlying reviews')
        if not row['reason'].strip():
            raise ValueError('adjudication reason must be nonempty')
        record['label'] = _label(row['human_label'])
        record['resolution_before_adjudication'] = record['resolution']
        record['resolution'] = 'adjudicated'
        record['adjudication'] = row
    return source


def _human_cell(value):
    if value is None:
        return ''
    text = str(value)
    if text.startswith(('\t', '\r', '\n')) or text.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + text
    return text


def _write_csv(path, fields, rows, human_facing=False):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _human_cell(row.get(key)) if human_facing else row.get(key, '')
                             for key in fields})


def _write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def import_reviews(samples, metadata, reviews, output_dir, *, min_reviewers=1, adjudication=None):
    """Validate all inputs, then create a new output directory; never overwrite.

    min_reviewers controls eligibility, not agreement. Conflicts and explicit
    ambiguity always require resolution, even when a majority has one polarity.
    Distinct reviewer IDs are supplied by the caller; this cannot verify actual
    reviewer identity or independent annotation outside the files.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'output directory already exists: {output_dir}')
    if isinstance(min_reviewers, bool) or not isinstance(min_reviewers, int) or min_reviewers < 1:
        raise ValueError('min_reviewers must be a positive integer')
    reviews = list(reviews)
    if not reviews:
        raise ValueError('at least one reviewer CSV export is required')
    records, inputs = _canonical(samples, metadata)
    by_sample, review_inputs, reviewers = _reviews(reviews, {r['sample_id']: r for r in records})
    agreement = _agreement(by_sample, reviewers)
    for record in records:
        record['reviews'] = by_sample[record['sample_id']]
        record['resolution'], record['label'] = _resolve(record['reviews'], min_reviewers)
    adjudication_input = _adjudications(adjudication, records) if adjudication else None
    eligible = [record for record in records if record['label'] is not None]
    unresolved = [record for record in records if record['label'] is None]
    resolutions = Counter(record['resolution'] for record in records)
    report = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
        canonical_inputs=inputs, review_inputs=review_inputs, adjudication_input=adjudication_input,
        min_reviewers=min_reviewers, reviewer_ids=reviewers,
        counts=dict(canonical_samples=len(records), review_rows=sum(map(len, by_sample.values())),
                    reviewed_rows=sum(r['review_status'] == 'reviewed' for rows in by_sample.values() for r in rows),
                    eligible_samples=len(eligible), unresolved_samples=len(unresolved),
                    adjudicated_samples=resolutions['adjudicated'], resolutions=dict(resolutions)),
        agreement_before_adjudication=agreement,
        rules=dict(label_map=LABELS, accepted_label_normalization='strip whitespace and casefold Positive/Negative only',
                   statuses=['pending', 'reviewed', 'ambiguous'],
                   review_dates='YYYY-MM-DD or ISO datetime with seconds and UTC offset/Z',
                   reviewer_independence='Distinct supplied reviewer IDs; actual identity/independence cannot be verified by CSV validation.',
                   eligibility='At least min_reviewers completed polarity reviews, unanimous and no ambiguous reviews; or explicit independent adjudication.',
                   source_labels='Canonical sample labels and metadata intended_label are never used for model labels.',
                   human_csv_formula_escape='Prefix apostrophe for leading tab/CR/LF or first non-whitespace =,+,-,@; raw values are in JSON.',
                   model_csv='reviewed_samples.csv preserves exact text; intended for model ingestion, not opening as a spreadsheet.',
                   audit_csv='Human-facing CSVs are display artifacts, not round-trip reviewer inputs.'))
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / 'reviewed_samples.csv', ('text', 'label'), eligible)
    _write_json(output_dir / 'reviewed_metadata.json', eligible)
    _write_json(output_dir / 'unresolved_samples.json', unresolved)
    audit_fields = REVIEW_FIELDS + ('normalized_label', 'export_path', 'export_row')
    _write_csv(output_dir / 'human_review_audit.csv', audit_fields,
               [row for rows in by_sample.values() for row in rows], human_facing=True)
    _write_csv(output_dir / 'unresolved_review.csv', ('sample_id', 'text', 'resolution', 'completed_reviews'),
               [dict(record, completed_reviews=sum(r['review_status'] != 'pending' for r in record['reviews']))
                for record in unresolved], human_facing=True)
    report['artifacts_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted(output_dir.iterdir())}
    _write_json(output_dir / 'import_report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, nargs='+', action='extend', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--min-reviewers', type=int, default=1)
    parser.add_argument('--adjudication', type=Path)
    args = parser.parse_args(argv)
    try:
        report = import_reviews(args.samples, args.metadata, args.reviews, args.output_dir,
                                min_reviewers=args.min_reviewers, adjudication=args.adjudication)
    except (OSError, ValueError) as exc:
        print(f'Import rejected: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
