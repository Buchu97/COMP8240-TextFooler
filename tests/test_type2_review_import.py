import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'scripts/import_type2_reviews.py'
REVIEW_FIELDS = ['sample_id', 'text', 'human_label', 'review_status',
                 'reviewer_id', 'reviewed_at', 'notes']
ADJUDICATION_FIELDS = ['sample_id', 'text', 'human_label', 'adjudicator_id',
                       'adjudicated_at', 'reason']


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


class Type2ReviewImportTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), 'Type 2 review importer is missing')
        spec = importlib.util.spec_from_file_location('type2_review_import', MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = self.root / 'samples.csv'
        self.metadata = self.root / 'metadata.csv'
        self.texts = ['A wonderful film.', 'A dreadful film.', '=SUM(1,1)', 'It was a film.']
        self.sample_rows = [{'text': text, 'label': ''} for text in self.texts]
        self.meta_rows = [dict(sample_id=f'S{i}', sample_row=str(i),
                               text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                               source_type='fixture', source_url=f'https://example.test/{i}',
                               intended_label='1') for i, text in enumerate(self.texts, 1)]
        self.write_source()

    def write_source(self):
        write_csv(self.samples, ['text', 'label'], self.sample_rows)
        write_csv(self.metadata, list(self.meta_rows[0]), self.meta_rows)

    def review(self, sample=1, label='Positive', reviewer='alice', status='reviewed', **changes):
        row = dict(sample_id=f'S{sample}', text=self.texts[sample - 1],
                   human_label=label, review_status=status, reviewer_id=reviewer,
                   reviewed_at='2026-11-03', notes='')
        row.update(changes)
        return row

    def reviews(self, rows, name='alice.csv'):
        path = self.root / name
        write_csv(path, REVIEW_FIELDS, rows)
        return path

    def run_import(self, paths, **kwargs):
        return self.module.import_reviews(self.samples, self.metadata, paths,
                                          self.root / 'imported', **kwargs)

    def test_normalizes_only_polarity_names_and_preserves_text_and_metadata(self):
        path = self.reviews([self.review(1, ' positive '), self.review(2, 'NEGATIVE')])
        report = self.run_import([path])
        self.assertEqual(read_csv(self.root / 'imported/reviewed_samples.csv'), [
            {'text': 'A wonderful film.', 'label': '1'},
            {'text': 'A dreadful film.', 'label': '0'}])
        metadata = json.loads((self.root / 'imported/reviewed_metadata.json').read_text())
        self.assertEqual(metadata[0]['source_metadata'], self.meta_rows[0])
        self.assertEqual(metadata[0]['sample_id'], 'S1')
        self.assertEqual(report['counts']['eligible_samples'], 2)
        self.assertEqual(report['counts']['unresolved_samples'], 2)
        self.assertEqual(report['agreement_before_adjudication'], [])

    def test_pending_and_explicitly_ambiguous_rows_cannot_receive_draft_labels(self):
        path = self.reviews([
            self.review(1, '', '', 'pending', reviewed_at=''),
            self.review(2, '', 'alice', 'ambiguous', notes='Mixed sentiment.')])
        report = self.run_import([path])
        self.assertEqual(report['counts']['eligible_samples'], 0)
        rows = json.loads((self.root / 'imported/unresolved_samples.json').read_text())
        self.assertEqual([row['resolution'] for row in rows],
                         ['pending', 'ambiguous', 'pending', 'pending'])
        self.assertEqual(read_csv(self.root / 'imported/reviewed_samples.csv'), [])

    def test_agreement_uses_overlap_before_adjudication_and_disagreement_stays_unresolved(self):
        alice = self.reviews([self.review(1), self.review(2, 'Negative'),
                              self.review(3), self.review(4, 'Negative')])
        bob = self.reviews([self.review(1, reviewer='bob'),
                            self.review(2, 'Negative', 'bob'),
                            self.review(3, 'Negative', 'bob')], 'bob.csv')
        report = self.run_import([alice, bob])
        agreement = report['agreement_before_adjudication'][0]
        self.assertEqual(agreement['n_overlap'], 3)
        self.assertAlmostEqual(agreement['percent_agreement'], 200 / 3)
        self.assertAlmostEqual(agreement['cohen_kappa'], 0.4)
        self.assertEqual(report['counts']['eligible_samples'], 3)
        unresolved = json.loads((self.root / 'imported/unresolved_samples.json').read_text())
        self.assertEqual([(r['sample_id'], r['resolution']) for r in unresolved],
                         [('S3', 'disagreement')])

    def test_disjoint_reviewers_do_not_get_agreement_values(self):
        a = self.reviews([self.review(1)])
        b = self.reviews([self.review(2, reviewer='bob')], 'bob.csv')
        item = self.run_import([a, b])['agreement_before_adjudication'][0]
        self.assertEqual(item['n_overlap'], 0)
        self.assertIsNone(item['percent_agreement'])
        self.assertIsNone(item['cohen_kappa'])
        self.assertEqual(item['kappa_unavailable_reason'], 'no_overlap')

    def test_full_reviewer_exports_can_share_unassigned_pending_rows(self):
        pending = [self.review(i, '', '', 'pending', reviewed_at='') for i in (3, 4)]
        a = self.reviews([self.review(1)] + pending)
        b = self.reviews([self.review(2, 'Negative', 'bob')] + pending, 'bob.csv')
        try:
            report = self.run_import([a, b])
        except ValueError as exc:
            self.fail(f'Unassigned pending placeholders are not duplicate human annotations: {exc}')
        self.assertEqual(report['counts']['eligible_samples'], 2)
        self.assertEqual(report['agreement_before_adjudication'][0]['n_overlap'], 0)
        unresolved = json.loads((self.root / 'imported/unresolved_samples.json').read_text())
        self.assertEqual([r['resolution'] for r in unresolved], ['pending', 'pending'])
        self.assertEqual(len(unresolved[0]['reviews']), 2)

    def test_original_draft_sample_labels_never_supply_an_imported_human_label(self):
        for row in self.sample_rows:
            row['label'] = '1'
        self.write_source()
        self.run_import([self.reviews([self.review(1, 'Negative')])])
        self.assertEqual(read_csv(self.root / 'imported/reviewed_samples.csv'),
                         [{'text': 'A wonderful film.', 'label': '0'}])

    def test_constant_agreement_has_undefined_kappa(self):
        a = self.reviews([self.review(1)])
        b = self.reviews([self.review(1, reviewer='bob')], 'bob.csv')
        item = self.run_import([a, b])['agreement_before_adjudication'][0]
        self.assertEqual(item['percent_agreement'], 100.0)
        self.assertIsNone(item['cohen_kappa'])
        self.assertEqual(item['kappa_unavailable_reason'], 'constant_label_marginals')

    def test_required_reviewer_count_keeps_single_review_unresolved(self):
        report = self.run_import([self.reviews([self.review()])], min_reviewers=2)
        self.assertEqual(report['counts']['eligible_samples'], 0)
        unresolved = json.loads((self.root / 'imported/unresolved_samples.json').read_text())
        self.assertEqual(unresolved[0]['resolution'], 'insufficient_reviews')

    def test_explicit_independent_adjudication_resolves_conflict_without_rewriting_agreement(self):
        a = self.reviews([self.review()])
        b = self.reviews([self.review(label='Negative', reviewer='bob')], 'bob.csv')
        adjudication = self.root / 'adjudication.csv'
        write_csv(adjudication, ADJUDICATION_FIELDS, [dict(sample_id='S1', text=self.texts[0],
                  human_label='Negative', adjudicator_id='carol', adjudicated_at='2026-11-03T09:00:00+11:00',
                  reason='The first reading missed a qualification.')])
        report = self.run_import([a, b], adjudication=adjudication)
        self.assertEqual(read_csv(self.root / 'imported/reviewed_samples.csv')[0]['label'], '0')
        self.assertEqual(report['agreement_before_adjudication'][0]['percent_agreement'], 0.0)
        self.assertEqual(report['counts']['adjudicated_samples'], 1)

    def test_formula_like_cells_are_escaped_only_in_human_facing_csv(self):
        path = self.reviews([self.review(3, notes='  =HYPERLINK("x")')])
        self.run_import([path])
        model = read_csv(self.root / 'imported/reviewed_samples.csv')
        audit = read_csv(self.root / 'imported/human_review_audit.csv')
        self.assertEqual(model[0]['text'], '=SUM(1,1)')
        self.assertEqual(audit[0]['text'], "'=SUM(1,1)")
        self.assertEqual(audit[0]['notes'], "'  =HYPERLINK(\"x\")")
        metadata = json.loads((self.root / 'imported/reviewed_metadata.json').read_text())
        self.assertEqual(metadata[0]['reviews'][0]['notes'], '  =HYPERLINK("x")')

    def test_invalid_timezone_offset_minutes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reviewed_at'):
            self.run_import([self.reviews([self.review(reviewed_at='2026-11-03T09:00:00+00:99')])])

    def test_adjudication_cannot_predate_its_underlying_reviews(self):
        a = self.reviews([self.review()])
        b = self.reviews([self.review(label='Negative', reviewer='bob')], 'bob.csv')
        path = self.root / 'adjudication.csv'
        write_csv(path, ADJUDICATION_FIELDS, [dict(sample_id='S1', text=self.texts[0],
                  human_label='Negative', adjudicator_id='carol', adjudicated_at='2026-11-02',
                  reason='Earlier decision')])
        with self.assertRaisesRegex(ValueError, 'predate'):
            self.run_import([a, b], adjudication=path)

    def test_adjudication_rejects_unknown_changed_duplicate_or_nonindependent_decisions(self):
        a = self.reviews([self.review()])
        b = self.reviews([self.review(label='Negative', reviewer='bob')], 'bob.csv')
        path = self.root / 'adjudication.csv'
        base = dict(sample_id='S1', text=self.texts[0], human_label='Negative',
                    adjudicator_id='carol', adjudicated_at='2026-11-03', reason='Human resolution')
        mutations = [dict(sample_id='S2', text=self.texts[1]), dict(sample_id='missing'),
                     dict(text='Changed'), dict(adjudicator_id='alice'), dict(adjudicator_id=''),
                     dict(adjudicated_at='2026-02-30'), dict(reason=' '), dict(human_label='0')]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                write_csv(path, ADJUDICATION_FIELDS, [dict(base, **mutation)])
                with self.assertRaises(ValueError):
                    self.run_import([a, b], adjudication=path)
                self.assertFalse((self.root / 'imported').exists())
        write_csv(path, ADJUDICATION_FIELDS, [base, base])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.run_import([a, b], adjudication=path)

    def test_changed_id_or_text_and_duplicate_rows_fail_before_any_output(self):
        cases = [
            [self.review(sample_id='UNKNOWN')],
            [self.review(text='A wonderful film. ')],
            [self.review(), self.review()],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    self.run_import([self.reviews(rows)])
                self.assertFalse((self.root / 'imported').exists())

    def test_same_review_cannot_be_counted_twice_across_exports(self):
        a = self.reviews([self.review()])
        b = self.reviews([self.review()], 'copy.csv')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.run_import([a, b])

    def test_mixed_reviewer_ids_in_one_export_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reviewer'):
            self.run_import([self.reviews([self.review(), self.review(2, reviewer='bob')])])

    def test_malformed_status_label_identity_and_dates_are_rejected(self):
        cases = [dict(human_label='1'), dict(human_label=''), dict(human_label='neutral'),
                 dict(review_status='done'), dict(reviewer_id=''), dict(reviewer_id=' alice '),
                 dict(reviewed_at='2026-02-30'), dict(reviewed_at='03/11/2026'),
                 dict(reviewed_at='2026-11-03T09:00:00'), dict(reviewed_at=''),
                 dict(review_status='pending'), dict(review_status='ambiguous')]
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.run_import([self.reviews([self.review(**changes)])])

    def test_missing_duplicate_headers_and_ragged_review_csv_are_rejected(self):
        path = self.root / 'bad.csv'
        for contents in ['sample_id,text\nS1,A wonderful film.\n',
                         ','.join(REVIEW_FIELDS + ['text']) + '\n',
                         ','.join(REVIEW_FIELDS) + '\nS1,A wonderful film.,Positive\n',
                         ','.join(REVIEW_FIELDS) + '\nS1,A wonderful film.,Positive,reviewed,alice,2026-11-03,,extra\n']:
            with self.subTest(contents=contents):
                path.write_text(contents, encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.run_import([path])

    def test_canonical_metadata_must_match_all_samples_and_sha256(self):
        mutations = [lambda: self.meta_rows[0].update(text_sha256='0' * 64),
                     lambda: self.meta_rows[0].update(sample_row='2'),
                     lambda: self.meta_rows[0].update(sample_id='S2'),
                     lambda: self.meta_rows.pop(),
                     lambda: self.sample_rows[0].update(text='Changed canonical text')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                original_meta = [dict(row) for row in self.meta_rows]
                original_samples = [dict(row) for row in self.sample_rows]
                mutate()
                self.write_source()
                with self.assertRaises(ValueError):
                    self.run_import([self.reviews([self.review()])])
                self.meta_rows, self.sample_rows = original_meta, original_samples

    def test_existing_output_directory_is_never_overwritten(self):
        output = self.root / 'imported'
        output.mkdir()
        (output / 'keep.txt').write_text('keep')
        with self.assertRaises(FileExistsError):
            self.run_import([self.reviews([self.review()])])
        self.assertEqual((output / 'keep.txt').read_text(), 'keep')
        self.assertEqual(list(output.iterdir()), [output / 'keep.txt'])

    def test_cli_supports_multiple_review_exports_and_reports_validation_failure(self):
        path = self.reviews([self.review()])
        second = self.reviews([self.review(reviewer='bob')], 'bob.csv')
        args = [sys.executable, str(MODULE), '--samples', str(self.samples),
                '--metadata', str(self.metadata), '--reviews', str(path),
                '--reviews', str(second),
                '--output-dir', str(self.root / 'cli')]
        result = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['counts']['eligible_samples'], 1)
        self.assertEqual(json.loads(result.stdout)['agreement_before_adjudication'][0]['n_overlap'], 1)
        self.reviews([self.review(human_label='bad')])
        result = subprocess.run(args[:-1] + [str(self.root / 'invalid')], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('human_label', result.stderr)
        self.assertFalse((self.root / 'invalid').exists())


if __name__ == '__main__':
    unittest.main()
