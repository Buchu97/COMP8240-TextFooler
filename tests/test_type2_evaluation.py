"""Evaluation preparation uses synthetic review fixtures, never project labels."""

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
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from import_type2_reviews import import_reviews, REVIEW_FIELDS


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Type2EvaluationTests(unittest.TestCase):
    def setUp(self):
        module_path = SCRIPTS / 'prepare_type2_evaluation.py'
        self.assertTrue(module_path.is_file(), 'Type2 evaluation bridge is missing')
        spec = importlib.util.spec_from_file_location('prepare_type2_evaluation', module_path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.texts = [' A wonderful café film.\nLoved it.', '=A dreadful movie.']
        samples = self.root / 'samples.csv'
        metadata = self.root / 'metadata.csv'
        # These intentionally conflict with the supplied human polarity reviews.
        write_csv(samples, ['text', 'label'], [dict(text=t, label=0) for t in self.texts])
        self.metadata = [dict(sample_id=f'FIXTURE{i}', sample_row=i,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            source_type='synthetic_test_fixture', intended_label=0,
            source_url=f'https://example.test/{i}') for i, text in enumerate(self.texts, 1)]
        write_csv(metadata, list(self.metadata[0]), self.metadata)
        self.sources = (samples, metadata)

    def make_import(self, *, pending=False, min_reviewers=1):
        reviews = self.root / 'reviewer.csv'
        write_csv(reviews, REVIEW_FIELDS, [dict(sample_id=f'FIXTURE{i}', text=text,
            human_label='' if pending else ('Positive' if i == 1 else 'Negative'),
            review_status='pending' if pending else 'reviewed',
            reviewer_id='' if pending else 'fixture_reviewer',
            reviewed_at='' if pending else '2026-11-03', notes='Synthetic test only')
            for i, text in enumerate(self.texts, 1)])
        self.imported = self.root / 'import_fixture'
        import_reviews(*self.sources, [reviews], self.imported, min_reviewers=min_reviewers)
        return self.imported

    def prepare(self, **kwargs):
        return self.module.prepare_evaluation(self.imported, self.root / 'evaluation_fixture',
            experiment_tag='TYPE2_FIXTURE_20261103', min_reviewers=1, **kwargs)

    def rewrite_import_hashes(self):
        report_path = self.imported / 'import_report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        for name in report['artifacts_sha256']:
            report['artifacts_sha256'][name] = digest(self.imported / name)
        report_path.write_text(json.dumps(report), encoding='utf-8')

    def test_preserves_exact_evaluation_bytes_ids_human_labels_and_provenance(self):
        self.make_import()
        manifest = self.prepare()
        output = self.root / 'evaluation_fixture'
        self.assertEqual((output / 'evaluation.csv').read_bytes(),
                         (self.imported / 'reviewed_samples.csv').read_bytes())
        self.assertEqual(read_csv(output / 'evaluation.csv'), [
            {'text': self.texts[0], 'label': '1'}, {'text': self.texts[1], 'label': '0'}])
        metadata = read_csv(output / 'evaluation_metadata.csv')
        self.assertEqual([r['source_id'] for r in metadata], ['FIXTURE1', 'FIXTURE2'])
        self.assertEqual([r['label_name'] for r in metadata], ['positive', 'negative'])
        self.assertEqual([r['text_sha256'] for r in metadata],
                         [r['text_sha256'] for r in self.metadata])
        original = json.loads((output / 'review_import/reviewed_metadata.json').read_text())
        self.assertEqual(original[0]['source_metadata']['intended_label'], '0')
        self.assertEqual(original[0]['source_metadata']['source_type'], 'synthetic_test_fixture')
        self.assertEqual(original[0]['reviews'][0]['reviewer_id'], 'fixture_reviewer')
        self.assertFalse(manifest['train_test_split_created'])
        self.assertEqual(manifest['evaluation']['examples'], 2)
        self.assertEqual(manifest['label2id'], {'negative': 0, 'positive': 1})
        self.assertEqual(sorted(p.name for p in output.iterdir()),
            ['evaluation.csv', 'evaluation_metadata.csv', 'manifest.json', 'review_import'])
        self.module.validate_prepared(output)

    def test_rejects_pending_empty_and_insufficient_review_imports(self):
        for pending, minimum in ((True, 1), (False, 2)):
            with self.subTest(pending=pending, minimum=minimum):
                self.make_import(pending=pending, min_reviewers=minimum)
                with self.assertRaisesRegex(ValueError, 'resolved|empty|review'):
                    self.module.prepare_evaluation(self.imported, self.root / 'bad',
                        experiment_tag='TYPE2_FIXTURE', min_reviewers=minimum)
                self.assertFalse((self.root / 'bad').exists())
                import shutil
                shutil.rmtree(self.imported)

    def test_rejects_partial_review_population(self):
        self.make_import()
        reviews = self.root / 'reviewer.csv'
        rows = read_csv(reviews)
        rows[1].update(human_label='', review_status='pending', reviewed_at='')
        write_csv(reviews, REVIEW_FIELDS, rows)
        import shutil
        shutil.rmtree(self.imported)
        import_reviews(*self.sources, [reviews], self.imported, min_reviewers=1)
        with self.assertRaisesRegex(ValueError, 'resolved|unresolved'):
            self.prepare()

    def test_explicit_partial_pilot_retains_unresolved_disagreement_and_full_counts(self):
        self.make_import()
        bob = self.root / 'bob.csv'
        rows = read_csv(self.root / 'reviewer.csv')
        for row in rows:
            row['reviewer_id'] = 'fixture_bob'
            row['human_label'] = 'Positive'
        write_csv(bob, REVIEW_FIELDS, rows)
        import shutil
        shutil.rmtree(self.imported)
        import_reviews(*self.sources, [self.root / 'reviewer.csv', bob],
                       self.imported, min_reviewers=2)
        output = self.root / 'pilot'
        manifest = self.module.prepare_evaluation(self.imported, output,
            experiment_tag='TYPE2_PILOT_FIXTURE', min_reviewers=2, allow_partial=True)
        self.assertEqual(manifest['review_provenance']['counts']['canonical_samples'], 2)
        self.assertEqual(manifest['review_provenance']['counts']['eligible_samples'], 1)
        self.assertEqual(manifest['review_provenance']['counts']['unresolved_samples'], 1)
        self.assertEqual(manifest['evaluation']['coverage_fraction'], 0.5)
        self.assertTrue(manifest['partial_pilot'])
        self.assertEqual(read_csv(output / 'evaluation.csv'),
                         [{'text': self.texts[0], 'label': '1'}])
        unresolved = json.loads((output / 'review_import/unresolved_samples.json').read_text())
        self.assertEqual([(r['sample_id'], r['resolution']) for r in unresolved],
                         [('FIXTURE2', 'disagreement')])
        with self.assertRaisesRegex(ValueError, 'partial|pilot'):
            self.module.validate_prepared(output)
        self.module.validate_prepared(output, allow_partial=True)

    def test_partial_mode_requires_pilot_tag_and_does_not_allow_empty_population(self):
        self.make_import(pending=True)
        with self.assertRaisesRegex(ValueError, 'tag|PILOT'):
            self.module.prepare_evaluation(self.imported, self.root / 'bad_tag',
                experiment_tag='TYPE2_FINAL_FIXTURE', min_reviewers=1, allow_partial=True)
        with self.assertRaisesRegex(ValueError, 'empty|nonempty'):
            self.module.prepare_evaluation(self.imported, self.root / 'bad_empty',
                experiment_tag='TYPE2_PILOT_FIXTURE', min_reviewers=1, allow_partial=True)
        self.assertFalse((self.root / 'bad_tag').exists())
        self.assertFalse((self.root / 'bad_empty').exists())

    def test_preflight_rejects_false_coverage_even_with_unchanged_review_artifacts(self):
        self.make_import()
        self.prepare()
        output = self.root / 'evaluation_fixture'
        path = output / 'manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['review_provenance']['counts']['canonical_samples'] = 500
        path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'provenance|coverage|population'):
            self.module.validate_prepared(output)

    def test_preflight_rejects_tampered_recipe_asset(self):
        config = json.loads(self.module.CONFIG.read_text(encoding='utf-8'))
        config['required_files_sha256'] = {'scripts/textfooler_paper_delta_0_7.py': '0' * 64}
        path = self.root / 'tampered_config.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        self.module.CONFIG = path
        with self.assertRaisesRegex(ValueError, 'asset hash mismatch'):
            self.module.verify_local_assets()

    def test_rejects_policy_mismatch_before_output_creation(self):
        self.make_import()
        with self.assertRaisesRegex(ValueError, 'min_reviewers'):
            self.module.prepare_evaluation(self.imported, self.root / 'bad',
                experiment_tag='TYPE2_FIXTURE', min_reviewers=2)
        self.assertFalse((self.root / 'bad').exists())

    def test_rejects_edited_artifact_and_hash_rewritten_unreviewed_label(self):
        self.make_import()
        path = self.imported / 'reviewed_metadata.json'
        rows = json.loads(path.read_text(encoding='utf-8'))
        rows[0]['label'] = 0
        path.write_text(json.dumps(rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash|SHA|replay'):
            self.prepare()
        # Forging a matching CSV and new report hashes still lacks review support.
        csv_rows = read_csv(self.imported / 'reviewed_samples.csv')
        csv_rows[0]['label'] = '0'
        write_csv(self.imported / 'reviewed_samples.csv', ['text', 'label'], csv_rows)
        self.rewrite_import_hashes()
        with self.assertRaisesRegex(ValueError, 'replay|provenance'):
            self.prepare()
        self.assertFalse((self.root / 'evaluation_fixture').exists())

    def test_rejects_changed_original_reviewer_source(self):
        self.make_import()
        with (self.root / 'reviewer.csv').open('a', encoding='utf-8') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'source|hash|SHA'):
            self.prepare()

    def test_tag_and_overwrite_policy(self):
        self.make_import()
        for tag in ('', '../bad', 'untagged lower case'):
            with self.assertRaisesRegex(ValueError, 'tag'):
                self.module.prepare_evaluation(self.imported, self.root / 'bad',
                    experiment_tag=tag, min_reviewers=1)
        self.prepare()
        before = (self.root / 'evaluation_fixture/evaluation.csv').read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((self.root / 'evaluation_fixture/evaluation.csv').read_bytes(), before)

    def test_preflight_rejects_metadata_edit_even_with_updated_manifest_hash(self):
        self.make_import()
        self.prepare()
        output = self.root / 'evaluation_fixture'
        path = output / 'evaluation_metadata.csv'
        rows = read_csv(path)
        rows[0]['source_id'] = 'OTHER'
        write_csv(path, list(rows[0]), rows)
        manifest_path = output / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['artifacts_sha256'][path.name] = digest(path)
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'metadata|alignment'):
            self.module.validate_prepared(output)

    @unittest.skipUnless(
        all((ROOT / asset).is_file() for asset in json.loads(
            (ROOT / 'notes/type2_execution_20261103.json').read_text(encoding='utf-8')
        )['required_files_sha256']),
        'requires the pinned Type 2 model and encoder assets',
    )
    def test_offline_dry_run_uses_local_pinned_assets_and_creates_no_attack_output(self):
        self.make_import()
        self.prepare()
        run_output = self.root / 'attack_fixture'
        code = (
            'import run_type2, sys; '
            'code=run_type2.main(sys.argv[1:]); '
            'assert "torch" not in sys.modules and "textattack" not in sys.modules; '
            'raise SystemExit(code)'
        )
        completed = subprocess.run([sys.executable, '-c', code,
            '--dataset-dir', str(self.root / 'evaluation_fixture'),
            '--output-dir', str(run_output), '--dry-run'], cwd=SCRIPTS,
            capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result['status'], 'preflight_passed_no_attack_executed')
        self.assertEqual(result['examples'], 2)
        self.assertEqual(result['model']['revision'], '2aeb2b03a543f2eb1f203ca40676fbaf27a3d4fd')
        self.assertEqual(result['attack']['N'], 50)
        self.assertEqual(result['attack']['word_cosine_delta'], 0.7)
        self.assertFalse(run_output.exists())

    def test_dry_run_rejects_existing_attack_output_before_loading_model(self):
        output = self.root / 'existing'
        output.mkdir()
        completed = subprocess.run([sys.executable, str(SCRIPTS / 'run_type2.py'),
            '--dataset-dir', str(self.root / 'missing'), '--output-dir', str(output), '--dry-run'],
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 2)
        self.assertIn('already exists', completed.stderr)


if __name__ == '__main__':
    unittest.main()
