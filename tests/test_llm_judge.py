import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import judge_protocol as protocol
from test_judge_data import source_fixture
from test_judge_runtime import configuration


class JudgeWorkflowTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/llm_judge.py'
        self.assertTrue(path.exists(), 'Judge command-line workflow is missing')
        spec = importlib.util.spec_from_file_location('llm_judge', path)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        spec.loader.exec_module(self.module)

    def test_end_to_end_fixture_judgements_keep_denominators_and_human_reviews_separate(self):
        import judge_data
        import judge_runtime
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            prepared = root / 'prepared'
            judge_data.prepare({'sources': [source_fixture(root)]}, prepared, development_per_dataset=2, audit_size=3)
            manifest, pairs, spaces = judge_data.load_prepared(prepared)
            jobs = self.module.jobs_for_phase(manifest, pairs, spaces, 'final')
            self.assertEqual(len(jobs), 15)
            bad_pair = jobs[0]['pair_id']
            answers = {}
            for job in jobs:
                body = judge_runtime.request_body(job, configuration())
                if job['kind'] == 'quality':
                    answer = {'meaning_preserved': 'no' if job['pair_id'] == bad_pair else 'yes',
                              'same_task_label': 'yes', 'naturalness_a': 4, 'naturalness_b': 4,
                              'meaning_reason': 'Fixture judgement.', 'label_reason': 'Fixture label judgement.',
                              'naturalness_reason': 'Fixture naturalness judgement.'}
                else:
                    answer = {'label': 'positive', 'reason': 'Fixture label.'}
                answers[protocol.digest(body)] = answer
            def transport(body, config):
                return {'id': 'test-fixture', 'model': 'test-model-v1',
                        'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answers[protocol.digest(body)])}}],
                        'usage': {'prompt_tokens': 20, 'completion_tokens': 10}}
            evaluation = root / 'evaluation'
            judge_runtime.execute_jobs(jobs, evaluation, configuration(), protocol.digest(manifest), execute=True,
                                       budget_usd=1, max_calls=100, transport=transport)
            report = self.module.summarize(prepared, evaluation, root / 'report')
            result = report['datasets']['reviews']
            self.assertEqual(result['selected'], 5)
            self.assertEqual(result['judged'], 5)
            self.assertEqual(result['accepted'], 4)
            self.assertEqual(result['rejected'], 1)
            self.assertEqual(result['judge_acceptance_percent'], 80)
            self.assertEqual(result['raw_attack_asr_percent'], 87.5)
            self.assertIsNone(result['full_population_judge_accepted_asr_percent'])
            self.assertEqual(report['human_comparison']['completed_reviews'], 0)
            diagnostics = report['protocol_diagnostics']['reviews']
            self.assertEqual(diagnostics['original_label_disagreements'], 0)
            self.assertEqual(diagnostics['meaning_yes'], 4)
            self.assertEqual(diagnostics['paired_vs_single_label_preservation']['n'], 5)
            self.assertEqual(diagnostics['paired_vs_single_label_preservation']['agreement_percent'], 100)
            audit = judge_data.read_csv(prepared / 'human_review.csv')
            chosen = next(r for r in audit if r['pair_id'] != bad_pair)
            chosen.update(meaning_preserved='yes', label_a='positive', label_b='positive',
                          naturalness_a='4', naturalness_b='4', reviewer_id='test_fixture_reviewer', reviewed_at='2026-10-30')
            protocol.write_csv(root / 'reviewed.csv', audit, list(audit[0]))
            report = self.module.summarize(prepared, evaluation, root / 'report2', human_review=root / 'reviewed.csv')
            self.assertEqual(report['human_comparison']['meaning']['n'], 1)
            self.assertEqual(report['human_comparison']['meaning']['agreement_percent'], 100)
            audit[0]['text_a'] = 'An unrelated text.'
            protocol.write_csv(root / 'tampered.csv', audit, list(audit[0]))
            with self.assertRaises(ValueError):
                self.module.summarize(prepared, evaluation, root / 'bad-report', human_review=root / 'tampered.csv')
            self.assertFalse((root / 'bad-report').exists())

    def test_agreement_handles_constant_ratings_and_partial_unresolved_decisions(self):
        result = self.module.agreement([('yes', 'yes'), ('no', 'yes'), ('uncertain', 'yes')])
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['agreement_percent'], 50)
        self.assertAlmostEqual(result['cohens_kappa'], 0)
        self.assertEqual(result['unresolved'], 1)
        self.assertIsNone(self.module.agreement([('yes', 'yes')])['cohens_kappa'])

    def test_protocol_diagnostics_exclude_uncertain_single_text_labels_from_agreement(self):
        def pair(original, edited, paired, status, correct):
            return dict(original_judge_label=original, perturbed_judge_label=edited,
                        same_task_label=paired, meaning_preserved='yes',
                        status=status, original_label_matches_gold=correct)

        rows = [
            pair('positive', 'positive', 'yes', 'accepted', True),
            pair('negative', 'positive', 'yes', 'uncertain', False),
            pair('positive', protocol.UNCERTAIN, 'yes', 'uncertain', True),
        ]
        result = self.module.protocol_diagnostics(rows)
        self.assertEqual(result['original_label_disagreements'], 1)
        self.assertEqual(result['uncertain_from_original_label'], 1)
        self.assertEqual(result['meaning_yes'], 3)
        self.assertEqual(result['paired_vs_single_label_preservation']['n'], 2)
        self.assertEqual(result['paired_vs_single_label_preservation']['unresolved'], 1)
        self.assertEqual(result['paired_vs_single_label_preservation']['agreement_percent'], 50)


if __name__ == '__main__':
    unittest.main()
