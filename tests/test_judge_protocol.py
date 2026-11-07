import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def pair():
    return {'pair_id': 'p1', 'dataset': 'mr', 'task': 'sentiment',
            'original_text': 'A delightful film.', 'perturbed_text': 'A dreadful film.',
            'gold_label': 'positive', 'original_prediction': 'SECRET_PREDICTION',
            'phase': 'final', 'control': ''}


def quality(meaning='yes', label='yes', naturalness=4):
    return {'meaning_preserved': meaning, 'same_task_label': label,
            'naturalness_a': naturalness, 'naturalness_b': naturalness,
            'meaning_reason': 'A short explanation.', 'label_reason': 'A label explanation.',
            'naturalness_reason': 'A naturalness explanation.'}


class JudgeProtocolTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/judge_protocol.py'
        self.assertTrue(path.exists(), 'Judge protocol is missing')
        spec = importlib.util.spec_from_file_location('judge_protocol', path)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        spec.loader.exec_module(self.module)

    def test_requests_hide_gold_and_predictions_and_independent_texts_are_separate(self):
        labels = {'negative': 'Unfavourable sentiment.', 'positive': 'Favourable sentiment.'}
        requests = self.module.build_requests(pair(), labels, 8240)
        self.assertEqual(set(requests), {'quality', 'original_label', 'perturbed_label'})
        for request in requests.values():
            self.assertNotIn('SECRET_PREDICTION', json.dumps(request))
            self.assertNotIn('gold_label', json.dumps(request))
        original = json.loads(requests['original_label']['messages'][1]['content'])
        changed = json.loads(requests['perturbed_label']['messages'][1]['content'])
        self.assertEqual(original['text'], 'A delightful film.')
        self.assertEqual(changed['text'], 'A dreadful film.')
        self.assertNotIn('A dreadful film.', json.dumps(original))
        self.assertNotIn('A delightful film.', json.dumps(changed))

    def test_answer_validation_rejects_boolean_ratings_unknown_labels_and_extra_keys(self):
        for bad in (dict(quality(), naturalness_a=True), dict(quality(), meaning_preserved='maybe'),
                    dict(quality(), extra='unrequested')):
            with self.assertRaises(ValueError):
                self.module.validate_answer('quality', bad, ['negative', 'positive'])
        with self.assertRaises(ValueError):
            self.module.validate_answer('original_label', {'label': 'politics', 'reason': 'x'}, ['negative', 'positive'])

    def test_acceptance_requires_semantics_pair_label_and_independent_gold_agreement(self):
        yes = {'label': 'positive', 'reason': 'Positive.'}
        no = {'label': 'negative', 'reason': 'Negative.'}
        self.assertEqual(self.module.assess(pair(), quality(), yes, yes)['status'], 'accepted')
        self.assertEqual(self.module.assess(pair(), quality('no'), yes, yes)['status'], 'rejected')
        self.assertEqual(self.module.assess(pair(), quality(), yes, no)['status'], 'rejected')
        self.assertEqual(self.module.assess(pair(), quality(), no, no)['status'], 'uncertain')
        self.assertEqual(self.module.assess(pair(), quality(), no, yes)['status'], 'uncertain')
        self.assertEqual(self.module.assess(pair(), quality('no'), no, no)['status'], 'uncertain')
        self.assertEqual(self.module.assess(pair(), quality('uncertain'), yes, yes)['status'], 'uncertain')
        self.assertEqual(self.module.assess(pair(), quality(naturalness=1), yes, yes)['status'], 'accepted')

    def test_pair_order_is_reproducible_and_scores_are_unblinded_correctly(self):
        labels = {'negative': 'Negative.', 'positive': 'Positive.'}
        request = self.module.build_requests(pair(), labels, 8240)['quality']
        self.assertEqual(request, self.module.build_requests(pair(), labels, 8240)['quality'])
        q = quality()
        q.update(naturalness_a=1, naturalness_b=5)
        expected = 5 if request['a_is_original'] else 1
        result = self.module.assess(pair(), q, {'label': 'positive'}, {'label': 'positive'}, request['a_is_original'])
        self.assertEqual(result['perturbed_naturalness'], expected)


if __name__ == '__main__':
    unittest.main()
