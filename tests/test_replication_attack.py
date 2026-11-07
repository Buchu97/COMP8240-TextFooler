import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class ReplicationAttackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'scripts/replication_attack.py'

    def test_ablation_preserves_every_other_constraint_and_hypothesis_restriction(self):
        self.assertTrue(self.path.exists(), 'Ablation recipe is missing')
        import numpy as np
        import torch
        from collections import OrderedDict
        from type1_models import configure_cache
        configure_cache(ROOT)
        from textattack.models.wrappers import ModelWrapper
        from textattack.shared import AttackedText
        from replication_attack import build_attack
        class Fixed(ModelWrapper):
            model = torch.nn.Identity()
            def __call__(self, texts):
                return np.tile([.8, .2], (len(texts), 1))
        full, ablated = build_attack(Fixed(), 'full'), build_attack(Fixed(), 'no_sentence_similarity')
        def names(attack):
            return [type(c).__name__ for c in attack.constraints]
        self.assertEqual(names(full), ['WordEmbeddingDistance', 'PartOfSpeech', 'UniversalSentenceEncoder'])
        self.assertEqual(names(ablated), ['WordEmbeddingDistance', 'PartOfSpeech'])
        self.assertEqual(ablated.constraints[0].min_cos_sim, .7)
        self.assertEqual(ablated.transformation.max_candidates, 50)
        text = AttackedText(OrderedDict([('premise', 'bright skies'), ('hypothesis', 'my canoe floats')]))
        text.attack_attrs['modified_indices'] = {3}
        allowed = set(range(text.num_words))
        for c in ablated.pre_transformation_constraints:
            allowed &= c(text, ablated.transformation)
        self.assertEqual(allowed, {4})
        with self.assertRaises(ValueError):
            build_attack(Fixed(), 'unknown')


if __name__ == '__main__':
    unittest.main()
