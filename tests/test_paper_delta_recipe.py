"""Regression checks against the installed TextAttack recipe."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from type1_models import configure_cache
configure_cache(ROOT)


class PaperDeltaRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from textattack.models.wrappers import ModelWrapper
        from textfooler_paper_delta_0_7 import Attack

        class FixedClassifier(ModelWrapper):
            def __init__(self):
                import torch
                self.model = torch.nn.Identity()

            def __call__(self, texts):
                return np.tile([0.8, 0.2], (len(texts), 1))

        cls.attack = Attack(FixedClassifier())

    def test_stopwords_and_already_modified_positions_are_not_attackable(self):
        from textattack.shared import AttackedText
        text = AttackedText("my canoe to make more steady")
        text.attack_attrs["modified_indices"] = {1}
        allowed = set(range(text.num_words))
        for constraint in self.attack.pre_transformation_constraints:
            allowed &= constraint(text, self.attack.transformation)
        self.assertEqual(allowed, {3, 5})

    def test_sentence_pair_premise_is_not_attackable(self):
        from collections import OrderedDict
        from textattack.shared import AttackedText
        text = AttackedText(OrderedDict([("premise", "bright skies"), ("hypothesis", "dark clouds")]))
        allowed = set(range(text.num_words))
        for constraint in self.attack.pre_transformation_constraints:
            allowed &= constraint(text, self.attack.transformation)
        self.assertFalse(allowed & {0, 1})

    def test_embedding_constraint_enforces_paper_delta(self):
        import numpy as np
        from textattack.constraints.semantics import WordEmbeddingDistance
        from textattack.shared import AttackedText, WordEmbedding
        embedding = [c for c in self.attack.constraints if isinstance(c, WordEmbeddingDistance)]
        self.assertEqual(len(embedding), 1)
        original_embedding = embedding[0].embedding
        embedding[0].embedding = WordEmbedding(
            np.array([[1., 0.], [.65, np.sqrt(1 - .65**2)], [.8, .6]]),
            {"canoe": 0, "raft": 1, "rowboat": 2}, {0: "canoe", 1: "raft", 2: "rowboat"})
        try:
            original = AttackedText("canoe")
            weak = original.replace_word_at_index(0, "raft")
            strong = original.replace_word_at_index(0, "rowboat")
            for candidate in (weak, strong):
                candidate.attack_attrs["last_transformation"] = self.attack.transformation
            self.assertFalse(embedding[0](weak, original))
            self.assertTrue(embedding[0](strong, original))
        finally:
            embedding[0].embedding = original_embedding


if __name__ == "__main__":
    unittest.main()
