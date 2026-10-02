"""TextFooler maintained recipe with the paper-stated word delta of 0.7.

Only the WordEmbeddingDistance threshold is changed relative to TextAttack
v0.3.10's TextFoolerJin2019 recipe. WordSwapEmbedding(max_candidates=50), the
goal function, the remaining constraints, and the search method are retained.
"""

from textattack import Attack as TextAttack
from textattack.attack_recipes import TextFoolerJin2019
from textattack.constraints.semantics import WordEmbeddingDistance


def Attack(model_wrapper):
    maintained = TextFoolerJin2019.build(model_wrapper)
    constraints = []
    replacements = 0

    for constraint in maintained.constraints:
        if isinstance(constraint, WordEmbeddingDistance):
            constraints.append(WordEmbeddingDistance(min_cos_sim=0.7))
            replacements += 1
        else:
            constraints.append(constraint)

    if replacements != 1:
        raise RuntimeError(
            "Expected exactly one WordEmbeddingDistance constraint; "
            f"found {replacements}."
        )

    return TextAttack(
        maintained.goal_function,
        constraints,
        maintained.transformation,
        maintained.search_method,
    )
