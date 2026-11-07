"""Full paper-delta TextFooler and its sentence-similarity ablation."""


def build_attack(wrapper, variant='full'):
    if variant not in ('full', 'no_sentence_similarity'):
        raise ValueError('Unknown attack variant')
    from textfooler_paper_delta_0_7 import Attack
    attack = Attack(wrapper)
    if variant == 'full':
        return attack
    from textattack import Attack as TextAttack
    from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
    remaining = [c for c in attack.constraints if not isinstance(c, UniversalSentenceEncoder)]
    if len(attack.constraints) - len(remaining) != 1:
        raise ValueError('Expected exactly one sentence-similarity constraint')
    return TextAttack(attack.goal_function,
                      list(attack.pre_transformation_constraints) + remaining,
                      attack.transformation, attack.search_method)
