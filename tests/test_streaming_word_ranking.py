from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


class StreamingWordRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from replication_runs import configure
        configure()

    def test_ranking_and_query_counts_match_the_batched_reference(self):
        from replication_runs import configure
        configure()
        from textattack.search_methods import GreedyWordSwapWIR
        from textattack.shared import AttackedText
        from textattack.goal_functions import UntargetedClassification
        from textattack.models.wrappers import ModelWrapper
        import torch
        from streaming_word_ranking import StreamingGreedyWordSwapWIR
        class Classifier(ModelWrapper):
            def __init__(self):self.model=torch.nn.Linear(1,2);self.batches=[]
            def __call__(self,texts):
                self.batches.append(len(texts))
                return torch.tensor([[3.,len(text)%13/20] for text in texts])
        text=AttackedText(' '.join(['bright']*70+['funny','story','with','good','actors']*100))
        observations=[]
        for kind in ('delete','unk'):
            observations=[]
            for method in (GreedyWordSwapWIR(kind),StreamingGreedyWordSwapWIR(kind)):
                wrapper=Classifier();goal=UntargetedClassification(wrapper,model_batch_size=32)
                goal.init_attack_example(text,0)
                method.goal_function=goal
                method.get_goal_results=goal.get_results
                method.get_indices_to_order=lambda value:(value.num_words,list(range(value.num_words)))
                order,stopped=method._get_index_order(text)
                observations.append((order.tolist(),stopped,goal.num_queries,wrapper.batches))
                if isinstance(method,StreamingGreedyWordSwapWIR):
                    self.assertLessEqual(len(goal._call_model_cache),256)
            self.assertEqual(observations[0],observations[1])

    def test_unsupported_ranking_is_rejected(self):
        from streaming_word_ranking import StreamingGreedyWordSwapWIR
        with self.assertRaises(ValueError):StreamingGreedyWordSwapWIR('random')


if __name__=='__main__':unittest.main()
