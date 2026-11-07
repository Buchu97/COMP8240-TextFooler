from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


class EncodedBatchCacheTests(unittest.TestCase):
    def setUp(self):
        from replication_runs import configure
        configure()
        import torch
        from encoded_batch_cache import EncodedBatchCachedWrapper
        class Tokenizer:
            model_max_length=4
            def __call__(self,texts,**kwargs):
                ids=torch.tensor([[int(token) for token in text.split()[:4]] for text in texts])
                return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'token_type_ids':torch.zeros_like(ids)}
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.anchor=torch.nn.Parameter(torch.tensor(.5));self.calls=0
            def forward(self,input_ids,**kwargs):
                self.calls+=1
                value=input_ids.float().sum(-1)+input_ids.float().sum()+self.anchor
                return SimpleNamespace(logits=torch.stack([value,-value],dim=-1))
        self.torch=torch;self.model=Model().eval()
        self.wrapper=EncodedBatchCachedWrapper(SimpleNamespace(model=self.model,tokenizer=Tokenizer()),capacity=2)

    def test_identical_encoded_batch_reuses_exact_logits_and_returns_fresh_tensors(self):
        first=self.wrapper(['1 2 3 4 alpha']);expected=first.clone()
        first[0,0]=999
        second=self.wrapper(['1 2 3 4 beta'])
        self.assertTrue(self.torch.equal(second,expected));self.assertEqual(self.model.calls,1)
        second[0,0]=888
        self.assertTrue(self.torch.equal(self.wrapper(['1 2 3 4 gamma']),expected))
        stats=self.wrapper.execution_statistics()
        self.assertEqual(stats['batch_requests'],3);self.assertEqual(stats['cache_hits'],2)
        self.assertEqual(stats['forward_batches'],1);self.assertEqual(stats['requested_examples'],3)

    def test_batch_shape_order_and_lru_eviction_are_not_conflated(self):
        self.wrapper(['1 2 3 4'])
        original=self.wrapper(['1 2 3 4','4 3 2 2'])
        reordered=self.wrapper(['4 3 2 2','1 2 3 4'])
        self.assertFalse(self.torch.equal(original,reordered))
        self.wrapper(['1 2 3 4'])
        self.assertEqual(self.model.calls,4)

    def test_training_or_weight_mutation_rejects_stale_evaluation_cache(self):
        self.wrapper(['1 2 3 4']);self.model.train()
        with self.assertRaises(ValueError):self.wrapper(['1 2 3 4'])
        self.model.eval()
        with self.torch.no_grad():self.model.anchor.add_(1)
        with self.assertRaises(ValueError):self.wrapper(['1 2 3 4'])

    def test_recovered_outcomes_do_not_invent_physical_execution_counts(self):
        from verify_replication_artifacts import validate_execution_statistics
        stats={'status':'unavailable_after_interruption','reason':'Completed CSV recovered after interruption'}
        self.assertEqual(validate_execution_statistics(stats,[{'num_queries':12}]),'unavailable_after_interruption')
        with self.assertRaises(ValueError):validate_execution_statistics(dict(stats,forward_batches=0),[])

    def test_physical_statistics_reconcile_only_the_executed_segment(self):
        from verify_replication_artifacts import validate_execution_statistics
        stats={'status':'available','start_row':1,'clean_baseline_included':False,'logical_candidate_queries':12,
               'batch_requests':3,'requested_examples':10,'forward_batches':1,'forward_examples':4,'cache_hits':2}
        rows=[{'num_queries':99},{'num_queries':12}]
        self.assertEqual(validate_execution_statistics(stats,rows),'available')
        with self.assertRaises(ValueError):validate_execution_statistics(dict(stats,start_row=-1),rows)
        with self.assertRaises(ValueError):validate_execution_statistics(dict(stats,logical_candidate_queries=111),rows)


if __name__=='__main__':unittest.main()
