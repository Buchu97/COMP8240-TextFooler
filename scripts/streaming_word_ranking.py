"""Memory-bounded TextFooler word ranking for long classification inputs."""
import math
import lru
import numpy as np
from textattack import Attack
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding
from fast_text_edits import FastTextEdits,IndexedWordSwapEmbedding


class StreamingGreedyWordSwapWIR(GreedyWordSwapWIR):
    def __init__(self,wir_method='delete',unk_token='[UNK]'):
        if wir_method not in ('delete','unk'):raise ValueError('Streaming ranking supports delete and unk methods only')
        super().__init__(wir_method,unk_token)
        self._edits=FastTextEdits()

    def _get_index_order(self,initial_text,max_len=-1):
        if not math.isinf(self.goal_function.query_budget):
            raise ValueError('Streaming ranking requires the unbounded TextFooler query budget')
        _,indices=self.get_indices_to_order(initial_text)
        size=self.goal_function.batch_size
        self.goal_function._call_model_cache=lru.LRU(max(256,size))
        scores=[];search_over=False
        caching=self.goal_function.use_cache
        self.goal_function.use_cache=False
        try:
            for start in range(0,len(indices),size):
                texts=[self._edits.replace(initial_text,index,'' if self.wir_method=='delete' else self.unk_token)
                       for index in indices[start:start+size]]
                results,search_over=self.get_goal_results(texts)
                scores.extend(result.score for result in results)
                del results,texts
        finally:
            self.goal_function.use_cache=caching
        return np.array(indices)[(-np.array(scores)).argsort()],search_over


def build_streaming_attack(attack):
    old=attack.search_method
    if type(old) is not GreedyWordSwapWIR or old.wir_method not in ('delete','unk'):
        raise ValueError('Expected maintained TextFooler word ranking')
    if type(attack.transformation) is not WordSwapEmbedding:
        raise ValueError('Expected the maintained embedding word substitution')
    transformation=IndexedWordSwapEmbedding(max_candidates=attack.transformation.max_candidates,
        embedding=attack.transformation.embedding,letters_to_insert=attack.transformation.letters_to_insert)
    attack.goal_function._call_model_cache=lru.LRU(256)
    return Attack(attack.goal_function,
        list(attack.pre_transformation_constraints)+list(attack.constraints),transformation,
        StreamingGreedyWordSwapWIR(old.wir_method,old.unk_token),
        transformation_cache_size=1,constraint_cache_size=256)
