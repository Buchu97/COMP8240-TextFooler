"""Exact whole-batch memoization for a frozen evaluation-mode classifier."""
from collections import OrderedDict
from itertools import chain
import torch
from textattack.models.wrappers import ModelWrapper


class EncodedBatchCachedWrapper(ModelWrapper):
    def __init__(self,wrapper,capacity=128):
        if capacity<1:raise ValueError('Cache capacity must be positive')
        self.model=wrapper.model;self.tokenizer=wrapper.tokenizer
        self.release_metadata=getattr(wrapper,'release_metadata',None)
        self.capacity=capacity;self._cache=OrderedDict()
        self._frozen_tensors=tuple(chain(self.model.parameters(),self.model.buffers()))
        self._frozen_signature=self._signature()
        if self.model.training:raise ValueError('Classifier must be in evaluation mode')
        self.reset_statistics()

    def _signature(self):
        return tuple((id(value),value.data_ptr(),value._version,str(value.device),value.dtype)
                     for value in chain(self.model.parameters(),self.model.buffers()))

    def reset_statistics(self,clear_cache=False):
        self._stats=dict(batch_requests=0,requested_examples=0,forward_batches=0,forward_examples=0,cache_hits=0)
        if clear_cache:self._cache.clear()

    def execution_statistics(self):return dict(self._stats)

    def __call__(self,texts):
        if self.model.training or self._signature()!=self._frozen_signature:
            raise ValueError('Classifier state changed; create a new evaluation cache')
        maximum=512 if self.tokenizer.model_max_length==int(1e30) else self.tokenizer.model_max_length
        inputs=self.tokenizer(texts,add_special_tokens=True,padding='max_length',max_length=maximum,
                              truncation=True,return_tensors='pt')
        key=tuple((name,str(value.dtype),tuple(value.shape),value.detach().cpu().contiguous().numpy().tobytes())
                  for name,value in sorted(inputs.items()))
        self._stats['batch_requests']+=1;self._stats['requested_examples']+=len(texts)
        device=next(self.model.parameters()).device
        if key in self._cache:
            self._stats['cache_hits']+=1
            value=self._cache.pop(key);self._cache[key]=value
            return value.to(device).clone()
        with torch.no_grad():
            outputs=self.model(**{name:value.to(device) for name,value in inputs.items()}).logits
        self._stats['forward_batches']+=1;self._stats['forward_examples']+=len(texts)
        self._cache[key]=outputs.detach().cpu().clone()
        while len(self._cache)>self.capacity:self._cache.popitem(last=False)
        return outputs
