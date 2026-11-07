"""Validated maintained and original-release classifier loaders."""
from pathlib import Path
import re

NLI_MODELS={
    'bert-base-uncased-snli':('textattack/bert-base-uncased-snli','d4ef8a69a50bc95cc074514f4b798c67f572163a'),
    'bert-base-uncased-mnli':('textattack/bert-base-uncased-MNLI','3a97f689528cbd91bcc71ab29ea6c20c089d8f28')}


def load_original_bert(root,dataset):
    import torch
    from transformers import BertConfig,BertTokenizer,BertForSequenceClassification
    from textattack.models.wrappers import HuggingFaceModelWrapper
    path=Path(root)/'models/original_textfooler'/('bert_'+dataset)
    config=BertConfig.from_json_file(path/'bert_config.json')
    config.num_labels=3 if dataset in ('snli','mnli') else 4 if dataset=='ag' else 2
    state=torch.load(path/'pytorch_model.bin',map_location='cpu',weights_only=True)
    renamed={key.replace('.gamma','.weight').replace('.beta','.bias'):value for key,value in state.items()}
    if len(renamed)!=len(state):
        raise ValueError('Legacy parameter names collide')
    model=BertForSequenceClassification(config)
    model.load_state_dict(renamed,strict=True)
    model.eval()
    maximum=256 if dataset in ('fake','imdb','yelp') else 128
    tokenizer=BertTokenizer(vocab_file=str(path/'vocab.txt'),do_lower_case=True,model_max_length=maximum)
    wrapper=HuggingFaceModelWrapper(model,tokenizer)
    wrapper.release_metadata={'kind':'authors_original_checkpoint','folder':str(path.resolve()),
                              'maximum_length':maximum,'parameter_loading':'strict; legacy gamma/beta names mapped to weight/bias'}
    return wrapper


def verify_glove_alignment(weights,words,vectors):
    import numpy as np
    offset=len(words)+2-len(weights)
    if offset not in (0,1) or weights.shape[1]!=vectors.shape[1]:
        raise ValueError('Original embedding dimensions do not match the GloVe vocabulary')
    count=len(words)-offset
    for start in range(0,count,4096):
        expected=np.asarray(vectors[offset+start:offset+min(start+4096,count)],dtype=np.float32)
        expected=expected/np.maximum(np.linalg.norm(expected,axis=1,keepdims=True),np.finfo(np.float32).tiny)
        actual=np.asarray(weights[start:min(start+4096,count)])
        if not np.allclose(expected,actual,atol=2e-6,rtol=2e-5):
            raise ValueError('Original checkpoint embedding order does not match the cached vocabulary')
    mapping={str(word):index for index,word in enumerate(words[offset:])}
    mapping.update({'<oov>':count,'<pad>':count+1})
    return mapping


def load_original_word(root,family,dataset):
    import numpy as np
    import torch
    from torch import nn
    from textattack.models.wrappers import ModelWrapper
    path=Path(root)/'models/original_textfooler'/f'{family}_{dataset}'/'pytorch_model.bin'
    state=torch.load(path,map_location='cpu',weights_only=True)
    embedding=state['emb_layer.embedding.weight']
    glove=Path(root)/'data/cache/textattack/word_embeddings/glove200'
    words=np.load(glove/'glove.wordlist.npy')
    vectors=np.load(glove/'glove.6B.200d.mat.npy',mmap_mode='r')
    mapping=verify_glove_alignment(embedding.numpy(),words,vectors)
    class OriginalWordModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb_layer=nn.Module()
            self.emb_layer.embedding=nn.Embedding(*embedding.shape)
            if family=='cnn':
                self.encoder=nn.Module()
                self.encoder.convs1=nn.ModuleList([nn.Conv2d(1,state[f'encoder.convs1.{i}.weight'].shape[0],
                    tuple(state[f'encoder.convs1.{i}.weight'].shape[2:])) for i in range(3)])
            else:
                depth=max(int(re.search(r'_l(\d+)',key).group(1)) for key in state if key.startswith('encoder.weight_ih_l'))+1
                self.encoder=nn.LSTM(embedding.shape[1],state['encoder.weight_hh_l0'].shape[1],depth,bidirectional=True)
            self.out=nn.Linear(state['out.weight'].shape[1],state['out.weight'].shape[0])
        def forward(self,ids):
            emb=self.emb_layer.embedding(ids)
            if family=='cnn':
                output=torch.cat([torch.relu(conv(emb.unsqueeze(1))).squeeze(3).max(dim=2).values for conv in self.encoder.convs1],dim=1)
            else:
                output=self.encoder(emb.transpose(0,1))[0].max(dim=0).values
            return self.out(output)
    model=OriginalWordModel()
    model.load_state_dict(state,strict=True)
    model.eval()
    class OriginalWordWrapper(ModelWrapper):
        def __init__(self):
            self.model=model
            self.tokenizer=None
            self.release_metadata={'kind':'authors_original_checkpoint','checkpoint':str(path.resolve()),
                'vocabulary':'Every original embedding row verified against normalized GloVe 6B 200d',
                'padding':'Left padding to batch maximum; minimum 5 tokens; no truncation'}
        def __call__(self,texts):
            outputs=[]
            device=next(self.model.parameters()).device
            for start in range(0,len(texts),32):
                batch=[text.split() for text in texts[start:start+32]]
                length=max(5,max(len(words) for words in batch))
                ids=[[mapping['<pad>']]*(length-len(words))+[mapping.get(word,mapping['<oov>']) for word in words] for words in batch]
                with torch.no_grad():
                    outputs.append(self.model(torch.tensor(ids,device=device)))
            return torch.cat(outputs)
    return OriginalWordWrapper()


def load_model(root,alias):
    if alias.startswith('original-'):
        _,family,dataset=alias.split('-')
        if family=='bert':
            return load_original_bert(root,dataset)
        if family in ('cnn','lstm'):
            return load_original_word(root,family,dataset)
        raise ValueError('Unknown original model family')
    if alias in NLI_MODELS:
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        from textattack.models.wrappers import HuggingFaceModelWrapper
        name,revision=NLI_MODELS[alias]
        cache=str(Path(root)/'data/cache/textattack')
        model=AutoModelForSequenceClassification.from_pretrained(name,revision=revision,cache_dir=cache,trust_remote_code=False)
        tokenizer=AutoTokenizer.from_pretrained(name,revision=revision,cache_dir=cache,trust_remote_code=False)
        tokenizer.model_max_length=128
        return HuggingFaceModelWrapper(model,tokenizer)
    from textattack.model_args import ModelArgs
    return ModelArgs._create_model_from_args(ModelArgs(model=alias))
