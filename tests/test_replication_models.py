import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


class ReplicationModelTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'scripts/replication_models.py').exists(),'Replication model loader is missing')
        from type1_models import configure_cache
        configure_cache(ROOT)
        import replication_models
        self.m=replication_models

    def test_legacy_bert_weights_load_strictly_with_documented_maximum_length(self):
        import torch
        from transformers import BertConfig,BertForSequenceClassification
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            root=Path(directory)
            folder=root/'models/original_textfooler/bert_fake'
            folder.mkdir(parents=True)
            config=BertConfig(vocab_size=8,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,num_labels=2)
            model=BertForSequenceClassification(config).eval()
            state={key.replace('.LayerNorm.weight','.LayerNorm.gamma').replace('.LayerNorm.bias','.LayerNorm.beta'):value for key,value in model.state_dict().items()}
            torch.save(state,folder/'pytorch_model.bin')
            (folder/'bert_config.json').write_text(config.to_json_string())
            (folder/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nperson\nmoves\n')
            wrapper=self.m.load_original_bert(root,'fake')
            self.assertEqual(wrapper.tokenizer.model_max_length,256)
            self.assertEqual(wrapper(['a person moves']).shape,(1,2))
            self.assertTrue(torch.equal(wrapper.model.state_dict()['classifier.weight'],model.state_dict()['classifier.weight']))

    def test_original_vocabulary_requires_numeric_embedding_alignment(self):
        import numpy as np
        self.assertTrue(hasattr(self.m,'verify_glove_alignment'),'Vocabulary alignment validation is missing')
        vectors=np.array([[1.,0.],[0.,1.],[1.,1.]],dtype=np.float32)
        normalized=vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
        weights=np.concatenate([normalized,np.zeros((2,2),dtype=np.float32)])
        mapping=self.m.verify_glove_alignment(weights,['one','two','three'],vectors)
        self.assertEqual(mapping,{'one':0,'two':1,'three':2,'<oov>':3,'<pad>':4})
        with self.assertRaises(ValueError):
            self.m.verify_glove_alignment(weights[[1,0,2,3,4]],['one','two','three'],vectors)


if __name__=='__main__': unittest.main()
