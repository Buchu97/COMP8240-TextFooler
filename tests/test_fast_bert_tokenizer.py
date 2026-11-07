from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


class FastBertTokenizerTests(unittest.TestCase):
    def test_converted_backend_preserves_vocabulary_special_ids_and_encoded_batches(self):
        from transformers import BertTokenizer
        from fast_bert_tokenizer import make_fast_tokenizer
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            path=Path(directory)/'vocab.txt'
            path.write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','a','good','bad',
                'film','the','can','not',"'",'t','.',',','!','?','(',')','##s','##ly','naive','cafe'])+'\n')
            slow=BertTokenizer(vocab_file=str(path),do_lower_case=True,model_max_length=16)
            fast=make_fast_tokenizer(slow)
            texts=['A GOOD film!',"The film can't be bad.",'naïve café','a '+('good '*50),
                   '[CLS] a [MASK] film [SEP]','\tgood\nfilm\x00','(good), bad?','中文电影']
            settings=dict(padding='max_length',max_length=16,truncation=True,add_special_tokens=True)
            self.assertEqual(dict(slow(texts,**settings)),dict(fast(texts,**settings)))
            self.assertEqual(slow.get_vocab(),fast.get_vocab())
            self.assertEqual(slow.all_special_ids,fast.all_special_ids)
            self.assertEqual(slow.model_max_length,fast.model_max_length)


if __name__=='__main__':unittest.main()
