"""Convert the released BERT tokenizer to an equivalent compiled WordPiece backend."""


def make_fast_tokenizer(slow):
    from transformers import BertTokenizer,BertTokenizerFast
    from transformers.convert_slow_tokenizer import convert_slow_tokenizer
    if type(slow) is not BertTokenizer or not slow.do_basic_tokenize:
        raise ValueError('Expected the released basic BERT tokenizer')
    fast=BertTokenizerFast(tokenizer_object=convert_slow_tokenizer(slow),
        do_lower_case=slow.do_lower_case,strip_accents=slow.basic_tokenizer.strip_accents,
        tokenize_chinese_chars=slow.basic_tokenizer.tokenize_chinese_chars,
        model_max_length=slow.model_max_length,padding_side=slow.padding_side,
        truncation_side=slow.truncation_side,**slow.special_tokens_map)
    if fast.get_vocab()!=slow.get_vocab() or fast.all_special_ids!=slow.all_special_ids:
        raise ValueError('Converted vocabulary or special-token IDs differ')
    return fast
