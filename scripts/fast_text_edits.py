"""Indexed single-word edits with the maintained AttackedText tracking semantics."""
from collections import OrderedDict
from types import MethodType
from textattack.shared import AttackedText
from textattack.shared.utils import words_from_text
from textattack.transformations import WordSwapEmbedding


def _cached_word_position(source,index):
    if 0<=index<len(source.words):return source._indexed_positions[index]
    return AttackedText._text_index_of_word_index(source,index)


def index_word_positions(source):
    raw=source.text
    if len(source._text_input)!=1 or not raw.isascii() or AttackedText.SPLIT_TOKEN in raw:return source
    if not hasattr(source,'_indexed_positions'):
        lower=raw.lower();cursor=0;positions=[]
        for word in source.words:
            start=lower.find(word.lower(),cursor)
            positions.append(start);cursor=start+len(word)
        source._indexed_positions=tuple(positions)
        source._text_index_of_word_index=MethodType(_cached_word_position,source)
    return source


class _IndexedAttackedText(AttackedText):
    def __init__(self,text_input,attack_attrs,known_words,prefix_positions,edit_index):
        self._known_words=known_words
        self._prefix_positions=prefix_positions
        self._edit_index=edit_index
        self._indexed_lower_text=None
        super().__init__(text_input,attack_attrs=attack_attrs)
        self._words=known_words
        self._words_per_input=[known_words]

    @property
    def words(self):
        return self._known_words

    def _text_index_of_word_index(self,index):
        if not 0<=index<len(self.words):return super()._text_index_of_word_index(index)
        if index<self._edit_index:return self._prefix_positions[index]
        if self._indexed_lower_text is None:self._indexed_lower_text=self.text.lower()
        cursor=0 if self._edit_index==0 else self._prefix_positions[self._edit_index-1]+len(self.words[self._edit_index-1])
        for word in self.words[self._edit_index:index+1]:
            start=self._indexed_lower_text.find(word.lower(),cursor);cursor=start+len(word)
        return start


class FastTextEdits:
    def __init__(self):
        self._source=None
        self._spans=[]

    def replace(self,source,index,replacement):
        words=source.words
        if (len(source._text_input)!=1 or not isinstance(replacement,str) or
            not 0<=index<len(words) or (replacement and len(words_from_text(replacement))!=1)):
            return source.replace_word_at_index(index,replacement)
        raw=next(iter(source._text_input.values()))
        if AttackedText.SPLIT_TOKEN in raw or AttackedText.SPLIT_TOKEN in replacement:
            return source.replace_word_at_index(index,replacement)
        if source is not self._source:
            cursor=0;spans=[]
            for word in words:
                start=raw.index(word,cursor);end=start+len(word)
                spans.append((start,end));cursor=end
            self._source=source;self._spans=spans
        start,end=self._spans[index]
        prefix,suffix=raw[:start],raw[end:]
        attrs={}
        if 'label_names' in source.attack_attrs:attrs['label_names']=source.attack_attrs['label_names']
        attrs.update(previous_attacked_text=source,newly_modified_indices=set(),
                     modified_indices=source.attack_attrs['modified_indices'].copy(),
                     original_index_map=source.attack_attrs['original_index_map'].copy())
        if replacement=='':
            attrs['modified_indices']={value if value<index else value-1 for value in attrs['modified_indices'] if value!=index}
            mapping=attrs['original_index_map']
            mapping[mapping==index]=-1;mapping[mapping>index]-=1
            if suffix:
                if index==0 and suffix.startswith(' '):suffix=suffix[1:]
                elif index>0 and prefix.endswith(' '):prefix=prefix[:-1]
        elif words[index]!=replacement:
            attrs['modified_indices'].add(index);attrs['newly_modified_indices'].add(index)
        key=next(iter(source._text_input))
        text_input=OrderedDict([(key,prefix+replacement+suffix)])
        if raw.isascii() and replacement.isascii():
            index_word_positions(source)
            known=words[:]
            if replacement=='':del known[index]
            else:known[index]=words_from_text(replacement)[0]
            return _IndexedAttackedText(text_input,attrs,known,source._indexed_positions,index)
        return AttackedText(text_input,attack_attrs=attrs)


class IndexedWordSwapEmbedding(WordSwapEmbedding):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._edits=FastTextEdits()

    def _get_transformations(self,current_text,indices_to_modify):
        results=[]
        for index in indices_to_modify:
            word=current_text.words[index]
            for replacement in self._get_replacement_words(word):
                if replacement!=word:results.append(self._edits.replace(current_text,index,replacement))
        return results
