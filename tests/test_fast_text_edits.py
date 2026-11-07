from collections import OrderedDict
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


class FastTextEditsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from replication_runs import configure
        configure()
        from textattack.shared import AttackedText
        cls.AttackedText=AttackedText

    def assert_equivalent(self,old,new):
        import numpy as np
        self.assertEqual(old._text_input,new._text_input)
        self.assertEqual(old.words,new.words)
        self.assertEqual(old.words_per_input,new.words_per_input)
        self.assertEqual(set(old.attack_attrs),set(new.attack_attrs))
        for field in ('modified_indices','newly_modified_indices'):
            self.assertEqual(old.attack_attrs[field],new.attack_attrs[field])
        np.testing.assert_array_equal(old.attack_attrs['original_index_map'],new.attack_attrs['original_index_map'])
        self.assertEqual(old.attack_attrs.get('label_names'),new.attack_attrs.get('label_names'))
        self.assertEqual(old.attack_attrs['previous_attacked_text'].text,new.attack_attrs['previous_attacked_text'].text)

    def test_single_edits_match_legacy_text_and_tracking(self):
        from fast_text_edits import FastTextEdits
        edits=FastTextEdits()
        examples=['a b','a b.','a  b, c?','(great) can\'t save a weak script!',
                  'red red red blue',r'They \( were \) good !','Café😊with odd—punctuation42',
                  'a <SPLIT> b','a b.c','a.*b','123abc 42foo foo42 x_foo alpha@beta']
        for text in examples:
            base=self.AttackedText(text,attack_attrs={'label_names':['negative','positive']})
            for index,word in enumerate(base.words):
                for replacement in ('better','','two words','not-good',"'better'",'<SPLIT>',word):
                    with self.subTest(text=text,index=index,replacement=replacement):
                        self.assert_equivalent(base.replace_word_at_index(index,replacement),edits.replace(base,index,replacement))

    def test_chained_swaps_deletions_and_pair_fallback_match_legacy(self):
        from fast_text_edits import FastTextEdits
        edits=FastTextEdits()
        old=self.AttackedText('a bright but weak story .')
        new=self.AttackedText('a bright but weak story .')
        for index,replacement in [(1,'clever'),(3,'poor'),(0,''),(2,'moving')]:
            old=old.replace_word_at_index(index,replacement)
            new=edits.replace(new,index,replacement)
            self.assert_equivalent(old,new)
        pair=self.AttackedText(OrderedDict(premise='A dog runs.',hypothesis='An animal moves.'))
        self.assert_equivalent(pair.replace_word_at_index(4,'creature'),edits.replace(pair,4,'creature'))

    def test_indexed_windows_match_reference_after_chained_edits(self):
        from fast_text_edits import FastTextEdits,index_word_positions
        examples=['A,a A! a  B b.c (D) e?',"'one' two-three can't FOUR five six seven eight nine ten",
                  '123abc 42foo foo42 x_foo alpha@beta', 'red '*120+'blue END.']
        for text in examples:
            old=self.AttackedText(text);new=index_word_positions(self.AttackedText(text))
            for position,replacement in [(0,'better'),(2,''),(1,"'good'"),(0,'')]:
                old=old.replace_word_at_index(position,replacement)
                new=FastTextEdits().replace(new,position,replacement)
                for index in range(old.num_words):
                    self.assertEqual(old._text_index_of_word_index(index),new._text_index_of_word_index(index))
                    for window in (1,2,3,15,200):
                        self.assertEqual(old.text_window_around_index(index,window),new.text_window_around_index(index,window))

    def test_indexed_lookup_avoids_repeated_reference_scan(self):
        from fast_text_edits import FastTextEdits,index_word_positions
        from unittest.mock import patch
        text='first '+'repeated '*200+'last'
        base=index_word_positions(self.AttackedText(text))
        changed=FastTextEdits().replace(base,190,'different')
        expected=self.AttackedText(changed.text).text_window_around_index(190,15)
        with patch.object(self.AttackedText,'_text_index_of_word_index',side_effect=AssertionError('Repeated full scan')):
            self.assertEqual(changed.text_window_around_index(190,15),expected)
            self.assertIn('last',base.text_window_around_index(201,15))


if __name__=='__main__':unittest.main()
