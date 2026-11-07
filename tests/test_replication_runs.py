import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class ReplicationRunTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/replication_runs.py'
        self.assertTrue(path.exists(), 'Replication runner is missing')
        import replication_runs
        self.r = replication_runs

    def test_nli_sample_filters_unlabelled_rows_and_records_explicit_label_mapping(self):
        source = [{'premise': 'A person walks.', 'hypothesis': 'Someone moves.', 'label': 0},
                  {'premise': 'A person walks.', 'hypothesis': 'Nobody moves.', 'label': 2},
                  {'premise': 'A person walks.', 'hypothesis': 'Someone sings.', 'label': 1},
                  {'premise': 'Unknown.', 'hypothesis': 'Unknown.', 'label': -1}]
        result = self.r.select_rows(source, ['premise','hypothesis'], {0:1,1:2,2:0}, 3, 8240)
        self.assertEqual({r['source_id']:r['label'] for r in result}, {0:1,1:0,2:2})
        self.assertEqual(len({r['source_id'] for r in result}), 3)
        self.assertEqual(result, self.r.select_rows(source, ['premise','hypothesis'], {0:1,1:2,2:0}, 3, 8240))
        with self.assertRaises(ValueError):
            self.r.select_rows(source, ['premise','hypothesis'], {0:1,1:2,2:0}, 4, 8240)

    def test_outcome_rejects_premise_change_and_false_success(self):
        row = {'label': 1, 'original_prediction': 1, 'perturbed_prediction': 0,
               'result_type':'Successful', 'original_premise':'A person walks.', 'perturbed_premise':'A person runs.'}
        with self.assertRaises(ValueError):
            self.r.validate_outcome(row)
        row['perturbed_premise'] = row['original_premise']
        self.r.validate_outcome(row)
        row['original_prediction'] = 0
        with self.assertRaises(ValueError):
            self.r.validate_outcome(row)

    def test_metrics_use_all_examples_for_accuracy_and_only_eligible_for_asr(self):
        common={'num_queries':10,'original_word_count':5,'changed_word_count':1}
        rows=[dict(common,label=1,original_prediction=1,perturbed_prediction=0,result_type='Successful'),
              dict(common,label=1,original_prediction=1,perturbed_prediction=1,result_type='Failed'),
              dict(common,label=1,original_prediction=0,perturbed_prediction=0,result_type='Skipped')]
        result=self.r.metrics(rows)
        self.assertAlmostEqual(result['original_accuracy_percent'], 200/3)
        self.assertAlmostEqual(result['after_attack_accuracy_percent'], 100/3)
        self.assertEqual(result['attack_success_rate_percent'], 50)
        self.assertEqual(result['mean_perturbed_word_percent_successful'], 20)

    def test_released_readers_follow_original_cleaning_and_preserve_nli_case(self):
        self.assertTrue(hasattr(self.r, 'read_released'), 'Released-data reader is missing')
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            path=Path(folder)/'data'
            path.write_text("1 Film's GOOD.\n0 BAD!\n",encoding='utf-8')
            rows,names=self.r.read_released(path,False)
            self.assertEqual(rows[0]['inputs'], {'text':"film 's good"})
            self.assertEqual(rows[1]['inputs'], {'text':'bad !'})
            path.write_text('1 (good)\n',encoding='utf-8')
            rows,names=self.r.read_released(path,False)
            self.assertEqual([ord(c) for c in rows[0]['inputs']['text']], [92,40,32,103,111,111,100,32,92,41])
            path.write_text('1 good?\n',encoding='utf-8')
            rows,names=self.r.read_released(path,False)
            self.assertEqual([ord(c) for c in rows[0]['inputs']['text']], [103,111,111,100,32,92,63])
            path.write_text('entailment\tA Person walks .\tSomeone moves .\ncontradiction\tA Person walks .\tNobody moves .\n',encoding='utf-8')
            rows,names=self.r.read_released(path,True)
            self.assertEqual(rows[0]['inputs']['premise'],'A Person walks .')
            self.assertEqual([r['label'] for r in rows],[1,0])

    def test_model_dataset_mismatch_is_rejected_before_inference(self):
        self.assertTrue(hasattr(self.r, 'validate_configuration'), 'Configuration validation is missing')
        with self.assertRaises(ValueError):
            self.r.validate_configuration('snli','cnn-imdb','full')

    def test_resume_rejects_changed_original_inputs(self):
        sample=[{'row':0,'source_id':2,'label':1,'inputs':{'text':'a good film'}}]
        row={'row':0,'source_id':2,'label':1,'original_text':'another film',
             'original_prediction':1,'perturbed_prediction':0,'result_type':'Successful'}
        with self.assertRaises(ValueError):
            self.r.check_existing([row],sample)

    def test_clean_predictions_preserve_pair_order_and_validate_output_classes(self):
        self.assertTrue(hasattr(self.r,'clean_predictions'),'Clean baseline validation is missing')
        import numpy as np
        sample=[{'row':0,'source_id':2,'label':1,'inputs':{'premise':'A walks','hypothesis':'A moves'}}]
        observed=[]
        def wrapper(inputs):
            observed.extend(inputs)
            return np.array([[0.,2.,1.]])
        rows=self.r.clean_predictions(wrapper,sample,3)
        self.assertEqual(observed,[('A walks','A moves')])
        self.assertEqual(rows[0]['prediction'],1)
        with self.assertRaises(ValueError):
            self.r.clean_predictions(lambda inputs:np.array([[0.,1.]]),sample,3)


if __name__ == '__main__':
    unittest.main()
