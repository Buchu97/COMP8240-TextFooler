from pathlib import Path
import sys
import unittest
import tempfile
import math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


class ReplicationReportTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'scripts/replication_report.py').exists(),'Replication report is missing')
        import replication_report
        self.r=replication_report

    def pair(self):
        common={'dataset':'mr','model_alias':'original-bert-mr','evaluation_sha256':'same-data',
                'model_state_sha256':'same-model','seed':8240,'tokenizer_max_length':128,
                'recipe_sha256':'same-recipe','ablation_sha256':'same-ablation'}
        full=[{'row':0,'source_id':1,'label':1,'original_text':'a good film','original_prediction':1,
               'perturbed_prediction':1,'result_type':'Failed','num_queries':10,'original_word_count':3,'changed_word_count':0},
              {'row':1,'source_id':2,'label':0,'original_text':'a bad film','original_prediction':0,
               'perturbed_prediction':1,'result_type':'Successful','num_queries':12,'original_word_count':3,'changed_word_count':1}]
        ablated=[dict(full[0],perturbed_prediction=0,result_type='Successful',num_queries=8,changed_word_count=1),dict(full[1])]
        return {'metadata':dict(common,variant='full'),'rows':full},{'metadata':dict(common,variant='no_sentence_similarity'),'rows':ablated}

    def test_paired_ablation_uses_shared_eligible_denominator_and_discordant_pairs(self):
        full,ablated=self.pair()
        result=self.r.compare_pair(full,ablated)
        self.assertEqual(result['eligible'],2)
        self.assertEqual(result['full_only_success'],0)
        self.assertEqual(result['no_sentence_only_success'],1)
        self.assertEqual(result['both_successful'],1)
        self.assertEqual(result['asr_change_points'],50)
        self.assertEqual(result['mcnemar_exact_p'],1)

    def test_paired_comparison_refuses_different_samples_or_initial_predictions(self):
        full,ablated=self.pair()
        ablated['metadata']['evaluation_sha256']='different'
        with self.assertRaises(ValueError):self.r.compare_pair(full,ablated)
        full,ablated=self.pair()
        ablated['rows'][0]['original_prediction']=0
        with self.assertRaises(ValueError):self.r.compare_pair(full,ablated)

    def test_pair_refuses_loader_runner_or_dependency_drift(self):
        for field,left,right in [('runner_sha256','a','b'),('model_loader_sha256','a','b'),
                                 ('packages',{'textattack':'0.3.10'},{'textattack':'other'})]:
            with self.subTest(field=field):
                full,ablated=self.pair()
                full['metadata'][field]=left;ablated['metadata'][field]=right
                with self.assertRaises(ValueError):self.r.compare_pair(full,ablated)

    def test_paper_comparison_keeps_metric_definitions_separate(self):
        self.assertTrue(hasattr(self.r,'paper_comparison'),'Paper comparison is missing')
        run={'metadata':{'dataset':'mr','model_alias':'original-bert-mr'},
             'metrics':{'original_accuracy_percent':90.4,'after_attack_accuracy_percent':24.8,
                        'mean_queries_eligible':63.,'mean_perturbed_percent_whitespace_successful':20.}}
        reference={'classification':{'table':'Table 3','model_order':['bert'],'dataset_order':['mr'],
            'original_accuracy_percent':[86.],'after_attack_accuracy_percent':[11.5],
            'queries':[166],'perturbed_words_percent':[16.7],'semantic_similarity':[.65]}}
        result=self.r.paper_comparison(run,reference)
        self.assertAlmostEqual(result['clean_difference_points'],4.4)
        self.assertAlmostEqual(result['after_attack_difference_points'],13.3)
        self.assertEqual(result['paper_reported_semantic_similarity'],.65)

    def test_similarity_cache_checks_rows_hashes_and_derived_means(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            path=Path(directory)
            (path/'outcomes.csv').write_text('fixed outcomes')
            run={'path':directory,'rows':[{'row':1,'source_id':7,'result_type':'Successful'}]}
            values=[{'row':1,'source_id':7,'cosine':.8,'angular_unscaled_1_minus_acos':1-math.acos(.8)}]
            self.r.write_table(path/'per_example_similarity.csv',values)
            record={'outcomes_sha256':self.r.sha(path/'outcomes.csv'),
                    'model_url':'https://tfhub.dev/google/universal-sentence-encoder-large/3',
                    'per_example_sha256':self.r.sha(path/'per_example_similarity.csv'),
                    'count':1,'mean_cosine':.8,'mean_angular_unscaled_1_minus_acos':1-math.acos(.8)}
            self.assertEqual(self.r.validate_similarity(run,record),record)
            with self.assertRaises(ValueError):self.r.validate_similarity(run,dict(record,mean_cosine=.2))
            values[0]['source_id']=8
            self.r.write_table(path/'per_example_similarity.csv',values)
            with self.assertRaises(ValueError):self.r.validate_similarity(run,record)
            record['per_example_sha256']=self.r.sha(path/'per_example_similarity.csv')
            with self.assertRaises(ValueError):self.r.validate_similarity(run,record)

    def test_semantic_ablation_uses_only_the_same_successful_inputs(self):
        full,ablated=self.pair()
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            for run,values,name in [(full,[(1,2,.9)],'full'),(ablated,[(0,1,.1),(1,2,.7)],'ablated')]:
                path=Path(directory)/name;path.mkdir();run['path']=str(path)
                (path/'outcomes.csv').write_text(name)
                rows=[{'row':row,'source_id':source,'cosine':value,'angular_unscaled_1_minus_acos':1-math.acos(value)} for row,source,value in values]
                self.r.write_table(path/'per_example_similarity.csv',rows)
                self.r.save(path/'similarity.json',{'outcomes_sha256':self.r.sha(path/'outcomes.csv'),
                    'model_url':'https://tfhub.dev/google/universal-sentence-encoder-large/3',
                    'per_example_sha256':self.r.sha(path/'per_example_similarity.csv'),'count':len(rows),
                    'mean_cosine':sum(row['cosine'] for row in rows)/len(rows),
                    'mean_angular_unscaled_1_minus_acos':sum(row['angular_unscaled_1_minus_acos'] for row in rows)/len(rows)})
            result=self.r.compare_pair(full,ablated)
            self.assertEqual(result['semantic_common_successes'],1)
            self.assertAlmostEqual(result['semantic_mean_cosine_full_common'],.9)
            self.assertAlmostEqual(result['semantic_mean_cosine_no_sentence_common'],.7)
            self.assertAlmostEqual(result['semantic_mean_paired_cosine_change_common'],-.2)


if __name__=='__main__':unittest.main()
