from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


class ReplicationSuiteTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'scripts/run_replication_suite.py').exists(),'Suite runner is missing')
        import run_replication_suite
        self.s=run_replication_suite

    def test_suite_covers_released_classification_matrix_and_paired_ablations(self):
        jobs=self.s.build_jobs()
        classification=[j for j in jobs if j['dataset'] in ('mr','imdb','yelp','ag','fake') and j['variant']=='full']
        self.assertEqual(len(classification),15)
        self.assertEqual({j['model'].split('-')[1] for j in classification},{'cnn','lstm','bert'})
        self.assertEqual(len({j['id'] for j in jobs}),20)
        self.assertEqual({j['dataset'] for j in jobs if j['variant']=='no_sentence_similarity'},{'mr','snli'})

    def test_missing_or_partial_model_assets_are_not_scheduled(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            root=Path(directory)
            job={'model':'original-bert-mr'}
            self.assertFalse(self.s.ready_model(root,job))
            folder=root/'models/original_textfooler/bert_mr'
            folder.mkdir(parents=True)
            (folder/'pytorch_model.bin.part').write_bytes(b'incomplete')
            self.assertFalse(self.s.ready_model(root,job))

    def suite_case(self,directory,child=True,sample_hash='sample'):
        root=Path(directory)
        samples=root/'samples/mr';samples.mkdir(parents=True)
        (samples/'evaluation.json').write_text('fixed population')
        from replication_runs import sha
        expected=sha(samples/'evaluation.json')
        self.s.save(samples/'manifest.json',{'evaluation_sha256':expected})
        job={'id':'j','model':'original-bert-mr','dataset':'mr','variant':'full'}
        output=root/'out';output.mkdir()
        folder=output/'j'
        process=Mock()
        def poll():
            self.s.save(folder/'summary.json',{})
            return None if process.poll.call_count==1 else 0
        process.poll.side_effect=poll
        if not child:self.s.save(folder/'summary.json',{})
        metadata={'model_alias':job['model'],'dataset':'mr','variant':'full',
                  'evaluation_sha256':expected if sample_hash=='sample' else sample_hash}
        arguments=['suite','--samples',str(root/'samples'),'--output-root',str(output),'--poll-seconds','1']
        with patch.object(sys,'argv',arguments),patch.object(self.s,'build_jobs',return_value=[job]),\
             patch.object(self.s,'ready_model',return_value=True),\
             patch.object(self.s,'verified_run',return_value={'metadata':metadata}),\
             patch.object(self.s.subprocess,'Popen',return_value=process),patch.object(self.s.time,'sleep'):
            with self.assertRaises(SystemExit) as stopped:self.s.main()
        return stopped.exception.code,process,self.s.read(output/'suite_status.json')

    def test_summary_publication_does_not_bypass_child_exit(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            code,process,status=self.suite_case(directory)
            self.assertEqual(code,0)
            self.assertGreaterEqual(process.poll.call_count,2)
            self.assertEqual(status['active'],[])
            self.assertEqual(status['completed'],['j'])

    def test_reuse_refuses_a_different_requested_population(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            code,process,status=self.suite_case(directory,child=False,sample_hash='other population')
            self.assertEqual(code,1)
            self.assertEqual(status['completed'],[])
            self.assertIn('j',status['failed'])


if __name__=='__main__':unittest.main()
