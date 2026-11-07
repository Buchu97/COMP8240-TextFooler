"""Independently reconcile completed runs with frozen inputs and source snapshots."""
import argparse
import math
from pathlib import Path
from replication_runs import read,save,sha,check_existing
from replication_report import verified_run,validate_similarity


def validate_execution_statistics(stats,rows):
    if stats.get('status')=='unavailable_after_interruption':
        if set(stats)!= {'status','reason'} or not isinstance(stats['reason'],str) or not stats['reason']:
            raise ValueError('Unavailable physical counters must have a reason and no invented counts')
        return 'unavailable_after_interruption'
    fields=('start_row','logical_candidate_queries','batch_requests','requested_examples',
            'forward_batches','forward_examples','cache_hits')
    if any(type(stats.get(field)) is not int or stats[field]<0 for field in fields):
        raise ValueError('Execution counters must be nonnegative integers')
    if (stats.get('status')!='available' or not 0<=stats['start_row']<=len(rows) or
        stats['batch_requests']!=stats['forward_batches']+stats['cache_hits'] or
        stats['requested_examples']<stats['forward_examples'] or stats.get('clean_baseline_included') is not False or
        stats['logical_candidate_queries']!=sum(row['num_queries'] for row in rows[stats['start_row']:])):
        raise ValueError('Execution statistics do not reconcile with the saved attack segment')
    return 'available'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Verification output already exists')
    records=[];excluded=[]
    for summary_path in sorted(args.root.glob('*/summary.json')):
        path=summary_path.parent
        if (path/'validation_issue.json').exists():continue
        result=verified_run(path)
        metadata=result['metadata']
        if not metadata['model_alias'].startswith('original-'):
            excluded.append({'run':path.name,'reason':'Auxiliary maintained-checkpoint comparison; outside the original-release matrix audit'})
            continue
        sample_dir=args.samples/metadata['dataset']
        manifest=read(sample_dir/'manifest.json')
        samples=read(sample_dir/'evaluation.json')
        assert sha(sample_dir/'evaluation.json')==manifest['evaluation_sha256']==metadata['evaluation_sha256']
        assert sha(sample_dir/'manifest.json')==metadata['sample_manifest_sha256']
        assert sha(sample_dir/'released_source.txt')==manifest['source_sha256']
        assert sha(sample_dir/'evaluation.csv')==manifest['display_csv_sha256']
        check_existing(result['rows'],samples)
        baseline=read(path/'baseline.json')
        assert len(baseline)==len(samples)
        for saved,row,sample in zip(baseline,result['rows'],samples):
            assert (saved['row'],saved['source_id'],saved['label'])==(sample['row'],sample['source_id'],sample['label'])
            assert saved['prediction']==row['original_prediction']
            probabilities=saved['probabilities']
            assert len(probabilities)==len({item['label'] for item in samples})
            assert all(math.isfinite(value) and 0<=value<=1 for value in probabilities)
            assert math.isclose(sum(probabilities),1,abs_tol=1e-5)
            assert saved['prediction']==max(range(len(probabilities)),key=lambda index:probabilities[index])
        for field,filename in [('runner_sha256','runner_snapshot.py'),('ablation_sha256','ablation_snapshot.py'),
                               ('recipe_sha256','recipe_snapshot.py'),('model_loader_sha256','model_loader_snapshot.py')]:
            assert sha(path/filename)==metadata[field],(path.name,field)
        execution_status='not_recorded_for_this_runtime'
        if metadata.get('streaming_word_ranking'):
            assert sha(path/'streaming_word_ranking_snapshot.py')==metadata['streaming_word_ranking']['sha256']
            if 'fast_text_edits_sha256' in metadata['streaming_word_ranking']:
                assert sha(path/'fast_text_edits_snapshot.py')==metadata['streaming_word_ranking']['fast_text_edits_sha256']
            if 'fast_bert_tokenizer_sha256' in metadata['streaming_word_ranking']:
                assert sha(path/'fast_bert_tokenizer_snapshot.py')==metadata['streaming_word_ranking']['fast_bert_tokenizer_sha256']
            if 'encoded_batch_cache_sha256' in metadata['streaming_word_ranking']:
                assert sha(path/'encoded_batch_cache_snapshot.py')==metadata['streaming_word_ranking']['encoded_batch_cache_sha256']
                stats=read(summary_path)['model_execution_statistics']
                execution_status=validate_execution_statistics(stats,result['rows'])
        if (path/'similarity.json').exists():validate_similarity(result,read(path/'similarity.json'))
        records.append({'run':path.name,'examples':len(samples),'summary_sha256':sha(summary_path),
                        'similarity_verified':(path/'similarity.json').exists(),'physical_execution_statistics':execution_status})
    save(args.output,{'status':'passed','verified_complete_runs':len(records),'runs':records,'excluded':excluded,
        'verification_script_sha256':sha(__file__),
        'checks':['Frozen source/evaluation/display hashes','Every input, source ID and label','All clean predictions and probabilities',
                  'Recounted outcome metrics and summary hashes','Run-specific source snapshots','Similarity CSV membership, hashes and aggregates when present'],
        'scope':'Completed artifacts only; unfinished configurations remain in suite_status.json'})
    print(f'Verified {len(records)} complete runs against frozen source rows and saved artifacts')


if __name__=='__main__':main()
