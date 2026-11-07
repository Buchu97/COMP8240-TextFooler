"""Verify original-data runs and compare matched sentence-constraint ablations."""
import argparse
import csv
import json
import math
from pathlib import Path

from replication_runs import check_existing,metrics,read,save,sha,validate_outcome


def verified_run(path,expected=1000):
    path=Path(path)
    if (path/'validation_issue.json').exists():
        raise ValueError('Superseded validation trial')
    summary,metadata=read(path/'summary.json'),read(path/'metadata.json')
    if summary['status']!='completed' or summary['metadata_sha256']!=sha(path/'metadata.json') or summary['outcomes_sha256']!=sha(path/'outcomes.csv'):
        raise ValueError('Incomplete or changed run artifacts')
    with (path/'outcomes.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    for index,row in enumerate(rows):
        for field in ('row','source_id','label','original_prediction','perturbed_prediction','num_queries','original_word_count','changed_word_count'):
            row[field]=int(row[field])
        if row['row']!=index:
            raise ValueError('Outcome rows are not sequential')
        validate_outcome(row)
    if len(rows)!=expected or len({r['source_id'] for r in rows})!=len(rows):
        raise ValueError('Run population differs from the planned unique sample')
    actual=metrics(rows)
    for field,value in actual.items():
        if field=='by_category':continue
        recorded=summary['metrics'][field]
        if value is None and recorded is None:continue
        if value is None or recorded is None or not math.isclose(value,recorded,abs_tol=1e-9):
            raise ValueError('Summary metric differs from independently counted outcomes')
    for field,filename in [('baseline_sha256','baseline.json')]:
        if field in summary and summary[field]!=sha(path/filename):
            raise ValueError('Saved clean baseline changed')
    successful=[r for r in rows if r['result_type']=='Successful']
    actual['mean_perturbed_percent_whitespace_successful']=(sum(100*r['changed_word_count']/len((r.get('original_hypothesis') or r['original_text']).split()) for r in successful)/len(successful) if successful else None)
    return {'path':str(path.resolve()),'metadata':metadata,'rows':rows,'metrics':actual}


def compare_pair(full,ablated):
    left,right=full['metadata'],ablated['metadata']
    if left['variant']!='full' or right['variant']!='no_sentence_similarity':
        raise ValueError('Expected full versus sentence-similarity ablation')
    for field in ('dataset','model_alias','evaluation_sha256','model_state_sha256','seed',
                  'tokenizer_max_length','recipe_sha256','ablation_sha256',
                  'runner_sha256','model_loader_sha256','packages','sample_manifest_sha256',
                  'input_columns','model_label_names','device','gpu','streaming_word_ranking'):
        if left.get(field)!=right.get(field):
            raise ValueError(f'Paired experiments differ in {field}')
    if len(full['rows'])!=len(ablated['rows']):
        raise ValueError('Ablation populations differ')
    eligible=[]
    for a,b in zip(full['rows'],ablated['rows']):
        for field in ('row','source_id','label','original_text','original_prediction'):
            if a[field]!=b[field]:raise ValueError('Ablation rows or clean predictions differ')
        validate_outcome(a);validate_outcome(b)
        if a['original_prediction']==a['label']:eligible.append((a,b))
    n=len(eligible)
    both=sum(a['result_type']==b['result_type']=='Successful' for a,b in eligible)
    left_only=sum(a['result_type']=='Successful' and b['result_type']!='Successful' for a,b in eligible)
    right_only=sum(b['result_type']=='Successful' and a['result_type']!='Successful' for a,b in eligible)
    discordant=left_only+right_only
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(left_only,right_only)+1))/2**discordant) if discordant else 1.
    difference=100*(right_only-left_only)/n if n else None
    interval=None
    if n:
        import numpy as np
        values=np.array([int(b['result_type']=='Successful')-int(a['result_type']=='Successful') for a,b in eligible])
        rng=np.random.default_rng(8240)
        bootstrap=[100*float(rng.choice(values,size=n,replace=True).mean()) for _ in range(2000)]
        interval=np.percentile(bootstrap,[2.5,97.5]).tolist()
    result={'dataset':left['dataset'],'model':left['model_alias'],'examples':len(full['rows']),
            'eligible':n,'both_successful':both,'full_only_success':left_only,
            'no_sentence_only_success':right_only,'neither_successful':n-both-left_only-right_only,
            'asr_change_points':difference,'paired_bootstrap_95_percent_ci_points':interval,
            'mcnemar_exact_p':p,'mean_query_change_eligible':sum(b['num_queries']-a['num_queries'] for a,b in eligible)/n if n else None,
            'outcome_scope':'Computational attack outcomes; semantic validity evaluated separately'}
    if all('path' in run and (Path(run['path'])/'similarity.json').exists() for run in (full,ablated)):
        scores=[]
        for run in (full,ablated):
            path=Path(run['path'])
            validate_similarity(run,read(path/'similarity.json'))
            with (path/'per_example_similarity.csv').open(encoding='utf-8',newline='') as stream:
                scores.append({int(row['row']):float(row['cosine']) for row in csv.DictReader(stream)})
        common=sorted(set(scores[0])&set(scores[1]))
        changes=[scores[1][row]-scores[0][row] for row in common]
        result.update(semantic_common_successes=len(common),
            semantic_mean_cosine_full_common=sum(scores[0][row] for row in common)/len(common) if common else None,
            semantic_mean_cosine_no_sentence_common=sum(scores[1][row] for row in common)/len(common) if common else None,
            semantic_mean_paired_cosine_change_common=sum(changes)/len(common) if common else None,
            semantic_population='Inputs successfully attacked by both variants; automatic USE-Large cosine, not verified semantic validity')
    return result


def validate_similarity(run,result):
    path=Path(run['path'])
    csv_path=path/'per_example_similarity.csv'
    if (result['outcomes_sha256']!=sha(path/'outcomes.csv') or
        result['per_example_sha256']!=sha(csv_path) or
        result['model_url']!='https://tfhub.dev/google/universal-sentence-encoder-large/3'):
        raise ValueError('Similarity artifacts or encoder differ from the recorded inputs')
    with csv_path.open(encoding='utf-8',newline='') as stream:values=list(csv.DictReader(stream))
    expected=[(row['row'],row['source_id']) for row in run['rows'] if row['result_type']=='Successful']
    if [(int(row['row']),int(row['source_id'])) for row in values]!=expected or result['count']!=len(values):
        raise ValueError('Similarity rows differ from the successful attack population')
    for row in values:
        value=float(row['cosine']);angular=float(row['angular_unscaled_1_minus_acos'])
        if not math.isfinite(value) or not -1<=value<=1 or not math.isclose(angular,1-math.acos(value),abs_tol=1e-9):
            raise ValueError('Invalid cosine or angular similarity value')
    for source,target in [('cosine','mean_cosine'),('angular_unscaled_1_minus_acos','mean_angular_unscaled_1_minus_acos')]:
        mean=sum(float(row[source]) for row in values)/len(values) if values else None
        if (mean is None and result[target] is not None) or (mean is not None and
            (result[target] is None or not math.isclose(mean,result[target],abs_tol=1e-9))):
            raise ValueError('Similarity aggregate differs from its per-example values')
    return result


def semantic_scores(run,encoder):
    path=Path(run['path'])
    target=path/'similarity.json'
    if target.exists():
        return validate_similarity(run,read(target))
    import numpy as np
    import tensorflow as tf
    from analyze_core_replication import cosine_rows
    values=[]
    successful=[row for row in run['rows'] if row['result_type']=='Successful']
    for start in range(0,len(successful),32):
        batch=successful[start:start+32]
        original=[r.get('original_hypothesis') or r['original_text'] for r in batch]
        changed=[r.get('perturbed_hypothesis') or r['perturbed_text'] for r in batch]
        a=np.asarray(encoder(text=tf.constant(original))['default'])
        b=np.asarray(encoder(text=tf.constant(changed))['default'])
        for row,value in zip(batch,cosine_rows(a,b).tolist()):
            if not math.isfinite(value):raise ValueError('Sentence encoder produced a non-finite similarity')
            value=max(-1.,min(1.,value))
            values.append({'row':row['row'],'source_id':row['source_id'],'cosine':value,
                           'angular_unscaled_1_minus_acos':1-math.acos(max(-1,min(1,value)))})
    temporary=path/'per_example_similarity.csv.tmp'
    with temporary.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['row','source_id','cosine','angular_unscaled_1_minus_acos'])
        writer.writeheader();writer.writerows(values)
    temporary.replace(path/'per_example_similarity.csv')
    result={'outcomes_sha256':sha(path/'outcomes.csv'),'model':'Universal Sentence Encoder Large v3',
            'model_url':'https://tfhub.dev/google/universal-sentence-encoder-large/3',
            'population':'Successful attacks only; hypothesis-only for NLI, full input for classification',
            'count':len(values),'mean_cosine':sum(r['cosine'] for r in values)/len(values) if values else None,
            'mean_angular_unscaled_1_minus_acos':sum(r['angular_unscaled_1_minus_acos'] for r in values)/len(values) if values else None,
            'per_example_sha256':sha(path/'per_example_similarity.csv'),
            'interpretation':'Automatic similarity, not verification of label or meaning preservation'}
    save(target,result)
    return validate_similarity(run,result)


def write_table(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or ['status']);writer.writeheader();writer.writerows(rows)


def paper_comparison(run,reference):
    meta,values=run['metadata'],run['metrics']
    alias=meta['model_alias']
    family=alias.split('-')[1] if alias.startswith('original-') else 'bert' if alias.startswith('bert') else alias.split('-')[0]
    dataset=meta['dataset']
    if dataset in ('snli','mnli_m','mnli_mm'):
        source=reference['bert_entailment'];index=source['datasets'].index(dataset)
    else:
        source=reference['classification']
        index=source['model_order'].index(family)*len(source['dataset_order'])+source['dataset_order'].index(dataset)
    row={'dataset':dataset,'model':alias,'paper_table':source['table'],
         'comparison_role':'primary_original_release' if alias.startswith('original-') else 'auxiliary_maintained_checkpoint',
         'paper_clean_accuracy_percent':source['original_accuracy_percent'][index],
         'replicated_clean_accuracy_percent':values['original_accuracy_percent'],
         'clean_difference_points':values['original_accuracy_percent']-source['original_accuracy_percent'][index],
         'paper_after_attack_accuracy_percent':source['after_attack_accuracy_percent'][index],
         'replicated_after_attack_accuracy_percent':values['after_attack_accuracy_percent'],
         'after_attack_difference_points':values['after_attack_accuracy_percent']-source['after_attack_accuracy_percent'][index],
         'paper_perturbed_words_percent':source['perturbed_words_percent'][index],
         'replicated_perturbed_percent_whitespace':values['mean_perturbed_percent_whitespace_successful'],
         'paper_queries':source['queries'][index],'replicated_mean_queries_eligible':values['mean_queries_eligible'],
         'paper_reported_semantic_similarity':source['semantic_similarity'][index]}
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    parser.add_argument('--similarity',action='store_true')
    args=parser.parse_args()
    if args.report_dir.exists():raise FileExistsError('Report directory already exists')
    runs=[];excluded=[]
    for path in sorted(args.root.glob('*/summary.json')):
        try:runs.append(verified_run(path.parent))
        except (ValueError,KeyError,OSError) as error:excluded.append({'path':str(path.parent),'reason':str(error)})
    encoder=None
    if args.similarity:
        import tensorflow_hub as hub
        encoder=hub.load(str(Path(__file__).resolve().parents[1]/'models/use-large-v3/model')).signatures['default']
    table=[];paper=[]
    reference=read(Path(__file__).resolve().parents[1]/'notes/textfooler_paper_reference.json')
    for run in runs:
        row={key:run['metadata'][key] for key in ('dataset','model_alias','variant')}
        row['comparison_role']='primary_original_release' if run['metadata']['model_alias'].startswith('original-') else 'auxiliary_maintained_checkpoint'
        row.update({key:value for key,value in run['metrics'].items() if key!='by_category'})
        execution=read(Path(run['path'])/'summary.json').get('model_execution_statistics')
        if execution:
            row['physical_counter_status']=execution['status']
            if execution['status']=='available':
                row.update(physical_forward_batches=execution['forward_batches'],
                    physical_forward_examples=execution['forward_examples'],
                    encoded_batch_requests=execution['batch_requests'],
                    encoded_batch_cache_hits=execution['cache_hits'],
                    physical_counter_start_row=execution['start_row'])
        if encoder is not None:
            scores=semantic_scores(run,encoder)
            row.update({key:scores[key] for key in ('mean_cosine','mean_angular_unscaled_1_minus_acos')})
        if run['metadata']['variant']=='full':
            comparison=paper_comparison(run,reference)
            if encoder is not None:
                comparison.update(replication_use_large_cosine=scores['mean_cosine'],
                                  replication_angular_unscaled=scores['mean_angular_unscaled_1_minus_acos'])
            paper.append(comparison)
        row['output']=run['path'];table.append(row)
    comparisons=[]
    for full in runs:
        if full['metadata']['variant']!='full':continue
        matches=[r for r in runs if r['metadata']['variant']=='no_sentence_similarity' and
                 r['metadata']['model_alias']==full['metadata']['model_alias'] and r['metadata']['dataset']==full['metadata']['dataset']]
        for ablated in matches:comparisons.append(compare_pair(full,ablated))
    args.report_dir.mkdir(parents=True)
    write_table(args.report_dir/'replication_summary.csv',table)
    write_table(args.report_dir/'ablation_comparison.csv',comparisons)
    write_table(args.report_dir/'paper_comparison.csv',paper)
    save(args.report_dir/'report.json',{'completed_runs':len(runs),'runs':table,'paired_ablations':comparisons,
        'paper_comparison':paper,'paper_reference':reference,'excluded':excluded,
        'primary_original_release_baselines':sum(row['comparison_role']=='primary_original_release' and row['variant']=='full' for row in table),
        'primary_ablation_runs':sum(row['comparison_role']=='primary_original_release' and row['variant']=='no_sentence_similarity' for row in table),
        'report_script_sha256':sha(__file__),
        'metric_notes':{'word_change':'TextAttack word counts and whitespace-token counts reported separately; whitespace counts follow the authors token-list denominator.',
                        'queries':'Reported query means are logical TextAttack candidate evaluations over initially correct examples, excluding the separate clean baseline. Exact encoded-batch reuse reduces physical forward passes without changing these logical counts; available execution counters are separately scoped to their attack segment.',
                        'similarity':'Reported paper score and explicitly defined post-hoc USE-Large cosine/angular scores are separate columns; equality of evaluation procedures is not assumed.',
                        'incomplete_runs':'Only independently verified complete runs enter tables. The suite status records pending and failed configurations.'}})
    print(json.dumps({'completed_runs':len(runs),'paired_ablations':len(comparisons),'excluded':excluded},indent=2))


if __name__=='__main__':main()
