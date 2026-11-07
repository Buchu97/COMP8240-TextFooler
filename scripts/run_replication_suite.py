"""Run the released-checkpoint matrix and paired sentence-similarity ablations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from replication_runs import read,save,sha
from replication_report import verified_run


def build_jobs():
    jobs=[]
    for dataset in ('mr','imdb','yelp','ag','fake'):
        for family in ('cnn','lstm','bert'):
            jobs.append({'id':f'original_{family}_{dataset}_full','dataset':dataset,
                         'model':f'original-{family}-{dataset}','variant':'full','stream_word_ranking':dataset=='fake'})
    for dataset in ('snli','mnli_m','mnli_mm'):
        jobs.append({'id':f'original_bert_{dataset}_full','dataset':dataset,
                     'model':'original-bert-'+('snli' if dataset=='snli' else 'mnli'),'variant':'full'})
    for dataset in ('mr','snli'):
        jobs.append({'id':f'original_bert_{dataset}_no_sentence_similarity','dataset':dataset,
                     'model':f'original-bert-{dataset}','variant':'no_sentence_similarity'})
    priority=['original_bert_mr_full','original_bert_snli_full','original_bert_mr_no_sentence_similarity',
              'original_bert_snli_no_sentence_similarity','original_bert_mnli_m_full',
              'original_bert_mnli_mm_full','original_bert_fake_full','original_cnn_imdb_full']
    return sorted(jobs,key=lambda job:priority.index(job['id']) if job['id'] in priority else len(priority))


def ready_model(root,job):
    _,family,dataset=job['model'].split('-')
    folder=Path(root)/'models/original_textfooler'/f'{family}_{dataset}'
    files=['pytorch_model.bin']+(['bert_config.json','vocab.txt'] if family=='bert' else [])
    for filename in files:
        path=folder/filename;record=path.with_suffix(path.suffix+'.source.json')
        if not path.is_file() or not record.is_file():return False
        if path.stat().st_size!=read(record)['bytes']:return False
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    parser.add_argument('--poll-seconds',type=int,default=15)
    args=parser.parse_args()
    from filelock import FileLock
    root=Path(__file__).resolve().parents[1]
    args.output_root.mkdir(parents=True,exist_ok=True)
    logs=args.output_root/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    jobs=build_jobs();active={};done=set();failed={};last=None
    with FileLock(str(args.output_root/'suite.lock'),timeout=0):
        while len(done)+len(failed)<len(jobs):
            for identity,(process,stream) in list(active.items()):
                code=process.poll()
                if code is None:continue
                stream.close();del active[identity]
                if code:
                    done.discard(identity)
                    failed[identity]={'exit_code':code,'log':str(logs/(identity+'.log'))}
                elif not (args.output_root/identity/'summary.json').exists():
                    failed[identity]={'validation_error':'Worker exited without a completed summary'}
            for job in jobs:
                if job['id'] in done or job['id'] in failed or job['id'] in active:continue
                folder=args.output_root/job['id']
                if (folder/'summary.json').exists():
                    try:
                        result=verified_run(folder)
                        if (result['metadata']['model_alias']!=job['model'] or
                            result['metadata']['dataset']!=job['dataset'] or result['metadata']['variant']!=job['variant'] or
                            bool(result['metadata'].get('streaming_word_ranking'))!=bool(job.get('stream_word_ranking'))):
                            raise ValueError('Completed configuration differs from the suite')
                        sample_dir=args.samples/job['dataset']
                        manifest=read(sample_dir/'manifest.json')
                        if (sha(sample_dir/'evaluation.json')!=manifest['evaluation_sha256'] or
                            result['metadata']['evaluation_sha256']!=manifest['evaluation_sha256']):
                            raise ValueError('Completed population differs from the requested fixed sample')
                        done.add(job['id'])
                    except (ValueError,OSError,KeyError) as error:
                        failed[job['id']]={'validation_error':str(error)}
            external=[p for p in args.output_root.glob('*/run.lock') if p.parent.name not in active
                      and not (p.parent/'summary.json').exists() and not (p.parent/'validation_issue.json').exists()]
            slots=max(0,args.workers-len(active)-len(external))
            for job in jobs:
                if not slots:break
                if job['id'] in done or job['id'] in failed or job['id'] in active:continue
                fake_word_jobs={'original_cnn_fake_full','original_lstm_fake_full'}
                if (job['id'] in fake_word_jobs and
                    fake_word_jobs.intersection(set(active)|{path.parent.name for path in external})):
                    continue
                folder=args.output_root/job['id']
                if (folder/'run.lock').exists() or not ready_model(root,job):continue
                command=[sys.executable,'-X','utf8','-u',str(root/'scripts/replication_runs.py'),'run',
                         '--sample-dir',str(args.samples/job['dataset']),'--output-dir',str(folder),
                         '--model',job['model'],'--variant',job['variant']]
                if job.get('stream_word_ranking'):command.append('--stream-word-ranking')
                stream=(logs/(job['id']+'.log')).open('a',encoding='utf-8')
                process=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active[job['id']]=(process,stream);slots-=1
                print('Starting '+job['id'],flush=True)
            status={'planned':len(jobs),'completed':sorted(done),'active':list(active),
                    'external_active':[p.parent.name for p in external],'failed':failed,
                    'waiting_for_assets':[j['id'] for j in jobs if j['id'] not in done and j['id'] not in failed and not ready_model(root,j)]}
            save(args.output_root/'suite_status.json',status)
            brief=f'{len(done)}/{len(jobs)} complete | active {list(active)} | external {status["external_active"]} | failed {len(failed)}'
            if brief!=last:print(brief,flush=True);last=brief
            if len(done)+len(failed)<len(jobs):time.sleep(max(1,args.poll_seconds))
    raise SystemExit(1 if failed else 0)


if __name__=='__main__':main()
