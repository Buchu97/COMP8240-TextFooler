"""Fixed-sample original-data replication with resumable per-example outcomes."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'mr': ('bert-base-uncased-mr', None), 'imdb': ('cnn-imdb', None),
            'yelp': ('cnn-yelp', None), 'ag': ('cnn-ag-news', None),
            'snli': ('bert-base-uncased-snli', 'test'),
            'mnli_m': ('bert-base-uncased-mnli', 'validation_matched'),
            'mnli_mm': ('bert-base-uncased-mnli', 'validation_mismatched')}
RELEASED_NAMES = {'mr':'mr','imdb':'imdb','yelp':'yelp','ag':'ag','fake':'fake',
                  'snli':'snli','mnli_m':'mnli_matched','mnli_mm':'mnli_mismatched'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def configure():
    for name, folder in {'TA_CACHE_DIR':'textattack', 'HF_HOME':'huggingface', 'TORCH_HOME':'torch',
                         'NLTK_DATA':'nltk', 'TFHUB_CACHE_DIR':'tensorflow_hub'}.items():
        os.environ[name] = str(ROOT / 'data/cache' / folder)
    os.environ.update(TF_CPP_MIN_LOG_LEVEL='3', HF_HUB_DISABLE_XET='1', PYTHONUTF8='1')


def select_rows(source, columns, label_map, count, seed):
    labels = source['label'] if not isinstance(source, list) else [r['label'] for r in source]
    if any(int(label) not in label_map and int(label) != -1 for label in labels):
        raise ValueError('Unexpected source label')
    eligible = [i for i, label in enumerate(labels) if int(label) in label_map]
    if not 1 <= count <= len(eligible):
        raise ValueError('Requested sample exceeds labelled source rows')
    rows = []
    for row_number, index in enumerate(random.Random(seed).sample(eligible, count)):
        raw = source[index]
        inputs = {column: raw[column] for column in columns}
        if any(not isinstance(text, str) or not text.strip() for text in inputs.values()):
            raise ValueError('Empty or invalid original input')
        rows.append({'row':row_number, 'source_id':index, 'inputs':inputs,
                     'source_label':int(raw['label']), 'label':label_map[int(raw['label'])]})
    return rows


def read_released(path, is_nli):
    rows=[]
    names={'0':'contradiction','1':'entailment','2':'neutral'} if is_nli else {}
    for index,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        if is_nli:
            label,premise,hypothesis=line.split('\t')[:3]
            if label=='-':
                continue
            label={'contradiction':0,'entailment':1,'neutral':2}[label]
            inputs={'premise':' '.join(premise.split()),'hypothesis':' '.join(hypothesis.split())}
        else:
            label, text=line.split(' ',1)
            label=int(label)
            text=re.sub(r"[^A-Za-z0-9(),!?\'\`]",' ',text)
            for pattern,replacement in [(r"\'s"," 's"),(r"\'ve"," 've"),(r"n\'t"," n't"),
                                        (r"\'re"," 're"),(r"\'d"," 'd"),(r"\'ll"," 'll"),
                                        (r',',' , '),(r'!',' ! '),(r'\(',r' \( '),
                                        (r'\)',r' \) '),(r'\?',r' \? ')]:
                text=re.sub(pattern,replacement,text)
            inputs={'text':' '.join(text.strip().lower().split())}
        if any(not text for text in inputs.values()):
            raise ValueError('Empty released input')
        rows.append({'row':len(rows),'source_id':index,'inputs':inputs,'label':label})
    return rows,names


def validate_configuration(dataset, model, variant):
    allowed = ([f'{family}-{dataset}' for family in ('cnn','lstm')] +
               [f'bert-base-uncased-{dataset}']) if dataset in ('mr','imdb','yelp','ag') else []
    if dataset=='snli':
        allowed=['bert-base-uncased-snli']
    if dataset in ('mnli_m','mnli_mm'):
        allowed=['bert-base-uncased-mnli']
    if dataset in ('mr','imdb','yelp','ag','fake'):
        allowed += [f'original-{family}-{dataset}' for family in ('cnn','lstm','bert')]
    elif dataset=='snli':
        allowed += ['original-bert-snli']
    elif dataset in ('mnli_m','mnli_mm'):
        allowed += ['original-bert-mnli']
    if model not in allowed or variant not in ('full','no_sentence_similarity'):
        raise ValueError('Model/dataset or attack variant mismatch')


def prepare_sample(dataset, output, count=1000, seed=8240, source_name='released'):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Sample folder already exists')
    if source_name=='released':
        source=ROOT/'repo/TextFooler-official/data'/RELEASED_NAMES[dataset]
        rows,names=read_released(source,dataset in ('snli','mnli_m','mnli_mm'))
        if not 1<=count<=len(rows):
            raise ValueError('Requested sample exceeds released rows')
        rows=rows[:count]
        if dataset in ('mr','imdb','yelp'):
            names={'0':'negative','1':'positive'}
        elif dataset=='ag':
            names={'0':'World','1':'Sports','2':'Business','3':'Sci/Tech'}
        commit=subprocess.check_output(['git','-C',str(source.parents[1]),'rev-parse','HEAD'],text=True).strip()
        output.mkdir(parents=True)
        shutil.copyfile(source,output/'released_source.txt')
        save(output/'evaluation.json',rows)
        columns=list(rows[0]['inputs'])
        with (output/'evaluation.csv').open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['row','source_id']+columns+['label'])
            writer.writeheader()
            writer.writerows({'row':r['row'],'source_id':r['source_id'],**r['inputs'],'label':r['label']} for r in rows)
        manifest={'dataset':dataset,'source_kind':'authors_released_attack_examples',
                  'source_url':f'https://github.com/jind11/TextFooler/blob/{commit}/data/{source.name}',
                  'source_revision':commit,'source_sha256':sha(source),'source_rows':len(rows),
                  'reader_reference':'jind11/TextFooler dataloader.read_corpus defaults; attack_nli.read_data(target_model=bert)',
                  'input_columns':columns,'model_label_names':names,'seed':seed,'examples':len(rows),
                  'selection':'Released file order; no resampling or correctness filtering',
                  'evaluation_sha256':sha(output/'evaluation.json'),'display_csv_sha256':sha(output/'evaluation.csv')}
        save(output/'manifest.json',manifest)
        print(json.dumps({'dataset':dataset,'source':'authors released examples','examples':len(rows)}),flush=True)
        return manifest
    configure()
    from textattack.dataset_args import DatasetArgs
    alias, split = DATASETS[dataset]
    args = DatasetArgs(dataset_by_model=alias, dataset_split=split)
    source = DatasetArgs._create_dataset_from_args(args)
    raw = source._dataset
    source_names = raw.features[source.output_column].names
    label_map = source.label_map or {i:i for i in range(len(source_names))}
    label_names = {str(mapped):source_names[original] for original, mapped in label_map.items()}
    rows = select_rows(raw, source.input_columns, label_map, count, seed)
    output.mkdir(parents=True)
    save(output / 'evaluation.json', rows)
    fields = ['row','source_id'] + list(source.input_columns) + ['label']
    with (output / 'evaluation.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({'row':r['row'], 'source_id':r['source_id'], **r['inputs'], 'label':r['label']} for r in rows)
    manifest = {'dataset':dataset, 'source_args':args.dataset_from_huggingface, 'source_split':split or 'test',
                'source_fingerprint':raw._fingerprint, 'source_rows':len(raw),
                'unlabelled_source_rows':sum(int(x)==-1 for x in raw['label']),
                'label_map':label_map, 'model_label_names':label_names, 'input_columns':list(source.input_columns),
                'label_map_source':'Installed TextAttack 0.3.10 dataset_args.py and source dataset label names',
                'selection':'Python Random(seed).sample of labelled held-out source rows before model inference',
                'seed':seed, 'examples':len(rows), 'evaluation_sha256':sha(output/'evaluation.json'),
                'display_csv_sha256':sha(output/'evaluation.csv')}
    save(output / 'manifest.json', manifest)
    print(json.dumps({'dataset':dataset, 'examples':len(rows), 'sample':str(output.resolve())}), flush=True)
    return manifest


def validate_outcome(row):
    correct = int(row['original_prediction']) == int(row['label'])
    after = int(row['perturbed_prediction']) == int(row['label'])
    expected = 'Skipped' if not correct else 'Failed' if after else 'Successful'
    if row['result_type'] != expected:
        raise ValueError('Attack outcome and predictions disagree')
    if row.get('original_premise') != row.get('perturbed_premise'):
        raise ValueError('Entailment premise was modified')


def metrics(rows):
    for row in rows:
        validate_outcome(row)
    from run_constructed_pilot import summarise_attacks
    return summarise_attacks([dict(row, category='all') for row in rows])


def state_digest(model):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def check_existing(rows, samples):
    if len(rows) > len(samples):
        raise ValueError('More outcomes than sample rows')
    for index, row in enumerate(rows):
        sample = samples[index]
        original = '\n'.join(sample['inputs'].values())
        if (int(row['row']) != index or int(row['source_id']) != sample['source_id']
                or int(row['label']) != sample['label'] or row['original_text'] != original):
            raise ValueError('Saved outcomes differ from the fixed sample')
        validate_outcome(row)


def clean_predictions(wrapper,samples,num_classes):
    import torch
    baseline=[]
    for sample in samples:
        values=list(sample['inputs'].values())
        inputs=values[0] if len(values)==1 else tuple(values)
        scores=torch.as_tensor(wrapper([inputs]))
        if scores.shape!=(1,num_classes) or not torch.isfinite(scores).all():
            raise ValueError('Classifier output has the wrong class dimension or invalid values')
        probabilities=torch.softmax(scores,dim=-1)[0].cpu().tolist()
        baseline.append({'row':sample['row'],'source_id':sample['source_id'],'label':sample['label'],
                         'prediction':max(range(num_classes),key=lambda index:probabilities[index]),
                         'probabilities':probabilities})
    return baseline


def run_experiment(sample_dir, output, model_alias, variant='full', seed=8240,stream_word_ranking=False):
    configure()
    import torch
    import textattack
    from replication_models import load_model
    from replication_attack import build_attack
    from run_type1 import add_utf8_text_log
    sample_dir, output = Path(sample_dir), Path(output)
    manifest, samples = read(sample_dir/'manifest.json'), read(sample_dir/'evaluation.json')
    validate_configuration(manifest['dataset'],model_alias,variant)
    if sha(sample_dir/'evaluation.json') != manifest['evaluation_sha256']:
        raise ValueError('Fixed sample hash mismatch')
    torch.set_num_threads(8)
    textattack.shared.utils.set_seed(seed)
    wrapper = load_model(ROOT,model_alias)
    compiled_tokenizer=stream_word_ranking and model_alias=='original-bert-fake'
    if compiled_tokenizer:
        from fast_bert_tokenizer import make_fast_tokenizer
        wrapper.tokenizer=make_fast_tokenizer(wrapper.tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wrapper.model.to(device).eval()
    model_hash = state_digest(wrapper.model)
    if compiled_tokenizer:
        from encoded_batch_cache import EncodedBatchCachedWrapper
        wrapper=EncodedBatchCachedWrapper(wrapper,capacity=128)
    attack = build_attack(wrapper, variant)
    if stream_word_ranking:
        from streaming_word_ranking import build_streaming_attack
        attack=build_streaming_attack(attack)
    tokenizer = wrapper.tokenizer
    metadata = {'dataset':manifest['dataset'], 'model_alias':model_alias, 'variant':variant,
                'model_state_sha256':model_hash, 'model_revision':getattr(getattr(wrapper.model,'config',None),'_commit_hash',None),
                'model_release':getattr(wrapper,'release_metadata',None),
                'sample_manifest_sha256':sha(sample_dir/'manifest.json'), 'evaluation_sha256':manifest['evaluation_sha256'],
                'input_columns':manifest['input_columns'], 'model_label_names':manifest['model_label_names'],
                'seed':seed, 'per_row_seed':'seed + saved row number', 'N':50, 'word_delta':.7,
                'tokenizer_max_length':getattr(tokenizer,'model_max_length',getattr(tokenizer,'max_length',None)),
                'recipe_sha256':sha(ROOT/'scripts/textfooler_paper_delta_0_7.py'),
                'ablation_sha256':sha(ROOT/'scripts/replication_attack.py'), 'runner_sha256':sha(Path(__file__)),
                'model_loader_sha256':sha(ROOT/'scripts/replication_models.py'),
                'packages':{name:version(name) for name in ('torch','transformers','textattack','datasets')},
                'device':str(device), 'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                'perturbation_denominator':'hypothesis words only for NLI; complete text words for classification',
                'sentence_constraint':None if variant=='no_sentence_similarity' else {
                    'encoder':'Universal Sentence Encoder v3 (non-large)', 'metric':'angular',
                    'threshold':.840845057, 'window_size':15, 'compare_against_original':False,
                    'skip_text_shorter_than_window':True},
                'semantic_label_validation':'Not performed by the attack; evaluated separately'}
    if stream_word_ranking:
        metadata['streaming_word_ranking']={'sha256':sha(ROOT/'scripts/streaming_word_ranking.py'),
            'fast_text_edits_sha256':sha(ROOT/'scripts/fast_text_edits.py'),
            'batch_size':attack.goal_function.batch_size,'model_and_constraint_cache_entries':256,
            'transformation_cache_entries':1,'ranking_model_cache':False,
            'purpose':'Bound word-ranking construction and caches without truncating or filtering inputs'}
        if compiled_tokenizer:
            metadata['streaming_word_ranking'].update(fast_bert_tokenizer_sha256=sha(ROOT/'scripts/fast_bert_tokenizer.py'),
                tokenizer_backend='BertTokenizerFast converted from the released slow tokenizer',
                tokenizers_version=version('tokenizers'),encoded_batch_cache_sha256=sha(ROOT/'scripts/encoded_batch_cache.py'),
                encoded_batch_cache_capacity=128,
                model_dtype=str(next(wrapper.model.parameters()).dtype),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                query_accounting='TextAttack logical candidate requests; physical model execution recorded separately')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'metadata.json').exists():
        if read(output/'metadata.json') != metadata:
            raise ValueError('Saved model, sample, code or runtime changed; use a new output folder')
    elif any(output.iterdir()):
        raise ValueError('Unrecognised nonempty output folder')
    else:
        save(output/'metadata.json', metadata)
        (output/'attack_configuration.txt').write_text(repr(attack)+'\n', encoding='utf-8')
        for filename,target in [('replication_runs.py','runner_snapshot.py'),
                                ('replication_attack.py','ablation_snapshot.py'),
                                ('replication_models.py','model_loader_snapshot.py'),
                                ('textfooler_paper_delta_0_7.py','recipe_snapshot.py')]:
            shutil.copyfile(ROOT/'scripts'/filename,output/target)
        if stream_word_ranking:
            shutil.copyfile(ROOT/'scripts/streaming_word_ranking.py',output/'streaming_word_ranking_snapshot.py')
            shutil.copyfile(ROOT/'scripts/fast_text_edits.py',output/'fast_text_edits_snapshot.py')
            if compiled_tokenizer:
                shutil.copyfile(ROOT/'scripts/fast_bert_tokenizer.py',output/'fast_bert_tokenizer_snapshot.py')
                shutil.copyfile(ROOT/'scripts/encoded_batch_cache.py',output/'encoded_batch_cache_snapshot.py')
    if (output/'baseline.json').exists():
        baseline=read(output/'baseline.json')
    else:
        classes=len(manifest['model_label_names']) or len({s['label'] for s in samples})
        baseline=clean_predictions(wrapper,samples,classes)
        save(output/'baseline.json',baseline)
    if len(baseline)!=len(samples) or any(row['row']!=sample['row'] or row['source_id']!=sample['source_id'] or row['label']!=sample['label'] for row,sample in zip(baseline,samples)):
        raise ValueError('Clean baseline does not match the fixed sample')
    print(f'{model_alias}/{manifest["dataset"]}: clean {sum(r["prediction"]==r["label"] for r in baseline)}/{len(samples)}',flush=True)
    rows = []
    if (output/'outcomes.csv').exists():
        with (output/'outcomes.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for field in ('row','source_id','label','original_prediction','perturbed_prediction','num_queries','original_word_count','changed_word_count'):
                row[field] = int(row[field])
        check_existing(rows, samples)
    if len(rows) == len(samples):
        if not (output/'summary.json').exists():
            recovered={'status':'completed','completed_at_utc':datetime.now(timezone.utc).isoformat(),
                'metadata_sha256':sha(output/'metadata.json'),'outcomes_sha256':sha(output/'outcomes.csv'),
                'baseline_sha256':sha(output/'baseline.json'),'metrics':metrics(rows),'human_or_llm_judging_completed':False}
            if compiled_tokenizer:
                recovered['model_execution_statistics']={'status':'unavailable_after_interruption',
                    'reason':'Completed outcomes recovered; physical execution counters were not persisted'}
            save(output/'summary.json',recovered)
        summary = read(output/'summary.json')
        if summary['outcomes_sha256'] != sha(output/'outcomes.csv'):
            raise ValueError('Completed outcomes changed')
        print(json.dumps({'status':'reused', 'output':str(output), 'metrics':summary['metrics']}), flush=True)
        return summary
    if (output/'run.lock').exists():
        raise RuntimeError('A run lock exists; verify no worker is active before removing a stale lock')
    descriptor = os.open(output/'run.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor,'w') as stream:
        stream.write(str(os.getpid()))
    logger = textattack.loggers.AttackLogManager(metrics=None)
    segment = len(rows)
    if compiled_tokenizer:wrapper.reset_statistics(clear_cache=True)
    add_utf8_text_log(logger, output/f'attack_results_{segment:04d}.txt')
    logger.add_output_csv(str(output/f'attack_results_{segment:04d}.csv'), color_method='file')
    started = time.monotonic()
    try:
        with (output/'outcomes.csv').open('a', encoding='utf-8', newline='') as stream:
            writer = None
            for sample in samples[segment:]:
                index = sample['row']
                textattack.shared.utils.set_seed(seed + index)
                inputs = OrderedDict((name,sample['inputs'][name]) for name in manifest['input_columns'])
                result = attack.attack(inputs, sample['label'])
                original, changed = result.original_result.attacked_text, result.perturbed_result.attacked_text
                original_inputs, changed_inputs = original._text_input, changed._text_input
                if dict(original_inputs) != sample['inputs']:
                    raise RuntimeError('Attack input differs from saved sample')
                premise = original_inputs.get('premise','')
                is_pair = 'premise' in original_inputs
                relevant = textattack.shared.AttackedText(original_inputs['hypothesis']) if is_pair else original
                row = {'row':index,'source_id':sample['source_id'],'label':sample['label'],
                       'original_text':'\n'.join(original_inputs.values()), 'perturbed_text':'\n'.join(changed_inputs.values()),
                       'original_premise':premise,'perturbed_premise':changed_inputs.get('premise',''),
                       'original_hypothesis':original_inputs.get('hypothesis',''),
                       'perturbed_hypothesis':changed_inputs.get('hypothesis',''),
                       'original_prediction':int(result.original_result.output),
                       'perturbed_prediction':int(result.perturbed_result.output),
                       'result_type':type(result).__name__.removesuffix('AttackResult'),
                       'num_queries':int(result.num_queries), 'original_word_count':relevant.num_words,
                       'total_original_word_count':original.num_words, 'changed_word_count':original.words_diff_num(changed),
                       'elapsed_seconds':time.monotonic()-started}
                validate_outcome(row)
                if row['original_prediction']!=baseline[index]['prediction']:
                    raise ValueError('Attack original prediction differs from the clean baseline')
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if segment==0:
                        writer.writeheader()
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                logger.log_result(result)
                if len(rows)%50==0 or len(rows)==len(samples):
                    logger.flush()
                    print(f'{model_alias}/{manifest["dataset"]}/{variant}: {len(rows)}/{len(samples)} | successful {sum(r["result_type"]=="Successful" for r in rows)} | {time.monotonic()-started:.0f}s', flush=True)
        summary = {'status':'completed', 'completed_at_utc':datetime.now(timezone.utc).isoformat(),
                   'metadata_sha256':sha(output/'metadata.json'), 'outcomes_sha256':sha(output/'outcomes.csv'),
                   'baseline_sha256':sha(output/'baseline.json'),
                   'metrics':metrics(rows), 'human_or_llm_judging_completed':False}
        if compiled_tokenizer:
            summary['model_execution_statistics']={'status':'available','start_row':segment,'clean_baseline_included':False,
                'logical_candidate_queries':sum(int(row['num_queries']) for row in rows[segment:]),
                **wrapper.execution_statistics()}
        save(output/'summary.json', summary)
        print(json.dumps(summary['metrics']), flush=True)
        return summary
    finally:
        logger.flush()
        for item in logger.loggers:
            item.close()
        (output/'run.lock').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--dataset',choices=list(RELEASED_NAMES),required=True)
    prep.add_argument('--source',choices=['released','hf'],default='released')
    prep.add_argument('--output-dir',type=Path,required=True)
    prep.add_argument('--num-examples',type=int,default=1000)
    prep.add_argument('--seed',type=int,default=8240)
    run = commands.add_parser('run')
    run.add_argument('--sample-dir',type=Path,required=True)
    run.add_argument('--output-dir',type=Path,required=True)
    run.add_argument('--model',required=True)
    run.add_argument('--variant',choices=['full','no_sentence_similarity'],default='full')
    run.add_argument('--seed',type=int,default=8240)
    run.add_argument('--stream-word-ranking',action='store_true')
    args = parser.parse_args()
    if args.command=='prepare':
        prepare_sample(args.dataset,args.output_dir,args.num_examples,args.seed,args.source)
    else:
        run_experiment(args.sample_dir,args.output_dir,args.model,args.variant,args.seed,args.stream_word_ranking)


if __name__=='__main__':
    main()
