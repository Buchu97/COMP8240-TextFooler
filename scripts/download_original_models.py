"""Download public checkpoints linked by the original TextFooler repository."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'bert_fake':{'pytorch_model.bin':('1ZF2Yo7UhotpUK40bPfa7OQvzw2iCZ9_L',437983779),
                 'bert_config.json':('1vAgJ66pF-9l4J9j3Nx_lAfBIbH6Wl8Bc',313),
                 'vocab.txt':('1HmMSFY9HgQZKi5-B5_X4gNks7z7pvFqH',231508)},
    'bert_mr':{'pytorch_model.bin':('1ceklt5EBHn2RVKRPmaQLxKKhcDObHMvQ',437983779),
               'bert_config.json':('1oPGXr66hZNOGY92GcEua5qjdcF2bPVxL',313),
               'vocab.txt':('1nxEF8F-33ENfcD7riz2Ns0nGARQxGshr',231508)},
    'bert_snli':{'pytorch_model.bin':('1OGIQO_F2KjVfxctgmdMUheFOtAh8gZ8w',437986855),
                 'bert_config.json':('1IHqOlEJZRWXDpVHtv-wtCo25kugJxnRR',313),
                 'vocab.txt':('1LBUp_C_J0U7Z5NISl1Hif88OrOqi2HmP',231508)},
    'bert_mnli':{'pytorch_model.bin':('1pbPRy-whHKKIeUuEJaZymkMUCCFlGyPO',437986855),
                 'bert_config.json':('1ddaxLSeOiJSKbXbOsPtgGthIZDe6MEOo',313),
                 'vocab.txt':('1CEAR6yxHC1b8b-1p5bfKWQH9-KeE5nBI',231508)},
    'cnn_fake':{'pytorch_model.bin':('1PtiDs88v2xKrp-PpAOgBYIShVmuHaaGJ',320966142)},
    'lstm_fake':{'pytorch_model.bin':('1KWcOND53xjr5Xh1LViBNasZKReleDmfY',320668489)}}
CATALOG=ROOT/'notes/original_checkpoint_catalog.json'
PINNED_HASHES={}
if CATALOG.exists():
    for model,entry in json.loads(CATALOG.read_text(encoding='utf-8-sig')).items():
        SOURCES[model]={name:(item['file_id'],item['bytes']) for name,item in entry['files'].items()}
        PINNED_HASHES.update({(model,name):item['sha256'] for name,item in entry['files'].items() if 'sha256' in item})


def digest(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch(url,path,expected_size,transport=None,expected_sha256=None):
    import requests
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    record_path=path.with_suffix(path.suffix+'.source.json')
    def publish(candidate):
        if candidate.stat().st_size!=expected_size:
            raise ValueError('Incomplete checkpoint download')
        actual=digest(candidate)
        if expected_sha256 is not None and actual!=expected_sha256:
            raise ValueError('Checkpoint differs from the pinned SHA256')
        record={'url':url,'bytes':expected_size,'sha256':actual,
                'downloaded_at_utc':datetime.now(timezone.utc).isoformat(),
                'source':'Public checkpoint linked from jind11/TextFooler README; local SHA256 recorded after transfer'}
        temporary=record_path.with_suffix(record_path.suffix+'.tmp')
        temporary.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        temporary.replace(record_path)
        if candidate!=path:candidate.replace(path)
        return record
    if path.exists():
        if not record_path.exists():
            if expected_sha256 is not None:return publish(path)
            raise ValueError('Existing download has no provenance record')
        record=json.loads(record_path.read_text(encoding='utf-8'))
        if (record['url']!=url or path.stat().st_size!=expected_size or digest(path)!=record['sha256'] or
            (expected_sha256 is not None and record['sha256']!=expected_sha256)):
            raise ValueError('Existing checkpoint differs from the verified download')
        return record
    partial=path.with_suffix(path.suffix+'.part')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset==expected_size:return publish(partial)
    if offset>expected_size:raise ValueError('Partial checkpoint exceeds the published size')
    headers={'Range':f'bytes={offset}-'} if offset else {}
    response=(transport or requests.get)(url,headers=headers,stream=True,timeout=(20,120))
    try:
        response.raise_for_status()
        if 'html' in response.headers.get('Content-Type','').lower():
            raise ValueError('Download returned HTML instead of checkpoint bytes')
        append=response.status_code==206 and offset>0
        if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
            raise ValueError('Download resume range differs')
        remaining=expected_size-offset if append else expected_size
        if int(response.headers.get('Content-Length',remaining))!=remaining:
            raise ValueError('Download size differs from published artifact metadata')
        with partial.open('ab' if append else 'wb') as stream:
            for chunk in response.iter_content(4*1024*1024):
                if chunk:
                    stream.write(chunk)
        return publish(partial)
    finally:
        response.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',choices=list(SOURCES),nargs='+',required=True)
    args=parser.parse_args()
    if any(model.startswith(('cnn_','lstm_')) for model in args.models):
        glove=ROOT/'data/cache/textattack/word_embeddings/glove200'
        if not all((glove/name).is_file() for name in ('glove.wordlist.npy','glove.6B.200d.mat.npy')):
            from replication_runs import configure
            configure()
            from textattack.shared import utils
            utils.download_from_s3('word_embeddings/glove200')
    for model in args.models:
        for filename,(identifier,size) in SOURCES[model].items():
            path=ROOT/'models/original_textfooler'/model/filename
            print(f'{model}: {filename} ({size} bytes)',flush=True)
            record=fetch(f'https://drive.usercontent.google.com/download?id={identifier}&export=download&authuser=0&confirm=t',path,size,
                         expected_sha256=PINNED_HASHES.get((model,filename)))
            print(f'verified: {record["sha256"]}',flush=True)


if __name__=='__main__':main()
