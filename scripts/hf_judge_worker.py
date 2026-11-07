"""Standalone, offline Hugging Face judge worker for physical H100 GPU 0."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def preflight(bundle_path, output):
    bundle_path, output = Path(bundle_path), Path(output)
    bundle = load(bundle_path)
    worker_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if worker_sha != bundle['worker_sha256']:
        raise ValueError('Worker code differs from the exported job pack')
    seen = set()
    for job in bundle['jobs']:
        expected_id = signature({'protocol': bundle['protocol'], 'pair_id': job['pair_id'],
                                 'kind': job['kind'], 'body': job['body']})
        if job['job_id'] != expected_id or job['job_id'] in seen:
            raise ValueError('Local job identity is invalid or duplicated')
        seen.add(job['job_id'])
    resolved_output, bundle_parent = output.resolve(), bundle_path.parent.resolve()
    if resolved_output == bundle_parent or bundle_parent.is_relative_to(resolved_output):
        raise ValueError('Worker outputs must be separate from input files')
    if (output / 'worker.lock').exists():
        raise FileExistsError('Another worker may be active; inspect the existing lock before proceeding')
    metadata = load(output / 'worker_metadata.json') if (output / 'worker_metadata.json').exists() else None
    if metadata and (metadata['bundle_signature'] != signature(bundle) or metadata['worker_sha256'] != worker_sha):
        raise ValueError('Existing worker results belong to a different bundle or implementation')
    missing = 0
    for job in bundle['jobs']:
        path = output / 'responses' / (job['job_id'] + '.json')
        if not path.exists():
            missing += 1
            continue
        record = load(path)
        if (not metadata or record['job_id'] != job['job_id']
                or record['request_sha256'] != signature(job['body'])
                or record['response_sha256'] != signature(record['response'])
                or record['bundle_signature'] != signature(bundle)
                or record['response'].get('id') != 'local_' + job['job_id']):
            raise ValueError('Saved local response identity or content changed')
    return bundle, metadata, missing


def run_jobs(bundle_path, output, generate, metadata, max_jobs):
    bundle_path, output = Path(bundle_path), Path(output)
    bundle, _, _ = preflight(bundle_path, output)
    if max_jobs < 1 or metadata['model'] != bundle['config']['model'] or not re.fullmatch('[a-f0-9]{40}', metadata['revision']):
        raise ValueError('Invalid model metadata or job limit')
    worker_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    expected = {'bundle_signature': signature(bundle), 'worker_sha256': worker_sha, 'model_metadata': metadata}
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'worker.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    generated = reused = 0
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(str(os.getpid()))
        if (output / 'worker_metadata.json').exists():
            if load(output / 'worker_metadata.json') != expected:
                raise ValueError('Worker model or bundle changed; preserve the old results and choose a new output directory')
        else:
            save(output / 'worker_metadata.json', expected)
        for job in bundle['jobs']:
            path = output / 'responses' / (job['job_id'] + '.json')
            if path.exists():
                record = load(path)
                if (record['job_id'] != job['job_id'] or record['request_sha256'] != signature(job['body'])
                        or record['response_sha256'] != signature(record['response'])
                        or record['bundle_signature'] != signature(bundle)):
                    raise ValueError('Saved local response identity or content changed')
                reused += 1
                continue
            if generated >= max_jobs:
                continue
            answer = generate(job['body'])
            if not isinstance(answer.get('text'), str):
                raise ValueError('Local generation returned no text')
            response = {'id': 'local_' + job['job_id'], 'model': metadata['model'] + '@' + metadata['revision'],
                        'choices': [{'finish_reason': answer.get('finish_reason', 'stop'),
                                     'message': {'role': 'assistant', 'content': answer['text']}}],
                        'usage': {'prompt_tokens': answer['prompt_tokens'], 'completion_tokens': answer['completion_tokens']}}
            save(path, {'job_id': job['job_id'], 'bundle_signature': signature(bundle),
                        'request_sha256': signature(job['body']), 'response_sha256': signature(response),
                        'response': response, 'elapsed_seconds': answer['elapsed_seconds'],
                        'completed_at_utc': datetime.now(timezone.utc).isoformat()})
            generated += 1
            print(f'[judge] {reused + generated}/{len(bundle["jobs"])} received | {answer["elapsed_seconds"]:.1f}s', flush=True)
    finally:
        lock.unlink(missing_ok=True)
    return {'generated': generated, 'reused': reused, 'remaining': len(bundle['jobs']) - generated - reused}


@contextmanager
def efficient_gqa():
    import torch.nn.functional as functional
    from torch.nn.attention import sdpa_kernel, SDPBackend
    original = functional.scaled_dot_product_attention
    def attention(query, key, value, *args, **kwargs):
        if query.device.type != 'cuda':
            return original(query, key, value, *args, **kwargs)
        groups = query.shape[-3] // key.shape[-3]
        if key.shape[-3] != value.shape[-3] or query.shape[-3] % key.shape[-3]:
            raise ValueError('Invalid grouped-query attention heads')
        if groups != 1:
            key = key.repeat_interleave(groups, dim=-3)
            value = value.repeat_interleave(groups, dim=-3)
        kwargs['enable_gqa'] = False
        backends = [SDPBackend.EFFICIENT_ATTENTION]
        if query.shape[-2] == 1:
            backends.append(SDPBackend.MATH)
        with sdpa_kernel(backends, set_priority=True):
            return original(query, key, value, *args, **kwargs)
    functional.scaled_dot_product_attention = attention
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = original


def load_generator(bundle, snapshot, max_input_tokens):
    snapshot = Path(snapshot).resolve()
    revision = snapshot.name
    expected_cache = 'models--' + bundle['config']['model'].replace('/', '--')
    if (not re.fullmatch('[a-f0-9]{40}', revision) or snapshot.parent.name != 'snapshots'
            or snapshot.parent.parent.name != expected_cache):
        raise ValueError('Use the immutable snapshot directory for the exact exported model in its existing Hugging Face cache')
    result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=uuid,name,memory.free',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    uuid, name, free = [part.strip() for part in result.stdout.strip().split(',')]
    if not uuid.startswith('GPU-') or 'H100' not in name or float(free) < 65 * 1024:
        raise RuntimeError('Physical GPU 0 must be an H100 with at least 65 GiB free before loading BF16 weights')
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=uuid,
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTORCH_CUDA_ALLOC_CONF='backend:native')
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, set_seed
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Refusing inference outside the isolated H100 GPU 0')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    def rendered(body):
        schema = json.dumps(body['response_format']['json_schema']['schema'], ensure_ascii=False)
        messages = [dict(message) for message in body['messages']]
        messages[0]['content'] += '\nReturn one JSON object matching this schema, with no markdown fences:\n' + schema
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False, preserve_thinking=False)
    model_config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    text_config = getattr(model_config, 'text_config', model_config)
    context = getattr(text_config, 'max_position_embeddings', max_input_tokens)
    for job in bundle['jobs']:
        tokens = len(tokenizer(rendered(job['body']), add_special_tokens=False)['input_ids'])
        output_limit = job['body'].get('max_completion_tokens', job['body'].get('max_tokens'))
        if tokens > max_input_tokens or tokens + output_limit > context:
            raise ValueError('A complete judge prompt exceeds the configured context; nothing was truncated')
    architecture = AutoModelForCausalLM
    if 'Qwen3_5ForConditionalGeneration' in (model_config.architectures or []):
        from transformers import Qwen3_5ForConditionalGeneration
        architecture = Qwen3_5ForConditionalGeneration
    print(f'[load] {bundle["config"]["model"]} | pinned snapshot | BF16 | physical GPU 0 | offline', flush=True)
    model = architecture.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                         dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
    model.generation_config.disable_compile = True
    if getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False):
        raise RuntimeError('Expected BF16 judge weights, not quantized weights')
    if any(p.device.type != 'cuda' or p.device.index != 0 for p in model.parameters()):
        raise RuntimeError('All judge parameters must stay on physical GPU 0')
    metadata = {'model': bundle['config']['model'], 'revision': revision, 'snapshot': str(snapshot),
                'config_sha256': hashlib.sha256((snapshot / 'config.json').read_bytes()).hexdigest(),
                'gpu_uuid': uuid, 'gpu': name, 'dtype': 'bfloat16', 'thinking': False,
                'attention': 'efficient_sdpa_explicit_gqa', 'do_sample': False,
                'schema_enforcement': 'post_generation_validation',
                'max_input_tokens': max_input_tokens,
                'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'accelerate')}}
    def generate(body):
        set_seed(8240)
        inputs = tokenizer(rendered(body), return_tensors='pt', add_special_tokens=False).to('cuda:0')
        limit = body.get('max_completion_tokens', body.get('max_tokens'))
        torch.cuda.synchronize(0)
        started = time.perf_counter()
        with torch.inference_mode(), efficient_gqa():
            output = model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize(0)
        generated = output[0, inputs['input_ids'].shape[1]:]
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        finished = bool(len(generated) and int(generated[-1]) in eos)
        return {'text': tokenizer.decode(generated, skip_special_tokens=True).strip(),
                'prompt_tokens': inputs['input_ids'].shape[1], 'completion_tokens': len(generated),
                'finish_reason': 'stop' if finished or len(generated) < limit else 'length',
                'elapsed_seconds': time.perf_counter() - started}
    return generate, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--snapshot-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-jobs', type=int, required=True)
    parser.add_argument('--max-input-tokens', type=int, default=8192)
    args = parser.parse_args()
    if args.max_jobs < 1 or args.max_input_tokens < 128:
        parser.error('Job limit must be positive and input limit at least 128 tokens')
    bundle, existing, missing = preflight(args.bundle, args.output_dir)
    if existing and existing['model_metadata']['revision'] != args.snapshot_dir.name:
        raise ValueError('Requested snapshot differs from the saved judge results')
    if not missing and existing:
        def no_generation(body):
            raise RuntimeError('Completed packs must not regenerate responses')
        print(json.dumps(run_jobs(args.bundle, args.output_dir, no_generation, existing['model_metadata'], args.max_jobs), indent=2))
        return
    generate, metadata = load_generator(bundle, args.snapshot_dir, args.max_input_tokens)
    print(json.dumps(run_jobs(args.bundle, args.output_dir, generate, metadata, args.max_jobs), indent=2))


if __name__ == '__main__':
    main()
