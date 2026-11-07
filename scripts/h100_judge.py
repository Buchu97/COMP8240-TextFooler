"""Create and run the standalone Windows H100 TextFooler judge environment."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def environment(root, download=False):
    root = Path(root).resolve()
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'HF_HOME', 'HF_HUB_CACHE',
                'HUGGINGFACE_HUB_CACHE', 'HF_DATASETS_CACHE', 'TRANSFORMERS_CACHE',
                'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_ENDPOINT'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', HF_HOME=str(root / 'hf_home'),
               HF_HUB_CACHE=str(root / 'model_cache'), HF_HUB_DISABLE_TELEMETRY='1',
               PIP_CACHE_DIR=str(root / 'pip_cache'))
    if not download:
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def expected_snapshot(root, model, revision):
    if not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('The model revision must be a full commit SHA')
    return Path(root).resolve() / 'model_cache' / ('models--' + model.replace('/', '--')) / 'snapshots' / revision


def check_files(snapshot):
    for filename in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        path = snapshot / filename
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Incomplete dedicated model download: {filename}')
    config = read(snapshot / 'config.json')
    if config.get('quantization_config'):
        raise ValueError('The judge requires the official BF16 weights')
    index = snapshot / 'model.safetensors.index.json'
    weights = set(read(index)['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if not weights:
        raise ValueError('The model weight index is empty')
    for name in weights:
        path = snapshot / name
        if Path(name).name != name or not path.is_file() or not path.stat().st_size:
            raise ValueError('Incomplete dedicated model weight download')


def ready_snapshot(root, verify_files=True):
    root = Path(root).resolve()
    pack, state = read(root / 'h100_pack.json'), read(root / 'model.json')
    if pack.get('model_revision') and state['revision'] != pack['model_revision']:
        raise ValueError('Saved model revision differs from the pinned package revision')
    expected = expected_snapshot(root, pack['model'], state['revision'])
    if (state['model'] != pack['model'] or not state['complete']
            or Path(state['snapshot']).is_absolute()
            or (root / state['snapshot']).resolve() != expected.resolve()
            or not expected.resolve().is_relative_to(root / 'model_cache')):
        raise ValueError('Model setup is incomplete or refers outside this job folder')
    if verify_files:
        check_files(expected)
    return expected


def download_model(root, api, downloader):
    root = Path(root).resolve()
    pack = read(root / 'h100_pack.json')
    model = pack['model']
    pinned_revision = pack.get('model_revision')
    if pinned_revision is not None:
        expected_snapshot(root, model, pinned_revision)
    state_path = root / 'model.json'
    if state_path.exists():
        state = read(state_path)
        if state['model'] != model or (pinned_revision and state['revision'] != pinned_revision):
            raise ValueError('Model differs from the saved setup; use a new standalone folder')
        if state['complete']:
            expected = ready_snapshot(root, verify_files=False)
            try:
                check_files(expected)
                return expected
            except (OSError, ValueError):
                state['complete'] = False
                write(state_path, state)
        revision = state['revision']
    else:
        revision = pinned_revision or api.model_info(model).sha
        snapshot = expected_snapshot(root, model, revision)
        state = {'model': model, 'revision': revision, 'complete': False,
                 'snapshot': snapshot.relative_to(root).as_posix()}
        write(state_path, state)
    expected = expected_snapshot(root, model, revision)
    print(f'[download] {model} | commit {revision} | dedicated model_cache', flush=True)
    downloaded = Path(downloader(repo_id=model, revision=revision, cache_dir=str(root / 'model_cache'),
                                allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model'],
                                max_workers=4)).resolve()
    if downloaded != expected.resolve() or not downloaded.is_relative_to(root / 'model_cache'):
        raise ValueError('Downloader returned a model outside the dedicated cache')
    check_files(expected)
    state['complete'] = True
    write(state_path, state)
    return ready_snapshot(root)


def python_path(root):
    executable = Path(root).resolve() / '.venv' / 'Scripts' / 'python.exe'
    if executable.exists() and not executable.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('The judge interpreter must belong to this standalone folder')
    return executable


@contextmanager
def setup_lock(root):
    # OS locks are released even after termination; the marker file may remain.
    with (Path(root) / 'setup.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another setup process is active in this folder') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def setup(root):
    root = Path(root).resolve()
    if os.name != 'nt' or sys.version_info[:2] != (3, 11):
        raise RuntimeError('Run setup on Windows with: py -3.11 .\\h100_judge.py setup')
    requirements = root / 'requirements_h100_judge.txt'
    read(root / 'h100_pack.json')
    with setup_lock(root):
        env = environment(root, download=True)
        executable = python_path(root)
        if not executable.exists():
            subprocess.run([sys.executable, '-I', '-m', 'venv', str(root / '.venv')], cwd=root, env=env, check=True)
        subprocess.run([str(executable), '-I', '-m', 'pip', '--isolated', 'install',
                        'torch==2.7.1', '--index-url', 'https://download.pytorch.org/whl/cu126'],
                       cwd=root, env=env, check=True)
        subprocess.run([str(executable), '-I', '-m', 'pip', '--isolated', 'install', '-r', str(requirements)],
                       cwd=root, env=env, check=True)
        subprocess.run([str(executable), '-I', '-m', 'pip', 'check'], cwd=root, env=env, check=True)
        frozen = subprocess.run([str(executable), '-I', '-m', 'pip', 'freeze'],
                                cwd=root, env=env, check=True, capture_output=True, text=True)
        write(root / 'runtime.json', {'python': '3.11', 'packages': frozen.stdout.splitlines(),
              'requirements_sha256': hashlib.sha256(requirements.read_bytes()).hexdigest()})
        subprocess.run([str(executable), '-I', str(root / 'h100_judge.py'), '_download'],
                       cwd=root, env=env, check=True)
        print('[ready] Dedicated environment and pinned model download complete', flush=True)


def run_phase(root, phase, max_jobs):
    root = Path(root).resolve()
    if max_jobs < 1:
        raise ValueError('max-jobs must be positive')
    pack = read(root / 'h100_pack.json')
    if phase not in pack['phases']:
        raise ValueError('This phase is absent from the job pack')
    jobs = (root / pack['phases'][phase]).resolve()
    if not jobs.is_relative_to(root):
        raise ValueError('Job files must belong to the standalone folder')
    executable = python_path(root)
    if not executable.is_file():
        raise ValueError('Run setup first to create the dedicated judge environment')
    snapshot = ready_snapshot(root)
    subprocess.run([str(executable), '-I', str(jobs / 'hf_judge_worker.py'),
                    '--bundle', str(jobs / 'bundle.json'), '--snapshot-dir', str(snapshot),
                    '--output-dir', str(root / 'results' / phase), '--max-jobs', str(max_jobs)],
                   cwd=root, env=environment(root), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('setup', help='Create the dedicated environment and download the pinned model')
    commands.add_parser('_download', help=argparse.SUPPRESS)
    run = commands.add_parser('run', help='Run/resume a phase with the dedicated local model')
    run.add_argument('--phase', choices=['development', 'control', 'final'], default='development')
    run.add_argument('--max-jobs', type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        if args.command == 'setup':
            setup(root)
        elif args.command == '_download':
            from huggingface_hub import HfApi, snapshot_download
            download_model(root, HfApi(), snapshot_download)
        else:
            run_phase(root, args.phase, args.max_jobs)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
