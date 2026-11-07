import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class StandaloneH100Tests(unittest.TestCase):
    def test_pack_revision_is_honoured_without_resolving_a_new_snapshot(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            root = Path(folder)
            self.pack(root)
            pack = json.loads((root / 'h100_pack.json').read_text())
            pack['model_revision'] = 'e' * 40
            (root / 'h100_pack.json').write_text(json.dumps(pack))
            def forbidden(*args):
                raise AssertionError('Pinned package unexpectedly resolved a different model revision')
            snapshot = self.module.download_model(root, SimpleNamespace(model_info=forbidden), self.downloader)
            self.assertEqual(snapshot.name, 'e' * 40)
            state = json.loads((root / 'model.json').read_text())
            state['revision'] = 'f' * 40
            state['snapshot'] = state['snapshot'].replace('e' * 40, 'f' * 40)
            (root / 'model.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'revision'):
                self.module.ready_snapshot(root, verify_files=False)

    def setUp(self):
        path = ROOT / 'scripts/h100_judge.py'
        self.assertTrue(path.exists(), 'Standalone H100 bootstrap is missing')
        spec = importlib.util.spec_from_file_location('standalone_h100', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def pack(self, root):
        (root / 'h100_pack.json').write_text(json.dumps({'model': 'Qwen/Qwen3.8-27B',
            'phases': {'development': 'h100_development', 'control': 'h100_controls'}}))

    def downloader(self, **kwargs):
        snapshot = Path(kwargs['cache_dir']) / 'models--Qwen--Qwen3.8-27B' / 'snapshots' / kwargs['revision']
        snapshot.mkdir(parents=True, exist_ok=True)
        for name, content in {'config.json': '{}', 'tokenizer.json': '{}',
                              'tokenizer_config.json': '{}', 'model.safetensors': 'fixture weights'}.items():
            (snapshot / name).write_text(content)
        return str(snapshot)

    def test_empty_cache_download_is_pinned_and_reusable_without_any_other_project(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            root = Path(folder)
            self.pack(root)
            api = SimpleNamespace(model_info=lambda model: SimpleNamespace(sha='a' * 40))
            snapshot = self.module.download_model(root, api, self.downloader)
            self.assertTrue(snapshot.is_relative_to(root / 'model_cache'))
            manifest = json.loads((root / 'model.json').read_text())
            self.assertEqual(manifest['revision'], 'a' * 40)
            self.assertTrue(manifest['complete'])
            def forbidden(*args, **kwargs):
                raise AssertionError('Ready local model unexpectedly contacted the network')
            self.assertEqual(self.module.download_model(root, SimpleNamespace(model_info=forbidden), forbidden), snapshot)
            (snapshot / 'model.safetensors').unlink()
            with self.assertRaises(ValueError):
                self.module.ready_snapshot(root)
            repaired = self.module.download_model(root, SimpleNamespace(model_info=forbidden), self.downloader)
            self.assertEqual(repaired, snapshot)
            self.assertTrue((repaired / 'model.safetensors').is_file())

    def test_setup_lock_recovers_after_abrupt_process_exit_and_blocks_live_setup(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            root = Path(folder)
            self.assertTrue(hasattr(self.module, 'setup_lock'), 'Crash-safe setup lock is missing')
            program = ('import importlib.util, os, sys; from pathlib import Path; '
                       's=importlib.util.spec_from_file_location("bootstrap", sys.argv[1]); '
                       'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                       'lock=m.setup_lock(Path(sys.argv[2])); lock.__enter__(); os._exit(0)')
            subprocess.run([sys.executable, '-I', '-c', program,
                            str(ROOT / 'scripts/h100_judge.py'), str(root)], check=True)
            with self.module.setup_lock(root):
                with self.assertRaises(RuntimeError):
                    with self.module.setup_lock(root):
                        self.fail('Two setup processes acquired the same live lock')

    def test_interrupted_download_reuses_saved_commit_and_rejects_external_cache(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            root = Path(folder)
            self.pack(root)
            api = SimpleNamespace(model_info=lambda model: SimpleNamespace(sha='b' * 40))
            def interrupted(**kwargs):
                raise ConnectionError('Fixture interrupted transfer')
            with self.assertRaises(ConnectionError):
                self.module.download_model(root, api, interrupted)
            self.assertFalse(json.loads((root / 'model.json').read_text())['complete'])
            snapshot = self.module.download_model(root, SimpleNamespace(model_info=lambda model: SimpleNamespace(sha='c' * 40)), self.downloader)
            self.assertEqual(snapshot.name, 'b' * 40)
            manifest = json.loads((root / 'model.json').read_text())
            manifest['snapshot'] = '../other_project/snapshot'
            (root / 'model.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                self.module.ready_snapshot(root)

    def test_run_uses_pack_local_environment_and_removes_inherited_project_paths(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            root = Path(folder)
            self.pack(root)
            self.module.download_model(root, SimpleNamespace(model_info=lambda model: SimpleNamespace(sha='d' * 40)), self.downloader)
            executable = root / '.venv' / 'Scripts' / 'python.exe'
            executable.parent.mkdir(parents=True)
            executable.touch()
            jobs = root / 'h100_development'
            jobs.mkdir()
            (jobs / 'hf_judge_worker.py').write_text('')
            (jobs / 'bundle.json').write_text('{}')
            observed = []
            def process(command, **kwargs):
                observed.append((command, kwargs))
                return SimpleNamespace(returncode=0)
            with patch.dict('os.environ', {'PYTHONPATH': 'other_project', 'HF_HOME': 'other_cache',
                                          'VIRTUAL_ENV': 'other_env', 'HF_ENDPOINT': 'https://other-host.invalid'}), \
                 patch.object(self.module.subprocess, 'run', side_effect=process):
                self.module.run_phase(root, 'development', 60)
            command, options = observed[0]
            self.assertEqual(Path(command[0]), executable)
            self.assertIn(str(root / 'results' / 'development'), command)
            self.assertNotIn('PYTHONPATH', options['env'])
            self.assertNotIn('VIRTUAL_ENV', options['env'])
            self.assertNotIn('HF_ENDPOINT', options['env'])
            self.assertEqual(options['env']['HF_HUB_CACHE'], str(root / 'model_cache'))
            self.assertEqual(options['env']['HF_HUB_OFFLINE'], '1')
            self.assertEqual(options['cwd'], root)


if __name__ == '__main__':
    unittest.main()
