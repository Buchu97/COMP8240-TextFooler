from pathlib import Path
import sys
import tempfile
import unittest
import json
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


class OriginalDownloadTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'scripts/download_original_models.py').exists(),'Original checkpoint downloader is missing')
        from download_original_models import fetch
        self.fetch=fetch

    def response(self, body, content_type='application/octet-stream'):
        class Response:
            status_code=200
            headers={'Content-Type':content_type,'Content-Length':str(len(body))}
            def raise_for_status(self): pass
            def iter_content(self, size): yield body
            def close(self): pass
        return lambda *args,**kwargs:Response()

    def test_download_verifies_size_and_records_hash_before_reuse(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            path=Path(directory)/'weights.bin'
            result=self.fetch('https://example.org/weights',path,5,self.response(b'hello'))
            self.assertEqual(path.read_bytes(),b'hello')
            self.assertEqual(result['sha256'],'2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824')
            path.write_bytes(b'other')
            with self.assertRaises(ValueError):
                self.fetch('https://example.org/weights',path,5,self.response(b'hello'))

    def test_html_and_short_downloads_never_become_model_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            path=Path(directory)/'weights.bin'
            for body,kind in [(b'hello','text/html'),(b'no','application/octet-stream')]:
                with self.assertRaises(ValueError):
                    self.fetch('https://example.org/weights',path,5,self.response(body,kind))
                self.assertFalse(path.exists())

    def test_complete_partial_and_orphaned_final_recover_without_network(self):
        expected='2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
        def unavailable(*args,**kwargs):self.fail('Complete verified bytes were redownloaded')
        for suffix in ('.part',''):
            with self.subTest(suffix=suffix),tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
                path=Path(directory)/'weights.bin'
                Path(str(path)+suffix).write_bytes(b'hello')
                result=self.fetch('https://example.org/weights',path,5,unavailable,expected_sha256=expected)
                self.assertEqual(path.read_bytes(),b'hello')
                self.assertEqual(result['sha256'],expected)
                self.assertTrue(path.with_suffix('.bin.source.json').is_file())

    def test_pinned_checksum_rejects_same_size_wrong_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            path=Path(directory)/'weights.bin'
            with self.assertRaises(ValueError):
                self.fetch('https://example.org/weights',path,5,self.response(b'hello'),expected_sha256='0'*64)
            self.assertFalse(path.exists())

    def test_interrupted_publication_keeps_complete_receipt_and_recoverable_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            path=Path(directory)/'weights.bin'
            original=Path.replace
            def interrupted(source,target):
                if source.name=='weights.bin.part':raise OSError('Interrupted publication')
                return original(source,target)
            with patch.object(Path,'replace',interrupted),self.assertRaises(OSError):
                self.fetch('https://example.org/weights',path,5,self.response(b'hello'))
            self.assertFalse(path.exists())
            record=json.loads(path.with_suffix('.bin.source.json').read_text())
            result=self.fetch('https://example.org/weights',path,5,expected_sha256=record['sha256'])
            self.assertEqual(result['sha256'],record['sha256'])
            self.assertEqual(path.read_bytes(),b'hello')


if __name__=='__main__': unittest.main()
