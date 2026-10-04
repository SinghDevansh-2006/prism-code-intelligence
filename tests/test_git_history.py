import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.versioned_retriever import VersionedSemanticIndex, EMBEDDING_DIM
from index_git_history import index_history

class FakeEncoder:
    def __init__(self): self.encodes = 0
    def encode_document(self, texts, **kwargs):
        self.encodes += len(texts)
        out = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        out[:, 0] = 1
        return out
    encode_query = encode_document

class GitHistoryTests(unittest.TestCase):
    def test_real_git_add_edit_delete_and_idempotence(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            repo = root / 'repo'; repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.STDOUT)
            git('init'); git('config', 'user.name', 'Test'); git('config', 'user.email', 'test@example.invalid')
            path = repo / 'example.py'
            for code in ['def f(): return 1\n', 'def f(): return 2\n', None]:
                if code is None: path.unlink()
                else: path.write_text(code)
                git('add', '-A'); git('commit', '-m', 'Fixture change')
            model = FakeEncoder()
            index = VersionedSemanticIndex(root / 'index', shared_model=model)
            events = index_history(repo, ['example.py'], index)
            self.assertEqual([e['operation'] for e in events], ['added', 'modified', 'deleted'])
            self.assertEqual(model.encodes, 2)
            self.assertEqual(index_history(repo, ['example.py'], index), [])
            self.assertEqual(len(index.history('example.py')), 3)
            self.assertEqual(len(index.search('find f')['results']), 2)
            loaded = VersionedSemanticIndex(root / 'index', shared_model=model)
            self.assertEqual(loaded.stats()['versions'], 3)
            bounded = VersionedSemanticIndex(root / 'bounded', shared_model=model)
            index_history(repo, ['example.py'], bounded, max_commits=2)
            with self.assertRaisesRegex(ValueError, 'older or divergent'):
                index_history(repo, ['example.py'], bounded, max_commits=3)

    def test_failed_write_rolls_back(self):
        with tempfile.TemporaryDirectory() as root:
            model = FakeEncoder(); index = VersionedSemanticIndex(root, shared_model=model)
            index.add_version('f', '1', 'def f(): return 1')
            before = index.embeddings_path.read_bytes()
            with patch.object(index, '_append_record', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    index.add_version('f', '2', 'def f(): return 2')
            self.assertEqual(index.embeddings_path.read_bytes(), before)
            self.assertEqual(VersionedSemanticIndex(root, shared_model=model).stats()['versions'], 1)
            index.add_version('f', '2', 'def f(): return 1')
            self.assertEqual(model.encodes, 2)  # successful initial + failed changed; identical reused

    def test_orphan_vectors_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'embeddings.f32').write_bytes(b'bad')
            with self.assertRaisesRegex(RuntimeError, 'without version metadata'):
                VersionedSemanticIndex(root, shared_model=FakeEncoder())

    def test_snapshot_selection_excludes_deleted_and_future_code(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); repo=root/'repo';repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.STDOUT).decode().strip()
            git('init');git('config','user.name','Test');git('config','user.email','test@example.invalid')
            commits=[]
            for code in ['def f(): return 1', 'def f(): return 2', None]:
                if code is None:(repo/'f.py').unlink()
                else:(repo/'f.py').write_text(code)
                git('add','-A');git('commit','-m','Version');commits.append(git('rev-parse','HEAD'))
            index=VersionedSemanticIndex(root/'index',shared_model=FakeEncoder())
            index_history(repo,['f.py'],index)
            self.assertEqual(index.search('f',scope='commit',commit=commits[0])['results'][0]['code'],'def f(): return 1')
            self.assertEqual(index.search('f',scope='commit',commit=commits[1])['results'][0]['code'],'def f(): return 2')
            self.assertEqual(index.search('f',scope='latest')['results'],[])
            self.assertEqual(index.search('f',scope='commit',commit=commits[2])['results'],[])
            self.assertEqual(len(index.search('f',scope='all')['results']),2)
            with self.assertRaises(ValueError):index.search('f',scope='commit',commit='missing')

    def test_sliding_import_window_preserves_unchanged_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); repo = root/'repo'; repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),*args], stderr=subprocess.STDOUT).decode().strip()
            git('init'); git('config','user.name','Test'); git('config','user.email','test@example.invalid')
            (repo/'f.py').write_text('def f(): return 1')
            git('add','.'); git('commit','-m','Add code')
            original = git('rev-parse','HEAD')
            index = VersionedSemanticIndex(root/'index',shared_model=FakeEncoder())
            index_history(repo,['f.py'],index,max_commits=2)
            for number in range(3):
                (repo/'notes.txt').write_text(str(number))
                git('add','.'); git('commit','-m','Unrelated change')
                self.assertEqual(index_history(repo,['f.py'],index,max_commits=2),[])
                self.assertEqual(index.search('f',scope='latest')['results'][0]['version'],original)
            (repo/'f.py').write_text('def f(): return 2')
            git('add','.'); git('commit','-m','Change code')
            events = index_history(repo,['f.py'],index,max_commits=1)
            self.assertEqual([event['operation'] for event in events],['modified'])
            self.assertEqual(len(index.history('f.py')),2)
