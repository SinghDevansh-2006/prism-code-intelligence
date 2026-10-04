import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from datasets import Dataset
from evaluate_mteb_pipeline import PrismSearch, fused_scores

class MTEBAdapterTests(unittest.TestCase):
    def test_fusion_uses_distributions_not_raw_weighted_cosines(self):
        gemma = np.array([0., 1., 2.], dtype=np.float32)
        qwen = np.array([20., 10., 0.], dtype=np.float32)
        scores = fused_scores(gemma, qwen)
        self.assertEqual(np.argsort(-scores).tolist(), [2,1,0])
        np.testing.assert_allclose(scores, fused_scores(gemma+10, qwen*2), atol=1e-6)
        self.assertTrue(np.isfinite(fused_scores(np.ones(3), np.ones(3))).all())

    def test_cached_ids_validated_and_ranked(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root/'document_ids.json').write_text(json.dumps(['a','b']))
            (root/'query_ids.json').write_text(json.dumps(['q']))
            for model in ['embeddinggemma','qwen']:
                np.save(root/f'{model}_documents.npy', np.eye(2, dtype=np.float32))
                np.save(root/f'{model}_queries.npy', np.array([[0,1]], dtype=np.float32))
            with PrismSearch(root) as adapter:
                with self.assertRaisesRegex(ValueError, 'IDs/order'):
                    adapter.index(Dataset.from_dict({'id':['b','a'], 'text':['','']}))
                adapter.index(Dataset.from_dict({'id':['a','b'], 'text':['','']}))
                result=adapter.search(Dataset.from_dict({'id':['q'], 'text':['']}), top_k=1)
                self.assertEqual(list(result['q']), ['b'])

    def test_close_releases_mappings_before_file_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'embeddings.npy'
            np.save(path,np.eye(2,dtype=np.float32))
            with PrismSearch(directory) as adapter:
                array=adapter.cached(path.name)
                mapping=array._mmap
                self.assertFalse(mapping.closed)
            self.assertTrue(mapping.closed)
            adapter.close()  # Repeated cleanup is safe.
            path.unlink()  # Windows must be able to remove the file now.

    def test_context_closes_mappings_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory)/'embeddings.npy',np.eye(2,dtype=np.float32))
            with self.assertRaisesRegex(RuntimeError,'fixture failure'):
                with PrismSearch(directory) as adapter:
                    mapping=adapter.cached('embeddings.npy')._mmap
                    raise RuntimeError('fixture failure')
            self.assertTrue(mapping.closed)
