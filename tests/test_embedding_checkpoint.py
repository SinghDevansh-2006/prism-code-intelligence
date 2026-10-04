import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from src.embedding_checkpoint import encode_checkpointed

class Encoder:
    max_seq_length=20
    dtype="torch.float32"
    device="cpu"
    def __init__(self):self.calls=0
    def __getitem__(self,index):return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash='fixture')))
    def encode_document(self,texts,**kwargs):
        self.calls+=1
        return np.array([[len(t),1] for t in texts],dtype=np.float32)
    encode_query=encode_document

class CheckpointTests(unittest.TestCase):
    def test_resume_restores_original_order_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            model=Encoder();texts=['a','abcdef','xy']
            result=encode_checkpointed(model,texts,'gemma_documents',root,2)
            np.testing.assert_array_equal(result[:,0],[1,6,2])
            encode_checkpointed(model,texts,'gemma_documents',root,2)
            self.assertEqual(model.calls,1)
            with self.assertRaisesRegex(ValueError,'inputs/config changed'):
                encode_checkpointed(model,['different'],'gemma_documents',root,2)

    def test_precision_change_rejects_checkpoint_reuse(self):
        with tempfile.TemporaryDirectory() as root:
            model=Encoder()
            encode_checkpointed(model,['code'],'qwen_documents',root,1)
            model.dtype='torch.bfloat16'
            with self.assertRaisesRegex(ValueError,'inputs/config changed'):
                encode_checkpointed(model,['code'],'qwen_documents',root,1)
            self.assertEqual(model.calls,1)
