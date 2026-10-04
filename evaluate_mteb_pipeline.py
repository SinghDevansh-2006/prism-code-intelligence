"""Run MTEB AppsRetrieval through PRISM's frozen 65/35 search adapter.

Default: fresh CPU document/query inference. --embedding-cache explicitly reuses
previous arrays, recomputes scores/rankings, and is NOT a fresh inference benchmark.
Outputs are separate from the submitted artifacts.
"""
import argparse
import hashlib
import gc
import json
import time
from pathlib import Path
import numpy as np
from src.code_normalizer import compact_large_literals
from src.embedding_checkpoint import encode_checkpointed


def fused_scores(gemma, qwen):
    def zscore(row):
        return (row - row.mean()) / (row.std() + 1e-8)
    return .65 * zscore(gemma) + .35 * zscore(qwen)


class PrismSearch:
    def __init__(self, embedding_cache=None, batch_size=8, output=None, qwen_batch_size=1):
        from mteb.models.model_meta import ModelMeta
        self.cache = Path(embedding_cache) if embedding_cache else None
        self.batch_size = batch_size
        self.qwen_batch_size = qwen_batch_size
        self.output = Path(output) if output else Path("cache/review/fresh-embeddings")
        self.hashes = {}
        self._cached_arrays = []
        required = {k: None for k, field in ModelMeta.model_fields.items() if field.is_required()}
        required.update(name='PRISM/frozen-fusion', revision='65-35-v1', framework=['Sentence Transformers', 'NumPy'])
        self.mteb_model_meta = ModelMeta(**required)

    def cached(self, filename):
        path = self.cache / filename
        self.hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        array = np.load(path, mmap_mode='r')
        self._cached_arrays.append(array)
        return array

    def close(self):
        """Release owned mappings before callers remove files (required on Windows)."""
        self.docs = []
        for array in self._cached_arrays:
            if not array._mmap.closed:
                array._mmap.close()
        self._cached_arrays.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def load_model(name):
        import torch
        from sentence_transformers import SentenceTransformer
        specifications = {
            'embeddinggemma': ('google/embeddinggemma-300m', '57c266a740f537b4dc058e1b0cda161fd15afa75', 2048),
            'qwen': ('Qwen/Qwen3-Embedding-0.6B', '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3', 4096),
        }
        model_id, revision, length = specifications[name]
        model = SentenceTransformer(model_id, revision=revision, device='cpu',
                                    model_kwargs={'torch_dtype': torch.float32})
        model.max_seq_length = length
        return model

    def index(self, corpus, **kwargs):
        self.doc_ids = list(corpus['id'])
        if self.cache:
            ids = json.loads((self.cache / 'document_ids.json').read_text())
            if ids != self.doc_ids:
                raise ValueError('Cached corpus IDs/order differ from MTEB corpus')
            self.docs = [self.cached(n + '_documents.npy') for n in ['embeddinggemma', 'qwen']]
        else:
            texts = [compact_large_literals(row['text']) for row in corpus]
            self.docs = []
            for name in ['embeddinggemma','qwen']:
                model = self.load_model(name)
                array = encode_checkpointed(model, texts, name+'_documents', self.output/('checkpoints' if name == 'embeddinggemma' else f'checkpoints-qwen-batch{self.qwen_batch_size}'), self.batch_size if name == 'embeddinggemma' else self.qwen_batch_size)
                del model
                gc.collect()
                self.docs.append(array)
                np.save(self.output / (name+'_documents.npy'), array)
            (self.output/'document_ids.json').write_text(json.dumps(self.doc_ids))

        if any(len(docs) != len(self.doc_ids) or not np.isfinite(docs).all() for docs in self.docs):
            raise ValueError('Invalid corpus embeddings')

    def search(self, queries, top_k, **kwargs):
        ids = list(queries['id'])
        if self.cache:
            if json.loads((self.cache / 'query_ids.json').read_text()) != ids:
                raise ValueError('Cached query IDs/order differ from MTEB queries')
            arrays = [self.cached(n + '_queries.npy') for n in ['embeddinggemma', 'qwen']]
        else:
            arrays = []
            for name in ['embeddinggemma','qwen']:
                model = self.load_model(name)
                array = encode_checkpointed(model, list(queries['text']), name+'_queries', self.output/('checkpoints' if name == 'embeddinggemma' else f'checkpoints-qwen-batch{self.qwen_batch_size}'), self.batch_size if name == 'embeddinggemma' else self.qwen_batch_size)
                del model
                gc.collect()
                arrays.append(array)
                np.save(self.output / (name+'_queries.npy'), array)
            (self.output/'query_ids.json').write_text(json.dumps(ids))

        if any(len(a) != len(ids) or not np.isfinite(a).all() for a in arrays):
            raise ValueError('Invalid query embeddings')
        output = {}
        for i, qid in enumerate(ids):
            scores = fused_scores(arrays[0][i] @ self.docs[0].T, arrays[1][i] @ self.docs[1].T)
            order = np.argsort(-scores, kind='stable')[:top_k]
            output[qid] = {self.doc_ids[int(j)]: float(scores[j]) for j in order}
        return output


def main():
    import mteb
    from mteb.tasks.retrieval.code.apps_retrieval import AppsRetrieval
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embedding-cache', type=Path)
    parser.add_argument('--output', type=Path, default=Path('cache/review/mteb'))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--qwen-batch-size', type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1 or args.qwen_batch_size < 1:
        parser.error("Batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'appsretrieval_results.json').exists():
        parser.error('Output already contains a result; use a new directory')
    with PrismSearch(args.embedding_cache, args.batch_size, args.output, args.qwen_batch_size) as model:
        start = time.perf_counter()
        result = mteb.evaluate(model, AppsRetrieval(), cache=None, co2_tracker=False,
                               prediction_folder=args.output, overwrite_strategy='always')
        payload = result.task_results[0].model_dump(mode='json')
        (args.output / 'appsretrieval_results.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
        provenance = {'method': 'mteb.evaluate with custom SearchProtocol',
            'fresh_embedding_inference': not bool(args.embedding_cache),
            'elapsed_seconds': time.perf_counter()-start, 'device': 'cpu',
            'timing_scope': 'This invocation only; interrupted inference time is excluded and matching checkpoints may be reused.',
            'batch_sizes': {'gemma': args.batch_size, 'qwen': args.qwen_batch_size},
            'fusion': '.65*zscore(Gemma cosine)+.35*zscore(Qwen cosine), across full corpus',
            'cached_embedding_sha256': model.hashes,
            'limitation': 'Cached embeddings exclude model inference time and do not prove fresh-download reproducibility.' if args.embedding_cache else 'Local CPU run; no cross-platform claim.'}
        (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
