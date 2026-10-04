"""Repair the audited 256 Qwen rows; preserve the original run and record reuse."""
import gc
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from evaluate_mteb_pipeline import PrismSearch
from src.code_normalizer import compact_large_literals
from src.embedding_checkpoint import encode_checkpointed

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'cache/review/mteb-fresh-complete'
OUTPUT=ROOT/'cache/review/mteb-uniform-arrays'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    audit=json.loads((ROOT/'docs/review/mteb-fresh-audit.json').read_text())
    for name, expected in audit['sha256'].items():
        if sha(SOURCE/name)!=expected:raise ValueError(f'Original evidence changed: {name}')
    rows=[json.loads(line) for line in (ROOT/'cache/runtime-full/corpus.jsonl').read_text().splitlines()]
    ids=json.loads((SOURCE/'document_ids.json').read_text())
    assert [row['id'] for row in rows]==ids
    texts=[compact_large_literals(row['code']) for row in rows]
    old=json.loads((SOURCE/'checkpoints-qwen-batch1/qwen_documents.json').read_text())
    signature=hashlib.sha256(json.dumps({'texts':texts,'revision':old['model_revision'],'max_length':4096,'name':'qwen_documents','batch_size':1},ensure_ascii=False).encode()).hexdigest()
    assert signature==old['signature'], 'Corpus or original configuration changed'
    order=np.argsort([-len(text) for text in texts],kind='stable')
    affected=order[:256]
    qwen=np.load(SOURCE/'qwen_documents.npy',allow_pickle=False).copy()
    assert np.allclose(np.linalg.norm(qwen[order[256:]],axis=1),1,atol=1e-5)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    model=PrismSearch.load_model('qwen')
    assert str(model.dtype)=='torch.float32' and str(model.device)=='cpu'
    repaired=encode_checkpointed(model,[texts[int(i)] for i in affected], 'qwen_documents', OUTPUT/'repair-checkpoints',1)
    del model;gc.collect()
    qwen[affected]=repaired
    assert qwen.shape==(8765,1024) and np.isfinite(qwen).all()
    assert np.allclose(np.linalg.norm(qwen,axis=1),1,atol=1e-5)
    np.save(OUTPUT/'qwen_documents.npy',qwen)
    reused=['embeddinggemma_documents.npy','embeddinggemma_queries.npy','qwen_queries.npy']
    for name in reused:
        a=np.load(SOURCE/name,allow_pickle=False)
        assert a.dtype==np.float32 and np.isfinite(a).all()
        assert np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5)
        shutil.copyfile(SOURCE/name,OUTPUT/name)
    for name in ['document_ids.json','query_ids.json']:shutil.copyfile(SOURCE/name,OUTPUT/name)
    manifest={'method':'Explicit float32 CPU repair of audited mixed-precision rows, followed by complete MTEB rescoring',
        'regenerated_qwen_document_rows':len(affected),'regenerated_document_ids':[ids[int(i)] for i in affected],
        'reused_qwen_document_rows':len(ids)-len(affected),'reused_arrays':reused,
        'reuse_basis':'All retained arrays/rows were generated with explicit float32 in the preceding fresh CPU run; they are not the original submitted benchmark cache. Source hashes and document order checked.',
        'dtype':'torch.float32','device':'cpu','source_sha256':audit['sha256'],
        'corrected_sha256':{p.name:sha(p) for p in OUTPUT.iterdir() if p.is_file() and p.suffix in {'.npy','.json'} and p.name!='repair-provenance.json'}}
    (OUTPUT/'repair-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Uniform float32 arrays verified; ready for full MTEB evaluation.',flush=True)

if __name__=='__main__':main()
