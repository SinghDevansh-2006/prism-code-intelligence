"""Resumable CPU embedding generation; never reuses unrelated experiment arrays."""
import hashlib
import json
from pathlib import Path
import numpy as np


def encode_checkpointed(model, texts, name, root, batch_size):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    revision = getattr(model[0].auto_model.config, '_commit_hash', None)
    signature = hashlib.sha256(json.dumps({'texts':texts,'revision':revision,
        'max_length':model.max_seq_length, 'name':name, 'batch_size':batch_size,
        'dtype':str(model.dtype), 'device':str(model.device), 'format_version':2},ensure_ascii=False).encode()).hexdigest()
    manifest = root / (name+'.json')
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError('Embedding checkpoint inputs/config changed; choose a new output directory')
    manifest.write_text(json.dumps({'signature':signature,'model_revision':revision,'rows':len(texts),'dtype':str(model.dtype),'device':str(model.device),'format_version':2})+'\n')
    order = np.argsort([-len(text) for text in texts], kind='stable')
    chunks=[]; reused=0
    method = model.encode_query if name.endswith('_queries') else model.encode_document
    for start in range(0,len(texts),64):
        ids=order[start:start+64]
        path=root/f'{name}.{start:06d}.npy'
        if path.exists():
            array=np.load(path,allow_pickle=False);reused+=len(ids)
        else:
            array=method([texts[int(i)] for i in ids], batch_size=batch_size,
                normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
            tmp=path.with_suffix('.tmp.npy');np.save(tmp,array);tmp.replace(path)
        if len(array)!=len(ids) or not np.isfinite(array).all():
            raise ValueError('Invalid embedding checkpoint')
        chunks.append((ids,array))
        print(f'{name}: {start+len(ids)}/{len(texts)} rows ({reused} resumed)',flush=True)
    output=np.empty((len(texts),chunks[0][1].shape[1]),dtype=np.float32)
    for ids,array in chunks:output[ids]=array
    return output
