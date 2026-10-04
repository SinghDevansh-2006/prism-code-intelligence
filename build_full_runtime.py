"""Build a separate 8,765-document runtime index from official data and embeddings."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from src.structural_metadata import extract_structural_metadata
from verify_submission import CORPUS_REVISION

ROOT = Path(__file__).resolve().parent


def build(corpus, embeddings, output):
    ids = list(corpus['_id'])
    expected = json.loads((embeddings / 'document_ids.json').read_text())
    if ids != expected or len(set(ids)) != 8765:
        raise ValueError('Official corpus/embedding IDs or order do not match')
    arrays = {}
    for name, dim in [('embeddinggemma',768),('qwen',1024)]:
        path = embeddings / f'{name}_documents.npy'
        array = np.load(path, mmap_mode='r')
        if array.shape != (8765, dim) or not np.isfinite(array).all():
            raise ValueError(f'Invalid {name} array shape or values')
        arrays[name] = path
    if output.exists():
        raise ValueError('Use a new output directory; existing indexes are never overwritten')
    output.mkdir(parents=True)
    parsed = 0
    with (output/'corpus.jsonl').open('w') as codes, (output/'structural_index.jsonl').open('w') as structures:
        for row in corpus:
            code = row['text']
            codes.write(json.dumps({'id':row['_id'],'code':code,'title':'','language':'PYTHON'})+'\n')
            metadata = extract_structural_metadata(code)
            parsed += bool(metadata['parsed'])
            structures.write(json.dumps({'id':row['_id'],**metadata})+'\n')
    config = json.loads((ROOT/'runtime_index/config.json').read_text())
    config.update(corpus_size=len(ids), corpus_scope='official-full', corpus_revision=CORPUS_REVISION)
    config['reranking']['enabled'] = False
    config.pop('validation', None)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'document_ids.json').write_text(json.dumps(ids)+'\n')
    for name,path in arrays.items(): shutil.copyfile(path,output/path.name)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    report={'corpus_documents':len(ids),'parsed_documents':parsed,'corpus_revision':CORPUS_REVISION,
            'embeddings':'Reused supplied embedding arrays; this build performs no new inference', 'sha256':hashes}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    from datasets import Dataset, load_dataset
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embedding-cache',type=Path,default=ROOT/'cache/official')
    parser.add_argument('--corpus-arrow',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'cache/runtime-full')
    args=parser.parse_args()
    corpus=Dataset.from_file(str(args.corpus_arrow)) if args.corpus_arrow else load_dataset(
        'CoIR-Retrieval/apps','corpus',revision=CORPUS_REVISION,split='corpus')
    print(json.dumps(build(corpus,args.embedding_cache,args.output),indent=2))


if __name__=='__main__':main()
