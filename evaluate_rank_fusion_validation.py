"""One pre-specified validation-only candidate; never reads official test labels."""
import json
import time
from pathlib import Path
import numpy as np
from datasets import Dataset

ROOT=Path(__file__).resolve().parent

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries-arrow',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'cache/review/rank-fusion-validation.json')
    args=parser.parse_args()
    protocol={'candidate':'0.65/(60+Gemma rank)+0.35/(60+Qwen rank)',
        'selection':'Candidate requires positive lower 95% paired-bootstrap NDCG delta and no MRR@100 decrease',
        'data':'Existing 1000-query validation from training partition; previously used for development, not a new untouched holdout',
        'official_test_labels_used':False,'bootstrap_seed':20261002,'bootstrap_samples':2000}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    split=json.loads((ROOT/'data/split.json').read_text()); relevant={x['query_id']:x['doc_id'] for x in split['validation']}
    rows=[r for r in Dataset.from_file(str(args.queries_arrow)) if r['partition']=='train' and r['_id'] in relevant]
    doc_ids=json.loads((ROOT/'runtime_index/document_ids.json').read_text()); lookup={d:i for i,d in enumerate(doc_ids)}
    target=np.array([lookup[relevant[r['_id']]] for r in rows])
    g=np.load(ROOT/'cache/embeddinggemma/queries.npy') @ np.load(ROOT/'cache/embeddinggemma/documents.npy').T
    q=np.load(ROOT/'cache/qwen06b/queries.npy') @ np.load(ROOT/'cache/qwen06b/documents.npy').T
    if g.shape != (1000,5000) or q.shape != g.shape: raise ValueError('Unexpected validation cache dimensions')
    def z(x):return (x-x.mean(axis=1,keepdims=True))/(x.std(axis=1,keepdims=True)+1e-8)
    def ranks(x):
        order=np.argsort(-x,axis=1,kind='stable');out=np.empty_like(order);np.put_along_axis(out,order,np.arange(1,x.shape[1]+1)[None,:],axis=1);return out
    start=time.perf_counter();base=ranks(.65*z(g)+.35*z(q));baseline_seconds=time.perf_counter()-start
    start=time.perf_counter();candidate=ranks(.65/(60+ranks(g))+.35/(60+ranks(q)));candidate_seconds=time.perf_counter()-start
    def metrics(rank):
        pos=rank[np.arange(len(target)),target]; ndcg=np.where(pos<=10,1/np.log2(pos+1),0);mrr=np.where(pos<=100,1/pos,0)
        return ndcg,mrr
    bn,bm=metrics(base);cn,cm=metrics(candidate)
    if abs(bn.mean()-.9177) > .0002:raise ValueError('Baseline did not reproduce; investigate cache order before trusting candidate')
    diff=cn-bn;rng=np.random.default_rng(protocol['bootstrap_seed']);boots=[rng.choice(diff,len(diff),replace=True).mean() for _ in range(2000)]
    ci=np.quantile(boots,[.025,.975]);promote=bool(ci[0]>0 and cm.mean()>=bm.mean())
    report={**protocol,'baseline':{'ndcg_at_10':float(bn.mean()),'mrr_at_100':float(bm.mean()),'fusion_sort_seconds':baseline_seconds},
        'candidate_result':{'ndcg_at_10':float(cn.mean()),'mrr_at_100':float(cm.mean()),'fusion_sort_seconds':candidate_seconds},
        'ndcg_delta_95pct_interval':ci.tolist(),'passes_predeclared_gate':promote,
        'deployment':'No runtime or submission changes made by this experiment'}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
