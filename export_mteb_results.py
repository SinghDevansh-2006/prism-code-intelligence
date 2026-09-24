"""Re-evaluate frozen rankings with MTEB; never reruns or modifies inference."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from datasets import Dataset, load_dataset
from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores, make_score_dict
from mteb.results.task_result import TaskResult
from mteb.tasks.retrieval.code.apps_retrieval import AppsRetrieval

ROOT = Path(__file__).resolve().parent
QRELS_REVISION = 'a4fb4d92996bcfe1e0b0af9e97ea4b70b80ec9d5'
EXPECTED_SHA = '519fe7b0ad8bb4ae80f1da5192ceea067bc1c68b74cad6328ee7ac8a2de644d4'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qrels-arrow', type=Path, help='Optional official cached test Arrow file for offline evaluation')
    args = parser.parse_args()
    source = ROOT / 'submission/AppsRetrieval_inference_results.json'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA:
        raise ValueError('Frozen rankings checksum changed')
    rankings = json.loads(source.read_text())
    dataset = (Dataset.from_file(str(args.qrels_arrow)) if args.qrels_arrow else load_dataset(
        'CoIR-Retrieval/apps-qrels', revision=QRELS_REVISION, split='test'))
    qrels = {}
    for row in dataset:
        qrels.setdefault(str(row['query_id']), {})[str(row['corpus_id'])] = int(row['score'])
    if set(rankings) != set(qrels) or len(rankings) != 3765:
        raise ValueError('Incorrect test-query coverage')
    for docs in rankings.values():
        if len(docs) != 100 or len(set(docs)) != 100 or not all(isinstance(d, str) for d in docs):
            raise ValueError('Each ranking must contain 100 unique string IDs')
    # Strictly descending ordinal scores preserve the saved order without ties.
    run = {qid: {doc: float(100-i) for i, doc in enumerate(docs)} for qid, docs in rankings.items()}
    start = time.perf_counter()
    metrics = calculate_retrieval_scores(run, qrels, [1, 3, 5, 10, 20, 100])
    scores = make_score_dict(ndcg=metrics.ndcg, _map=metrics.map, recall=metrics.recall,
        precision=metrics.precision, mrr=metrics.mrr, naucs={}, naucs_mrr={},
        hit_rate=metrics.hit_rate, task_scores={})
    scores['main_score'] = scores['ndcg_at_10']
    assert abs(scores['main_score'] - .853100) < .00001
    assert abs(scores['mrr_at_10'] - .821803) < .000001
    result = TaskResult.from_task_results(AppsRetrieval(), {'test': {'default': scores}}, time.perf_counter()-start)
    payload = result.to_dict()
    TaskResult.model_validate(payload)
    destination = ROOT / 'submission/appsretrieval_results.json'
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    TaskResult.model_validate_json(destination.read_text())
    provenance = {
        'method': 'MTEB retrieval metric evaluation of saved rankings; not a new mteb.evaluate inference run',
        'ranking_sha256': digest, 'mteb_version': result.mteb_version,
        'qrels_dataset': 'CoIR-Retrieval/apps-qrels', 'qrels_revision': QRELS_REVISION,
        'qrels_source': 'provided Arrow file' if args.qrels_arrow else 'pinned Hugging Face dataset',
        'qrels_rows_sha256': hashlib.sha256(json.dumps(qrels, sort_keys=True).encode()).hexdigest(),
        'score_semantics': '100 minus zero-based rank; confidence metrics omitted',
        'evaluation_time_scope': 'metric computation only, excludes inference and data loading',
        'result_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'queries': len(rankings), 'retrieved_documents_per_query': 100,
    }
    (destination.parent / 'appsretrieval_results.provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'file': str(destination), 'ndcg_at_10': scores['main_score'], 'mrr_at_10': scores['mrr_at_10'], 'queries': len(rankings)}, indent=2))

if __name__ == '__main__':
    main()
