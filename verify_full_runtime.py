"""Check the full-corpus API and Git snapshot modes against real indexed data."""
import argparse
import json
import time
from pathlib import Path
from datasets import Dataset
from fastapi.testclient import TestClient
import api
from src.query_router import route_query


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries-arrow',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('cache/review/full-runtime-verification.json'))
    args=parser.parse_args();client=TestClient(api.app)
    health=client.get('/health').json();assert health['corpus_documents']==8765,health
    frozen=json.loads((api.ROOT/'submission/AppsRetrieval_inference_results.json').read_text())
    examples=[]
    for row in Dataset.from_file(str(args.queries_arrow)):
        qid=row['_id'];query=row['text']
        if qid not in frozen or len(query)>20000 or route_query(query).route!='semantic':continue
        expected=frozen[qid][0]
        if int(expected[1:])<=5000:continue
        start=time.perf_counter();response=client.post('/search',json={'query':query,'top_k':5})
        assert response.status_code==200,response.text
        results=response.json()['results']
        examples.append({'query_id':qid,'frozen_top1':expected,'api_top1':results[0]['id'],
                         'matches_frozen_top1':results[0]['id']==expected,'elapsed_ms':(time.perf_counter()-start)*1000})
        if len(examples)==5:break
    assert len(examples)==5
    snapshots=client.get('/versions').json()['commits'];assert snapshots
    scopes=[]
    for scope,commit in [('all',None),('latest',None),('commit',snapshots[0])]:
        response=client.post('/search',json={'query':'find the API health endpoint','version_scope':scope,'version_commit':commit})
        assert response.status_code==200,response.text
        payload=response.json();assert payload['plan']['route']=='evolution' and payload['results']
        scopes.append({'scope':scope,'commit':commit,'versions':[r['version'] for r in payload['results']]})
    assert client.post('/search',json={'query':'API','version_scope':'commit','version_commit':'invalid'}).status_code==422
    report={'health':health,'semantic_examples':examples,'version_scopes':scopes,
            'scope':'Five integration/parity examples, not a new accuracy benchmark'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
