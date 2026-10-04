"""Small CPU robustness/latency sample; not an accuracy or generalization benchmark."""
import argparse
import json
import platform
import time
from pathlib import Path
from fastapi.testclient import TestClient
def peak_rss_mb():
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if platform.system() == 'Darwin' else 1024)

import api

CASES = [
    ('semantic', 'Find an algorithm that merges overlapping intervals'),
    ('semantic', 'Find code that checks whether parentheses are balanced'),
    ('semantic', 'Find shortest paths in a weighted graph'),
    ('exact_usage', 'which files import collections?'),
    ('exact_usage', 'which functions call sorted?'),
    ('structural', 'which functions call range before print?'),
    ('structural', 'find recursive functions'),
    ('evolution', 'show binary search version history'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('cache/review/cpu-smoke.json'))
    args = parser.parse_args()
    client = TestClient(api.app)
    records = []
    for route, query in CASES:
        for iteration in range(2):
            start = time.perf_counter()
            response = client.post('/search', json={'query': query, 'top_k': 5})
            elapsed = (time.perf_counter()-start)*1000
            payload = response.json()
            if response.status_code != 200 or payload['plan']['route'] != route or not payload['results']:
                raise RuntimeError(f'{route} failed: {payload}')
            records.append({'query': query, 'route': route, 'iteration': iteration,
                'elapsed_ms': elapsed, 'process_peak_rss_mb': peak_rss_mb(),
                'top_result': {k: v for k,v in payload['results'][0].items() if k not in ('code','timeline')},
                'result_ids': [r.get('id', r.get('logical_id')) for r in payload['results']]})
    for query, status in [('   ', 422), ('show dependency graph', 422)]:
        assert client.post('/search', json={'query': query}).status_code == status
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'platform': platform.platform(), 'machine': platform.machine(),
        'python': platform.python_version(), 'device': api.DEVICE,
        'model_cache': 'existing local cache; no fresh-download claim',
        'accuracy': 'Not scored: these queries have no independent relevance labels',
        'memory': 'Process lifetime peak RSS; unavailable on Windows',
        'records': records}, indent=2)+'\n')
    print(f'{len(records)} successful searches; output: {args.output}')


if __name__ == '__main__': main()
