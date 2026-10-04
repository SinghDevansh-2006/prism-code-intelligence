import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from fastapi.testclient import TestClient
import api
from src.agentic_engine import AgenticCodeEngine

class APIResilienceTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_invalid_requests_never_load_models(self):
        with patch.object(api.engine, 'search') as search:
            for body in [{'query': ' \n\t '}, {'query': ''}, {'query': 'x', 'top_k': 0},
                         {'query': 'x', 'top_k': 51}, {'query': 'x' * 20001}]:
                self.assertEqual(self.client.post('/search', json=body).status_code, 422)
            search.assert_not_called()

    def test_normalization_and_unavailable(self):
        with patch.object(api.engine, 'search', return_value={'results': []}) as search:
            self.assertEqual(self.client.post('/search', json={'query': '  find code  '}).status_code, 200)
            search.assert_called_once_with('find code', top_k=10, version_scope='auto', version_commit=None)
        with patch.object(api.engine, 'search', side_effect=OSError('private path')):
            response = self.client.post('/search', json={'query': 'find code'})
            self.assertEqual(response.status_code, 503)
            self.assertIn('README', response.json()['detail'])
            self.assertNotIn('private path', response.text)

    def test_concurrent_lazy_initialization(self):
        engine = AgenticCodeEngine()
        def construct(**kwargs):
            time.sleep(.03)
            return object()
        with patch('src.agentic_engine.StructuralSearchEngine', side_effect=construct) as constructor:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: engine._get_structural_engine(), range(4)))
            self.assertEqual(constructor.call_count, 1)
            self.assertTrue(all(item is results[0] for item in results))

    def test_version_scope_validation_and_forwarding(self):
        with patch.object(api.engine, 'search', return_value={'results': []}) as search:
            self.assertEqual(self.client.post('/search', json={'query':'health endpoint', 'version_scope':'unknown'}).status_code, 422)
            search.assert_not_called()
            response = self.client.post('/search', json={'query':'health endpoint', 'version_scope':'commit', 'version_commit':'a'*40})
            self.assertEqual(response.status_code, 200)
            search.assert_called_once_with('health endpoint', top_k=10, version_scope='commit', version_commit='a'*40)
