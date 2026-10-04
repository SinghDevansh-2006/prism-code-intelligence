import json
import tempfile
import unittest
from pathlib import Path
from verify_submission import load_rankings, validate_rankings, ndcg_at_k, reciprocal_rank

class RankingIntegrityTests(unittest.TestCase):
    def test_valid_and_metrics(self):
        validate_rankings({'q': ['a', 'b']}, {'q': {'b': 1}}, {'a', 'b'}, 2)
        self.assertAlmostEqual(reciprocal_rank(['a', 'b'], {'b': 1}, 10), .5)
        self.assertAlmostEqual(ndcg_at_k(['a', 'b'], {'b': 1}, 10), 1 / 1.584962500721156)

    def test_bad_rankings(self):
        for run in [{}, {'q': ['a', 'b'], 'extra': ['a', 'b']}, {'q': ['a', 'a']},
                    {'q': ['a', 'unknown']}, {'q': ['a', 1]}, {'q': ['a']}, {'q': 'ab'}]:
            with self.subTest(run=run), self.assertRaises(ValueError):
                validate_rankings(run, {'q': {'b': 1}}, {'a', 'b'}, 2)

    def test_duplicate_json_query_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'run.json'
            path.write_text('{"q": ["a"], "q": ["b"]}')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_rankings(path)
