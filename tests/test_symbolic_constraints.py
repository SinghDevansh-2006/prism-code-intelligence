import unittest
from src.structural_search import StructuralSearchEngine

class SymbolicConstraintsTests(unittest.TestCase):
    def setUp(self):
        self.engine = StructuralSearchEngine.__new__(StructuralSearchEngine)
        self.engine.records = [dict(id='a', functions=[], classes=[], imports=[], calls=[dict(name='collections', line=1)], identifiers=['collections']),
            dict(id='b', functions=[], classes=[], imports=[dict(name='collections.defaultdict', line=2)], calls=[], identifiers=[])]
        self.engine.corpus = {i: dict(code='', title='', language='python') for i in ['a','b']}

    def test_import_does_not_match_call_or_identifier(self):
        results = self.engine.search('which files import collections?')['results']
        self.assertEqual([r['id'] for r in results], ['b'])
        self.assertEqual(results[0]['evidence'][0]['type'], 'import')

    def test_identifier_boundaries_and_case(self):
        match = self.engine._matches
        self.assertFalse(match('sprint', 'print'))
        self.assertFalse(match('MyClass', 'myclass'))
        self.assertTrue(match('obj.print', 'print'))

    def test_unsupported_graph_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'not implemented'):
            self.engine.search('show dependency graph')

    def test_multiple_imports_require_all(self):
        self.assertEqual(self.engine.search('which files import collections and math?')['results'], [])

    def test_single_letter_identifier(self):
        from src.query_router import route_query
        self.assertEqual(route_query('which functions call f?').extracted_terms, ['f'])
