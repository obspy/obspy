from obspy.core.json import Default, get_dump_kwargs
from obspy.core.quakeml import readQuakeML
import os
import unittest
import warnings
import json

warnings.filterwarnings("ignore")


class JSONTestCase(unittest.TestCase):
    """Test JSON module classes and functions"""
    def setUp(self):
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        qml_file = os.path.join(self.path, 'qml-example-1.2-RC3.xml')
        c = readQuakeML(qml_file)
        self.event = c.events[0]

    def verify_json(self, s):
        """Test an output is a string and is JSON"""
        self.assertIsInstance(s, str)
        j = json.loads(s)
        self.assertIsInstance(j, dict)

    def test_default(self):
        """Test Default function class"""
        default = Default()
        self.assertTrue(hasattr(default, '_catalog_attrib'))
        self.assertTrue(hasattr(default, 'OMIT_NULLS'))
        self.assertTrue(hasattr(default, 'TIME_FORMAT'))
        self.assertTrue(hasattr(default, '__init__'))
        self.assertTrue(hasattr(default, '__call__'))
        s = json.dumps(self.event, default=default)
        self.verify_json(s)

    def test_get_dump_kwargs(self):
        """Test getting kwargs for json.dumps"""
        kw = get_dump_kwargs()
        self.assertTrue('default' in kw)
        self.assertTrue('separators' in kw)
        self.assertIsInstance(kw['default'], Default)
        self.assertTrue(kw['default'].OMIT_NULLS)
        self.assertEqual(kw['separators'], (',', ':'))
        s1 = json.dumps(self.event, **kw)
        self.verify_json(s1)
        kw = get_dump_kwargs(minify=False, no_nulls=False)
        self.assertTrue('default' in kw)
        self.assertTrue('separators' not in kw)
        self.assertIsInstance(kw['default'], Default)
        self.assertFalse(kw['default'].OMIT_NULLS)
        s2 = json.dumps(self.event, **kw)
        self.verify_json(s2)
        # Compacted version is smaller
        self.assertLess(len(s1), len(s2))

    def tearDown(self):
        del self.event


def suite():
    return unittest.makeSuite(JSONTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
