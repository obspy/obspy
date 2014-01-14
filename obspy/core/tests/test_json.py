# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from future import standard_library  # NOQA
from future.builtins import str
from future.utils import native_str

from obspy.core.json import (Default, get_dump_kwargs, writeJSON)
from obspy.core.quakeml import readQuakeML
from obspy.core import compatibility
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
        self.c = readQuakeML(qml_file)
        self.event = self.c.events[0]

    def verify_json(self, s):
        """Test an output is a string and is JSON"""
        self.assertTrue(isinstance(s, (str, native_str)))
        j = json.loads(s)
        self.assertTrue(isinstance(j, dict))

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
        self.assertTrue(isinstance(kw['default'], Default))
        self.assertTrue(kw['default'].OMIT_NULLS)
        self.assertEqual(kw['separators'], (',', ':'))
        s1 = json.dumps(self.event, **kw)
        self.verify_json(s1)
        kw = get_dump_kwargs(minify=False, no_nulls=False)
        self.assertTrue('default' in kw)
        self.assertTrue('separators' not in kw)
        self.assertTrue(isinstance(kw['default'], Default))
        self.assertFalse(kw['default'].OMIT_NULLS)
        s2 = json.dumps(self.event, **kw)
        self.verify_json(s2)
        # Compacted version is smaller
        self.assertTrue(len(s1) < len(s2))

    def test_write_json(self):
        memfile = compatibility.StringIO()
        writeJSON(self.c, memfile)
        memfile.seek(0, 0)
        # Verify json module can load
        j = json.load(memfile)
        self.assertTrue(isinstance(j, dict))
        # Test registered method call
        memfile = compatibility.StringIO()
        self.c.write(memfile, format="json")
        memfile.seek(0, 0)
        # Verify json module can load
        j = json.load(memfile)
        self.assertTrue(isinstance(j, dict))

    def tearDown(self):
        del self.event


def suite():
    return unittest.makeSuite(JSONTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
