# -*- coding: utf-8 -*-
import io
import json
import os
import unittest

from obspy.io.json.default import Default
from obspy.io.json.core import get_dump_kwargs, _write_json
from obspy.io.quakeml.core import _read_quakeml


class JSONTestCase(unittest.TestCase):
    """Test JSON module classes and functions"""
    def setUp(self):
        self.path = os.path.join(os.path.dirname(__file__))
        qml_file = os.path.join(self.path, "..", "..", "quakeml", "tests",
                                "data", "qml-example-1.2-RC3.xml")
        self.c = _read_quakeml(qml_file)
        self.event = self.c.events[0]

    def verify_json(self, s):
        """Test an output is a string and is JSON"""
        assert isinstance(s, str)
        j = json.loads(s)
        assert isinstance(j, dict)

    def test_default(self):
        """Test Default function class"""
        default = Default()
        assert hasattr(default, '_catalog_attrib')
        assert hasattr(default, 'OMIT_NULLS')
        assert hasattr(default, 'TIME_FORMAT')
        assert hasattr(default, '__init__')
        assert hasattr(default, '__call__')
        s = json.dumps(self.event, default=default)
        self.verify_json(s)

    def test_get_dump_kwargs(self):
        """Test getting kwargs for json.dumps"""
        kw = get_dump_kwargs()
        assert 'default' in kw
        assert 'separators' in kw
        assert isinstance(kw['default'], Default)
        assert kw['default'].OMIT_NULLS
        assert kw['separators'] == (',', ':')
        s1 = json.dumps(self.event, **kw)
        self.verify_json(s1)
        kw = get_dump_kwargs(minify=False, no_nulls=False)
        assert 'default' in kw
        assert 'separators' not in kw
        assert isinstance(kw['default'], Default)
        assert not kw['default'].OMIT_NULLS
        s2 = json.dumps(self.event, **kw)
        self.verify_json(s2)
        # Compacted version is smaller
        assert len(s1) < len(s2)

    def test_write_json(self):
        memfile = io.StringIO()
        _write_json(self.c, memfile)
        memfile.seek(0, 0)
        # Verify json module can load
        j = json.load(memfile)
        assert isinstance(j, dict)
        # Test registered method call
        memfile = io.StringIO()
        self.c.write(memfile, format="json")
        memfile.seek(0, 0)
        # Verify json module can load
        j = json.load(memfile)
        assert isinstance(j, dict)

    def tearDown(self):
        del self.event
