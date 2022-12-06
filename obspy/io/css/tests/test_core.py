#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.io.css.core test suite.
"""
import gzip
import os
import unittest

import numpy as np

from obspy import read
from obspy.core import Stream, Trace, UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.io.css.core import (_is_css, _read_css, _is_nnsa_kb_core,
                               _read_nnsa_kb_core)


class CoreTestCase(unittest.TestCase):
    """
    Test cases for css core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        self.filename_css = os.path.join(self.path, 'test_css.wfdisc')
        self.filename_nnsa = os.path.join(self.path, 'test_nnsa.wfdisc')
        self.filename_css_2 = os.path.join(self.path, 'test_css_2.wfdisc')
        self.filename_css_3 = os.path.join(self.path, 'test_css_3.wfdisc')
        # set up stream for validation
        header = {}
        header['station'] = 'TEST'
        header['starttime'] = UTCDateTime(1296474900.0)
        header['sampling_rate'] = 80.0
        header['calib'] = 1.0
        header['calper'] = 1.0
        header['_format'] = 'CSS'
        filename = os.path.join(self.path, '201101311155.10.ascii.gz')
        with gzip.open(filename, 'rb') as fp:
            data = np.loadtxt(fp, dtype=np.int_)
        # traces in the test files are sorted ZEN
        st = Stream()
        for x, cha in zip(data.reshape((3, 4800)), ('HHZ', 'HHE', 'HHN')):
            # big-endian copy
            tr = Trace(x, header.copy())
            tr.stats.station += 'be'
            tr.stats.channel = cha
            st += tr
            # little-endian copy
            tr = Trace(x, header.copy())
            tr.stats.station += 'le'
            tr.stats.channel = cha
            st += tr
        self.st_result_css = st.copy()
        for tr in st:
            tr.stats['_format'] = "NNSA_KB_CORE"
        self.st_result_nnsa = st

    def test_is_css(self):
        """
        Read files via obspy.core.stream.read function.
        """
        # 1
        assert _is_css(self.filename_css)
        # check that empty files are not recognized as CSS
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            fh = open(tempfile, "wb")
            fh.close()
            assert not _is_css(tempfile)

    def test_is_nnsa_kb_core(self):
        """
        Read files via obspy.core.stream.read function.
        """
        # 1
        assert _is_nnsa_kb_core(self.filename_nnsa)
        # check that empty files are not recognized as NNSA_KB_CORE
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            fh = open(tempfile, "wb")
            fh.close()
            assert not _is_nnsa_kb_core(tempfile)

    def test_is_not_this_format_core(self):
        # check that NNSA files are not recognized as CSS
        assert not _is_css(self.filename_nnsa)
        # check that CSS file is not recognized as NNSA_KB_CORE
        assert not _is_nnsa_kb_core(self.filename_css)

    def test_css_read_via_obspy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        # 1
        st = read(self.filename_css)
        self.assertEqual(st, self.st_result_css)

    def test_css_read_via_module(self):
        """
        Read files via obspy.io.css.core._read_css function.
        """
        # 1
        st = _read_css(self.filename_css)
        # _format entry is not present when using low-level function
        for tr in self.st_result_css:
            tr.stats.pop('_format')
        self.assertEqual(st, self.st_result_css)

    def test_nnsa_kb_core_read_via_obspy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        # 1
        st = read(self.filename_nnsa)
        self.assertEqual(st, self.st_result_nnsa)

    def test_nnsa_kb_core_read_via_module(self):
        """
        Read files via obspy.io.css.core._read_nnsa_kb_core function.
        """
        # 1
        st = _read_nnsa_kb_core(self.filename_nnsa)
        # _format entry is not present when using low-level function
        for tr in self.st_result_nnsa:
            tr.stats.pop('_format')
        self.assertEqual(st, self.st_result_nnsa)

    def test_css_2_read_via_module(self):
        """
        Read files via obspy.io.css.core._read_css function.
        Read gzipped waveforms.
        """
        # 1
        st = _read_css(self.filename_css_2)
        # _format entry is not present when using low-level function
        for tr in self.st_result_css:
            tr.stats.pop('_format')
        self.assertEqual(st, self.st_result_css)

    def test_css_3_read_via_module(self):
        """
        Read files via obspy.io.css.core._read_css function.
        Exception if waveform file is missing.
        """
        # 1
        self.assertRaises(FileNotFoundError, _read_css, self.filename_css_3)
