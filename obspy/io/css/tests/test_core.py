#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.io.css.core test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import gzip
import os
import unittest

import numpy as np

from obspy import read
from obspy.core import Stream, Trace, UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.io.css.core import _is_css, _read_css


class CoreTestCase(unittest.TestCase):
    """
    Test cases for css core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(self.path, 'test.wfdisc')
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
        self.st_result = st

    def test_is_css(self):
        """
        Read files via obspy.core.stream.read function.
        """
        # 1
        assert(_is_css(self.filename))
        # check that empty files are not recognized as CSS
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            fh = open(tempfile, "wb")
            fh.close()
            assert(not _is_css(tempfile))

    def test_read_via_obspy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        # 1
        st = read(self.filename)
        self.assertEqual(st, self.st_result)

    def test_read_via_module(self):
        """
        Read files via obspy.io.css.core._read_css function.
        """
        # 1
        st = _read_css(self.filename)
        # _format entry is not present when using low-level function
        for tr in self.st_result:
            tr.stats.pop('_format')
        self.assertEqual(st, self.st_result)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
