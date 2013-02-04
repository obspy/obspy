#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.css.core test suite.
"""

from obspy import read
from obspy.core import UTCDateTime, Trace, Stream
from obspy.core.util import NamedTemporaryFile
from obspy.css.core import readCSS, isCSS
import os
import numpy as np
import unittest


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
        data = np.loadtxt(filename, dtype='int')
        # traces in the test files are sorted ZEN
        st = Stream()
        for x, cha in zip(data.reshape((3, 4800)), ('HHZ', 'HHE', 'HHN')):
            tr = Trace(x, header.copy())
            tr.stats.channel = cha
            st += tr
        self.st_result = st

    def test_isCSS(self):
        """
        Read files via obspy.core.stream.read function.
        """
        # 1
        assert(isCSS(self.filename))
        # check that empty files are not recognized as CSS
        tempfile = NamedTemporaryFile().name
        with open(tempfile, "wb") as fh:
            pass
        try:
            assert(not isCSS(tempfile))
        finally:
            # cleanup
            os.remove(tempfile)

    def test_readViaObsPy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        # 1
        st = read(self.filename)
        self.assertTrue(st == self.st_result)

    def test_readViaModule(self):
        """
        Read files via obspy.css.core.readCSS function.
        """
        # 1
        st = readCSS(self.filename)
        # _format entry is not present when using low-level function
        for tr in self.st_result:
            tr.stats.pop('_format')
        self.assertTrue(st == self.st_result)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
