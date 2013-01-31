#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.css.core test suite.
"""

from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.css.core import readCSS, isCSS
import os
import unittest


class CoreTestCase(unittest.TestCase):
    """
    Test cases for css core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(self.path, 'test.wfdisc')
        self.filename2 = os.path.join(self.path, '201101311155.10.ascii.gz')

    def test_isCSS(self):
        """
        Read files via obspy.core.stream.read function.
        """
        # 1
        assert(isCSS(self.filename))

    def test_readViaObsPy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        # 1
        st = read(self.filename)
        st.verify()
        self.assertEquals(len(st), 3)
        for tr in st:
            self.assertEquals(tr.stats.starttime,
                              UTCDateTime(2011, 1, 31, 11, 55))
            self.assertEquals(len(tr), 4800)
            self.assertEquals(tr.stats.sampling_rate, 80.0)
            self.assertEquals(tr.stats.channel[:-1], 'HH')
        # XXX also test data here

    def test_readViaModule(self):
        """
        Read files via obspy.css.core.readCSS function.
        """
        # 1
        st = readCSS(self.filename)
        st.verify()
        self.assertEquals(len(st), 3)
        for tr in st:
            self.assertEquals(tr.stats.starttime,
                              UTCDateTime(2011, 1, 31, 11, 55))
            self.assertEquals(len(tr), 4800)
            self.assertEquals(tr.stats.sampling_rate, 80.0)
            self.assertEquals(tr.stats.channel[:-1], 'HH')
        # XXX also test data here


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
