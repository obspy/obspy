#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.datamark.core test suite.
"""

from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.datamark.core import readDATAMARK
import os
import unittest


class CoreTestCase(unittest.TestCase):
    """
    Test cases for libgse2 core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_readViaObsPy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = os.path.join(self.path, '10030302.00')
        # 1
        st = read(filename)
        st.verify()
        self.assertEquals(len(st), 2)
        self.assertEquals(st[0].stats.starttime,
                          UTCDateTime('2010-03-03T02:00:00.000000Z'))
        self.assertEquals(st[0].stats.endtime,
                          UTCDateTime('2010-03-03T02:00:59.990000Z'))
        self.assertEquals(st[0].stats.starttime,
                          UTCDateTime('2010-03-03T02:00:00.000000Z'))
        self.assertEquals(len(st[0]), 6000)
        self.assertAlmostEquals(st[0].stats.sampling_rate, 100.0)
        self.assertEquals(st[0].stats.channel, '0')

    def test_readViaModule(self):
        """
        Read files via obspy.datamark.core.readDATAMARK function.
        """
        filename = os.path.join(self.path, '10030302.00')
        # 1
        st = readDATAMARK(filename)
        st.verify()
        self.assertEquals(len(st), 2)
        self.assertEquals(st[0].stats.starttime,
                          UTCDateTime('2010-03-03T02:00:00.000000Z'))
        self.assertEquals(st[0].stats.endtime,
                          UTCDateTime('2010-03-03T02:00:59.990000Z'))
        self.assertEquals(st[0].stats.starttime,
                          UTCDateTime('2010-03-03T02:00:00.000000Z'))
        self.assertEquals(len(st[0]), 6000)
        self.assertAlmostEquals(st[0].stats.sampling_rate, 100.0)
        self.assertEquals(st[0].stats.channel, '0')


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
