#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.io.dmx.core test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.dmx.core import _read_dmx


class CoreTestCase(unittest.TestCase):
    """
    Test cases for dmx core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_read_via_obspy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = os.path.join(self.path, '181223_120000.DMX')
        # 1
        st = read(filename)
        st.verify()
        self.assertEqual(len(st), 186)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime(2018, 12, 23, 12, 0))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime(2018, 12, 23, 12, 0, 59, 980000))
        self.assertTrue("dmx" in st[0].stats)
        self.assertEqual(len(st[0]), 3000)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 50.0)
        self.assertEqual(st[0].stats.channel, 'E')

    def test_read_via_module(self):
        """
        Read files via obspy.io.mdx.core._read_dmx function directly.
        """
        filename = os.path.join(self.path, '181223_120000.DMX')
        # 1
        st = _read_dmx(filename)
        st.verify()
        self.assertEqual(len(st), 186)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime(2018, 12, 23, 12, 0))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime(2018, 12, 23, 12, 0, 59, 980000))
        self.assertTrue("dmx" in st[0].stats)
        self.assertEqual(len(st[0]), 3000)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 50.0)
        self.assertEqual(st[0].stats.channel, 'E')
        self.assertEqual(st[0].id, 'IT.STR1..E')

    def test_read_with_station(self):
        """
        Read files and passing a station keyword argument.
        """
        filename = os.path.join(self.path, '181223_120000.DMX')
        # 1
        st = read(filename, station='EMPL')
        st.verify()

        self.assertEqual(len(st), 3)
        self.assertEqual(st[0].id, "IT.EMPL..E")
        for tr in st:
            self.assertEqual(tr.stats.station, "EMPL")


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
