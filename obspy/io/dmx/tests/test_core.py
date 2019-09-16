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
        filename = os.path.join(self.path, '131114_090600.dmx')
        # 1
        st = read(filename)
        st.verify()
        self.assertEqual(len(st), 2)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime(2013, 11, 14, 9, 6))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime(2013, 11, 14, 9, 6, 59, 990000))
        self.assertTrue("dmx" in st[0].stats)
        self.assertEqual(len(st[0]), 6000)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 100.0)
        self.assertEqual(st[0].stats.channel, 'Z')
        self.assertEqual(st[0].id, 'ETNA.EMFO..Z')

    def test_read_via_module(self):
        """
        Read files via obspy.io.mdx.core._read_dmx function directly.
        """
        filename = os.path.join(self.path, '131114_090600.dmx')
        # 1
        st = _read_dmx(filename)
        st.verify()
        self.assertEqual(len(st), 2)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime(2013, 11, 14, 9, 6))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime(2013, 11, 14, 9, 6, 59, 990000))
        self.assertTrue("dmx" in st[0].stats)
        self.assertEqual(len(st[0]), 6000)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 100.0)
        self.assertEqual(st[0].stats.channel, 'Z')
        self.assertEqual(st[0].id, 'ETNA.EMFO..Z')

    def test_read_with_station(self):
        """
        Read files and passing a station keyword argument.
        """
        filename = os.path.join(self.path, '131114_090600.dmx')
        # 1
        st = read(filename, station='EMPL')
        st.verify()

        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].id, "ETNA.EMPL..Z")
        for tr in st:
            self.assertEqual(tr.stats.station, "EMPL")


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
