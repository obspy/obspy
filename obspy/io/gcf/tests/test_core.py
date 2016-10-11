#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.io.gcf.core test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import numpy as np

from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.gcf.core import _read_gcf, merge_gcf_stream


EXPECTED = np.array([-49378, -49213, -49273, -49277, -49341, -49415, -49289,
                     -49309, -49277, -49381, -49441, -49276, -49331, -49268,
                     -49250, -49407, -49421, -49282, -49224, -49281],
                    dtype=np.int32)


class CoreTestCase(unittest.TestCase):
    """
    Test cases for gcf core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_read_via_obspy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = os.path.join(self.path, '20160603_1955n.gcf')
        # 1
        st = read(filename)
        st.verify()
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2016-06-03T19:55:00.000000Z'))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime('2016-06-03T19:55:02.990000Z'))
        self.assertEqual(len(st[0]), 300)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 100.0)
        self.assertEqual(st[0].stats.channel, 'HHN')
        self.assertEqual(st[0].stats.station, '6018')
        np.testing.assert_array_equal(EXPECTED, st[0].data[:20])

    def test_read_head_via_obspy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = os.path.join(self.path, '20160603_1955n.gcf')
        # 1
        st = read(filename, headonly=True)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2016-06-03T19:55:00.000000Z'))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime('2016-06-03T19:55:02.990000Z'))
        self.assertEqual(st[0].stats.npts, 300)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 100.0)
        self.assertEqual(st[0].stats.channel, 'HHN')
        self.assertEqual(st[0].stats.station, '6018')

    def test_read_via_module(self):
        """
        Read files via obspy.io.gcf.core._read_gcf function.
        """
        filename = os.path.join(self.path, '20160603_1955n.gcf')
        # 1
        st = _read_gcf(filename)
        st.verify()
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2016-06-03T19:55:00.000000Z'))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime('2016-06-03T19:55:02.990000Z'))
        self.assertEqual(len(st[0]), 300)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 100.0)
        self.assertEqual(st[0].stats.channel, 'HHN')
        self.assertEqual(st[0].stats.station, '6018')
        np.testing.assert_array_equal(EXPECTED, st[0].data[:20])

    def test_read_head_via_module(self):
        """
        Read files via obspy.io.gcf.core._read_gcf function.
        """
        filename = os.path.join(self.path, '20160603_1955n.gcf')
        # 1
        st = _read_gcf(filename, headonly=True)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2016-06-03T19:55:00.000000Z'))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime('2016-06-03T19:55:02.990000Z'))
        self.assertEqual(st[0].stats.npts, 300)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 100.0)
        self.assertEqual(st[0].stats.channel, 'HHN')
        self.assertEqual(st[0].stats.station, '6018')

    def test_read_channel_prefix_via_obspy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = os.path.join(self.path, '20160603_1955n.gcf')
        # 1
        st = read(filename, headonly=True, channel_prefix="HN")
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2016-06-03T19:55:00.000000Z'))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime('2016-06-03T19:55:02.990000Z'))
        self.assertEqual(st[0].stats.npts, 300)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 100.0)
        self.assertEqual(st[0].stats.channel, 'HNN')
        self.assertEqual(st[0].stats.station, '6018')

    def test_merge_gcf_stream(self):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = os.path.join(self.path, '20160603_1955n.gcf')
        # 1
        st1 = read(filename, headonly=True, channel_prefix="HN")
        st2 = st1.copy()
        tr1 = st1[0].copy()
        tr2 = st1[0].copy()
        tr1.stats.starttime = UTCDateTime('2016-06-03T19:55:02.000000Z')
        tr1.stats.npts = 100
        tr2.stats.npts = 200
        st2.traces = [tr1, tr2]
        st2 = merge_gcf_stream(st2)
        self.assertEqual(len(st1), len(st2))
        self.assertEqual(st2[0].stats.starttime,
                         UTCDateTime('2016-06-03T19:55:00.000000Z'))
        self.assertEqual(st2[0].stats.endtime,
                         UTCDateTime('2016-06-03T19:55:02.990000Z'))
        self.assertEqual(st2[0].stats.npts, 300)
        self.assertAlmostEqual(st2[0].stats.sampling_rate, 100.0)
        self.assertEqual(st2[0].stats.channel, 'HNN')
        self.assertEqual(st2[0].stats.station, '6018')

    def test_sps_d(self):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = os.path.join(self.path, '20160603_1910n.gcf')
        # 1
        st = read(filename, headonly=True, channel_prefix="HN")
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2016-06-03T19:10:00.000000Z'))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime('2016-06-03T19:10:01.998000Z'))
        self.assertEqual(st[0].stats.npts, 1000)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 500.0)
        self.assertEqual(st[0].stats.channel, 'HNN')
        self.assertEqual(st[0].stats.station, '6018')


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
