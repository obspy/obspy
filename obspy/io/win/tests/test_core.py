#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.io.win.core test suite.
"""
import os
import unittest

from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.win.core import _read_win


class CoreTestCase(unittest.TestCase):
    """
    Test cases for win core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_read_via_obspy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = os.path.join(self.path, '10030302.00')
        # 1
        st = read(filename)
        st.verify()
        st.sort(keys=['channel'])
        self.assertEqual(len(st), 2)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2010-03-03T02:00:00.000000Z'))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime('2010-03-03T02:00:59.990000Z'))
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2010-03-03T02:00:00.000000Z'))
        self.assertEqual(len(st[0]), 6000)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 100.0)
        self.assertEqual(st[0].stats.channel, 'a100')

    def test_read_via_module(self):
        """
        Read files via obspy.io.win.core._read_win function.
        """
        filename = os.path.join(self.path, '10030302.00')
        # 1
        st = _read_win(filename)
        st.verify()
        st.sort(keys=['channel'])
        self.assertEqual(len(st), 2)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2010-03-03T02:00:00.000000Z'))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime('2010-03-03T02:00:59.990000Z'))
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2010-03-03T02:00:00.000000Z'))
        self.assertEqual(len(st[0]), 6000)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 100.0)
        self.assertEqual(st[0].stats.channel, 'a100')
