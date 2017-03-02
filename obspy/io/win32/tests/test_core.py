#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.io.win32.core test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.win32.core import _read_win32


class CoreTestCase(unittest.TestCase):
    """
    Test cases for datamark core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_read_via_obspy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = os.path.join(self.path, '2014010100000101VM.cnt')
        ct_filename = os.path.join(self.path, '01_01_20140101.euc.ch')
        # 1
        st = read(filename, channel_table_filename=ct_filename)
        st.sort(keys=['channel'])
        self.assertEqual(len(st), 3)
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2014-01-01T00:00:00.000000Z'))
        self.assertEqual(st[2].stats.endtime,
                         UTCDateTime('2014-01-01T00:00:59.990000Z'))
        self.assertEqual(len(st[2]), 6000)
        self.assertAlmostEqual(st[2].stats.sampling_rate, 100.0)
        self.assertEqual(st[2].stats.win32.channel_id, '77e3')
        self.assertAlmostEqual(st[2].max(), 45710)

    def test_read_via_module(self):
        """
        Read files via obspy.io.datamark.core._read_datamark function.
        """
        filename = os.path.join(self.path, '2014010100000101VM.cnt')
        ct_filename = os.path.join(self.path, '01_01_20140101.euc.ch')
        # 1
        st = _read_win32(filename, channel_table_filename=ct_filename)
        st.sort(keys=['channel'])
        self.assertEqual(len(st), 3)
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2014-01-01T00:00:00.000000Z'))
        self.assertEqual(st[2].stats.endtime,
                         UTCDateTime('2014-01-01T00:00:59.990000Z'))
        self.assertEqual(len(st[2]), 6000)
        self.assertAlmostEqual(st[2].stats.sampling_rate, 100.0)
        self.assertEqual(st[2].stats.win32.channel_id, '77e3')
        self.assertAlmostEqual(st[2].max(), 45710)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
