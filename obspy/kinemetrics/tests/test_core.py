#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.kinemetrics.core test suite.
"""
import os
import unittest

from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.kinemetrics.core import is_evt, read_evt


class CoreTestCase(unittest.TestCase):
    """
    Test cases for kinemetrics core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_is_evt(self):
        """
        Test for the the is_evt() function.
        """
        valid_files = [os.path.join(self.path, "BI008_MEMA-04823.kinemetrics"),
                       os.path.join(self.path, "BX456_MOLA-02351.kinemetrics")]
        invalid_files = []
        py_dir = os.path.join(self.path, os.pardir, os.pardir)
        for filename in os.listdir(py_dir):
            if filename.endswith(".py"):
                invalid_files.append(
                    os.path.abspath(os.path.join(py_dir, filename)))
        self.assertTrue(len(invalid_files) > 0)

        for filename in valid_files:
            self.assertTrue(is_evt(filename))
        for filename in invalid_files:
            self.assertFalse(is_evt(filename))

    def test_read_via_ObsPy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = os.path.join(self.path, 'BI008_MEMA-04823.kinemetrics')
        # 1
        st = read(filename)
        st.verify()
        self.assertEqual(len(st), 3)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(len(st[0]), 230*25)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 250.0)
        self.assertEqual(st[0].stats.channel, '0')
        self.assertEqual(st[0].stats.station, 'MEMA')

        filename = os.path.join(self.path, 'BX456_MOLA-02351.kinemetrics')
        # 2
        st = read(filename)
        st.verify()
        self.assertEqual(len(st), 6)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[3].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[4].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[5].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(len(st[0]), 390*25)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 250.0)
        self.assertEqual(st[0].stats.channel, '0')
        self.assertEqual(st[0].stats.station, 'MOLA')

    def test_read_via_module(self):
        """
        Read files via obspy.kinemetrics.core.read_evt function.
        """
        filename = os.path.join(self.path, 'BI008_MEMA-04823.kinemetrics')
        # 1
        st = read_evt(filename)
        st.verify()
        self.assertEqual(len(st), 3)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2013-08-15T09:20:28.000000Z'))
        self.assertEqual(len(st[0]), 230*25)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 250.0)
        self.assertEqual(st[0].stats.channel, '0')
        self.assertEqual(st[0].stats.station, 'MEMA')

        filename = os.path.join(self.path, 'BX456_MOLA-02351.kinemetrics')
        # 2
        st = read_evt(filename)
        st.verify()
        self.assertEqual(len(st), 6)
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[1].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[2].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[3].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[4].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(st[5].stats.starttime,
                         UTCDateTime('2012-01-17T09:54:36.000000Z'))
        self.assertEqual(len(st[0]), 390*25)
        self.assertAlmostEqual(st[0].stats.sampling_rate, 250.0)
        self.assertEqual(st[0].stats.channel, '0')
        self.assertEqual(st[0].stats.station, 'MOLA')


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
