#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.datamark.core test suite.
"""

from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.evt.core import readEVT
import os
import unittest


class CoreTestCase(unittest.TestCase):
    """
Test cases for evt core interface
"""
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_readViaObsPy(self):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = os.path.join(self.path, 'BI008_MEMA-04823.evt')
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

        filename = os.path.join(self.path, 'BX456_MOLA-02351.evt')
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

    def test_readViaModule(self):
        """
        Read files via obspy.evt.core.readEVT function.
        """
        filename = os.path.join(self.path, 'BI008_MEMA-04823.evt')
        # 1
        st = readEVT(filename)
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

        filename = os.path.join(self.path, 'BX456_MOLA-02351.evt')
        # 2
        st = readEVT(filename)
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
