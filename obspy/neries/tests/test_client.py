# -*- coding: utf-8 -*-
"""
The obspy.neries.client test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy import UTCDateTime, read
from obspy.core.util import NamedTemporaryFile
from obspy.neries import Client


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.neries.client.Client.
    """
    def test_getTravelTimes(self):
        """
        Testing request method for calculating travel times.
        """
        client = Client()
        # 1
        result = client.getTravelTimes(20, 20, 10, [(48, 12)], 'ak135')
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['P'], 356988.24732429383)
        self.assertAlmostEqual(result[0]['S'], 645775.5623471631)
        # 2
        result = client.getTravelTimes(0, 0, 10,
                                       [(120, 0), (150, 0), (180, 0)])
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0]['P'], 605519.0321213702)
        self.assertAlmostEqual(result[0]['S'], 1097834.6352750373)
        self.assertAlmostEqual(result[1]['P'], 367256.0587305712)
        self.assertAlmostEqual(result[1]['S'], 665027.0583152708)
        self.assertEqual(result[2], {})

    def test_saveWaveform(self):
        """
        """
        # initialize client
        client = Client(user='test@obspy.org')
        start = UTCDateTime(2012, 1, 1)
        end = start + 10
        with NamedTemporaryFile() as tf:
            mseedfile = tf.name
            # MiniSEED
            client.saveWaveform(mseedfile, 'BW', 'MANZ', '', 'EHZ', start, end)
            st = read(mseedfile)
            # MiniSEED may not start with Volume Index Control Headers (V)
            with open(mseedfile, 'rb') as fp:
                self.assertNotEqual(fp.read(8)[6:7], b"V")
        # ArcLink cuts on record base
        self.assertTrue(st[0].stats.starttime <= start)
        self.assertTrue(st[0].stats.endtime >= end)
        self.assertEqual(st[0].stats.network, 'BW')
        self.assertEqual(st[0].stats.station, 'MANZ')
        self.assertEqual(st[0].stats.location, '')
        self.assertEqual(st[0].stats.channel, 'EHZ')
        # Full SEED
        with NamedTemporaryFile() as tf:
            fseedfile = tf.name
            client.saveWaveform(fseedfile, 'BW', 'MANZ', '', 'EHZ', start, end,
                                format='FSEED')
            st = read(fseedfile)
            # Full SEED must start with Volume Index Control Headers (V)
            with open(fseedfile, 'rb') as fp:
                self.assertEqual(fp.read(8)[6:7], b"V")
        # ArcLink cuts on record base
        self.assertTrue(st[0].stats.starttime <= start)
        self.assertTrue(st[0].stats.endtime >= end)
        self.assertEqual(st[0].stats.network, 'BW')
        self.assertEqual(st[0].stats.station, 'MANZ')
        self.assertEqual(st[0].stats.location, '')
        self.assertEqual(st[0].stats.channel, 'EHZ')

    def test_getInventory(self):
        """
        Testing inventory requests.
        """
        client = Client(user='test@obspy.org')
        dt1 = UTCDateTime("1974-01-01T00:00:00")
        dt2 = UTCDateTime("2011-01-01T00:00:00")
        # 1 - XML w/ instruments
        result = client.getInventory('GE', 'SNAA', '', 'BHZ', dt1, dt2,
                                     format='XML')
        self.assertTrue(result.startswith(b'<?xml'))
        self.assertTrue(b'code="GE"' in result)
        # 2 - SUDS object w/o instruments
        result = client.getInventory('GE', 'SNAA', '', 'BHZ', dt1, dt2,
                                     instruments=False)
        self.assertTrue(isinstance(result, object))
        self.assertEqual(result.ArclinkInventory.inventory.network._code, 'GE')
        # 3 - SUDS object w/ instruments
        result = client.getInventory('GE', 'SNAA', '', 'BHZ', dt1, dt2,
                                     instruments=True)
        self.assertTrue(isinstance(result, object))
        self.assertEqual(result.ArclinkInventory.inventory.network._code, 'GE')
        self.assertTrue('sensor' in result.ArclinkInventory.inventory)
        self.assertTrue('responsePAZ' in result.ArclinkInventory.inventory)
        # 4 - SUDS object with spatial filters
        client = Client(user='test@obspy.org')
        result = client.getInventory('GE', 'SNAA', '', 'BHZ', dt1, dt2,
                                     min_latitude=-72.0, max_latitude=-71.0,
                                     min_longitude=-3, max_longitude=-2)
        self.assertTrue(isinstance(result, object))
        self.assertEqual(result.ArclinkInventory.inventory.network._code, 'GE')
        # 5 - SUDS object with spatial filters with incorrect coordinates
        client = Client(user='test@obspy.org')
        result = client.getInventory('GE', 'SNAA', '', 'BHZ', dt1, dt2,
                                     min_latitude=-71.0, max_latitude=-72.0,
                                     min_longitude=-2, max_longitude=-3)
        self.assertTrue(isinstance(result, object))
        self.assertEqual(result.ArclinkInventory.inventory.network._code, 'GE')


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
