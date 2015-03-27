# -*- coding: utf-8 -*-
"""
The obspy.clients.earthworm.client test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import unittest

from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.core.util.decorator import skip_on_network_error
from obspy.clients.earthworm import Client


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.clients.earthworm.client.Client.
    """
    def setUp(self):
        # Monkey patch: set lower default precision of all UTCDateTime objects
        UTCDateTime.DEFAULT_PRECISION = 4
        self.client = Client("pele.ess.washington.edu", 16017, timeout=7)

    def tearDown(self):
        # restore default precision of all UTCDateTime objects
        UTCDateTime.DEFAULT_PRECISION = 6

    @skip_on_network_error
    def test_getWaveform(self):
        """
        Tests get_waveforms method.
        """
        client = self.client
        start = UTCDateTime(2013, 1, 17)
        end = start + 30
        # example 1 -- 1 channel, cleanup
        stream = client.get_waveforms('UW', 'TUCA', '', 'BHZ', start, end)
        self.assertEqual(len(stream), 1)
        delta = stream[0].stats.delta
        trace = stream[0]
        self.assertTrue(len(trace) == 1201)
        self.assertTrue(trace.stats.starttime >= start - delta)
        self.assertLessEqual(trace.stats.starttime, start + delta)
        self.assertTrue(trace.stats.endtime >= end - delta)
        self.assertLessEqual(trace.stats.endtime, end + delta)
        self.assertEqual(trace.stats.network, 'UW')
        self.assertEqual(trace.stats.station, 'TUCA')
        self.assertEqual(trace.stats.location, '')
        self.assertEqual(trace.stats.channel, 'BHZ')
        # example 2 -- 1 channel, no cleanup
        stream = client.get_waveforms('UW', 'TUCA', '', 'BHZ', start, end,
                                      cleanup=False)
        self.assertTrue(len(stream) >= 2)
        summed_length = sum(len(tr) for tr in stream)
        self.assertTrue(summed_length == 1201)
        self.assertTrue(stream[0].stats.starttime >= start - delta)
        self.assertLessEqual(stream[0].stats.starttime, start + delta)
        self.assertTrue(stream[-1].stats.endtime >= end - delta)
        self.assertLessEqual(stream[-1].stats.endtime, end + delta)
        for trace in stream:
            self.assertEqual(trace.stats.network, 'UW')
            self.assertEqual(trace.stats.station, 'TUCA')
            self.assertEqual(trace.stats.location, '')
            self.assertEqual(trace.stats.channel, 'BHZ')
        # example 3 -- component wildcarded with '?'
        stream = client.get_waveforms('UW', 'TUCA', '', 'BH?', start, end)
        self.assertEqual(len(stream), 3)
        for trace in stream:
            self.assertTrue(len(trace) == 1201)
            self.assertTrue(trace.stats.starttime >= start - delta)
            self.assertLessEqual(trace.stats.starttime, start + delta)
            self.assertTrue(trace.stats.endtime >= end - delta)
            self.assertLessEqual(trace.stats.endtime, end + delta)
            self.assertEqual(trace.stats.network, 'UW')
            self.assertEqual(trace.stats.station, 'TUCA')
            self.assertEqual(trace.stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[1].stats.channel, 'BHN')
        self.assertEqual(stream[2].stats.channel, 'BHE')

    @skip_on_network_error
    def test_saveWaveform(self):
        """
        Tests save_waveforms method.
        """
        # initialize client
        client = self.client
        start = UTCDateTime(2013, 1, 17)
        end = start + 30
        with NamedTemporaryFile() as tf:
            testfile = tf.name
            # 1 channel, cleanup (using SLIST to avoid dependencies)
            client.save_waveforms(testfile, 'UW', 'TUCA', '', 'BHZ', start,
                                  end, format="SLIST")
            stream = read(testfile)
        self.assertEqual(len(stream), 1)
        delta = stream[0].stats.delta
        trace = stream[0]
        self.assertTrue(len(trace) == 1201)
        self.assertTrue(trace.stats.starttime >= start - delta)
        self.assertLessEqual(trace.stats.starttime, start + delta)
        self.assertTrue(trace.stats.endtime >= end - delta)
        self.assertLessEqual(trace.stats.endtime, end + delta)
        self.assertEqual(trace.stats.network, 'UW')
        self.assertEqual(trace.stats.station, 'TUCA')
        self.assertEqual(trace.stats.location, '')
        self.assertEqual(trace.stats.channel, 'BHZ')

    @skip_on_network_error
    def test_availability(self):
        data = self.client.get_availability()
        seeds = ["%s.%s.%s.%s" % (d[0], d[1], d[2], d[3]) for d in data]
        self.assertIn('UW.TUCA.--.BHZ', seeds)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
