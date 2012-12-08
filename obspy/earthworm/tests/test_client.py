# -*- coding: utf-8 -*-
"""
The obspy.earthworm.client test suite.
"""

from obspy.earthworm import Client
from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile
import os
import unittest
from numpy import array


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.earthworm.client.Client.
    """
    def setUp(self):
        # Monkey patch: set lower default precision of all UTCDateTime objects
        UTCDateTime.DEFAULT_PRECISION = 4

    def tearDown(self):
        # restore default precision of all UTCDateTime objects
        UTCDateTime.DEFAULT_PRECISION = 6

    def test_getWaveform(self):
        """
        Tests getWaveform method.
        """
        client = Client("pele.ess.washington.edu", 16017)
        start = UTCDateTime() - 24 * 3600
        end = start + 30
        # example 1 -- 1 channel, cleanup
        stream = client.getWaveform('UW', 'TUCA', '', 'BHZ', start, end)
        self.assertEquals(len(stream), 1)
        delta = stream[0].stats.delta
        trace = stream[0]
        self.assertTrue(len(trace) == 1201)
        self.assertTrue(trace.stats.starttime >= start - delta)
        self.assertTrue(trace.stats.starttime <= start + delta)
        self.assertTrue(trace.stats.endtime >= end - delta)
        self.assertTrue(trace.stats.endtime <= end + delta)
        self.assertEquals(trace.stats.network, 'UW')
        self.assertEquals(trace.stats.station, 'TUCA')
        self.assertEquals(trace.stats.location, '')
        self.assertEquals(trace.stats.channel, 'BHZ')
        # example 2 -- 1 channel, no cleanup
        stream = client.getWaveform('UW', 'TUCA', '', 'BHZ', start, end,
                                    cleanup=False)
        self.assertTrue(len(stream) >= 2)
        summed_length = array([len(trace) for trace in stream]).sum()
        self.assertTrue(summed_length == 1201)
        self.assertTrue(stream[0].stats.starttime >= start - delta)
        self.assertTrue(stream[0].stats.starttime <= start + delta)
        self.assertTrue(stream[-1].stats.endtime >= end - delta)
        self.assertTrue(stream[-1].stats.endtime <= end + delta)
        for trace in stream:
            self.assertEquals(trace.stats.network, 'UW')
            self.assertEquals(trace.stats.station, 'TUCA')
            self.assertEquals(trace.stats.location, '')
            self.assertEquals(trace.stats.channel, 'BHZ')
        # example 3 -- component wildcarded with '?'
        stream = client.getWaveform('UW', 'TUCA', '', 'BH?', start, end)
        self.assertEquals(len(stream), 3)
        for trace in stream:
            self.assertTrue(len(trace) == 1201)
            self.assertTrue(trace.stats.starttime >= start - delta)
            self.assertTrue(trace.stats.starttime <= start + delta)
            self.assertTrue(trace.stats.endtime >= end - delta)
            self.assertTrue(trace.stats.endtime <= end + delta)
            self.assertEquals(trace.stats.network, 'UW')
            self.assertEquals(trace.stats.station, 'TUCA')
            self.assertEquals(trace.stats.location, '')
        self.assertEquals(stream[0].stats.channel, 'BHZ')
        self.assertEquals(stream[1].stats.channel, 'BHN')
        self.assertEquals(stream[2].stats.channel, 'BHE')

    def test_saveWaveform(self):
        """
        Tests saveWaveform method.
        """
        testfile = NamedTemporaryFile().name
        try:
            # initialize client
            client = Client("pele.ess.washington.edu", 16017)
            start = UTCDateTime() - 24 * 3600
            end = start + 30
            # 1 channel, cleanup (using SLIST to avoid dependencies)
            client.saveWaveform(testfile, 'UW', 'TUCA', '', 'BHZ', start, end,
                                format="SLIST")
            stream = read(testfile)
            self.assertEquals(len(stream), 1)
            delta = stream[0].stats.delta
            trace = stream[0]
            self.assertTrue(len(trace) == 1201)
            self.assertTrue(trace.stats.starttime >= start - delta)
            self.assertTrue(trace.stats.starttime <= start + delta)
            self.assertTrue(trace.stats.endtime >= end - delta)
            self.assertTrue(trace.stats.endtime <= end + delta)
            self.assertEquals(trace.stats.network, 'UW')
            self.assertEquals(trace.stats.station, 'TUCA')
            self.assertEquals(trace.stats.location, '')
            self.assertEquals(trace.stats.channel, 'BHZ')
        finally:
            os.remove(testfile)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
