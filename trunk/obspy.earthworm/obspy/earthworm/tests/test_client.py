# -*- coding: utf-8 -*-
"""
The obspy.earthworm.client test suite.
"""

from obspy.earthworm import Client
from obspy.core import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile
import os
import unittest


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.earthworm.client.Client.
    """
    def test_getWaveform(self):
        """
        Tests getWaveform method.
        """
        client = Client("hood.ess.washington.edu", 16021)
        start = UTCDateTime(2011, 12, 17, 21, 31, 55)
        end = start + 30
        # example 1 -- 1 channel, cleanup
        stream = client.getWaveform('UW', 'LON', '', 'BHZ', start, end)
        self.assertEquals(len(stream), 1)
        trace = stream[0]
        self.assertTrue(len(trace) == 1201)
        self.assertTrue(trace.stats.starttime == \
                        UTCDateTime("2011-12-17T21:31:55.005000Z"))
        self.assertTrue(trace.stats.endtime == \
                        UTCDateTime("2011-12-17T21:32:25.005000Z"))
        self.assertEquals(trace.stats.network, 'UW')
        self.assertEquals(trace.stats.station, 'LON')
        self.assertEquals(trace.stats.location, '--')
        self.assertEquals(trace.stats.channel, 'BHZ')
        # example 2 -- 1 channel, no cleanup
        stream = client.getWaveform('UW', 'LON', '', 'BHZ', start, end,
                                    cleanup=False)
        self.assertEquals(len(stream), 2)
        self.assertTrue(len(stream[0]) == 848)
        self.assertTrue(len(stream[1]) == 353)
        self.assertTrue(stream[0].stats.starttime == \
                        UTCDateTime("2011-12-17T21:31:55.005000Z"))
        self.assertTrue(stream[0].stats.endtime == \
                        UTCDateTime("2011-12-17T21:32:16.180000Z"))
        self.assertTrue(stream[1].stats.starttime == \
                        UTCDateTime("2011-12-17T21:32:16.205000Z"))
        self.assertTrue(stream[1].stats.endtime == \
                        UTCDateTime("2011-12-17T21:32:25.005000Z"))
        for trace in stream:
            self.assertEquals(trace.stats.network, 'UW')
            self.assertEquals(trace.stats.station, 'LON')
            self.assertEquals(trace.stats.location, '--')
            self.assertEquals(trace.stats.channel, 'BHZ')
        # example 3 -- component wildcarded with '?'
        stream = client.getWaveform('UW', 'LON', '', 'BH?', start, end)
        self.assertEquals(len(stream), 3)
        for trace in stream:
            self.assertTrue(len(trace) == 1201)
            self.assertTrue(trace.stats.starttime == \
                            UTCDateTime("2011-12-17T21:31:55.005000Z"))
            self.assertTrue(trace.stats.endtime == \
                            UTCDateTime("2011-12-17T21:32:25.005000Z"))
            self.assertEquals(trace.stats.network, 'UW')
            self.assertEquals(trace.stats.station, 'LON')
            self.assertEquals(trace.stats.location, '--')
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
            client = Client("hood.ess.washington.edu", 16021)
            start = UTCDateTime(2011, 12, 17, 21, 31, 55)
            end = start + 30
            # example 1 -- 1 channel, cleanup (using SLIST to avoid dependencies)
            client.saveWaveform(testfile, 'UW', 'LON', '', 'BHZ', start, end,
                                format="SLIST")
            stream = read(testfile)
            self.assertEquals(len(stream), 1)
            trace = stream[0]
            self.assertTrue(len(trace) == 1201)
            self.assertTrue(trace.stats.starttime == \
                            UTCDateTime("2011-12-17T21:31:55.005000Z"))
            self.assertTrue(trace.stats.endtime == \
                            UTCDateTime("2011-12-17T21:32:25.005000Z"))
            self.assertEquals(trace.stats.network, 'UW')
            self.assertEquals(trace.stats.station, 'LON')
            self.assertEquals(trace.stats.location, '--')
            self.assertEquals(trace.stats.channel, 'BHZ')
        finally:
            os.remove(testfile)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
