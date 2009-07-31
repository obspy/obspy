# -*- coding: utf-8 -*-
"""
The obspy.arclink.client test suite.
"""

from obspy.arclink import Client
from obspy.core.utcdatetime import UTCDateTime
import unittest


class ClientTestCase(unittest.TestCase):
    """
    Test cases for L{obspy.arclink.client.Client}.
    """

    def test_getWaveform(self):
        """
        """
        client = Client()
        # example 1
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        stream = client.getWaveform('BW', 'MANZ', '', 'EH*', start, end)
        self.assertEquals(len(stream), 3)
        for trace in stream:
            self.assertEquals(trace.stats.starttime, start)
            self.assertEquals(trace.stats.endtime, end)
            self.assertEquals(trace.stats.network, 'BW')
            self.assertEquals(trace.stats.station, 'MANZ')
        # example 2
        start = UTCDateTime(2009, 1, 1)
        end = start + 100
        stream2 = client.getWaveform('BW', 'RJOB', '', 'EHE', start, end)
        self.assertEquals(len(stream2), 1)
        trace2 = stream2[0]
        self.assertEquals(trace2.stats.starttime, start)
        self.assertEquals(trace2.stats.endtime, end)
        self.assertEquals(trace2.stats.network, 'BW')
        self.assertEquals(trace2.stats.station, 'RJOB')
        self.assertEquals(trace2.stats.location, '')
        self.assertEquals(trace2.stats.channel, 'EHE')

    def test_getMissingWaveform(self):
        """
        """
        client = Client()
        # example 1
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        stream = client.getWaveform('AA', 'AAAAA', '', '*', start, end)
        self.assertEquals(len(stream), 0)
        # example 2
        start = UTCDateTime(1008, 1, 1)
        end = start + 1
        stream = client.getWaveform('BW', 'MANZ', '', '*', start, end)
        self.assertEquals(len(stream), 0)

    def test_getNetworks(self):
        """
        """
        client = Client()
        # example 1
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        result = client.getNetworks(start, end)
        self.assertTrue('BW' in result.keys())
        self.assertEquals(result['BW']['code'], 'BW')
        self.assertEquals(result['BW']['type'], 'SP/BB')
        self.assertEquals(result['BW']['institutions'], u'Uni München')


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
