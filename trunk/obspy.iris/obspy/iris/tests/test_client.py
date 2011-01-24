# -*- coding: utf-8 -*-
"""
The obspy.iris.client test suite.
"""

from obspy.core.utcdatetime import UTCDateTime
from obspy.iris import Client
import unittest


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.iris.client.Client.
    """
    def test_getWaveform(self):
        """
        Testing simple waveform request method.
        """
        #1 - simple example
        client = Client()
        start = UTCDateTime("2010-02-27T06:30:00.019538Z")
        end = start + 20
        stream = client.getWaveform("IU", "ANMO", "00", "BHZ", start, end)
        self.assertEquals(len(stream), 1)
        self.assertEquals(stream[0].stats.starttime, start)
        self.assertEquals(stream[0].stats.endtime, end)
        self.assertEquals(stream[0].stats.network, 'IU')
        self.assertEquals(stream[0].stats.station, 'ANMO')
        self.assertEquals(stream[0].stats.location, '00')
        self.assertEquals(stream[0].stats.channel, 'BHZ')
        #2 - no data raises an exception
        self.assertRaises(Exception, client.getWaveform, "YY", "XXXX", "00",
                          "BHZ", start, end)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
