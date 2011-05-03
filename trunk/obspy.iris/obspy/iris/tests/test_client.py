# -*- coding: utf-8 -*-
"""
The obspy.iris.client test suite.
"""

from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.iris import Client
import os
import unittest
import filecmp


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.iris.client.Client.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

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

    def test_saveResponse(self):
        """
        Fetches and stores response information as SEED RESP file.
        """
        client = Client()
        start = UTCDateTime("2005-001T00:00:00")
        end = UTCDateTime("2008-001T00:00:00")
        #1 - RESP, single channel
        origfile = os.path.join(self.path, 'data', 'RESP.ANMO.IU.00.BHZ')
        tempfile = NamedTemporaryFile().name
        client.saveResponse(tempfile, "IU", "ANMO", "00", "BHZ", start, end)
        self.assertTrue(filecmp.cmp(origfile, tempfile))
        os.remove(tempfile)
        #2 - RESP, multiple channels
        origfile = os.path.join(self.path, 'data', 'RESP.ANMO.IU._._')
        tempfile = NamedTemporaryFile().name
        client.saveResponse(tempfile, "IU", "ANMO", "*", "*", start, end)
        self.assertTrue(filecmp.cmp(origfile, tempfile))
        os.remove(tempfile)
        #3 - StationXML, single channel
        tempfile = NamedTemporaryFile().name
        client.saveResponse(tempfile, "IU", "ANMO", "00", "BHZ", start, end,
                            format="StationXML")
        data = open(tempfile).read()
        self.assertTrue('<Station net_code="IU" sta_code="ANMO">' in data)
        os.remove(tempfile)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
