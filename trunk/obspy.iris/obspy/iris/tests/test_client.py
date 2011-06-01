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

    def test_saveWaveform(self):
        """
        Testing simple waveform file save method.
        """
        #1 - file identical to file retrieved via web interface
        client = Client()
        start = UTCDateTime("2010-02-27T06:30:00")
        end = UTCDateTime("2010-02-27T06:31:00")
        origfile = os.path.join(self.path, 'data', 'IU.ANMO.00.BHZ.mseed')
        tempfile = NamedTemporaryFile().name
        client.saveWaveform(tempfile, "IU", "ANMO", "00", "BHZ", start, end)
        self.assertTrue(filecmp.cmp(origfile, tempfile))
        os.remove(tempfile)
        #2 - no data raises an exception
        self.assertRaises(Exception, client.saveWaveform, "YY", "XXXX", "00",
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
        origfile = os.path.join(self.path, 'data', 'RESP.ANMO.IU._.BH_')
        tempfile = NamedTemporaryFile().name
        client.saveResponse(tempfile, "IU", "ANMO", "*", "BH?", start, end)
        self.assertTrue(filecmp.cmp(origfile, tempfile))
        os.remove(tempfile)
        #3 - StationXML, single channel
        tempfile = NamedTemporaryFile().name
        client.saveResponse(tempfile, "IU", "ANMO", "00", "BHZ", start, end,
                            format="StationXML")
        data = open(tempfile).read()
        self.assertTrue('<Station net_code="IU" sta_code="ANMO">' in data)
        os.remove(tempfile)

    def test_sacpz(self):
        """
        Fetches SAC poles and zeros information.
        Can not be tested in docstring because creation date on server is
        included in resulting text.
        """
        client = Client()

        t1 = UTCDateTime("2005-01-01")
        t2 = UTCDateTime("2008-01-01")
        got = client.sacpz(network="IU", station="ANMO", location="00",
                           channel="BHZ", starttime=t1, endtime=t2)
        got = got.split("\n")
        expected = ['**************************************************',
                    '* NETWORK\tIU',
                    '* STATION\tANMO',
                    '* CHANNEL\tBHZ',
                    '* LOCATION\t00',
                    '* CREATED\t2011/05/05 09:56:08.639',
                    '* START\t\t2002/11/19 00:00:00.000',
                    '* END\t\t2008/06/30 00:00:00.000',
                    '* DESCRIPTION\tAlbuquerque, New Mexico, USA',
                    '* LATITUDE\t34.94598',
                    '* LONGITUDE\t-106.45713',
                    '* ELEVATION\t1671.0',
                    '* DEPTH\t\t145.0',
                    '* DIP\t\t0.0',
                    '* AZIMUTH\t0.0',
                    '* SAMPLE RATE\t20.0',
                    '* INPUT UNIT\tM',
                    '* OUTPUT UNIT\tCOUNTS',
                    '* INSTTYPE\tGeotech KS-54000 Borehole Seismometer',
                    '* INSTGAIN\t2.204000e+03',
                    '* SENSITIVITY\t9.244000e+08',
                    '* A0\t\t8.608300e+04',
                    '* ****',
                    'ZEROS\t3',
                    '\t+0.000000e+00\t+0.000000e+00\t',
                    '\t+0.000000e+00\t+0.000000e+00\t',
                    '\t+0.000000e+00\t+0.000000e+00\t',
                    'POLES\t5',
                    '\t-5.943130e+01\t+0.000000e+00\t',
                    '\t-2.271210e+01\t+2.710650e+01\t',
                    '\t-2.271210e+01\t-2.710650e+01\t',
                    '\t-4.800400e-03\t+0.000000e+00\t',
                    '\t-7.319900e-02\t+0.000000e+00\t',
                    'CONSTANT\t7.957513e+13',
                    '',
                    '',
                    '']
        # drop lines with creation date (current time during request)
        got.pop(5)
        expected.pop(5)
        self.assertEquals(got, expected)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__': # pragma: no cover
    unittest.main(defaultTest='suite')
