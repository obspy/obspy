# -*- coding: utf-8 -*-
"""
The obspy.arclink.client test suite.
"""

from obspy.arclink import Client
from obspy.arclink.client import ArcLinkException
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile
import obspy
import os
import unittest


class ClientTestCase(unittest.TestCase):
    """
    Test cases for L{obspy.arclink.client.Client}.
    """
    def test_getWaveform(self):
        """
        """
        # initialize client
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

    def test_getNotExistingWaveform(self):
        """
        """
        # initialize client
        client = Client()
        # example 1
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        self.assertRaises(ArcLinkException, client.getWaveform, 'AA', 'AAAAA',
                          '', '*', start, end)
        # example 2
        start = UTCDateTime(1008, 1, 1)
        end = start + 1
        self.assertRaises(ArcLinkException, client.getWaveform, 'BW', 'MANZ',
                          '', '*', start, end)

    def test_getWaveformWrongPattern(self):
        """
        """
        # initialize client
        client = Client()
        # example 1
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        self.assertRaises(ArcLinkException, client.getWaveform, 'BW', 'MAN*',
                          '', '*', start, end)

    def test_getNetworks(self):
        """
        """
        # initialize client
        client = Client()
        # example 1
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        result = client.getNetworks(start, end)
        self.assertTrue('BW' in result.keys())
        self.assertEquals(result['BW']['code'], 'BW')
        self.assertEquals(result['BW']['type'], 'SP/BB')
        self.assertEquals(result['BW']['institutions'][0:3], u'Uni')

    def test_saveWaveform(self):
        """
        """
        mseedfile = NamedTemporaryFile().name
        fseedfile = NamedTemporaryFile().name
        # initialize client
        client = Client()
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        # MiniSEED
        client.saveWaveform(mseedfile, 'BW', 'MANZ', '', 'EHZ', start, end)
        stats = os.stat(mseedfile)
        st = obspy.read(mseedfile)
        self.assertEquals(stats.st_size, 1024)
        # ArcLink cuts on record base
        self.assertTrue(st[0].stats.starttime <= start)
        self.assertTrue(st[0].stats.endtime >= end)
        self.assertEquals(st[0].stats.network, 'BW')
        self.assertEquals(st[0].stats.station, 'MANZ')
        self.assertEquals(st[0].stats.location, '')
        self.assertEquals(st[0].stats.channel, 'EHZ')
        os.remove(mseedfile)
        # Full SEED
        client.saveWaveform(fseedfile, 'BW', 'MANZ', '', 'EHZ', start, end,
                            format='FSEED')
        stats = os.stat(fseedfile)
        st = obspy.read(fseedfile)
        self.assertTrue(stats.st_size > 1024)
        self.assertEquals(open(fseedfile).read(8), "000001V ")
        # ArcLink cuts on record base
        self.assertTrue(st[0].stats.starttime <= start)
        self.assertTrue(st[0].stats.endtime >= end)
        self.assertEquals(st[0].stats.network, 'BW')
        self.assertEquals(st[0].stats.station, 'MANZ')
        self.assertEquals(st[0].stats.location, '')
        self.assertEquals(st[0].stats.channel, 'EHZ')
        os.remove(fseedfile)

    def test_getCompressedWaveform(self):
        """
        """
        # initialize client
        client = Client()
        # example 1
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        stream = client.getWaveform('BW', 'MANZ', '', 'EH*', start, end,
                                    compressed=False)
        self.assertEquals(len(stream), 3)
        for trace in stream:
            self.assertEquals(trace.stats.starttime, start)
            self.assertEquals(trace.stats.endtime, end)
            self.assertEquals(trace.stats.network, 'BW')
            self.assertEquals(trace.stats.station, 'MANZ')
        # example 2
        start = UTCDateTime(2009, 1, 1)
        end = start + 100
        stream2 = client.getWaveform('BW', 'RJOB', '', 'EHE', start, end,
                                     compressed=False)
        self.assertEquals(len(stream2), 1)
        trace2 = stream2[0]
        self.assertEquals(trace2.stats.starttime, start)
        self.assertEquals(trace2.stats.endtime, end)
        self.assertEquals(trace2.stats.network, 'BW')
        self.assertEquals(trace2.stats.station, 'RJOB')
        self.assertEquals(trace2.stats.location, '')
        self.assertEquals(trace2.stats.channel, 'EHE')

    def test_saveCompressedWaveform(self):
        """
        """
        mseedfile = NamedTemporaryFile().name
        fseedfile = NamedTemporaryFile().name
        # initialize client
        client = Client()
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        # MiniSEED
        client.saveWaveform(mseedfile, 'BW', 'MANZ', '', 'EHZ', start, end,
                            compressed=False)
        stats = os.stat(mseedfile)
        st = obspy.read(mseedfile)
        self.assertEquals(stats.st_size, 1024)
        # ArcLink cuts on record base
        self.assertTrue(st[0].stats.starttime <= start)
        self.assertTrue(st[0].stats.endtime >= end)
        self.assertEquals(st[0].stats.network, 'BW')
        self.assertEquals(st[0].stats.station, 'MANZ')
        self.assertEquals(st[0].stats.location, '')
        self.assertEquals(st[0].stats.channel, 'EHZ')
        os.remove(mseedfile)
        # Full SEED
        client.saveWaveform(fseedfile, 'BW', 'MANZ', '', 'EHZ', start, end,
                            format='FSEED')
        stats = os.stat(fseedfile)
        st = obspy.read(fseedfile)
        self.assertTrue(stats.st_size > 1024)
        self.assertEquals(open(fseedfile).read(8), "000001V ")
        # ArcLink cuts on record base
        self.assertTrue(st[0].stats.starttime <= start)
        self.assertTrue(st[0].stats.endtime >= end)
        self.assertEquals(st[0].stats.network, 'BW')
        self.assertEquals(st[0].stats.station, 'MANZ')
        self.assertEquals(st[0].stats.location, '')
        self.assertEquals(st[0].stats.channel, 'EHZ')
        os.remove(fseedfile)

    def test_getPAZ(self):
        """
        Test for the Client.getPAZ function.
        
        As reference the EHZ channel of MANZ is taken, the result is compared 
        to the entries of the local response file of the Bavarian network.
        """
        # reference values
        zeros = [0j, 0j]
        poles = [-3.700400e-02 + 3.701600e-02j, -3.700400e-02 - 3.701600e-02j,
                 - 2.513300e+02 + 0.000000e+00j, -1.310400e+02 - 4.672900e+02j,
                 - 1.310400e+02 + 4.672900e+02j]
        gain = 6.0077e+07
        sensitivity = 2.5168e+09
        # set start and endtime
        start = UTCDateTime(2009, 1, 1)
        end = start + 1
        # initialize client
        client = Client()
        # fetch poles and zeros
        paz = client.getPAZ('BW', 'MANZ', '', 'EHZ', start, end)
        self.assertEqual(gain, paz['gain'])
        self.assertEqual(poles, paz['poles'])
        self.assertEqual(zeros, paz['zeros'])
        # can only compare four decimal places
        self.assertAlmostEqual(sensitivity / 1e9,
                               paz['sensitivity'] / 1e9, places=4)

    def test_getPAZ2(self):
        """
        Test for the Client.getPAZ function for erde.
        """
        poles = [-3.700400e-02 + 3.701600e-02j, -3.700400e-02 - 3.701600e-02j]
        t = UTCDateTime(2009, 1, 1)
        client = Client("erde.geophysik.uni-muenchen.de")
        # fetch poles and zeros
        paz = client.getPAZ('BW', 'MANZ', '', 'EHZ', t, t+1)
        self.assertEqual(len(poles), 5)
        self.assertEqual(poles, paz['poles'][:2])

    def test_saveResponse(self):
        """
        Fetches and stores response information as Dataless SEED volume.
        """
        tempfile = NamedTemporaryFile().name
        client = Client()
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        # Dataless SEED
        client.saveResponse(tempfile, 'BW', 'MANZ', '', 'EHZ', start, end)
        self.assertEquals(open(tempfile).read(8), "000001V ")
        os.remove(tempfile)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
