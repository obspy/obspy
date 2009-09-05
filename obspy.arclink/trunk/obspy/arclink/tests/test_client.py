# -*- coding: utf-8 -*-
"""
The obspy.arclink.client test suite.
"""

from obspy.arclink import Client
from obspy.arclink.client import ArcLinkException
from obspy.core.utcdatetime import UTCDateTime
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
        client = Client()
        # example 1
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        self.assertRaises(ArcLinkException, client.getWaveform, 'BW', 'MAN*',
                          '', '*', start, end)

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
        self.assertEquals(result['BW']['institutions'][0:3], u'Uni')

    def test_saveWaveform(self):
        """
        """
        client = Client()
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        # MiniSEED
        client.saveWaveform('test.mseed', 'BW', 'MANZ', '', 'EHZ', start, end)
        stats = os.stat('test.mseed')
        st = obspy.read('test.mseed')
        self.assertEquals(stats.st_size, 1024)
        # ArcLink cuts on record base
        self.assertTrue(st[0].stats.starttime <= start)
        self.assertTrue(st[0].stats.endtime >= end)
        self.assertEquals(st[0].stats.network, 'BW')
        self.assertEquals(st[0].stats.station, 'MANZ')
        self.assertEquals(st[0].stats.location, '')
        self.assertEquals(st[0].stats.channel, 'EHZ')
        os.remove('test.mseed')
        # Full SEED
        client.saveWaveform('test.fseed', 'BW', 'MANZ', '', 'EHZ', start, end,
                            format='FSEED')
        stats = os.stat('test.fseed')
        st = obspy.read('test.fseed')
        self.assertTrue(stats.st_size > 1024)
        self.assertEquals(open('test.fseed').read(8), "000001V ")
        # ArcLink cuts on record base
        self.assertTrue(st[0].stats.starttime <= start)
        self.assertTrue(st[0].stats.endtime >= end)
        self.assertEquals(st[0].stats.network, 'BW')
        self.assertEquals(st[0].stats.station, 'MANZ')
        self.assertEquals(st[0].stats.location, '')
        self.assertEquals(st[0].stats.channel, 'EHZ')
        os.remove('test.fseed')

    def test_getPAZ(self):
        """
        Test for the Client.getPAZ function. As reference the EHZ channel
        of MANZ is taken, the result is compared to the entries of the
        local response file of the bavarian network.
        """
        zeros = [0j,0j]
        poles = [-3.700400E-02 +3.701600E-02j, -3.700400E-02 -3.701600E-02j,
                 -2.513300E+02 +0.000000E+00j, -1.310400E+02 -4.672900E+02j, 
                 -1.310400E+02 +4.672900E+02j]
        gain = 6.0077E+07
        sensitivity = 2.516800E+09
        #
        client = Client()
        start = UTCDateTime(2009, 1, 1)
        end = start + 1
        # poles and zeros
        paz = client.getPAZ('BW', 'MANZ', '', 'EHZ', start, end)
        self.assertEqual(gain, paz['gain'])
        self.assertEqual(poles, paz['poles'])
        self.assertEqual(zeros, paz['zeros'])
        # can only compare dezimal places
        self.assertAlmostEqual(sensitivity/1e9, paz['sensitivity']/1e9, places=4)
    
    
    def test_saveResponse(self):
        """
        """
        client = Client()
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        # Dataless SEED
        client.saveResponse('test.dseed', 'BW', 'MANZ', '', 'EHZ', start, end)
        self.assertEquals(open('test.dseed').read(8), "000001V ")
        os.remove('test.dseed')


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
