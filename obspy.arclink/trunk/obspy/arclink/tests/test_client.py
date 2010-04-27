# -*- coding: utf-8 -*-
"""
The obspy.arclink.client test suite.
"""

from obspy.arclink import Client
from obspy.arclink.client import ArcLinkException
from obspy.core import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile
import numpy as np
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
        st = read(mseedfile)
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
        st = read(fseedfile)
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
        st = read(mseedfile)
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
        st = read(fseedfile)
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
        pazs = client.getPAZ('BW', 'MANZ', '', 'EHZ', start, end)
        # compare first instrument
        paz = pazs.values()[0]
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
        pazs = client.getPAZ('BW', 'MANZ', '', 'EHZ', t, t + 1)
        paz = pazs['STS-2/N/g=1500']
        self.assertEqual(len(poles), 2)
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

    def test_SRL(self):
        """
        Tests if example in ObsPy paper submitted to the Electronic
        Seismologist section of SRL is still working. The test shouldn't be
        changed because the reference gets wrong.
        """
        paz = {'gain': 60077000.0,
               'poles': [(-0.037004000000000002 + 0.037016j),
                         (-0.037004000000000002 - 0.037016j),
                         (-251.33000000000001 + 0j),
                         (-131.03999999999999 - 467.29000000000002j),
                         (-131.03999999999999 + 467.29000000000002j)],
               'sensitivity': 2516778600.0,
               'zeros': [0j, 0j] }
        dat1 = np.array([288, 300, 292, 285, 265, 287, 279, 250, 278, 278])
        dat2 = np.array([445, 432, 425, 400, 397, 471, 426, 390, 450, 442])
        # Retrieve Data via Arclink
        client = Client(host="webdc.eu", port=18001)
        t = UTCDateTime("2009-08-24 00:20:03")
        st = client.getWaveform("BW", "RJOB", "", "EHZ", t, t + 30)
        poles_zeros = client.getPAZ("BW", "RJOB", "", "EHZ",
                                    t, t + 30).values()[0]
        self.assertEquals(paz['gain'], poles_zeros['gain'])
        self.assertEquals(paz['poles'], poles_zeros['poles'])
        self.assertEquals(paz['sensitivity'], poles_zeros['sensitivity'])
        self.assertEquals(paz['zeros'], poles_zeros['zeros'])
        self.assertEquals('BW', st[0].stats['network'])
        self.assertEquals('RJOB', st[0].stats['station'])
        self.assertEquals(200.0, st[0].stats['sampling_rate'])
        self.assertEquals(6001, st[0].stats['npts'])
        self.assertEquals('2009-08-24T00:20:03.000000Z',
                          str(st[0].stats['starttime']))
        np.testing.assert_array_equal(dat1, st[0].data[:10])
        np.testing.assert_array_equal(dat2, st[0].data[-10:])


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
