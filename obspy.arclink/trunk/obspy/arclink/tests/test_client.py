# -*- coding: utf-8 -*-
"""
The obspy.arclink.client test suite.
"""

from obspy.arclink import Client
from obspy.arclink.client import ArcLinkException
from obspy.core import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile, AttribDict
import numpy as np
import os
import unittest


def dict_checker(dict1, dict2):
    for key, value in dict1.iteritems():
        if not dict2.has_key(key):
            print "dict2: missing key %s" % key
            continue
        if isinstance(value, AttribDict):
            dict_checker(dict1[key], dict2[key])
        elif dict2[key] != value:
            print "%s: %s != %s" % (key, value, dict2[key])


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

    def test_delayedRequest(self):
        """
        """
        client = Client(host='webdc.eu', port=18002, command_delay=0.1)
        start = UTCDateTime(2010, 1, 1)
        end = start + 100
        # getWaveform with 0.1 delay 
        stream = client.getWaveform('BW', 'MANZ', '', 'EHE', start, end)
        self.assertEquals(len(stream), 1)
        # getRouting with 0.1 delay 
        results = client.getRouting('BW', '*', start, end)
        self.assertEquals(len(results), 1)
        self.assertTrue(results.has_key('BW.'))
        self.assertEquals(len(results['BW.']), 2)

    def test_getRouting(self):
        """
        Tests getRouting method on various ArcLink nodes.
        """
        dt = UTCDateTime(2010, 1, 1)
        #1 - BW network via erde.geophysik.uni-muenchen.de:18001
        client = Client(host="erde.geophysik.uni-muenchen.de", port=18001)
        results = client.getRouting('BW', 'RJOB', dt, dt + 1)
        self.assertEquals(results, {'BW.RJOB': []})
        #2 - BW network via webdc:18001
        client = Client(host="webdc.eu", port=18001)
        results = client.getRouting('BW', 'RJOB', dt, dt + 1)
        self.assertEquals(results,
            {'BW.': [{'priority': 1, 'start': UTCDateTime(1980, 1, 1, 0, 0),
                      'host': 'webdc.eu', 'end': None, 'port': 18001}]})
        #3 - BW network via webdc:18002
        client = Client(host="webdc.eu", port=18002)
        results = client.getRouting('BW', 'RJOB', dt, dt + 1)
        self.assertEquals(results,
            {'BW.': [{'priority': 2, 'start': UTCDateTime(1980, 1, 1, 0, 0),
                      'host': 'webdc.eu', 'end': None, 'port': 18002},
                     {'priority': 1, 'start': UTCDateTime(1980, 1, 1, 0, 0),
                      'host': 'erde.geophysik.uni-muenchen.de', 'end': None,
                      'port': 18001}]})
        #4 - BW network via bhlsa03.knmi.nl:18001
        client = Client(host="bhlsa03.knmi.nl", port=18001)
        results = client.getRouting('BW', 'RJOB', dt, dt + 1)
        self.assertEquals(results,
            {'BW.': [{'priority': 1, 'start': UTCDateTime(1980, 1, 1, 0, 0),
                      'host': 'webdc.eu', 'end': None, 'port': 18001}]})
        #5 - BW network via bhlsa03.knmi.nl:18002
        client = Client(host="bhlsa03.knmi.nl", port=18002)
        results = client.getRouting('BW', 'RJOB', dt, dt + 1)
        self.assertEquals(results,
            {'BW.': [{'priority': 2, 'start': UTCDateTime(1980, 1, 1, 0, 0),
                      'host': 'webdc.eu', 'end': None, 'port': 18002},
                     {'priority': 1, 'start': UTCDateTime(1980, 1, 1, 0, 0),
                      'host': 'erde.geophysik.uni-muenchen.de', 'end': None,
                      'port': 18001}]})
        #6 - IV network has no routing entry in webdc.eu:18001
        client = Client(host="webdc.eu", port=18001)
        results = client.getRouting('IV', '', dt, dt + 1)
        self.assertEquals(results, {})
        #7 - IV network via webdc.eu:18002
        client = Client(host="webdc.eu", port=18002)
        results = client.getRouting('IV', '', dt, dt + 1)
        self.assertEquals(results,
            {'IV.': [{'priority': 1, 'start': UTCDateTime(1980, 1, 1, 0, 0),
                      'host': 'eida.rm.ingv.it', 'end': None, 'port': 18002},
                     {'priority': 1, 'start': UTCDateTime(1980, 1, 1, 0, 0),
                      'host': 'eida.rm.ingv.it', 'end': None, 'port': 18001}]})
        #8 - GE.APE via webdc.eu:18001
        client = Client(host="webdc.eu", port=18001)
        results = client.getRouting('GE', 'APE', dt, dt + 1)
        self.assertEquals(results,
            {'GE.': [{'priority': 1, 'start': UTCDateTime(1980, 1, 1, 0, 0),
                      'host': 'webdc.eu', 'end': None, 'port': 18001}]})
        #9 - GE.APE via webdc.eu:18002
        client = Client(host="webdc.eu", port=18002)
        results = client.getRouting('GE', 'APE', dt, dt + 1)
        self.assertEquals(results,
            {'GE.': [{'priority': 1, 'start': UTCDateTime(1980, 1, 1, 0, 0),
                      'host': 'webdc.eu', 'end': None, 'port': 18002}]})
        #10 - unknown network 00 via webdc.eu:18002
        client = Client(host="webdc.eu", port=18002)
        results = client.getRouting('00', '', dt, dt + 1)
        self.assertEquals(results, {})

    def test_getInventory(self):
        """
        Tests getInventory method on various ArcLink nodes.
        """
        dt = UTCDateTime(2010, 1, 1)
        # expected results for inventory schmea 0.2 and 1.0
        expected_results_0_2 = AttribDict({
            'remark': '', 'code': 'BW', 'end': None,
            'description': 'BayernNetz',
            'stations': AttribDict({
                'MANZ': AttribDict({
                    'remark': '', 'code': 'MANZ', 'elevation': 635.0,
                    'description': 'Manzenberg,Bavaria',
                    'start': UTCDateTime(2005, 12, 6, 0, 0),
                    'restricted': False, 'archive_net': 'BW',
                    'longitude': 12.1083, 'affiliation': '', 'depth': 4.0,
                    'place': 'Manzenberg, Bavaria', 'country': 'Germany',
                    'latitude': 49.986199999999997, 'end': None})}),
            'restricted': False, 'region': 'Germany', 'archive': 'LMU',
            'start': UTCDateTime(1980, 1, 1, 0, 0), 'net_class': 'p',
            'type': 'SP/BB', 'institutions': u'Uni M\xfcnchen'})
        expected_results_1_0 = AttribDict({
            'remark': '', 'code': 'BW', 'end': None,
            'description': 'BayernNetz',
            'stations': AttribDict({
                'MANZ': AttribDict({
                    'remark': '', 'code': 'MANZ', 'elevation': 635.0,
                    'description': 'Manzenberg,Bavaria',
                    'start': UTCDateTime(2005, 12, 6, 0, 0),
                    'restricted': False, 'archive_net': '',
                    'longitude': 12.1083, 'affiliation': '', 'depth': None,
                    'place': 'Manzenberg, Bavaria', 'country': 'Germany',
                    'latitude': 49.986199999999997, 'end': None})}),
            'restricted': False, 'region': 'Germany', 'archive': 'LMU',
            'start': UTCDateTime(1980, 1, 1, 0, 0), 'net_class': '',
            'type': 'SP/BB', 'institutions': u'Uni M\xfcnchen'})
        #1 - BW network via erde.geophysik.uni-muenchen.de:18001
        client = Client(host="erde.geophysik.uni-muenchen.de", port=18001)
        results = client.getInventory('BW', 'MANZ', dt, dt + 1)
        self.assertEquals(len(results), 1)
        self.assertTrue('BW' in results)
        self.assertEquals(results.BW, expected_results_0_2)
        #2 - BW network via webdc:18001
        client = Client(host="webdc.eu", port=18001)
        results = client.getInventory('BW', 'MANZ', dt, dt + 1)
        self.assertEquals(len(results), 1)
        self.assertTrue('BW' in results)
        self.assertEquals(results.BW, expected_results_0_2)
        #3 - BW network via webdc:18002
        client = Client(host="webdc.eu", port=18002)
        results = client.getInventory('BW', 'MANZ', dt, dt + 1)
        self.assertEquals(len(results), 1)
        self.assertTrue('BW' in results)
        self.assertEquals(results.BW, expected_results_1_0)
        #4 - BW network via bhlsa03.knmi.nl:18001
        client = Client(host="bhlsa03.knmi.nl", port=18001)
        results = client.getInventory('BW', 'MANZ', dt, dt + 1)
        self.assertEquals(len(results), 1)
        self.assertTrue('BW' in results)
        self.assertEquals(results.BW, expected_results_0_2)
        #5 - BW network via bhlsa03.knmi.nl:18002
        client = Client(host="bhlsa03.knmi.nl", port=18002)
        results = client.getInventory('BW', 'MANZ', dt, dt + 1)
        self.assertEquals(len(results), 1)
        self.assertTrue('BW' in results)
        self.assertEquals(results.BW, expected_results_1_0)
        #10 - unknown network 00 via webdc.eu:18002
        client = Client(host="webdc.eu", port=18002)
        results = client.getInventory('00', '', dt, dt + 1)
        self.assertFalse(results)

    def test_getWaveform_with_Metadata(self):
        """
        """
        # initialize client
        client = Client()
        # example 1
        t = UTCDateTime("2010-08-01T12:00:00")
        st = client.getWaveform("BW", "RJOB", "", "EHZ", t, t + 60,
                                getPAZ=True, getCoordinates=True)
        statsdict = st[0].stats.__dict__
        statsdict.pop("endtime")
        statsdict.pop("delta")
        results = {
            'network': 'BW',
            '_format': 'MSEED',
            'paz': AttribDict({
                'poles': [(-0.037004000000000002 + 0.037016j),
                          (-0.037004000000000002 - 0.037016j),
                          (-251.33000000000001 + 0j),
                          (-131.03999999999999 - 467.29000000000002j),
                          (-131.03999999999999 + 467.29000000000002j)],
                'sensitivity': 2516778600.0,
                'zeros': [0j, 0j],
                'gain': 60077000.0}),
            'mseed': AttribDict({'dataquality': 'D',
                                 'record_length': 512,
                                 'encoding': 'STEIM1',
                                 'byteorder': '>'}),
            'coordinates': AttribDict({'latitude': 47.737166999999999,
                                       'elevation': 860.0,
                                       'longitude': 12.795714}),
            'station': 'RJOB',
            'location': '',
            'starttime': UTCDateTime(2010, 8, 1, 12, 0),
            'npts': 546,
            'calib': 1.0,
            'sampling_rate': 200.0,
            'channel': 'EHZ'}
        self.assertEquals(statsdict, results)
        st = client.getWaveform("CZ", "VRAC", "", "BHZ", t, t + 60,
                                getPAZ=True, getCoordinates=True)
        statsdict = st[0].stats.__dict__
        statsdict.pop("endtime")
        statsdict.pop("delta")
        results = {
            'network': 'CZ',
            '_format': 'MSEED',
            'paz': AttribDict({
                'poles': [(-0.037004000000000002 + 0.037016j),
                          (-0.037004000000000002 - 0.037016j),
                          (-251.33000000000001 + 0j),
                          (-131.03999999999999 - 467.29000000000002j),
                          (-131.03999999999999 + 467.29000000000002j)],
                'sensitivity': 8200000000.0,
                'zeros': [0j, 0j],
                'gain': 60077000.0}),
            'mseed': AttribDict({'dataquality': 'D',
                                 'record_length': 512,
                                 'encoding': 'STEIM1',
                                 'byteorder': '>'}),
            'coordinates': AttribDict({'latitude': 49.308399999999999,
                                       'elevation': 470.0,
                                       'longitude': 16.593299999999999}),
            'station': 'VRAC',
            'location': '',
            'starttime': UTCDateTime(2010, 8, 1, 11, 59, 59, 993400),
            'npts': 2401,
            'calib': 1.0,
            'sampling_rate': 40.0,
            'channel': 'BHZ'}
        self.assertEquals(statsdict, results)

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

    def test_getStations(self):
        """
        """
        # initialize client
        client = Client()
        # example 1
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        result = client.getStations(start, end, 'BW')
        self.assertTrue(
            AttribDict({'remark': '', 'code': 'RWMO', 'elevation': 763.0,
                        'description': 'Wildenmoos, Bavaria',
                        'affiliation': '', 'country': 'Germany',
                        'longitude': 12.729887,
                        'start': UTCDateTime(2006, 6, 4, 0, 0), 'depth': 1.0,
                        'place': 'Wildenmoos, Bavaria', 'archive_net': 'BW',
                        'latitude': 47.744171999999999, 'end': None,
                        'restricted': False}) in result)

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
        # set start and end time
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
        Test for the Client.getPAZ function for erde.geophysik.uni-muenchen.de.
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
