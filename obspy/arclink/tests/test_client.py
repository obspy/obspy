# -*- coding: utf-8 -*-
"""
The obspy.arclink.client test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import io
import operator
import unittest
import warnings

import numpy as np

from obspy import read
from obspy.arclink import Client
from obspy.arclink.client import ArcLinkException
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import AttribDict, NamedTemporaryFile


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.arclink.client.Client.
    """
    def test_getWaveform(self):
        """
        Tests getWaveform method.
        """
        # example 1
        client = Client(user='test@obspy.org')
        start = UTCDateTime(2010, 1, 1)
        end = start + 1
        stream = client.getWaveform('BW', 'MANZ', '', 'EH*', start, end)
        self.assertEqual(len(stream), 3)
        for trace in stream:
            self.assertTrue(trace.stats.starttime <= start)
            self.assertTrue(trace.stats.endtime >= end)
            self.assertEqual(trace.stats.network, 'BW')
            self.assertEqual(trace.stats.station, 'MANZ')
            self.assertEqual(trace.stats.location, '')
            self.assertEqual(trace.stats.channel[0:2], 'EH')
        # example 2
        client = Client(user='test@obspy.org')
        start = UTCDateTime("2010-12-31T23:59:50.495000Z")
        end = start + 100
        stream = client.getWaveform('GE', 'APE', '', 'BHE', start, end)
        self.assertEqual(len(stream), 1)
        trace = stream[0]
        self.assertTrue(trace.stats.starttime <= start)
        self.assertTrue(trace.stats.endtime >= end)
        self.assertEqual(trace.stats.network, 'GE')
        self.assertEqual(trace.stats.station, 'APE')
        self.assertEqual(trace.stats.location, '')
        self.assertEqual(trace.stats.channel, 'BHE')

    def test_getWaveformNoRouting(self):
        """
        Tests routing parameter of getWaveform method.
        """
        # 1 - requesting BW data w/o routing on webdc.eu
        client = Client(user='test@obspy.org')
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        self.assertRaises(ArcLinkException, client.getWaveform, 'BW', 'MANZ',
                          '', 'EH*', start, end, route=False)
        # 2 - requesting BW data w/o routing directly from BW ArcLink node
        client = Client(host='erde.geophysik.uni-muenchen.de', port=18001,
                        user='test@obspy.org')
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        stream = client.getWaveform('BW', 'MANZ', '', 'EH*', start, end,
                                    route=False)
        for trace in stream:
            self.assertTrue(trace.stats.starttime <= start)
            self.assertTrue(trace.stats.endtime >= end)
            self.assertEqual(trace.stats.network, 'BW')
            self.assertEqual(trace.stats.station, 'MANZ')
            self.assertEqual(trace.stats.location, '')
            self.assertEqual(trace.stats.channel[0:2], 'EH')

    def test_delayedRequest(self):
        """
        """
        # initialize client with 0.1 delay
        client = Client(host='webdc.eu', port=18002, command_delay=0.1,
                        user='test@obspy.org')
        start = UTCDateTime(2010, 1, 1)
        end = start + 100
        # getWaveform with 0.1 delay
        stream = client.getWaveform('GR', 'FUR', '', 'HHE', start, end)
        self.assertEqual(len(stream), 1)
        # getRouting with 0.1 delay
        results = client.getRouting('GR', 'FUR', start, end)
        self.assertTrue('GR...' in results)

    def test_getRouting(self):
        """
        Tests getRouting method on various ArcLink nodes.
        """
        dt = UTCDateTime(2010, 1, 1)
        # 1 - BW network via erde.geophysik.uni-muenchen.de:18001
        client = Client(host="erde.geophysik.uni-muenchen.de", port=18001,
                        user='test@obspy.org')
        results = client.getRouting('BW', 'RJOB', dt, dt + 1)
        self.assertEqual(
            results,
            {'BW.RJOB..': [{'priority': 1,
                            'start': UTCDateTime(1980, 1, 1, 0, 0),
                            'host': '141.84.11.2', 'end': None,
                            'port': 18001}]})
        # 2 - BW network via webdc:18001
        client = Client(host="webdc.eu", port=18001, user='test@obspy.org')
        results = client.getRouting('BW', 'RJOB', dt, dt + 1)
        self.assertEqual(
            results,
            {'BW.RJOB..': [{'priority': 1,
                            'start': UTCDateTime(1980, 1, 1, 0, 0),
                            'host': '141.84.11.2',
                            'end': None,
                            'port': 18001}]})
        # 3 - BW network via webdc:18002
        client = Client(host="webdc.eu", port=18002, user='test@obspy.org')
        results = client.getRouting('BW', 'RJOB', dt, dt + 1)
        self.assertEqual(
            results,
            {'BW.RJOB..': [{'priority': 1,
                            'start': UTCDateTime(1980, 1, 1, 0, 0),
                            'host': '141.84.11.2',
                            'end': None,
                            'port': 18001}]})
        # 4 - IV network via webdc.eu:18001
        client = Client(host="webdc.eu", port=18001, user='test@obspy.org')
        results = client.getRouting('IV', '', dt, dt + 1)
        self.assertEqual(
            results,
            {'IV...': [{'priority': 1, 'start': UTCDateTime(1980, 1, 1, 0, 0),
                        'host': 'eida.rm.ingv.it', 'end': None,
                        'port': 18002}]})
        # 5 - IV network via webdc.eu:18002
        client = Client(host="webdc.eu", port=18002, user='test@obspy.org')
        results = client.getRouting('IV', '', dt, dt + 1)
        self.assertEqual(
            results,
            {'IV...': [{'priority': 1, 'start': UTCDateTime(1980, 1, 1, 0, 0),
                        'host': 'eida.rm.ingv.it', 'end': None,
                        'port': 18002}]})
        # 6 - GE.APE via webdc.eu:18001
        client = Client(host="webdc.eu", port=18001, user='test@obspy.org')
        results = client.getRouting('GE', 'APE', dt, dt + 1)
        self.assertEqual(
            results,
            {'GE...': [{'priority': 1, 'start': UTCDateTime(1993, 1, 1, 0, 0),
                        'host': 'eida.gfz-potsdam.de', 'end': None,
                        'port': 18002}]})
        # 7 - GE.APE via webdc.eu:18002
        client = Client(host="webdc.eu", port=18002, user='test@obspy.org')
        results = client.getRouting('GE', 'APE', dt, dt + 1)
        self.assertEqual(
            results,
            {'GE...': [{'priority': 1, 'start': UTCDateTime(1993, 1, 1, 0, 0),
                        'host': 'eida.gfz-potsdam.de', 'end': None,
                        'port': 18002}]})
        # 8 - unknown network 00 via webdc.eu:18002
        client = Client(host="webdc.eu", port=18002, user='test@obspy.org')
        results = client.getRouting('00', '', dt, dt + 1)
        self.assertEqual(results, {})

    def test_getInventory(self):
        """
        Tests getInventory method on various ArcLink nodes.
        """
        client = Client(user='test@obspy.org')
        dt = UTCDateTime(2010, 1, 1)
        # 1 - GE network
        result = client.getInventory('GE', 'APE', starttime=dt, endtime=dt + 1)
        self.assertTrue('GE' in result)
        self.assertTrue('GE.APE' in result)
        # 2 - GE network
        result = client.getInventory('GE', 'APE', '', 'BHE', starttime=dt,
                                     endtime=dt + 1, instruments=True)
        self.assertTrue('GE' in result)
        self.assertTrue('GE.APE' in result)
        self.assertTrue('GE.APE..BHE' in result)  # only for instruments=True
        # 3 - BW network
        result = client.getInventory('BW', 'RJOB', starttime=dt,
                                     endtime=dt + 1)
        self.assertTrue('BW' in result)
        self.assertTrue('BW.RJOB' in result)
        # 4 - BW network
        result = client.getInventory('BW', 'MANZ', '', 'EHE', starttime=dt,
                                     endtime=dt + 1, instruments=True)
        self.assertTrue('BW' in result)
        self.assertTrue('BW.MANZ' in result)
        self.assertTrue('BW.MANZ..EHE' in result)
        # 5 - unknown network 00 via webdc.eu:18002
        self.assertRaises(ArcLinkException, client.getInventory, '00', '',
                          starttime=dt, endtime=dt + 1)
        # 6 - get channel gain without PAZ
        start = UTCDateTime("1970-01-01 00:00:00")
        end = UTCDateTime("2020-10-19 00:00:00")
        result = client.getInventory('BW', 'MANZ', '', 'EHE', start, end)
        self.assertTrue('BW' in result)
        self.assertTrue('BW.MANZ' in result)
        self.assertTrue('BW.MANZ..EHE' in result)
        self.assertEqual(len(result['BW.MANZ..EHE']), 2)
        self.assertTrue('gain' in result['BW.MANZ..EHE'][0])
        self.assertTrue('paz' not in result['BW.MANZ..EHE'][0])
        # 7 - history of instruments
        # GE.SNAA sometimes needs a while therefore we use command_delay=0.1
        client = Client(user='test@obspy.org', command_delay=0.1)
        result = client.getInventory('GE', 'SNAA', '', 'BHZ', start, end,
                                     instruments=True)
        self.assertTrue('GE' in result)
        self.assertTrue('GE.SNAA' in result)
        self.assertTrue('GE.SNAA..BHZ' in result)
        self.assertEqual(len(result['GE.SNAA..BHZ']), 4)
        # sort channel results
        channel = result['GE.SNAA..BHZ']
        channel = sorted(channel, key=operator.itemgetter('starttime'))
        # check for required attributes
        self.assertEqual(channel[0].starttime, UTCDateTime("1997-03-03"))
        self.assertEqual(channel[0].endtime, UTCDateTime("1999-10-11"))
        self.assertEqual(channel[0].gain, 596224500.0)
        self.assertEqual(channel[1].starttime, UTCDateTime("1999-10-11"))
        self.assertEqual(channel[1].endtime, UTCDateTime("2003-01-10"))
        self.assertEqual(channel[1].gain, 596224500.0)
        self.assertEqual(channel[2].starttime, UTCDateTime("2003-01-10"))
        self.assertEqual(channel[2].endtime, UTCDateTime(2011, 1, 15, 9, 56))
        self.assertEqual(channel[2].gain, 588000000.0)

    def test_getInventoryTwice(self):
        """
        Requesting inventory data twice should not fail.
        """
        client = Client(user='test@obspy.org')
        dt = UTCDateTime(2009, 1, 1)
        # station
        client.getInventory('BW', 'MANZ', starttime=dt, endtime=dt + 1)
        client.getInventory('BW', 'MANZ', starttime=dt, endtime=dt + 1)
        # network
        client.getInventory('BW', starttime=dt, endtime=dt + 1)
        client.getInventory('BW', starttime=dt, endtime=dt + 1)

    def test_getWaveformWithMetadata(self):
        """
        """
        # initialize client
        client = Client(user='test@obspy.org')
        # example 1
        t = UTCDateTime("2010-08-01T12:00:00")
        st = client.getWaveform("BW", "RJOB", "", "EHZ", t, t + 60,
                                metadata=True)
        results = {
            'network': 'BW',
            '_format': 'MSEED',
            'paz': AttribDict({
                'normalization_factor': 60077000.0,
                'name': 'RJOB.2007.351.HZ',
                'sensitivity': 2516800000.0,
                'normalization_frequency': 1.0,
                'sensor_manufacturer': 'Streckeisen',
                'sensitivity_unit': 'M/S',
                'sensitivity_frequency': 0.02,
                'poles': [(-0.037004 + 0.037016j), (-0.037004 - 0.037016j),
                          (-251.33 + 0j), (-131.04 - 467.29j),
                          (-131.04 + 467.29j)],
                'gain': 60077000.0,
                'zeros': [0j, 0j],
                'sensor_model': 'STS-2'}),
            'mseed': AttribDict({
                'record_length': 512,
                'encoding': 'STEIM1',
                'filesize': 30720,
                'dataquality': 'D',
                'number_of_records': 60,
                'byteorder': '>'}),
            'coordinates': AttribDict({
                'latitude': 47.737167,
                'elevation': 860.0,
                'longitude': 12.795714}),
            'sampling_rate': 200.0,
            'station': 'RJOB',
            'location': '',
            'starttime': UTCDateTime(2010, 8, 1, 12, 0),
            'delta': 0.005,
            'calib': 1.0,
            'npts': 1370,
            'endtime': UTCDateTime(2010, 8, 1, 12, 0, 6, 845000),
            'channel': 'EHZ'}
        results['processing'] = st[0].stats['processing']
        self.assertEqual(st[0].stats, results)
        # example 2
        client = Client(user='test@obspy.org')
        st = client.getWaveform("CZ", "VRAC", "", "BHZ", t, t + 60,
                                metadata=True)
        results = {
            'network': 'CZ',
            '_format': 'MSEED',
            'paz': AttribDict({
                'normalization_factor': 60077000.0,
                'name': 'GFZ:CZ1980:STS-2/N/g=20000',
                'sensitivity': 8200000000.0,
                'normalization_frequency': 1.0,
                'sensor_manufacturer': 'Streckeisen',
                'sensitivity_unit': 'M/S',
                'sensitivity_frequency': 0.02,
                'zeros': [0j, 0j],
                'gain': 60077000.0,
                'poles': [(-0.037004 + 0.037016j), (-0.037004 - 0.037016j),
                          (-251.33 + 0j), (-131.04 - 467.29j),
                          (-131.04 + 467.29j)],
                'sensor_model': 'STS-2/N'}),
            'mseed': AttribDict({
                'record_length': 512,
                'encoding': 'STEIM1',
                'filesize': 3584,
                'dataquality': 'D',
                'number_of_records': 7,
                'byteorder': '>'}),
            'coordinates': AttribDict({
                'latitude': 49.3084,
                'elevation': 470.0,
                'longitude': 16.5933}),
            'delta': 0.025,
            'station': 'VRAC',
            'location': '',
            'starttime': UTCDateTime(2010, 8, 1, 11, 59, 59, 993400),
            'endtime': UTCDateTime(2010, 8, 1, 12, 0, 59, 993400),
            'npts': 2401,
            'calib': 1.0,
            'sampling_rate': 40.0,
            'channel': 'BHZ'}
        results['processing'] = st[0].stats['processing']
        self.assertEqual(st[0].stats, results)

    def test_getNotExistingWaveform(self):
        """
        """
        # initialize client
        client = Client(user='test@obspy.org')
        # example 1
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        self.assertRaises(ArcLinkException, client.getWaveform, 'AA', 'AAAAA',
                          '', '*', start, end)
        # example 2
        start = UTCDateTime(2038, 1, 1)
        end = start + 1
        self.assertRaises(ArcLinkException, client.getWaveform, 'BW', 'MANZ',
                          '', '*', start, end)

    def test_getWaveformWrongPattern(self):
        """
        """
        # initialize client
        client = Client(user='test@obspy.org')
        # example 1
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        self.assertRaises(ArcLinkException, client.getWaveform, 'BW', 'MAN*',
                          '', '*', start, end)

    def test_getNetworks(self):
        """
        """
        # initialize client
        client = Client(user='test@obspy.org')
        # example 1
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        result = client.getNetworks(start, end)
        self.assertTrue('BW' in result.keys())
        self.assertEqual(result['BW']['code'], 'BW')
        self.assertEqual(result['BW']['description'], 'BayernNetz')

    def test_getStations(self):
        """
        """
        # initialize client
        client = Client(user='test@obspy.org')
        # example 1
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        result = client.getStations(start, end, 'BW')
        self.assertTrue(
            AttribDict({'remark': '', 'code': 'RWMO', 'elevation': 763.0,
                        'description': 'Wildenmoos, Bavaria, BW-Net',
                        'start': UTCDateTime(2006, 7, 4, 0, 0),
                        'restricted': False, 'archive_net': '',
                        'longitude': 12.729887, 'affiliation': 'BayernNetz',
                        'depth': None, 'place': 'Wildenmoos',
                        'country': ' BW-Net', 'latitude': 47.744171,
                        'end': None}) in result)

    def test_saveWaveform(self):
        """
        Default behavior is requesting data compressed and unpack on the fly.
        """
        # initialize client
        client = Client("erde.geophysik.uni-muenchen.de", 18001,
                        user='test@obspy.org')
        start = UTCDateTime(2008, 1, 1)
        end = start + 10
        # MiniSEED
        with NamedTemporaryFile(suffix='.bz2') as tf:
            mseedfile = tf.name
            client.saveWaveform(mseedfile, 'BW', 'MANZ', '', 'EHZ', start, end)
            st = read(mseedfile)
            # MiniSEED may not start with Volume Index Control Headers (V)
            with open(mseedfile, 'rb') as fp:
                self.assertNotEqual(fp.read(8)[6:7], b"V")
            # ArcLink cuts on record base
            self.assertTrue(st[0].stats.starttime <= start)
            self.assertTrue(st[0].stats.endtime >= end)
            self.assertEqual(st[0].stats.network, 'BW')
            self.assertEqual(st[0].stats.station, 'MANZ')
            self.assertEqual(st[0].stats.location, '')
            self.assertEqual(st[0].stats.channel, 'EHZ')
        # Full SEED
        with NamedTemporaryFile(suffix='.bz2') as tf:
            fseedfile = tf.name
            client.saveWaveform(fseedfile, 'BW', 'MANZ', '', 'EHZ', start, end,
                                format='FSEED')
            st = read(fseedfile)
            # Full SEED must start with Volume Index Control Headers (V)
            with open(fseedfile, 'rb') as fp:
                self.assertEqual(fp.read(8)[6:7], b"V")
            # ArcLink cuts on record base
            self.assertTrue(st[0].stats.starttime <= start)
            self.assertTrue(st[0].stats.endtime >= end)
            self.assertEqual(st[0].stats.network, 'BW')
            self.assertEqual(st[0].stats.station, 'MANZ')
            self.assertEqual(st[0].stats.location, '')
            self.assertEqual(st[0].stats.channel, 'EHZ')

    def test_getWaveformNoCompression(self):
        """
        Disabling compression during waveform request.
        """
        # initialize client
        client = Client(user='test@obspy.org')
        start = UTCDateTime(2011, 1, 1, 0, 0)
        end = start + 10
        stream = client.getWaveform('BW', 'MANZ', '', 'EH*', start, end,
                                    compressed=False)
        self.assertEqual(len(stream), 3)
        for trace in stream:
            self.assertEqual(trace.stats.network, 'BW')
            self.assertEqual(trace.stats.station, 'MANZ')

    def test_saveWaveformNoCompression(self):
        """
        Explicitly disable compression during waveform request and save it
        directly to disk.
        """
        # initialize client
        client = Client(user='test@obspy.org')
        start = UTCDateTime(2010, 1, 1, 0, 0)
        end = start + 1
        # MiniSEED
        with NamedTemporaryFile(suffix='.bz2') as tf:
            mseedfile = tf.name
            client.saveWaveform(mseedfile, 'GE', 'APE', '', 'BHZ', start, end,
                                compressed=False)
            st = read(mseedfile)
            # MiniSEED may not start with Volume Index Control Headers (V)
            with open(mseedfile, 'rb') as fp:
                self.assertNotEqual(fp.read(8)[6:7], b"V")
            # ArcLink cuts on record base
            self.assertEqual(st[0].stats.network, 'GE')
            self.assertEqual(st[0].stats.station, 'APE')
            self.assertEqual(st[0].stats.location, '')
            self.assertEqual(st[0].stats.channel, 'BHZ')
        # Full SEED
        with NamedTemporaryFile(suffix='.bz2') as tf:
            fseedfile = tf.name
            client.saveWaveform(fseedfile, 'GE', 'APE', '', 'BHZ', start, end,
                                format='FSEED')
            st = read(fseedfile)
            # Full SEED
            client.saveWaveform(fseedfile, 'BW', 'MANZ', '', 'EHZ', start, end,
                                format='FSEED')
            # ArcLink cuts on record base
            self.assertEqual(st[0].stats.network, 'GE')
            self.assertEqual(st[0].stats.station, 'APE')
            self.assertEqual(st[0].stats.location, '')
            self.assertEqual(st[0].stats.channel, 'BHZ')

    def test_saveWaveformCompressed(self):
        """
        Tests saving compressed and not unpacked bzip2 files to disk.
        """
        # initialize client
        client = Client(user='test@obspy.org')
        start = UTCDateTime(2008, 1, 1, 0, 0)
        end = start + 1
        # MiniSEED
        with NamedTemporaryFile(suffix='.bz2') as tf:
            mseedfile = tf.name
            client.saveWaveform(mseedfile, 'GE', 'APE', '', 'BHZ', start, end,
                                unpack=False)
            # check if compressed
            with open(mseedfile, 'rb') as fp:
                self.assertEqual(fp.read(2), b'BZ')
            # importing via read should work too
            read(mseedfile)
        # Full SEED
        with NamedTemporaryFile(suffix='.bz2') as tf:
            fseedfile = tf.name
            client.saveWaveform(fseedfile, 'GE', 'APE', '', 'BHZ', start, end,
                                format="FSEED", unpack=False)
            # check if compressed
            with open(fseedfile, 'rb') as fp:
                self.assertEqual(fp.read(2), b'BZ')
            # importing via read should work too
            read(fseedfile)

    def test_getPAZ(self):
        """
        Test for the Client.getPAZ function.

        As reference the EHZ channel of MANZ is taken, the result is compared
        to the entries of the local response file of the Bavarian network.
        """
        # reference values
        zeros = [0j, 0j]
        poles = [-3.700400e-02 + 3.701600e-02j, -3.700400e-02 - 3.701600e-02j,
                 -2.513300e+02 + 0.000000e+00j, -1.310400e+02 - 4.672900e+02j,
                 -1.310400e+02 + 4.672900e+02j]
        normalization_factor = 6.0077e+07
        sensitivity = 2.5168e+09
        # initialize client
        client = Client('erde.geophysik.uni-muenchen.de', 18001,
                        user='test@obspy.org')
        # fetch poles and zeros
        dt = UTCDateTime(2009, 1, 1)
        paz = client.getPAZ('BW', 'MANZ', '', 'EHZ', dt)
        # compare instrument
        self.assertEqual(normalization_factor, paz.normalization_factor)
        self.assertEqual(poles, paz.poles)
        self.assertEqual(zeros, paz.zeros)
        self.assertAlmostEqual(sensitivity / 1e9, paz.sensitivity / 1e9, 4)
        # PAZ over multiple channels should raise an exception
        self.assertRaises(ArcLinkException, client.getPAZ, 'BW', 'MANZ', '',
                          'EH*', dt)

    def test_getPAZ2(self):
        """
        Test for the Client.getPAZ function for erde.geophysik.uni-muenchen.de.
        """
        poles = [-3.700400e-02 + 3.701600e-02j, -3.700400e-02 - 3.701600e-02j]
        dt = UTCDateTime(2009, 1, 1)
        client = Client("erde.geophysik.uni-muenchen.de", 18001,
                        user='test@obspy.org')
        # fetch poles and zeros
        paz = client.getPAZ('BW', 'MANZ', '', 'EHZ', dt)
        self.assertEqual(len(poles), 2)
        self.assertEqual(poles, paz['poles'][:2])

    def test_saveResponse(self):
        """
        Fetches and stores response information as Dataless SEED volume.
        """
        client = Client(user='test@obspy.org')
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            # Dataless SEED
            client.saveResponse(tempfile, 'BW', 'MANZ', '', 'EHZ', start, end)
            with open(tempfile, 'rb') as fp:
                self.assertEqual(fp.read(8), b"000001V ")

        # Try again but write to a BytesIO instance.
        file_object = io.BytesIO()
        client = Client(user='test@obspy.org')
        start = UTCDateTime(2008, 1, 1)
        end = start + 1
        # Dataless SEED
        client.saveResponse(file_object, 'BW', 'MANZ', '', 'EHZ', start, end)
        file_object.seek(0, 0)
        self.assertEqual(file_object.read(8), b"000001V ")

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
               'sensitivity': 2516800000.0,
               'zeros': [0j, 0j]}
        dat1 = np.array([288, 300, 292, 285, 265, 287, 279, 250, 278, 278])
        dat2 = np.array([445, 432, 425, 400, 397, 471, 426, 390, 450, 442])
        # Retrieve data via ArcLink
        client = Client(host="webdc.eu", port=18001, user='test@obspy.org')
        t = UTCDateTime("2009-08-24 00:20:03")
        st = client.getWaveform("BW", "RJOB", "", "EHZ", t, t + 30)
        # original but deprecated call
        # poles_zeros = list(client.getPAZ("BW", "RJOB", "", "EHZ",
        #                                 t, t+30).values())[0]
        poles_zeros = client.getPAZ("BW", "RJOB", "", "EHZ", t)
        self.assertEqual(paz['gain'], poles_zeros['gain'])
        self.assertEqual(paz['poles'], poles_zeros['poles'])
        self.assertEqual(paz['sensitivity'], poles_zeros['sensitivity'])
        self.assertEqual(paz['zeros'], poles_zeros['zeros'])
        self.assertEqual('BW', st[0].stats['network'])
        self.assertEqual('RJOB', st[0].stats['station'])
        self.assertEqual(200.0, st[0].stats['sampling_rate'])
        self.assertEqual(6001, st[0].stats['npts'])
        self.assertEqual(
            '2009-08-24T00:20:03.000000Z', str(st[0].stats['starttime']))
        np.testing.assert_array_equal(dat1, st[0].data[:10])
        np.testing.assert_array_equal(dat2, st[0].data[-10:])

    def test_issue311(self):
        """
        Testing issue #311.
        """
        client = Client("webdc.eu", 18001, user='test@obspy.org')
        t = UTCDateTime("2009-08-20 04:03:12")
        # 1
        st = client.getWaveform("BW", "MANZ", "", "EH*", t - 3, t + 15,
                                metadata=False)
        self.assertEqual(len(st), 3)
        self.assertTrue('paz' not in st[0].stats)
        self.assertTrue('coordinates' not in st[0].stats)
        # 2
        st = client.getWaveform("BW", "MANZ", "", "EH*", t - 3, t + 15,
                                metadata=True)
        self.assertEqual(len(st), 3)
        self.assertTrue('paz' in st[0].stats)
        self.assertTrue('coordinates' in st[0].stats)

    def test_issue372(self):
        """
        Test case for issue #372.
        """
        dt = UTCDateTime("20120729070000")
        client = Client(user='test@obspy.org')
        st = client.getWaveform("BS", "JMB", "", "BH*", dt, dt + 7200,
                                metadata=True)
        for tr in st:
            self.assertTrue('paz' in tr.stats)
            self.assertTrue('coordinates' in tr.stats)
            self.assertTrue('poles' in tr.stats.paz)
            self.assertTrue('zeros' in tr.stats.paz)
            self.assertTrue('latitude' in tr.stats.coordinates)

    def test_getInventoryInstrumentChange(self):
        """
        Check results of getInventory if instrumentation has been changed.

        Sensitivity change for GE.SNAA..BHZ at 2003-01-10T00:00:00
        """
        client = Client(user='test@obspy.org')
        # one instrument in given time span
        dt = UTCDateTime("2003-01-09T00:00:00")
        inv = client.getInventory("GE", "SNAA", "", "BHZ", dt, dt + 10,
                                  instruments=True, route=False)
        self.assertTrue(len(inv['GE.SNAA..BHZ']), 1)
        # two instruments in given time span
        dt = UTCDateTime("2003-01-09T23:59:59")
        inv = client.getInventory("GE", "SNAA", "", "BHZ", dt, dt + 10,
                                  instruments=True, route=False)
        self.assertTrue(len(inv['GE.SNAA..BHZ']), 2)
        # one instrument in given time span
        dt = UTCDateTime("2003-01-10T00:00:00")
        inv = client.getInventory("GE", "SNAA", "", "BHZ", dt, dt + 10,
                                  instruments=True, route=False)
        self.assertTrue(len(inv['GE.SNAA..BHZ']), 1)

    def test_getWaveformInstrumentChange(self):
        """
        Check results of getWaveform if instrumentation has been changed.

        Sensitivity change for GE.SNAA..BHZ at 2003-01-10T00:00:00
        """
        client = Client(user='test@obspy.org')
        # one instrument in given time span
        dt = UTCDateTime("2003-01-09T00:00:00")
        st = client.getWaveform("GE", "SNAA", "", "BHZ", dt, dt + 10,
                                metadata=True)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.paz.sensitivity, 596224500.0)
        # two instruments in given time span
        dt = UTCDateTime("2003-01-09T23:59:00")
        st = client.getWaveform("GE", "SNAA", "", "BHZ", dt, dt + 120,
                                metadata=True)
        # results into two traces
        self.assertEqual(len(st), 2)
        # with different PAZ
        st.sort()
        self.assertEqual(st[0].stats.paz.sensitivity, 596224500.0)
        self.assertEqual(st[1].stats.paz.sensitivity, 588000000.0)
        # one instrument in given time span
        dt = UTCDateTime("2003-01-10T01:00:00")
        st = client.getWaveform("GE", "SNAA", "", "BHZ", dt, dt + 10,
                                metadata=True)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.paz.sensitivity, 588000000.0)

    def test_init(self):
        """
        Testing client initialization.
        """
        # user is a required keyword - raises now a deprecation warning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', DeprecationWarning)
            self.assertRaises(DeprecationWarning, Client)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
