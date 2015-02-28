# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import numpy as np

from obspy.core.preview import createPreview
from obspy.core.trace import Trace
from obspy.core.utcdatetime import UTCDateTime
from obspy.db.client import Client
from obspy.db.db import WaveformChannel, WaveformFile, WaveformPath


class ClientTestCase(unittest.TestCase):
    """
    Test suite for obspy.db.client.
    """
    # unfortunately no py2.6 syntax
    # @classmethod
    # def setUpClass(cls):
    def __init__(self, *args, **kwargs):
        super(ClientTestCase, self).__init__(*args, **kwargs)
        # Create a in memory database only once for test suite
        url = 'sqlite:///:memory:'
        self.client = Client(url)
        # add paths
        session = self.client.session()
        path1 = WaveformPath({'path': '/path/to/1'})
        path2 = WaveformPath({'path': '/path/to/2'})
        session.add_all([path1, path2])
        # add files
        file1 = WaveformFile(
            {'file': 'file_001.mseed', 'size': 2000,
                'mtime': UTCDateTime('20120101').timestamp, 'format': 'MSEED'})
        file2 = WaveformFile(
            {'file': 'file_002.mseed', 'size': 2000,
                'mtime': UTCDateTime('20120102').timestamp, 'format': 'MSEED'})
        file3 = WaveformFile(
            {'file': 'file_001.gse2', 'size': 2000,
                'mtime': UTCDateTime('20120102').timestamp, 'format': 'GSE2'})
        path1.files.append(file1)
        path1.files.append(file2)
        path2.files.append(file3)
        session.add_all([file1, file2, file3])
        # add channels
        channel1 = WaveformChannel(
            {'network': 'BW', 'station': 'MANZ',
                'location': '', 'channel': 'EHZ',
                'starttime':
                UTCDateTime('2012-01-01 00:00:00.000000').datetime,
                'endtime': UTCDateTime('2012-01-01 23:59:59.999999').datetime,
                'npts': 3000, 'sampling_rate': 100.0})
        channel2 = WaveformChannel(
            {'network': 'BW', 'station': 'MANZ',
                'location': '', 'channel': 'EHZ',
                'starttime':
                UTCDateTime('2012-01-02 01:00:00.000000').datetime,
                'endtime':
                UTCDateTime('2012-01-02 23:59:59.999999').datetime,
                'npts': 3000,
                'sampling_rate': 100.0})
        # create a channel with preview
        header = {'network': 'GE', 'station': 'FUR',
                  'location': '00', 'channel': 'BHZ',
                  'starttime': UTCDateTime('2012-01-01 00:00:00.000000'),
                  'sampling_rate': 100.0}
        # linear trend
        data = np.linspace(0, 1, 3000000)
        # some peaks
        data[20000] = 15
        data[20001] = -15
        data[1000000] = 22
        data[1000001] = -22
        data[2000000] = 14
        data[2000001] = -14
        tr = Trace(data=data, header=header)
        self.preview = createPreview(tr, 30).data
        header = dict(tr.stats)
        header['starttime'] = tr.stats.starttime.datetime
        header['endtime'] = tr.stats.endtime.datetime
        channel3 = WaveformChannel(header)
        channel3.preview = self.preview.dumps()
        file1.channels.append(channel1)
        file2.channels.append(channel2)
        file3.channels.append(channel3)
        session.add_all([channel1, channel2, channel3])
        session.commit()
        session.close()

    def test_getNetworkIds(self):
        """
        Tests for method getNetworkIds.
        """
        data = self.client.getNetworkIDs()
        self.assertEqual(len(data), 2)
        self.assertTrue('BW' in data)
        self.assertTrue('GE' in data)

    def test_getStationIds(self):
        """
        Tests for method getStationIds.
        """
        # 1 - all
        data = self.client.getStationIds()
        self.assertEqual(len(data), 2)
        self.assertTrue('MANZ' in data)
        self.assertTrue('FUR' in data)
        # 2 - BW network
        data = self.client.getStationIds(network='BW')
        self.assertEqual(len(data), 1)
        self.assertTrue('MANZ' in data)
        # 3 - not existing network
        data = self.client.getStationIds(network='XX')
        self.assertEqual(len(data), 0)

    def test_getLocationIds(self):
        """
        Tests for method getLocationIds.
        """
        data = self.client.getLocationIds()
        self.assertEqual(len(data), 2)
        self.assertTrue('' in data)
        self.assertTrue('00' in data)
        # 2 - BW network
        data = self.client.getLocationIds(network='BW')
        self.assertEqual(len(data), 1)
        self.assertTrue('' in data)
        # 3 - not existing network
        data = self.client.getLocationIds(network='XX')
        self.assertEqual(len(data), 0)
        # 4 - MANZ station
        data = self.client.getLocationIds(station='MANZ')
        self.assertEqual(len(data), 1)
        self.assertTrue('' in data)
        # 5 - not existing station
        data = self.client.getLocationIds(station='XXXXX')
        self.assertEqual(len(data), 0)
        # 4 - GE network, FUR station
        data = self.client.getLocationIds(network='GE', station='FUR')
        self.assertEqual(len(data), 1)
        self.assertTrue('00' in data)

    def test_getChannelIds(self):
        """
        Tests for method getChannelIds.
        """
        data = self.client.getChannelIds()
        self.assertEqual(len(data), 2)
        self.assertTrue('EHZ' in data)
        self.assertTrue('BHZ' in data)

    def test_getEndtimes(self):
        """
        Tests for method getEndtimes.
        """
        # 1
        data = self.client.getEndtimes()
        self.assertEqual(len(data), 2)
        self.assertEqual(data['BW.MANZ..EHZ'],
                         UTCDateTime(2012, 1, 2, 23, 59, 59, 999999))
        self.assertEqual(data['GE.FUR.00.BHZ'],
                         UTCDateTime(2012, 1, 1, 8, 19, 59, 990000))
        # 2 - using wildcards
        data = self.client.getEndtimes(network='?W', station='M*', location='')
        self.assertEqual(len(data), 1)
        self.assertEqual(data['BW.MANZ..EHZ'],
                         UTCDateTime(2012, 1, 2, 23, 59, 59, 999999))
        # 3 - no data
        data = self.client.getEndtimes(network='GE', station='*', location='')
        self.assertEqual(len(data), 0)

    def test_getWaveformPath(self):
        """
        Tests for method getWaveformPath.
        """
        # 1
        dt = UTCDateTime('2012-01-01 00:00:00.000000')
        data = self.client.getWaveformPath(starttime=dt, endtime=dt + 5)
        self.assertEqual(len(data), 2)
        self.assertEqual(data['BW.MANZ..EHZ'],
                         [os.path.join('/path/to/1', 'file_001.mseed')])
        self.assertEqual(data['GE.FUR.00.BHZ'],
                         [os.path.join('/path/to/2', 'file_001.gse2')])
        # 2 - no data
        dt = UTCDateTime('2012-01-01 00:00:00.000000')
        data = self.client.getWaveformPath(starttime=dt - 5, endtime=dt - 4)
        self.assertEqual(data, {})
        # 3
        dt = UTCDateTime('2012-01-02 01:00:00.000000')
        data = self.client.getWaveformPath(starttime=dt, endtime=dt + 5)
        self.assertEqual(len(data), 1)
        self.assertEqual(data['BW.MANZ..EHZ'],
                         [os.path.join('/path/to/1', 'file_002.mseed')])
        # 4 - filter by network
        dt = UTCDateTime('2012-01-01 00:00:00.000000')
        dt2 = UTCDateTime('2012-01-02 23:00:00.000000')
        data = self.client.getWaveformPath(starttime=dt, endtime=dt2,
                                           network='BW')
        self.assertEqual(len(data), 1)
        self.assertEqual(data['BW.MANZ..EHZ'],
                         [os.path.join('/path/to/1', 'file_001.mseed'),
                          os.path.join('/path/to/1', 'file_002.mseed')])
        # 5 - filter by network and station using wildcards
        data = self.client.getWaveformPath(starttime=dt, endtime=dt2,
                                           network='BW', station='MA*')
        self.assertEqual(len(data), 1)
        # 6 - filter by channel and location
        data = self.client.getWaveformPath(starttime=dt, endtime=dt2,
                                           channel='?HZ', location='')
        self.assertEqual(len(data), 1)

    def test_getPreview(self):
        """
        Tests for method getPreview.
        """
        # 1
        dt = UTCDateTime('2012-01-01 00:00:00.000000')
        dt2 = UTCDateTime('2012-01-01T08:19:30.000000Z')
        st = self.client.getPreview(starttime=dt, endtime=dt2)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].id, 'GE.FUR.00.BHZ')
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('2012-01-01T00:00:00.000000Z'))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime('2012-01-01T08:19:30.000000Z'))
        self.assertEqual(st[0].stats.delta, 30.0)
        self.assertEqual(st[0].stats.npts, 1000)
        self.assertEqual(st[0].stats.preview, True)
        np.testing.assert_equal(st[0].data, self.preview)
        # 2 - no data
        st = self.client.getPreview(network='XX', starttime=dt, endtime=dt + 2)
        self.assertEqual(len(st), 0)
        # 3 - trimmed
        dt = UTCDateTime('2012-01-01 00:00:00.000000')
        dt2 = UTCDateTime('2012-01-01T04:09:30.000000Z')
        st = self.client.getPreview(network='G?', location='00', station='*',
                                    starttime=dt, endtime=dt2)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.npts, 500)
        # 4 - using trace_ids and pad=True
        dt = UTCDateTime('2011-12-31 00:00:00.000000')
        dt2 = UTCDateTime('2012-01-01T04:09:30.000000Z')
        st = self.client.getPreview(trace_ids=['GE.FUR.00.BHZ',
                                               'GE.FUR.00.BHN'], pad=True,
                                    starttime=dt, endtime=dt2)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.npts, 3380)


def suite():
    try:
        import sqlite3  # @UnusedImport # NOQA
    except ImportError:
        # skip the whole test suite if module sqlite3 is missing
        return unittest.makeSuite(object, 'test')
    else:
        return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
