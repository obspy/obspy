#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest

import numpy as np

from obspy import UTCDateTime, read
from obspy.io.ah.core import _is_ah, _read_ah, _write_ah1, _read_ah1
from obspy.core.util import NamedTemporaryFile


class CoreTestCase(unittest.TestCase):
    """
    AH (Ad Hoc) file test suite.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_is_ah(self):
        """
        Testing AH file format.
        """
        # AH v1
        testfile = os.path.join(self.path, 'TSG', 'BRV.TSG.DS.lE21.resp')
        self.assertTrue(_is_ah(testfile))
        testfile = os.path.join(self.path, 'TSG', 'BRV.TSG.KSM.sE12.resp')
        self.assertTrue(_is_ah(testfile))
        testfile = os.path.join(self.path, 'ah1.f')
        self.assertTrue(_is_ah(testfile))
        testfile = os.path.join(self.path, 'ah1.c')
        self.assertTrue(_is_ah(testfile))
        testfile = os.path.join(self.path, 'ah1.t')
        self.assertTrue(_is_ah(testfile))
        testfile = os.path.join(self.path, 'hrv.lh.zne')
        self.assertTrue(_is_ah(testfile))

        # AH v2
        testfile = os.path.join(self.path, 'ah2.f')
        self.assertTrue(_is_ah(testfile))
        testfile = os.path.join(self.path, 'ah2.f-e')
        self.assertTrue(_is_ah(testfile))
        testfile = os.path.join(self.path, 'ah2.c')
        self.assertTrue(_is_ah(testfile))
        testfile = os.path.join(self.path, 'ah2.t')
        self.assertTrue(_is_ah(testfile))

        # non AH files
        testfile = os.path.join(self.path, 'TSG', 'BRV.TSG.DS.lE21.asc')
        self.assertFalse(_is_ah(testfile))
        testfile = os.path.join(self.path, 'TSG', 'BRV.TSG.KSM.sE12.asc')
        self.assertFalse(_is_ah(testfile))
        testfile = os.path.join(self.path, 'TSG', 'Readme_TSG_response.txt')
        self.assertFalse(_is_ah(testfile))

    def test_read(self):
        """
        Testing reading AH file format using read() function.
        """
        # AH v1
        testfile = os.path.join(self.path, 'hrv.lh.zne')
        st = read(testfile)
        self.assertEqual(len(st), 3)
        testfile = os.path.join(self.path, 'ah1.f')
        st = read(testfile)
        self.assertEqual(len(st), 4)
        # not supported data types (vector, complex, tensor)
        testfile = os.path.join(self.path, 'ah1.c')
        self.assertRaises(NotImplementedError, _read_ah, testfile)
        testfile = os.path.join(self.path, 'ah1.t')
        self.assertRaises(NotImplementedError, _read_ah, testfile)

        # AH v2
        # float
        testfile = os.path.join(self.path, 'ah2.f')
        st = read(testfile)
        self.assertEqual(len(st), 4)

    def test_read_ah(self):
        """
        Testing reading AH file format using _read_ah() function.
        """
        # AH v1
        testfile = os.path.join(self.path, 'ah1.f')
        st = _read_ah(testfile)
        self.assertEqual(len(st), 4)
        tr = st[0]
        ah = tr.stats.ah
        # station
        self.assertEqual(ah.version, '1.0')
        self.assertEqual(ah.station.code, 'RSCP')
        self.assertEqual(ah.station.channel, 'IPZ')
        self.assertEqual(ah.station.type, 'null')
        self.assertAlmostEqual(ah.station.latitude, 35.599899, 6)
        self.assertAlmostEqual(ah.station.longitude, -85.568802, 6)
        self.assertEqual(ah.station.elevation, 481.0)
        self.assertAlmostEqual(ah.station.gain, 64200.121094, 6)
        self.assertEqual(len(ah.station.poles), 24)
        self.assertEqual(len(ah.station.zeros), 7)
        # event
        self.assertEqual(ah.event.latitude, 0.0)
        self.assertEqual(ah.event.longitude, 0.0)
        self.assertEqual(ah.event.depth, 0.0)
        self.assertEqual(ah.event.origin_time, None)
        self.assertEqual(ah.event.comment, 'null')
        # record
        self.assertEqual(ah.record.type, 1)
        self.assertEqual(ah.record.ndata, 720)
        self.assertEqual(tr.stats.npts, 720)
        self.assertEqual(len(tr), 720)
        self.assertEqual(tr.data.dtype, np.float64)
        self.assertAlmostEqual(ah.record.delta, 0.25, 6)
        self.assertAlmostEqual(tr.stats.delta, 0.25, 6)
        self.assertAlmostEqual(ah.record.max_amplitude, 0.0, 6)
        dt = UTCDateTime(1984, 4, 20, 6, 42, 0, 120000)
        self.assertEqual(ah.record.start_time, dt)
        self.assertEqual(tr.stats.starttime, dt)
        self.assertEqual(ah.record.comment, 'null')
        self.assertEqual(ah.record.log, 'gdsn_tape;demeaned;')
        # data
        np.testing.assert_array_almost_equal(tr.data[:4], np.array([
            -731.41247559, -724.41247559, -622.41247559, -470.4125061]))
        np.testing.assert_array_almost_equal(tr.data[-4:], np.array([
            -1421.41247559, 118.58750153, 88.58750153, -982.41247559]))

        # not supported data types (vector, complex, tensor)
        testfile = os.path.join(self.path, 'ah1.c')
        self.assertRaises(NotImplementedError, _read_ah, testfile)
        testfile = os.path.join(self.path, 'ah1.t')
        self.assertRaises(NotImplementedError, _read_ah, testfile)

        # AH v2
        testfile = os.path.join(self.path, 'ah2.f')
        st = _read_ah(testfile)
        self.assertEqual(len(st), 4)
        tr = st[0]
        ah = tr.stats.ah
        self.assertEqual(ah.version, '2.0')
        # station
        self.assertEqual(ah.station.code, 'RSCP')
        self.assertEqual(ah.station.channel, 'IPZ')
        self.assertEqual(ah.station.type, 'null')
        self.assertAlmostEqual(ah.station.latitude, 35.599899, 6)
        self.assertAlmostEqual(ah.station.longitude, -85.568802, 6)
        self.assertEqual(ah.station.elevation, 481.0)
        self.assertAlmostEqual(ah.station.gain, 64200.121094, 6)
        self.assertEqual(len(ah.station.poles), 24)
        self.assertEqual(len(ah.station.zeros), 7)
        # event
        self.assertEqual(ah.event.latitude, 0.0)
        self.assertEqual(ah.event.longitude, 0.0)
        self.assertEqual(ah.event.depth, 0.0)
        self.assertEqual(ah.event.origin_time, None)
        self.assertEqual(ah.event.comment, 'null')
        # record
        self.assertEqual(ah.record.type, 1)
        self.assertEqual(ah.record.ndata, 720)
        self.assertEqual(tr.stats.npts, 720)
        self.assertEqual(len(tr), 720)
        self.assertEqual(tr.data.dtype, np.float64)
        self.assertAlmostEqual(ah.record.delta, 0.25, 6)
        self.assertAlmostEqual(tr.stats.delta, 0.25, 6)
        self.assertAlmostEqual(ah.record.max_amplitude, 0.0, 6)
        dt = UTCDateTime(1984, 4, 20, 6, 42, 0, 120000)
        self.assertEqual(ah.record.start_time, dt)
        self.assertEqual(tr.stats.starttime, dt)
        self.assertEqual(ah.record.comment, 'null')
        self.assertEqual(ah.record.log, 'gdsn_tape;demeaned;')
        # data
        np.testing.assert_array_almost_equal(tr.data[:4], np.array([
            -731.41247559, -724.41247559, -622.41247559, -470.4125061]))
        np.testing.assert_array_almost_equal(tr.data[-4:], np.array([
            -1421.41247559, 118.58750153, 88.58750153, -982.41247559]))

        # not supported data types (vector, complex, tensor)
        testfile = os.path.join(self.path, 'ah2.t')
        self.assertRaises(NotImplementedError, _read_ah, testfile)

    def test_tsg(self):
        """
        Test reading AH v1 files of the STsR-TSG System at Borovoye.

        .. seealso:: https://www.ldeo.columbia.edu/res/pi/Monitoring/Data/
        """
        # 1 - BRV.TSG.DS.lE21
        testfile = os.path.join(self.path, 'TSG', 'BRV.TSG.DS.lE21.resp')
        st = _read_ah(testfile)
        self.assertEqual(len(st), 1)
        tr = st[0]
        ah = tr.stats.ah
        self.assertEqual(ah.version, '1.0')
        # station
        self.assertEqual(ah.station.code, 'BRVK')
        self.assertEqual(ah.station.channel, 'lE21')
        self.assertEqual(ah.station.type, 'TSG-DS')
        self.assertAlmostEqual(ah.station.latitude, 53.058060, 6)
        self.assertAlmostEqual(ah.station.longitude, 70.282799, 6)
        self.assertEqual(ah.station.elevation, 300.0)
        self.assertAlmostEqual(ah.station.gain, 0.05, 6)
        self.assertAlmostEqual(ah.station.normalization, 40.009960, 6)
        self.assertAlmostEqual(ah.station.longitude, 70.282799, 6)
        # calibration
        self.assertEqual(len(ah.station.poles), 7)
        self.assertAlmostEqual(ah.station.poles[0],
                               complex(-1.342653e-01, 1.168836e-01), 6)
        self.assertAlmostEqual(ah.station.poles[1],
                               complex(-1.342653e-01, -1.168836e-01), 6)
        self.assertEqual(len(ah.station.zeros), 4)
        self.assertAlmostEqual(ah.station.zeros[0], complex(0.0, 0.0), 6)
        self.assertAlmostEqual(ah.station.zeros[1], complex(0.0, 0.0), 6)
        self.assertAlmostEqual(ah.station.zeros[2], complex(0.0, 0.0), 6)
        self.assertAlmostEqual(ah.station.zeros[3], complex(0.0, 0.0), 6)
        # event
        self.assertAlmostEqual(ah.event.latitude, 49.833000, 6)
        self.assertAlmostEqual(ah.event.longitude, 78.807999, 6)
        self.assertEqual(ah.event.depth, 0.5)
        self.assertEqual(ah.event.origin_time, UTCDateTime(1988, 2, 8, 15, 23))
        self.assertEqual(ah.event.comment, 'Calibration_for_hg_TSG')
        # record
        self.assertEqual(ah.record.type, 1)
        self.assertEqual(ah.record.ndata, 225)
        self.assertEqual(tr.stats.npts, 225)
        self.assertEqual(len(tr), 225)
        self.assertEqual(tr.data.dtype, np.float64)
        self.assertAlmostEqual(ah.record.delta, 0.312, 6)
        self.assertAlmostEqual(tr.stats.delta, 0.312, 6)
        self.assertAlmostEqual(ah.record.max_amplitude, 785.805786, 6)
        dt = UTCDateTime(1988, 2, 8, 15, 24, 50.136002)
        self.assertEqual(ah.record.start_time, dt)
        self.assertEqual(tr.stats.starttime, dt)
        self.assertEqual(ah.record.abscissa_min, 0.0)
        self.assertEqual(ah.record.comment, 'DS response in counts/nm;')
        self.assertEqual(ah.record.log,
                         'brv2ah: ahtedit;demeaned;modhead;modhead;ahtedit;')
        # extras
        self.assertEqual(len(ah.extras), 21)
        self.assertEqual(ah.extras[0], 0.0)
        self.assertAlmostEqual(ah.extras[1], 0.1, 6)
        self.assertAlmostEqual(ah.extras[2], 0.1, 6)
        self.assertEqual(ah.extras[3], 0.0)
        # data
        np.testing.assert_array_almost_equal(tr.data[:24], np.array([
            -1.19425595, -1.19425595, -1.19425595, -1.19425595,
            -1.19425595, -1.19425595, -1.19425595, -1.19425595,
            -1.19425595, -1.19425595, -1.19425595, -1.19425595,
            -1.19425595, -1.19425595, -1.19425595, -1.19425595,
            -1.19425595, -1.19425595, -1.19425595, -1.19425595,
            52.8057518, 175.80580139, 322.80578613, 463.80578613]))
        np.testing.assert_array_almost_equal(tr.data[-4:], np.array([
            1.80574405, 2.80574393, 3.80574393, 3.80574393]))

    def test_write_ah1(self):
        """
        Testing writing AH1 file format using _write_ah1() function.
        """
        # AH v1
        testfile = os.path.join(self.path, 'st.ah')
        stream_orig = _read_ah(testfile)

        with NamedTemporaryFile() as tf:
            tmpfile = tf.name + '.AH'
            # write testfile
            _write_ah1(stream_orig, tmpfile)
            # read again
            st = _read_ah1(tmpfile)
            self.assertEqual(len(st), 1)
            tr = st[0]
            ah = tr.stats.ah
            stats = tr.stats
            # stream header
            self.assertEqual(stats.network, '')
            self.assertEqual(stats.station, 'ALE')
            self.assertEqual(stats.location, '')
            self.assertEqual(stats.channel, 'VHZ')
            starttime = UTCDateTime(1994, 6, 9, 0, 40, 45)
            endtime = UTCDateTime(1994, 6, 12, 8, 55, 4, 724522)
            self.assertEqual(stats.starttime, starttime)
            self.assertEqual(stats.endtime, endtime)
            self.assertAlmostEqual(stats.sampling_rate, 0.100000, 6)
            self.assertAlmostEqual(stats.delta, 9.999990, 6)
            self.assertEqual(stats.npts, 28887)
            self.assertEqual(len(tr), 28887)
            self.assertEqual(stats.calib, 1.0)

            # station
            self.assertEqual(ah.version, '1.0')
            self.assertEqual(ah.station.code, 'ALE')
            self.assertEqual(ah.station.channel, 'VHZ')
            self.assertEqual(ah.station.type, 'Global S')
            self.assertEqual(ah.station.latitude, 82.50330352783203)
            self.assertEqual(ah.station.longitude, -62.349998474121094)
            self.assertEqual(ah.station.elevation, 60.0)
            self.assertEqual(ah.station.gain, 265302864.0)
            self.assertEqual(len(ah.station.poles), 13)
            self.assertEqual(len(ah.station.zeros), 6)
            # event
            self.assertEqual(ah.event.latitude, -13.872200012207031)
            self.assertEqual(ah.event.longitude, -67.51249694824219)
            self.assertEqual(ah.event.depth, 640000.0)
            origintime = UTCDateTime(1994, 6, 9, 0, 33, 16)
            self.assertEqual(ah.event.origin_time, origintime)
            self.assertEqual(ah.event.comment, 'null')
            # record
            self.assertEqual(ah.record.type, 1)
            self.assertEqual(ah.record.ndata, 28887)
            self.assertEqual(tr.data.dtype, np.float64)
            self.assertAlmostEqual(ah.record.delta, 9.999990, 6)
            self.assertEqual(ah.record.max_amplitude, 9.265750885009766)
            rstarttime = UTCDateTime(1994, 6, 9, 0, 40, 45)
            self.assertEqual(ah.record.start_time, rstarttime)
            comment = 'Comp azm=0.0,inc=-90.0; Disp (m);'
            self.assertEqual(ah.record.comment, comment)
            self.assertEqual(ah.record.log, 'null')
            # data
            np.testing.assert_array_almost_equal(tr.data[:4], np.array([
                -236., -242., -252., -262.]))
            np.testing.assert_array_almost_equal(tr.data[-4:], np.array([
                101., 106., 107., 104.]))
