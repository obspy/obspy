#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from obspy import UTCDateTime, read
from obspy.io.ah.core import _is_ah, _read_ah, _write_ah1, _read_ah1
from obspy.core.util import NamedTemporaryFile
import pytest


class TestCore():
    """
    AH (Ad Hoc) file test suite.
    """
    def test_is_ah(self, testdata, datapath):
        """
        Testing AH file format.
        """
        # AH v1
        testfile = datapath / 'TSG' / 'BRV.TSG.DS.lE21.resp'
        assert _is_ah(testfile)
        testfile = datapath / 'TSG' / 'BRV.TSG.KSM.sE12.resp'
        assert _is_ah(testfile)
        testfile = testdata['ah1.f']
        assert _is_ah(testfile)
        testfile = testdata['ah1.c']
        assert _is_ah(testfile)
        testfile = testdata['ah1.t']
        assert _is_ah(testfile)
        testfile = testdata['hrv.lh.zne']
        assert _is_ah(testfile)

        # AH v2
        testfile = testdata['ah2.f']
        assert _is_ah(testfile)
        testfile = testdata['ah2.f-e']
        assert _is_ah(testfile)
        testfile = testdata['ah2.c']
        assert _is_ah(testfile)
        testfile = testdata['ah2.t']
        assert _is_ah(testfile)

        # non AH files
        testfile = datapath / 'TSG' / 'BRV.TSG.DS.lE21.asc'
        assert not _is_ah(testfile)
        testfile = datapath / 'TSG' / 'BRV.TSG.KSM.sE12.asc'
        assert not _is_ah(testfile)
        testfile = datapath / 'TSG' / 'Readme_TSG_response.txt'
        assert not _is_ah(testfile)

    def test_read(self, testdata):
        """
        Testing reading AH file format using read() function.
        """
        # AH v1
        testfile = testdata['hrv.lh.zne']
        st = read(testfile)
        assert len(st) == 3
        testfile = testdata['ah1.f']
        st = read(testfile)
        assert len(st) == 4
        # not supported data types (vector, complex, tensor)
        testfile = testdata['ah1.c']
        with pytest.raises(NotImplementedError):
            _read_ah(testfile)
        testfile = testdata['ah1.t']
        with pytest.raises(NotImplementedError):
            _read_ah(testfile)

        # AH v2
        # float
        testfile = testdata['ah2.f']
        st = read(testfile)
        assert len(st) == 4

    def test_read_ah(self, testdata):
        """
        Testing reading AH file format using _read_ah() function.
        """
        # AH v1
        testfile = testdata['ah1.f']
        st = _read_ah(testfile)
        assert len(st) == 4
        tr = st[0]
        ah = tr.stats.ah
        # station
        assert ah.version == '1.0'
        assert ah.station.code == 'RSCP'
        assert ah.station.channel == 'IPZ'
        assert ah.station.type == 'null'
        assert round(abs(ah.station.latitude-35.599899), 6) == 0
        assert round(abs(ah.station.longitude--85.568802), 6) == 0
        assert ah.station.elevation == 481.0
        assert round(abs(ah.station.gain-64200.121094), 6) == 0
        assert len(ah.station.poles) == 24
        assert len(ah.station.zeros) == 7
        # event
        assert ah.event.latitude == 0.0
        assert ah.event.longitude == 0.0
        assert ah.event.depth == 0.0
        assert ah.event.origin_time is None
        assert ah.event.comment == 'null'
        # record
        assert ah.record.type == 1
        assert ah.record.ndata == 720
        assert tr.stats.npts == 720
        assert len(tr) == 720
        assert tr.data.dtype == np.float64
        assert round(abs(ah.record.delta-0.25), 6) == 0
        assert round(abs(tr.stats.delta-0.25), 6) == 0
        assert round(abs(ah.record.max_amplitude-0.0), 6) == 0
        dt = UTCDateTime(1984, 4, 20, 6, 42, 0, 120000)
        assert ah.record.start_time == dt
        assert tr.stats.starttime == dt
        assert ah.record.comment == 'null'
        assert ah.record.log == 'gdsn_tape;demeaned;'
        # data
        np.testing.assert_array_almost_equal(tr.data[:4], np.array([
            -731.41247559, -724.41247559, -622.41247559, -470.4125061]))
        np.testing.assert_array_almost_equal(tr.data[-4:], np.array([
            -1421.41247559, 118.58750153, 88.58750153, -982.41247559]))

        # not supported data types (vector, complex, tensor)
        testfile = testdata['ah1.c']
        with pytest.raises(NotImplementedError):
            _read_ah(testfile)
        testfile = testdata['ah1.t']
        with pytest.raises(NotImplementedError):
            _read_ah(testfile)

        # AH v2
        testfile = testdata['ah2.f']
        st = _read_ah(testfile)
        assert len(st) == 4
        tr = st[0]
        ah = tr.stats.ah
        assert ah.version == '2.0'
        # station
        assert ah.station.code == 'RSCP'
        assert ah.station.channel == 'IPZ'
        assert ah.station.type == 'null'
        assert round(abs(ah.station.latitude-35.599899), 6) == 0
        assert round(abs(ah.station.longitude--85.568802), 6) == 0
        assert ah.station.elevation == 481.0
        assert round(abs(ah.station.gain-64200.121094), 6) == 0
        assert len(ah.station.poles) == 24
        assert len(ah.station.zeros) == 7
        # event
        assert ah.event.latitude == 0.0
        assert ah.event.longitude == 0.0
        assert ah.event.depth == 0.0
        assert ah.event.origin_time is None
        assert ah.event.comment == 'null'
        # record
        assert ah.record.type == 1
        assert ah.record.ndata == 720
        assert tr.stats.npts == 720
        assert len(tr) == 720
        assert tr.data.dtype == np.float64
        assert round(abs(ah.record.delta-0.25), 6) == 0
        assert round(abs(tr.stats.delta-0.25), 6) == 0
        assert round(abs(ah.record.max_amplitude-0.0), 6) == 0
        dt = UTCDateTime(1984, 4, 20, 6, 42, 0, 120000)
        assert ah.record.start_time == dt
        assert tr.stats.starttime == dt
        assert ah.record.comment == 'null'
        assert ah.record.log == 'gdsn_tape;demeaned;'
        # data
        np.testing.assert_array_almost_equal(tr.data[:4], np.array([
            -731.41247559, -724.41247559, -622.41247559, -470.4125061]))
        np.testing.assert_array_almost_equal(tr.data[-4:], np.array([
            -1421.41247559, 118.58750153, 88.58750153, -982.41247559]))

        # not supported data types (vector, complex, tensor)
        testfile = testdata['ah2.t']
        with pytest.raises(NotImplementedError):
            _read_ah(testfile)

    def test_tsg(self, datapath):
        """
        Test reading AH v1 files of the STsR-TSG System at Borovoye.

        .. seealso:: https://www.ldeo.columbia.edu/res/pi/Monitoring/Data/
        """
        # 1 - BRV.TSG.DS.lE21
        testfile = datapath / 'TSG' / 'BRV.TSG.DS.lE21.resp'
        st = _read_ah(testfile)
        assert len(st) == 1
        tr = st[0]
        ah = tr.stats.ah
        assert ah.version == '1.0'
        # station
        assert ah.station.code == 'BRVK'
        assert ah.station.channel == 'lE21'
        assert ah.station.type == 'TSG-DS'
        assert round(abs(ah.station.latitude-53.058060), 6) == 0
        assert round(abs(ah.station.longitude-70.282799), 6) == 0
        assert ah.station.elevation == 300.0
        assert round(abs(ah.station.gain-0.05), 6) == 0
        assert round(abs(ah.station.normalization-40.009960), 6) == 0
        assert round(abs(ah.station.longitude-70.282799), 6) == 0
        # calibration
        assert len(ah.station.poles) == 7
        assert round(abs(
            ah.station.poles[0]-complex(-1.342653e-01, 1.168836e-01)), 6) == 0
        assert round(abs(
            ah.station.poles[1]-complex(-1.342653e-01, -1.168836e-01)), 6) == 0
        assert len(ah.station.zeros) == 4
        assert round(abs(ah.station.zeros[0]-complex(0.0, 0.0)), 6) == 0
        assert round(abs(ah.station.zeros[1]-complex(0.0, 0.0)), 6) == 0
        assert round(abs(ah.station.zeros[2]-complex(0.0, 0.0)), 6) == 0
        assert round(abs(ah.station.zeros[3]-complex(0.0, 0.0)), 6) == 0
        # event
        assert round(abs(ah.event.latitude-49.833000), 6) == 0
        assert round(abs(ah.event.longitude-78.807999), 6) == 0
        assert ah.event.depth == 0.5
        assert ah.event.origin_time == UTCDateTime(1988, 2, 8, 15, 23)
        assert ah.event.comment == 'Calibration_for_hg_TSG'
        # record
        assert ah.record.type == 1
        assert ah.record.ndata == 225
        assert tr.stats.npts == 225
        assert len(tr) == 225
        assert tr.data.dtype == np.float64
        assert round(abs(ah.record.delta-0.312), 6) == 0
        assert round(abs(tr.stats.delta-0.312), 6) == 0
        assert round(abs(ah.record.max_amplitude-785.805786), 6) == 0
        dt = UTCDateTime(1988, 2, 8, 15, 24, 50.136002)
        assert ah.record.start_time == dt
        assert tr.stats.starttime == dt
        assert ah.record.abscissa_min == 0.0
        assert ah.record.comment == 'DS response in counts/nm;'
        assert ah.record.log == \
            'brv2ah: ahtedit;demeaned;modhead;modhead;ahtedit;'
        # extras
        assert len(ah.extras) == 21
        assert ah.extras[0] == 0.0
        assert round(abs(ah.extras[1]-0.1), 6) == 0
        assert round(abs(ah.extras[2]-0.1), 6) == 0
        assert ah.extras[3] == 0.0
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

    def test_write_ah1(self, testdata):
        """
        Testing writing AH1 file format using _write_ah1() function.
        """
        # AH v1
        testfile = testdata['st.ah']
        stream_orig = _read_ah(testfile)

        with NamedTemporaryFile() as tf:
            tmpfile = tf.name + '.AH'
            # write testfile
            _write_ah1(stream_orig, tmpfile)
            # read again
            st = _read_ah1(tmpfile)
            assert len(st) == 1
            tr = st[0]
            ah = tr.stats.ah
            stats = tr.stats
            # stream header
            assert stats.network == ''
            assert stats.station == 'ALE'
            assert stats.location == ''
            assert stats.channel == 'VHZ'
            starttime = UTCDateTime(1994, 6, 9, 0, 40, 45)
            endtime = UTCDateTime(1994, 6, 12, 8, 55, 4, 724522)
            assert stats.starttime == starttime
            assert stats.endtime == endtime
            assert round(abs(stats.sampling_rate-0.100000), 6) == 0
            assert round(abs(stats.delta-9.999990), 6) == 0
            assert stats.npts == 28887
            assert len(tr) == 28887
            assert stats.calib == 1.0

            # station
            assert ah.version == '1.0'
            assert ah.station.code == 'ALE'
            assert ah.station.channel == 'VHZ'
            assert ah.station.type == 'Global S'
            assert ah.station.latitude == 82.50330352783203
            assert ah.station.longitude == -62.349998474121094
            assert ah.station.elevation == 60.0
            assert ah.station.gain == 265302864.0
            assert len(ah.station.poles) == 13
            assert len(ah.station.zeros) == 6
            # event
            assert ah.event.latitude == -13.872200012207031
            assert ah.event.longitude == -67.51249694824219
            assert ah.event.depth == 640000.0
            origintime = UTCDateTime(1994, 6, 9, 0, 33, 16)
            assert ah.event.origin_time == origintime
            assert ah.event.comment == 'null'
            # record
            assert ah.record.type == 1
            assert ah.record.ndata == 28887
            assert tr.data.dtype == np.float64
            assert round(abs(ah.record.delta-9.999990), 6) == 0
            assert ah.record.max_amplitude == 9.265750885009766
            rstarttime = UTCDateTime(1994, 6, 9, 0, 40, 45)
            assert ah.record.start_time == rstarttime
            comment = 'Comp azm=0.0,inc=-90.0; Disp (m);'
            assert ah.record.comment == comment
            assert ah.record.log == 'null'
            # data
            np.testing.assert_array_almost_equal(tr.data[:4], np.array([
                -236., -242., -252., -262.]))
            np.testing.assert_array_almost_equal(tr.data[-4:], np.array([
                101., 106., 107., 104.]))
