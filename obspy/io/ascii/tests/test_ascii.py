# -*- coding: utf-8 -*-
import numpy as np

from obspy import Trace, UTCDateTime, read
from obspy.io.ascii.core import (_determine_dtype, _is_slist, _is_tspair,
                                 _read_slist, _read_tspair, _write_slist,
                                 _write_tspair)
from obspy.core.util import NamedTemporaryFile
import pytest


class TestASCII():
    """
    """
    def test_is_slist_file(self, testdata, datapath):
        """
        Testing SLIST file format.
        """
        testfile = testdata['slist.ascii']
        assert _is_slist(testfile)
        testfile = testdata['slist_2_traces.ascii']
        assert _is_slist(testfile)
        testfile = testdata['tspair.ascii']
        assert not _is_slist(testfile)
        # not existing file should fail
        testfile = datapath / 'xyz'
        assert not _is_slist(testfile)

    def test_read_slist_file_single_trace(self, testdata):
        """
        Read SLIST file test via obspy.core.ascii._read_slist.
        """
        testfile = testdata['slist.ascii']
        # read
        stream = _read_slist(testfile)
        stream.verify()
        assert stream[0].stats.network == 'XX'
        assert stream[0].stats.station == 'TEST'
        assert stream[0].stats.location == ''
        assert stream[0].stats.channel == 'BHZ'
        assert stream[0].stats.sampling_rate == 40.0
        assert stream[0].stats.npts == 635
        assert stream[0].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[0].stats.calib == 1.0e-00
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)

    def test_read_slist_file_multiple_traces(self, testdata):
        """
        Read SLIST file test via obspy.core.ascii._read_slist.
        """
        testfile = testdata['slist_2_traces.ascii']
        # read
        stream = _read_slist(testfile)
        stream.verify()
        assert stream[0].stats.network == 'XX'
        assert stream[0].stats.station == 'TEST'
        assert stream[0].stats.location == ''
        assert stream[0].stats.channel == 'BHZ'
        assert stream[0].stats.sampling_rate == 40.0
        assert stream[0].stats.npts == 635
        assert stream[0].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[0].stats.calib == 1.0e-00
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)
        # second trace
        assert stream[1].stats.network == 'XX'
        assert stream[1].stats.station == 'TEST'
        assert stream[1].stats.location == ''
        assert stream[1].stats.channel == 'BHE'
        assert stream[1].stats.sampling_rate == 40.0
        assert stream[1].stats.npts == 630
        assert stream[1].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[1].stats.calib == 1.0e-00
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data)
        # check last 4 samples
        data = [781, 785, 778, 772]
        np.testing.assert_array_almost_equal(stream[1].data[-4:], data)

    def test_read_slist_file_head_only(self, testdata):
        """
        Read SLIST file test via obspy.core.ascii._read_slist.
        """
        testfile = testdata['slist.ascii']
        # read
        stream = _read_slist(testfile, headonly=True)
        assert stream[0].stats.network == 'XX'
        assert stream[0].stats.station == 'TEST'
        assert stream[0].stats.location == ''
        assert stream[0].stats.channel == 'BHZ'
        assert stream[0].stats.sampling_rate == 40.0
        assert stream[0].stats.npts == 635
        assert stream[0].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[0].stats.calib == 1.0e-00
        assert len(stream[0].data) == 0

    def test_read_slist_file_encoding(self, testdata):
        """
        Read SLIST file test via obspy.core.ascii._read_slist.
        """
        # float32
        testfile = testdata['slist_float.ascii']
        stream = _read_slist(testfile)
        assert stream[0].stats.network == 'XX'
        assert stream[0].stats.station == 'TEST'
        assert stream[0].stats.location == ''
        assert stream[0].stats.channel == 'BHZ'
        assert stream[0].stats.sampling_rate == 40.0
        assert stream[0].stats.npts == 12
        assert stream[0].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[0].stats.calib == 1.0e-00
        data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
        np.testing.assert_array_almost_equal(stream[0].data, data, decimal=2)
        # unknown encoding
        testfile = testdata['slist_unknown.ascii']
        with pytest.raises(NotImplementedError):
            _read_slist(testfile)

    def test_is_tspair_file(self, testdata, datapath):
        """
        Testing TSPAIR file format.
        """
        testfile = testdata['tspair.ascii']
        assert _is_tspair(testfile)
        testfile = testdata['tspair_2_traces.ascii']
        assert _is_tspair(testfile)
        testfile = testdata['slist.ascii']
        assert not _is_tspair(testfile)
        # not existing file should fail
        testfile = datapath / 'xyz'
        assert not _is_tspair(testfile)

    def test_read_tspair_file_single_trace(self, testdata):
        """
        Read TSPAIR file test via obspy.core.ascii._read_tspair.
        """
        testfile = testdata['tspair.ascii']
        # read
        stream = _read_tspair(testfile)
        stream.verify()
        assert stream[0].stats.network == 'XX'
        assert stream[0].stats.station == 'TEST'
        assert stream[0].stats.location == ''
        assert stream[0].stats.channel == 'BHZ'
        assert stream[0].stats.sampling_rate == 40.0
        assert stream[0].stats.npts == 635
        assert stream[0].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[0].stats.calib == 1.0e-00
        assert stream[0].stats.mseed.dataquality == 'R'
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)

    def test_read_tspair_file_multiple_traces(self, testdata):
        """
        Read TSPAIR file test via obspy.core.ascii._read_tspair.
        """
        testfile = testdata['tspair_2_traces.ascii']
        # read
        stream = _read_tspair(testfile)
        stream.verify()
        # sort traces to ensure comparable results
        stream.sort()
        assert stream[1].stats.network == 'XX'
        assert stream[1].stats.station == 'TEST'
        assert stream[1].stats.location == ''
        assert stream[1].stats.channel == 'BHZ'
        assert stream[1].stats.sampling_rate == 40.0
        assert stream[1].stats.npts == 635
        assert stream[1].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[1].stats.calib == 1.0e-00
        assert stream[1].stats.mseed.dataquality == 'R'
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[1].data[-4:], data)
        # second trace
        assert stream[0].stats.network == 'XX'
        assert stream[0].stats.station == 'TEST'
        assert stream[0].stats.location == ''
        assert stream[0].stats.channel == 'BHE'
        assert stream[0].stats.sampling_rate == 40.0
        assert stream[0].stats.npts == 630
        assert stream[0].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[0].stats.calib == 1.0e-00
        assert stream[0].stats.mseed.dataquality == 'R'
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [781, 785, 778, 772]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)

    def test_read_tspair_head_only(self, testdata):
        """
        Read TSPAIR file test via obspy.core.ascii._read_tspair.
        """
        testfile = testdata['tspair.ascii']
        # read
        stream = _read_tspair(testfile, headonly=True)
        assert stream[0].stats.network == 'XX'
        assert stream[0].stats.station == 'TEST'
        assert stream[0].stats.location == ''
        assert stream[0].stats.channel == 'BHZ'
        assert stream[0].stats.sampling_rate == 40.0
        assert stream[0].stats.npts == 635
        assert stream[0].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[0].stats.calib == 1.0e-00
        assert stream[0].stats.mseed.dataquality == 'R'
        assert len(stream[0].data) == 0

    def test_read_tspair_file_encoding(self, testdata):
        """
        Read TSPAIR file test via obspy.core.ascii._read_tspair.
        """
        # float32
        testfile = testdata['tspair_float.ascii']
        stream = _read_tspair(testfile)
        stream.verify()
        assert stream[0].stats.network == 'XX'
        assert stream[0].stats.station == 'TEST'
        assert stream[0].stats.location == ''
        assert stream[0].stats.channel == 'BHZ'
        assert stream[0].stats.sampling_rate == 40.0
        assert stream[0].stats.npts == 12
        assert stream[0].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[0].stats.calib == 1.0e-00
        assert stream[0].stats.mseed.dataquality == 'R'
        data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
        np.testing.assert_array_almost_equal(stream[0].data, data, decimal=2)
        # unknown encoding
        testfile = testdata['tspair_unknown.ascii']
        with pytest.raises(NotImplementedError):
            _read_tspair(testfile)

    def test_write_tspair(self, testdata):
        """
        Write TSPAIR file test via obspy.core.ascii._write_tspair.
        """
        # float32
        testfile = testdata['tspair_float.ascii']
        stream_orig = _read_tspair(testfile)
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write
            _write_tspair(stream_orig, tmpfile)
            # look at the raw data
            with open(tmpfile, 'rt') as f:
                lines = f.readlines()
            assert lines[0].strip() == \
                'TIMESERIES XX_TEST__BHZ_R, 12 samples, 40 sps, ' + \
                '2008-01-15T00:00:00.025000, TSPAIR, FLOAT, Counts'
            assert lines[1].strip() == \
                '2008-01-15T00:00:00.025000  +1.8500999450e+02'
            # read again
            stream = _read_tspair(tmpfile)
            stream.verify()
            assert stream[0].stats.network == 'XX'
            assert stream[0].stats.station == 'TEST'
            assert stream[0].stats.location == ''
            assert stream[0].stats.channel == 'BHZ'
            assert stream[0].stats.sampling_rate == 40.0
            assert stream[0].stats.npts == 12
            assert stream[0].stats.starttime == \
                UTCDateTime("2008-01-15T00:00:00.025000")
            assert stream[0].stats.calib == 1.0e-00
            assert stream[0].stats.mseed.dataquality == 'R'
            data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                    209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
            np.testing.assert_array_almost_equal(stream[0].data, data,
                                                 decimal=2)
            # compare raw header
            with open(testfile, 'rt') as f:
                lines_orig = f.readlines()
            with open(tmpfile, 'rt') as f:
                lines_new = f.readlines()
        assert lines_orig[0] == lines_new[0]

    def test_write_tspair_custom_fmt(self, testdata):
        """
        Write TSPAIR file test via obspy.core.ascii._write_tspair.
        """
        # float
        testfile_orig = testdata['tspair_float.ascii']
        testfile = testdata['tspair_float_custom_fmt.ascii']
        stream_orig = _read_tspair(testfile_orig)
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write
            _write_tspair(stream_orig, tmpfile, custom_fmt='%3.14f')
            # look at the raw data
            with open(tmpfile, 'rt') as f:
                lines = f.readlines()
            assert lines[0].strip() == \
                'TIMESERIES XX_TEST__BHZ_R, 12 samples, 40 sps, ' + \
                '2008-01-15T00:00:00.025000, TSPAIR, FLOAT, Counts'
            assert lines[1].strip() == \
                '2008-01-15T00:00:00.025000  185.00999450000000'
            # read again
            stream = _read_tspair(tmpfile)
            stream.verify()
            assert stream[0].stats.network == 'XX'
            assert stream[0].stats.station == 'TEST'
            assert stream[0].stats.location == ''
            assert stream[0].stats.channel == 'BHZ'
            assert stream[0].stats.sampling_rate == 40.0
            assert stream[0].stats.npts == 12
            assert stream[0].stats.starttime == \
                UTCDateTime("2008-01-15T00:00:00.025000")
            assert stream[0].stats.calib == 1.0e-00
            assert stream[0].stats.mseed.dataquality == 'R'
            data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                    209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
            np.testing.assert_array_almost_equal(stream[0].data, data,
                                                 decimal=2)
            # compare raw header
            with open(testfile, 'rt') as f:
                lines_orig = f.readlines()
            with open(tmpfile, 'rt') as f:
                lines_new = f.readlines()
        assert lines_orig[0] == lines_new[0]

    def test_write_tspair_custom_fmt_custom(self, testdata):
        """
        Write TSPAIR file test via obspy.core.ascii._write_tspair.
        """
        # float
        testfile_orig = testdata['tspair_float.ascii']
        stream_orig = _read_tspair(testfile_orig)
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write
            _write_tspair(stream_orig, tmpfile, custom_fmt='%+r')
            with pytest.raises(NotImplementedError):
                _read_tspair(tmpfile)
            # look at the raw data
            with open(tmpfile, 'rt') as f:
                lines = f.readlines()
            assert lines[0].strip() == \
                'TIMESERIES XX_TEST__BHZ_R, 12 samples, 40 sps, ' + \
                '2008-01-15T00:00:00.025000, TSPAIR, CUSTOM, Counts'
            assert lines[1].strip() == \
                '2008-01-15T00:00:00.025000  185.0099945'

    def test_write_tspair_file_multiple_traces(self, testdata):
        """
        Write TSPAIR file test via obspy.core.ascii._write_tspair.
        """
        testfile = testdata['tspair_2_traces.ascii']
        stream_orig = _read_tspair(testfile)
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write
            _write_tspair(stream_orig, tmpfile)
            # look at the raw data
            with open(tmpfile, 'rt') as f:
                lines = f.readlines()
            assert lines[0].startswith('TIMESERIES')
            assert 'TSPAIR' in lines[0]
            assert lines[1] == '2008-01-15T00:00:00.025000  185\n'
            # test issue #321 (problems in time stamping)
            assert lines[-1] == '2008-01-15T00:00:15.750000  772\n'
            # read again
            stream = _read_tspair(tmpfile)
        stream.verify()
        # sort traces to ensure comparable results
        stream.sort()
        assert stream[0].stats.network == 'XX'
        assert stream[0].stats.station == 'TEST'
        assert stream[0].stats.location == ''
        assert stream[0].stats.channel == 'BHE'
        assert stream[0].stats.sampling_rate == 40.0
        assert stream[0].stats.npts == 630
        assert stream[0].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[0].stats.calib == 1.0e-00
        assert stream[0].stats.mseed.dataquality == 'R'
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [781, 785, 778, 772]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)
        # second trace
        assert stream[1].stats.network == 'XX'
        assert stream[1].stats.station == 'TEST'
        assert stream[1].stats.location == ''
        assert stream[1].stats.channel == 'BHZ'
        assert stream[1].stats.sampling_rate == 40.0
        assert stream[1].stats.npts == 635
        assert stream[1].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[1].stats.calib == 1.0e-00
        assert stream[0].stats.mseed.dataquality == 'R'
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[1].data[-4:], data)

    def test_write_slist(self, testdata):
        """
        Write SLIST file test via obspy.core.ascii._write_tspair.
        """
        # float32
        testfile = testdata['slist_float.ascii']
        stream_orig = _read_slist(testfile)
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write
            _write_slist(stream_orig, tmpfile)
            # look at the raw data
            with open(tmpfile, 'rt') as f:
                lines = f.readlines()
            assert lines[0].strip() == \
                'TIMESERIES XX_TEST__BHZ_R, 12 samples, 40 sps, ' + \
                '2008-01-15T00:00:00.025000, SLIST, FLOAT, Counts'
            assert lines[1].strip() == \
                '+1.8500999450e+02\t+1.8102000430e+02\t+1.8502999880e+02\t' + \
                '+1.8903999330e+02\t+1.9405000310e+02\t+2.0505999760e+02'
            # read again
            stream = _read_slist(tmpfile)
            stream.verify()
            assert stream[0].stats.network == 'XX'
            assert stream[0].stats.station == 'TEST'
            assert stream[0].stats.location == ''
            assert stream[0].stats.channel == 'BHZ'
            assert stream[0].stats.sampling_rate == 40.0
            assert stream[0].stats.npts == 12
            assert stream[0].stats.starttime == \
                UTCDateTime("2008-01-15T00:00:00.025000")
            assert stream[0].stats.calib == 1.0e-00
            assert stream[0].stats.mseed.dataquality == 'R'
            data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                    209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
            np.testing.assert_array_almost_equal(stream[0].data, data,
                                                 decimal=2)
            # compare raw header
            with open(testfile, 'rt') as f:
                lines_orig = f.readlines()
            with open(tmpfile, 'rt') as f:
                lines_new = f.readlines()
        assert lines_orig[0] == lines_new[0]

    def test_write_slist_custom_fmt_float(self, testdata):
        """
        Write SLIST file test via obspy.core.ascii._write_tspair.
        """
        # float
        testfile_orig = testdata['slist_float.ascii']
        testfile = testdata['slist_float_custom_fmt.ascii']
        stream_orig = _read_slist(testfile_orig)
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write
            _write_slist(stream_orig, tmpfile, custom_fmt='%3.14f')
            # look at the raw data
            with open(tmpfile, 'rt') as f:
                lines = f.readlines()
            assert lines[0].strip() == \
                'TIMESERIES XX_TEST__BHZ_R, 12 samples, 40 sps, ' + \
                '2008-01-15T00:00:00.025000, SLIST, FLOAT, Counts'
            assert lines[1].strip() == \
                '185.00999450000000\t181.02000430000001\t' + \
                '185.02999879999999\t189.03999329999999\t' + \
                '194.05000310000000\t205.05999760000000'
            # read again
            stream = _read_slist(tmpfile)
            stream.verify()
            assert stream[0].stats.network == 'XX'
            assert stream[0].stats.station == 'TEST'
            assert stream[0].stats.location == ''
            assert stream[0].stats.channel == 'BHZ'
            assert stream[0].stats.sampling_rate == 40.0
            assert stream[0].stats.npts == 12
            assert stream[0].stats.starttime == \
                UTCDateTime("2008-01-15T00:00:00.025000")
            assert stream[0].stats.calib == 1.0e-00
            assert stream[0].stats.mseed.dataquality == 'R'
            data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                    209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
            np.testing.assert_array_almost_equal(stream[0].data, data,
                                                 decimal=2)
            # compare raw header
            with open(testfile, 'rt') as f:
                lines_orig = f.readlines()
            with open(tmpfile, 'rt') as f:
                lines_new = f.readlines()
        assert lines_orig[0] == lines_new[0]

    def test_write_slist_custom_fmt_custom(self, testdata):
        """
        Write SLIST file test via obspy.core.ascii._write_tspair.
        """
        # float
        testfile_orig = testdata['slist_float.ascii']
        stream_orig = _read_slist(testfile_orig)
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write
            _write_slist(stream_orig, tmpfile, custom_fmt='%+r')
            with pytest.raises(NotImplementedError):
                _read_slist(tmpfile)
            # look at the raw data
            with open(tmpfile, 'rt') as f:
                lines = f.readlines()
            assert lines[0].strip() == \
                'TIMESERIES XX_TEST__BHZ_R, 12 samples, 40 sps, ' + \
                '2008-01-15T00:00:00.025000, SLIST, CUSTOM, Counts'
            assert lines[1].strip() == \
                '185.0099945\t181.02000430000001\t' + \
                '185.02999879999999\t189.03999329999999\t' + \
                '194.0500031\t205.0599976'

    def test_write_slist_file_multiple_traces(self, testdata):
        """
        Write SLIST file test via obspy.core.ascii._write_tspair.
        """
        testfile = testdata['slist_2_traces.ascii']
        stream_orig = _read_slist(testfile)
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write
            _write_slist(stream_orig, tmpfile)
            # look at the raw data
            with open(tmpfile, 'rt') as f:
                lines = f.readlines()
            assert lines[0].startswith('TIMESERIES')
            assert 'SLIST' in lines[0]
            assert lines[1].strip() == '185\t181\t185\t189\t194\t205'
            # read again
            stream = _read_slist(tmpfile)
        stream.verify()
        # sort traces to ensure comparable results
        stream.sort()
        assert stream[0].stats.network == 'XX'
        assert stream[0].stats.station == 'TEST'
        assert stream[0].stats.location == ''
        assert stream[0].stats.channel == 'BHE'
        assert stream[0].stats.sampling_rate == 40.0
        assert stream[0].stats.npts == 630
        assert stream[0].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[0].stats.calib == 1.0e-00
        assert stream[0].stats.mseed.dataquality == 'R'
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [781, 785, 778, 772]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)
        # second trace
        assert stream[1].stats.network == 'XX'
        assert stream[1].stats.station == 'TEST'
        assert stream[1].stats.location == ''
        assert stream[1].stats.channel == 'BHZ'
        assert stream[1].stats.sampling_rate == 40.0
        assert stream[1].stats.npts == 635
        assert stream[1].stats.starttime == \
            UTCDateTime("2008-01-15T00:00:00.025000")
        assert stream[1].stats.calib == 1.0e-00
        assert stream[0].stats.mseed.dataquality == 'R'
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[1].data[-4:], data)

    def test_write_small_trace(self):
        """
        Tests writing Traces containing 0, 1 or 2 samples only.
        """
        for format in ['SLIST', 'TSPAIR']:
            for num in range(0, 4):
                tr = Trace(data=np.arange(num))
                with NamedTemporaryFile() as tf:
                    tempfile = tf.name
                    tr.write(tempfile, format=format)
                    # test results
                    st = read(tempfile, format=format)
                assert len(st) == 1
                assert len(st[0]) == num

    def test_float_sampling_rates_write_and_read(self):
        """
        Tests writing and reading Traces with floating point and with less than
        1 Hz sampling rates.
        """
        tr = Trace(np.arange(10))
        check_sampling_rates = (0.000000001, 1.000000001, 100.000000001,
                                99.999999999, 1.5, 1.666666, 10000.0001)
        for format in ['SLIST', 'TSPAIR']:
            for sps in check_sampling_rates:
                tr.stats.sampling_rate = sps
                with NamedTemporaryFile() as tf:
                    tempfile = tf.name
                    tr.write(tempfile, format=format)
                    # test results
                    got = read(tempfile, format=format)[0]
                assert tr.stats.sampling_rate == got.stats.sampling_rate

    def test_determine_dtype(self):
        """
        Tests _determine_dtype for properly returned types
        """
        float_formats = ['%+10.10e', '%+.10e', '%.3e',
                         '%+10.10E', '%+.10E', '%.3E',
                         '%+10.10f', '%+.10f', '%.3f',
                         '%+10.10F', '%+.10F', '%.3F',
                         '%+10.10g', '%+.10g', '%.3g',
                         '%+10.10G', '%+.10G', '%.3G']

        int_formats = ['%+10.10i', '%+.10i', '%.3i',
                       '%+10.10I', '%+.10I', '%.3I',
                       '%+10.10d', '%+.10d', '%.3d',
                       '%+10.10D', '%+.10D', '%.3D']

        custom_formats = ['%+10.10s', '%+.10s', '%.3s',
                          '%+10.10x', '%+.10x', '%.3x',
                          '%+10.10k', '%+.10k', '%.3k',
                          '%+10.10z', '%+.10z', '%.3z',
                          '%+10.10w', '%+.10w', '%.3w',
                          '%+10.10q', '%+.10q', '%.3q']

        for format in float_formats:
            assert 'FLOAT' == _determine_dtype(format)

        for format in int_formats:
            assert 'INTEGER' == _determine_dtype(format)

        for format in custom_formats:
            assert 'CUSTOM' == _determine_dtype(format)

        with pytest.raises(ValueError):
            _determine_dtype('')

    def test_regression_against_mseed2ascii(self, testdata):
        """
        Regression test against issue #2165.
        """
        mseed_file = testdata["miniseed_record.mseed"]
        mseed2ascii_file = testdata["mseed2ascii_miniseed_record.txt"]

        with NamedTemporaryFile() as tf:
            # Write as TSPAIR
            read(mseed_file).write(tf.name, format="TSPAIR")
            # Check all lines aside from the first as they differ.
            with open(tf.name, "rt") as fh:
                actual_lines = fh.readlines()[1:]
            with open(mseed2ascii_file, "rt") as fh:
                expected_lines = fh.readlines()[1:]

        for actual, expected in zip(actual_lines, expected_lines):
            assert actual.strip() == expected.strip()
