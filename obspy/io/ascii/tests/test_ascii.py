# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import os
import unittest

import numpy as np

from obspy import Trace, UTCDateTime, read
from obspy.io.ascii.core import (_is_slist, _is_tspair, _read_slist,
                                 _read_tspair, _write_slist, _write_tspair)
from obspy.core.util import NamedTemporaryFile


class ASCIITestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_is_slist_file(self):
        """
        Testing SLIST file format.
        """
        testfile = os.path.join(self.path, 'data', 'slist.ascii')
        self.assertEqual(_is_slist(testfile), True)
        testfile = os.path.join(self.path, 'data', 'slist_2_traces.ascii')
        self.assertEqual(_is_slist(testfile), True)
        testfile = os.path.join(self.path, 'data', 'tspair.ascii')
        self.assertEqual(_is_slist(testfile), False)
        # not existing file should fail
        testfile = os.path.join(self.path, 'data', 'xyz')
        self.assertEqual(_is_slist(testfile), False)

    def test_read_slist_file_single_trace(self):
        """
        Read SLIST file test via obspy.core.ascii._read_slist.
        """
        testfile = os.path.join(self.path, 'data', 'slist.ascii')
        # read
        stream = _read_slist(testfile)
        stream.verify()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 635)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)

    def test_read_slist_file_multiple_traces(self):
        """
        Read SLIST file test via obspy.core.ascii._read_slist.
        """
        testfile = os.path.join(self.path, 'data', 'slist_2_traces.ascii')
        # read
        stream = _read_slist(testfile)
        stream.verify()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 635)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)
        # second trace
        self.assertEqual(stream[1].stats.network, 'XX')
        self.assertEqual(stream[1].stats.station, 'TEST')
        self.assertEqual(stream[1].stats.location, '')
        self.assertEqual(stream[1].stats.channel, 'BHE')
        self.assertEqual(stream[1].stats.sampling_rate, 40.0)
        self.assertEqual(stream[1].stats.npts, 630)
        self.assertEqual(stream[1].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[1].stats.calib, 1.0e-00)
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data)
        # check last 4 samples
        data = [781, 785, 778, 772]
        np.testing.assert_array_almost_equal(stream[1].data[-4:], data)

    def test_read_slist_file_head_only(self):
        """
        Read SLIST file test via obspy.core.ascii._read_slist.
        """
        testfile = os.path.join(self.path, 'data', 'slist.ascii')
        # read
        stream = _read_slist(testfile, headonly=True)
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 635)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(len(stream[0].data), 0)

    def test_read_slist_file_encoding(self):
        """
        Read SLIST file test via obspy.core.ascii._read_slist.
        """
        # float32
        testfile = os.path.join(self.path, 'data', 'slist_float.ascii')
        stream = _read_slist(testfile)
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 12)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
        np.testing.assert_array_almost_equal(stream[0].data, data, decimal=2)
        # unknown encoding
        testfile = os.path.join(self.path, 'data', 'slist_unknown.ascii')
        self.assertRaises(NotImplementedError, _read_slist, testfile)

    def test_is_tspair_file(self):
        """
        Testing TSPAIR file format.
        """
        testfile = os.path.join(self.path, 'data', 'tspair.ascii')
        self.assertEqual(_is_tspair(testfile), True)
        testfile = os.path.join(self.path, 'data', 'tspair_2_traces.ascii')
        self.assertEqual(_is_tspair(testfile), True)
        testfile = os.path.join(self.path, 'data', 'slist.ascii')
        self.assertEqual(_is_tspair(testfile), False)
        # not existing file should fail
        testfile = os.path.join(self.path, 'data', 'xyz')
        self.assertEqual(_is_tspair(testfile), False)

    def test_read_tspair_file_single_trace(self):
        """
        Read TSPAIR file test via obspy.core.ascii._read_tspair.
        """
        testfile = os.path.join(self.path, 'data', 'tspair.ascii')
        # read
        stream = _read_tspair(testfile)
        stream.verify()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 635)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)

    def test_read_tspair_file_multiple_traces(self):
        """
        Read TSPAIR file test via obspy.core.ascii._read_tspair.
        """
        testfile = os.path.join(self.path, 'data', 'tspair_2_traces.ascii')
        # read
        stream = _read_tspair(testfile)
        stream.verify()
        # sort traces to ensure comparable results
        stream.sort()
        self.assertEqual(stream[1].stats.network, 'XX')
        self.assertEqual(stream[1].stats.station, 'TEST')
        self.assertEqual(stream[1].stats.location, '')
        self.assertEqual(stream[1].stats.channel, 'BHZ')
        self.assertEqual(stream[1].stats.sampling_rate, 40.0)
        self.assertEqual(stream[1].stats.npts, 635)
        self.assertEqual(stream[1].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[1].stats.calib, 1.0e-00)
        self.assertEqual(stream[1].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[1].data[-4:], data)
        # second trace
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHE')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 630)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [781, 785, 778, 772]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)

    def test_read_tspair_head_only(self):
        """
        Read TSPAIR file test via obspy.core.ascii._read_tspair.
        """
        testfile = os.path.join(self.path, 'data', 'tspair.ascii')
        # read
        stream = _read_tspair(testfile, headonly=True)
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 635)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        self.assertEqual(len(stream[0].data), 0)

    def test_read_tspair_file_encoding(self):
        """
        Read TSPAIR file test via obspy.core.ascii._read_tspair.
        """
        # float32
        testfile = os.path.join(self.path, 'data', 'tspair_float.ascii')
        stream = _read_tspair(testfile)
        stream.verify()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 12)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
        np.testing.assert_array_almost_equal(stream[0].data, data, decimal=2)
        # unknown encoding
        testfile = os.path.join(self.path, 'data', 'tspair_unknown.ascii')
        self.assertRaises(NotImplementedError, _read_tspair, testfile)

    def test_write_tspair(self):
        """
        Write TSPAIR file test via obspy.core.ascii._write_tspair.
        """
        # float32
        testfile = os.path.join(self.path, 'data', 'tspair_float.ascii')
        stream_orig = _read_tspair(testfile)
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write
            _write_tspair(stream_orig, tmpfile)
            # read again
            stream = _read_tspair(tmpfile)
            stream.verify()
            self.assertEqual(stream[0].stats.network, 'XX')
            self.assertEqual(stream[0].stats.station, 'TEST')
            self.assertEqual(stream[0].stats.location, '')
            self.assertEqual(stream[0].stats.channel, 'BHZ')
            self.assertEqual(stream[0].stats.sampling_rate, 40.0)
            self.assertEqual(stream[0].stats.npts, 12)
            self.assertEqual(stream[0].stats.starttime,
                             UTCDateTime("2008-01-15T00:00:00.025000"))
            self.assertEqual(stream[0].stats.calib, 1.0e-00)
            self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
            data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                    209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
            np.testing.assert_array_almost_equal(stream[0].data, data,
                                                 decimal=2)
            # compare raw header
            with open(testfile, 'rt') as f:
                lines_orig = f.readlines()
            with open(tmpfile, 'rt') as f:
                lines_new = f.readlines()
        self.assertEqual(lines_orig[0], lines_new[0])

    def test_write_tspair_file_multiple_traces(self):
        """
        Write TSPAIR file test via obspy.core.ascii._write_tspair.
        """
        testfile = os.path.join(self.path, 'data', 'tspair_2_traces.ascii')
        stream_orig = _read_tspair(testfile)
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write
            _write_tspair(stream_orig, tmpfile)
            # look at the raw data
            with open(tmpfile, 'rt') as f:
                lines = f.readlines()
            self.assertTrue(lines[0].startswith('TIMESERIES'))
            self.assertIn('TSPAIR', lines[0])
            self.assertEqual(lines[1], '2008-01-15T00:00:00.025000  185\n')
            # test issue #321 (problems in time stamping)
            self.assertEqual(lines[-1], '2008-01-15T00:00:15.750000  772\n')
            # read again
            stream = _read_tspair(tmpfile)
        stream.verify()
        # sort traces to ensure comparable results
        stream.sort()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHE')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 630)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [781, 785, 778, 772]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)
        # second trace
        self.assertEqual(stream[1].stats.network, 'XX')
        self.assertEqual(stream[1].stats.station, 'TEST')
        self.assertEqual(stream[1].stats.location, '')
        self.assertEqual(stream[1].stats.channel, 'BHZ')
        self.assertEqual(stream[1].stats.sampling_rate, 40.0)
        self.assertEqual(stream[1].stats.npts, 635)
        self.assertEqual(stream[1].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[1].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[1].data[-4:], data)

    def test_write_slist(self):
        """
        Write SLIST file test via obspy.core.ascii._write_tspair.
        """
        # float32
        testfile = os.path.join(self.path, 'data', 'slist_float.ascii')
        stream_orig = _read_slist(testfile)
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write
            _write_slist(stream_orig, tmpfile)
            # look at the raw data
            with open(tmpfile, 'rt') as f:
                lines = f.readlines()
            self.assertEqual(
                lines[0].strip(),
                'TIMESERIES XX_TEST__BHZ_R, 12 samples, 40 sps, ' +
                '2008-01-15T00:00:00.025000, SLIST, FLOAT, Counts')
            self.assertEqual(
                lines[1].strip(),
                '185.009995\t181.020004\t185.029999\t189.039993\t' +
                '194.050003\t205.059998')
            # read again
            stream = _read_slist(tmpfile)
            stream.verify()
            self.assertEqual(stream[0].stats.network, 'XX')
            self.assertEqual(stream[0].stats.station, 'TEST')
            self.assertEqual(stream[0].stats.location, '')
            self.assertEqual(stream[0].stats.channel, 'BHZ')
            self.assertEqual(stream[0].stats.sampling_rate, 40.0)
            self.assertEqual(stream[0].stats.npts, 12)
            self.assertEqual(stream[0].stats.starttime,
                             UTCDateTime("2008-01-15T00:00:00.025000"))
            self.assertEqual(stream[0].stats.calib, 1.0e-00)
            self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
            data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                    209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
            np.testing.assert_array_almost_equal(stream[0].data, data,
                                                 decimal=2)
            # compare raw header
            with open(testfile, 'rt') as f:
                lines_orig = f.readlines()
            with open(tmpfile, 'rt') as f:
                lines_new = f.readlines()
        self.assertEqual(lines_orig[0], lines_new[0])

    def test_write_slist_file_multiple_traces(self):
        """
        Write SLIST file test via obspy.core.ascii._write_tspair.
        """
        testfile = os.path.join(self.path, 'data', 'slist_2_traces.ascii')
        stream_orig = _read_slist(testfile)
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write
            _write_slist(stream_orig, tmpfile)
            # look at the raw data
            with open(tmpfile, 'rt') as f:
                lines = f.readlines()
            self.assertTrue(lines[0].startswith('TIMESERIES'))
            self.assertIn('SLIST', lines[0])
            self.assertEqual(lines[1].strip(), '185\t181\t185\t189\t194\t205')
            # read again
            stream = _read_slist(tmpfile)
        stream.verify()
        # sort traces to ensure comparable results
        stream.sort()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHE')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 630)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [781, 785, 778, 772]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)
        # second trace
        self.assertEqual(stream[1].stats.network, 'XX')
        self.assertEqual(stream[1].stats.station, 'TEST')
        self.assertEqual(stream[1].stats.location, '')
        self.assertEqual(stream[1].stats.channel, 'BHZ')
        self.assertEqual(stream[1].stats.sampling_rate, 40.0)
        self.assertEqual(stream[1].stats.npts, 635)
        self.assertEqual(stream[1].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[1].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
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
                self.assertEqual(len(st), 1)
                self.assertEqual(len(st[0]), num)

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
                self.assertEqual(tr.stats.sampling_rate,
                                 got.stats.sampling_rate)


def suite():
    return unittest.makeSuite(ASCIITestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
