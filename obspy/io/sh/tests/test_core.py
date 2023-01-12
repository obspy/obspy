# -*- coding: utf-8 -*-
import os
import warnings

import numpy as np

from obspy import Trace, UTCDateTime, read
from obspy.core.util import NamedTemporaryFile
from obspy.io.sh.core import (STANDARD_ASC_HEADERS, _is_asc, _is_q, _read_asc,
                              _read_q, _write_asc, _write_q)


class TestCore():
    """
    """
    def test_read_101_traces(self, testdata):
        """
        Testing reading Q file with more than 100 traces.
        """
        testfile = testdata['101.QHD']
        # read
        stream = _read_q(testfile)
        stream.verify()
        assert len(stream) == 101

    def test_is_asc_file(self, testdata):
        """
        Testing ASC file format.
        """
        testfile = testdata['TEST_090101_0101.ASC']
        assert _is_asc(testfile)
        testfile = testdata['QFILE-TEST-SUN.QHD']
        assert not _is_asc(testfile)

    def test_is_q_file(self, testdata):
        """
        Testing Q header file format.
        """
        testfile = testdata['QFILE-TEST-SUN.QHD']
        assert _is_q(testfile)
        testfile = testdata['QFILE-TEST-SUN.QBN']
        assert not _is_q(testfile)
        testfile = testdata['TEST_090101_0101.ASC']
        assert not _is_q(testfile)

    def test_read_single_channel_asc_file(self, testdata):
        """
        Read ASC file test via obspy.io.sh.core._read_asc.
        """
        testfile = testdata['TEST_090101_0101.ASC']
        # read
        stream = _read_asc(testfile)
        stream.verify()
        assert stream[0].stats.delta == 5.000000e-02
        assert stream[0].stats.npts == 1000
        assert stream[0].stats.starttime == UTCDateTime(2009, 1, 1, 1, 1, 1)
        assert stream[0].stats.channel == 'BHZ'
        assert stream[0].stats.station == 'TEST'
        assert stream[0].stats.calib == 1.0e-00
        # check last 4 samples
        data = [2.176000e+01, 2.195485e+01, 2.213356e+01, 2.229618e+01]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)

    def _compare_stream(self, stream):
        """
        Helper function to verify stream from file 'data/QFILE-TEST*'.
        """
        # channel 1
        assert stream[0].stats.delta == 5.000000e-02
        assert stream[0].stats.npts == 801
        assert stream[0].stats.sh.COMMENT == 'TEST TRACE IN QFILE #1'
        assert stream[0].stats.starttime == UTCDateTime(2009, 10, 1, 12, 46, 1)
        assert stream[0].stats.channel == 'BHN'
        assert stream[0].stats.station == 'TEST'
        assert stream[0].stats.calib == 1.500000e+00
        # check last 4 samples
        data = [-4.070354e+01, -4.033876e+01, -3.995153e+01, -3.954230e+01]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data, 5)
        # channel 2
        assert stream[1].stats.delta == 5.000000e-02
        assert stream[1].stats.npts == 801
        assert stream[1].stats.sh.COMMENT == 'TEST TRACE IN QFILE #2'
        assert stream[1].stats.starttime == UTCDateTime(2009, 10, 1, 12, 46, 1)
        assert stream[1].stats.channel == 'BHE'
        assert stream[1].stats.station == 'TEST'
        assert stream[1].stats.calib == 1.500000e+00
        # check first 4 samples
        data = [-3.995153e+01, -4.033876e+01, -4.070354e+01, -4.104543e+01]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data, 5)
        # channel 3
        assert stream[2].stats.delta == 1.000000e-02
        assert stream[2].stats.npts == 4001
        assert stream[2].stats.sh.COMMENT == '******'
        assert stream[2].stats.starttime == \
            UTCDateTime(2010, 1, 1, 1, 1, 5, 999000)
        assert stream[2].stats.channel == 'HHZ'
        assert stream[2].stats.station == 'WET'
        assert stream[2].stats.calib == 1.059300e+00
        # check first 4 samples
        data = [4.449060e+02, 4.279572e+02, 4.120677e+02, 4.237200e+02]
        np.testing.assert_array_almost_equal(stream[2].data[0:4], data, 4)

    def test_read_and_write_multi_channel_asc_file(self, testdata):
        """
        Read and write ASC file via obspy.io.sh.core._read_asc.
        """
        origfile = testdata['QFILE-TEST-ASC.ASC']
        # read original
        stream1 = _read_asc(origfile)
        stream1.verify()
        self._compare_stream(stream1)
        # write
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            _write_asc(stream1, tempfile, STANDARD_ASC_HEADERS + ['COMMENT'])
            # read both files and compare the content
            with open(origfile, 'rt') as f:
                text1 = f.readlines()
            with open(tempfile, 'rt') as f:
                text2 = f.readlines()
            assert text1 == text2
            # read again
            stream2 = _read_asc(tempfile)
            stream2.verify()
            self._compare_stream(stream2)

    def test_read_and_write_multi_channel_asc_file_via_obspy(self, testdata):
        """
        Read and write ASC file test via obspy.core.
        """
        origfile = testdata['QFILE-TEST-ASC.ASC']
        # read original
        stream1 = read(origfile, format="SH_ASC")
        stream1.verify()
        self._compare_stream(stream1)
        # write
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            hd = STANDARD_ASC_HEADERS + ['COMMENT']
            stream1.write(tempfile, format="SH_ASC", included_headers=hd)
            # read again w/ auto detection
            stream2 = read(tempfile)
            stream2.verify()
            self._compare_stream(stream2)

    def test_read_and_write_multi_channel_q_file(self, testdata):
        """
        Read and write Q file via obspy.io.sh.core._read_q.
        """
        # 1 - little endian (PC)
        origfile = testdata['QFILE-TEST.QHD']
        # read original
        stream1 = _read_q(origfile)
        stream1.verify()
        self._compare_stream(stream1)
        # write
        with NamedTemporaryFile(suffix='.QHD') as tf:
            tempfile = tf.name
            _write_q(stream1, tempfile, append=False)
            # read again
            stream2 = _read_q(tempfile)
            stream2.verify()
            self._compare_stream(stream2)
            # remove binary file too (dynamically created)
            os.remove(os.path.splitext(tempfile)[0] + '.QBN')
        # 2 - big endian (SUN)
        origfile = testdata['QFILE-TEST-SUN.QHD']
        # read original
        stream1 = _read_q(origfile, byteorder=">")
        stream1.verify()
        self._compare_stream(stream1)
        # write
        with NamedTemporaryFile(suffix='.QHD') as tf:
            tempfile = tf.name
            _write_q(stream1, tempfile, byteorder=">", append=False)
            # read again
            stream2 = _read_q(tempfile, byteorder=">")
            stream2.verify()
            self._compare_stream(stream2)
            # remove binary file too (dynamically created)
            os.remove(os.path.splitext(tempfile)[0] + '.QBN')

    def test_read_and_write_multi_channel_q_file_via_obspy(self, testdata):
        """
        Read and write Q file test via obspy.core.
        """
        # 1 - little endian (PC)
        origfile = testdata['QFILE-TEST.QHD']
        # read original
        stream1 = read(origfile, format="Q")
        stream1.verify()
        self._compare_stream(stream1)
        # write
        with NamedTemporaryFile(suffix='.QHD') as tf:
            tempfile = tf.name
            stream1.write(tempfile, format="Q", append=False)
            # read again w/ auto detection
            stream2 = read(tempfile)
            stream2.verify()
            self._compare_stream(stream2)
            # remove binary file too (dynamically created)
            os.remove(os.path.splitext(tempfile)[0] + '.QBN')
        # 2 - big endian (SUN)
        origfile = testdata['QFILE-TEST-SUN.QHD']
        # read original
        stream1 = read(origfile, format="Q", byteorder=">")
        stream1.verify()
        self._compare_stream(stream1)
        # write
        with NamedTemporaryFile(suffix='.QHD') as tf:
            tempfile = tf.name
            stream1.write(tempfile, format="Q", byteorder=">", append=False)
            # read again w/ auto detection
            stream2 = read(tempfile, byteorder=">")
            stream2.verify()
            self._compare_stream(stream2)
            # remove binary file too (dynamically created)
            os.remove(os.path.splitext(tempfile)[0] + '.QBN')

    def test_skip_asc_lines(self, testdata):
        testfile = testdata['QFILE-TEST-ASC.ASC']
        # read
        stream = _read_asc(testfile, skip=100, delta=0.1, length=2)
        stream.verify()
        # skip force one trace only
        assert len(stream) == 1
        # headers
        assert stream[0].stats.delta == 1.000000e-01
        assert stream[0].stats.npts == 2
        # check samples
        assert len(stream[0].data) == 2
        assert round(abs(stream[0].data[0]-111.7009), 4) == 0
        assert round(abs(stream[0].data[1]-119.5831), 4) == 0

    def test_write_small_trace(self):
        """
        Tests writing Traces containing 0, 1 or 2 samples only.
        """
        for format in ['SH_ASC', 'Q']:
            for num in range(0, 4):
                tr = Trace(data=np.arange(num))
                with NamedTemporaryFile() as tf:
                    tempfile = tf.name
                    if format == 'Q':
                        tempfile += '.QHD'
                    tr.write(tempfile, format=format)
                    # test results
                    with warnings.catch_warnings() as _:  # NOQA
                        warnings.simplefilter("ignore")
                        st = read(tempfile, format=format)
                    assert len(st) == 1
                    assert len(st[0]) == num
                    # Q files consist of two files - deleting additional file
                    if format == 'Q':
                        os.remove(tempfile[:-4] + '.QBN')
                        os.remove(tempfile[:-4] + '.QHD')

    def test_write_long_header(self):
        """
        Test for issue #1526
        """
        tr = read()[0]
        comment = 'This is a long comment for testing purposes.'
        tr.stats.sh = {'COMMENT': ' '.join(4 * [comment])}
        with NamedTemporaryFile(suffix='.QHD') as tf:
            tempfile = tf.name
            tr.write(tempfile, format="Q")
            tr2 = read(tempfile)[0]
            # remove binary file too (dynamically created)
            os.remove(os.path.splitext(tempfile)[0] + '.QBN')
        assert tr.stats.sh.COMMENT == tr2.stats.sh.COMMENT

    def test_header_whitespaces(self):
        """
        Test for issue #1552
        """
        tr = read()[0]
        tr.stats.sh = {'COMMENT': 30 * '   *   '}
        with NamedTemporaryFile(suffix='.QHD') as tf:
            tempfile = tf.name
            tr.write(tempfile, format="Q")
            tr2 = read(tempfile)[0]
            # remove binary file too (dynamically created)
            os.remove(os.path.splitext(tempfile)[0] + '.QBN')
        assert len(tr.stats.sh.COMMENT) == len(tr2.stats.sh.COMMENT)
        assert tr.stats.sh.COMMENT == tr2.stats.sh.COMMENT

    def test_append_traces(self):
        """
        Test for issue #2870
        """
        stream = read()
        with NamedTemporaryFile(suffix='.QHD') as tf:
            tempfile = tf.name
            stream.write(tempfile, format="Q")
            with open(tempfile) as f:
                header1 = f.read()
            # remove binary file too (dynamically created)
            os.remove(os.path.splitext(tempfile)[0] + '.QBN')
        with NamedTemporaryFile(suffix='.QHD') as tf:
            tempfile = tf.name
            stream[:1].write(tempfile, format="Q")
            stream[1:].write(tempfile, format="Q", append=True)
            with open(tempfile) as f:
                header2 = f.read()
            # remove binary file too (dynamically created)
            os.remove(os.path.splitext(tempfile)[0] + '.QBN')
        assert header2 == header1
