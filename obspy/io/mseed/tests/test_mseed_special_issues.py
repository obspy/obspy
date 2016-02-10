# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import ctypes as C
import io
import multiprocessing
import os
import platform
import random
import signal
import sys
import unittest
import warnings

import numpy as np

from obspy import Stream, Trace, UTCDateTime, read
from obspy.core.compatibility import from_buffer
from obspy.core.util import NamedTemporaryFile
from obspy.core.util.attribdict import AttribDict
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from obspy.io.mseed import util
from obspy.io.mseed.core import _read_mseed, _write_mseed, \
    InternalMSEEDReadingError, InternalMSEEDReadingWarning
from obspy.io.mseed.headers import clibmseed
from obspy.io.mseed.msstruct import _MSStruct


# some Python version don't support negative timestamps
NO_NEGATIVE_TIMESTAMPS = False
try:
    UTCDateTime(-50000)
except:
    NO_NEGATIVE_TIMESTAMPS = True


def _test_function(filename):
    """
    Internal function used by MSEEDSpecialIssueTestCase.test_infinite_loop
    """
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            st = read(filename)  # noqa @UnusedVariable
        except (ValueError, InternalMSEEDReadingError):
            # Should occur with broken files
            pass


class MSEEDSpecialIssueTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)
        # mseed steim compression is big endian
        if sys.byteorder == 'little':
            self.swap = 1
        else:
            self.swap = 0

    def test_invalid_record_length(self):
        """
        An invalid record length should raise an exception.
        """
        npts = 6000
        np.random.seed(815)  # make test reproducible
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            data = np.random.randint(-1000, 1000, npts).astype(np.int32)
            st = Stream([Trace(data=data)])
            # Writing should fail with invalid record lengths.
            # Not a power of 2.
            self.assertRaises(ValueError, _write_mseed, st, tempfile,
                              format="MSEED", reclen=1000)
            # Too small.
            self.assertRaises(ValueError, _write_mseed, st, tempfile,
                              format="MSEED", reclen=8)
            # Not a number.
            self.assertRaises(ValueError, _write_mseed, st, tempfile,
                              format="MSEED", reclen='A')

    def test_invalid_encoding(self):
        """
        An invalid encoding should raise an exception.
        """
        npts = 6000
        np.random.seed(815)  # make test reproducible
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            data = np.random.randint(-1000, 1000, npts).astype(np.int32)
            st = Stream([Trace(data=data)])
            # Writing should fail with invalid record lengths.
            # Wrong number.
            self.assertRaises(ValueError, _write_mseed, st, tempfile,
                              format="MSEED", encoding=2)
            # Wrong Text.
            self.assertRaises(ValueError, _write_mseed, st, tempfile,
                              format="MSEED", encoding='FLOAT_64')

    def test_ctypes_arg_types(self):
        """
        Test that ctypes argtypes are set for type checking
        """
        argument_error = C.ArgumentError
        cl = clibmseed
        args = [C.pointer(C.pointer(C.c_int())), 'a', 1, 1.5, 1, 0, 0, 0, 0]
        self.assertRaises(argument_error, cl.ms_readtraces, *args)
        self.assertRaises(TypeError, cl.ms_readtraces, *args[:-1])
        self.assertRaises(argument_error, cl.ms_readmsr_r, *args)
        self.assertRaises(TypeError, cl.ms_readmsr_r, *args[:-1])
        self.assertRaises(argument_error, cl.mst_printtracelist, *args[:5])
        self.assertRaises(argument_error, cl.ms_detect, *args[:4])
        args.append(1)  # 10 argument function
        self.assertRaises(argument_error, cl.mst_packgroup, *args)
        args = ['hallo']  # one argument functions
        self.assertRaises(argument_error, cl.msr_starttime, *args)
        self.assertRaises(argument_error, cl.msr_endtime, *args)
        self.assertRaises(argument_error, cl.mst_init, *args)
        self.assertRaises(argument_error, cl.mst_free, *args)
        self.assertRaises(argument_error, cl.mst_initgroup, *args)
        self.assertRaises(argument_error, cl.mst_freegroup, *args)
        self.assertRaises(argument_error, cl.msr_init, *args)

    def test_broken_last_record(self):
        """
        Test if Libmseed is able to read files with broken last record. Use
        both methods, readMSTracesViaRecords and readMSTraces
        """
        file = os.path.join(self.path, "data", "brokenlastrecord.mseed")
        # independent reading of the data
        with open(file, 'rb') as fp:
            data_string = fp.read()[128:]  # 128 Bytes header
        data = util._unpack_steim_2(data_string, 5980, swapflag=self.swap,
                                    verbose=0)
        # test readMSTraces. Will raise an internal warning.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data_record = _read_mseed(file)[0].data

        self.assertEqual(len(w), 1)
        self.assertEqual(w[0].category, InternalMSEEDReadingWarning)

        np.testing.assert_array_equal(data, data_record)

    def test_one_sample_overlap(self):
        """
        Both methods readMSTraces and readMSTracesViaRecords should recognize a
        single sample overlap.
        """
        # create a stream with one sample overlapping
        trace1 = Trace(data=np.zeros(1000))
        trace2 = Trace(data=np.zeros(10))
        trace2.stats.starttime = UTCDateTime(999)
        st = Stream([trace1, trace2])
        # write into MSEED
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            _write_mseed(st, tempfile, format="MSEED")
            # read it again
            new_stream = _read_mseed(tempfile)
            self.assertEqual(len(new_stream), 2)

    def test_bug_write_read_float32_seed_win32(self):
        """
        Test case for issue #64.
        """
        # create stream object
        data = np.array([395.07809448, 395.0782, 1060.28112793, -1157.37487793,
                         -1236.56237793, 355.07028198, -1181.42175293],
                        dtype=np.float32)
        st = Stream([Trace(data=data)])
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            _write_mseed(st, tempfile, format="MSEED")
            # read temp file directly without libmseed
            with open(tempfile, 'rb') as fp:
                fp.seek(56)
                dtype = np.dtype(native_str('>f4'))
                bin_data = from_buffer(fp.read(7 * dtype.itemsize),
                                       dtype=dtype)
            np.testing.assert_array_equal(data, bin_data)
            # read via ObsPy
            st2 = _read_mseed(tempfile)
        # test results
        np.testing.assert_array_equal(data, st2[0].data)

    @unittest.skipIf(NO_NEGATIVE_TIMESTAMPS,
                     'times before 1970 are not supported on this operation '
                     'system')
    def test_write_with_date_time_before_1970(self):
        """
        Write an stream via libmseed with a datetime before 1970.

        This test depends on the platform specific localtime()/gmtime()
        function.
        """
        # create trace
        tr = Trace(data=np.empty(1000))
        tr.stats.starttime = UTCDateTime("1969-01-01T00:00:00")
        # write file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            _write_mseed(Stream([tr]), tempfile, format="MSEED")
            # read again
            stream = _read_mseed(tempfile)
            stream.verify()

    def test_invalid_data_type(self):
        """
        Writing data of type int64 and int16 are not supported.
        """
        npts = 6000
        np.random.seed(815)  # make test reproducible
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            # int64
            data = np.random.randint(-1000, 1000, npts).astype(np.int64)
            st = Stream([Trace(data=data)])
            self.assertRaises(Exception, st.write, tempfile, format="MSEED")
            # int8
            data = np.random.randint(-1000, 1000, npts).astype(np.int8)
            st = Stream([Trace(data=data)])
            self.assertRaises(Exception, st.write, tempfile, format="MSEED")

    def test_write_wrong_encoding(self):
        """
        Test to write a floating point mseed file with encoding STEIM1.
        An exception should be raised.
        """
        file = os.path.join(self.path, "data",
                            "BW.BGLD.__.EHE.D.2008.001.first_record")
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            # Read the data and convert them to float
            st = read(file)
            st[0].data = st[0].data.astype(np.float32) + .5
            # Type is not consistent float32 cannot be compressed with STEIM1,
            # therefore a exception should be raised.
            self.assertRaises(Exception, st.write, tempfile, format="MSEED",
                              encoding=10)

    def test_write_wrong_encoding_via_mseed_stats(self):
        """
        Test to write a floating point mseed file with encoding STEIM1 with the
        encoding set in stats.mseed.encoding.
        This will just raise a warning.
        """
        file = os.path.join(self.path, "data",
                            "BW.BGLD.__.EHE.D.2008.001.first_record")
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            # Read the data and convert them to float
            st = read(file)
            st[0].data = st[0].data.astype(np.float32) + .5
            # Type is not consistent float32 cannot be compressed with STEIM1,
            # therefore a warning should be raised.
            self.assertEqual(st[0].stats.mseed.encoding, 'STEIM1')
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('error', UserWarning)
                self.assertRaises(UserWarning, st.write, tempfile,
                                  format="MSEED")

    def test_wrong_record_length_as_argument(self):
        """
        Specifying a wrong record length should raise an error.
        """
        file = os.path.join(self.path, 'data', 'libmseed',
                            'float32_Float32_bigEndian.mseed')
        self.assertRaises(Exception, read, file, reclen=4096)

    def test_read_quality_information_warns(self):
        """
        Reading the quality information while reading the data files is no more
        supported in newer obspy.io.mseed versions. Check that a warning is
        raised.
        Similar functionality is included in obspy.io.mseed.util.
        """
        timingqual = os.path.join(self.path, 'data', 'timingquality.mseed')
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', ObsPyDeprecationWarning)
            # This should not raise a warning.
            read(timingqual)
            # This should warn.
            self.assertRaises(ObsPyDeprecationWarning, read, timingqual,
                              quality=True)

    def test_read_with_missing_blockette010(self):
        """
        Reading a Full/Mini-SEED w/o blockette 010 but blockette 008.
        """
        # 1 - Mini-SEED
        file = os.path.join(self.path, 'data', 'blockette008.mseed')
        tr = read(file)[0]
        self.assertEqual('BW.PART..EHZ', tr.id)
        self.assertEqual(1642, tr.stats.npts)
        # 2 - full SEED
        file = os.path.join(self.path, 'data',
                            'RJOB.BW.EHZ.D.300806.0000.fullseed')
        tr = read(file)[0]
        self.assertEqual('BW.RJOB..EHZ', tr.id)
        self.assertEqual(412, tr.stats.npts)

    def test_issue160(self):
        """
        Tests issue #160.

        Reading the header of SEED file.
        """
        file = os.path.join(self.path, 'data',
                            'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        tr_one = read(file)[0]
        tr_two = read(file, headonly=True)[0]
        ms = AttribDict({'record_length': 512, 'encoding': 'STEIM1',
                         'filesize': 5120, 'dataquality': 'D',
                         'number_of_records': 10, 'byteorder': '>'})
        for tr in tr_one, tr_two:
            self.assertEqual('BW.BGLD..EHE', tr.id)
            self.assertEqual(ms, tr.stats.mseed)
            self.assertEqual(4120, tr.stats.npts)
            self.assertEqual(UTCDateTime(2008, 1, 1, 0, 0, 20, 510000),
                             tr.stats.endtime)

    def test_issue217(self):
        """
        Tests issue #217.

        Reading a MiniSEED file without sequence numbers and a record length of
        1024.
        """
        file = os.path.join(self.path, 'data',
                            'reclen_1024_without_sequence_numbers.mseed')
        tr = read(file)[0]
        ms = AttribDict({'record_length': 1024, 'encoding': 'STEIM1',
                         'filesize': 2048, 'dataquality': 'D',
                         'number_of_records': 2, 'byteorder': '>'})
        self.assertEqual('XX.STF1..HHN', tr.id)
        self.assertEqual(ms, tr.stats.mseed)
        self.assertEqual(932, tr.stats.npts)
        self.assertEqual(UTCDateTime(2007, 5, 31, 22, 45, 46, 720000),
                         tr.stats.endtime)

    def test_issue296(self):
        """
        Tests issue #296.
        """
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            # 1 - transform to np.float64 values
            st = read()
            for tr in st:
                tr.data = tr.data.astype(np.float64)
            # write a single trace automatically detecting encoding
            st[0].write(tempfile, format="MSEED")
            # write a single trace automatically detecting encoding
            st.write(tempfile, format="MSEED")
            # write a single trace with encoding 5
            st[0].write(tempfile, format="MSEED", encoding=5)
            # write a single trace with encoding 5
            st.write(tempfile, format="MSEED", encoding=5)
            # 2 - transform to np.float32 values
            st = read()
            for tr in st:
                tr.data = tr.data.astype(np.float32)
            # write a single trace automatically detecting encoding
            st[0].write(tempfile, format="MSEED")
            # write a single trace automatically detecting encoding
            st.write(tempfile, format="MSEED")
            # write a single trace with encoding 4
            st[0].write(tempfile, format="MSEED", encoding=4)
            # write a single trace with encoding 4
            st.write(tempfile, format="MSEED", encoding=4)
            # 3 - transform to np.int32 values
            st = read()
            for tr in st:
                tr.data = tr.data.astype(np.int32)
            # write a single trace automatically detecting encoding
            st[0].write(tempfile, format="MSEED")
            # write a single trace automatically detecting encoding
            st.write(tempfile, format="MSEED")
            # write a single trace with encoding 3
            st[0].write(tempfile, format="MSEED", encoding=3)
            # write the whole stream with encoding 3
            st.write(tempfile, format="MSEED", encoding=3)
            # write a single trace with encoding 10
            st[0].write(tempfile, format="MSEED", encoding=10)
            # write the whole stream with encoding 10
            st.write(tempfile, format="MSEED", encoding=10)
            # write a single trace with encoding 11
            st[0].write(tempfile, format="MSEED", encoding=11)
            # write the whole stream with encoding 11
            st.write(tempfile, format="MSEED", encoding=11)
            # 4 - transform to np.int16 values
            st = read()
            for tr in st:
                tr.data = tr.data.astype(np.int16)
            # write a single trace automatically detecting encoding
            st[0].write(tempfile, format="MSEED")
            # write a single trace automatically detecting encoding
            st.write(tempfile, format="MSEED")
            # write a single trace with encoding 1
            st[0].write(tempfile, format="MSEED", encoding=1)
            # write the whole stream with encoding 1
            st.write(tempfile, format="MSEED", encoding=1)
            # 5 - transform to ASCII values
            st = read()
            for tr in st:
                tr.data = tr.data.astype(native_str('|S1'))
            # write a single trace automatically detecting encoding
            st[0].write(tempfile, format="MSEED")
            # write a single trace automatically detecting encoding
            st.write(tempfile, format="MSEED")
            # write a single trace with encoding 0
            st[0].write(tempfile, format="MSEED", encoding=0)
            # write the whole stream with encoding 0
            st.write(tempfile, format="MSEED", encoding=0)

    def test_issue289(self):
        """
        Tests issue #289.

        Reading MiniSEED using start/end time outside of data should result in
        an empty Stream object.
        """
        # 1
        file = os.path.join(self.path, 'data', 'steim2.mseed')
        st = read(file, starttime=UTCDateTime() - 10, endtime=UTCDateTime())
        self.assertEqual(len(st), 0)
        # 2
        file = os.path.join(self.path, 'data', 'fullseed.mseed')
        st = read(file, starttime=UTCDateTime() - 10, endtime=UTCDateTime())
        self.assertEqual(len(st), 0)

    def test_issue312(self):
        """
        Tests issue #312

        The blkt_link struct was defined wrong.
        """
        filename = os.path.join(self.path, 'data',
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        # start and end time
        ms = _MSStruct(filename)
        ms.read(-1, 0, 1, 0)
        blkt_link = ms.msr.contents.blkts.contents
        # The first blockette usually begins after 48 bytes. In the test file
        # it does.
        self.assertEqual(blkt_link.blktoffset, 48)
        # The first blockette is blockette 1000 in this file.
        self.assertEqual(blkt_link.blkt_type, 1000)
        # Only one blockette.
        self.assertEqual(blkt_link.next_blkt, 0)
        # Blockette data is 8 bytes - 4 bytes for the blockette header.
        self.assertEqual(blkt_link.blktdatalen, 4)
        del ms

    def test_issue272(self):
        """
        Tests issue #272

        Option headonly should not read the actual waveform data.
        """
        filename = os.path.join(self.path, 'data',
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        # everything
        st = read(filename)
        self.assertEqual(st[0].stats.npts, 4120)
        self.assertEqual(len(st[0].data), 4120)
        # headers only
        st = read(filename, headonly=True)
        self.assertEqual(st[0].stats.npts, 4120)
        self.assertEqual(len(st[0].data), 0)

    def test_issue325(self):
        """
        Tests issue #325: Use selection with non default dataquality flag.
        """
        filename = os.path.join(self.path, 'data', 'dataquality-m.mseed')
        # 1 - read all
        st = read(filename)
        self.assertEqual(len(st), 3)
        t1 = st[0].stats.starttime
        t2 = st[0].stats.endtime
        # 2 - select full time window
        st2 = read(filename, starttime=t1, endtime=t2)
        self.assertEqual(len(st2), 3)
        for tr in st2:
            del tr.stats.processing
        self.assertEqual(st, st2)
        # 3 - use selection
        st2 = read(filename, starttime=t1, endtime=t2, sourcename='*.*.*.*')
        self.assertEqual(len(st2), 3)
        for tr in st2:
            del tr.stats.processing
        self.assertEqual(st, st2)
        st2 = read(filename, starttime=t1, endtime=t2, sourcename='*')
        self.assertEqual(len(st2), 3)
        for tr in st2:
            del tr.stats.processing
        self.assertEqual(st, st2)
        # 4 - selection without times
        st2 = read(filename, sourcename='*.*.*.*')
        self.assertEqual(len(st2), 3)
        self.assertEqual(st, st2)
        st2 = read(filename, sourcename='*')
        self.assertEqual(len(st2), 3)
        self.assertEqual(st, st2)

    def test_issue332(self):
        """
        Tests issue #332

        Writing traces with wrong encoding in stats should raise only a user
        warning.
        """
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            st = read()
            tr = st[0]
            tr.data = tr.data.astype(np.float64) + .5
            tr.stats.mseed = {'encoding': 0}
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('error', UserWarning)
                self.assertRaises(UserWarning, st.write, tempfile,
                                  format="MSEED")

    def test_issue341(self):
        """
        Tests issue #341

        Read/write of MiniSEED files with huge sampling rates/delta values.
        """
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            # 1 - sampling rate
            st = read()
            tr = st[0]
            tr.stats.sampling_rate = 1000000000.0
            tr.write(tempfile, format="MSEED")
            # read again
            st = read(tempfile)
            self.assertEqual(st[0].stats.sampling_rate, 1000000000.0)
            # 2 - delta
            st = read()
            tr = st[0]
            tr.stats.delta = 10000000.0
            tr.write(tempfile, format="MSEED")
            # read again
            st = read(tempfile)
            self.assertAlmostEqual(st[0].stats.delta, 10000000.0, 0)

    def test_issue485(self):
        """
        Test reading floats and doubles, which are bytswapped nans
        """
        ref = [-1188.07800293, 638.16400146, 395.07809448, 1060.28112793]
        for filename in ('nan_float32.mseed', 'nan_float64.mseed'):
            filename = os.path.join(self.path, 'data', 'encoding', filename)
            data = read(filename)[0].data.tolist()
            np.testing.assert_array_almost_equal(
                data, ref, decimal=8, err_msg='Data of file %s not equal' %
                filename)

    def test_enforcing_reading_byteorder(self):
        """
        Tests if setting the byte order of the header for reading is passed to
        the C functions.

        Quite simple. It just checks if reading with the correct byte order
        works and reading with the wrong byte order fails.
        """
        tr = Trace(data=np.arange(10, dtype=np.int32))

        # Test with little endian.
        memfile = io.BytesIO()
        tr.write(memfile, format="mseed", byteorder="<")
        memfile.seek(0, 0)
        # Reading little endian should work just fine.
        tr2 = read(memfile, header_byteorder="<")[0]
        memfile.seek(0, 0)
        self.assertEqual(tr2.stats.mseed.byteorder, "<")
        # Remove the mseed specific header fields. These are obviously not
        # equal.
        del tr2.stats.mseed
        del tr2.stats._format
        self.assertEqual(tr, tr2)
        # Wrong byte order raises.
        self.assertRaises(ValueError, read, memfile, header_byteorder=">")

        # Same test with big endian
        memfile = io.BytesIO()
        tr.write(memfile, format="mseed", byteorder=">")
        memfile.seek(0, 0)
        # Reading big endian should work just fine.
        tr2 = read(memfile, header_byteorder=">")[0]
        memfile.seek(0, 0)
        self.assertEqual(tr2.stats.mseed.byteorder, ">")
        # Remove the mseed specific header fields. These are obviously not
        # equal.
        del tr2.stats.mseed
        del tr2.stats._format
        self.assertEqual(tr, tr2)
        # Wrong byte order raises.
        self.assertRaises(ValueError, read, memfile, header_byteorder="<")

    def test_long_year_range(self):
        """
        Tests reading and writing years 1900 to 2100.
        """
        tr = Trace(np.arange(5, dtype=np.float32))

        # Year 2056 is non-deterministic for days 1, 256 and 257. These three
        # dates are simply simply not supported right now. See the libmseed
        # documentation for more details.
        # Use every 5th year. Otherwise the test takes too long. Use 1901 as
        # start to get year 2056.
        years = range(1901, 2101, 5)
        for year in years:
            for byteorder in ["<", ">"]:
                memfile = io.BytesIO()
                # Get some random time with the year and the byte order as the
                # seed.
                random.seed(year + ord(byteorder))
                tr.stats.starttime = UTCDateTime(
                    year,
                    julday=random.randrange(1, 365),
                    hour=random.randrange(0, 24),
                    minute=random.randrange(0, 60),
                    second=random.randrange(0, 60))
                if year == 2056:
                    tr.stats.starttime = UTCDateTime(2056, 2, 1)
                tr.write(memfile, format="mseed")
                st2 = read(memfile)
                self.assertEqual(len(st2), 1)
                tr2 = st2[0]
                # Remove the mseed specific header fields. These are obviously
                # not equal.
                del tr2.stats.mseed
                del tr2.stats._format
                self.assertEqual(tr, tr2)

    def test_full_seed_with_non_default_dataquality(self):
        """
        Tests the reading of full SEED files with dataqualities other then D.
        """
        # Test the normal one first.
        filename = os.path.join(self.path, 'data', 'fullseed.mseed')
        st = read(filename)
        self.assertEqual(st[0].stats.mseed.dataquality, "D")

        # Test the others. They should also have identical data.
        filename = os.path.join(self.path, 'data',
                                'fullseed_dataquality_M.mseed')
        st = read(filename)
        data_m = st[0].data
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.mseed.dataquality, "M")

        filename = os.path.join(self.path, 'data',
                                'fullseed_dataquality_R.mseed')
        st = read(filename)
        data_r = st[0].data
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.mseed.dataquality, "R")

        filename = os.path.join(self.path, 'data',
                                'fullseed_dataquality_Q.mseed')
        st = read(filename)
        data_q = st[0].data
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.mseed.dataquality, "Q")

        # Assert that the data is the same.
        np.testing.assert_array_equal(data_m, data_r)
        np.testing.assert_array_equal(data_m, data_q)

    def test_infinite_loop(self):
        """
        Tests that libmseed doesn't enter an infinite loop on buggy files.
        """
        filename = os.path.join(self.path, 'data', 'infinite-loop.mseed')

        process = multiprocessing.Process(target=_test_function,
                                          args=(filename, ))
        process.start()
        process.join(60)

        fail = process.is_alive()
        process.terminate()
        if process.is_alive():
            if platform.system() == 'Windows':
                os.kill(process.pid, signal.CTRL_BREAK_EVENT)
            else:
                os.kill(process.pid, signal.SIGKILL)
        self.assertFalse(fail)

    def test_writing_blockette_100(self):
        """
        Tests that blockette 100 is written correctly. It is only used if
        the sampling rate is higher than 32727 Hz or smaller than 1.0 /
        32727.0 Hz.
        """
        # Three traces, only the middle one needs it.
        tr = Trace(data=np.linspace(0, 100, 101))
        st = Stream(traces=[tr.copy(), tr.copy(), tr.copy()])

        st[1].stats.sampling_rate = 60000.0

        with io.BytesIO() as buf:
            st.write(buf, format="mseed")
            buf.seek(0, 0)
            st2 = read(buf)

        self.assertTrue(np.allclose(
            st[0].stats.sampling_rate,
            st2[0].stats.sampling_rate))
        self.assertTrue(np.allclose(
            st[1].stats.sampling_rate,
            st2[1].stats.sampling_rate))
        self.assertTrue(np.allclose(
            st[2].stats.sampling_rate,
            st2[2].stats.sampling_rate))

        st[1].stats.sampling_rate = 1.0 / 60000.0

        with io.BytesIO() as buf:
            st.write(buf, format="mseed")
            buf.seek(0, 0)
            st2 = read(buf)

        self.assertTrue(np.allclose(
            st[0].stats.sampling_rate,
            st2[0].stats.sampling_rate))
        self.assertTrue(np.allclose(
            st[1].stats.sampling_rate,
            st2[1].stats.sampling_rate))
        self.assertTrue(np.allclose(
            st[2].stats.sampling_rate,
            st2[2].stats.sampling_rate))

    def test_microsecond_accuracy_reading_and_writing_before_1970(self):
        """
        Tests that reading and writing data with microsecond accuracy and
        before 1970 works as expected.
        """
        # Test a couple of timestamps. Positive and negative ones.
        timestamps = [123456.789123, -123456.789123, 1.123400, 1.123412,
                      1.123449, 1.123450, 1.123499, -1.123400, -1.123412,
                      -1.123449, -1.123450, -1.123451, -1.123499]

        for timestamp in timestamps:
            starttime = UTCDateTime(timestamp)
            self.assertEqual(starttime.timestamp, timestamp)

            tr = Trace(data=np.linspace(0, 100, 101))
            tr.stats.starttime = starttime

            with io.BytesIO() as fh:
                tr.write(fh, format="mseed")
                fh.seek(0, 0)
                tr2 = read(fh)[0]

            del tr2.stats.mseed
            del tr2.stats._format

            self.assertEqual(tr2.stats.starttime, starttime)
            self.assertEqual(tr2, tr)


def suite():
    return unittest.makeSuite(MSEEDSpecialIssueTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
