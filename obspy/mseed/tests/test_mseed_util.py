# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str
from datetime import datetime
from struct import pack, unpack

from obspy import UTCDateTime
from obspy.mseed import util
from obspy.mseed.headers import FIXED_HEADER_ACTIVITY_FLAGS, \
    FIXED_HEADER_DATA_QUAL_FLAGS, FIXED_HEADER_IO_CLOCK_FLAGS
from obspy.mseed.core import readMSEED
from obspy.core.util import NamedTemporaryFile
from obspy.core import Stream, Trace
from obspy.mseed.util import set_flags_in_fixed_headers

import io
import numpy as np
import os
import random
import sys
import unittest
import warnings
import copy


class MSEEDUtilTestCase(unittest.TestCase):
    """
    Tests suite for util module of obspy.mseed.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)
        # mseed steim compression is big endian
        if sys.byteorder == 'little':
            self.swap = 1
        else:
            self.swap = 0

    def test_convertDatetime(self):
        """
        Tests all time conversion methods.
        """
        # These values are created using the Linux "date -u -d @TIMESTRING"
        # command. These values are assumed to be correct.
        timesdict = {
            1234567890: UTCDateTime(2009, 2, 13, 23, 31, 30),
            1111111111: UTCDateTime(2005, 3, 18, 1, 58, 31),
            1212121212: UTCDateTime(2008, 5, 30, 4, 20, 12),
            1313131313: UTCDateTime(2011, 8, 12, 6, 41, 53),
            100000: UTCDateTime(1970, 1, 2, 3, 46, 40),
            100000.111112: UTCDateTime(1970, 1, 2, 3, 46, 40, 111112),
            200000000: UTCDateTime(1976, 5, 3, 19, 33, 20),
            # test rounding of 7th digit of microseconds
            1388479508.871572: UTCDateTime(1388479508.8715718),
        }
        # Loop over timesdict.
        for ts, dt in timesdict.items():
            self.assertEqual(dt, util._convertMSTimeToDatetime(ts * 1000000))
            self.assertEqual(ts * 1000000, util._convertDatetimeToMSTime(dt))
        # Additional sanity tests.
        # Today.
        now = UTCDateTime()
        self.assertEqual(now, util._convertMSTimeToDatetime(
            util._convertDatetimeToMSTime(now)))
        # Some random date.
        random.seed(815)  # make test reproducible
        timestring = random.randint(0, 2000000) * 1e6
        self.assertEqual(timestring, util._convertDatetimeToMSTime(
            util._convertMSTimeToDatetime(timestring)))

    def test_getRecordInformation(self):
        """
        Tests the util._getMSFileInfo method with known values.
        """
        filename = os.path.join(self.path, 'data',
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        # Simply reading the file.
        info = util.getRecordInformation(filename)
        self.assertEqual(info['filesize'], 5120)
        self.assertEqual(info['record_length'], 512)
        self.assertEqual(info['number_of_records'], 10)
        self.assertEqual(info['excess_bytes'], 0)
        # Now with an open file. This should work regardless of the current
        # value of the file pointer and it should also not change the file
        # pointer.
        with open(filename, 'rb') as open_file:
            open_file.seek(1234)
            info = util.getRecordInformation(open_file)
            self.assertEqual(info['filesize'], 5120 - 1234)
            self.assertEqual(info['record_length'], 512)
            self.assertEqual(info['number_of_records'], 7)
            self.assertEqual(info['excess_bytes'], 302)
            self.assertEqual(open_file.tell(), 1234)
        # Now test with a BytesIO with the first ten percent.
        with open(filename, 'rb') as open_file:
            open_file_string = io.BytesIO(open_file.read())
        open_file_string.seek(111)
        info = util.getRecordInformation(open_file_string)
        self.assertEqual(info['filesize'], 5120 - 111)
        self.assertEqual(info['record_length'], 512)
        self.assertEqual(info['number_of_records'], 9)
        self.assertEqual(info['excess_bytes'], 401)
        self.assertEqual(open_file_string.tell(), 111)
        # One more file containing two records.
        filename = os.path.join(self.path, 'data', 'test.mseed')
        info = util.getRecordInformation(filename)
        self.assertEqual(info['filesize'], 8192)
        self.assertEqual(info['record_length'], 4096)
        self.assertEqual(info['number_of_records'], 2)
        self.assertEqual(info['excess_bytes'], 0)

    def test_getDataQuality(self):
        """
        This test reads a self-made Mini-SEED file with set Data Quality Bits.
        A real test file would be better as this test tests a file that was
        created by the inverse method that reads the bits.
        """
        filename = os.path.join(self.path, 'data', 'qualityflags.mseed')
        # Read quality flags.
        result = util.getTimingAndDataQuality(filename)
        # The test file contains 18 records. The first record has no set bit,
        # bit 0 of the second record is set, bit 1 of the third, ..., bit 7 of
        # the 9th record is set. The last nine records have 0 to 8 set bits,
        # starting with 0 bits, bit 0 is set, bits 0 and 1 are set...
        # Altogether the file contains 44 set bits.
        self.assertEqual(result,
                         {'data_quality_flags': [9, 8, 7, 6, 5, 4, 3, 2]})
        # No set quality flags should result in a list of zeros.
        filename = os.path.join(self.path, 'data', 'test.mseed')
        result = util.getTimingAndDataQuality(filename)
        self.assertEqual(result,
                         {'data_quality_flags': [0, 0, 0, 0, 0, 0, 0, 0]})

    def test_getStartAndEndTime(self):
        """
        Tests getting the start- and endtime of a file.

        The values are compared with the results of reading the full files.
        """
        mseed_filenames = ['BW.BGLD.__.EHE.D.2008.001.first_10_records',
                           'test.mseed', 'timingquality.mseed']
        for _i in mseed_filenames:
            filename = os.path.join(self.path, 'data', _i)
            # Get the start- and end time.
            (start, end) = util.getStartAndEndTime(filename)
            # Parse the whole file.
            stream = readMSEED(filename)
            self.assertEqual(start, stream[0].stats.starttime)
            self.assertEqual(end, stream[0].stats.endtime)

    def test_getTimingQuality(self):
        """
        This test reads a self-made Mini-SEED file with Timing Quality
        information in Blockette 1001. A real test file would be better.

        The test file contains 101 records with the timing quality ranging from
        0 to 100 in steps of 1.

        The result is compared to the result from the following R command:

        V <- 0:100; min(V); max(V); mean(V); median(V); quantile(V, 0.75,
        type = 3); quantile(V, 0.25, type = 3)
        """
        filename = os.path.join(self.path, 'data', 'timingquality.mseed')
        result = util.getTimingAndDataQuality(filename)
        self.assertEqual(result,
                         {'timing_quality_upper_quantile': 75.0,
                          'data_quality_flags': [0, 0, 0, 0, 0, 0, 0, 0],
                          'timing_quality_min': 0.0,
                          'timing_quality_lower_quantile': 25.0,
                          'timing_quality_average': 50.0,
                          'timing_quality_median': 50.0,
                          'timing_quality_max': 100.0})
        # No timing quality set should result in an empty dictionary.
        filename = os.path.join(self.path, 'data',
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        result = util.getTimingAndDataQuality(filename)
        self.assertEqual(result,
                         {'data_quality_flags': [0, 0, 0, 0, 0, 0, 0, 0]})
        result = util.getTimingAndDataQuality(filename)
        self.assertEqual(result,
                         {'data_quality_flags': [0, 0, 0, 0, 0, 0, 0, 0]})

    def test_unpackSteim1(self):
        """
        Test decompression of Steim1 strings. Remove 64 Bytes of header
        by hand, see SEEDManual_V2.4.pdf page 100.
        """
        steim1_file = os.path.join(self.path, 'data',
                                   'BW.BGLD.__.EHE.D.2008.001.first_record')
        # 64 Bytes header.
        with open(steim1_file, 'rb') as fp:
            data_string = fp.read()[64:]
        data = util._unpackSteim1(data_string, 412, swapflag=self.swap,
                                  verbose=0)
        data_record = readMSEED(steim1_file)[0].data
        np.testing.assert_array_equal(data, data_record)

    def test_unpackSteim2(self):
        """
        Test decompression of Steim2 strings. Remove 128 Bytes of header
        by hand, see SEEDManual_V2.4.pdf page 100.
        """
        steim2_file = os.path.join(self.path, 'data', 'steim2.mseed')
        # 128 Bytes header.
        with open(steim2_file, 'rb') as fp:
            data_string = fp.read()[128:]
        data = util._unpackSteim2(data_string, 5980, swapflag=self.swap,
                                  verbose=0)
        data_record = readMSEED(steim2_file)[0].data
        np.testing.assert_array_equal(data, data_record)

    def test_time_shifting(self):
        """
        Tests the shiftTimeOfFile() function.
        """
        with NamedTemporaryFile() as tf:
            output_filename = tf.name
            # Test a normal file first.
            filename = os.path.join(
                self.path, 'data',
                "BW.BGLD.__.EHE.D.2008.001.first_10_records")
            # Shift by one second.
            util.shiftTimeOfFile(filename, output_filename, 10000)
            st_before = readMSEED(filename)
            st_after = readMSEED(output_filename)
            st_before[0].stats.starttime += 1
            self.assertEqual(st_before, st_after)
            # Shift by 22 seconds in the other direction.
            util.shiftTimeOfFile(filename, output_filename, -220000)
            st_before = readMSEED(filename)
            st_after = readMSEED(output_filename)
            st_before[0].stats.starttime -= 22
            self.assertEqual(st_before, st_after)
            # Shift by 11.33 seconds.
            util.shiftTimeOfFile(filename, output_filename, 113300)
            st_before = readMSEED(filename)
            st_after = readMSEED(output_filename)
            st_before[0].stats.starttime += 11.33
            self.assertEqual(st_before, st_after)

            # Test a special case with the time correction applied flag set but
            # no actual time correction in the field.
            filename = os.path.join(
                self.path, 'data',
                "one_record_time_corr_applied_but_time_corr_is_zero.mseed")
            # Positive shift.
            util.shiftTimeOfFile(filename, output_filename, 22000)
            st_before = readMSEED(filename)
            st_after = readMSEED(output_filename)
            st_before[0].stats.starttime += 2.2
            self.assertEqual(st_before, st_after)
            # Negative shift.
            util.shiftTimeOfFile(filename, output_filename, -333000)
            st_before = readMSEED(filename)
            st_after = readMSEED(output_filename)
            st_before[0].stats.starttime -= 33.3
            self.assertEqual(st_before, st_after)

    def test_time_shifting_special_case(self):
        """
        Sometimes actually changing the time value is necessary. This works but
        is considered experimental and thus emits a warning. Therefore Python
        >= 2.6 only.
        """
        with NamedTemporaryFile() as tf:
            output_filename = tf.name
            # This file was created only for testing purposes.
            filename = os.path.join(
                self.path, 'data',
                "one_record_already_applied_time_correction.mseed")
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('error', UserWarning)
                self.assertRaises(UserWarning, util.shiftTimeOfFile,
                                  input_file=filename,
                                  output_file=output_filename,
                                  timeshift=123400)
                # Now ignore the warnings and test the default values.
                warnings.simplefilter('ignore', UserWarning)
                util.shiftTimeOfFile(input_file=filename,
                                     output_file=output_filename,
                                     timeshift=123400)
            st_before = readMSEED(filename)
            st_after = readMSEED(output_filename)
            st_before[0].stats.starttime += 12.34
            self.assertEqual(st_before, st_after)

            # Test negative shifts.
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('ignore', UserWarning)
                util.shiftTimeOfFile(input_file=filename,
                                     output_file=output_filename,
                                     timeshift=-22222)
            st_before = readMSEED(filename)
            st_after = readMSEED(output_filename)
            st_before[0].stats.starttime -= 2.2222
            self.assertEqual(st_before, st_after)

    def test_checkFlagValue(self):
        """
        Test case for obspy.mseed.util._checkFlagValue
        """

        # Valid value for a boolean flag
        corrected_flag = util._checkFlagValue(True)
        self.assertTrue(isinstance(corrected_flag, bool))
        self.assertTrue(corrected_flag)
        corrected_flag = util._checkFlagValue(False)
        self.assertTrue(isinstance(corrected_flag, bool))
        self.assertFalse(corrected_flag)

        # Valid value for an instant flag #1
        flag_value = {"INSTANT": UTCDateTime("2009-12-23T06:00:00.0")}
        corrected_flag = util._checkFlagValue(flag_value)
        self.assertTrue(isinstance(corrected_flag, list))
        self.assertEqual(corrected_flag[0][0],
                         UTCDateTime("2009-12-23T06:00:00.0"))
        self.assertEqual(corrected_flag[0][1],
                         UTCDateTime("2009-12-23T06:00:00.0"))

        # Valid value for an instant flag #2
        flag_value = {"INSTANT": [UTCDateTime("2009-12-24T06:00:00.0")]}
        corrected_flag = util._checkFlagValue(flag_value)
        self.assertTrue(isinstance(corrected_flag, list))
        self.assertEqual(corrected_flag[0][0],
                         UTCDateTime("2009-12-24T06:00:00.0"))
        self.assertEqual(corrected_flag[0][1],
                         UTCDateTime("2009-12-24T06:00:00.0"))

        # Valid value for several instant flags
        flag_value = {"INSTANT": [UTCDateTime("2009-12-24T06:00:00.0"),
                                  UTCDateTime("2009-12-24T06:01:00.0")]}
        corrected_flag = util._checkFlagValue(flag_value)
        self.assertTrue(isinstance(corrected_flag, list))
        self.assertEqual(corrected_flag[0][0],
                         UTCDateTime("2009-12-24T06:00:00.0"))
        self.assertEqual(corrected_flag[0][1],
                         UTCDateTime("2009-12-24T06:00:00.0"))
        self.assertEqual(corrected_flag[1][0],
                         UTCDateTime("2009-12-24T06:01:00.0"))
        self.assertEqual(corrected_flag[1][1],
                         UTCDateTime("2009-12-24T06:01:00.0"))

        # Valid value for a duration #1
        flag_value = {"DURATION": [UTCDateTime("2009-12-25T06:00:00.0"),
                                   UTCDateTime("2009-12-25T06:10:00.0")]}
        corrected_flag = util._checkFlagValue(flag_value)
        self.assertTrue(isinstance(corrected_flag, list))
        self.assertEqual(corrected_flag[0][0],
                         UTCDateTime("2009-12-25T06:00:00.0"))
        self.assertEqual(corrected_flag[0][1],
                         UTCDateTime("2009-12-25T06:10:00.0"))

        # Valid value for a duration #2
        flag_value = {"DURATION": [(UTCDateTime("2009-12-25T16:00:00.0"),
                                    UTCDateTime("2009-12-25T16:10:00.0"))]}
        corrected_flag = util._checkFlagValue(flag_value)
        self.assertTrue(isinstance(corrected_flag, list))
        self.assertEqual(corrected_flag[0][0],
                         UTCDateTime("2009-12-25T16:00:00.0"))
        self.assertEqual(corrected_flag[0][1],
                         UTCDateTime("2009-12-25T16:10:00.0"))

        # Valid value for several durations #1
        flag_value = {"DURATION": [UTCDateTime("2009-12-24T06:00:00.0"),
                                   UTCDateTime("2009-12-24T06:10:00.0"),
                                   UTCDateTime("2009-12-24T07:00:00.0"),
                                   UTCDateTime("2009-12-24T07:10:00.0")]}
        corrected_flag = util._checkFlagValue(flag_value)
        self.assertTrue(isinstance(corrected_flag, list))
        self.assertEqual(corrected_flag[0][0],
                         UTCDateTime("2009-12-24T06:00:00.0"))
        self.assertEqual(corrected_flag[0][1],
                         UTCDateTime("2009-12-24T06:10:00.0"))
        self.assertEqual(corrected_flag[1][0],
                         UTCDateTime("2009-12-24T07:00:00.0"))
        self.assertEqual(corrected_flag[1][1],
                         UTCDateTime("2009-12-24T07:10:00.0"))

        # Valid value for several durations #2
        flag_value = {"DURATION": [(UTCDateTime("2009-12-25T06:00:00.0"),
                                    UTCDateTime("2009-12-25T06:10:00.0")),
                                   (UTCDateTime("2009-12-25T07:00:00.0"),
                                    UTCDateTime("2009-12-25T07:10:00.0"))]}
        corrected_flag = util._checkFlagValue(flag_value)
        self.assertTrue(isinstance(corrected_flag, list))
        self.assertEqual(corrected_flag[0][0],
                         UTCDateTime("2009-12-25T06:00:00.0"))
        self.assertEqual(corrected_flag[0][1],
                         UTCDateTime("2009-12-25T06:10:00.0"))
        self.assertEqual(corrected_flag[1][0],
                         UTCDateTime("2009-12-25T07:00:00.0"))
        self.assertEqual(corrected_flag[1][1],
                         UTCDateTime("2009-12-25T07:10:00.0"))

        # Test of the (valid) example 1 written in set_flags_in_fixed_headers's
        # docstring
        date1 = UTCDateTime("2009-12-23T06:00:00.0")
        date2 = UTCDateTime("2009-12-23T06:30:00.0")
        date3 = UTCDateTime("2009-12-24T10:00:00.0")
        date4 = UTCDateTime("2009-12-24T10:30:00.0")
        date5 = UTCDateTime("2009-12-26T18:00:00.0")
        date6 = UTCDateTime("2009-12-26T18:04:00.0")
        flag_value = {"INSTANT": [date5, date6],
                      "DURATION": [(date1, date2), (date3, date4)]}
        corrected_flag = util._checkFlagValue(flag_value)
        self.assertTrue(isinstance(corrected_flag, list))
        self.assertEqual(len(corrected_flag), 4)
        # Sort by start date to ensure uniqueness of the list
        corrected_flag.sort(key=lambda val: val[0])
        self.assertEqual(len(corrected_flag), 4)
        self.assertEqual(corrected_flag[0][0],
                         UTCDateTime("2009-12-23T06:00:00.0"))
        self.assertEqual(corrected_flag[0][1],
                         UTCDateTime("2009-12-23T06:30:00.0"))
        self.assertEqual(corrected_flag[1][0],
                         UTCDateTime("2009-12-24T10:00:00.0"))
        self.assertEqual(corrected_flag[1][1],
                         UTCDateTime("2009-12-24T10:30:00.0"))
        self.assertEqual(corrected_flag[2][0],
                         UTCDateTime("2009-12-26T18:00:00.0"))
        self.assertEqual(corrected_flag[2][1],
                         UTCDateTime("2009-12-26T18:00:00.0"))
        self.assertEqual(corrected_flag[3][0],
                         UTCDateTime("2009-12-26T18:04:00.0"))
        self.assertEqual(corrected_flag[3][1],
                         UTCDateTime("2009-12-26T18:04:00.0"))

        # Test of the (valid) example 2 written in set_flags_in_fixed_headers's
        # docstring
        flag_value = {"INSTANT": [date5, date6],
                      "DURATION": [date1, date2, date3, date4]}
        self.assertEqual(len(corrected_flag), 4)
        # Sort by start date to ensure uniqueness of the list
        corrected_flag.sort(key=lambda val: val[0])
        self.assertEqual(len(corrected_flag), 4)
        self.assertEqual(corrected_flag[0][0],
                         UTCDateTime("2009-12-23T06:00:00.0"))
        self.assertEqual(corrected_flag[0][1],
                         UTCDateTime("2009-12-23T06:30:00.0"))
        self.assertEqual(corrected_flag[1][0],
                         UTCDateTime("2009-12-24T10:00:00.0"))
        self.assertEqual(corrected_flag[1][1],
                         UTCDateTime("2009-12-24T10:30:00.0"))
        self.assertEqual(corrected_flag[2][0],
                         UTCDateTime("2009-12-26T18:00:00.0"))
        self.assertEqual(corrected_flag[2][1],
                         UTCDateTime("2009-12-26T18:00:00.0"))
        self.assertEqual(corrected_flag[3][0],
                         UTCDateTime("2009-12-26T18:04:00.0"))
        self.assertEqual(corrected_flag[3][1],
                         UTCDateTime("2009-12-26T18:04:00.0"))

        # Invalid type for datation flag
        flag_value = "invalid because str"
        self.assertRaises(ValueError, util._checkFlagValue, flag_value)

        # Invalid key (neither "INSTANT" nor "DURATION") for datation dict
        flag_value = {"INVALID_KEY": [UTCDateTime("2009-12-25T07:10:00.0")]}
        self.assertRaises(ValueError, util._checkFlagValue, flag_value)

        # Invalid value type for key "INSTANT"
        flag_value = {"INSTANT": "invalid because str"}
        self.assertRaises(ValueError, util._checkFlagValue, flag_value)

        # Invalid value type for key "DURATION"
        flag_value = {"DURATION": "invalid because str"}
        self.assertRaises(ValueError, util._checkFlagValue, flag_value)

        # Invalid len for key DURATION #1
        flag_value = {"DURATION": [UTCDateTime("2009-12-25T06:00:00.0"),
                                   UTCDateTime("2009-12-25T06:10:00.0"),
                                   UTCDateTime("2009-12-25T06:20:00.0")]}
        self.assertRaises(ValueError, util._checkFlagValue, flag_value)

        # Invalid type in DURATION
        flag_value = {"DURATION": [UTCDateTime("2009-12-25T06:00:00.0"),
                                   UTCDateTime("2009-12-25T06:10:00.0"),
                                   UTCDateTime("2009-12-25T06:20:00.0"),
                                   "invalid because str"]}
        self.assertRaises(ValueError, util._checkFlagValue, flag_value)

        # Start after end in DURATION
        flag_value = {"DURATION": [UTCDateTime("2010-12-27T19:50:59.0"),
                                   UTCDateTime("2009-12-25T06:00:00.0")]}
        self.assertRaises(ValueError, util._checkFlagValue, flag_value)

    def test_searchFlagInBlockette(self):
        """
        Test case for obspy.mseed.util._searchFlagInBlockette
        """
        # Write dummy file
        npts = 2000
        np.random.seed(42)  # make test reproducible
        data = np.random.randint(-1000, 1000, npts).astype(np.int32)
        # This header ensures presence of blockettes 1000 and 1001
        stat_header = {'network': 'NE', 'station': 'STATI', 'location': 'LO',
                       'channel': 'CHA', 'npts': len(data), 'sampling_rate': 1,
                       'mseed': {'dataquality': 'D',
                                 'blkt1001': {'timing_quality': 63}}}
        stat_header['starttime'] = UTCDateTime(datetime(2012, 8, 1,
                                                        12, 0, 0, 42))
        trace1 = Trace(data=data, header=stat_header)
        st = Stream([trace1])
        with NamedTemporaryFile() as tf:
            st.write(tf, format="mseed", encoding=11, reclen=512)
            tf.seek(0, os.SEEK_SET)
            file_name = tf.name

            with open(file_name, "rb") as file_desc:
                file_desc.seek(0, os.SEEK_SET)
                # Test from file start
                read_bytes = util._searchFlagInBlockette(file_desc,
                                                         48, 1001, 4, 1)
                self.assertFalse(read_bytes is None)
                self.assertEqual(unpack(native_str(">B"), read_bytes)[0], 63)

                # Test from middle of a record header
                file_desc.seek(14, os.SEEK_CUR)
                file_pos = file_desc.tell()
                read_bytes = util._searchFlagInBlockette(file_desc,
                                                         34, 1000, 6, 1)
                self.assertFalse(read_bytes is None)
                self.assertEqual(unpack(native_str(">B"), read_bytes)[0], 9)
                # Check that file_desc position has not changed
                self.assertEqual(file_desc.tell(), file_pos)

                # Test from middle of a record data
                file_desc.seek(60, os.SEEK_CUR)
                read_bytes = util._searchFlagInBlockette(file_desc,
                                                         -26, 1001, 5, 1)
                self.assertFalse(read_bytes is None)
                self.assertEqual(unpack(native_str(">B"), read_bytes)[0], 42)

                # Test another record. There is at least 3 records in a
                # mseed with 2000 data points and 512 bytes record length
                file_desc.seek(1040, os.SEEK_SET)
                read_bytes = util._searchFlagInBlockette(file_desc,
                                                         32, 1001, 4, 1)
                self.assertEqual(unpack(native_str(">B"), read_bytes)[0], 63)

                # Test missing blockette
                read_bytes = util._searchFlagInBlockette(file_desc,
                                                         32, 201, 4, 4)
                self.assertTrue(read_bytes is None)

    def test_convertFlagsToRawByte(self):
        """
        Test case for obspy.mseed.util._convertFlagsToRawByte
        """

        recstart = UTCDateTime("2009-12-25T06:00:00.0")
        recend = UTCDateTime("2009-12-26T06:00:00.0")
        user_flags = {
            # boolean flags
            'calib_signal': True,
            'time_correction': False,
            # instant value
            'begin_event': [(UTCDateTime("2009-12-25T07:00:00.0"),
                             UTCDateTime("2009-12-25T07:00:00.0"))],
            # duration value (inside record)
            'end_event': [(UTCDateTime("2009-12-25T07:00:00.0"),
                           UTCDateTime("2009-12-25T07:04:00.0"))],
            # instant value at the end of the record
            'positive_leap': [(UTCDateTime("2009-12-26T06:00:00.0"),
                               UTCDateTime("2009-12-26T06:00:00.0"))],
            # value before record start
            'negative_leap': [(UTCDateTime("2001-12-25T06:00:00.0"),
                               UTCDateTime("2001-12-25T06:00:00.0"))],
            # value after record end
            'event_in_progress': [(UTCDateTime("2020-12-25T06:00:00.0"),
                                   UTCDateTime("2020-12-25T06:00:00.0"))]}

        act_flags = util._convertFlagsToRawByte(FIXED_HEADER_ACTIVITY_FLAGS,
                                                user_flags,
                                                recstart, recend)
        self.assertEqual(act_flags, 13)

        user_flags = {
            # duration value (across record start)
            'sta_vol_parity_error_possible':
                [(UTCDateTime("2009-12-25T00:00:00.0"),
                  UTCDateTime("2009-12-26T00:00:00.0"))],
            # duration value (across record end)
            'long_record_read': [(UTCDateTime("2009-12-26T00:00:00.0"),
                                  UTCDateTime("2009-12-27T00:00:00.0"))],
            # duration value (record inside duration)
            'short_record_read': [(UTCDateTime("2009-12-25T00:00:00.0"),
                                   UTCDateTime("2009-12-27T00:00:00.0"))],
            # several datation periods, one matching the record
            'start_of_time_series': [(UTCDateTime("2008-12-25T00:00:00.0"),
                                      UTCDateTime("2008-12-27T00:00:00.0")),
                                     (UTCDateTime("2009-12-26T00:00:00.0"),
                                      UTCDateTime("2009-12-26T00:00:00.0")),
                                     (UTCDateTime("2010-12-25T00:00:00.0"),
                                      UTCDateTime("2010-12-27T00:00:00.0"))]}
        io_clock = util._convertFlagsToRawByte(FIXED_HEADER_IO_CLOCK_FLAGS,
                                               user_flags,
                                               recstart, recend)
        self.assertEqual(io_clock, 15)

        # Quick check of data quality flags
        user_flags = {'amplifier_sat_detected': True,
                      'digitizer_clipping_detected': False,
                      'spikes_detected': True,
                      'glitches_detected': False,
                      'missing_padded_data_present': True,
                      'telemetry_sync_error': False,
                      'digital_filter_maybe_charging': True,
                      'time_tag_questionable': False}
        data_qual = util._convertFlagsToRawByte(FIXED_HEADER_DATA_QUAL_FLAGS,
                                                user_flags,
                                                recstart, recend)
        self.assertEqual(data_qual, 85)

    def test_set_flags_in_fixed_header(self):
        """
        Test case for obspy.mseed.util.set_flags_in_fixed_headers
        """

        # Write mseed file with several traces

        npts = 1000
        np.random.seed(42)  # make test reproducible
        data = np.random.randint(-1000, 1000, npts).astype(np.int32)

        # Test valid data
        stat_header = {'network': 'NE', 'station': 'STATI', 'location': 'LO',
                       'channel': 'CHA', 'npts': len(data), 'sampling_rate': 1,
                       'mseed': {'dataquality': 'D',
                                 'blkt1001': {'timing_quality': 63}}}

        stat_header['starttime'] = UTCDateTime(datetime(2012, 8, 1,
                                                        12, 0, 0, 0))
        trace1 = Trace(data=data, header=stat_header)

        stat_header['channel'] = 'CHB'
        trace2 = Trace(data=data, header=stat_header)

        stat_header['station'] = 'STATJ'
        trace3 = Trace(data=data, header=stat_header)

        st = Stream([trace1, trace2, trace3])
        with NamedTemporaryFile() as tf:
            st.write(tf, format="mseed", encoding=11, reclen=512)
            tf.seek(0, os.SEEK_SET)
            file_name = tf.name

            # Initialize dummy flags with known binary value
            # Matching bytes are 0x15, 0x28, 0x88
            classic_flags = {'activity_flags': {'calib_signal': True,
                                                'begin_event': True,
                                                'positive_leap': True,
                                                'negative_leap': False},
                             'io_clock_flags': {'start_of_time_series': True,
                                                'clock_locked': True},
                             'data_qual_flags': {
                                 'glitches_detected': True,
                                 'time_tag_questionable': True}}

            expected_classic = pack(native_str('BBB'), 0x15, 0x28, 0x88)
            expected_leap_mod = pack(native_str('BBB'), 0x05, 0x28, 0x88)
            expected_glitch_mod = pack(native_str('BBB'), 0x15, 0x28, 0x88)

            # Test update all traces
            all_traces = {'...': copy.deepcopy(classic_flags)}
            set_flags_in_fixed_headers(file_name, all_traces)
            # Check that values changed
            self._check_values(tf, '...', [], expected_classic, 512)

            # Update one trace
            one_trace = {'NE.STATI.LO.CHA': copy.deepcopy(classic_flags)}
            cur_dict = one_trace['NE.STATI.LO.CHA']['activity_flags']
            cur_dict['positive_leap'] = False
            set_flags_in_fixed_headers(file_name, one_trace)
            # Check that values changed
            self._check_values(tf, 'NE.STATI.LO.CHA', [], expected_leap_mod,
                               512)
            # Check that values that should not change, have not
            self._check_values(tf, 'NE.STATI.LO.CHB', [], expected_classic,
                               512)
            self._check_values(tf, 'NE.STATJ.LO.CHB', [], expected_classic,
                               512)
            # Put back previous values
            set_flags_in_fixed_headers(file_name, all_traces)

            # Update specific trace without paying attention to station name
            no_sta = {'NE..LO.CHB': copy.deepcopy(classic_flags)}
            no_sta['NE..LO.CHB']['activity_flags']['positive_leap'] = False
            set_flags_in_fixed_headers(file_name, no_sta)
            self._check_values(tf, 'NE.STATI.LO.CHA', [], expected_classic,
                               512)
            self._check_values(tf, 'NE.STATI.LO.CHB', [], expected_leap_mod,
                               512)
            self._check_values(tf, 'NE.STATJ.LO.CHB', [], expected_leap_mod,
                               512)
            # Put back previous values
            set_flags_in_fixed_headers(file_name, all_traces)

            # Wildcard plus specific traces
            wild_plus = {'NE..LO.CHB': copy.deepcopy(classic_flags),
                         'NE.STATI.LO.CHB': copy.deepcopy(classic_flags)}
            wild_plus['NE..LO.CHB']['activity_flags']['positive_leap'] = False
            cur_dict = wild_plus['NE.STATI.LO.CHB']['data_qual_flags']
            cur_dict['glitches_detected'] = True
            set_flags_in_fixed_headers(file_name, wild_plus)
            self._check_values(tf, 'NE.STATI.LO.CHA', [], expected_classic,
                               512)
            self._check_values(tf, 'NE.STATI.LO.CHB', [],
                               expected_glitch_mod, 512)
            self._check_values(tf, 'NE.STATJ.LO.CHB', [],
                               expected_leap_mod, 512)
            # Put back previous values
            set_flags_in_fixed_headers(file_name, all_traces)

            # Update trace not present in the file
            not_pres = {'NE.NOSTA.LO.CHA': copy.deepcopy(classic_flags)}
            cur_dict = not_pres['NE.NOSTA.LO.CHA']['data_qual_flags']
            cur_dict['glitches_detected'] = False
            set_flags_in_fixed_headers(file_name, not_pres)
            self._check_values(tf, '...', [], expected_classic, 512)
            # Put back previous values
            set_flags_in_fixed_headers(file_name, all_traces)
            self._check_values(tf, '...', [], expected_classic, 512)
            # Non-existing flag values
            wrong_flag = {'...': copy.deepcopy(classic_flags)}
            wrong_flag['...']['activity_flags']['inexistent'] = True
            wrong_flag['...']['wrong_flag_group'] = {}
            wrong_flag['...']['wrong_flag_group']['inexistent_too'] = True
            self.assertRaises(ValueError, set_flags_in_fixed_headers,
                              file_name, wrong_flag)

            # Put back previous values
            set_flags_in_fixed_headers(file_name, all_traces)

            # Test dated flags
            dated_flags = {'activity_flags': {
                # calib should be at first record
                'calib_signal': UTCDateTime("2012-08-01T12:00:30.0"),
                # begin event should be at second record
                'begin_event':
                    {"INSTANT": [datetime(2012, 8, 1, 12, 13, 20, 0),
                                 datetime(2012, 8, 1, 12, 13, 20, 0)]},
                # positive leap should span from first to second record
                'positive_leap':
                    {"DURATION": [UTCDateTime("2012-08-01T12:00:30.0"),
                                  datetime(2012, 8, 1, 12, 13, 20, 0)]},
                'negative_leap': False}}

            expected_first = pack(native_str('BBB'), 0x11, 0x00, 0x00)
            expected_second = pack(native_str('BBB'), 0x10, 0x00, 0x00)
            expected_fourth = pack(native_str('BBB'), 0x14, 0x00, 0x00)
            expected_afterfourth = pack(native_str('BBB'), 0x00, 0x00, 0x00)
            # Test update all traces
            dated_traces = {'NE.STATI.LO.CHA': copy.deepcopy(dated_flags)}
            set_flags_in_fixed_headers(file_name, dated_traces)
            # Verification of the expected flags
            self._check_values(tf, 'NE.STATI.LO.CHA', [0], expected_first, 512)
            self._check_values(tf, 'NE.STATI.LO.CHA', [1, 2],
                               expected_second, 512)
            self._check_values(tf, 'NE.STATI.LO.CHA', [3], expected_fourth,
                               512)
            self._check_values(tf, 'NE.STATI.LO.CHA', [10],
                               expected_afterfourth, 512)

            # Incorrect trace identification
            wrong_trace = {'not_three_points': copy.deepcopy(classic_flags)}
            self.assertRaises(ValueError, set_flags_in_fixed_headers,
                              file_name, wrong_trace)

    def _check_values(self, file_bfr, trace_id, record_numbers, expected_bytes,
                      reclen):
        """
        Check fixed header flags value in a file. Raises AssertError if the
        result is not the one expected by expected_bytes.

        This method is meant to be used by test_set_flags_in_fixed_header. It
        checks the value of the fixed header bytes against expected_bytes in
        every record matching the trace identifier trace_id.

        Trace identification is expected as  a string looking like
        Network.Station.Location.Channel. Empty fields are allowed and will
        match any value.

        :type file_bfr: File or NamedTemporaryFile
        :param file_bfr: the file to test.
        :type trace_id: str
        :param trace_id: trace identification: Network.Station.Location.Channel
        :type record_numbers: list
        :param record_numbers: list of indexes of records to check. Record is
        counted only if it matches the wanted trace_id. Empty list to check all
        records.
        :type expected_bytes: bytes
        :param expected_bytes: the values of the expected flags
        :type reclen: int
        :param reclen: record length across the file
        """

        prev_pos = file_bfr.tell()
        file_bfr.seek(0, os.SEEK_END)
        filesize = file_bfr.tell()
        file_bfr.seek(0, os.SEEK_SET)

        record_count = 0
        while file_bfr.tell() < filesize:
            file_bfr.seek(8, os.SEEK_CUR)
            # Read trace id
            sta = file_bfr.read(5)
            loc = file_bfr.read(2)
            cha = file_bfr.read(3)
            net = file_bfr.read(2)

            # Check whether we want to check this trace
            expectedtrace = trace_id.split(".")
            exp_net = expectedtrace[0]
            exp_sta = expectedtrace[1]
            exp_loc = expectedtrace[2]
            exp_cha = expectedtrace[3]

            if (exp_net == "" or exp_net == net) and \
               (exp_sta == "" or exp_sta == sta) and \
               (exp_loc == "" or exp_loc == loc) and \
               (exp_cha == "" or exp_cha == cha):

                if len(record_numbers) == 0 or record_count in record_numbers:
                    file_bfr.seek(16, os.SEEK_CUR)
                    readbytes = file_bfr.read(3)
                    self.assertEqual(readbytes, expected_bytes,
                                     "Expected bytes")
                else:
                    file_bfr.seek(19, os.SEEK_CUR)
                record_count += 1

                # Move to the next record
                file_bfr.seek(reclen - 39, os.SEEK_CUR)
            else:
                # No match, move directly to the next record
                file_bfr.seek(reclen - 20, os.SEEK_CUR)

        # Move the file_bfr to where it was before
        file_bfr.seek(prev_pos, os.SEEK_SET)


def suite():
    return unittest.makeSuite(MSEEDUtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
