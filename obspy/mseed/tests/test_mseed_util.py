# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str
from datetime import datetime
from _struct import pack

from obspy import UTCDateTime
from obspy.mseed import util
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


    def test_set_flags_in_fixed_header(self):
        """
        Test case for obspy.mseed.util.set_flags_in_fixed_headers
        """

        # Write mseed file with several traces

        npts = 1000
        np.random.seed(42)  # make test reproducable
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
                             'data_qual_flags': {'glitches_detected': True,
                                                 'time_tag_questionable': 1}}

            expected_classic = pack(native_str('BBB'), 0x15, 0x28, 0x88)
            expected_leap_mod = pack(native_str('BBB'), 0x05, 0x28, 0x88)
            expected_glitch_mod = pack(native_str('BBB'), 0x15, 0x28, 0x88)

            # Test update all traces
            all_traces = {'...': copy.deepcopy(classic_flags)}
            set_flags_in_fixed_headers(file_name, all_traces)
            # Check that values changed
            self._check_values(tf, '...', expected_classic, 512)

            # Update one trace
            one_trace = {'NE.STATI.LO.CHA': copy.deepcopy(classic_flags)}
            cur_dict = one_trace['NE.STATI.LO.CHA']['activity_flags']
            cur_dict['positive_leap'] = False
            set_flags_in_fixed_headers(file_name, one_trace)
            # Check that values changed
            self._check_values(tf, 'NE.STATI.LO.CHA', expected_leap_mod, 512)
            # Check that values that should not change, have not
            self._check_values(tf, 'NE.STATI.LO.CHB', expected_classic, 512)
            self._check_values(tf, 'NE.STATJ.LO.CHB', expected_classic, 512)
            # Put back previous values
            set_flags_in_fixed_headers(file_name, all_traces)

            # Update specific trace without paying attention to station name
            no_sta = {'NE..LO.CHB': copy.deepcopy(classic_flags)}
            no_sta['NE..LO.CHB']['activity_flags']['positive_leap'] = False
            set_flags_in_fixed_headers(file_name, no_sta)
            self._check_values(tf, 'NE.STATI.LO.CHA', expected_classic, 512)
            self._check_values(tf, 'NE.STATI.LO.CHB', expected_leap_mod, 512)
            self._check_values(tf, 'NE.STATJ.LO.CHB', expected_leap_mod, 512)
            # Put back previous values
            set_flags_in_fixed_headers(file_name, all_traces)

            # Wildcard plus specific traces
            wild_plus = {'NE..LO.CHB': copy.deepcopy(classic_flags),
                         'NE.STATI.LO.CHB': copy.deepcopy(classic_flags)}
            wild_plus['NE..LO.CHB']['activity_flags']['positive_leap'] = False
            cur_dict = wild_plus['NE.STATI.LO.CHB']['data_qual_flags']
            cur_dict['glitches_detected'] = False
            set_flags_in_fixed_headers(file_name, wild_plus)
            self._check_values(tf, 'NE.STATI.LO.CHA', expected_classic, 512)
            self._check_values(tf, 'NE.STATI.LO.CHB', expected_glitch_mod, 512)
            self._check_values(tf, 'NE.STATJ.LO.CHB', expected_leap_mod, 512)
            # Put back previous values
            set_flags_in_fixed_headers(file_name, all_traces)

            # Update trace not present in the file
            not_pres = {'NE.NOSTA.LO.CHA': copy.deepcopy(classic_flags)}
            cur_dict = not_pres['NE.NOSTA.LO.CHA']['data_qual_flags']
            cur_dict['glitches_detected'] = False
            set_flags_in_fixed_headers(file_name, not_pres)
            self._check_values(tf, '...', expected_classic, 512)
            # Put back previous values
            set_flags_in_fixed_headers(file_name, all_traces)
            self._check_values(tf, '...', expected_classic, 512)
            # Non-existing flag values
            wrong_flag = {'...': copy.deepcopy(classic_flags)}
            wrong_flag['...']['activity_flags']['inexistent'] = True
            wrong_flag['...']['wrong_flag_group'] = {}
            wrong_flag['...']['wrong_flag_group']['inexistent_too'] = True
            set_flags_in_fixed_headers(file_name, wrong_flag)
            self._check_values(tf, '...', expected_classic, 512)
            # Put back previous values
            set_flags_in_fixed_headers(file_name, all_traces)

            # Incorrect trace identification
            wrong_trace = {'not_three_points': copy.deepcopy(classic_flags)}
            self.assertRaises(ValueError, set_flags_in_fixed_headers,
                              file_name, wrong_trace)

    def _check_values(self, file_bfr, trace_id, expected_bytes, reclen):
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
        :type trace_id: string
        :param trace_id: trace identification: Network.Station.Location.Channel
        :type expected_bytes: binary string
        :param expected_bytes: the values of the expected flags
        :type reclen: int
        :param reclen: record length across the file
        """

        prev_pos = file_bfr.tell()
        file_bfr.seek(0, os.SEEK_END)
        filesize = file_bfr.tell()
        file_bfr.seek(0, os.SEEK_SET)

        while file_bfr.tell() < filesize:
            file_bfr.seek(8, os.SEEK_CUR)
            # Read trace id
            sta = file_bfr.read(5)
            loc = file_bfr.read(2)
            cha = file_bfr.read(3)
            net = file_bfr.read(2)

            # Check wether we want to check this trace
            expectedtrace = trace_id.split(".")
            exp_net = expectedtrace[0]
            exp_sta = expectedtrace[1]
            exp_loc = expectedtrace[2]
            exp_cha = expectedtrace[3]

            if (exp_net == "" or exp_net == net) and \
               (exp_sta == "" or exp_sta == sta) and \
               (exp_loc == "" or exp_loc == loc) and \
               (exp_cha == "" or exp_net == cha):

                file_bfr.seek(16, os.SEEK_CUR)
                readbytes = file_bfr.read(3)
                self.assertEqual(readbytes, expected_bytes, "Expected bytes")
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
