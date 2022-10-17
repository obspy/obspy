# -*- coding: utf-8 -*-
"""
The obspy.io.segy core test suite.
"""
import io
import os
import unittest
from struct import unpack
import warnings

import numpy as np
import pytest

from obspy import UTCDateTime, read, Trace, Stream
from obspy.core.util import NamedTemporaryFile, AttribDict
from obspy.core.util.testing import WarningsCapture
from obspy.io.segy.core import (SEGYCoreWritingError, SEGYSampleIntervalError,
                                _is_segy, _is_su, _read_segy, _read_su,
                                _write_segy, _write_su)
from obspy.io.segy.segy import _read_segy as _read_segy_internal
from obspy.io.segy.segy import SEGYError, SEGYFile, SEGYTrace, \
    SEGYBinaryFileHeader
from obspy.io.segy.tests import _patch_header
from obspy.io.segy.tests.header import DTYPES, FILES


class SEGYCoreTestCase(unittest.TestCase):
    """
    Test cases for SEG Y reading and writing..
    """
    def setUp(self):
        # directory where the test files are located
        self.dir = os.path.dirname(__file__)
        self.path = os.path.join(self.dir, 'data')
        # All the files and information about them. These files will be used in
        # most tests. data_sample_enc is the encoding of the data value and
        # sample_size the size in bytes of these samples.
        self.files = FILES
        self.dtypes = DTYPES

    def test_is_segy_file(self):
        """
        Tests the _is_segy method.
        """
        # Test all files in the test directory.
        for file in self.files.keys():
            file = os.path.join(self.path, file)
            self.assertEqual(_is_segy(file), True)
        # Also check all the other files in the test directory and they should
        # not work. Just check certain files to ensure reproducibility.
        files = ['test_core.py', 'test_segy.py', '__init__.py']
        for file in files:
            file = os.path.join(self.dir, file)
            self.assertEqual(_is_segy(file), False)

    def test_is_su_file(self):
        """
        Tests the _is_su method.
        """
        # Test all SEG Y files in the test directory.
        for file in self.files.keys():
            file = os.path.join(self.path, file)
            self.assertEqual(_is_su(file), False)
        # Also check all the other files in the test directory and they should
        # not work. Just check certain files to ensure reproducibility.
        files = ['test_core.py', 'test_segy.py', '__init__.py']
        for file in files:
            file = os.path.join(self.dir, file)
            self.assertEqual(_is_su(file), False)
        # Check an actual Seismic Unix file.
        file = os.path.join(self.path, '1.su_first_trace')
        self.assertEqual(_is_su(file), True)

    def test_read_head_only(self):
        """
        Tests headonly flag on _read_segy and _read_su functions.
        """
        # _read_segy
        file = os.path.join(self.path, '1.sgy_first_trace')
        st = _read_segy(file, headonly=True)
        self.assertEqual(st[0].stats.npts, 8000)
        self.assertEqual(len(st[0].data), 0)
        # _read_su
        file = os.path.join(self.path, '1.su_first_trace')
        st = _read_su(file, headonly=True)
        self.assertEqual(st[0].stats.npts, 8000)
        self.assertEqual(len(st[0].data), 0)

    def test_enforcing_textual_header_encoding_while_reading(self):
        """
        Tests whether or not the enforcing of the encoding of the textual file
        header actually works.
        """
        # File ld0042_file_00018.sgy_first_trace has an EBCDIC encoding.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        # Read once with EBCDIC encoding and check if it is correct.
        st1 = _read_segy(file, textual_header_encoding='EBCDIC')
        self.assertEqual(st1.stats.textual_file_header[3:21],
                         b'CLIENT: LITHOPROBE')
        # This should also be written the stats dictionary.
        self.assertEqual(st1.stats.textual_file_header_encoding,
                         'EBCDIC')
        # Reading again with ASCII should yield bad results. Lowercase keyword
        # argument should also work.
        st2 = _read_segy(file, textual_header_encoding='ascii')
        self.assertFalse(st2.stats.textual_file_header[3:21] ==
                         b'CLIENT: LITHOPROBE')
        self.assertEqual(st2.stats.textual_file_header_encoding,
                         'ASCII')
        # Autodetection should also write the textual file header encoding to
        # the stats dictionary.
        st3 = _read_segy(file)
        self.assertEqual(st3.stats.textual_file_header_encoding,
                         'EBCDIC')

    def test_enforcing_endianness_while_writing(self):
        """
        Tests whether or not the enforcing of the endianness while writing
        works.
        """
        # File ld0042_file_00018.sgy_first_trace is in big endian.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        st1 = _read_segy(file)
        # First write should be big endian.
        with NamedTemporaryFile() as tf:
            out_file = tf.name
            _write_segy(st1, out_file)
            st2 = _read_segy(out_file)
            self.assertEqual(st2.stats.endian, '>')
            # Do once again to enforce big endian.
            _write_segy(st1, out_file, byteorder='>')
            st3 = _read_segy(out_file)
            self.assertEqual(st3.stats.endian, '>')
            # Enforce little endian.
            _write_segy(st1, out_file, byteorder='<')
            st4 = _read_segy(out_file)
            self.assertEqual(st4.stats.endian, '<')

    def test_setting_data_encoding_works(self):
        """
        Test whether or not the enforcing the data encoding works.
        """
        # File ld0042_file_00018.sgy_first_trace uses IBM floating point
        # representation.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        st = _read_segy(file)
        # First test if it even works.
        with NamedTemporaryFile() as tf:
            out_file = tf.name
            _write_segy(st, out_file)
            with open(out_file, 'rb') as f:
                data1 = f.read()
            # Write again and enforce encoding one which should yield the same
            # result.
            _write_segy(st, out_file, data_encoding=1)
            with open(out_file, 'rb') as f:
                data2 = f.read()
            self.assertEqual(data1, data2)
            # Writing IEEE floats which should not require any dtype changes.
            _write_segy(st, out_file, data_encoding=5)
            with open(out_file, 'rb') as f:
                data3 = f.read()
            self.assertFalse(data1 == data3)

    def test_reading_and_writing_different_data_encodings(self):
        """
        Writes and reads different data encodings and checks if the data
        remains the same.
        """
        # The file uses IBM data encoding.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        st = _read_segy(file)
        data = st[0].data
        # All working encodings with corresponding dtypes.
        encodings = {1: np.float32,
                     2: np.int32,
                     3: np.int16,
                     5: np.float32}
        with NamedTemporaryFile() as tf:
            out_file = tf.name
            # Loop over all encodings.
            for data_encoding, dtype in encodings.items():
                this_data = np.require(data.copy(), dtype)
                st[0].data = this_data
                _write_segy(st, out_file, data_encoding=data_encoding)
                # Read again and compare data.
                this_stream = _read_segy(out_file)
                # Both should now be equal. Usually converting from IBM to IEEE
                # floating point numbers might result in small rounding errors
                # but in this case it seems to work. Might be different on
                # different computers.
                np.testing.assert_array_equal(this_data, this_stream[0].data)

    def test_not_matching_data_encoding_and_dtype_raises(self):
        """
        obspy.io.segy does not automatically convert to the corresponding
        dtype.
        """
        encodings = [1, 2, 3, 5]
        # The file uses IBM data encoding.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        st = _read_segy(file)
        # Use float64 as the wrong encoding in every case.
        st[0].data = np.require(st[0].data, np.float64)
        with NamedTemporaryFile() as tf:
            out_file = tf.name
            # Loop over all encodings.
            for data_encoding in encodings:
                self.assertRaises(SEGYCoreWritingError, _write_segy, st,
                                  out_file, data_encoding=data_encoding)

    def test_invalid_data_encoding_raises(self):
        """
        Using an invalid data encoding raises an error.
        """
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        st = _read_segy(file)
        with NamedTemporaryFile() as tf:
            out_file = tf.name
            self.assertRaises(SEGYCoreWritingError, _write_segy, st, out_file,
                              data_encoding=0)
            self.assertRaises(SEGYCoreWritingError, _write_segy, st, out_file,
                              data_encoding='')

    def test_enforcing_textual_header_encoding_while_writing(self):
        """
        Tests whether or not the enforcing of the endianness while writing
        works.
        """
        # File ld0042_file_00018.sgy_first_trace has an EBCDIC encoding.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        st1 = _read_segy(file)
        # Save the header to compare it later on.
        with open(file, 'rb') as f:
            header = f.read(3200)

        # All newly written header will have the file revision number and
        # the end header mark set - just also set them in the old header.
        header = _patch_header(header, ebcdic=True)

        # First write should remain EBCDIC.
        with NamedTemporaryFile() as tf:
            out_file = tf.name
            _write_segy(st1, out_file)
            st2 = _read_segy(out_file)
            # Compare header.
            with open(out_file, 'rb') as f:
                new_header = f.read(3200)
        # re-encode both to ASCII to easily compare them.
        self.assertEqual(
            header.decode("EBCDIC-CP-BE").encode("ASCII"),
            new_header.decode("EBCDIC-CP-BE").encode("ASCII"))
        self.assertEqual(st2.stats.textual_file_header_encoding,
                         'EBCDIC')
        # Do once again to enforce EBCDIC.
        _write_segy(st1, out_file, textual_header_encoding='EBCDIC')
        st3 = _read_segy(out_file)
        # Compare header.
        with open(out_file, 'rb') as f:
            new_header = f.read(3200)
        self.assertEqual(header, new_header)
        os.remove(out_file)
        self.assertEqual(st3.stats.textual_file_header_encoding,
                         'EBCDIC')
        # Enforce ASCII
        _write_segy(st1, out_file, textual_header_encoding='ASCII')
        st4 = _read_segy(out_file)
        # Compare header. Should not be equal this time.
        with open(out_file, 'rb') as f:
            new_header = f.read(3200)
        self.assertFalse(header == new_header)
        os.remove(out_file)
        self.assertEqual(st4.stats.textual_file_header_encoding,
                         'ASCII')

    def test_enforcing_endianness_while_reading(self):
        """
        Tests whether or not enforcing the endianness while reading a file
        works. It will actually just deactivate the autodetection in case it
        produced a wrong result. Using a wrong endianness while reading a file
        will still produce an error because the data format will most likely be
        wrong and therefore obspy.io.segy cannot unpack the data.
        """
        # File ld0042_file_00018.sgy_first_trace is in big endian.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        # This should work and write big endian to the stats dictionary.
        st1 = _read_segy(file)
        self.assertEqual(st1.stats.endian, '>')
        # Doing the same with the right endianness should still work.
        st2 = _read_segy(file, byteorder='>')
        self.assertEqual(st2.stats.endian, '>')
        # The wrong endianness should yield an key error because the routine to
        # unpack the wrong data format code cannot be found.
        self.assertRaises(KeyError, _read_segy, file, byteorder='<')

    def test_reading_using_core(self):
        """
        This tests checks whether or not all necessary information is read
        during reading with core. It actually just assumes the internal
        SEGYFile object, which is thoroughly tested in
        obspy.io.segy.tests.test_segy, is correct and compared all values to
        it. This seems to be the easiest way to test everything.
        """
        for file, _ in self.files.items():
            file = os.path.join(self.path, file)
            # Read the file with the internal SEGY representation.
            segy_file = _read_segy_internal(file)
            # Read again using core.
            st = _read_segy(file)
            # They all should have length one because all additional traces
            # have been removed.
            self.assertEqual(len(st), 1)
            # Assert the data is the same.
            np.testing.assert_array_equal(segy_file.traces[0].data, st[0].data)
            # Textual header.
            self.assertEqual(segy_file.textual_file_header,
                             st.stats.textual_file_header)
            # Textual_header_encoding.
            self.assertEqual(segy_file.textual_header_encoding,
                             st.stats.textual_file_header_encoding)
            # Endianness.
            self.assertEqual(segy_file.endian, st.stats.endian)
            # Data encoding.
            self.assertEqual(segy_file.data_encoding,
                             st.stats.data_encoding)
            # Test the file and trace binary headers.
            for key, value in \
                    segy_file.binary_file_header.__dict__.items():
                self.assertEqual(getattr(st.stats.binary_file_header,
                                 key), value)
            for key, value in \
                    segy_file.traces[0].header.__dict__.items():
                self.assertEqual(getattr(st[0].stats.segy.trace_header, key),
                                 value)

    def test_writing_using_core(self):
        """
        Tests the writing of SEGY rev1 files using obspy.core. It just compares
        the output of writing using obspy.core with the output of writing the
        files using the internal SEGY object which is thoroughly tested in
        obspy.io.segy.tests.test_segy.
        """
        for file, _ in self.files.items():
            file = os.path.join(self.path, file)
            # Read the file with the internal SEGY representation.
            segy_file = _read_segy_internal(file)
            # Read again using core.
            st = _read_segy(file)
            # Create two temporary files to write to.
            with NamedTemporaryFile() as tf1:
                out_file1 = tf1.name
                with NamedTemporaryFile() as tf2:
                    out_file2 = tf2.name
                    # Write twice and catch header warnings
                    with WarningsCapture():
                        warnings.simplefilter("ignore")
                        segy_file.write(out_file1)
                        _write_segy(st, out_file2)
                    # Read and delete files.
                    with open(out_file1, 'rb') as f1:
                        data1 = f1.read()
                    with open(out_file2, 'rb') as f2:
                        data2 = f2.read()
            # Test if they are equal.
            self.assertEqual(data1[3200:3600], data2[3200:3600])

    def test_invalid_values_for_textual_header_encoding(self):
        """
        Invalid keyword arguments should be caught gracefully.
        """
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        self.assertRaises(SEGYError, _read_segy, file,
                          textual_header_encoding='BLUB')

    def test_setting_delta_and_sampling_rate_in_stats(self):
        """
        Just checks if the delta and sampling rate attributes are correctly
        set.
        Testing the delta value is enough because the stats attribute takes
        care that delta/sampling rate always match.
        """
        file = os.path.join(self.path, '1.sgy_first_trace')
        segy = _read_segy(file)
        self.assertEqual(segy[0].stats.delta, 250E-6)
        # The same with the Seismic Unix file.
        file = os.path.join(self.path, '1.su_first_trace')
        su = _read_su(file)
        self.assertEqual(su[0].stats.delta, 250E-6)

    def test_writing_new_sampling_rate(self):
        """
        Setting a new sample rate works.
        """
        file = os.path.join(self.path, '1.sgy_first_trace')
        segy = _read_segy(file)
        segy.stats.textual_file_header = \
            _patch_header(segy.stats.textual_file_header)
        segy[0].stats.sampling_rate = 20
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            _write_segy(segy, outfile)
            new_segy = _read_segy(outfile)
        self.assertEqual(new_segy[0].stats.sampling_rate, 20)
        # The same with the Seismic Unix file.
        file = os.path.join(self.path, '1.su_first_trace')
        _read_su(file)

    def test_reading_date(self):
        """
        Reads one file with a set date. The date has been read with SeisView 2
        by the DMNG.
        """
        # Date as read by SeisView 2.
        date = UTCDateTime(year=2005, julday=353, hour=15, minute=7, second=54)
        file = os.path.join(self.path, '1.sgy_first_trace')
        segy = _read_segy(file)
        self.assertEqual(date, segy[0].stats.starttime)
        # The same with the Seismic Unix file.
        file = os.path.join(self.path, '1.su_first_trace')
        su = _read_su(file)
        self.assertEqual(date, su[0].stats.starttime)

    def test_large_sample_rate_interval_raises(self):
        """
        SEG Y supports a sample interval from 1 to 65535 microseconds in steps
        of 1 microsecond. Larger intervals cannot be supported due to the
        definition of the SEG Y format. Therefore the smallest possible
        sampling rate is ~ 15.26 Hz.
        """
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            # Test for SEG Y.
            file = os.path.join(self.path, '1.sgy_first_trace')
            segy = _read_segy(file)
            segy.stats.textual_file_header = \
                _patch_header(segy.stats.textual_file_header)
            # Set the largest possible delta value which should just work.
            segy[0].stats.delta = 0.065535
            _write_segy(segy, outfile)
            # Slightly larger should raise.
            segy[0].stats.delta = 0.065536
            self.assertRaises(SEGYSampleIntervalError, _write_segy, segy,
                              outfile)
            # Same for SU.
            file = os.path.join(self.path, '1.su_first_trace')
            su = _read_su(file)
            # Set the largest possible delta value which should just work.
            su[0].stats.delta = 0.065535
            _write_su(su, outfile)
        # Slightly larger should raise.
        su[0].stats.delta = 0.065536
        self.assertRaises(SEGYSampleIntervalError, _write_su, su, outfile)

    def test_writing_su_file_with_no_header(self):
        """
        If the trace has no trace.su attribute, one should still be able to
        write a SeismicUnix file.

        This is not recommended because most Trace.stats attributes will be
        lost while writing SU.
        """
        st = read()
        del st[1:]
        st[0].data = np.require(st[0].data, np.float32)
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            st.write(outfile, format='SU')
            st2 = read(outfile)
            # Compare new and old stream objects. All the other header
            # attributes will not be set.
            np.testing.assert_array_equal(st[0].data, st2[0].data)
            self.assertEqual(st[0].stats.starttime, st2[0].stats.starttime)
            self.assertEqual(st[0].stats.endtime, st2[0].stats.endtime)
            self.assertEqual(st[0].stats.sampling_rate,
                             st2[0].stats.sampling_rate)
            # Writing and reading this new stream object should not change
            # anything.
            st2.write(outfile, format='SU')
            st3 = read(outfile)
        np.testing.assert_array_equal(st2[0].data, st3[0].data)
        # Remove the su attributes because they will not be equal due to lazy
        # header attributes.
        del st2[0].stats.su
        del st3[0].stats.su
        self.assertEqual(st2[0].stats, st3[0].stats)

    def test_writing_modified_date(self):
        """
        Tests if the date in Trace.stats.starttime is correctly written in SU
        and SEGY files.
        """
        # Define new date!
        new_date = UTCDateTime(2010, 7, 7, 2, 2, 2)
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            # Test for SEGY.
            file = os.path.join(self.path, 'example.y_first_trace')
            segy = _read_segy(file)
            segy[0].stats.starttime = new_date
            _write_segy(segy, outfile)
            segy_new = _read_segy(outfile)
            self.assertEqual(new_date, segy_new[0].stats.starttime)
            # Test for SU.
            file = os.path.join(self.path, '1.su_first_trace')
            su = _read_su(file)
            su[0].stats.starttime = new_date
            _write_su(su, outfile)
            su_new = _read_su(outfile)
        self.assertEqual(new_date, su_new[0].stats.starttime)

    def test_writing_starttime_timestamp_0(self):
        """
        If the starttime of the Trace is UTCDateTime(0) it will be interpreted
        as a missing starttime is not written. Test if this holds True.
        """
        file = os.path.join(self.path, '1.sgy_first_trace')
        # This file has a set date!
        with open(file, 'rb') as f:
            f.seek(3600 + 156, 0)
            date_time = f.read(10)
        year, julday, hour, minute, second = unpack(b'>5h', date_time)
        self.assertEqual([year == 2005, julday == 353, hour == 15, minute == 7,
                          second == 54], 5 * [True])
        # Read and set zero time.
        segy = _read_segy(file)
        segy.stats.textual_file_header = \
            _patch_header(segy.stats.textual_file_header)
        segy[0].stats.starttime = UTCDateTime(0)
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            _write_segy(segy, outfile)
            # Check the new date.
            with open(outfile, 'rb') as f:
                f.seek(3600 + 156, 0)
                date_time = f.read(10)
        year, julday, hour, minute, second = unpack(b'>5h', date_time)
        self.assertEqual([year == 0, julday == 0, hour == 0, minute == 0,
                          second == 0], 5 * [True])
        # The same for SU.
        file = os.path.join(self.path, '1.su_first_trace')
        # This file has a set date!
        with open(file, 'rb') as f:
            f.seek(156, 0)
            date_time = f.read(10)
        year, julday, hour, minute, second = unpack(b'<5h', date_time)
        self.assertEqual([year == 2005, julday == 353, hour == 15, minute == 7,
                          second == 54], 5 * [True])
        # Read and set zero time.
        su = _read_su(file)
        su[0].stats.starttime = UTCDateTime(0)
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            _write_su(su, outfile)
            # Check the new date.
            with open(outfile, 'rb') as f:
                f.seek(156, 0)
                date_time = f.read(10)
        year, julday, hour, minute, second = unpack(b'<5h', date_time)
        self.assertEqual([year == 0, julday == 0, hour == 0, minute == 0,
                          second == 0], 5 * [True])

    def test_two_digit_years_segy(self):
        """
        Even tough not specified in the 1975 SEG Y rev 1 standard, 2 digit
        years should be read correctly. Some programs produce them.

        Every two digit year < 30 will be mapped to 2000-2029 and every two
        digit year >=30 <100 will be mapped to 1930-1999.
        """
        # Read two artificial test files and check the years.
        filename = os.path.join(self.path, 'one_trace_year_11.sgy')
        st = _read_segy(filename)
        self.assertEqual(2011, st[0].stats.starttime.year)
        filename = os.path.join(self.path, 'one_trace_year_99.sgy')
        st = _read_segy(filename)
        self.assertEqual(1999, st[0].stats.starttime.year)

    def test_two_digit_years_su(self):
        """
        Same test as test_TwoDigitYearsSEGY just for Seismic Unix files.
        """
        # Read two artificial test files and check the years.
        filename = os.path.join(self.path, 'one_trace_year_11.su')
        st = _read_su(filename)
        self.assertEqual(2011, st[0].stats.starttime.year)
        filename = os.path.join(self.path, 'one_trace_year_99.su')
        st = _read_su(filename)
        self.assertEqual(1999, st[0].stats.starttime.year)

    def test_issue_377(self):
        """
        Tests that _read_segy() and stream.write() should handle negative trace
        header values.
        """
        filename = os.path.join(self.path, 'one_trace_year_11.sgy')
        st = _read_segy(filename)
        st[0].stats.segy.trace_header['source_coordinate_x'] = -1
        st.stats.textual_file_header = \
            _patch_header(st.stats.textual_file_header)
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            st.write(outfile, format='SEGY')

    def test_comparing_still_packed_trace_headers(self):
        """
        Regression test to guard against an issue that caused an exception
        to be raised when attempting to compare two still packed trace headers.

        The exception only occured when reading the `obspy.read()`.
        """
        file = os.path.join(self.path, '1.sgy_first_trace')
        # The exception was
        header_a = read(file)[0].stats.segy.trace_header
        header_b = read(file)[0].stats.segy.trace_header
        self.assertEqual(header_a, header_b)

    def test_reading_and_writing_with_unset_dates(self):
        f = SEGYFile()
        f.binary_file_header = SEGYBinaryFileHeader()
        s = SEGYTrace()
        f.traces = [s]
        s.data = np.ones(10, dtype=np.float32)

        # Create a dummy file with only the year set.
        s.header.year_data_recorded = 2015
        with io.BytesIO() as buf:
            f.write(buf, data_encoding=5)
            buf.seek(0, 0)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                st = read(buf, format="segy")
        # Should result in the correct year.
        self.assertEqual(st[0].stats.starttime, UTCDateTime(2015, 1, 1))
        self.assertEqual(len(w), 1)
        self.assertEqual(
            w[0].message.args[0],
            "Trace starttime does not store a proper date (day of year is "
            "zero). Using January 1st 00:00 as trace start time.")

        # No date set at all.
        s.header.year_data_recorded = 0
        with io.BytesIO() as buf:
            f.write(buf, data_encoding=5)
            buf.seek(0, 0)
            st = read(buf, format="segy")
        # Results in 1970, 1, 1
        self.assertEqual(st[0].stats.starttime, UTCDateTime(0))

    def test_writing_text_and_binary_textual_file_headers(self):
        """
        Make sure the textual file header can be written if has been passed
        either as text or as a bytestring.
        """
        # Loop over bytes/text and the textual header encoding.
        for textual_file_header in [b"12345", "12345"]:
            for encoding in ["ASCII", "EBCDIC"]:
                st = read()
                for tr in st:
                    tr.data = np.require(tr.data, dtype=np.float32)
                st.stats = AttribDict()
                st.stats.textual_file_header = textual_file_header
                with io.BytesIO() as buf:
                    # Warning raised to create a complete header.
                    with pytest.warns(UserWarning):
                        st.write(buf, format="SEGY", data_encoding=5,
                                 textual_header_encoding=encoding)
                    buf.seek(0, 0)
                    # Read with SEG-Y to preserve the textual file header.
                    st2 = _read_segy(buf)
                self.assertEqual(
                    # Ignore the auto-generated parts of the header.
                    st2.stats.textual_file_header.decode().split()[0],
                    "12345")

    def test_writing_too_long_trace(self):
        """
        Test nice exception message when trying to write a too long trace
        (#1393)
        """
        x = np.arange(32768, dtype=np.int32)
        tr = Trace(x)
        tr.stats.sampling_rate = 100
        st = Stream([tr])
        bio = io.BytesIO()
        with self.assertRaises(ValueError) as e:
            _write_segy(st, bio, data_encoding=2)
        self.assertEqual(
            str(e.exception),
            "Can not write traces with more than 32767 samples (trace at "
            "index 0):\n... | 1970-01-01T00:00:00.000000Z - "
            "1970-01-01T00:05:27.670000Z | 100.0 Hz, 32768 samples")
