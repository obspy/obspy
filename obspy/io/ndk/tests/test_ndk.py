#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import warnings

from obspy import UTCDateTime, read_events
from obspy.io.ndk.core import (ObsPyNDKException, _parse_date_time, _is_ndk,
                               _read_ndk)
import pytest


class TestNDK():
    """
    Test suite for obspy.io.ndk
    """
    def test_read_single_ndk(self, testdata):
        """
        Test reading a single event from an NDK file and comparing it to a
        QuakeML file that has been manually checked to contain all the
        information in the NDK file.
        """
        filename = testdata['C200604092050A.ndk']
        cat = _read_ndk(filename)

        reference = testdata['C200604092050A.xml']
        ref_cat = read_events(reference)

        assert cat == ref_cat

    def test_read_multiple_events(self, testdata):
        """
        Tests the reading of multiple events in one file. The file has been
        edited to test a variety of settings.
        """
        filename = testdata['multiple_events.ndk']
        cat = _read_ndk(filename)

        assert len(cat) == 6

        # Test the type of moment tensor inverted for.
        assert [
            i.focal_mechanisms[0].moment_tensor.inversion_type
            for i in cat] == ["general", "zero trace", "double couple"] * 2

        # Test the type and duration of the moment rate function.
        assert [
            i.focal_mechanisms[0].moment_tensor.source_time_function.type
            for i in cat] == ["triangle", "box car"] * 3
        assert [
            i.focal_mechanisms[0].moment_tensor.source_time_function.duration
            for i in cat] == [2.6, 7.4, 9.0, 1.8, 2.0, 1.6]

        # Test the type of depth setting.
        assert [i.preferred_origin().depth_type for i in cat] == \
            ["from moment tensor inversion", "from location",
             "from modeling of broad-band P waveforms"] * 2

        # Check the solution type.
        for event in cat[:3]:
            assert "Standard" in \
                          event.focal_mechanisms[0].comments[0].text
        for event in cat[3:]:
            assert "Quick" in \
                          event.focal_mechanisms[0].comments[0].text

    def test_is_ndk(self, testdata, datapath):
        """
        Test for the the _is_ndk() function.
        """
        valid_files = [testdata['C200604092050A.ndk'],
                       testdata['multiple_events.ndk']]
        invalid_files = list(datapath.parent.glob('*.py'))
        assert len(invalid_files) > 0

        for filename in valid_files:
            assert _is_ndk(filename)
        for filename in invalid_files:
            assert not _is_ndk(filename)

    def test_reading_using_obspy_plugin(self, testdata):
        """
        Checks that reading with the read_events() function works correctly.
        """
        filename = testdata['C200604092050A.ndk']
        cat = read_events(filename)

        reference = testdata['C200604092050A.xml']
        ref_cat = read_events(reference)

        assert cat == ref_cat

    def test_reading_from_string_io(self, testdata):
        """
        Tests reading from StringIO.
        """
        filename = testdata['C200604092050A.ndk']
        with open(filename, "rt") as fh:
            file_object = io.StringIO(fh.read())

        cat = read_events(file_object)
        file_object.close()

        reference = testdata['C200604092050A.xml']
        ref_cat = read_events(reference)

        assert cat == ref_cat

    def test_reading_from_bytes_io(self, testdata):
        """
        Tests reading from BytesIO.
        """
        filename = testdata['C200604092050A.ndk']
        with open(filename, "rb") as fh:
            file_object = io.BytesIO(fh.read())

        cat = read_events(file_object)
        file_object.close()

        reference = testdata['C200604092050A.xml']
        ref_cat = read_events(reference)

        assert cat == ref_cat

    def test_reading_from_open_file_in_text_mode(self, testdata):
        """
        Tests reading from an open file in text mode.
        """
        filename = testdata['C200604092050A.ndk']
        with open(filename, "rt") as fh:
            cat = read_events(fh)

        reference = testdata['C200604092050A.xml']
        ref_cat = read_events(reference)

        assert cat == ref_cat

    def test_reading_from_open_file_in_binary_mode(self, testdata):
        """
        Tests reading from an open file in binary mode.
        """
        filename = testdata['C200604092050A.ndk']
        with open(filename, "rb") as fh:
            cat = read_events(fh)

        reference = testdata['C200604092050A.xml']
        ref_cat = read_events(reference)

        assert cat == ref_cat

    def test_reading_the_same_file_twice_does_not_raise_a_warnings(
            self, testdata):
        """
        Asserts that reading the same file twice does not raise a warning
        due to resource identifier already in use.
        """
        filename = testdata['C200604092050A.ndk']
        cat_1 = read_events(filename)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat_2 = read_events(filename)

        assert len(w) == 0
        assert cat_1 == cat_2

        filename = testdata['multiple_events.ndk']
        cat_1 = read_events(filename)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat_2 = read_events(filename)

        assert len(w) == 0
        assert cat_1 == cat_2

    def test_is_ndk_for_file_with_invalid_date(self, testdata):
        """
        Tests the _is_ndk function for a file with invalid date.
        """
        assert not _is_ndk(testdata["faulty_invalid_date.ndk"])

    def test_is_ndk_for_file_with_invalid_latitude(self, testdata):
        """
        Tests the _is_ndk function a file with an invalid latitude.
        """
        assert not _is_ndk(testdata["faulty_invalid_latitude.ndk"])

    def test_is_ndk_for_file_with_infeasible_latitude(self, testdata):
        """
        Tests the _is_ndk function a file with an unfeasible latitude.
        """
        assert not _is_ndk(testdata["faulty_infeasible_latitude.ndk"])

    def test_reading_file_with_multiple_errors(self, testdata):
        """
        Tests reading a file with multiple errors.
        """
        filename = testdata['faulty_multiple_events.ndk']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat = read_events(filename)

        assert len(w) == 6
        assert "Invalid time in event 2" in str(w[0])
        assert "Unknown data type" in str(w[1])
        assert "Moment rate function" in str(w[2])
        assert "Unknown source type" in str(w[3])
        assert "Unknown type of depth" in str(w[4])
        assert "Invalid CMT timestamp" in str(w[5])

        # One event should still be available.
        assert len(cat) == 1

    def test_reading_from_string(self, testdata):
        """
        Tests reading from a string.
        """
        filename = testdata['C200604092050A.ndk']

        reference = testdata['C200604092050A.xml']
        ref_cat = read_events(reference)

        with io.open(filename, "rt") as fh:
            data = fh.read()

        assert _is_ndk(data)
        cat = _read_ndk(data)

        assert cat == ref_cat

    def test_reading_from_bytestring(self, testdata):
        """
        Tests reading from a byte string.
        """
        filename = testdata['C200604092050A.ndk']

        reference = testdata['C200604092050A.xml']
        ref_cat = read_events(reference)

        with io.open(filename, "rb") as fh:
            data = fh.read()

        assert _is_ndk(data)
        cat = _read_ndk(data)

        assert cat == ref_cat

    def test_missing_lines(self, testdata):
        """
        Tests the raised warning if an event has less then 5 lines.
        """
        with open(testdata['multiple_events.ndk'], "rt") \
                as fh:
            lines = [_i.rstrip() for _i in fh.readlines()]

        # Assemble anew and skip last line.
        data = io.StringIO("\n".join(lines[:-1]))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat = read_events(data)

        data.close()

        assert len(w) == 1
        assert "Not a multiple of 5 lines" in str(w[0])
        # Only five events will have been read.
        assert len(cat) == 5

    def test_reading_event_with_faulty_but_often_occurring_timestamp(
            self, testdata):
        """
        The timestamp "O-00000000000000" is not valid according to the NDK
        definition but occurs a lot in the GCMT catalog thus we include it
        here.
        """
        filename = testdata['faulty_cmt_timestamp.ndk']

        cat = read_events(filename)

        assert len(cat) == 1
        comments = cat[0].focal_mechanisms[0].comments
        assert "CMT Analysis Type: Unknown" in comments[0].text
        assert "CMT Timestamp: O-000000000" in comments[1].text

    def test_raise_exception_if_no_events_in_file(self, testdata):
        """
        The parser is fairly relaxed and will skip invalid files. This test
        assures that an exception is raised if every event has been skipped.
        """
        with open(testdata['C200604092050A.ndk'], "rt") \
                as fh:
            lines = [_i.rstrip() for _i in fh.readlines()]

        # Assemble anew and skip last line.
        data = io.StringIO("\n".join(lines[:-1]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ObsPyNDKException):
                read_events(data)

    def test_parse_date_time_function(self):
        """
        Tests the _parse_date_time() function.
        """
        # Simple tests for some valid times.
        date, time = "1997/11/03", "19:17:33.8"
        assert _parse_date_time(date, time) == \
            UTCDateTime(1997, 11, 3, 19, 17, 33, int(8E5))
        date, time = "1996/11/20", "19:42:56.1"
        assert _parse_date_time(date, time) == \
            UTCDateTime(1996, 11, 20, 19, 42, 56, int(1E5))
        date, time = "2005/01/01", "01:20:05.4"
        assert _parse_date_time(date, time) == \
            UTCDateTime(2005, 1, 1, 1, 20, 5, int(4E5))
        date, time = "2013/03/01", "03:29:46.8"
        assert _parse_date_time(date, time) == \
            UTCDateTime(2013, 3, 1, 3, 29, 46, int(8E5))
        date, time = "2013/03/02", "07:53:43.8"
        assert _parse_date_time(date, time) == \
            UTCDateTime(2013, 3, 2, 7, 53, 43, int(8E5))

        # Some more tests for 60s. The tested values are all values occurring
        # in a big NDK test file.
        date, time = "1998/09/27", "00:57:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(1998, 9, 27, 0, 58)
        date, time = "2000/12/22", "16:29:60.0"
        assert _parse_date_time(date, time) == \
            UTCDateTime(2000, 12, 22, 16, 30)
        date, time = "2003/06/19", "23:04:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(2003, 6, 19, 23, 5)
        date, time = "2005/06/20", "02:32:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(2005, 6, 20, 2, 33)
        date, time = "2006/03/02", "17:16:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(2006, 3, 2, 17, 17)
        date, time = "2006/05/26", "10:25:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(2006, 5, 26, 10, 26)
        date, time = "2006/08/20", "13:34:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(2006, 8, 20, 13, 35)
        date, time = "2007/04/20", "00:30:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(2007, 4, 20, 0, 31)
        date, time = "2007/07/02", "00:54:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(2007, 7, 2, 0, 55)
        date, time = "2007/08/27", "17:11:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(2007, 8, 27, 17, 12)
        date, time = "2008/09/24", "01:36:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(2008, 9, 24, 1, 37)
        date, time = "2008/10/05", "10:44:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(2008, 10, 5, 10, 45)
        date, time = "2009/04/17", "04:09:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(2009, 4, 17, 4, 10)
        date, time = "2009/06/03", "14:30:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(2009, 6, 3, 14, 31)
        date, time = "2009/07/20", "10:44:60.0"
        assert _parse_date_time(date, time) == UTCDateTime(2009, 7, 20, 10, 45)
