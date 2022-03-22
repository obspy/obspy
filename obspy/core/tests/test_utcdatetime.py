# -*- coding: utf-8 -*-
import copy
import datetime
import itertools
import warnings
from functools import partial
from operator import ge, eq, lt, le, gt, ne

from packaging.version import parse as parse_version
import numpy as np

from obspy import UTCDateTime as UTC
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
import pytest


class TestUTCDateTime:
    """
    Test suite for obspy.core.utcdatetime.UTCDateTime.
    """
    def test_from_string(self):
        """
        Tests initialization from a given time string not ISO8601 compatible.
        """
        # some supported non ISO8601 patterns
        dt = UTC("1970-01-01 12:23:34")
        assert dt == UTC(1970, 1, 1, 12, 23, 34)
        dt = UTC("1970,01,01,12:23:34")
        assert dt == UTC(1970, 1, 1, 12, 23, 34)
        dt = UTC("1970,001,12:23:34")
        assert dt == UTC(1970, 1, 1, 12, 23, 34)
        dt = UTC("20090701121212")
        assert dt == UTC(2009, 7, 1, 12, 12, 12)
        dt = UTC("19700101")
        assert dt == UTC(1970, 1, 1, 0, 0)
        dt = UTC("1970/01/17 12:23:34")
        assert dt == UTC(1970, 1, 17, 12, 23, 34)
        # other non ISO8601 strings should raise an exception
        with pytest.raises(Exception):
            UTC("1970,001,12:23:34", iso8601=True)

    def test_from_numpy_string(self):
        """
        Tests importing from NumPy strings.
        """
        # some strange patterns
        dt = UTC(np.string_("1970-01-01 12:23:34"))
        assert dt == UTC(1970, 1, 1, 12, 23, 34)
        dt = UTC(np.string_("1970,01,01,12:23:34"))
        assert dt == UTC(1970, 1, 1, 12, 23, 34)
        dt = UTC(np.string_("1970,001,12:23:34"))
        assert dt == UTC(1970, 1, 1, 12, 23, 34)
        dt = UTC(np.string_("20090701121212"))
        assert dt == UTC(2009, 7, 1, 12, 12, 12)
        dt = UTC(np.string_("19700101"))
        assert dt == UTC(1970, 1, 1, 0, 0)
        # non ISO8601 strings should raise an exception
        with pytest.raises(Exception):
            UTC(np.string_("1970,001,12:23:34"), iso8601=True)

    def test_from_python_date_time(self):
        """
        Tests initialization from a given time string not ISO8601 compatible.
        """
        dt = UTC(datetime.datetime(1970, 1, 1, 12, 23, 34, 123456))
        assert dt == UTC(1970, 1, 1, 12, 23, 34, 123456)
        dt = UTC(datetime.datetime(1970, 1, 1, 12, 23, 34))
        assert dt == UTC(1970, 1, 1, 12, 23, 34)
        dt = UTC(datetime.datetime(1970, 1, 1))
        assert dt == UTC(1970, 1, 1)
        dt = UTC(datetime.date(1970, 1, 1))
        assert dt == UTC(1970, 1, 1)

    def test_from_numeric(self):
        """
        Tests initialization from a given a numeric value.
        """
        dt = UTC(0.0)
        assert dt == UTC(1970, 1, 1, 0, 0, 0)
        dt = UTC(1240561632.005)
        assert dt == UTC(2009, 4, 24, 8, 27, 12, 5000)
        dt = UTC(1240561632)
        assert dt == UTC(2009, 4, 24, 8, 27, 12)

    def test_from_iso8601_calendar_date_string(self):
        """
        Tests initialization from a given ISO8601 calendar date representation.
        """
        # w/o trailing Z
        dt = UTC("2009-12-31T12:23:34.5")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 500000)
        dt = UTC("2009-12-31T12:23:34.500000")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 500000)
        dt = UTC("2009-12-31T12:23:34.000005")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 5)
        dt = UTC("2009-12-31T12:23:34")
        assert dt == UTC(2009, 12, 31, 12, 23, 34)
        dt = UTC("2009-12-31T12:23")
        assert dt == UTC(2009, 12, 31, 12, 23)
        dt = UTC("2009-12-31T12")
        assert dt == UTC(2009, 12, 31, 12)
        # enforce ISO8601 - no chance to detect that format
        dt = UTC("2009-12-31", iso8601=True)
        assert dt == UTC(2009, 12, 31)
        # compact
        dt = UTC("20091231T122334.5")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 500000)
        dt = UTC("20091231T122334.500000")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 500000)
        dt = UTC("20091231T122334.000005")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 5)
        dt = UTC("20091231T122334")
        assert dt == UTC(2009, 12, 31, 12, 23, 34)
        dt = UTC("20091231T1223")
        assert dt == UTC(2009, 12, 31, 12, 23)
        dt = UTC("20091231T12")
        assert dt == UTC(2009, 12, 31, 12)
        # enforce ISO8601 - no chance to detect that format
        dt = UTC("20091231", iso8601=True)
        assert dt == UTC(2009, 12, 31)
        # w/ trailing Z
        dt = UTC("2009-12-31T12:23:34.5Z")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 500000)
        dt = UTC("2009-12-31T12:23:34.500000Z")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 500000)
        dt = UTC("2009-12-31T12:23:34.000005Z")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 5)
        dt = UTC("2009-12-31T12:23:34Z")
        assert dt == UTC(2009, 12, 31, 12, 23, 34)
        dt = UTC("2009-12-31T12:23Z")
        assert dt == UTC(2009, 12, 31, 12, 23)
        dt = UTC("2009-12-31T12Z")
        assert dt == UTC(2009, 12, 31, 12)
        # compact
        dt = UTC("20091231T122334.5Z")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 500000)
        dt = UTC("20091231T122334.500000Z")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 500000)
        dt = UTC("20091231T122334.000005Z")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 5)
        dt = UTC("20091231T122334Z")
        assert dt == UTC(2009, 12, 31, 12, 23, 34)
        dt = UTC("20091231T1223Z")
        assert dt == UTC(2009, 12, 31, 12, 23)
        dt = UTC("20091231T12Z")
        assert dt == UTC(2009, 12, 31, 12)
        # time zones
        dt = UTC("2009-12-31T12:23:34-01:15")
        assert dt == UTC(2009, 12, 31, 13, 38, 34)
        dt = UTC("2009-12-31T12:23:34.5-01:15")
        assert dt == UTC(2009, 12, 31, 13, 38, 34, 500000)
        dt = UTC("2009-12-31T12:23:34.000005-01:15")
        assert dt == UTC(2009, 12, 31, 13, 38, 34, 5)
        dt = UTC("2009-12-31T12:23:34+01:15")
        assert dt == UTC(2009, 12, 31, 11, 8, 34)
        dt = UTC("2009-12-31T12:23:34.5+01:15")
        assert dt == UTC(2009, 12, 31, 11, 8, 34, 500000)
        dt = UTC("2009-12-31T12:23:34.000005+01:15")
        assert dt == UTC(2009, 12, 31, 11, 8, 34, 5)

    def test_from_iso8601_ordinal_date_string(self):
        """
        Tests initialization from a given ISO8601 ordinal date representation.
        """
        # w/o trailing Z
        dt = UTC("2009-365T12:23:34.5")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 500000)
        dt = UTC("2009-001T12:23:34")
        assert dt == UTC(2009, 1, 1, 12, 23, 34)
        dt = UTC("2009-001T12:23")
        assert dt == UTC(2009, 1, 1, 12, 23)
        dt = UTC("2009-001T12")
        assert dt == UTC(2009, 1, 1, 12)
        dt = UTC("2009-355")
        assert dt == UTC(2009, 12, 21)
        # enforce ISO8601 - no chance to detect that format
        dt = UTC("2009-001", iso8601=True)
        assert dt == UTC(2009, 1, 1)
        # compact
        dt = UTC("2009365T122334.5")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 500000)
        dt = UTC("2009001T122334")
        assert dt == UTC(2009, 1, 1, 12, 23, 34)
        dt = UTC("2009001T1223")
        assert dt == UTC(2009, 1, 1, 12, 23)
        dt = UTC("2009001T12")
        assert dt == UTC(2009, 1, 1, 12)
        dt = UTC("2009355")
        assert dt == UTC(2009, 12, 21)
        # enforce ISO8601 - no chance to detect that format
        dt = UTC("2009001", iso8601=True)
        assert dt == UTC(2009, 1, 1)
        # Compact day 360 - see issues #2868
        dt = UTC("2012360T")
        assert dt == UTC(2012, 12, 25)  # Note leapyear
        # w/ trailing Z
        dt = UTC("2009-365T12:23:34.5Z")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 500000)
        dt = UTC("2009-001T12:23:34Z")
        assert dt == UTC(2009, 1, 1, 12, 23, 34)
        dt = UTC("2009-001T12:23Z")
        assert dt == UTC(2009, 1, 1, 12, 23)
        dt = UTC("2009-001T12Z")
        assert dt == UTC(2009, 1, 1, 12)
        # compact
        dt = UTC("2009365T122334.5Z")
        assert dt == UTC(2009, 12, 31, 12, 23, 34, 500000)
        dt = UTC("2009001T122334Z")
        assert dt == UTC(2009, 1, 1, 12, 23, 34)
        dt = UTC("2009001T1223Z")
        assert dt == UTC(2009, 1, 1, 12, 23)
        dt = UTC("2009001T12Z")
        assert dt == UTC(2009, 1, 1, 12)

    def test_from_iso8601_week_date_string(self):
        """
        Tests initialization from a given ISO8601 week date representation.
        """
        # w/o trailing Z
        dt = UTC("2009-W53-7T12:23:34.5")
        assert dt == UTC(2010, 1, 3, 12, 23, 34, 500000)
        dt = UTC("2009-W01-1T12:23:34")
        assert dt == UTC(2008, 12, 29, 12, 23, 34)
        dt = UTC("2009-W01-1T12:23")
        assert dt == UTC(2008, 12, 29, 12, 23)
        dt = UTC("2009-W01-1T12")
        assert dt == UTC(2008, 12, 29, 12)
        # enforce ISO8601 - no chance to detect that format
        dt = UTC("2009-W01-1", iso8601=True)
        assert dt == UTC(2008, 12, 29)
        # compact
        dt = UTC("2009W537T122334.5")
        assert dt == UTC(2010, 1, 3, 12, 23, 34, 500000)
        dt = UTC("2009W011T122334")
        assert dt == UTC(2008, 12, 29, 12, 23, 34)
        dt = UTC("2009W011T1223")
        assert dt == UTC(2008, 12, 29, 12, 23)
        dt = UTC("2009W011T12")
        assert dt == UTC(2008, 12, 29, 12)
        # enforce ISO8601 - no chance to detect that format
        dt = UTC("2009W011", iso8601=True)
        assert dt == UTC(2008, 12, 29)
        # w/ trailing Z
        dt = UTC("2009-W53-7T12:23:34.5Z")
        assert dt == UTC(2010, 1, 3, 12, 23, 34, 500000)
        dt = UTC("2009-W01-1T12:23:34Z")
        assert dt == UTC(2008, 12, 29, 12, 23, 34)
        dt = UTC("2009-W01-1T12:23Z")
        assert dt == UTC(2008, 12, 29, 12, 23)
        dt = UTC("2009-W01-1T12Z")
        assert dt == UTC(2008, 12, 29, 12)
        # compact
        dt = UTC("2009W537T122334.5Z")
        assert dt == UTC(2010, 1, 3, 12, 23, 34, 500000)
        dt = UTC("2009W011T122334Z")
        assert dt == UTC(2008, 12, 29, 12, 23, 34)
        dt = UTC("2009W011T1223Z")
        assert dt == UTC(2008, 12, 29, 12, 23)
        dt = UTC("2009W011T12Z")
        assert dt == UTC(2008, 12, 29, 12)

    def test_to_string(self):
        """
        Tests __str__ method.
        """
        dt = UTC(1970, 1, 1, 12, 23, 34)
        assert str(dt) == '1970-01-01T12:23:34.000000Z'
        dt = UTC(1970, 1, 1, 12, 23, 34, 500000)
        assert str(dt) == '1970-01-01T12:23:34.500000Z'
        dt = UTC(1970, 1, 1, 12, 23, 34.500000)
        assert str(dt) == '1970-01-01T12:23:34.500000Z'
        dt = UTC(1970, 1, 1, 12, 23, 34, 5)
        assert str(dt) == '1970-01-01T12:23:34.000005Z'
        dt = UTC(1970, 1, 1)
        assert str(dt) == '1970-01-01T00:00:00.000000Z'

    def test_deepcopy(self):
        dt = UTC(1240561632.005001)
        dt2 = copy.deepcopy(dt)
        dt += 68
        assert dt2.timestamp == 1240561632.005001
        assert dt.timestamp == 1240561700.005001

    def test_add(self):
        a = UTC(0.0)
        assert a + 1 == UTC(1970, 1, 1, 0, 0, 1)
        assert a + int(1) == UTC(1970, 1, 1, 0, 0, 1)
        assert a + np.int32(1) == UTC(1970, 1, 1, 0, 0, 1)
        assert a + np.int64(1) == UTC(1970, 1, 1, 0, 0, 1)
        assert a + np.float32(1) == UTC(1970, 1, 1, 0, 0, 1)
        assert a + np.float64(1) == UTC(1970, 1, 1, 0, 0, 1)
        assert a + 1.123456 == UTC(1970, 1, 1, 0, 0, 1, 123456)
        assert a + 60 * 60 * 24 * 31 + 0.1 == UTC(1970, 2, 1, 0, 0, 0, 100000)
        assert a + -0.5 == UTC(1969, 12, 31, 23, 59, 59, 500000)
        td = datetime.timedelta(seconds=1)
        assert a + td == UTC(1970, 1, 1, 0, 0, 1)

    def test_sub(self):
        # 1
        start = UTC(2000, 1, 1, 0, 0, 0, 0)
        end = UTC(2000, 1, 1, 0, 0, 4, 995000)
        assert round(abs(end - start-4.995), 7) == 0
        # 2
        start = UTC(1000, 1, 1, 0, 0, 0, 0)
        end = UTC(1000, 1, 1, 0, 0, 4, 0)
        assert round(abs(end - start-4), 7) == 0
        # 3
        start = UTC(0)
        td = datetime.timedelta(seconds=1)
        assert start - td == UTC(1969, 12, 31, 23, 59, 59)
        # 4
        start = UTC(2000, 1, 1, 0, 0, 0, 999999)
        end = UTC(2000, 1, 1, 0, 0, 1, 1)
        assert round(abs(end - start-0.000002), 6) == 0

    def test_negative_timestamp(self):
        dt = UTC(-1000.1)
        assert str(dt) == "1969-12-31T23:43:19.900000Z"
        assert dt.timestamp == -1000.1

    def test_sub_with_negative_time_stamp(self):
        start = UTC(0)
        end = UTC(-1000.5)
        assert round(abs(end - start--1000.5), 7) == 0

    def test_small_negative_utc_date_time(self):
        """
        Windows OS supports only negative timestamps < -43200
        """
        # 0
        dt = UTC(0)
        assert dt.timestamp == 0
        assert str(dt) == "1970-01-01T00:00:00.000000Z"
        dt = UTC("1970-01-01T00:00:00.000000Z")
        assert dt.timestamp == 0
        assert str(dt) == "1970-01-01T00:00:00.000000Z"
        # -1
        dt = UTC(-1)
        assert dt.timestamp == -1
        assert str(dt) == "1969-12-31T23:59:59.000000Z"
        dt = UTC("1969-12-31T23:59:59.000000Z")
        assert dt.timestamp == -1
        assert str(dt) == "1969-12-31T23:59:59.000000Z"
        # -1.000001
        dt = UTC(-1.000001)
        assert dt.timestamp == -1.000001
        assert str(dt) == "1969-12-31T23:59:58.999999Z"
        dt = UTC("1969-12-31T23:59:58.999999Z")
        assert round(abs(dt.timestamp--1.000001), 6) == 0
        assert str(dt) == "1969-12-31T23:59:58.999999Z"
        # -0.000001
        dt = UTC("1969-12-31T23:59:59.999999Z")
        assert round(abs(dt.timestamp--0.000001), 6) == 0
        assert str(dt) == "1969-12-31T23:59:59.999999Z"
        dt = UTC(-0.000001)
        assert round(abs(dt.timestamp--0.000001), 6) == 0
        assert str(dt) == "1969-12-31T23:59:59.999999Z"
        # -0.000000001 - max precision is nanosecond!
        dt = UTC(-0.000000001, precision=9)
        assert dt.timestamp == -0.000000001
        assert str(dt) == "1969-12-31T23:59:59.999999999Z"
        # -1000.1
        dt = UTC("1969-12-31T23:43:19.900000Z")
        assert dt.timestamp == -1000.1
        assert str(dt) == "1969-12-31T23:43:19.900000Z"
        # -43199.123456
        dt = UTC(-43199.123456)
        assert round(abs(dt.timestamp--43199.123456), 6) == 0
        assert str(dt) == "1969-12-31T12:00:00.876544Z"

    def test_big_negative_UTC(self):
        # 1
        dt = UTC("1969-12-31T23:43:19.900000Z")
        assert dt.timestamp == -1000.1
        assert str(dt) == "1969-12-31T23:43:19.900000Z"
        # 2
        dt = UTC("1905-01-01T12:23:34.123456Z")
        assert dt.timestamp == -2051177785.876544
        assert str(dt) == "1905-01-01T12:23:34.123456Z"

    def test_init_UTC(self):
        dt = UTC(year=2008, month=1, day=1)
        assert str(dt) == "2008-01-01T00:00:00.000000Z"
        dt = UTC(year=2008, julday=1, hour=12, microsecond=5000)
        assert str(dt) == "2008-01-01T12:00:00.005000Z"
        # without parameters returns current date time
        UTC()

    def test_init_UTC_mixing_keyworks_with_arguments(self):
        # times
        dt = UTC(2008, 1, 1, hour=12)
        assert dt == UTC(2008, 1, 1, 12)
        dt = UTC(2008, 1, 1, 12, minute=59)
        assert dt == UTC(2008, 1, 1, 12, 59)
        dt = UTC(2008, 1, 1, 12, 59, second=59)
        assert dt == UTC(2008, 1, 1, 12, 59, 59)
        dt = UTC(2008, 1, 1, 12, 59, 59, microsecond=123456)
        assert dt == UTC(2008, 1, 1, 12, 59, 59, 123456)
        dt = UTC(2008, 1, 1, hour=12, minute=59, second=59,
                 microsecond=123456)
        assert dt == UTC(2008, 1, 1, 12, 59, 59, 123456)
        # dates
        dt = UTC(2008, month=1, day=1)
        assert dt == UTC(2008, 1, 1)
        dt = UTC(2008, 1, day=1)
        assert dt == UTC(2008, 1, 1)
        dt = UTC(2008, julday=1)
        assert dt == UTC(2008, 1, 1)
        # combined
        dt = UTC(2008, julday=1, hour=12, minute=59, second=59,
                 microsecond=123456)
        assert dt == UTC(2008, 1, 1, 12, 59, 59, 123456)

    def test_to_python_date_time_objects(self):
        """
        Tests getDate, getTime, getTimestamp and getDateTime methods.
        """
        dt = UTC(1970, 1, 1, 12, 23, 34, 456789)
        # as function
        assert dt._get_date() == datetime.date(1970, 1, 1)
        assert dt._get_time() == datetime.time(12, 23, 34, 456789)
        assert dt._get_datetime() == \
               datetime.datetime(1970, 1, 1, 12, 23, 34, 456789)
        assert round(abs(dt._get_timestamp()-44614.456789), 7) == 0
        # as property
        assert dt.date == datetime.date(1970, 1, 1)
        assert dt.time == datetime.time(12, 23, 34, 456789)
        assert dt.datetime == \
               datetime.datetime(1970, 1, 1, 12, 23, 34, 456789)
        assert round(abs(dt.timestamp-44614.456789), 7) == 0

    def test_sub_add_float(self):
        """
        Tests subtraction of floats from UTC
        """
        time = UTC(2010, 0o5, 31, 19, 54, 24.490)
        delta = -0.045149
        expected = UTC("2010-05-31T19:54:24.535149Z")

        got1 = time + (-delta)
        got2 = time - delta
        assert round(abs(got1 - got2-0.0), 7) == 0
        assert round(abs(expected.timestamp-got1.timestamp), 6) == 0

    def test_issue_159(self):
        """
        Test case for issue #159.
        """
        dt = UTC("2010-2-13T2:13:11")
        assert dt == UTC(2010, 2, 13, 2, 13, 11)
        dt = UTC("2010-2-13T02:13:11")
        assert dt == UTC(2010, 2, 13, 2, 13, 11)
        dt = UTC("2010-2-13T2:13:11.123456")
        assert dt == UTC(2010, 2, 13, 2, 13, 11, 123456)
        dt = UTC("2010-2-13T02:9:9.123456")
        assert dt == UTC(2010, 2, 13, 2, 9, 9, 123456)

    def test_invalid_dates(self):
        """
        Tests invalid dates.
        """
        # Both should raise a value error that the day is too large for the
        # month.
        with pytest.raises(ValueError):
            UTC(2010, 9, 31)
        with pytest.raises(ValueError):
            UTC('2010-09-31')
        # invalid julday
        with pytest.raises(ValueError):
            UTC(year=2010, julday=999)
        # testing some strange patterns
        with pytest.raises(TypeError):
            UTC("ABC")
        with pytest.raises(TypeError):
            UTC("12X3T")
        with pytest.raises(ValueError):
            UTC(2010, 9, 31)

    def test_invalid_times(self):
        """
        Tests invalid times.
        """
        # wrong time information
        with pytest.raises(ValueError):
            UTC("2010-02-13T99999", iso8601=True)
        with pytest.raises(ValueError):
            UTC("2010-02-13 99999", iso8601=True)
        with pytest.raises(ValueError):
            UTC("2010-02-13T99999")
        with pytest.raises(TypeError):
            UTC("2010-02-13T02:09:09.XXXXX")

    def test_issue_168(self):
        """
        Couldn't calculate julday before 1900.
        """
        # 1
        dt = UTC("2010-01-01")
        assert dt.julday == 1
        # 2
        dt = UTC("1905-12-31")
        assert dt.julday == 365
        # 3
        dt = UTC("1906-12-31T23:59:59.999999Z")
        assert dt.julday == 365

    def test_format_seed(self):
        """
        Tests format_seed method
        """
        # 1
        dt = UTC("2010-01-01")
        assert dt.format_seed(compact=True) == "2010,001"
        # 2
        dt = UTC("2010-01-01T00:00:00.000000")
        assert dt.format_seed(compact=True) == "2010,001"
        # 3
        dt = UTC("2010-01-01T12:00:00")
        assert dt.format_seed(compact=True) == "2010,001,12"
        # 4
        dt = UTC("2010-01-01T12:34:00")
        assert dt.format_seed(compact=True) == "2010,001,12:34"
        # 5
        dt = UTC("2010-01-01T12:34:56")
        assert dt.format_seed(compact=True) == "2010,001,12:34:56"
        # 6
        dt = UTC("2010-01-01T12:34:56.123456")
        assert dt.format_seed(compact=True) == "2010,001,12:34:56.1234"
        # 7 - explicit disabling compact flag still results into compact date
        # if no time information is given
        dt = UTC("2010-01-01")
        assert dt.format_seed(compact=False) == "2010,001"

    def test_eq(self):
        """
        Tests __eq__ operators.
        """
        assert UTC(999) == UTC(999)
        assert not (UTC(1) == UTC(999))
        # w/ default precision of 6 digits
        assert UTC(999.000001) == UTC(999.000001)
        assert UTC(999.999999) == UTC(999.999999)
        assert not (UTC(999.0000001) == UTC(999.0000009))
        assert not (UTC(999.9999990) == UTC(999.9999999))
        assert UTC(999.00000001) == UTC(999.00000009)
        assert UTC(999.99999900) == UTC(999.99999909)
        # w/ precision of 7 digits
        assert UTC(999.00000001, precision=7) != \
               UTC(999.00000009, precision=7)
        assert UTC(999.99999990, precision=7) != \
               UTC(999.99999999, precision=7)
        assert UTC(999.000000001, precision=7) == \
               UTC(999.000000009, precision=7)
        assert UTC(999.999999900, precision=7) == \
               UTC(999.999999909, precision=7)

    def test_ne(self):
        """
        Tests __ne__ operators.
        """
        assert not (UTC(999) != UTC(999))
        assert UTC(1) != UTC(999)
        # w/ default precision of 6 digits
        assert not (UTC(999.000001) != UTC(999.000001))
        assert not (UTC(999.999999) != UTC(999.999999))
        assert UTC(999.0000001) != UTC(999.0000009)
        assert UTC(999.9999990) != UTC(999.9999999)
        assert not (UTC(999.00000001) != UTC(999.00000009))
        assert not (UTC(999.99999900) != UTC(999.99999909))
        # w/ precision of 7 digits
        assert UTC(999.00000001, precision=7) != \
               UTC(999.00000009, precision=7)
        assert UTC(999.99999990, precision=7) != \
               UTC(999.99999999, precision=7)
        assert not (UTC(999.000000001, precision=7) !=
                    UTC(999.000000009, precision=7))
        assert not (UTC(999.999999900, precision=7) !=
                    UTC(999.999999909, precision=7))

    def test_lt(self):
        """
        Tests __lt__ operators.
        """
        assert not (UTC(999) < UTC(999))
        assert UTC(1) < UTC(999)
        assert not (UTC(999) < UTC(1))
        # w/ default precision of 6 digits
        assert not (UTC(999.000001) < UTC(999.000001))
        assert not (UTC(999.999999) < UTC(999.999999))
        assert UTC(999.0000001) < UTC(999.0000009)
        assert not (UTC(999.0000009) < UTC(999.0000001))
        assert UTC(999.9999990) < UTC(999.9999999)
        assert not (UTC(999.9999999) < UTC(999.9999990))
        assert not (UTC(999.00000001) < UTC(999.00000009))
        assert not (UTC(999.00000009) < UTC(999.00000001))
        assert not (UTC(999.99999900) < UTC(999.99999909))
        assert not (UTC(999.99999909) < UTC(999.99999900))
        # w/ precision of 7 digits
        assert UTC(999.00000001, precision=7) < \
               UTC(999.00000009, precision=7)
        assert not (UTC(999.00000009, precision=7) <
                    UTC(999.00000001, precision=7))
        assert UTC(999.99999990, precision=7) < \
               UTC(999.99999999, precision=7)
        assert not (UTC(999.99999999, precision=7) <
                    UTC(999.99999990, precision=7))
        assert not (UTC(999.000000001, precision=7) <
                    UTC(999.000000009, precision=7))
        assert not (UTC(999.000000009, precision=7) <
                    UTC(999.000000001, precision=7))
        assert not (UTC(999.999999900, precision=7) <
                    UTC(999.999999909, precision=7))
        assert not (UTC(999.999999909, precision=7) <
                    UTC(999.999999900, precision=7))

    def test_le(self):
        """
        Tests __le__ operators.
        """
        assert UTC(999) <= UTC(999)
        assert UTC(1) <= UTC(999)
        assert not (UTC(999) <= UTC(1))
        # w/ default precision of 6 digits
        assert UTC(999.000001) <= UTC(999.000001)
        assert UTC(999.999999) <= UTC(999.999999)
        assert UTC(999.0000001) <= UTC(999.0000009)
        assert not (UTC(999.0000009) <= UTC(999.0000001))
        assert UTC(999.9999990) <= UTC(999.9999999)
        assert not (UTC(999.9999999) <= UTC(999.9999990))
        assert UTC(999.00000001) <= UTC(999.00000009)
        assert UTC(999.00000009) <= UTC(999.00000001)
        assert UTC(999.99999900) <= UTC(999.99999909)
        assert UTC(999.99999909) <= UTC(999.99999900)
        # w/ precision of 7 digits
        assert UTC(999.00000001, precision=7) <= \
               UTC(999.00000009, precision=7)
        assert not (UTC(999.00000009, precision=7) <=
                    UTC(999.00000001, precision=7))
        assert UTC(999.99999990, precision=7) <= \
               UTC(999.99999999, precision=7)
        assert not (UTC(999.99999999, precision=7) <=
                    UTC(999.99999990, precision=7))
        assert UTC(999.000000001, precision=7) <= \
               UTC(999.000000009, precision=7)
        assert UTC(999.000000009, precision=7) <= \
               UTC(999.000000001, precision=7)
        assert UTC(999.999999900, precision=7) <= \
               UTC(999.999999909, precision=7)
        assert UTC(999.999999909, precision=7) <= \
               UTC(999.999999900, precision=7)

    def test_gt(self):
        """
        Tests __gt__ operators.
        """
        assert not (UTC(999) > UTC(999))
        assert not (UTC(1) > UTC(999))
        assert UTC(999) > UTC(1)
        # w/ default precision of 6 digits
        assert not (UTC(999.000001) > UTC(999.000001))
        assert not (UTC(999.999999) > UTC(999.999999))
        assert not (UTC(999.0000001) > UTC(999.0000009))
        assert UTC(999.0000009) > UTC(999.0000001)
        assert not (UTC(999.9999990) > UTC(999.9999999))
        assert UTC(999.9999999) > UTC(999.9999990)
        assert not (UTC(999.00000001) > UTC(999.00000009))
        assert not (UTC(999.00000009) > UTC(999.00000001))
        assert not (UTC(999.99999900) > UTC(999.99999909))
        assert not (UTC(999.99999909) > UTC(999.99999900))
        # w/ precision of 7 digits
        assert not (UTC(999.00000001, precision=7) >
                    UTC(999.00000009, precision=7))
        assert UTC(999.00000009, precision=7) > \
               UTC(999.00000001, precision=7)
        assert not (UTC(999.99999990, precision=7) >
                    UTC(999.99999999, precision=7))
        assert UTC(999.99999999, precision=7) > UTC(999.99999990, precision=7)
        assert not (UTC(999.000000001, precision=7) >
                    UTC(999.000000009, precision=7))
        assert not (UTC(999.000000009, precision=7) >
                    UTC(999.000000001, precision=7))
        assert not (UTC(999.999999900, precision=7) >
                    UTC(999.999999909, precision=7))
        assert not (UTC(999.999999909, precision=7) >
                    UTC(999.999999900, precision=7))

    def test_ge(self):
        """
        Tests __ge__ operators.
        """
        assert UTC(999) >= UTC(999)
        assert not (UTC(1) >= UTC(999))
        assert UTC(999) >= UTC(1)
        # w/ default precision of 6 digits
        assert UTC(999.000001) >= UTC(999.000001)
        assert UTC(999.999999) >= UTC(999.999999)
        assert not (UTC(999.0000001) >= UTC(999.0000009))
        assert UTC(999.0000009) >= UTC(999.0000001)
        assert not (UTC(999.9999990) >= UTC(999.9999999))
        assert UTC(999.9999999) >= UTC(999.9999990)
        assert UTC(999.00000001) >= UTC(999.00000009)
        assert UTC(999.00000009) >= UTC(999.00000001)
        assert UTC(999.99999900) >= UTC(999.99999909)
        assert UTC(999.99999909) >= UTC(999.99999900)
        # w/ precision of 7 digits
        assert not (UTC(999.00000001, precision=7) >=
               UTC(999.00000009, precision=7))
        assert UTC(999.00000009, precision=7) >= \
               UTC(999.00000001, precision=7)
        assert not (UTC(999.99999990, precision=7) >=
               UTC(999.99999999, precision=7))
        assert UTC(999.99999999, precision=7) >= \
               UTC(999.99999990, precision=7)
        assert UTC(999.000000001, precision=7) >= \
               UTC(999.000000009, precision=7)
        assert UTC(999.000000009, precision=7) >= \
               UTC(999.000000001, precision=7)
        assert UTC(999.999999900, precision=7) >= \
               UTC(999.999999909, precision=7)
        assert UTC(999.999999909, precision=7) >= \
               UTC(999.999999900, precision=7)

    def test_toordinal(self):
        """
        Short test if toordinal() is working.
        Matplotlib's date2num() function depends on this which is used a lot in
        plotting.
        """
        dt = UTC("2012-03-04T11:05:09.123456Z")
        assert dt.toordinal() == 734566

    def test_weekday(self):
        """
        Tests weekday method.
        """
        dt = UTC(2008, 10, 1, 12, 30, 35, 45020)
        assert dt.weekday == 2
        assert dt._get_weekday() == 2

    def test_default_precision(self):
        """
        Tests setting of default precisions via monkey patching.
        """
        dt = UTC()
        # instance
        assert dt.precision == 6
        assert dt.DEFAULT_PRECISION == 6
        # class
        assert UTC.DEFAULT_PRECISION == 6
        dt = UTC()
        # set new default precision
        UTC.DEFAULT_PRECISION = 3
        dt2 = UTC()
        # first instance should be unchanged
        assert dt.precision == 6
        # but class attribute has changed
        assert dt.DEFAULT_PRECISION == 3
        # class
        assert UTC.DEFAULT_PRECISION == 3
        # second instance should use new precision
        assert dt2.DEFAULT_PRECISION == 3
        assert dt2.precision == 3
        # cleanup
        UTC.DEFAULT_PRECISION = 6
        # class
        assert UTC.DEFAULT_PRECISION == 6

    def test_to_string_precision(self):
        """
        Tests __str__ method while using a precision.
        """
        # precision 7
        dt = UTC(1980, 2, 3, 12, 23, 34, precision=7)
        assert str(dt) == '1980-02-03T12:23:34.0000000Z'
        dt = UTC(1980, 2, 3, 12, 23, 34, 500000, precision=7)
        assert str(dt) == '1980-02-03T12:23:34.5000000Z'
        dt = UTC(1980, 2, 3, 12, 23, 34.500000, precision=7)
        assert str(dt) == '1980-02-03T12:23:34.5000000Z'
        dt = UTC(1980, 2, 3, 12, 23, 34, 5, precision=7)
        assert str(dt) == '1980-02-03T12:23:34.0000050Z'
        dt = UTC(1980, 2, 3, precision=7)
        assert str(dt) == '1980-02-03T00:00:00.0000000Z'
        dt = UTC(1980, 2, 3, 12, 23, 34, 444999, precision=7)
        assert str(dt) == '1980-02-03T12:23:34.4449990Z'
        # precision 3
        dt = UTC(1980, 2, 3, 12, 23, 34, precision=3)
        assert str(dt) == '1980-02-03T12:23:34.000Z'
        dt = UTC(1980, 2, 3, 12, 23, 34, 500000, precision=3)
        assert str(dt) == '1980-02-03T12:23:34.500Z'
        dt = UTC(1980, 2, 3, 12, 23, 34.500000, precision=3)
        assert str(dt) == '1980-02-03T12:23:34.500Z'
        dt = UTC(1980, 2, 3, 12, 23, 34, 5, precision=3)
        assert str(dt) == '1980-02-03T12:23:34.000Z'
        dt = UTC(1980, 2, 3, precision=3)
        assert str(dt) == '1980-02-03T00:00:00.000Z'
        dt = UTC(1980, 2, 3, 12, 23, 34, 444999, precision=3)
        assert str(dt) == '1980-02-03T12:23:34.445Z'

    def test_precision_above_9_issues_warning(self):
        """
        Precisions above 9 should raise a warning as they cannot be
        represented internally as a int of nanoseconds.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            UTC(precision=10)
        assert len(w) == 1
        assert 'precision above 9' in str(w[-1].message)

    def test_rich_comparision_numeric_objects(self):
        """
        Tests basic rich comparison operations against numeric objects.
        """
        t1 = UTC(2005, 3, 4, 12, 33, 44)
        t2 = UTC(2005, 3, 4, 12, 33, 44, 123456)
        t1_int = 1109939624
        t2_int = 1109939624
        t1_float = 1109939624.0
        t2_float = 1109939624.123456
        # test (not) equal
        assert t1 == t1_int
        assert t1 == t1_float
        assert not (t2 == t2_int)
        assert t2 == t2_float
        assert not (t1 != t1_int)
        assert not (t1 != t1_float)
        assert t2 != t2_int
        assert not (t2 != t2_float)
        # test less/greater(equal)
        assert t1 >= t1_int
        assert t1 <= t1_int
        assert not (t1 > t1_int)
        assert not (t1 < t1_int)
        assert t1 >= t1_float
        assert t1 <= t1_float
        assert not (t1 > t1_float)
        assert not (t1 < t1_float)
        assert t2 >= t2_int
        assert not (t2 <= t2_int)
        assert t2 > t2_int
        assert not (t2 < t2_int)
        assert t2 >= t2_float
        assert t2 <= t2_float
        assert not (t2 > t2_float)
        assert not (t2 < t2_float)

    def test_rich_comparision_numeric_types(self):
        """
        Tests basic rich comparison operations against non-numeric objects.
        """
        dt = UTC()
        for obj in [None, 'string', object()]:
            assert dt != obj
            assert not (dt <= obj)
            assert not (dt < obj)
            assert not (dt >= obj)
            assert not (dt > obj)
            assert obj != dt
            assert not (obj <= dt)
            assert not (obj < dt)
            assert not (obj >= dt)
            assert not (obj > dt)

    def test_rich_comparision_fuzzy(self):
        """
        UTC fuzzy comparisons break sorting, max, min - see #1765
        """
        # 1 - precision set to 6 - 3
        for precision in [6, 5, 4, 3]:
            dt1 = UTC(0.001, precision=precision)
            dt2 = UTC(0.004, precision=precision)
            dt3 = UTC(0.007, precision=precision)
            sorted_times = [dt1, dt2, dt3]
            # comparison
            for utc1, utc2 in itertools.combinations(sorted_times, 2):
                assert utc1 != utc2
                assert utc2 != utc1
                assert utc1 < utc2
                assert utc1 <= utc2
                assert utc2 > utc1
                assert utc2 >= utc1
            # sorting
            for unsorted_times in itertools.permutations(sorted_times):
                assert sorted(unsorted_times) == sorted_times
                # min, max
                assert max(unsorted_times) == dt3
                assert min(unsorted_times) == dt1

        # 2 - precision set to 2
        dt1 = UTC(0.001, precision=2)  # == 0.00
        dt2 = UTC(0.004, precision=2)  # == 0.00
        dt3 = UTC(0.007, precision=2)  # == 0.01
        # comparison
        assert dt1 == dt2
        assert dt2 != dt3
        assert dt1 != dt3
        assert dt1 >= dt2
        assert dt2 < dt3
        assert dt1 < dt3
        assert dt1 <= dt2
        assert dt2 <= dt3
        assert dt1 <= dt3
        assert dt1 <= dt2
        assert dt2 <= dt3
        assert dt1 <= dt3
        assert dt1 >= dt2
        assert dt2 < dt3
        assert dt1 < dt3
        # sorting
        times = [dt3, dt2, dt1]
        sorted_times = sorted(times)
        assert sorted_times[0] <= sorted_times[2]
        assert sorted_times[0] < sorted_times[2]
        assert sorted_times[0] != sorted_times[2]
        assert sorted_times[0] <= sorted_times[2]
        assert sorted_times[0] < sorted_times[2]
        assert sorted_times == [dt2, dt1, dt3]  # expected
        assert sorted_times == [dt2, dt1, dt3]  # due to precision
        # check correct sort order
        assert sorted_times[0]._ns == dt2._ns
        assert sorted_times[2]._ns == dt3._ns
        # min, max
        max_times = max(dt2, dt1, dt3)
        assert max_times == dt3
        max_times = max(dt1, dt2, dt3)
        assert max_times == dt3
        # min, max lists
        times = [dt2, dt1, dt3]
        assert max(times) == dt3
        assert min(times) == dt2  # expected
        assert min(times) == dt1  # due to precision
        times = [dt1, dt2, dt3]
        assert max(times) == dt3
        assert min(times) == dt1  # expected
        assert min(times) == dt2  # due to precision

    def test_datetime_with_timezone(self):
        """
        UTC from timezone-aware datetime.datetime

        .. seealso:: https://github.com/obspy/obspy/issues/553
        """
        class ManilaTime(datetime.tzinfo):

            def utcoffset(self, dt):  # @UnusedVariable
                return datetime.timedelta(hours=8)

        dt = datetime.datetime(2006, 11, 21, 16, 30, tzinfo=ManilaTime())
        assert dt.isoformat() == '2006-11-21T16:30:00+08:00'
        assert UTC(dt.isoformat()) == UTC(dt)

    def test_hash(self):
        """
        Test __hash__ method of UTC class.
        """
        assert UTC().__hash__() is None

    def test_now(self):
        """
        Test now class method of UTC class.
        """
        dt = UTC()
        assert UTC.now() >= dt

    def test_utcnow(self):
        """
        Test utcnow class method of UTCDateTime class.
        """
        dt = UTC()
        assert UTC.utcnow() >= dt

    def test_abs(self):
        """
        Test __abs__ method of UTCDateTime class.
        """
        dt = UTC(1970, 1, 1, 0, 0, 1)
        assert abs(dt) == 1
        dt = UTC(1970, 1, 1, 0, 0, 1, 500000)
        assert abs(dt) == 1.5
        dt = UTC(1970, 1, 1)
        assert abs(dt) == 0
        dt = UTC(1969, 12, 31, 23, 59, 59)
        assert abs(dt) == 1
        dt = UTC(1969, 12, 31, 23, 59, 59, 500000)
        assert abs(dt) == 0.5

    def test_string_with_timezone(self):
        """
        Test that all valid ISO time zone specifications are parsed properly
        https://en.wikipedia.org/wiki/ISO_8601#Time_offsets_from_UTC
        """
        # positive
        t = UTC("2013-09-01T12:34:56Z")
        time_strings = \
            ["2013-09-01T14:34:56+02", "2013-09-01T14:34:56+02:00",
             "2013-09-01T14:34:56+0200", "2013-09-01T14:49:56+02:15",
             "2013-09-01T12:34:56+00:00", "2013-09-01T12:34:56+00",
             "2013-09-01T12:34:56+0000"]
        for time_string in time_strings:
            assert t == UTC(time_string)

        # negative
        t = UTC("2013-09-01T12:34:56Z")
        time_strings = \
            ["2013-09-01T10:34:56-02", "2013-09-01T10:34:56-02:00",
             "2013-09-01T10:34:56-0200", "2013-09-01T10:19:56-02:15",
             "2013-09-01T12:34:56-00:00", "2013-09-01T12:34:56-00",
             "2013-09-01T12:34:56-0000"]
        for time_string in time_strings:
            assert t == UTC(time_string)

    def test_year_2038_problem(self):
        """
        See issue #805
        """
        dt = UTC(2004, 1, 10, 13, 37, 4)
        assert dt.__str__() == '2004-01-10T13:37:04.000000Z'
        dt = UTC(2038, 1, 19, 3, 14, 8)
        assert dt.__str__() == '2038-01-19T03:14:08.000000Z'
        dt = UTC(2106, 2, 7, 6, 28, 16)
        assert dt.__str__() == '2106-02-07T06:28:16.000000Z'

    def test_format_iris_webservice(self):
        """
        Tests the format IRIS webservice function.

        See issue #1096.
        """
        # These are parse slightly differently (1 microsecond difference but
        # the IRIS webservice string should be identical as its only
        # accurate to three digits.
        d1 = UTC(2011, 1, 25, 15, 32, 12.26)
        d2 = UTC("2011-01-25T15:32:12.26")

        assert d1.format_iris_web_service() == d2.format_iris_web_service()

    def test_floating_point_second_initialization(self):
        """
        Tests floating point precision issues in initialization of UTCDateTime
        objects with floating point seconds.

        See issue #1096.
        """
        for microns in range(0, 5999):
            t = UTC(2011, 1, 25, 15, 32, 12 + microns / 1e6)
            assert microns == t.microsecond

    def test_issue_1215(self):
        """
        Tests some non-ISO8601 strings which should be also properly parsed.

        See issue #1215.
        """
        assert UTC('2015-07-03-06') == UTC(2015, 7, 3, 6, 0, 0)
        assert UTC('2015-07-03-06-42') == UTC(2015, 7, 3, 6, 42, 0)
        assert UTC('2015-07-03-06-42-1') == UTC(2015, 7, 3, 6, 42, 1)
        utc1 = UTC('2015-07-03-06-42-1.5123')
        assert utc1 == UTC(2015, 7, 3, 6, 42, 1, 512300)

    def test_matplotlib_date(self):
        """
        Test convenience method and property for conversion to matplotlib
        datetime float numbers.
        """
        from matplotlib import __version__

        for t_, expected_old, expected in zip(
                ("1986-05-02T13:44:12.567890Z", "2009-08-24T00:20:07.700000Z",
                 "2026-11-27T03:12:45.4"),
                (725128.5723676839, 733643.0139780092, 739947.1338587963),
                (5965.57236768, 14480.013978, 20784.1338588),
                ):
            t = UTC(t_)
            if parse_version(__version__) < parse_version('3.3'):
                expected = expected_old
            np.testing.assert_almost_equal(
                t.matplotlib_date, expected, decimal=8)

    def test_add_error_message(self):
        t = UTC()
        t2 = UTC()
        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            t + t2

    def test_nanoseconds(self):
        """
        Various nanosecond tests.

        Also tests #1318.
        """
        # 1
        dt = UTC(1, 1, 1, 0, 0, 0, 0)
        assert dt._ns == -62135596800000000000
        assert dt.timestamp == -62135596800.0
        assert dt.microsecond == 0
        assert dt.datetime == datetime.datetime(1, 1, 1, 0, 0, 0, 0)
        assert str(dt) == '0001-01-01T00:00:00.000000Z'
        # 2
        dt = UTC(1, 1, 1, 0, 0, 0, 1)
        assert dt._ns == -62135596799999999000
        assert dt.microsecond == 1
        assert dt.datetime == datetime.datetime(1, 1, 1, 0, 0, 0, 1)
        assert str(dt) == '0001-01-01T00:00:00.000001Z'
        assert dt.timestamp == -62135596800.000001
        # 3
        dt = UTC(1, 1, 1, 0, 0, 0, 999999)
        assert dt._ns == -62135596799000001000
        # dt.timestamp should be -62135596799.000001 - not possible to display
        # correctly using python floats
        assert dt.timestamp == -62135596799.0
        assert dt.microsecond == 999999
        assert dt.datetime == datetime.datetime(1, 1, 1, 0, 0, 0, 999999)
        assert str(dt) == '0001-01-01T00:00:00.999999Z'
        # 4
        dt = UTC(1, 1, 1, 0, 0, 1, 0)
        assert dt._ns == -62135596799000000000
        assert dt.timestamp == -62135596799.0
        assert dt.microsecond == 0
        assert dt.datetime == datetime.datetime(1, 1, 1, 0, 0, 1, 0)
        assert str(dt) == '0001-01-01T00:00:01.000000Z'
        # 5
        dt = UTC(1, 1, 1, 0, 0, 1, 1)
        assert dt._ns == -62135596798999999000
        # dt.timestamp should be -62135596799.000001 - not possible to display
        # correctly using python floats
        assert dt.timestamp == -62135596799.0
        assert dt.microsecond == 1
        assert dt.datetime == datetime.datetime(1, 1, 1, 0, 0, 1, 1)
        assert str(dt) == '0001-01-01T00:00:01.000001Z'
        # 6
        dt = UTC(1970, 1, 1, 0, 0, 0, 1)
        assert dt._ns == 1000
        assert dt.timestamp == 0.000001
        assert dt.microsecond == 1
        assert dt.datetime == datetime.datetime(1970, 1, 1, 0, 0, 0, 1)
        assert str(dt) == '1970-01-01T00:00:00.000001Z'
        # 7
        dt = UTC(1970, 1, 1, 0, 0, 0, 999999)
        assert dt._ns == 999999000
        assert dt.timestamp == 0.999999
        assert dt.microsecond == 999999
        assert dt.datetime == datetime.datetime(1970, 1, 1, 0, 0, 0, 999999)
        assert str(dt) == '1970-01-01T00:00:00.999999Z'
        # 8
        dt = UTC(3000, 1, 1, 0, 0, 0, 500000)
        assert dt._ns == 32503680000500000000
        assert dt.timestamp == 32503680000.5
        assert dt.microsecond == 500000
        assert dt.datetime == datetime.datetime(3000, 1, 1, 0, 0, 0, 500000)
        assert str(dt) == '3000-01-01T00:00:00.500000Z'
        # 9
        dt = UTC(9999, 1, 1, 0, 0, 0, 500000)
        assert dt._ns == 253370764800500000000
        assert dt.timestamp == 253370764800.5
        assert dt.microsecond == 500000
        assert dt.datetime == datetime.datetime(9999, 1, 1, 0, 0, 0, 500000)
        assert str(dt) == '9999-01-01T00:00:00.500000Z'

    def test_utcdatetime_from_utcdatetime(self):
        a = UTC(1, 1, 1, 1, 1, 1, 999999)
        assert UTC(a)._ns == a._ns
        assert str(UTC(a)) == str(a)

    def test_issue_1008(self):
        """
        see #1008
        """
        assert str(UTC("9999-12-31T23:59:59.9999")) == \
               "9999-12-31T23:59:59.999900Z"
        assert str(UTC("9999-12-31T23:59:59.999999")) == \
               "9999-12-31T23:59:59.999999Z"

    def test_issue_1652(self):
        """
        Comparing UTCDateTime and datetime.datetime objects - see #1652
        """
        a = datetime.datetime(1990, 1, 1, 0, 0)
        e = UTC(2000, 1, 2, 1, 39, 37)
        assert a < e
        assert not (a > e)
        assert a <= e
        assert not (e <= a)
        assert not (a > e)
        assert e > a
        assert not (a >= e)
        assert e >= a
        assert not (a == e)
        assert not (e == a)

    def test_issue_2165(self):
        """
        When a timestamp gets rounded it should increment seconds and not
        result in 1_000_000 microsecond value. See #2072.
        """
        time = UTC(1.466387732999999762e+09)
        # test microseconds are rounded
        assert time.microsecond == 0
        # test __repr__
        expected_repr = "UTCDateTime(2016, 6, 20, 1, 55, 33)"
        assert time.__repr__() == expected_repr
        # test __str__
        expected_str = "2016-06-20T01:55:33.000000Z"
        assert str(time) == expected_str

    def test_ns_public_attribute(self):
        """
        Basic test for public ns interface to UTCDateTime
        """
        t = UTC('2018-01-17T12:34:56.789012Z')
        # test getter
        assert t.ns == 1516192496789012000
        # test init with ns (set attr is depreciated)
        x = 1516426162899012123
        t = UTC(ns=x)
        assert t.ns == x
        assert t.day == 20
        assert t.microsecond == 899012

    def test_timestamp_can_serialize_with_time_attrs(self):
        """
        Test that the datetime attrs can be used to serialize UTCDateTime
        objects inited from floats (given default precision of 6) - see #2034
        """
        time_attrs = ('year', 'month', 'day', 'hour', 'minute', 'second',
                      'microsecond')
        close_timestamps = [1515174511.1984465, 1515174511.1984463,
                            1515174511.1984460, 1515174511.1984458]
        close_utc = [UTC(x) for x in close_timestamps]

        for utc in close_utc:
            utc2 = UTC(**{x: getattr(utc, x) for x in time_attrs})
            assert utc == utc2

    def test_str_ms_equal_ms(self):
        """
        Test that the microseconds in the str representation are equal to
        the microseconds attr - see #2034
        """
        close_timestamps = [1515174511.1984465, 1515174511.1984463,
                            1515174511.1984460, 1515174511.1984458]
        close_utc = [UTC(x) for x in close_timestamps]

        for utc in close_utc:
            str_ms = int(str(utc).split('.')[-1][:-1])  # get ms from str rep
            ms = utc.microsecond
            assert str_ms == ms

    def test_close_utc_are_equal(self):
        """
        Ensure UTCs init'ed with floats that are very close together are
        equal - see 2034

        Note: Due to the rounding nanosecond attribute before comparision
        we can no longer guarantee equality based on the difference in
        nanoseconds. This trade-off was made to ensure UTCDateTime objects
        are always equal to their string representation when precision <= 6.
        See issue #2034.
        """
        # get an array of floats as close together as possible
        def yield_close_floats(start, length):
            for _ in range(length):
                start = np.nextafter(start, 0)
                yield start

        # convert to UTC objects
        float0 = 1515174511.1984458
        for precision in range(1, 10):
            close_timestamps = list(yield_close_floats(float0, 10))
            close_utc = [UTC(x, precision=precision)
                         for x in close_timestamps]

            # if str are equal then objects should be equal and visa versa
            for num in range(len(close_utc) - 1):
                utc1 = close_utc[num]
                utc2 = close_utc[num + 1]
                if utc1 == utc2:
                    assert str(utc1) == str(utc2)
                if str(utc1) == str(utc2):
                    assert utc1 == utc2

    def test_comparing_different_precision_utcs_warns(self):
        """
        Comparing UTCDateTime instances with different precisions should
        raise a warning.
        """
        utc1 = UTC(precision=9)
        utc2 = UTC(precision=6)
        for operator in [ge, eq, lt, le, gt, ne]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                operator(utc1, utc2)
            assert len(w) == 1
            assert 'different precision' in str(w[-1].message)

    def test_string_representation_various_precisions(self):
        """
        Ensure string representation works for many different precisions
        """
        precisions = range(-9, 9)
        for precision in precisions:
            utc = UTC(0.0, precision=precision)
            utc_str = str(utc)
            assert UTC(utc_str, precision=precision) == utc
            assert isinstance(utc_str, str)

    def test_zero_precision_doesnt_print_dot(self):
        """
        UTC with precision of 0 should not print a decimal in str rep.
        """
        utc = UTC(precision=0)
        utc_str = str(utc)
        assert '.' not in utc_str

    def test_change_time_attr_raises_warning(self):
        """
        Changing the time representation on the UTCDateTime instances should
        raise a depreciation warning as a path towards immutability
        (see #2072).
        """
        utc = UTC()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            utc.hour = 2
        assert len(w) == 1
        warn = w[0]
        assert 'will raise an Exception' in str(warn.message)
        assert isinstance(warn.message, ObsPyDeprecationWarning)

    def test_change_precision_raises_warning(self):
        """
        Changing the precision on the UTCDateTime instances should raise a
        depreciation warning as a path towards immutability (see #2072).
        """
        utc = UTC()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            utc.precision = 2
        assert len(w) == 1
        warn = w[0]
        assert 'will raise an Exception' in str(warn.message)
        assert isinstance(warn.message, ObsPyDeprecationWarning)

    def test_compare_utc_different_precision_raises_warning(self):
        """
        Comparing UTCDateTime objects of different precisions should raise a
        depreciation warning (see #2072)
        """
        utc1 = UTC(0, precision=2)
        utc2 = UTC(0, precision=3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            utc_equals = utc1 == utc2
        assert utc_equals
        assert len(w) == 1
        warn = w[0]
        assert 'will raise an Exception' in str(warn.message)
        assert isinstance(warn.message, ObsPyDeprecationWarning)

    def test_replace(self):
        """
        Tests for the replace method of UTCDateTime
        """
        test_dict = dict(
            year=2017,
            month=9,
            day=18,
            hour=18,
            minute=30,
            second=11,
            microsecond=122255,
        )

        utc = UTC(**test_dict)

        # iterate over each settable parameter and change
        for attr in test_dict:
            new_value = test_dict[attr] + 1
            utc2 = utc.replace(**{attr: new_value})
            assert isinstance(utc2, UTC)
            # make sure only the settable parameter changed in utc2
            for time_attribute in test_dict:
                default = getattr(utc, time_attribute)
                current = getattr(utc2, time_attribute)
                if time_attribute == attr:
                    assert current == default + 1
                else:
                    assert current == default

        # test julian day
        utc2 = utc.replace(julday=utc.julday + 1)
        assert utc2.julday == utc.julday + 1

    def test_replace_with_julday_and_month_raises(self):
        """
        The replace method cannot use julday with either day or month.
        """
        utc = UTC(0)
        with pytest.raises(ValueError):
            utc.replace(julday=100, day=2)
        with pytest.raises(ValueError):
            utc.replace(julday=100, month=2)
        with pytest.raises(ValueError):
            utc.replace(julday=100, day=2, month=2)

    def test_unsupported_replace_argument_raises(self):
        """
        The replace method should raise a value error if any unsupported
        arguments are passed to it.
        """
        utc = UTC(0)
        with pytest.raises(ValueError, match='zweite'):
            utc.replace(zweite=22)

    def test_hour_minute_second_overflow(self):
        """
        Tests for allowing hour, minute, and second to exceed usual limits.
        This only applies when using dates as kwargs to the UTCDateTime
        constructor. See #2222.
        """
        # Create a UTC constructor with default values using partial
        kwargs = dict(year=2017, month=9, day=18, hour=0, minute=0, second=0)
        base_utc = partial(UTC, **kwargs)
        # ensure hour can exceed 23 and is equal to the day ticking forward
        utc = base_utc(hour=25, strict=False)
        assert utc == base_utc(day=19, hour=1)
        # ensure minute can exceed 60
        utc = base_utc(minute=61, strict=False)
        assert utc == base_utc(hour=1, minute=1)
        # ensure second can exceed 60
        utc = base_utc(second=120, strict=False)
        assert utc == base_utc(minute=2)
        # ensure microsecond can exceed 1_000_000
        utc = base_utc(microsecond=10000000, strict=False)
        assert utc == base_utc(second=10)
        # ensure not all kwargs are required for overflow behavior
        utc = UTC(year=2017, month=9, day=18, second=60, strict=False)
        assert utc == base_utc(minute=1)
        # test for combination of args and kwargs
        utc1 = UTC(2017, 5, 4, second=120, strict=False)
        utc2 = UTC(2017, 5, 4, minute=2)
        assert utc1 == utc2
        # if strict == True a ValueError should be raised
        with pytest.raises(ValueError, match='hour must be in'):
            base_utc(hour=60)

    def test_hour_minute_second_overflow_with_replace(self):
        """
        The replace method should also support the changes described in #2222.
        """
        utc = UTC('2017-09-18T00:00:00')
        assert utc.replace(hour=25, strict=False) == utc + 25 * 3600
        assert utc.replace(minute=1000, strict=False) == utc + 60000
        assert utc.replace(second=60, strict=False) == utc + 60

    def test_strftime_with_years_less_than_1900(self):
        """
        Try that some strftime commands we use (e.g. in plotting) work even
        with years less than 1900 (underlying datetime.datetime.strftime raises
        ValueError if year <1900.
        """
        t = UTC(1888, 1, 2, 1, 39, 37)
        assert t.strftime('%Y-%m-%d') == '1888-01-02'
        t = UTC(998, 11, 9, 1, 39, 37)
        assert '0998-11-09' == t.strftime('%Y-%m-%d')

    def test_string_parsing_at_instantiating_before_1000(self):
        """
        Try instantiating the UTCDateTime object with strings containing years
        before 1000.
        """
        for value in ["998-01-01", "98-01-01", "9-01-01"]:
            msg = "'%s' does not start with a 4 digit year" % value
            with pytest.raises(ValueError, match=msg):
                UTC(value)

    def test_leap_years(self):
        """
        Test for issue #2369, correct implementation of juldays for leap years.

        Test one leap year (2016; valid juldays 365, 366; invalid julday 367)
        and one regular year (2018; valid juldays 364, 365; invalid julday 366)
        """
        # these should fail
        with pytest.raises(ValueError):
            UTC(year=2018, julday=366)
        with pytest.raises(ValueError):
            UTC(year=2016, julday=367)

        # these should work and check we got the expected output
        got = UTC(year=2018, julday=364)
        expected = UTC(2018, 12, 30)
        assert got == expected

        got = UTC(year=2018, julday=365)
        expected = UTC(2018, 12, 31)
        assert got == expected

        got = UTC(year=2016, julday=365)
        expected = UTC(2016, 12, 30)
        assert got == expected

        got = UTC(year=2016, julday=366)
        expected = UTC(2016, 12, 31)
        assert got == expected

    def test_issue_2447(self):
        """
        Setting iso8601=False should disable ISO8601 parsing.

        See issue #2447.
        """
        # auto detection
        assert UTC('2019-01-01T02-02:33') == \
               UTC(2019, 1, 1, 4, 33, 0)
        assert UTC('2019-01-01 02-02:33') == \
               UTC(2019, 1, 1, 2, 2, 33)
        # enforce ISO8601 mode
        assert UTC('2019-01-01T02-02:33', iso8601=True) == \
               UTC(2019, 1, 1, 4, 33, 0)
        # skip ISO8601 mode
        assert UTC('2019-01-01T02-02:33', iso8601=False) == \
               UTC(2019, 1, 1, 2, 2, 33)
