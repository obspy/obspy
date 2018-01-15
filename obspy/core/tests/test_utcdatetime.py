# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import copy
import datetime
import unittest

import numpy as np

from obspy import UTCDateTime


class UTCDateTimeTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.utcdatetime.UTCDateTime.
    """

    def test_from_string(self):
        """
        Tests initialization from a given time string not ISO8601 compatible.
        """
        # some supported non ISO8601 patterns
        dt = UTCDateTime("1970-01-01 12:23:34")
        self.assertEqual(dt, UTCDateTime(1970, 1, 1, 12, 23, 34))
        dt = UTCDateTime("1970,01,01,12:23:34")
        self.assertEqual(dt, UTCDateTime(1970, 1, 1, 12, 23, 34))
        dt = UTCDateTime("1970,001,12:23:34")
        self.assertEqual(dt, UTCDateTime(1970, 1, 1, 12, 23, 34))
        dt = UTCDateTime("20090701121212")
        self.assertEqual(dt, UTCDateTime(2009, 7, 1, 12, 12, 12))
        dt = UTCDateTime("19700101")
        self.assertEqual(dt, UTCDateTime(1970, 1, 1, 0, 0))
        dt = UTCDateTime("1970/01/17 12:23:34")
        self.assertEqual(dt, UTCDateTime(1970, 1, 17, 12, 23, 34))
        # other non ISO8601 strings should raise an exception
        self.assertRaises(Exception, UTCDateTime, "1970,001,12:23:34",
                          iso8601=True)

    def test_from_numpy_string(self):
        """
        Tests importing from NumPy strings.
        """
        # some strange patterns
        dt = UTCDateTime(np.string_("1970-01-01 12:23:34"))
        self.assertEqual(dt, UTCDateTime(1970, 1, 1, 12, 23, 34))
        dt = UTCDateTime(np.string_("1970,01,01,12:23:34"))
        self.assertEqual(dt, UTCDateTime(1970, 1, 1, 12, 23, 34))
        dt = UTCDateTime(np.string_("1970,001,12:23:34"))
        self.assertEqual(dt, UTCDateTime(1970, 1, 1, 12, 23, 34))
        dt = UTCDateTime(np.string_("20090701121212"))
        self.assertEqual(dt, UTCDateTime(2009, 7, 1, 12, 12, 12))
        dt = UTCDateTime(np.string_("19700101"))
        self.assertEqual(dt, UTCDateTime(1970, 1, 1, 0, 0))
        # non ISO8601 strings should raise an exception
        self.assertRaises(Exception, UTCDateTime,
                          np.string_("1970,001,12:23:34"), iso8601=True)

    def test_from_python_date_time(self):
        """
        Tests initialization from a given time string not ISO8601 compatible.
        """
        dt = UTCDateTime(datetime.datetime(1970, 1, 1, 12, 23, 34, 123456))
        self.assertEqual(dt, UTCDateTime(1970, 1, 1, 12, 23, 34, 123456))
        dt = UTCDateTime(datetime.datetime(1970, 1, 1, 12, 23, 34))
        self.assertEqual(dt, UTCDateTime(1970, 1, 1, 12, 23, 34))
        dt = UTCDateTime(datetime.datetime(1970, 1, 1))
        self.assertEqual(dt, UTCDateTime(1970, 1, 1))
        dt = UTCDateTime(datetime.date(1970, 1, 1))
        self.assertEqual(dt, UTCDateTime(1970, 1, 1))

    def test_from_numeric(self):
        """
        Tests initialization from a given a numeric value.
        """
        dt = UTCDateTime(0.0)
        self.assertEqual(dt, UTCDateTime(1970, 1, 1, 0, 0, 0))
        dt = UTCDateTime(1240561632.005)
        self.assertEqual(dt, UTCDateTime(2009, 4, 24, 8, 27, 12, 5000))
        dt = UTCDateTime(1240561632)
        self.assertEqual(dt, UTCDateTime(2009, 4, 24, 8, 27, 12))

    def test_from_iso8601_calendar_date_string(self):
        """
        Tests initialization from a given ISO8601 calendar date representation.
        """
        # w/o trailing Z
        dt = UTCDateTime("2009-12-31T12:23:34.5")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 500000))
        dt = UTCDateTime("2009-12-31T12:23:34.500000")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 500000))
        dt = UTCDateTime("2009-12-31T12:23:34.000005")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 5))
        dt = UTCDateTime("2009-12-31T12:23:34")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34))
        dt = UTCDateTime("2009-12-31T12:23")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23))
        dt = UTCDateTime("2009-12-31T12")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12))
        # enforce ISO8601 - no chance to detect that format
        dt = UTCDateTime("2009-12-31", iso8601=True)
        self.assertEqual(dt, UTCDateTime(2009, 12, 31))
        # compact
        dt = UTCDateTime("20091231T122334.5")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 500000))
        dt = UTCDateTime("20091231T122334.500000")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 500000))
        dt = UTCDateTime("20091231T122334.000005")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 5))
        dt = UTCDateTime("20091231T122334")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34))
        dt = UTCDateTime("20091231T1223")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23))
        dt = UTCDateTime("20091231T12")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12))
        # enforce ISO8601 - no chance to detect that format
        dt = UTCDateTime("20091231", iso8601=True)
        self.assertEqual(dt, UTCDateTime(2009, 12, 31))
        # w/ trailing Z
        dt = UTCDateTime("2009-12-31T12:23:34.5Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 500000))
        dt = UTCDateTime("2009-12-31T12:23:34.500000Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 500000))
        dt = UTCDateTime("2009-12-31T12:23:34.000005Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 5))
        dt = UTCDateTime("2009-12-31T12:23:34Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34))
        dt = UTCDateTime("2009-12-31T12:23Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23))
        dt = UTCDateTime("2009-12-31T12Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12))
        # compact
        dt = UTCDateTime("20091231T122334.5Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 500000))
        dt = UTCDateTime("20091231T122334.500000Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 500000))
        dt = UTCDateTime("20091231T122334.000005Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 5))
        dt = UTCDateTime("20091231T122334Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34))
        dt = UTCDateTime("20091231T1223Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23))
        dt = UTCDateTime("20091231T12Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12))
        # time zones
        dt = UTCDateTime("2009-12-31T12:23:34-01:15")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 13, 38, 34))
        dt = UTCDateTime("2009-12-31T12:23:34.5-01:15")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 13, 38, 34, 500000))
        dt = UTCDateTime("2009-12-31T12:23:34.000005-01:15")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 13, 38, 34, 5))
        dt = UTCDateTime("2009-12-31T12:23:34+01:15")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 11, 8, 34))
        dt = UTCDateTime("2009-12-31T12:23:34.5+01:15")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 11, 8, 34, 500000))
        dt = UTCDateTime("2009-12-31T12:23:34.000005+01:15")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 11, 8, 34, 5))

    def test_from_iso8601_ordinal_date_string(self):
        """
        Tests initialization from a given ISO8601 ordinal date representation.
        """
        # w/o trailing Z
        dt = UTCDateTime("2009-365T12:23:34.5")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 500000))
        dt = UTCDateTime("2009-001T12:23:34")
        self.assertEqual(dt, UTCDateTime(2009, 1, 1, 12, 23, 34))
        dt = UTCDateTime("2009-001T12:23")
        self.assertEqual(dt, UTCDateTime(2009, 1, 1, 12, 23))
        dt = UTCDateTime("2009-001T12")
        self.assertEqual(dt, UTCDateTime(2009, 1, 1, 12))
        dt = UTCDateTime("2009-355")
        self.assertEqual(dt, UTCDateTime(2009, 12, 21))
        # enforce ISO8601 - no chance to detect that format
        dt = UTCDateTime("2009-001", iso8601=True)
        self.assertEqual(dt, UTCDateTime(2009, 1, 1))
        # compact
        dt = UTCDateTime("2009365T122334.5")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 500000))
        dt = UTCDateTime("2009001T122334")
        self.assertEqual(dt, UTCDateTime(2009, 1, 1, 12, 23, 34))
        dt = UTCDateTime("2009001T1223")
        self.assertEqual(dt, UTCDateTime(2009, 1, 1, 12, 23))
        dt = UTCDateTime("2009001T12")
        self.assertEqual(dt, UTCDateTime(2009, 1, 1, 12))
        dt = UTCDateTime("2009355")
        self.assertEqual(dt, UTCDateTime(2009, 12, 21))
        # enforce ISO8601 - no chance to detect that format
        dt = UTCDateTime("2009001", iso8601=True)
        self.assertEqual(dt, UTCDateTime(2009, 1, 1))
        # w/ trailing Z
        dt = UTCDateTime("2009-365T12:23:34.5Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 500000))
        dt = UTCDateTime("2009-001T12:23:34Z")
        self.assertEqual(dt, UTCDateTime(2009, 1, 1, 12, 23, 34))
        dt = UTCDateTime("2009-001T12:23Z")
        self.assertEqual(dt, UTCDateTime(2009, 1, 1, 12, 23))
        dt = UTCDateTime("2009-001T12Z")
        self.assertEqual(dt, UTCDateTime(2009, 1, 1, 12))
        # compact
        dt = UTCDateTime("2009365T122334.5Z")
        self.assertEqual(dt, UTCDateTime(2009, 12, 31, 12, 23, 34, 500000))
        dt = UTCDateTime("2009001T122334Z")
        self.assertEqual(dt, UTCDateTime(2009, 1, 1, 12, 23, 34))
        dt = UTCDateTime("2009001T1223Z")
        self.assertEqual(dt, UTCDateTime(2009, 1, 1, 12, 23))
        dt = UTCDateTime("2009001T12Z")
        self.assertEqual(dt, UTCDateTime(2009, 1, 1, 12))

    def test_from_iso8601_week_date_string(self):
        """
        Tests initialization from a given ISO8601 week date representation.
        """
        # w/o trailing Z
        dt = UTCDateTime("2009-W53-7T12:23:34.5")
        self.assertEqual(dt, UTCDateTime(2010, 1, 3, 12, 23, 34, 500000))
        dt = UTCDateTime("2009-W01-1T12:23:34")
        self.assertEqual(dt, UTCDateTime(2008, 12, 29, 12, 23, 34))
        dt = UTCDateTime("2009-W01-1T12:23")
        self.assertEqual(dt, UTCDateTime(2008, 12, 29, 12, 23))
        dt = UTCDateTime("2009-W01-1T12")
        self.assertEqual(dt, UTCDateTime(2008, 12, 29, 12))
        # enforce ISO8601 - no chance to detect that format
        dt = UTCDateTime("2009-W01-1", iso8601=True)
        self.assertEqual(dt, UTCDateTime(2008, 12, 29))
        # compact
        dt = UTCDateTime("2009W537T122334.5")
        self.assertEqual(dt, UTCDateTime(2010, 1, 3, 12, 23, 34, 500000))
        dt = UTCDateTime("2009W011T122334")
        self.assertEqual(dt, UTCDateTime(2008, 12, 29, 12, 23, 34))
        dt = UTCDateTime("2009W011T1223")
        self.assertEqual(dt, UTCDateTime(2008, 12, 29, 12, 23))
        dt = UTCDateTime("2009W011T12")
        self.assertEqual(dt, UTCDateTime(2008, 12, 29, 12))
        # enforce ISO8601 - no chance to detect that format
        dt = UTCDateTime("2009W011", iso8601=True)
        self.assertEqual(dt, UTCDateTime(2008, 12, 29))
        # w/ trailing Z
        dt = UTCDateTime("2009-W53-7T12:23:34.5Z")
        self.assertEqual(dt, UTCDateTime(2010, 1, 3, 12, 23, 34, 500000))
        dt = UTCDateTime("2009-W01-1T12:23:34Z")
        self.assertEqual(dt, UTCDateTime(2008, 12, 29, 12, 23, 34))
        dt = UTCDateTime("2009-W01-1T12:23Z")
        self.assertEqual(dt, UTCDateTime(2008, 12, 29, 12, 23))
        dt = UTCDateTime("2009-W01-1T12Z")
        self.assertEqual(dt, UTCDateTime(2008, 12, 29, 12))
        # compact
        dt = UTCDateTime("2009W537T122334.5Z")
        self.assertEqual(dt, UTCDateTime(2010, 1, 3, 12, 23, 34, 500000))
        dt = UTCDateTime("2009W011T122334Z")
        self.assertEqual(dt, UTCDateTime(2008, 12, 29, 12, 23, 34))
        dt = UTCDateTime("2009W011T1223Z")
        self.assertEqual(dt, UTCDateTime(2008, 12, 29, 12, 23))
        dt = UTCDateTime("2009W011T12Z")
        self.assertEqual(dt, UTCDateTime(2008, 12, 29, 12))

    def test_to_string(self):
        """
        Tests __str__ method.
        """
        dt = UTCDateTime(1970, 1, 1, 12, 23, 34)
        self.assertEqual(str(dt), '1970-01-01T12:23:34.000000Z')
        dt = UTCDateTime(1970, 1, 1, 12, 23, 34, 500000)
        self.assertEqual(str(dt), '1970-01-01T12:23:34.500000Z')
        dt = UTCDateTime(1970, 1, 1, 12, 23, 34.500000)
        self.assertEqual(str(dt), '1970-01-01T12:23:34.500000Z')
        dt = UTCDateTime(1970, 1, 1, 12, 23, 34, 5)
        self.assertEqual(str(dt), '1970-01-01T12:23:34.000005Z')
        dt = UTCDateTime(1970, 1, 1)
        self.assertEqual(str(dt), '1970-01-01T00:00:00.000000Z')

    def test_deepcopy(self):
        dt = UTCDateTime(1240561632.0050001)
        dt2 = copy.deepcopy(dt)
        dt += 68
        self.assertEqual(dt2.timestamp, 1240561632.0050001)
        self.assertEqual(dt.timestamp, 1240561700.0050001)

    def test_add(self):
        a = UTCDateTime(0.0)
        self.assertEqual(a + 1, UTCDateTime(1970, 1, 1, 0, 0, 1))
        self.assertEqual(a + int(1), UTCDateTime(1970, 1, 1, 0, 0, 1))
        self.assertEqual(a + np.int32(1), UTCDateTime(1970, 1, 1, 0, 0, 1))
        self.assertEqual(a + np.int64(1), UTCDateTime(1970, 1, 1, 0, 0, 1))
        self.assertEqual(a + np.float32(1), UTCDateTime(1970, 1, 1, 0, 0, 1))
        self.assertEqual(a + np.float64(1), UTCDateTime(1970, 1, 1, 0, 0, 1))
        self.assertEqual(
            a + 1.123456, UTCDateTime(1970, 1, 1, 0, 0, 1, 123456))
        self.assertEqual(
            a + 60 * 60 * 24 * 31 + 0.1,
            UTCDateTime(1970, 2, 1, 0, 0, 0, 100000))
        self.assertEqual(
            a + -0.5, UTCDateTime(1969, 12, 31, 23, 59, 59, 500000))
        td = datetime.timedelta(seconds=1)
        self.assertEqual(a + td, UTCDateTime(1970, 1, 1, 0, 0, 1))

    def test_sub(self):
        # 1
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        end = UTCDateTime(2000, 1, 1, 0, 0, 4, 995000)
        self.assertAlmostEqual(end - start, 4.995)
        # 2
        start = UTCDateTime(1000, 1, 1, 0, 0, 0, 0)
        end = UTCDateTime(1000, 1, 1, 0, 0, 4, 0)
        self.assertAlmostEqual(end - start, 4, 5)
        # 3
        start = UTCDateTime(0)
        td = datetime.timedelta(seconds=1)
        self.assertEqual(start - td, UTCDateTime(1969, 12, 31, 23, 59, 59))
        # 4
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 999999)
        end = UTCDateTime(2000, 1, 1, 0, 0, 1, 1)
        self.assertAlmostEqual(end - start, 0.000002, 6)

    def test_negative_timestamp(self):
        dt = UTCDateTime(-1000.1)
        self.assertEqual(str(dt), "1969-12-31T23:43:19.900000Z")
        self.assertEqual(dt.timestamp, -1000.1)

    def test_sub_with_negative_time_stamp(self):
        start = UTCDateTime(0)
        end = UTCDateTime(-1000.5)
        self.assertAlmostEqual(end - start, -1000.5)

    def test_small_negative_utc_date_time(self):
        """
        Windows OS supports only negative timestamps < -43200
        """
        # 0
        dt = UTCDateTime(0)
        self.assertEqual(dt.timestamp, 0)
        self.assertEqual(str(dt), "1970-01-01T00:00:00.000000Z")
        dt = UTCDateTime("1970-01-01T00:00:00.000000Z")
        self.assertEqual(dt.timestamp, 0)
        self.assertEqual(str(dt), "1970-01-01T00:00:00.000000Z")
        # -1
        dt = UTCDateTime(-1)
        self.assertEqual(dt.timestamp, -1)
        self.assertEqual(str(dt), "1969-12-31T23:59:59.000000Z")
        dt = UTCDateTime("1969-12-31T23:59:59.000000Z")
        self.assertEqual(dt.timestamp, -1)
        self.assertEqual(str(dt), "1969-12-31T23:59:59.000000Z")
        # -1.000001
        dt = UTCDateTime(-1.000001)
        self.assertEqual(dt.timestamp, -1.000001)
        self.assertEqual(str(dt), "1969-12-31T23:59:58.999999Z")
        dt = UTCDateTime("1969-12-31T23:59:58.999999Z")
        self.assertAlmostEqual(dt.timestamp, -1.000001, 6)
        self.assertEqual(str(dt), "1969-12-31T23:59:58.999999Z")
        # -0.000001
        dt = UTCDateTime("1969-12-31T23:59:59.999999Z")
        self.assertAlmostEqual(dt.timestamp, -0.000001, 6)
        self.assertEqual(str(dt), "1969-12-31T23:59:59.999999Z")
        dt = UTCDateTime(-0.000001)
        self.assertAlmostEqual(dt.timestamp, -0.000001, 6)
        self.assertEqual(str(dt), "1969-12-31T23:59:59.999999Z")
        # -0.000000001 - max precision is nanosecond!
        dt = UTCDateTime(-0.000000001)
        self.assertEqual(dt.timestamp, -0.000000001)
        self.assertEqual(str(dt), "1969-12-31T23:59:59.000000Z")
        # -1000.1
        dt = UTCDateTime("1969-12-31T23:43:19.900000Z")
        self.assertEqual(dt.timestamp, -1000.1)
        self.assertEqual(str(dt), "1969-12-31T23:43:19.900000Z")
        # -43199.123456
        dt = UTCDateTime(-43199.123456)
        self.assertAlmostEqual(dt.timestamp, -43199.123456, 6)
        self.assertEqual(str(dt), "1969-12-31T12:00:00.876544Z")

    def test_big_negative_utcdatetime(self):
        # 1
        dt = UTCDateTime("1969-12-31T23:43:19.900000Z")
        self.assertEqual(dt.timestamp, -1000.1)
        self.assertEqual(str(dt), "1969-12-31T23:43:19.900000Z")
        # 2
        dt = UTCDateTime("1905-01-01T12:23:34.123456Z")
        self.assertEqual(dt.timestamp, -2051177785.876544)
        self.assertEqual(str(dt), "1905-01-01T12:23:34.123456Z")

    def test_init_utcdatetime(self):
        dt = UTCDateTime(year=2008, month=1, day=1)
        self.assertEqual(str(dt), "2008-01-01T00:00:00.000000Z")
        dt = UTCDateTime(year=2008, julday=1, hour=12, microsecond=5000)
        self.assertEqual(str(dt), "2008-01-01T12:00:00.005000Z")
        # without parameters returns current date time
        UTCDateTime()

    def test_init_utcdatetime_mixing_keyworks_with_arguments(self):
        # times
        dt = UTCDateTime(2008, 1, 1, hour=12)
        self.assertEqual(dt, UTCDateTime(2008, 1, 1, 12))
        dt = UTCDateTime(2008, 1, 1, 12, minute=59)
        self.assertEqual(dt, UTCDateTime(2008, 1, 1, 12, 59))
        dt = UTCDateTime(2008, 1, 1, 12, 59, second=59)
        self.assertEqual(dt, UTCDateTime(2008, 1, 1, 12, 59, 59))
        dt = UTCDateTime(2008, 1, 1, 12, 59, 59, microsecond=123456)
        self.assertEqual(dt, UTCDateTime(2008, 1, 1, 12, 59, 59, 123456))
        dt = UTCDateTime(2008, 1, 1, hour=12, minute=59, second=59,
                         microsecond=123456)
        self.assertEqual(dt, UTCDateTime(2008, 1, 1, 12, 59, 59, 123456))
        # dates
        dt = UTCDateTime(2008, month=1, day=1)
        self.assertEqual(dt, UTCDateTime(2008, 1, 1))
        dt = UTCDateTime(2008, 1, day=1)
        self.assertEqual(dt, UTCDateTime(2008, 1, 1))
        dt = UTCDateTime(2008, julday=1)
        self.assertEqual(dt, UTCDateTime(2008, 1, 1))
        # combined
        dt = UTCDateTime(2008, julday=1, hour=12, minute=59, second=59,
                         microsecond=123456)
        self.assertEqual(dt, UTCDateTime(2008, 1, 1, 12, 59, 59, 123456))

    def test_to_python_date_time_objects(self):
        """
        Tests getDate, getTime, getTimestamp and getDateTime methods.
        """
        dt = UTCDateTime(1970, 1, 1, 12, 23, 34, 456789)
        # as function
        self.assertEqual(dt._get_date(), datetime.date(1970, 1, 1))
        self.assertEqual(dt._get_time(), datetime.time(12, 23, 34, 456789))
        self.assertEqual(dt._get_datetime(),
                         datetime.datetime(1970, 1, 1, 12, 23, 34, 456789))
        self.assertAlmostEqual(dt._get_timestamp(), 44614.456789)
        # as property
        self.assertEqual(dt.date, datetime.date(1970, 1, 1))
        self.assertEqual(dt.time, datetime.time(12, 23, 34, 456789))
        self.assertEqual(dt.datetime,
                         datetime.datetime(1970, 1, 1, 12, 23, 34, 456789))
        self.assertAlmostEqual(dt.timestamp, 44614.456789)

    def test_sub_add_float(self):
        """
        Tests subtraction of floats from UTCDateTime
        """
        time = UTCDateTime(2010, 0o5, 31, 19, 54, 24.490)
        delta = -0.045149
        expected = UTCDateTime("2010-05-31T19:54:24.535149Z")

        got1 = time + (-delta)
        got2 = time - delta
        self.assertAlmostEqual(got1 - got2, 0.0)
        self.assertAlmostEqual(expected.timestamp, got1.timestamp, 6)

    def test_issue_159(self):
        """
        Test case for issue #159.
        """
        dt = UTCDateTime("2010-2-13T2:13:11")
        self.assertEqual(dt, UTCDateTime(2010, 2, 13, 2, 13, 11))
        dt = UTCDateTime("2010-2-13T02:13:11")
        self.assertEqual(dt, UTCDateTime(2010, 2, 13, 2, 13, 11))
        dt = UTCDateTime("2010-2-13T2:13:11.123456")
        self.assertEqual(dt, UTCDateTime(2010, 2, 13, 2, 13, 11, 123456))
        dt = UTCDateTime("2010-2-13T02:9:9.123456")
        self.assertEqual(dt, UTCDateTime(2010, 2, 13, 2, 9, 9, 123456))

    def test_invalid_dates(self):
        """
        Tests invalid dates.
        """
        # Both should raise a value error that the day is too large for the
        # month.
        self.assertRaises(ValueError, UTCDateTime, 2010, 9, 31)
        self.assertRaises(ValueError, UTCDateTime, '2010-09-31')
        # invalid julday
        self.assertRaises(TypeError, UTCDateTime, year=2010, julday=999)
        # testing some strange patterns
        self.assertRaises(TypeError, UTCDateTime, "ABC")
        self.assertRaises(TypeError, UTCDateTime, "12X3T")
        self.assertRaises(ValueError, UTCDateTime, 2010, 9, 31)

    def test_invalid_times(self):
        """
        Tests invalid times.
        """
        # wrong time information
        self.assertRaises(ValueError, UTCDateTime, "2010-02-13T99999",
                          iso8601=True)
        self.assertRaises(ValueError, UTCDateTime, "2010-02-13 99999",
                          iso8601=True)
        self.assertRaises(ValueError, UTCDateTime, "2010-02-13T99999")
        self.assertRaises(TypeError, UTCDateTime, "2010-02-13T02:09:09.XXXXX")

    def test_issue_168(self):
        """
        Couldn't calculate julday before 1900.
        """
        # 1
        dt = UTCDateTime("2010-01-01")
        self.assertEqual(dt.julday, 1)
        # 2
        dt = UTCDateTime("1905-12-31")
        self.assertEqual(dt.julday, 365)
        # 3
        dt = UTCDateTime("1906-12-31T23:59:59.999999Z")
        self.assertEqual(dt.julday, 365)

    def test_format_seed(self):
        """
        Tests format_seed method
        """
        # 1
        dt = UTCDateTime("2010-01-01")
        self.assertEqual(dt.format_seed(compact=True), "2010,001")
        # 2
        dt = UTCDateTime("2010-01-01T00:00:00.000000")
        self.assertEqual(dt.format_seed(compact=True), "2010,001")
        # 3
        dt = UTCDateTime("2010-01-01T12:00:00")
        self.assertEqual(dt.format_seed(compact=True), "2010,001,12")
        # 4
        dt = UTCDateTime("2010-01-01T12:34:00")
        self.assertEqual(dt.format_seed(compact=True), "2010,001,12:34")
        # 5
        dt = UTCDateTime("2010-01-01T12:34:56")
        self.assertEqual(dt.format_seed(compact=True), "2010,001,12:34:56")
        # 6
        dt = UTCDateTime("2010-01-01T12:34:56.123456")
        self.assertEqual(dt.format_seed(compact=True),
                         "2010,001,12:34:56.1234")
        # 7 - explicit disabling compact flag still results into compact date
        # if no time information is given
        dt = UTCDateTime("2010-01-01")
        self.assertEqual(dt.format_seed(compact=False), "2010,001")

    def test_eq(self):
        """
        Tests __eq__ operators.
        """
        self.assertEqual(UTCDateTime(999), UTCDateTime(999))
        self.assertFalse(UTCDateTime(1) == UTCDateTime(999))
        # w/ default precision of 6 digits
        self.assertEqual(UTCDateTime(999.000001), UTCDateTime(999.000001))
        self.assertEqual(UTCDateTime(999.999999), UTCDateTime(999.999999))
        self.assertFalse(UTCDateTime(999.0000001) == UTCDateTime(999.0000009))
        self.assertFalse(UTCDateTime(999.9999990) == UTCDateTime(999.9999999))
        self.assertEqual(UTCDateTime(999.00000001), UTCDateTime(999.00000009))
        self.assertEqual(UTCDateTime(999.99999900), UTCDateTime(999.99999909))
        # w/ precision of 7 digits
        self.assertNotEqual(UTCDateTime(999.00000001, precision=7),
                            UTCDateTime(999.00000009, precision=7))
        self.assertNotEqual(UTCDateTime(999.99999990, precision=7),
                            UTCDateTime(999.99999999, precision=7))
        self.assertEqual(UTCDateTime(999.000000001, precision=7),
                         UTCDateTime(999.000000009, precision=7))
        self.assertEqual(UTCDateTime(999.999999900, precision=7),
                         UTCDateTime(999.999999909, precision=7))

    def test_ne(self):
        """
        Tests __ne__ operators.
        """
        self.assertFalse(UTCDateTime(999) != UTCDateTime(999))
        self.assertNotEqual(UTCDateTime(1), UTCDateTime(999))
        # w/ default precision of 6 digits
        self.assertFalse(UTCDateTime(999.000001) != UTCDateTime(999.000001))
        self.assertFalse(UTCDateTime(999.999999) != UTCDateTime(999.999999))
        self.assertNotEqual(UTCDateTime(999.0000001), UTCDateTime(999.0000009))
        self.assertNotEqual(UTCDateTime(999.9999990), UTCDateTime(999.9999999))
        self.assertFalse(UTCDateTime(999.00000001) !=
                         UTCDateTime(999.00000009))
        self.assertFalse(UTCDateTime(999.99999900) !=
                         UTCDateTime(999.99999909))
        # w/ precision of 7 digits
        self.assertNotEqual(UTCDateTime(999.00000001, precision=7),
                            UTCDateTime(999.00000009, precision=7))
        self.assertNotEqual(UTCDateTime(999.99999990, precision=7),
                            UTCDateTime(999.99999999, precision=7))
        self.assertFalse(UTCDateTime(999.000000001, precision=7) !=
                         UTCDateTime(999.000000009, precision=7))
        self.assertFalse(UTCDateTime(999.999999900, precision=7) !=
                         UTCDateTime(999.999999909, precision=7))

    def test_lt(self):
        """
        Tests __lt__ operators.
        """
        self.assertFalse(UTCDateTime(999) < UTCDateTime(999))
        self.assertTrue(UTCDateTime(1) < UTCDateTime(999))
        self.assertFalse(UTCDateTime(999) < UTCDateTime(1))
        # w/ default precision of 6 digits
        self.assertFalse(UTCDateTime(999.000001) < UTCDateTime(999.000001))
        self.assertFalse(UTCDateTime(999.999999) < UTCDateTime(999.999999))
        self.assertTrue(UTCDateTime(999.0000001) < UTCDateTime(999.0000009))
        self.assertFalse(UTCDateTime(999.0000009) < UTCDateTime(999.0000001))
        self.assertTrue(UTCDateTime(999.9999990) < UTCDateTime(999.9999999))
        self.assertFalse(UTCDateTime(999.9999999) < UTCDateTime(999.9999990))
        self.assertFalse(UTCDateTime(999.00000001) < UTCDateTime(999.00000009))
        self.assertFalse(UTCDateTime(999.00000009) < UTCDateTime(999.00000001))
        self.assertFalse(UTCDateTime(999.99999900) < UTCDateTime(999.99999909))
        self.assertFalse(UTCDateTime(999.99999909) < UTCDateTime(999.99999900))
        # w/ precision of 7 digits
        self.assertTrue(UTCDateTime(999.00000001, precision=7) <
                        UTCDateTime(999.00000009, precision=7))
        self.assertFalse(UTCDateTime(999.00000009, precision=7) <
                         UTCDateTime(999.00000001, precision=7))
        self.assertTrue(UTCDateTime(999.99999990, precision=7) <
                        UTCDateTime(999.99999999, precision=7))
        self.assertFalse(UTCDateTime(999.99999999, precision=7) <
                         UTCDateTime(999.99999990, precision=7))
        self.assertFalse(UTCDateTime(999.000000001, precision=7) <
                         UTCDateTime(999.000000009, precision=7))
        self.assertFalse(UTCDateTime(999.000000009, precision=7) <
                         UTCDateTime(999.000000001, precision=7))
        self.assertFalse(UTCDateTime(999.999999900, precision=7) <
                         UTCDateTime(999.999999909, precision=7))
        self.assertFalse(UTCDateTime(999.999999909, precision=7) <
                         UTCDateTime(999.999999900, precision=7))

    def test_le(self):
        """
        Tests __le__ operators.
        """
        self.assertLessEqual(UTCDateTime(999), UTCDateTime(999))
        self.assertLessEqual(UTCDateTime(1), UTCDateTime(999))
        self.assertFalse(UTCDateTime(999) <= UTCDateTime(1))
        # w/ default precision of 6 digits
        self.assertLessEqual(UTCDateTime(999.000001), UTCDateTime(999.000001))
        self.assertLessEqual(UTCDateTime(999.999999), UTCDateTime(999.999999))
        self.assertLessEqual(UTCDateTime(999.0000001),
                             UTCDateTime(999.0000009))
        self.assertFalse(UTCDateTime(999.0000009) <= UTCDateTime(999.0000001))
        self.assertLessEqual(UTCDateTime(999.9999990),
                             UTCDateTime(999.9999999))
        self.assertFalse(UTCDateTime(999.9999999) <= UTCDateTime(999.9999990))
        self.assertLessEqual(UTCDateTime(999.00000001),
                             UTCDateTime(999.00000009))
        self.assertLessEqual(UTCDateTime(999.00000009),
                             UTCDateTime(999.00000001))
        self.assertLessEqual(UTCDateTime(999.99999900),
                             UTCDateTime(999.99999909))
        self.assertLessEqual(UTCDateTime(999.99999909),
                             UTCDateTime(999.99999900))
        # w/ precision of 7 digits
        self.assertLessEqual(UTCDateTime(999.00000001, precision=7),
                             UTCDateTime(999.00000009, precision=7))
        self.assertFalse(UTCDateTime(999.00000009, precision=7) <=
                         UTCDateTime(999.00000001, precision=7))
        self.assertLessEqual(UTCDateTime(999.99999990, precision=7),
                             UTCDateTime(999.99999999, precision=7))
        self.assertFalse(UTCDateTime(999.99999999, precision=7) <=
                         UTCDateTime(999.99999990, precision=7))
        self.assertLessEqual(UTCDateTime(999.000000001, precision=7),
                             UTCDateTime(999.000000009, precision=7))
        self.assertLessEqual(UTCDateTime(999.000000009, precision=7),
                             UTCDateTime(999.000000001, precision=7))
        self.assertLessEqual(UTCDateTime(999.999999900, precision=7),
                             UTCDateTime(999.999999909, precision=7))
        self.assertLessEqual(UTCDateTime(999.999999909, precision=7),
                             UTCDateTime(999.999999900, precision=7))

    def test_gt(self):
        """
        Tests __gt__ operators.
        """
        self.assertFalse(UTCDateTime(999) > UTCDateTime(999))
        self.assertFalse(UTCDateTime(1) > UTCDateTime(999))
        self.assertGreater(UTCDateTime(999), UTCDateTime(1))
        # w/ default precision of 6 digits
        self.assertFalse(UTCDateTime(999.000001) > UTCDateTime(999.000001))
        self.assertFalse(UTCDateTime(999.999999) > UTCDateTime(999.999999))
        self.assertFalse(UTCDateTime(999.0000001) > UTCDateTime(999.0000009))
        self.assertGreater(UTCDateTime(999.0000009), UTCDateTime(999.0000001))
        self.assertFalse(UTCDateTime(999.9999990) > UTCDateTime(999.9999999))
        self.assertGreater(UTCDateTime(999.9999999), UTCDateTime(999.9999990))
        self.assertFalse(UTCDateTime(999.00000001) > UTCDateTime(999.00000009))
        self.assertFalse(UTCDateTime(999.00000009) > UTCDateTime(999.00000001))
        self.assertFalse(UTCDateTime(999.99999900) > UTCDateTime(999.99999909))
        self.assertFalse(UTCDateTime(999.99999909) > UTCDateTime(999.99999900))
        # w/ precision of 7 digits
        self.assertFalse(UTCDateTime(999.00000001, precision=7) >
                         UTCDateTime(999.00000009, precision=7))
        self.assertGreater(UTCDateTime(999.00000009, precision=7),
                           UTCDateTime(999.00000001, precision=7))
        self.assertFalse(UTCDateTime(999.99999990, precision=7) >
                         UTCDateTime(999.99999999, precision=7))
        self.assertGreater(UTCDateTime(999.99999999, precision=7),
                           UTCDateTime(999.99999990, precision=7))
        self.assertFalse(UTCDateTime(999.000000001, precision=7) >
                         UTCDateTime(999.000000009, precision=7))
        self.assertFalse(UTCDateTime(999.000000009, precision=7) >
                         UTCDateTime(999.000000001, precision=7))
        self.assertFalse(UTCDateTime(999.999999900, precision=7) >
                         UTCDateTime(999.999999909, precision=7))
        self.assertFalse(UTCDateTime(999.999999909, precision=7) >
                         UTCDateTime(999.999999900, precision=7))

    def test_ge(self):
        """
        Tests __ge__ operators.
        """
        self.assertGreaterEqual(UTCDateTime(999), UTCDateTime(999))
        self.assertFalse(UTCDateTime(1) >= UTCDateTime(999))
        self.assertGreaterEqual(UTCDateTime(999), UTCDateTime(1))
        # w/ default precision of 6 digits
        self.assertGreaterEqual(UTCDateTime(999.000001),
                                UTCDateTime(999.000001))
        self.assertGreaterEqual(UTCDateTime(999.999999),
                                UTCDateTime(999.999999))
        self.assertFalse(UTCDateTime(999.0000001) >= UTCDateTime(999.0000009))
        self.assertGreaterEqual(UTCDateTime(999.0000009),
                                UTCDateTime(999.0000001))
        self.assertFalse(UTCDateTime(999.9999990) >= UTCDateTime(999.9999999))
        self.assertGreaterEqual(UTCDateTime(999.9999999),
                                UTCDateTime(999.9999990))
        self.assertGreaterEqual(UTCDateTime(999.00000001),
                                UTCDateTime(999.00000009))
        self.assertGreaterEqual(UTCDateTime(999.00000009),
                                UTCDateTime(999.00000001))
        self.assertGreaterEqual(UTCDateTime(999.99999900),
                                UTCDateTime(999.99999909))
        self.assertGreaterEqual(UTCDateTime(999.99999909),
                                UTCDateTime(999.99999900))
        # w/ precision of 7 digits
        self.assertFalse(UTCDateTime(999.00000001, precision=7) >=
                         UTCDateTime(999.00000009, precision=7))
        self.assertGreaterEqual(UTCDateTime(999.00000009, precision=7),
                                UTCDateTime(999.00000001, precision=7))
        self.assertFalse(UTCDateTime(999.99999990, precision=7) >=
                         UTCDateTime(999.99999999, precision=7))
        self.assertGreaterEqual(UTCDateTime(999.99999999, precision=7),
                                UTCDateTime(999.99999990, precision=7))
        self.assertGreaterEqual(UTCDateTime(999.000000001, precision=7),
                                UTCDateTime(999.000000009, precision=7))
        self.assertGreaterEqual(UTCDateTime(999.000000009, precision=7),
                                UTCDateTime(999.000000001, precision=7))
        self.assertGreaterEqual(UTCDateTime(999.999999900, precision=7),
                                UTCDateTime(999.999999909, precision=7))
        self.assertGreaterEqual(UTCDateTime(999.999999909, precision=7),
                                UTCDateTime(999.999999900, precision=7))

    def test_toordinal(self):
        """
        Short test if toordinal() is working.
        Matplotlib's date2num() function depends on this which is used a lot in
        plotting.
        """
        dt = UTCDateTime("2012-03-04T11:05:09.123456Z")
        self.assertEqual(dt.toordinal(), 734566)

    def test_weekday(self):
        """
        Tests weekday method.
        """
        dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        self.assertEqual(dt.weekday, 2)
        self.assertEqual(dt._get_weekday(), 2)

    def test_default_precision(self):
        """
        Tests setting of default precisions via monkey patching.
        """
        dt = UTCDateTime()
        # instance
        self.assertEqual(dt.precision, 6)
        self.assertEqual(dt.DEFAULT_PRECISION, 6)
        # class
        self.assertEqual(UTCDateTime.DEFAULT_PRECISION, 6)
        dt = UTCDateTime()
        # set new default precision
        UTCDateTime.DEFAULT_PRECISION = 3
        dt2 = UTCDateTime()
        # first instance should be unchanged
        self.assertEqual(dt.precision, 6)
        # but class attribute has changed
        self.assertEqual(dt.DEFAULT_PRECISION, 3)
        # class
        self.assertEqual(UTCDateTime.DEFAULT_PRECISION, 3)
        # second instance should use new precision
        self.assertEqual(dt2.DEFAULT_PRECISION, 3)
        self.assertEqual(dt2.precision, 3)
        # cleanup
        UTCDateTime.DEFAULT_PRECISION = 6
        # class
        self.assertEqual(UTCDateTime.DEFAULT_PRECISION, 6)

    def test_to_string_precision(self):
        """
        Tests __str__ method while using a precision.
        """
        # precision 7
        dt = UTCDateTime(1980, 2, 3, 12, 23, 34, precision=7)
        self.assertEqual(str(dt), '1980-02-03T12:23:34.0000000Z')
        dt = UTCDateTime(1980, 2, 3, 12, 23, 34, 500000, precision=7)
        self.assertEqual(str(dt), '1980-02-03T12:23:34.5000000Z')
        dt = UTCDateTime(1980, 2, 3, 12, 23, 34.500000, precision=7)
        self.assertEqual(str(dt), '1980-02-03T12:23:34.5000000Z')
        dt = UTCDateTime(1980, 2, 3, 12, 23, 34, 5, precision=7)
        self.assertEqual(str(dt), '1980-02-03T12:23:34.0000050Z')
        dt = UTCDateTime(1980, 2, 3, precision=7)
        self.assertEqual(str(dt), '1980-02-03T00:00:00.0000000Z')
        dt = UTCDateTime(1980, 2, 3, 12, 23, 34, 444999, precision=7)
        self.assertEqual(str(dt), '1980-02-03T12:23:34.4449990Z')
        # precision 3
        dt = UTCDateTime(1980, 2, 3, 12, 23, 34, precision=3)
        self.assertEqual(str(dt), '1980-02-03T12:23:34.000Z')
        dt = UTCDateTime(1980, 2, 3, 12, 23, 34, 500000, precision=3)
        self.assertEqual(str(dt), '1980-02-03T12:23:34.500Z')
        dt = UTCDateTime(1980, 2, 3, 12, 23, 34.500000, precision=3)
        self.assertEqual(str(dt), '1980-02-03T12:23:34.500Z')
        dt = UTCDateTime(1980, 2, 3, 12, 23, 34, 5, precision=3)
        self.assertEqual(str(dt), '1980-02-03T12:23:34.000Z')
        dt = UTCDateTime(1980, 2, 3, precision=3)
        self.assertEqual(str(dt), '1980-02-03T00:00:00.000Z')
        dt = UTCDateTime(1980, 2, 3, 12, 23, 34, 444999, precision=3)
        self.assertEqual(str(dt), '1980-02-03T12:23:34.445Z')

    def test_rich_comparision_numeric_objects(self):
        """
        Tests basic rich comparison operations against numeric objects.
        """
        t1 = UTCDateTime(2005, 3, 4, 12, 33, 44)
        t2 = UTCDateTime(2005, 3, 4, 12, 33, 44, 123456)
        t1_int = 1109939624
        t2_int = 1109939624
        t1_float = 1109939624.0
        t2_float = 1109939624.123456
        # test (not) equal
        self.assertEqual(t1, t1_int)
        self.assertEqual(t1, t1_float)
        self.assertFalse(t2 == t2_int)
        self.assertEqual(t2, t2_float)
        self.assertFalse(t1 != t1_int)
        self.assertFalse(t1 != t1_float)
        self.assertNotEqual(t2, t2_int)
        self.assertFalse(t2 != t2_float)
        # test less/greater(equal)
        self.assertGreaterEqual(t1, t1_int)
        self.assertLessEqual(t1, t1_int)
        self.assertFalse(t1 > t1_int)
        self.assertFalse(t1 < t1_int)
        self.assertGreaterEqual(t1, t1_float)
        self.assertLessEqual(t1, t1_float)
        self.assertFalse(t1 > t1_float)
        self.assertFalse(t1 < t1_float)
        self.assertGreaterEqual(t2, t2_int)
        self.assertFalse(t2 <= t2_int)
        self.assertGreater(t2, t2_int)
        self.assertFalse(t2 < t2_int)
        self.assertGreaterEqual(t2, t2_float)
        self.assertLessEqual(t2, t2_float)
        self.assertFalse(t2 > t2_float)
        self.assertFalse(t2 < t2_float)

    def test_rich_comparision_numeric_types(self):
        """
        Tests basic rich comparison operations against non-numeric objects.
        """
        dt = UTCDateTime()
        for obj in [None, 'string', object()]:
            self.assertFalse(dt == obj)
            self.assertNotEqual(dt, obj)
            self.assertFalse(dt <= obj)
            self.assertFalse(dt < obj)
            self.assertFalse(dt >= obj)
            self.assertFalse(dt > obj)
            self.assertFalse(obj == dt)
            self.assertNotEqual(obj, dt)
            self.assertFalse(obj <= dt)
            self.assertFalse(obj < dt)
            self.assertFalse(obj >= dt)
            self.assertFalse(obj > dt)

    def test_rich_comparision_fuzzy(self):
        """
        UTCDateTime fuzzy comparisons break sorting, max, min - see #1765
        """
        # 1 - precision set to 6 - 3
        for precision in [6, 5, 4, 3]:
            dt1 = UTCDateTime(0.001, precision=precision)
            dt2 = UTCDateTime(0.004, precision=precision)
            dt3 = UTCDateTime(0.007, precision=precision)
            # comparison
            self.assertEqual(dt1 == dt2, False)
            self.assertEqual(dt2 == dt3, False)
            self.assertEqual(dt1 == dt3, False)
            self.assertEqual(dt1 < dt2, True)
            self.assertEqual(dt2 < dt3, True)
            self.assertEqual(dt1 < dt3, True)
            self.assertEqual(dt1 <= dt2, True)
            self.assertEqual(dt2 <= dt3, True)
            self.assertEqual(dt1 <= dt3, True)
            self.assertEqual(dt1 > dt2, False)
            self.assertEqual(dt2 > dt3, False)
            self.assertEqual(dt1 > dt3, False)
            self.assertEqual(dt1 >= dt2, False)
            self.assertEqual(dt2 >= dt3, False)
            self.assertEqual(dt1 >= dt3, False)
            # sorting
            times = [dt3, dt2, dt1]
            sorted_times = sorted(times)
            self.assertEqual(sorted_times[0] <= sorted_times[2], True)
            self.assertEqual(sorted_times[0] < sorted_times[2], True)
            self.assertEqual(sorted_times[0] == sorted_times[2], False)
            self.assertEqual(sorted_times, [dt1, dt2, dt3])
            # min, max
            max_times = max(dt2, dt1, dt3)
            self.assertEqual(max_times, dt3)
            self.assertEqual(max_times._ns, 7000000)
            max_times = max(dt1, dt2, dt3)
            self.assertEqual(max_times, dt3)
            self.assertEqual(max_times._ns, 7000000)
            times = [dt2, dt1, dt3]
            self.assertEqual(max(times), dt3)
            self.assertEqual(max(times)._ns, 7000000)
            self.assertEqual(min(times), dt1)
            self.assertEqual(min(times)._ns, 1000000)
            self.assertEqual(max(times) - min(times), 0.006)
            times = [dt1, dt2, dt3]
            self.assertEqual(max(times) - min(times), 0.006)

        # 2 - precision set to 2
        dt1 = UTCDateTime(0.001, precision=2)  # == 0.00
        dt2 = UTCDateTime(0.004, precision=2)  # == 0.00
        dt3 = UTCDateTime(0.007, precision=2)  # == 0.01
        # comparison
        self.assertEqual(dt1 == dt2, True)
        self.assertEqual(dt2 == dt3, False)
        self.assertEqual(dt1 == dt3, False)
        self.assertEqual(dt1 < dt2, False)
        self.assertEqual(dt2 < dt3, True)
        self.assertEqual(dt1 < dt3, True)
        self.assertEqual(dt1 <= dt2, True)
        self.assertEqual(dt2 <= dt3, True)
        self.assertEqual(dt1 <= dt3, True)
        self.assertEqual(dt1 > dt2, False)
        self.assertEqual(dt2 > dt3, False)
        self.assertEqual(dt1 > dt3, False)
        self.assertEqual(dt1 >= dt2, True)
        self.assertEqual(dt2 >= dt3, False)
        self.assertEqual(dt1 >= dt3, False)
        # sorting
        times = [dt3, dt2, dt1]
        sorted_times = sorted(times)
        self.assertEqual(sorted_times[0] <= sorted_times[2], True)
        self.assertEqual(sorted_times[0] < sorted_times[2], True)
        self.assertEqual(sorted_times[0] == sorted_times[2], False)
        self.assertEqual(sorted_times[0] > sorted_times[2], False)
        self.assertEqual(sorted_times[0] >= sorted_times[2], False)
        self.assertEqual(sorted_times, [dt2, dt1, dt3])  # expected
        self.assertEqual(sorted_times, [dt2, dt1, dt3])  # due to precision
        # check correct sort order
        self.assertEqual(sorted_times[0]._ns, dt2._ns)
        self.assertEqual(sorted_times[2]._ns, dt3._ns)
        # min, max
        max_times = max(dt2, dt1, dt3)
        self.assertEqual(max_times, dt3)
        self.assertEqual(max_times._ns, 7000000)
        max_times = max(dt1, dt2, dt3)
        self.assertEqual(max_times, dt3)
        self.assertEqual(max_times._ns, 7000000)
        # min, max lists
        times = [dt2, dt1, dt3]
        self.assertEqual(max(times), dt3)
        self.assertEqual(max(times)._ns, 7000000)
        self.assertEqual(min(times), dt2)  # expected
        self.assertEqual(min(times), dt1)  # due to precision
        self.assertEqual(min(times)._ns, 4000000)  # due to given list order
        self.assertEqual(max(times) - min(times), 0.01)
        times = [dt1, dt2, dt3]
        self.assertEqual(max(times), dt3)
        self.assertEqual(max(times)._ns, 7000000)
        self.assertEqual(min(times), dt1)  # expected
        self.assertEqual(min(times), dt2)  # due to precision
        self.assertEqual(min(times)._ns, 1000000)  # due to given list order
        self.assertEqual(max(times) - min(times), 0.01)

    def test_datetime_with_timezone(self):
        """
        UTCDateTime from timezone-aware datetime.datetime

        .. seealso:: https://github.com/obspy/obspy/issues/553
        """
        class ManilaTime(datetime.tzinfo):

            def utcoffset(self, dt):  # @UnusedVariable
                return datetime.timedelta(hours=8)

        dt = datetime.datetime(2006, 11, 21, 16, 30, tzinfo=ManilaTime())
        self.assertEqual(dt.isoformat(), '2006-11-21T16:30:00+08:00')
        self.assertEqual(UTCDateTime(dt.isoformat()), UTCDateTime(dt))

    def test_hash(self):
        """
        Test __hash__ method of UTCDateTime class.
        """
        self.assertEqual(UTCDateTime().__hash__(), None)

    def test_now(self):
        """
        Test now class method of UTCDateTime class.
        """
        dt = UTCDateTime()
        self.assertGreaterEqual(UTCDateTime.now(), dt)

    def test_utcnow(self):
        """
        Test utcnow class method of UTCDateTime class.
        """
        dt = UTCDateTime()
        self.assertGreaterEqual(UTCDateTime.utcnow(), dt)

    def test_abs(self):
        """
        Test __abs__ method of UTCDateTime class.
        """
        dt = UTCDateTime(1970, 1, 1, 0, 0, 1)
        self.assertEqual(abs(dt), 1)
        dt = UTCDateTime(1970, 1, 1, 0, 0, 1, 500000)
        self.assertEqual(abs(dt), 1.5)
        dt = UTCDateTime(1970, 1, 1)
        self.assertEqual(abs(dt), 0)
        dt = UTCDateTime(1969, 12, 31, 23, 59, 59)
        self.assertEqual(abs(dt), 1)
        dt = UTCDateTime(1969, 12, 31, 23, 59, 59, 500000)
        self.assertEqual(abs(dt), 0.5)

    def test_string_with_timezone(self):
        """
        Test that all valid ISO time zone specifications are parsed properly
        https://en.wikipedia.org/wiki/ISO_8601#Time_offsets_from_UTC
        """
        # positive
        t = UTCDateTime("2013-09-01T12:34:56Z")
        time_strings = \
            ["2013-09-01T14:34:56+02", "2013-09-01T14:34:56+02:00",
             "2013-09-01T14:34:56+0200", "2013-09-01T14:49:56+02:15",
             "2013-09-01T12:34:56+00:00", "2013-09-01T12:34:56+00",
             "2013-09-01T12:34:56+0000"]
        for time_string in time_strings:
            self.assertEqual(t, UTCDateTime(time_string))

        # negative
        t = UTCDateTime("2013-09-01T12:34:56Z")
        time_strings = \
            ["2013-09-01T10:34:56-02", "2013-09-01T10:34:56-02:00",
             "2013-09-01T10:34:56-0200", "2013-09-01T10:19:56-02:15",
             "2013-09-01T12:34:56-00:00", "2013-09-01T12:34:56-00",
             "2013-09-01T12:34:56-0000"]
        for time_string in time_strings:
            self.assertEqual(t, UTCDateTime(time_string))

    def test_year_2038_problem(self):
        """
        See issue #805
        """
        dt = UTCDateTime(2004, 1, 10, 13, 37, 4)
        self.assertEqual(dt.__str__(), '2004-01-10T13:37:04.000000Z')
        dt = UTCDateTime(2038, 1, 19, 3, 14, 8)
        self.assertEqual(dt.__str__(), '2038-01-19T03:14:08.000000Z')
        dt = UTCDateTime(2106, 2, 7, 6, 28, 16)
        self.assertEqual(dt.__str__(), '2106-02-07T06:28:16.000000Z')

    def test_format_iris_webservice(self):
        """
        Tests the format IRIS webservice function.

        See issue #1096.
        """
        # These are parse slightly differently (1 microsecond difference but
        # the IRIS webservice string should be identical as its only
        # accurate to three digits.
        d1 = UTCDateTime(2011, 1, 25, 15, 32, 12.26)
        d2 = UTCDateTime("2011-01-25T15:32:12.26")

        self.assertEqual(d1.format_iris_web_service(),
                         d2.format_iris_web_service())

    def test_floating_point_second_initialization(self):
        """
        Tests floating point precision issues in initialization of UTCDateTime
        objects with floating point seconds.

        See issue #1096.
        """
        for microns in np.arange(0, 5999, dtype=np.int):
            t = UTCDateTime(2011, 1, 25, 15, 32, 12 + microns / 1e6)
            self.assertEqual(microns, t.microsecond)

    def test_issue_1215(self):
        """
        Tests some non-ISO8601 strings which should be also properly parsed.

        See issue #1215.
        """
        self.assertEqual(UTCDateTime('2015-07-03-06'),
                         UTCDateTime(2015, 7, 3, 6, 0, 0))
        self.assertEqual(UTCDateTime('2015-07-03-06-42'),
                         UTCDateTime(2015, 7, 3, 6, 42, 0))
        self.assertEqual(UTCDateTime('2015-07-03-06-42-1'),
                         UTCDateTime(2015, 7, 3, 6, 42, 1))
        self.assertEqual(UTCDateTime('2015-07-03-06-42-1.5123'),
                         UTCDateTime(2015, 7, 3, 6, 42, 1, 512300))

    def test_matplotlib_date(self):
        """
        Test convenience method and property for conversion to matplotlib
        datetime float numbers.
        """
        for t_, expected in zip(
                ("1986-05-02T13:44:12.567890Z", "2009-08-24T00:20:07.700000Z",
                 "2026-11-27T03:12:45.4"),
                (725128.5723676839, 733643.0139780092, 739947.1338587963)):
            t = UTCDateTime(t_)
            np.testing.assert_almost_equal(
                t.matplotlib_date, expected, decimal=8)

    def test_add_error_message(self):
        t = UTCDateTime()
        t2 = UTCDateTime()
        with self.assertRaises(TypeError) as context:
            t + t2
        self.assertEqual(
            str(context.exception),
            "unsupported operand type(s) for +: 'UTCDateTime' and "
            "'UTCDateTime'")

    def test_nanoseconds(self):
        """
        Various nanosecond tests.

        Also tests #1318.
        """
        # 1
        dt = UTCDateTime(1, 1, 1, 0, 0, 0, 0)
        self.assertEqual(dt._ns, -62135596800000000000)
        self.assertEqual(dt.timestamp, -62135596800.0)
        self.assertEqual(dt.microsecond, 0)
        self.assertEqual(dt.datetime, datetime.datetime(1, 1, 1, 0, 0, 0, 0))
        self.assertEqual(str(dt), '0001-01-01T00:00:00.000000Z')
        # 2
        dt = UTCDateTime(1, 1, 1, 0, 0, 0, 1)
        self.assertEqual(dt._ns, -62135596799999999000)
        self.assertEqual(dt.microsecond, 1)
        self.assertEqual(dt.datetime, datetime.datetime(1, 1, 1, 0, 0, 0, 1))
        self.assertEqual(str(dt), '0001-01-01T00:00:00.000001Z')
        self.assertEqual(dt.timestamp, -62135596800.000001)
        # 3
        dt = UTCDateTime(1, 1, 1, 0, 0, 0, 999999)
        self.assertEqual(dt._ns, -62135596799000001000)
        # dt.timestamp should be -62135596799.000001 - not possible to display
        # correctly using python floats
        self.assertEqual(dt.timestamp, -62135596799.0)
        self.assertEqual(dt.microsecond, 999999)
        self.assertEqual(dt.datetime,
                         datetime.datetime(1, 1, 1, 0, 0, 0, 999999))
        self.assertEqual(str(dt), '0001-01-01T00:00:00.999999Z')
        # 4
        dt = UTCDateTime(1, 1, 1, 0, 0, 1, 0)
        self.assertEqual(dt._ns, -62135596799000000000)
        self.assertEqual(dt.timestamp, -62135596799.0)
        self.assertEqual(dt.microsecond, 0)
        self.assertEqual(dt.datetime,
                         datetime.datetime(1, 1, 1, 0, 0, 1, 0))
        self.assertEqual(str(dt), '0001-01-01T00:00:01.000000Z')
        # 5
        dt = UTCDateTime(1, 1, 1, 0, 0, 1, 1)
        self.assertEqual(dt._ns, -62135596798999999000)
        # dt.timestamp should be -62135596799.000001 - not possible to display
        # correctly using python floats
        self.assertEqual(dt.timestamp, -62135596799.0)
        self.assertEqual(dt.microsecond, 1)
        self.assertEqual(dt.datetime,
                         datetime.datetime(1, 1, 1, 0, 0, 1, 1))
        self.assertEqual(str(dt), '0001-01-01T00:00:01.000001Z')
        # 6
        dt = UTCDateTime(1970, 1, 1, 0, 0, 0, 1)
        self.assertEqual(dt._ns, 1000)
        self.assertEqual(dt.timestamp, 0.000001)
        self.assertEqual(dt.microsecond, 1)
        self.assertEqual(dt.datetime,
                         datetime.datetime(1970, 1, 1, 0, 0, 0, 1))
        self.assertEqual(str(dt), '1970-01-01T00:00:00.000001Z')
        # 7
        dt = UTCDateTime(1970, 1, 1, 0, 0, 0, 999999)
        self.assertEqual(dt._ns, 999999000)
        self.assertEqual(dt.timestamp, 0.999999)
        self.assertEqual(dt.microsecond, 999999)
        self.assertEqual(dt.datetime,
                         datetime.datetime(1970, 1, 1, 0, 0, 0, 999999))
        self.assertEqual(str(dt), '1970-01-01T00:00:00.999999Z')
        # 8
        dt = UTCDateTime(3000, 1, 1, 0, 0, 0, 500000)
        self.assertEqual(dt._ns, 32503680000500000000)
        self.assertEqual(dt.timestamp, 32503680000.5)
        self.assertEqual(dt.microsecond, 500000)
        self.assertEqual(dt.datetime,
                         datetime.datetime(3000, 1, 1, 0, 0, 0, 500000))
        self.assertEqual(str(dt), '3000-01-01T00:00:00.500000Z')
        # 9
        dt = UTCDateTime(9999, 1, 1, 0, 0, 0, 500000)
        self.assertEqual(dt._ns, 253370764800500000000)
        self.assertEqual(dt.timestamp, 253370764800.5)
        self.assertEqual(dt.microsecond, 500000)
        self.assertEqual(dt.datetime,
                         datetime.datetime(9999, 1, 1, 0, 0, 0, 500000))
        self.assertEqual(str(dt), '9999-01-01T00:00:00.500000Z')

    def test_utcdatetime_from_utcdatetime(self):
        a = UTCDateTime(1, 1, 1, 1, 1, 1, 999999)
        self.assertEqual(UTCDateTime(a)._ns, a._ns)
        self.assertEqual(str(UTCDateTime(a)), str(a))

    def test_issue_1008(self):
        """
        see #1008
        """
        self.assertEqual(str(UTCDateTime("9999-12-31T23:59:59.9999")),
                         "9999-12-31T23:59:59.999900Z")
        self.assertEqual(str(UTCDateTime("9999-12-31T23:59:59.999999")),
                         "9999-12-31T23:59:59.999999Z")

    def test_issue_1652(self):
        """
        Comparing UTCDateTime and datetime.datetime objects - see #1652
        """
        a = datetime.datetime(1990, 1, 1, 0, 0)
        e = UTCDateTime(2000, 1, 2, 1, 39, 37)
        self.assertTrue(a < e)
        self.assertFalse(a > e)
        self.assertTrue(a <= e)
        self.assertFalse(e <= a)
        self.assertFalse(a > e)
        self.assertTrue(e > a)
        self.assertFalse(a >= e)
        self.assertTrue(e >= a)
        self.assertFalse(a == e)
        self.assertFalse(e == a)

    def test_timestamp_can_serialize_with_time_attrs(self):
        """
        Test that the datetime attrs can be used to serialize UTCDateTime
        objects inited from floats (given default precision of 6) - see 2034
        """
        time_attrs = ('year', 'month', 'day', 'hour', 'minute', 'second',
                      'microsecond')
        close_timestamps = [1515174511.1984465, 1515174511.1984463,
                            1515174511.1984460, 1515174511.1984458]
        close_utc = [UTCDateTime(x) for x in close_timestamps]

        for utc in close_utc:
            utc2 = UTCDateTime(**{x: getattr(utc, x) for x in time_attrs})
            self.assertEqual(utc, utc2)

    def test_str_ms_equal_ms(self):
        """
        Test that the microseconds in the str representation are equal to
        the microseconds attr - see 2034
        """
        close_timestamps = [1515174511.1984465, 1515174511.1984463,
                            1515174511.1984460, 1515174511.1984458]
        close_utc = [UTCDateTime(x) for x in close_timestamps]

        for utc in close_utc:
            str_ms = int(str(utc).split('.')[-1][:-1])  # get ms from str rep
            ms = utc.microsecond
            self.assertEqual(str_ms, ms)

    def test_close_utc_are_equal(self):
        """
        Ensure UTCs init'ed with floats that are very close together are
        equal - see 2034

        Note: Due to the rounding nanosecond attribute before comparision
        we can no longer guarantee equality based on the difference in
        nanoseconds. This trade-off was made to ensure UTCDateTime objects
        are always equal to their string representation when precision <= 6.
        See issue #2034 and pr # 1766.
        """
        # get an array of floats as close together as possible
        def yield_close_floats(start, length):
            for _ in range(length):
                start = np.nextafter(start, 0)
                yield start

        # convert to UTCDateTime objects
        close_timestamps = list(yield_close_floats(1515174511.1984458, 10))
        close_utc = [UTCDateTime(x) for x in close_timestamps]

        # if str are equal then objects should be equal and visa versa
        for num in range(len(close_utc) - 1):
            utc1 = close_utc[num]
            utc2 = close_utc[num + 1]
            if utc1 == utc2:
                self.assertEqual(str(utc1), str(utc2))
            if str(utc1) == str(utc2):
                self.assertEqual(utc1, utc2)


def suite():
    return unittest.makeSuite(UTCDateTimeTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
