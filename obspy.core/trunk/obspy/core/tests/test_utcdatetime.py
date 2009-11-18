# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime
import datetime
import copy
import unittest


class UTCDateTimeTestCase(unittest.TestCase):
    """
    Test suite for L{obspy.core.UTCDateTime}.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fromString(self):
        """
        Tests initialization from a given time string.
        """
        # without trailing Z
        dt = UTCDateTime("1970-01-01T12:23:34")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34))
        dt = UTCDateTime("1970-01-01T12:23:34.5")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34, 500000))
        dt = UTCDateTime("1970-01-01T12:23:34.000005")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34, 5))
        dt = UTCDateTime("1969-12-31T23:43:19.900000")
        self.assertEquals(dt, UTCDateTime(1969, 12, 31, 23, 43, 19, 900000))
        # with trailing Z
        dt = UTCDateTime("1970-01-01T12:23:34Z")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34))
        dt = UTCDateTime("1970-01-01T12:23:34.5Z")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34, 500000))
        dt = UTCDateTime("1970-01-01T12:23:34.000005Z")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34, 5))
        dt = UTCDateTime("1969-12-31T23:43:19.900000Z")
        self.assertEquals(dt, UTCDateTime(1969, 12, 31, 23, 43, 19, 900000))

    def test_fromDateStringWithoutSeparator(self):
        """
        Tests initialization from a given time string.
        """
        dt = UTCDateTime("19700101122334")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34))
        dt = UTCDateTime("19700101122334.5")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34, 500000))
        dt = UTCDateTime("19700101122334.000005")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34, 5))
        dt = UTCDateTime("19691231234319.900000")
        self.assertEquals(dt, UTCDateTime(1969, 12, 31, 23, 43, 19, 900000))

    def test_fromOrdinalDateString(self):
        """
        Tests initialization from a given time string.
        """
        dt = UTCDateTime("1970,001,12:23:34")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34))
        dt = UTCDateTime("1970,001,12:23:34.5")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34, 500000))
        dt = UTCDateTime("1970,001,12:23:34.000005")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34, 5))
        dt = UTCDateTime("1969,365,23:43:19.900000")
        self.assertEquals(dt, UTCDateTime(1969, 12, 31, 23, 43, 19, 900000))

    def test_toString(self):
        """
        Tests __str__ method.
        """
        dt = UTCDateTime(1970, 1, 1, 12, 23, 34)
        self.assertEquals(str(dt), '1970-01-01T12:23:34.000000Z')
        dt = UTCDateTime(1970, 1, 1, 12, 23, 34, 500000)
        self.assertEquals(str(dt), '1970-01-01T12:23:34.500000Z')
        dt = UTCDateTime(1970, 1, 1, 12, 23, 34, 5)
        self.assertEquals(str(dt), '1970-01-01T12:23:34.000005Z')

    def test_deepcopy(self):
        dt = UTCDateTime(1240561632.0050001)
        dt2 = copy.deepcopy(dt)
        dt += 68
        self.assertEquals(dt2.timestamp, 1240561632.0050001)
        self.assertEquals(dt.timestamp, 1240561700.0050001)

    def test_add(self):
        a = UTCDateTime(0.0)
        self.assertEquals(a + 1, UTCDateTime(1970, 1, 1, 0, 0, 1))
        self.assertEquals(a + 1.123456,
                          UTCDateTime(1970, 1, 1, 0, 0, 1, 123456))
        self.assertEquals(a + 60 * 60 * 24 * 31 + 0.1,
                          UTCDateTime(1970, 2, 1, 0, 0, 0, 100000))
        self.assertEquals(a + -0.5,
                          UTCDateTime(1969, 12, 31, 23, 59, 59, 500000))
        self.assertEquals(UTCDateTime(0.5) + UTCDateTime(10.5),
                          11.0)
        td = datetime.timedelta(seconds=1)
        self.assertEquals(a + td, UTCDateTime(1970, 1, 1, 0, 0, 1))

    def test_sub(self):
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        end = UTCDateTime(2000, 1, 1, 0, 0, 4, 995000)
        self.assertAlmostEquals(end - start, 4.995)
        start = UTCDateTime(1000, 1, 1, 0, 0, 0, 0)
        end = UTCDateTime(1000, 1, 1, 0, 0, 4, 0)
        self.assertAlmostEquals(end - start, 4)
        start = UTCDateTime(0)
        end = UTCDateTime(-1000.5)
        self.assertAlmostEquals(end - start, -1000.5)
        td = datetime.timedelta(seconds=1)
        self.assertEquals(start - td, UTCDateTime(1969, 12, 31, 23, 59, 59))

    def test_negativeTimestamp(self):
        dt = UTCDateTime(-1000.1)
        self.assertEquals(str(dt), "1969-12-31T23:43:19.900000Z")
        self.assertEquals(dt.timestamp, -1000.1)
        dt = UTCDateTime(-1000.1)

    def test_initUTCDateTime(self):
        dt = UTCDateTime(year=2008, month=1, day=1)
        self.assertEquals(str(dt), "2008-01-01T00:00:00.000000Z")
        dt = UTCDateTime(year=2008, julday=1, hour=12, microsecond=5000)
        self.assertEquals(str(dt), "2008-01-01T12:00:00.005000Z")

    def test_toPythonDateTimeObjects(self):
        """
        Tests getDate, getTime, getTimestamp and getDateTime methods.
        """
        dt = UTCDateTime(1970, 1, 1, 12, 23, 34, 456789)
        # as function
        self.assertEquals(dt.getDate(), datetime.date(1970, 1, 1))
        self.assertEquals(dt.getTime(), datetime.time(12, 23, 34, 456789))
        self.assertEquals(dt.getDateTime(), datetime.datetime(1970, 1, 1, 12,
                                                              23, 34, 456789))
        self.assertAlmostEquals(dt.getTimeStamp(), 44614.456789)
        # as property
        self.assertEquals(dt.date, datetime.date(1970, 1, 1))
        self.assertEquals(dt.time, datetime.time(12, 23, 34, 456789))
        self.assertEquals(dt.datetime, datetime.datetime(1970, 1, 1, 12, 23,
                                                         34, 456789))
        self.assertAlmostEquals(dt.timestamp, 44614.456789)


def suite():
    return unittest.makeSuite(UTCDateTimeTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
