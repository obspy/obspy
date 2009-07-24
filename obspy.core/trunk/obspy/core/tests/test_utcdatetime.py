# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime
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
        dt = UTCDateTime("1970-01-01T12:23:34")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34))
        dt = UTCDateTime("1970-01-01T12:23:34.5")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34, 500000))
        dt = UTCDateTime("1970-01-01T12:23:34.000005")
        self.assertEquals(dt, UTCDateTime(1970, 1, 1, 12, 23, 34, 5))
        dt = UTCDateTime("1969-12-31T23:43:19.900000")
        self.assertEquals(dt, UTCDateTime(1969, 12, 31, 23, 43, 19, 900000))

    def test_toString(self):
        """
        Tests __str__ method.
        """
        dt = UTCDateTime(1970, 1, 1, 12, 23, 34)
        self.assertEquals(str(dt), '1970-01-01T12:23:34.000000')
        dt = UTCDateTime(1970, 1, 1, 12, 23, 34, 500000)
        self.assertEquals(str(dt), '1970-01-01T12:23:34.500000')
        dt = UTCDateTime(1970, 1, 1, 12, 23, 34, 5)
        self.assertEquals(str(dt), '1970-01-01T12:23:34.000005')

    def test_deepcopy(self):
        dt = UTCDateTime(1240561632.0050001)
        dt2 = copy.deepcopy(dt)
        dt += 68
        self.assertEquals(dt2.timestamp, 1240561632.0050001)
        self.assertEquals(dt.timestamp, 1240561700.0050001)

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

    def test_negativeTimestamp(self):
        dt = UTCDateTime(-1000.1)
        self.assertEquals(str(dt), "1969-12-31T23:43:19.900000")
        self.assertEquals(dt.timestamp, -1000.1)
        dt = UTCDateTime(-1000.1)


def suite():
    return unittest.makeSuite(UTCDateTimeTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
