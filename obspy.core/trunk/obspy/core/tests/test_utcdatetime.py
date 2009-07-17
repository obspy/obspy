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
        dt += 1
        self.assertEquals(dt2.timestamp, 1240561632.0050001)
        self.assertEquals(dt.timestamp, 1240561633.0050001)


def suite():
    return unittest.makeSuite(UTCDateTimeTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
