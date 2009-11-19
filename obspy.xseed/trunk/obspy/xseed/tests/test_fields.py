# -*- coding: utf-8 -*-

from StringIO import StringIO
from obspy.core import UTCDateTime
from obspy.xseed.fields import Float, VariableString
import unittest


class FieldsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_formatExponential(self):
        field = Float(1, "test", 12, mask='%+1.5e', strict=True)
        self.assertEquals(field.write('2.5'), '+2.50000E+00')

    def test_dateTimeConversion(self):
        field = VariableString(1, "test", 1, 22, 'T', strict=True)
        #1
        orig = '1992,002,00:00:00.0000~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, UTCDateTime('1992-01-02T00:00:00'))
        self.assertEquals(field.write(dt), '1992,002~')
        #2
        orig = '1992,005,01:02:03.4567~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, UTCDateTime('1992-01-05T01:02:03.456700'))
        self.assertEquals(field.write(dt), orig)
        #3
        orig = '1992,005,01:02:03.0001~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, UTCDateTime('1992-01-05T01:02:03.000100'))
        self.assertEquals(field.write(dt), orig)
        #4
        orig = '1992,005,01:02:03.1000~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, UTCDateTime('1992-01-05T01:02:03.100000'))
        self.assertEquals(field.write(dt), orig)
        #5
        orig = '1987,023,04:23:05.1~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, UTCDateTime('1987-01-23T04:23:05.100000'))
        self.assertEquals(field.write(dt), '1987,023,04:23:05.1000~')
        #6
        orig = '1987,023,04:23:05.123~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, UTCDateTime('1987-01-23T04:23:05.123000'))
        self.assertEquals(field.write(dt), '1987,023,04:23:05.1230~')

    def test_dateConversion(self):
        field = VariableString(1, "test", 1, 22, 'T', strict=True,
                               compact=True)
        #1
        orig = '1992,002~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, UTCDateTime('1992-01-02'))
        self.assertEquals(field.write(dt), orig)
        #2
        orig = '2007,199~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, UTCDateTime('2007-07-18'))
        self.assertEquals(field.write(dt), orig)
        #3 - bad syntax
        #orig = '1992,2~'
        #dt = field.read(StringIO(orig))
        #self.assertEquals(dt, UTCDateTime('1992-01-02'))
        #self.assertEquals(field.write(dt), '1992,002~')
        #4 - wrong syntax
        orig = '1992~'
        self.assertRaises(Exception, field.read, StringIO(orig))
        #5 - wrong syntax
        orig = '1992'
        self.assertRaises(Exception, field.read, StringIO(orig))
        #6
        orig = '~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, '')
        self.assertEquals(field.write(dt), '~')
        #7 - bad syntax
        orig = ''
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, '')
        self.assertEquals(field.write(dt), '~')
        #8 - bad syntax
        orig = '2007,199'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, UTCDateTime('2007-07-18'))
        self.assertEquals(field.write(dt), '2007,199~')
        #9
        orig = '2009,074,12~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, UTCDateTime(2009, 3, 15, 12))
        self.assertEquals(field.write(dt), orig)
        #10
        orig = '2009,074,12:20~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, UTCDateTime(2009, 3, 15, 12, 20))
        self.assertEquals(field.write(dt), orig)


def suite():
    return unittest.makeSuite(FieldsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
