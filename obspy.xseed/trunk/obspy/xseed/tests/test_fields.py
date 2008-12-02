# -*- coding: utf-8 -*-

from StringIO import StringIO
from obspy.xseed.fields import Float, VariableString
import datetime
import unittest


class FieldsTestCase(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_formatExponential(self):
        field = Float(1, "test", 12, mask='%+1.5e')
        self.assertEquals(field.write('2.5'), '+2.50000E+00')
    
    def test_dateTimeConversion(self):
        field = VariableString(1, "test", 1, 22, 'T')
        orig = '1992,002,00:00:00.0000~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, '1992-01-02T00:00:00')
        self.assertEquals(field.write(dt), orig)
        orig = '1992,005,01:02:03.4567~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, '1992-01-05T01:02:03.456700')
        self.assertEquals(field.write(dt), orig)
        orig = '1992,005,01:02:03.0001~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, '1992-01-05T01:02:03.000100')
        self.assertEquals(field.write(dt), orig)
        orig = '1992,005,01:02:03.1000~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, '1992-01-05T01:02:03.100000')
        self.assertEquals(field.write(dt), orig)
    
    def test_dateConversion(self):
        field = VariableString(1, "test", 1, 22, 'T')
        orig = '1992,002~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, '1992-01-02')
        self.assertEquals(field.write(dt), orig)
        orig = '2007,199~'
        dt = field.read(StringIO(orig))
        self.assertEquals(dt, '2007-07-18')
        self.assertEquals(field.write(dt), orig)


def suite():
    return unittest.makeSuite(FieldsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
