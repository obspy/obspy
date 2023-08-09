# -*- coding: utf-8 -*-
import io
import string
import unittest

from obspy import UTCDateTime
from obspy.io.xseed.fields import Float, VariableString, FixedString


class FieldsTestCase(unittest.TestCase):
    """
    Fields test suite.
    """
    def test_format_exponential(self):
        field = Float(1, "test", 12, mask='%+1.5e', strict=True)
        self.assertEqual(field.write('2.5'), b'+2.50000E+00')

    def test_read_date_time(self):
        field = VariableString(1, "test", 1, 22, 'T', strict=True)
        # 1
        orig = b'1992,002,00:00:00.0000~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(1992, 1, 2))
        self.assertEqual(field.write(dt), b'1992,002~')
        # 1
        orig = b'1992,002~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(1992, 1, 2))
        self.assertEqual(field.write(dt), b'1992,002~')
        # 2
        orig = b'1992,005,01:02:03.4567~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(1992, 1, 5, 1, 2, 3, 456700))
        self.assertEqual(field.write(dt), orig)
        # 3
        orig = b'1992,005,01:02:03.0001~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(1992, 1, 5, 1, 2, 3, 100))
        self.assertEqual(field.write(dt), orig)
        # 4
        orig = b'1992,005,01:02:03.1000~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(1992, 1, 5, 1, 2, 3, 100000))
        self.assertEqual(field.write(dt), orig)
        # 5
        orig = b'1987,023,04:23:05.1~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(1987, 1, 23, 4, 23, 5, 100000))
        self.assertEqual(field.write(dt), b'1987,023,04:23:05.1000~')
        # 6
        orig = b'1987,023,04:23:05.123~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(1987, 1, 23, 4, 23, 5, 123000))
        self.assertEqual(field.write(dt), b'1987,023,04:23:05.1230~')
        #
        orig = b'2008,358,01:30:22.0987~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(2008, 12, 23, 1, 30, 22, 98700))
        self.assertEqual(field.write(dt), orig)
        #
        orig = b'2008,358,01:30:22.9876~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(2008, 12, 23, 1, 30, 22, 987600))
        self.assertEqual(field.write(dt), orig)
        #
        orig = b'2008,358,01:30:22.0005~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(2008, 12, 23, 1, 30, 22, 500))
        self.assertEqual(field.write(dt), orig)
        #
        orig = b'2008,358,01:30:22.0000~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(2008, 12, 23, 1, 30, 22, 0))
        self.assertEqual(field.write(dt), orig)

    def test_read_compact_date_time(self):
        field = VariableString(1, "test", 0, 22, 'T', strict=True,
                               compact=True)
        # 1
        orig = b'1992,002~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(1992, 1, 2))
        self.assertEqual(field.write(dt), orig)
        # 2
        orig = b'2007,199~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(2007, 7, 18))
        self.assertEqual(field.write(dt), orig)
        # 3 - wrong syntax
        orig = b'1992'
        self.assertRaises(Exception, field.read, io.BytesIO(orig))
        orig = b'1992,'
        self.assertRaises(Exception, field.read, io.BytesIO(orig))
        orig = b'1992~'
        self.assertRaises(Exception, field.read, io.BytesIO(orig))
        orig = b'1992,~'
        self.assertRaises(Exception, field.read, io.BytesIO(orig))
        # 5 - empty datetime
        orig = b'~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, '')
        self.assertEqual(field.write(dt), b'~')
        # 6 - bad syntax
        orig = b''
        self.assertRaises(Exception, field.read, io.BytesIO(orig))
        self.assertEqual(field.write(dt), b'~')
        # 7
        orig = b'2007,199~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(2007, 7, 18))
        self.assertEqual(field.write(dt), b'2007,199~')
        # 8
        orig = b'2009,074,12~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(2009, 3, 15, 12))
        self.assertEqual(field.write(dt), orig)
        # 9
        orig = b'2008,358,01:30:22.0012~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(2008, 12, 23, 1, 30, 22, 1200))
        self.assertEqual(field.write(dt), orig)
        #
        orig = b'2008,358,00:00:22~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(2008, 12, 23, 0, 0, 22, 0))
        self.assertEqual(field.write(dt), orig)
        #
        orig = b'2008,358,00:30~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(2008, 12, 23, 0, 30, 0, 0))
        self.assertEqual(field.write(dt), orig)
        #
        orig = b'2008,358,01~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(2008, 12, 23, 1, 0, 0, 0))
        self.assertEqual(field.write(dt), orig)
        #
        orig = b'2008,358~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(2008, 12, 23, 0, 0, 0, 0))
        self.assertEqual(field.write(dt), orig)
        #
        orig = b'2008,358,01:30:22.5~'
        dt = field.read(io.BytesIO(orig))
        self.assertEqual(dt, UTCDateTime(2008, 12, 23, 1, 30, 22, 500000))
        self.assertEqual(field.write(dt), b'2008,358,01:30:22.5000~')

    def test_wrong_spaces(self):
        """
        VariableString(..., 'S') should remove leading or trailing spaces.
        """
        field = VariableString(1, "test", 1, 60, 'UNLPS', strict=True)
        orig = b' T3930_b A6689 3930 ~'
        result = field.read(io.BytesIO(orig))
        self.assertEqual(result, 'T3930_b A6689 3930')

    def test_fixed_string_any_punctuation(self):
        """
        FixedString(..., 'P') should allow any punctuation character.
        """
        field = FixedString(1, "test", 9, 'UP')
        for character in string.punctuation:
            orig = b'ABC' + character.encode("utf-8") + b'DEF'
            result = field.read(io.BytesIO(orig))
            self.assertEqual(result, 'ABC' + character + 'DEF')

    def test_variable_string_any_punctuation(self):
        """
        VariableString(..., 'P') should allow any punctuation character.
        """
        field = VariableString(1, "test", 1, 30, 'UP')
        for character in string.punctuation:
            # need to skip tilde as it is also the end marker of a variable
            # field
            if character == '~':
                continue
            orig = b'ABC' + character.encode("utf-8") + b'DEF~'
            result = field.read(io.BytesIO(orig))
            self.assertEqual(result, 'ABC' + character + 'DEF')
