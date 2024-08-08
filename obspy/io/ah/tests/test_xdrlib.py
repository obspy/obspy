"""
Changes:
    * Changed import line

PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
--------------------------------------------

1. This LICENSE AGREEMENT is between the Python Software Foundation
("PSF"), and the Individual or Organization ("Licensee") accessing and
otherwise using this software ("Python") in source or binary form and
its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF hereby
grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
analyze, test, perform and/or display publicly, prepare derivative works,
distribute, and otherwise use Python alone or in any derivative version,
provided, however, that PSF's License Agreement and PSF's notice of copyright,
i.e., "Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022
Python Software Foundation;
All Rights Reserved" are retained in Python alone or in any derivative version
prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on
or incorporates Python or any part thereof, and wants to make
the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to Python.

4. PSF is making Python available to Licensee on an "AS IS"
basis.  PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON WILL NOT
INFRINGE ANY THIRD PARTY RIGHTS.

5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON
FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS
A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON,
OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between PSF and
Licensee.  This License Agreement does not grant permission to use PSF
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using Python, Licensee
agrees to be bound by the terms and conditions of this License
Agreement.
"""

from obspy.io.ah import xdrlib
import pytest


class TestXDR():

    def test_xdr(self):
        p = xdrlib.Packer()

        s = b'hello world'
        a = [b'what', b'is', b'hapnin', b'doctor']

        p.pack_int(42)
        p.pack_int(-17)
        p.pack_uint(9)
        p.pack_bool(True)
        p.pack_bool(False)
        p.pack_uhyper(45)
        p.pack_float(1.9)
        p.pack_double(1.9)
        p.pack_string(s)
        p.pack_list(range(5), p.pack_uint)
        p.pack_array(a, p.pack_string)

        # now verify
        data = p.get_buffer()
        up = xdrlib.Unpacker(data)

        assert up.get_position() == 0

        assert up.unpack_int() == 42
        assert up.unpack_int() == -17
        assert up.unpack_uint() == 9
        assert up.unpack_bool() is True

        # remember position
        pos = up.get_position()
        assert up.unpack_bool() is False

        # rewind and unpack again
        up.set_position(pos)
        assert up.unpack_bool() is False

        assert up.unpack_uhyper() == 45
        assert round(abs(up.unpack_float()-1.9), 7) == 0
        assert round(abs(up.unpack_double()-1.9), 7) == 0
        assert up.unpack_string() == s
        assert up.unpack_list(up.unpack_uint) == list(range(5))
        assert up.unpack_array(up.unpack_string) == a
        up.done()
        with pytest.raises(EOFError):
            up.unpack_uint()


class TestConversionError():

    def setup_method(self):
        self.packer = xdrlib.Packer()

    def assertRaisesConversion(self, *args):
        with pytest.raises(xdrlib.ConversionError):
            args[0](*args[1:])

    def test_pack_int(self):
        self.assertRaisesConversion(self.packer.pack_int, 'string')

    def test_pack_uint(self):
        self.assertRaisesConversion(self.packer.pack_uint, 'string')

    def test_float(self):
        self.assertRaisesConversion(self.packer.pack_float, 'string')

    def test_double(self):
        self.assertRaisesConversion(self.packer.pack_double, 'string')

    def test_uhyper(self):
        self.assertRaisesConversion(self.packer.pack_uhyper, 'string')
