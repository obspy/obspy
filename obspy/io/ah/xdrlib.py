"""Implements (a subset of) Sun XDR -- eXternal Data Representation.
See: RFC 1014

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
2010,2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022
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

import struct
from io import BytesIO
from functools import wraps


__all__ = ["Error", "Packer", "Unpacker", "ConversionError"]

# exceptions


class Error(Exception):
    """Exception class for this module. Use:
    except xdrlib.Error as var:
        # var has the Error instance for the exception
    Public ivars:
        msg -- contains the message
    """

    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return repr(self.msg)

    def __str__(self):
        return str(self.msg)


class ConversionError(Error):
    pass


def raise_conversion_error(function):
    """ Wrap any raised struct.errors in a ConversionError. """

    @wraps(function)
    def result(self, value):
        try:
            return function(self, value)
        except struct.error as e:
            raise ConversionError(e.args[0]) from None
    return result


class Packer:
    """Pack various data representations into a buffer."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.__buf = BytesIO()

    def get_buffer(self):
        return self.__buf.getvalue()
    # backwards compatibility
    get_buf = get_buffer

    @raise_conversion_error
    def pack_uint(self, x):
        self.__buf.write(struct.pack('>L', x))

    @raise_conversion_error
    def pack_int(self, x):
        self.__buf.write(struct.pack('>l', x))

    pack_enum = pack_int

    def pack_bool(self, x):
        if x:
            self.__buf.write(b'\0\0\0\1')
        else:
            self.__buf.write(b'\0\0\0\0')

    def pack_uhyper(self, x):
        try:
            self.pack_uint(x >> 32 & 0xffffffff)
        except (TypeError, struct.error) as e:
            raise ConversionError(e.args[0]) from None
        try:
            self.pack_uint(x & 0xffffffff)
        except (TypeError, struct.error) as e:
            raise ConversionError(e.args[0]) from None

    pack_hyper = pack_uhyper

    @raise_conversion_error
    def pack_float(self, x):
        self.__buf.write(struct.pack('>f', x))

    @raise_conversion_error
    def pack_double(self, x):
        self.__buf.write(struct.pack('>d', x))

    def pack_fstring(self, n, s):
        if n < 0:
            raise ValueError('fstring size must be nonnegative')
        data = s[:n]
        n = ((n+3)//4)*4
        data = data + (n - len(data)) * b'\0'
        self.__buf.write(data)

    pack_fopaque = pack_fstring

    def pack_string(self, s):
        n = len(s)
        self.pack_uint(n)
        self.pack_fstring(n, s)

    pack_opaque = pack_string
    pack_bytes = pack_string

    def pack_list(self, list, pack_item):
        for item in list:
            self.pack_uint(1)
            pack_item(item)
        self.pack_uint(0)

    def pack_farray(self, n, list, pack_item):
        if len(list) != n:
            raise ValueError('wrong array size')
        for item in list:
            pack_item(item)

    def pack_array(self, list, pack_item):
        n = len(list)
        self.pack_uint(n)
        self.pack_farray(n, list, pack_item)


class Unpacker:
    """Unpacks various data representations from the given buffer."""

    def __init__(self, data):
        self.reset(data)

    def reset(self, data):
        self.__buf = data
        self.__pos = 0

    def get_position(self):
        return self.__pos

    def set_position(self, position):
        self.__pos = position

    def get_buffer(self):
        return self.__buf

    def done(self):
        if self.__pos < len(self.__buf):
            raise Error('unextracted data remains')

    def unpack_uint(self):
        i = self.__pos
        self.__pos = j = i+4
        data = self.__buf[i:j]
        if len(data) < 4:
            raise EOFError
        return struct.unpack('>L', data)[0]

    def unpack_int(self):
        i = self.__pos
        self.__pos = j = i+4
        data = self.__buf[i:j]
        if len(data) < 4:
            raise EOFError
        return struct.unpack('>l', data)[0]

    unpack_enum = unpack_int

    def unpack_bool(self):
        return bool(self.unpack_int())

    def unpack_uhyper(self):
        hi = self.unpack_uint()
        lo = self.unpack_uint()
        return int(hi) << 32 | lo

    def unpack_hyper(self):
        x = self.unpack_uhyper()
        if x >= 0x8000000000000000:
            x = x - 0x10000000000000000
        return x

    def unpack_float(self):
        i = self.__pos
        self.__pos = j = i+4
        data = self.__buf[i:j]
        if len(data) < 4:
            raise EOFError
        return struct.unpack('>f', data)[0]

    def unpack_double(self):
        i = self.__pos
        self.__pos = j = i+8
        data = self.__buf[i:j]
        if len(data) < 8:
            raise EOFError
        return struct.unpack('>d', data)[0]

    def unpack_fstring(self, n):
        if n < 0:
            raise ValueError('fstring size must be nonnegative')
        i = self.__pos
        j = i + (n+3)//4*4
        if j > len(self.__buf):
            raise EOFError
        self.__pos = j
        return self.__buf[i:i+n]

    unpack_fopaque = unpack_fstring

    def unpack_string(self):
        n = self.unpack_uint()
        return self.unpack_fstring(n)

    unpack_opaque = unpack_string
    unpack_bytes = unpack_string

    def unpack_list(self, unpack_item):
        list = []
        while 1:
            x = self.unpack_uint()
            if x == 0:
                break
            if x != 1:
                raise ConversionError('0 or 1 expected, got %r' % (x,))
            item = unpack_item()
            list.append(item)
        return list

    def unpack_farray(self, n, unpack_item):
        list = []
        for i in range(n):
            list.append(unpack_item())
        return list

    def unpack_array(self, unpack_item):
        n = self.unpack_uint()
        return self.unpack_farray(n, unpack_item)
