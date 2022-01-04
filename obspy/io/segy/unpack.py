# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
#  Filename: unpack.py
#  Purpose: Routines for unpacking SEG Y data formats.
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#
# Copyright (C) 2010 Lion Krischer
# --------------------------------------------------------------------
"""
Functions that will all take a file pointer and the sample count and return a
NumPy array with the unpacked values.
"""
import ctypes as C  # NOQA
from pathlib import Path
import sys
import warnings

import numpy as np

from obspy.core.compatibility import from_buffer
from .util import clibsegy


# Get the system byte order.
BYTEORDER = sys.byteorder
if BYTEORDER == 'little':
    BYTEORDER = '<'
else:
    BYTEORDER = '>'


clibsegy.ibm2ieee.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                           flags='C_CONTIGUOUS'),
    C.c_int]
clibsegy.ibm2ieee.restype = C.c_void_p


def unpack_4byte_ibm(file, count, endian='>'):
    """
    Unpacks 4 byte IBM floating points.
    """
    # Read as 4 byte integer so bit shifting works.
    data = from_buffer(file.read(count * 4), dtype=np.float32)
    # Swap the byte order if necessary.
    if BYTEORDER != endian:
        data = data.byteswap()
    length = len(data)
    # Call the C code which transforms the data inplace.
    clibsegy.ibm2ieee(data, length)
    return data


# Old pure Python/NumPy code
#
# def unpack_4byte_ibm(file, count, endian='>'):
#    """
#    Unpacks 4 byte IBM floating points.
#    """
#    # Read as 4 byte integer so bit shifting works.
#    data = np.fromstring(file.read(count * 4), dtype=np.int32)
#    # Swap the byte order if necessary.
#    if BYTEORDER != endian:
#        data = data.byteswap()
#    # See https://mail.scipy.org/pipermail/scipy-user/2009-January/019392.html
#    # XXX: Might need check for values out of range:
#    # https://bytes.com/topic/c/answers/
#    #         221981-c-code-converting-ibm-370-floating-point-ieee-754-a
#    sign = np.bitwise_and(np.right_shift(data, 31), 0x01)
#    exponent = np.bitwise_and(np.right_shift(data, 24), 0x7f)
#    mantissa = np.bitwise_and(data, 0x00ffffff)
#    # Force single precision.
#    mantissa = np.require(mantissa, 'float32')
#    mantissa /= 0x1000000
#    # Do the following calculation in a weird way to avoid autocasting to
#    # float64.
#    # data = (1.0 - 2.0 * sign) * mantissa * 16.0 ** (exponent - 64.0)
#    sign *= -2.0
#    sign += 1.0
#    mantissa *= 16.0 ** (exponent - 64)
#    mantissa *= sign
#    return mantissa


def unpack_4byte_integer(file, count, endian='>'):
    """
    Unpacks 4 byte integers.
    """
    # Read as 4 byte integer so bit shifting works.
    data = from_buffer(file.read(count * 4), dtype=np.int32)
    # Swap the byte order if necessary.
    if BYTEORDER != endian:
        data = data.byteswap()
    return data


def unpack_2byte_integer(file, count, endian='>'):
    """
    Unpacks 2 byte integers.
    """
    # Read as 4 byte integer so bit shifting works.
    data = from_buffer(file.read(count * 2), dtype=np.int16)
    # Swap the byte order if necessary.
    if BYTEORDER != endian:
        data = data.byteswap()
    return data


def unpack_4byte_fixed_point(file, count, endian='>'):
    raise NotImplementedError


def unpack_4byte_ieee(file, count, endian='>'):
    """
    Unpacks 4 byte IEEE floating points.
    """
    # Read as 4 byte integer so bit shifting works.
    data = from_buffer(file.read(count * 4), dtype=np.float32)
    # Swap the byte order if necessary.
    if BYTEORDER != endian:
        data = data.byteswap()
    return data


def unpack_1byte_integer(file, count, endian='>'):
    raise NotImplementedError


class OnTheFlyDataUnpacker:
    """
    Tie-up a data sample unpack function with its parameters.

    This class allows for data to be read directly from the disk as needed,
    preventing the need to store data in memory.
    """
    def __init__(self, unpack_function, filename, filemode, seek, count,
                 endian='>'):
        self.unpack_function = unpack_function
        self.filename = filename
        self.filemode = filemode
        self.seek = seek
        self.count = count
        self.endian = endian
        self.mtime = Path(self.filename).stat().st_mtime

    def __call__(self):
        mtime = Path(self.filename).stat().st_mtime
        if mtime != self.mtime:
            msg = "File '%s' changed since reading headers" % self.filename
            msg += "; data may be read incorrectly "
            msg += "(modification time = %s)." % mtime
            warnings.warn(msg)
        with open(self.filename, self.filemode) as fp:
            fp.seek(self.seek)
            raw = self.unpack_function(fp, self.count, endian=self.endian)
        return raw
