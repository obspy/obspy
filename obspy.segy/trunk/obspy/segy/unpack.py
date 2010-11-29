# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
#  Filename: unpack.py
#  Purpose: Routines for unpacking SEG Y data formats.
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#
# Copyright (C) 2010 Lion Krischer
#---------------------------------------------------------------------
"""
Functions that will all take a file pointer and the sample count and return a
numpy array with the unpacked values.
"""

import numpy as np
import sys

BYTEORDER = sys.byteorder

def unpack_4byte_IBM(file, count, endian='>'):
    """
    Unpacks 4 byte IBM floating points. Will return data in double precission
    to minimize rounding errors.
    """
    # Read as 4 byte integer so bit shifting works.
    data = np.fromstring(file.read(count * 4), dtype='int32')
    # If the native byte order is little endian swap it if big endian is
    # wanted.
    if BYTEORDER == 'little' and endian == '>':
        data = data.byteswap()
    # Same the other way around.
    if BYTEORDER == 'big' and endian == '<':
        data = data.byteswap()
    # See http://mail.scipy.org/pipermail/scipy-user/2009-January/019392.html
    # XXX: Might need check for values out of range:
    # http://bytes.com/topic/c/answers/221981-c-code-converting-ibm-370-floating-point-ieee-754-a
    sign = np.bitwise_and(np.right_shift(data, 31),  0x01)
    exponent = np.bitwise_and(np.right_shift(data, 24), 0x7f)
    mantissa = np.bitwise_and(data, 0x00ffffff)
    # Force double precission.
    mantissa = np.require(mantissa, 'float64')
    mantissa /= 0x1000000
    # This should now also be double precission.
    data = (1.0 - 2.0 * sign) * mantissa * 16.0 ** (exponent - 64.0)
    return data

def unpack_4byte_Integer(file, count, endian='>'):
    """
    Unpacks 4 byte integers.
    """
    # Read as 4 byte integer so bit shifting works.
    data = np.fromstring(file.read(count * 4), dtype='int32')
    # If the native byte order is little endian swap it if big endian is
    # wanted.
    if BYTEORDER == 'little' and endian == '>':
        data = data.byteswap()
    # Same the other way around.
    if BYTEORDER == 'big' and endian == '<':
        data = data.byteswap()
    return data

def unpack_2byte_Integer(file, count, endian='>'):
    """
    Unpacks 2 byte integers.
    """
    # Read as 4 byte integer so bit shifting works.
    data = np.fromstring(file.read(count * 2), dtype='int16')
    # If the native byte order is little endian swap it if big endian is
    # wanted.
    if BYTEORDER == 'little' and endian == '>':
        data = data.byteswap()
    # Same the other way around.
    if BYTEORDER == 'big' and endian == '<':
        data = data.byteswap()
    return data

def unpack_4byte_Fixed_point(file, count, endian='>'):
    raise NotImplementedError

def unpack_4byte_IEEE(file, count, endian='>'):
    raise NotImplementedError

def unpack_1byte_Integer(file, count, endian='>'):
    raise NotImplementedError
