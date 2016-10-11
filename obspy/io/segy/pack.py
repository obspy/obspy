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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import sys

import numpy as np


LOG2 = 0.3010299956639812
# Get the system byte order.
BYTEORDER = sys.byteorder
if BYTEORDER == 'little':
    BYTEORDER = '<'
else:
    BYTEORDER = '>'


class WrongDtypeException(Exception):
    pass


def pack_4byte_ibm(file, data, endian='>'):
    """
    Packs 4 byte IBM floating points. This will only work if the host system
    internally uses little endian byte orders.
    """
    # Check the dtype and raise exception otherwise!
    if data.dtype != np.float64 and data.dtype != np.float32:
        raise WrongDtypeException
    # Calculate the values. The theory is explained in
    # https://www.codeproject.com/KB/applications/libnumber.aspx

    # Calculate the signs.
    signs = np.empty(len(data), dtype=np.uint8)
    temp_signs = np.sign(data)
    # Negative numbers are encoded as sign bit 1, positive ones as bit 0.
    signs[temp_signs == 1] = 0
    signs[temp_signs == -1] = 128

    # Make absolute values.
    data = np.abs(data)

    # Store the zeros and add an offset for numerical stability,
    # they will be set to zero later on again
    zeros = np.where(data == 0.0)
    data[zeros] += 1e-32

    # Calculate the exponent for the IBM data format.
    exponent = ((np.log10(data) / LOG2) * 0.25 + 65).astype(np.uint32)

    # Now calculate the fraction using single precision.
    fraction = np.require(
        data, np.float32) / (16.0 ** (np.require(exponent, np.float32) - 64))

    # Normalization.
    while True:
        # Find numbers smaller than 1/16 but not zero.
        non_normalized = np.where(np.where(fraction, fraction, 1) < 0.0625)[0]
        if len(non_normalized) == 0:
            break
        fraction[non_normalized] *= 16
        exponent[non_normalized] -= 1

    # If the fraction is one, change it to 1/16 and increase the exponent by
    # one.
    ones = np.where(fraction == 1.0)
    fraction[ones] = 0.0625
    exponent[ones] += 1

    # Times 2^24 to be able to get a long.
    fraction *= 16777216.0
    # Convert to unsigned long.
    fraction = np.require(fraction, np.uint64)

    # Use 8 bit integers to be able to store every byte separately.
    new_data = np.zeros(4 * len(data), np.uint8)

    # The first bit is the sign and the following 7 are the exponent.
    byte_0 = np.require(signs + exponent, np.uint8)
    # All following 24 bit are the fraction.
    byte_1 = np.require(np.right_shift(np.bitwise_and(fraction, 0x00ff0000),
                                       16), np.uint8)
    byte_2 = np.require(np.right_shift(np.bitwise_and(fraction, 0x0000ff00),
                                       8), np.uint8)
    byte_3 = np.require(np.bitwise_and(fraction, 0x000000ff), np.uint8)

    # Depending on the endianness store the data different.
    # big endian.
    if endian == '>':
        new_data[0::4] = byte_0
        new_data[1::4] = byte_1
        new_data[2::4] = byte_2
        new_data[3::4] = byte_3
    # little endian>
    elif endian == '<':
        new_data[0::4] = byte_3
        new_data[1::4] = byte_2
        new_data[2::4] = byte_1
        new_data[3::4] = byte_0
    # Should not happen.
    else:
        raise Exception
    # Write the zeros again.
    new_data.dtype = np.uint32
    new_data[zeros] = 0
    # Write to file.
    file.write(new_data.tostring())


def pack_4byte_integer(file, data, endian='>'):
    """
    Packs 4 byte integers.
    """
    # Check the dtype and raise exception otherwise!
    if data.dtype != np.int32:
        raise WrongDtypeException
    # Swap the byte order if necessary.
    if BYTEORDER != endian:
        data = data.byteswap()
    # Write the file.
    file.write(data.tostring())


def pack_2byte_integer(file, data, endian='>'):
    """
    Packs 2 byte integers.
    """
    # Check the dtype and raise exception otherwise!
    if data.dtype != np.int16:
        raise WrongDtypeException
    # Swap the byte order if necessary.
    if BYTEORDER != endian:
        data = data.byteswap()
    # Write the file.
    file.write(data.tostring())


def pack_4byte_fixed_point(file, data, endian='>'):
    raise NotImplementedError


def pack_4byte_ieee(file, data, endian='>'):
    """
    Packs 4 byte IEEE floating points.
    """
    # Check the dtype and raise exception otherwise!
    if data.dtype != np.float32:
        raise WrongDtypeException
    # Swap the byte order if necessary.
    if BYTEORDER != endian:
        data = data.byteswap()
    # Write the file.
    file.write(data.tostring())


def pack_1byte_integer(file, data, endian='>'):
    raise NotImplementedError
