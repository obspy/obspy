from codecs import encode

import decorator
import numpy as np


@decorator.decorator
def _open_file(func, *args, **kwargs):
    """
    Decorator to ensure a file buffer is passed as first argument to the
    decorated function.
    :param func:
        callable that takes at least one argument; the first argument must
        be treated as a buffer.
    :return: callable
    """
    first_arg = args[0]
    try:
        with open(first_arg, 'rb') as fi:
            args = tuple([fi] + list(args[1:]))
            return func(*args, **kwargs)
    except TypeError:  # assume we have been passed a buffer
        if not hasattr(args[0], 'read'):
            raise  # type error was in function call, not in opening file
        out = func(*args, **kwargs)
        first_arg.seek(0)  # reset position to start of file
    return out


def _read(fi, position, length, dtype, left_part=True):
    """
    Read one or more bytes using provided datatype.
    :param fi: A buffer containing the bytes to read.
    :param position: Byte position to start reading.
    :type position: int
    :param length: Length, in bytes, of data to read.
    :type length: int or float
    :param dtype:
        - bcd
        - binary
        - IEEE
    :type dtype: str
    :param left_part: If True, start the reading from the first half part
                      of the byte position.
                      If False, start the reading from the second half part
                      of the byte position.
    :type left_part: boolean
    """
    fi.seek(position)
    if dtype == 'bcd':
        return _read_bcd(fi, length, left_part)
    if dtype == 'binary':
        return _read_binary(fi, length, left_part)
    if dtype == 'IEEE':
        data = np.frombuffer(fi.read(int(length)), '>f4')
        return data[0] if len(data) == 1 else data


def _read_bcd(fi, length, left_part):
    """
    Interprets a byte string as binary coded decimals. See:
    https://en.wikipedia.org/wiki/Binary-coded_decimal#Basics
    :param fi: A buffer containing the bytes to read.
    :param length: number of bytes to read.
    :type length: int or float
    :param left_part:  If True, start the reading from the first half part
                       of the first byte. If False, start the reading from
                       the second half part of the first byte.
    :type left_part: boolean
    """
    tens = np.power(10, range(12))[::-1]
    nbr_half_bytes = round(2*length)
    if isinstance(length, float):
        length = int(length) + 1
    byte_values = fi.read(length)
    ints = np.frombuffer(byte_values, dtype='<u1', count=length)
    if left_part is True:
        unpack_bits = np.unpackbits(ints).reshape(-1, 4)[0:nbr_half_bytes]
    else:
        unpack_bits = np.unpackbits(ints).reshape(-1, 4)[1:nbr_half_bytes+1]
    bits = np.dot(unpack_bits, np.array([1, 2, 4, 8])[::-1].reshape(4, 1))
    if np.any(bits > 9):
        raise ValueError('invalid bcd values encountered')
    return np.dot(tens[-len(bits):], bits)[0]


def _read_binary(fi, length, left_part):
    """
    Read raw bytes and convert them in integer
    :param fi: A buffer containing the bytes to read.
    :param length: number of bytes to read.
    :type length: int or float
    :param left_part:  If True, start the reading from the first half part
                       of the byte.
    :type left_part: boolean
    """
    if isinstance(length, float):
        if np.isclose(length, 0.5):
            ints = np.frombuffer(fi.read(1), dtype='<u1')[0]
            if left_part is True:
                return np.bitwise_and(ints >> 4, 0x0f)
            else:
                return np.bitwise_and(ints, 0x0f)
        else:
            raise ValueError('invalid length of bytes to read.\
                             It has to be an integer or 0.5')
    else:
        return int(encode(fi.read(length), 'hex'), 16)
