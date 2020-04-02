# -*- coding: utf-8 -*-
import codecs

import numpy as np

from obspy import UTCDateTime
from obspy.core.compatibility import from_buffer


def bcd(_i):
    return (_i >> 4 & 0xF).astype(np.uint32) * 10 + (_i & 0xF)


def bcd_16bit_int(_i):
    _i = bcd(_i)
    return _i[::, 0] * 100 + _i[::, 1]


def bcd_hex(_i):
    m = _i.shape[1]
    _bcd = codecs.encode(_i.ravel(), "hex_codec").decode("ASCII").upper()
    return from_buffer(_bcd, dtype="|S%d" % (m * 2))


def bcd_8bit_hex(_i):
    return np.array(["{:X}".format(int(x)) for x in _i], dtype="|S2")


def bcd_julian_day_string_to_nanoseconds_of_year(_i):
    timestrings = bcd_hex(_i)
    return _timestrings_to_nanoseconds(timestrings)


_timegm_cache = {}


def _get_nanoseconds_for_start_of_year(year):
    # Reftek 130 data format stores only the last two digits of the year.
    # We currently assume that 00-49 are years 2000-2049 and 50-99 are years
    # 2050-2099. We deliberately raise an exception in the read routine if the
    # current year will become 2050 (just in case someone really still uses
    # this code then.. ;-)
    if year < 50:
        year += 2000
    else:
        year += 1900
    try:
        ns = _timegm_cache[year]
    except KeyError:
        ns = UTCDateTime(year, 1, 1)._ns
        _timegm_cache[year] = ns
    return ns


def _timestrings_to_nanoseconds(timestrings):
    """
    Helper routine to convert timestrings of form "DDDHHMMSSsss" to array of
    integer nanosecond POSIX timestamp.

    :param timestring: numpy.ndarray
    :rtype: numpy.ndarray
    """
    # split up the time string into tuple of
    # (day of year, hours, minutes, seconds, milliseconds), still as string
    nanoseconds = [(string[:3], string[3:5], string[5:7],
                    string[7:9], string[9:]) for string in timestrings]
    nanoseconds = np.array(nanoseconds, dtype="S3").astype(np.int64)
    # now scale the columns of the array, so that everything is in nanoseconds
    to_nano = 1000000000
    nanoseconds[:, 0] -= 1
    nanoseconds[:, 0] *= 86400 * to_nano
    nanoseconds[:, 1] *= 3600 * to_nano
    nanoseconds[:, 2] *= 60 * to_nano
    nanoseconds[:, 3] *= to_nano
    nanoseconds[:, 4] *= 1000000
    # sum up days, hours, minutes etc. for every row of the array
    nanoseconds = nanoseconds.sum(axis=1)
    return nanoseconds


def _decode_ascii(chars):
    return chars.decode("ASCII")


def _parse_long_time(time_bytestring, decode=True):
    """
    :returns: POSIX timestamp as integer nanoseconds
    """
    if decode:
        time_string = time_bytestring.decode()
    else:
        time_string = time_bytestring
    if not time_string.strip():
        return None
    time_string, milliseconds = time_string[:-3], int(time_string[-3:])
    t = UTCDateTime.strptime(time_string, '%Y%j%H%M%S')
    nanoseconds = t._ns
    nanoseconds += milliseconds * 1000000
    return nanoseconds


def _16_tuple_ascii(bytestring):
    item_count = 16
    chars = bytestring.decode("ASCII")
    if len(chars) % item_count != 0:
        raise NotImplementedError("Should not happen, contact developers.")
    item_size = int(len(chars) / item_count)
    result = []
    for i in range(item_count):
        chars_ = chars[i * item_size:(i + 1) * item_size]
        result.append(chars_.strip() or None)
    return tuple(result)


def _16_tuple_int(bytestring):
    ascii_tuple = _16_tuple_ascii(bytestring)
    result = []
    for chars in ascii_tuple:
        if chars is None or not chars.strip():
            result.append(None)
            continue
        result.append(int(chars))
    return tuple(result)


def _16_tuple_float(bytestring):
    ascii_tuple = _16_tuple_ascii(bytestring)
    result = []
    for chars in ascii_tuple:
        if chars is None or not chars.strip():
            result.append(None)
            continue
        result.append(float(chars))
    return tuple(result)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
