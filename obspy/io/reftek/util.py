# -*- coding: utf-8 -*-
"""
REFTEK130 read support, utility functions.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import calendar
import codecs
import datetime

import numpy as np
from obspy import UTCDateTime
# from obspy.io.mseed.util import _unpack_steim_1


NOW = UTCDateTime()

_timegm_cache = {}


def _get_timestamp_for_start_of_year(year):
    if NOW.year > 2050:
        raise NotImplementedError()
    if year < 50:
        year += 2000
    else:
        year += 1900

    try:
        t = _timegm_cache[year]
    except KeyError:
        t = calendar.timegm(datetime.datetime(year, 1, 1).utctimetuple())
        _timegm_cache[year] = t
    return t


def _bcd_str(chars):
    return codecs.encode(chars, "hex_codec").decode("ASCII").upper()


def _bits(char):
    bits = np.unpackbits(np.fromstring(char, dtype=np.uint8))
    return bits.astype(np.bool_).tolist()


def _flags(char):
    bits = _bits(char)
    keys = ("first_packet", "last_packet", "second_EH_ET", "unused",
            "ST_command_trigger_event", "stacked_data_in_packet",
            "overscaled_data_detected_during_packet",
            "calibration_signal_enabled_during_packet")
    return {key: bit for key, bit in zip(keys, bits)}


def _bcd_int(chars):
    return int(codecs.encode(chars, "hex_codec").decode("ASCII")) \
        if chars else None


def _prepare_time(packets):
    """
    Helper routine to set POSIX timestamp information for a list of packets,
    based on their raw time information (year as integer without century and
    timestring DDDHHMMSSsss)
    """
    # split up the time string into tuple of
    # (day of year, hours, minutes, seconds, milliseconds), still as string
    seconds = [(p._time_raw[1][:3], p._time_raw[1][3:5], p._time_raw[1][5:7],
                p._time_raw[1][7:9], p._time_raw[1][9:]) for p in packets]
    seconds = np.array(seconds, dtype="S3").astype(np.float64)
    # now scale the columns of the array, so that everything is in seconds
    seconds[:, 0] -= 1
    seconds[:, 0] *= 86400
    seconds[:, 1] *= 3600
    seconds[:, 2] *= 60
    seconds[:, 4] *= 1e-3
    # sum up days, hours, minutes etc. for every row of the array
    seconds = seconds.sum(axis=1)
    # now set every packet's time with the base POSIX timestamp of the start of
    # respective year (which we calculate only once) and adding the previously
    # calculated seconds from start of year
    for p, s in zip(packets, seconds):
        p._time = _get_timestamp_for_start_of_year(p._time_raw[0]) + s


def _parse_short_time(year, time_string):
    t = _get_timestamp_for_start_of_year(year)
    t += (86400 * (int(time_string[:3]) - 1) + 3600 * int(time_string[3:5]) +
          60 * int(time_string[5:7]) + int(time_string[7:9]) +
          1e-3 * int(time_string[9:]))
    return t


def _parse_long_time(time_bytestring, decode=True):
    if decode:
        time_string = time_bytestring.decode()
    else:
        time_string = time_bytestring
    if not time_string.strip():
        return None
    time_string, milliseconds = time_string[:-3], int(time_string[-3:])
    return (UTCDateTime().strptime(time_string, '%Y%j%H%M%S') +
            1e-3 * milliseconds)


# def _parse_data(data):
#     npts = _bcd_int(data[0:2])
#     # flags = _bcd_int(data[2])
#     data_format = _bcd_hexstr(data[3])
#     data = data[4:]
#     if data_format == "C0":
#         data = data[40:]
#         # XXX why need to swap? verbose for now..
#         return _unpack_steim_1(data, npts, swapflag=1, verbose=True)
#     else:
#         raise NotImplementedError()


def _decode_ascii(chars):
    return chars.decode("ASCII")


def _16_tuple_ascii(bytestring):
    item_count = 16
    chars = bytestring.decode("ASCII")
    if len(chars) % item_count != 0:
        raise NotImplementedError("Should not happen, contact developers.")
    item_size = int(len(chars) / item_count)
    result = []
    for i in range(item_count):
        chars_ = chars[i*item_size:(i+1)*item_size]
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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
