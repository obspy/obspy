# -*- coding: utf-8 -*-
"""
REFTEK130 read support.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import calendar
import codecs
import datetime
import io

import numpy as np

from obspy import UTCDateTime


NOW = UTCDateTime()


_timegm_cache = {}


def _get_timestamp_for_start_of_year(year):
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


def _timestrings_to_seconds(timestrings):
    """
    Helper routine to convert timestrings of form "DDDHHMMSSsss" to array of
    floating point seconds.

    :param timestring: numpy.ndarray
    :rtype: numpy.ndarray
    """
    # split up the time string into tuple of
    # (day of year, hours, minutes, seconds, milliseconds), still as string
    seconds = [(string[:3], string[3:5], string[5:7],
                string[7:9], string[9:]) for string in timestrings]
    seconds = np.array(seconds, dtype="S3").astype(np.float64)
    # now scale the columns of the array, so that everything is in seconds
    seconds[:, 0] -= 1
    seconds[:, 0] *= 86400
    seconds[:, 1] *= 3600
    seconds[:, 2] *= 60
    seconds[:, 4] *= 1e-3
    # sum up days, hours, minutes etc. for every row of the array
    seconds = seconds.sum(axis=1)
    return seconds


# # All the other headers as a dictionary of dictionaries.
# OTHER_HEADERS = {
#     "EH": {
#         "trigger_time_message": ("|S33",)):
#         "time_source": ("|S1",)):
#         "time_quality": ("|S1",)):
#         "station_name_extension": ("|S1",)):
#         "station_name": ("|S4",)):
#         "stream_name": ("|S16",)):
#         "_reserved_2": ("b1", 8)):
#         "sampling_rate": ("f32",)):
#         "trigger_type": ("|S4",)):
#         "trigger_time": "long_time"):
#         "first_sample_time": "long_time"):
#         "detrigger_time": "long_time"):
#         "last_sample_time": "long_time"):
#         "channel_adjusted_nominal_bit_weights": _16_tuple_ascii):
#         "channel_true_bit_weights": _16_tuple_ascii):
#         "channel_gain_code": _16_tuple_ascii):
#         "channel_ad_resolution_code": _16_tuple_ascii):
#         "channel_fsa_code": _16_tuple_ascii):
#         "channel_code": _16_tuple_ascii):
#         "channel_sensor_fsa_code": _16_tuple_ascii):
#         "channel_sensor_vpu": _16_tuple_int):
#         "channel_sensor_units_code": _16_tuple_ascii):
#         "station_channel_number": _16_tuple_int):
#         "_reserved_3": _decode_ascii):
#         "total_installed_channels": int):
#         "station_comment": _decode_ascii):
#         "digital_filter_list": _decode_ascii):
#         "position": _decode_ascii):
#         "reftek_120": None):
#     }
# }


def bcd(_i):
    return (_i >> 4 & 0xF).astype(np.uint32) * 10 + (_i & 0xF)


def bcd_16bit_int(_i):
    _i = bcd(_i)
    return _i[::, 0] * 100 + _i[::, 1]


def bcd_hex(_i):
    result = [codecs.encode(chars, "hex_codec").decode("ASCII").upper()
              for chars in _i]
    return np.array(result)


def bcd_8bit_hex(_i):
    return np.array(["{:X}".format(x) for x in _i], dtype="|S2")


def bcd_julian_day_string_to_seconds_of_year(_i):
    timestrings = bcd_hex(_i)
    return _timestrings_to_seconds(timestrings)


# The extended header which is the same for EH/ET/DT packets.
# tuples are:
#  - field name
#  - dtype during initial reading
#  - conversion routine (if any)
#  - dtype after conversion
PACKET = [
    ("packet_type", native_str("|S2"), None, native_str("S2")),
    ("experiment_number", np.uint8, bcd, np.uint8),
    ("year", np.uint8, bcd, np.uint8),
    ("unit_id", (np.uint8, 2), bcd_hex, native_str("S4")),
    ("time", (np.uint8, 6), bcd_julian_day_string_to_seconds_of_year,
     np.float64),
    ("byte_count", (np.uint8, 2), bcd_16bit_int, np.uint16),
    ("packet_sequence", (np.uint8, 2), bcd_16bit_int, np.uint16),
    ("event_number", (np.uint8, 2), bcd_16bit_int, np.uint16),
    ("data_stream_number", np.uint8, bcd, np.uint8),
    ("channel_number", np.uint8, bcd, np.uint8),
    ("number_of_samples", (np.uint8, 2), bcd_16bit_int, np.uint32),
    ("flags", np.uint8, None, np.uint8),
    ("data_format", np.uint8, bcd_8bit_hex, native_str("S2")),
    # Temporarily store the payload here.
    ("payload", (np.uint8, 1000), None, (np.uint8, 1000)),
]


packet_initial_unpack_dtype = np.dtype([
    (native_str(name), dtype_initial)
    for name, dtype_initial, converter, dtype_final in PACKET])

packet_final_dtype = np.dtype([
    (native_str(name), dtype_final)
    for name, dtype_initial, converter, dtype_final in PACKET])


def _initial_unpack_packets(bytestring):
    data = np.fromstring(
        bytestring, dtype=packet_initial_unpack_dtype)
    result = np.empty_like(data, dtype=packet_final_dtype)

    for name, dtype_initial, converter, dtype_final in PACKET:
        if converter is None:
            result[name][:] = data[name][:]
            continue
        result[name][:] = converter(data[name])
    # time unpacking is special and needs some additional work.
    # we need to add the POSIX timestamp of the start of respective year to the
    # already unpacked seconds into the respective year..
    result['time'][:] += [_get_timestamp_for_start_of_year(y)
                          for y in result['year']]

    return result


class Reftek130(object):
    def __init__(self, filename):
        with io.open(filename, "rb") as fh:
            string = fh.read(1024*3)
        self._data = _initial_unpack_packets(string)

    def to_stream(self):
        raise NotImplementedError()


def _read_reftek130(filename):
    if NOW.year > 2050:
        raise NotImplementedError()
    return Reftek130(filename).to_stream()


if __name__ == "__main__":
    filename = "_helpers/000000000_0036EE80.rt130"
    rt = Reftek130(filename)
    from IPython.core.debugger import Tracer; Tracer(colors="Linux")()
