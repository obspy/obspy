# -*- coding: utf-8 -*-
"""
REFTEK130 read support.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import io

import numpy as np


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


def bcd_16bit_hex(_i):
    return np.array(["{:X}".format(x) for x in
                    (_i[::, 0] * 256 + _i[::, 1])], dtype="|S4")


def bcd_8bit_hex(_i):
    return np.array(["{:X}".format(x) for x in _i], dtype="|S2")


def bcd_time(_i):
    # Time is a bit wild.
    t = bcd(_i)
    julday = t[::, 0] * 10 + t[::, 1] // 10.0
    # XXX
    hour = t[::, 1] % 10 * 10 + t[::, 2] // 10.0
    minute = t[::, 2] % 10 * 10 + t[::, 3] // 10.0
    second = t[::, 3] % 10 * 10 + t[::, 4] // 10.0
    microsecond = t[::, 4] % 10 * 100 + t[::, 5] * 1000
    return julday

# The extended header which is the same for EH/ET/DT packets.
# tuples are:
#  - field name
#  - dtype during initial reading
#  - conversion routine (if any)
#  - dtype after conversion
PACKET = [
    ("packet_type", native_str("|S2"), None, native_str("S2")),
    ("experiment_number", np.uint8, bcd, np.uint32),
    ("year", np.uint8, bcd, None),
    ("unit_id", (np.uint8, 2), bcd_16bit_hex, native_str("S4")),
    ("time", (np.uint8, 6), bcd_time, np.float64),
    ("byte_count", (np.uint8, 2), bcd_16bit_int, np.uint32),
    ("packet_sequence", (np.uint8, 2), bcd_16bit_int, np.uint32),
    ("event_number", (np.uint8, 2), bcd_16bit_int, np.uint32),
    ("data_stream_number", np.uint8, bcd, np.uint32),
    ("channel_number", np.uint8, bcd, np.uint32),
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
    for name, dtype_initial, converter, dtype_final in PACKET
    if dtype_final is not None])


def _initial_unpack_packets(bytestring):
    data = np.fromstring(
        bytestring, dtype=packet_initial_unpack_dtype)
    result = np.empty_like(data, dtype=packet_final_dtype)

    for name, dtype_initial, converter, dtype_final in PACKET:
        if dtype_final is None:
            continue
        if converter is None:
            result[name][:] = data[name][:]
            continue
        result[name][:] = converter(data[name])

    return result


class Reftek130(object):
    def __init__(self, filename):
        with io.open(filename, "rb") as fh:
            string = fh.read(1024*3)
        self._data = _initial_unpack_packets(string)

    def to_stream(self):
        raise NotImplementedError()


def _read_reftek130(filename):
    return Reftek130(filename).to_stream()


if __name__ == "__main__":
    filename = "_helpers/000000000_0036EE80.rt130"
    rt = Reftek130(filename)
    from IPython.core.debugger import Tracer; Tracer(colors="Linux")()
