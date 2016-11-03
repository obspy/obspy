import io
import numpy as np


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


dtype = [
    ("packet_type", "|S2"),
    ("experiment_number", "uint8"),
    ("year", "uint8"),
    ("unit_id", "uint8", 2),
    ("time", "uint8", 6),
    ("byte_count", "uint8", 2),
    ("packet_sequence", "uint8", 2),
    ("payload", "unit8", 1024 - 16)
]

# Defines the types of header values and how to unpack them.
HEADER_TYPES = {
    "bcd": (("uint8", ), bcd),
    "bcd_16bit_int": (("uint8", 2), bcd_16bit_int),
    "bcd_16bit_hex": (("uint8", 2), bcd_16bit_hex),
    "bcd_8bit_hex": (("uint8", 1), bcd_8bit_hex),
    "bcd_time": (("uint8", 6), bcd_time)
}

# The fixed header which is the same for all packets.
EXTENDED_HEADER = [
    ("packet_type", ("|S2", )),
    ("experiment_number", "bcd"),
    ("year", "bcd"),
    ("unit_id", "bcd_16bit_hex"),
    ("time", "bcd_time"),
    ("byte_count", "bcd_16bit_int"),
    ("packet_sequence", "bcd_16bit_int"),
    ("event_number", "bcd_16bit_int"),
    ("data_stream_number", "bcd"),
    ("channel_number", "bcd"),
    ("number_of_samples", "bcd_16bit_int"),
    ("flags", ("uint8",)),
    ("data_format", "bcd_8bit_hex"),
    # Temporarily store the payload here.
    ("payload", ("uint8", 1000)),
]


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


def _header_to_dtype(h):
    dtype = []
    for name, h_type in h:
        dtype.append((name,) + HEADER_TYPES[h_type][0]
                     if h_type in HEADER_TYPES else (name,) + h_type)
    return dtype


def _unpack_extended_header(string):
    data = np.fromstring(string, dtype=_header_to_dtype(EXTENDED_HEADER))
    results = {}
    dtype = []
    # XXX
    for name, h_type in EXTENDED_HEADER:
        if h_type not in HEADER_TYPES:
            results[name] = data[name]
            continue
        results[name] = HEADER_TYPES[h_type][1](data[name])
    for name, h_type in EXTENDED_HEADER:
        if h_type not in HEADER_TYPES:
            results[name] = data[name]
            continue
        results[name] = HEADER_TYPES[h_type][1](data[name])
    return results


class Reftek130(object):
    def __init__(self, filename):
        with io.open(filename, "rb") as fh:
            string = fh.read(1024*3)
        self._data = _unpack_extended_header(string)

    def to_stream(self):
        raise NotImplementedError()


def _read_reftek130(filename):
    return Reftek130(filename).to_stream()


if __name__ == "__main__":
    filename = "_helpers/000000000_0036EE80.rt130"
    rt = Reftek130(filename)
    from IPython.core.debugger import Tracer; Tracer(colors="Linux")()
