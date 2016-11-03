# -*- coding: utf-8 -*-
"""
Routines for handling of Reftek130 packets.

Currently only event header (EH), event trailer (ET) and data (DT) packets are
handled. These three packets have more or less the same meaning in the first 8
bytes of the payload which makes the first 24 bytes the so called extended
header.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import numpy as np

from obspy import UTCDateTime

from .util import (
    _decode_ascii, _parse_long_time, _16_tuple_ascii, _16_tuple_int, bcd,
    bcd_hex, bcd_julian_day_string_to_seconds_of_year, bcd_16bit_int,
    bcd_8bit_hex, _get_timestamp_for_start_of_year)


PACKET_TYPES_IMPLEMENTED = ("EH", "ET", "DT")
PACKET_TYPES_NOT_IMPLEMENTED = ("AD", "CD", "DS", "FD", "OM", "SC", "SH")
PACKET_TYPES = PACKET_TYPES_IMPLEMENTED + PACKET_TYPES_NOT_IMPLEMENTED


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


# name, offset, length (bytes) and converter routine for EH/ET packet payload
EH_PAYLOAD = {
    "trigger_time_message": (0, 33, _decode_ascii),
    "time_source": (33, 1, _decode_ascii),
    "time_quality": (34, 1, _decode_ascii),
    "station_name_extension": (35, 1, _decode_ascii),
    "station_name": (36, 4, _decode_ascii),
    "stream_name": (40, 16, _decode_ascii),
    "_reserved_2": (56, 8, _decode_ascii),
    "sampling_rate": (64, 4, int),
    "trigger_type": (68, 4, _decode_ascii),
    "trigger_time": (72, 16, _parse_long_time),
    "first_sample_time": (88, 16, _parse_long_time),
    "detrigger_time": (104, 16, _parse_long_time),
    "last_sample_time": (120, 16, _parse_long_time),
    "channel_adjusted_nominal_bit_weights": (136, 128, _16_tuple_ascii),
    "channel_true_bit_weights": (264, 128, _16_tuple_ascii),
    "channel_gain_code": (392, 16, _16_tuple_ascii),
    "channel_ad_resolution_code": (408, 16, _16_tuple_ascii),
    "channel_fsa_code": (424, 16, _16_tuple_ascii),
    "channel_code": (440, 64, _16_tuple_ascii),
    "channel_sensor_fsa_code": (504, 16, _16_tuple_ascii),
    "channel_sensor_vpu": (520, 96, _16_tuple_int),
    "channel_sensor_units_code": (616, 16, _16_tuple_ascii),
    "station_channel_number": (632, 48, _16_tuple_int),
    "_reserved_3": (680, 156, _decode_ascii),
    "total_installed_channels": (836, 2, int),
    "station_comment": (838, 40, _decode_ascii),
    "digital_filter_list": (878, 16, _decode_ascii),
    "position": (894, 26, _decode_ascii),
    "reftek_120": (920, 80, None),
    }


class EHPacket(object):
    __slots__ = ["_data"] + list(EH_PAYLOAD.keys())
    _headers = ('experiment_number', 'unit_id', 'byte_count',
                'packet_sequence', 'time', 'event_number',
                'data_stream_number', 'data_format', 'flags')

    def __init__(self, data):
        self._data = data
        payload = self._data["payload"].tobytes()
        for name, (start, length, converter) in EH_PAYLOAD.items():
            data = payload[start:start+length]
            if converter is not None:
                data = converter(data)
            setattr(self, name, data)

    @property
    def type(self):
        return self._data['packet_type'].item()

    def __getattr__(self, name):
        if name in self._headers:
            return self._data[name].item()

    @property
    def timestamp(self):
        return self._data['time'].item()

    @property
    def time(self):
        return UTCDateTime(self._data['time'].item())

    def _to_dict(self):
        """
        Convert to dictionary structure.
        """
        return {key: getattr(self, key) for key in EH_PAYLOAD.keys()}


PACKET_INITIAL_UNPACK_DTYPE = np.dtype([
    (native_str(name), dtype_initial)
    for name, dtype_initial, converter, dtype_final in PACKET])

PACKET_FINAL_DTYPE = np.dtype([
    (native_str(name), dtype_final)
    for name, dtype_initial, converter, dtype_final in PACKET])


def _initial_unpack_packets(bytestring):
    """
    First unpack data with dtype matching itemsize of storage in the reftek
    file, than allocate result array with dtypes for storage of python
    objects/arrays and fill it with the unpacked data.
    """
    data = np.fromstring(
        bytestring, dtype=PACKET_INITIAL_UNPACK_DTYPE)
    result = np.empty_like(data, dtype=PACKET_FINAL_DTYPE)

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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
