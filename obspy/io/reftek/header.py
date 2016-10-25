# -*- coding: utf-8 -*-
"""
REFTEK130 read support, header definitions.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .util import (
    _bcd_int, _bcd_str, _parse_long_time,
    _flags, _decode_ascii, _16_tuple_ascii, _16_tuple_int)


PACKETS_IMPLEMENTED = {
    "AD": False,
    "CD": False,
    "DS": False,
    "DT": True,
    "EH": True,
    "ET": True,
    "FD": False,
    "OM": False,
    "SC": False,
    "SH": False,
    }


PAYLOAD = {
    "EH": (
        (0, 2, "event_number", _bcd_int),
        (2, 1, "data_stream_number", _bcd_int),
        (3, 3, "_reserved_1", None),
        (6, 1, "flags", _flags),
        (7, 1, "data_format", _bcd_str),
        (8, 33, "trigger_time_message", _decode_ascii),
        (41, 1, "time_source", _decode_ascii),
        (42, 1, "time_quality", _decode_ascii),
        (43, 1, "station_name_extension", _decode_ascii),
        (44, 4, "station_name", _decode_ascii),
        (48, 16, "stream_name", _decode_ascii),
        (64, 8, "_reserved_2", _decode_ascii),
        (72, 4, "sampling_rate", int),
        (76, 4, "trigger_type", _decode_ascii),
        (80, 16, "trigger_time", _parse_long_time),
        (96, 16, "first_sample_time", _parse_long_time),
        (112, 16, "detrigger_time", _parse_long_time),
        (128, 16, "last_sample_time", _parse_long_time),
        (144, 128, "channel_adjusted_nominal_bit_weights", _16_tuple_ascii),
        (272, 128, "channel_true_bit_weights", _16_tuple_ascii),
        (400, 16, "channel_gain_code", _16_tuple_ascii),
        (416, 16, "channel_ad_resolution_code", _16_tuple_ascii),
        (432, 16, "channel_fsa_code", _16_tuple_ascii),
        (448, 64, "channel_code", _16_tuple_ascii),
        (512, 16, "channel_sensor_fsa_code", _16_tuple_ascii),
        (528, 96, "channel_sensor_vpu", _16_tuple_int),
        (624, 16, "channel_sensor_units_code", _16_tuple_ascii),
        (640, 48, "station_channel_number", _16_tuple_int),
        (688, 156, "_reserved_3", _decode_ascii),
        (844, 2, "total_installed_channels", int),
        (846, 40, "station_comment", _decode_ascii),
        (886, 16, "digital_filter_list", _decode_ascii),
        (902, 26, "position", _decode_ascii),
        (928, 80, "reftek_120", None),
        ),
    "DT": (
        (0, 2, "event_number", _bcd_str),
        (2, 1, "data_stream_number", _bcd_int),
        (3, 1, "channel_number", _bcd_int),
        (4, 2, "number_of_samples", _bcd_int),
        (6, 1, "flags", _flags),
        (7, 1, "data_format", _bcd_str),
        (4, 1004, "sample_data", None),
        ),
    }
PAYLOAD["ET"] = PAYLOAD["EH"]
