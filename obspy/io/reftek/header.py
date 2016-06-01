# -*- coding: utf-8 -*-
"""
REFTEK130 read support, header definitions.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .util import (
    _bcd_int, _bcd_str, _bcd_hexstr, _parse_long_time,
    _flags, _channel_codes)


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
        (7, 1, "data_format", _bcd_hexstr),
        (8, 33, "trigger_time_message", None),
        (41, 1, "time_source", None),
        (42, 1, "time_quality", None),
        (43, 1, "station_name_extension", None),
        (44, 4, "station_name", None),
        (48, 16, "stream_name", None),
        (64, 8, "_reserved_2", None),
        (72, 4, "sampling_rate", int),
        (76, 4, "trigger_type", None),
        (80, 16, "trigger_time", _parse_long_time),
        (96, 16, "first_sample_time", _parse_long_time),
        (112, 16, "detrigger_time", _parse_long_time),
        (128, 16, "last_sample_time", _parse_long_time),
        (144, 128, "channel_adjusted_nominal_bit_weights", None),
        (272, 128, "channel_true_bit_weights", None),
        (400, 16, "channel_gain", tuple),
        (416, 16, "channel_ad_resolution", tuple),
        (432, 16, "channel_fsa", tuple),
        (448, 64, "channel_code", _channel_codes),
        (512, 16, "channel_sensor_fsa", tuple),
        (528, 96, "channel_sensor_vpu", None),
        (624, 16, "channel_sensor_units", tuple),
        (640, 48, "station_channel_number", None),
        (688, 156, "_reserved_3", None),
        (844, 2, "total_installed_channels", None),
        (846, 40, "station_comment", None),
        (886, 16, "digital_filter_list", None),
        (902, 26, "position", None),
        (928, 80, "reftek_120", None),
        ),
    "DT": (
        (0, 2, "event_number", _bcd_str),
        (2, 1, "data_stream_number", _bcd_int),
        (3, 1, "channel_number", _bcd_int),
        (4, 2, "number_of_samples", _bcd_int),
        (6, 1, "flags", _flags),
        (7, 1, "data_format", _bcd_hexstr),
        (4, 1004, "sample_data", None),
        ),
    }
PAYLOAD["ET"] = PAYLOAD["EH"]
