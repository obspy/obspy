# -*- coding: utf-8 -*-
"""
Defines the header structures and some other dictionaries needed for SEG Y read
and write support.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np

from . import pack, unpack


# The format of the 400 byte long binary file header.
BINARY_FILE_HEADER_FORMAT = [
    # [length, name, mandatory]
    [4, 'job_identification_number', False],
    [4, 'line_number', False],
    [4, 'reel_number', False],
    [2, 'number_of_data_traces_per_ensemble', False],
    [2, 'number_of_auxiliary_traces_per_ensemble', False],
    [2, 'sample_interval_in_microseconds', True],
    [2, 'sample_interval_in_microseconds_of_original_field_recording', False],
    [2, 'number_of_samples_per_data_trace', True],
    [2, 'number_of_samples_per_data_trace_for_original_field_recording',
     False],
    [2, 'data_sample_format_code', True],
    [2, 'ensemble_fold', False],
    [2, 'trace_sorting_code', False],
    [2, 'vertical_sum_code', False],
    [2, 'sweep_frequency_at_start', False],
    [2, 'sweep_frequency_at_end', False],
    [2, 'sweep_length', False],
    [2, 'sweep_type_code', False],
    [2, 'trace_number_of_sweep_channel', False],
    [2, 'sweep_trace_taper_length_in_ms_at_start', False],
    [2, 'sweep_trace_taper_length_in_ms_at_end', False],
    [2, 'taper_type', False],
    [2, 'correlated_data_traces', False],
    [2, 'binary_gain_recovered', False],
    [2, 'amplitude_recovery_method', False],
    [2, 'measurement_system', False],
    [2, 'impulse_signal_polarity', False],
    [2, 'vibratory_polarity_code', False],
    [240, 'unassigned_1', False],
    [2, 'seg_y_format_revision_number', True],
    [2, 'fixed_length_trace_flag', True],
    [2, 'number_of_3200_byte_ext_file_header_records_following', True],
    [94, 'unassigned_2', False]]

# The format of the 240 byte long trace header.
TRACE_HEADER_FORMAT = [
    # [length, name, special_type, start_byte]
    # Special type enforces a different format while unpacking using struct.
    [4, 'trace_sequence_number_within_line', False, 0],
    [4, 'trace_sequence_number_within_segy_file', False, 4],
    [4, 'original_field_record_number', False, 8],
    [4, 'trace_number_within_the_original_field_record', False, 12],
    [4, 'energy_source_point_number', False, 16],
    [4, 'ensemble_number', False, 20],
    [4, 'trace_number_within_the_ensemble', False, 24],
    [2, 'trace_identification_code', False, 28],
    [2, 'number_of_vertically_summed_traces_yielding_this_trace', False, 30],
    [2, 'number_of_horizontally_stacked_traces_yielding_this_trace', False,
     32],
    [2, 'data_use', False, 34],
    [4, 'distance_from_center_of_the_source_point_to_' +
     'the_center_of_the_receiver_group', False, 36],
    [4, 'receiver_group_elevation', False, 40],
    [4, 'surface_elevation_at_source', False, 44],
    [4, 'source_depth_below_surface', False, 48],
    [4, 'datum_elevation_at_receiver_group', False, 52],
    [4, 'datum_elevation_at_source', False, 56],
    [4, 'water_depth_at_source', False, 60],
    [4, 'water_depth_at_group', False, 64],
    [2, 'scalar_to_be_applied_to_all_elevations_and_depths', False, 68],
    [2, 'scalar_to_be_applied_to_all_coordinates', False, 70],
    [4, 'source_coordinate_x', False, 72],
    [4, 'source_coordinate_y', False, 76],
    [4, 'group_coordinate_x', False, 80],
    [4, 'group_coordinate_y', False, 84],
    [2, 'coordinate_units', False, 88],
    [2, 'weathering_velocity', False, 90],
    [2, 'subweathering_velocity', False, 92],
    [2, 'uphole_time_at_source_in_ms', False, 94],
    [2, 'uphole_time_at_group_in_ms', False, 96],
    [2, 'source_static_correction_in_ms', False, 98],
    [2, 'group_static_correction_in_ms', False, 100],
    [2, 'total_static_applied_in_ms', False, 102],
    [2, 'lag_time_A', False, 104],
    [2, 'lag_time_B', False, 106],
    [2, 'delay_recording_time', False, 108],
    [2, 'mute_time_start_time_in_ms', False, 110],
    [2, 'mute_time_end_time_in_ms', False, 112],
    [2, 'number_of_samples_in_this_trace', 'H', 114],
    [2, 'sample_interval_in_ms_for_this_trace', 'H', 116],
    [2, 'gain_type_of_field_instruments', False, 118],
    [2, 'instrument_gain_constant', False, 120],
    [2, 'instrument_early_or_initial_gain', False, 122],
    [2, 'correlated', False, 124],
    [2, 'sweep_frequency_at_start', False, 126],
    [2, 'sweep_frequency_at_end', False, 128],
    [2, 'sweep_length_in_ms', False, 130],
    [2, 'sweep_type', False, 132],
    [2, 'sweep_trace_taper_length_at_start_in_ms', False, 134],
    [2, 'sweep_trace_taper_length_at_end_in_ms', False, 136],
    [2, 'taper_type', False, 138],
    [2, 'alias_filter_frequency', False, 140],
    [2, 'alias_filter_slope', False, 142],
    [2, 'notch_filter_frequency', False, 144],
    [2, 'notch_filter_slope', False, 146],
    [2, 'low_cut_frequency', False, 148],
    [2, 'high_cut_frequency', False, 150],
    [2, 'low_cut_slope', False, 152],
    [2, 'high_cut_slope', False, 154],
    [2, 'year_data_recorded', False, 156],
    [2, 'day_of_year', False, 158],
    [2, 'hour_of_day', False, 160],
    [2, 'minute_of_hour', False, 162],
    [2, 'second_of_minute', False, 164],
    [2, 'time_basis_code', False, 166],
    [2, 'trace_weighting_factor', False, 168],
    [2, 'geophone_group_number_of_roll_switch_position_one', False, 170],
    [2, 'geophone_group_number_of_trace_number_one', False, 172],
    [2, 'geophone_group_number_of_last_trace', False, 174],
    [2, 'gap_size', False, 176],
    [2, 'over_travel_associated_with_taper', False, 178],
    [4, 'x_coordinate_of_ensemble_position_of_this_trace', False, 180],
    [4, 'y_coordinate_of_ensemble_position_of_this_trace', False, 184],
    [4, 'for_3d_poststack_data_this_field_is_for_in_line_number', False, 188],
    [4, 'for_3d_poststack_data_this_field_is_for_cross_line_number', False,
     192],
    [4, 'shotpoint_number', False, 196],
    [2, 'scalar_to_be_applied_to_the_shotpoint_number', False, 200],
    [2, 'trace_value_measurement_unit', False, 202],
    # The transduction constant is encoded with the mantissa and the power of
    # the exponent, e.g.:
    # transduction_constant =
    # transduction_constant_mantissa * 10 ** transduction_constant_exponent
    [4, 'transduction_constant_mantissa', False, 204],
    [2, 'transduction_constant_exponent', False, 208],
    [2, 'transduction_units', False, 210],
    [2, 'device_trace_identifier', False, 212],
    [2, 'scalar_to_be_applied_to_times', False, 214],
    [2, 'source_type_orientation', False, 216],
    # XXX: In the SEGY manual it is unclear how the source energy direction
    # with respect to the source orientation is actually defined. It has 6
    # bytes but it seems like it would just need 4. It is encoded as tenths of
    # degrees, e.g. 347.8 is encoded as 3478.
    # As I am totally unclear how this relates to the 6 byte long field I
    # assume that the source energy direction is also encoded as the mantissa
    # and the power of the exponent, e.g.: source_energy_direction =
    # source_energy_direction_mantissa * 10 ** source_energy_direction_exponent
    # Any clarification on the subject is very welcome.
    [4, 'source_energy_direction_mantissa', False, 218],
    [2, 'source_energy_direction_exponent', False, 222],
    # The source measurement is encoded with the mantissa and the power of
    # the exponent, e.g.:
    # source_measurement =
    # source_measurement_mantissa * 10 ** source_measurement_exponent
    [4, 'source_measurement_mantissa', False, 224],
    [2, 'source_measurement_exponent', False, 228],
    [2, 'source_measurement_unit', False, 230],
    [8, 'unassigned', False, 232]]

TRACE_HEADER_KEYS = [_i[1] for _i in TRACE_HEADER_FORMAT]


# Functions that unpack the chosen data format. The keys correspond to the
# number given for each format by the SEG Y format reference.
DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS = {
    1: unpack.unpack_4byte_ibm,
    2: unpack.unpack_4byte_integer,
    3: unpack.unpack_2byte_integer,
    4: unpack.unpack_4byte_fixed_point,
    5: unpack.unpack_4byte_ieee,
    8: unpack.unpack_1byte_integer,
}

# Functions that pack the chosen data format. The keys correspond to the
# number given for each format by the SEG Y format reference.
DATA_SAMPLE_FORMAT_PACK_FUNCTIONS = {
    1: pack.pack_4byte_ibm,
    2: pack.pack_4byte_integer,
    3: pack.pack_2byte_integer,
    4: pack.pack_4byte_fixed_point,
    5: pack.pack_4byte_ieee,
    8: pack.pack_1byte_integer,
}

# Size of one sample.
DATA_SAMPLE_FORMAT_SAMPLE_SIZE = {
    1: 4,
    2: 4,
    3: 2,
    4: 4,
    5: 4,
    8: 1,
}

# Map the data format sample code and the corresponding dtype.
DATA_SAMPLE_FORMAT_CODE_DTYPE = {
    1: np.float32,
    2: np.int32,
    3: np.int16,
    5: np.float32}

# Map the endianness to bigger/smaller sign.
ENDIAN = {
    'big': '>',
    'little': '<',
    '>': '>',
    '<': '<'}
