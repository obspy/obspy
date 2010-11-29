# -*- coding: utf-8 -*-
"""
Defines the header structures and some other dictionaries needed for SEG Y read
and write support.
"""

from pack import *
from unpack import *

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
    [2, 'number_of_samples_per_data_trace_for_original_field_recording', False],
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
    [94, 'unassigned_2', False]
]

# The format of the 240 byte long trace header.
TRACE_HEADER_FORMAT = [
    # [length, name]
    [4, 'trace_sequence_number_within_line'],
    [4, 'trace_sequence_number_within_segy_file'],
    [4, 'original_field_record_number'],
    [4, 'trace_number_within_the_original_field_record'],
    [4, 'energy_source_point_number'],
    [4, 'ensemble_number'],
    [4, 'trace_number_within_the_ensemble'],
    [2, 'trace_identification_code'],
    [2, 'number_of_vertically_summed_traces_yielding_this_trace'],
    [2, 'number_of_horizontally_stacked_traces_yielding_this_trace'],
    [2, 'data_use'],
    [4,
     'distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group'],
    [4, 'receiver_group_elevation'],
    [4, 'surface_elevation_at_source'],
    [4, 'source_depth_below_surface'],
    [4, 'datum_elevation_at_receiver_group'],
    [4, 'datum_elevation_at_source'],
    [4, 'water_depth_at_source'],
    [4, 'water_depth_at_group'],
    [2, 'scalar_to_be_applied_to_all_elevations_and_depths'],
    [2, 'scalar_to_be_applied_to_all_coordinates'],
    [4, 'source_coordinate-x'],
    [4, 'source_coordinate-y'],
    [4, 'group_coordinate-x'],
    [4, 'group_coordinate-y'],
    [2, 'coordinate_units'],
    [2, 'weathering_velocity'],
    [2, 'subweathering_velocity'],
    [2, 'uphole_time_at_source_in_ms'],
    [2, 'uphole_time_at_group_in_ms'],
    [2, 'source_static_correction_in_ms'],
    [2, 'group_static_correction_in_ms'],
    [2, 'total_static_applied_in_ms'],
    [2, 'lag_time_A'],
    [2, 'lag_time_B'],
    [2, 'delay_recording_time'],
    [2, 'mute_time-start_time_in_ms'],
    [2, 'mute_time-end_time_in_ms'],
    [2, 'number_of_samples_in_this_trace'],
    [2, 'sample_interval_in_ms_for_this_trace'],
    [2, 'gain_type_of_field_instruments'],
    [2, 'instrument_gain_constant'],
    [2, 'instrument_early_or_initial_gain'],
    [2, 'correlated'],
    [2, 'sweep_frequency_at_start'],
    [2, 'sweep_frequency_at_end'],
    [2, 'sweep_length_in_ms'],
    [2, 'sweep_type'],
    [2, 'sweep_trace_taper_length_at_start_in_ms'],
    [2, 'sweep_trace_taper_length_at_end_in_ms'],
    [2, 'taper_type'],
    [2, 'alias_filter_frequency'],
    [2, 'alias_filter_slope'],
    [2, 'notch_filter_frequency'],
    [2, 'notch_filter_slope'],
    [2, 'low_cut_frequency'],
    [2, 'high_cut_frequency'],
    [2, 'low_cut_slope'],
    [2, 'high_cut_slope'],
    [2, 'year_data_recorded'],
    [2, 'day_of_year'],
    [2, 'hour_of_day'],
    [2, 'minute_of_hour'],
    [2, 'second_of_minute'],
    [2, 'time_basis_code'],
    [2, 'trace_weighting_factor'],
    [2, 'geophone_group_number_of_roll_switch_position_one'],
    [2, 'geophone_group_number_of_trace_number_one'],
    [2, 'geophone_group_number_of_last_trace'],
    [2, 'gap_size'],
    [2, 'over_travel_associated_with_taper'],
    [4, 'x_coordinate_of_ensemble_position_of_this_trace'],
    [4, 'y_coordinate_of_ensemble_position_of_this_trace'],
    [4, 'for_3d_poststack_data_this_field_is_for_in-line_number'],
    [4, 'for_3d_poststack_data_this_field_is_for_cross-line_number'],
    [4, 'shotpoint_number'],
    [2, 'scalar_to_be_applied_to_the_shotpoint_number'],
    [2, 'trace_value_measurement_unit'],
    # The transduction constant is encoded with the mantissa and the power of
    # the exponent, e.g.:
    # transduction_constant = 
    # transduction_constant_mantissa * 10 ** transduction_constant_exponent
    [4, 'transduction_constant_mantissa'],
    [2, 'transduction_constant_exponent'],
    [2, 'transduction_units'],
    [2, 'device/trace_identifier'],
    [2, 'scalar_to_be_applied_to_times'],
    [2, 'source/type_orientation'],
    # XXX: In the SEGY manual it is unclear how the source energy direction
    # with respect to the source orientation is actually defined. It has 6
    # bytes but it seems like it would just need 4. It is encoded as tenths of
    # degrees, e.g. 347.8 is encoded as 3478.
    # As I am totally unclear how this relates to the 6 byte long field I
    # assume that the source energy direction is also encoded as the mantissa
    # and the power of the exponent, e.g.: source_energy_direction = 
    # source_energy_direction_mantissa * 10 ** source_energy_direction_exponent
    # Any clarification on the subject is very welcome.
    [4, 'source_energy_direction_mantissa'],
    [2, 'source_energy_direction_exponent'],
    # The source measurement is encoded with the mantissa and the power of
    # the exponent, e.g.:
    # source_measurement = 
    # source_measurement_mantissa * 10 ** source_measurement_exponent
    [4, 'source_measurement_mantissa'],
    [2, 'source_measurement_exponent'],
    [2, 'source_measurement_unit'],
    [8, 'unassigned']
]

# Functions that unpack the chosen data format. The keys correspond to the
# number given for each format by the SEG Y format reference.
DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS = {
    1: unpack_4byte_IBM,
    2: unpack_4byte_Integer,
    3: unpack_2byte_Integer,
    4: unpack_4byte_Fixed_point,
    5: unpack_4byte_IEEE,
    8: unpack_1byte_Integer,
}

# Functions that pack the chosen data format. The keys correspond to the
# number given for each format by the SEG Y format reference.
DATA_SAMPLE_FORMAT_PACK_FUNCTIONS = {
    1: pack_4byte_IBM,
    2: pack_4byte_Integer,
    3: pack_2byte_Integer,
    4: pack_4byte_Fixed_point,
    5: pack_4byte_IEEE,
    8: pack_1byte_Integer,
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

# Map the endianness to bigger/smaller sign.
ENDIAN = {
    'big': '>',
    'little': '<'
}
