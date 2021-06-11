# -*- coding: utf-8 -*-
"""
Defines the header structures and some other dictionaries needed for SEG Y read
and write support.
"""
import numpy as np

from . import pack, unpack


#: The format of the 400 byte long binary file header.
BINARY_FILE_HEADER_FORMAT = [
    # [length, name, mandatory]
    [4, 'job_identification_number', False], # bytes 3201--3204
    [4, 'line_number', False], # bytes 3205--3208
    [4, 'reel_number', False], # bytes 3209--3212
    [2, 'number_of_data_traces_per_ensemble', False], # bytes 3213--3214
    [2, 'number_of_auxiliary_traces_per_ensemble', False], # bytes 3215--3216
    [2, 'sample_interval_in_microseconds', True], # bytes 3217--3218
    [2, 'sample_interval_in_microseconds_of_original_field_recording', False], # bytes 3219--3220
    [2, 'number_of_samples_per_data_trace', True], # bytes 3221--3222
    [2, 'number_of_samples_per_data_trace_for_original_field_recording', False], # bytes 3223--3224
    [2, 'data_sample_format_code', True], # bytes 3225--3226
    [2, 'ensemble_fold', False],  # bytes 3227--3228
    [2, 'trace_sorting_code', False], # bytes 3229--3230
    [2, 'vertical_sum_code', False], # bytes 3231--3232
    [2, 'sweep_frequency_at_start', False], # bytes 3233--3234
    [2, 'sweep_frequency_at_end', False], # bytes 3235--3236
    [2, 'sweep_length', False], # bytes 3237--3238
    [2, 'sweep_type_code', False], # bytes 3239--3240
    [2, 'trace_number_of_sweep_channel', False], # bytes 3241--3242
    [2, 'sweep_trace_taper_length_in_ms_at_start', False], # bytes 3243--3244
    [2, 'sweep_trace_taper_length_in_ms_at_end', False], # bytes 3245--3246
    [2, 'taper_type', False], # bytes 3247--3248
    [2, 'correlated_data_traces', False], # bytes 3249--3250
    [2, 'binary_gain_recovered', False], # bytes 3251--3252
    [2, 'amplitude_recovery_method', False], # bytes 3253--3254
    [2, 'measurement_system', False], # bytes 3255--3256
    [2, 'impulse_signal_polarity', False], # bytes 3257--3258
    [2, 'vibratory_polarity_code', False], # bytes 3259--3260
    [4, 'extended_number_of_data_traces_per_ensemble', False], # bytes 3261--3264
    [4, 'extended_number_of_auxiliary_traces_per_ensemble', False], # bytes 3265--3268
    [4, 'extended_number_of_samples_per_data_trace', False], # bytes 3269--3272
    [8, 'double_extended_sample_interval_in_microseconds', False], # bytes 3273--3280
    [8, 'double_extended_sample_interval_in_microseconds_for_original_recording', False], # bytes 3281--3288
    [4, 'extended_number_of_samples_per_data_trace_for_original_recording', False], # bytes 3289--3292
    [4, 'extended_ensemble_fold', False], # bytes 3293--3296
    [4, 'byte_ordering_constant', False], # bytes 3297--3300
    [200, 'unassigned_1', False], # bytes 3301--3500
    [2, 'seg_y_format_revision_number', True], # bytes 3501--3502
    [2, 'fixed_length_trace_flag', True], # bytes 3503--3504
    [2, 'number_of_3200_byte_ext_file_header_records_following', True], # bytes 3505--3506
    [4, 'number_of_additional_trace_headers', False], # bytes 3507--3510
    [2, 'time_basis_code', False], # bytes 3511--3512
    [8, 'unsignedint_number_of_traces_in_file', False], # bytes 3513--3520
    [8, 'unsignedint_byte_offset_of_first_trace', False], # bytes 3521--3528
    [4, 'signedint_number_of_trailer_stanzas', False], # bytes 3529--3532
    [68, 'unassigned_2', False]] # bytes 3533--3600

#: The format of the 240 byte long trace header.
TRACE_HEADER_FORMAT = [
    # [length, name, special_type, start_byte]
    # Special type enforces a different format while unpacking using struct.
    [4, 'trace_sequence_number_within_line', False, 0], # bytes 1--4
    [4, 'trace_sequence_number_within_segy_file', False, 4], # bytes 5--8
    [4, 'original_field_record_number', False, 8], # bytes 9--12
    [4, 'trace_number_within_the_original_field_record', False, 12], # bytes 13--16
    [4, 'energy_source_point_number', False, 16], # bytes 17--20
    [4, 'ensemble_number', False, 20], # bytes 21--24
    [4, 'trace_number_within_the_ensemble', False, 24], # bytes 25--28
    [2, 'trace_identification_code', False, 28], # bytes 29--30
    [2, 'number_of_vertically_summed_traces_yielding_this_trace', False, 30], # bytes 31--32
    [2, 'number_of_horizontally_stacked_traces_yielding_this_trace', False, 32], # bytes 33--34
    [2, 'data_use', False, 34], # bytes 35--36
    [4, 'distance_from_center_of_the_source_point_to_' + 'the_center_of_the_receiver_group', False, 36], # bytes 37--40
    [4, 'receiver_group_elevation', False, 40], # bytes 41--44
    [4, 'surface_elevation_at_source', False, 44], # bytes 45--48
    [4, 'source_depth_below_surface', False, 48], # bytes 49--52
    [4, 'datum_elevation_at_receiver_group', False, 52], # bytes 53--56
    [4, 'datum_elevation_at_source', False, 56], # bytes 57--60
    [4, 'water_depth_at_source', False, 60], # bytes 61--64
    [4, 'water_depth_at_group', False, 64], # bytes 65--68
    [2, 'scalar_to_be_applied_to_all_elevations_and_depths', False, 68], # bytes 69--70
    [2, 'scalar_to_be_applied_to_all_coordinates', False, 70], # bytes 71--72
    [4, 'source_coordinate_x', False, 72], # bytes 73--76
    [4, 'source_coordinate_y', False, 76], # bytes 77-80
    [4, 'group_coordinate_x', False, 80], # bytes 81--84
    [4, 'group_coordinate_y', False, 84], # bytes 85--88
    [2, 'coordinate_units', False, 88], # bytes 89--90
    [2, 'weathering_velocity', False, 90], # bytes 91--92
    [2, 'subweathering_velocity', False, 92], # bytes 93--94
    [2, 'uphole_time_at_source_in_ms', False, 94], # bytes 95--96
    [2, 'uphole_time_at_group_in_ms', False, 96], # bytes 97--98
    [2, 'source_static_correction_in_ms', False, 98], # bytes 99--100
    [2, 'group_static_correction_in_ms', False, 100], # bytes 101--102
    [2, 'total_static_applied_in_ms', False, 102], # bytes 103--104
    [2, 'lag_time_A', False, 104], # bytes 105--106
    [2, 'lag_time_B', False, 106], # bytes 107--108
    [2, 'delay_recording_time', False, 108], # bytes 109--110
    [2, 'mute_time_start_time_in_ms', False, 110], # bytes 111--112
    [2, 'mute_time_end_time_in_ms', False, 112], # bytes 113--114
    [2, 'number_of_samples_in_this_trace', 'H', 114], # bytes 115--116
    [2, 'sample_interval_in_ms_for_this_trace', 'H', 116], # bytes 117--118
    [2, 'gain_type_of_field_instruments', False, 118], # bytes 119--120
    [2, 'instrument_gain_constant', False, 120], # bytes 121--122
    [2, 'instrument_early_or_initial_gain', False, 122], # bytes 123--124
    [2, 'correlated', False, 124], # bytes 125--126
    [2, 'sweep_frequency_at_start', False, 126], # bytes 127--128
    [2, 'sweep_frequency_at_end', False, 128], # bytes 129--130
    [2, 'sweep_length_in_ms', False, 130], # bytes 131--132
    [2, 'sweep_type', False, 132], # bytes 133--134
    [2, 'sweep_trace_taper_length_at_start_in_ms', False, 134], # bytes 135--136
    [2, 'sweep_trace_taper_length_at_end_in_ms', False, 136], # bytes 137--138
    [2, 'taper_type', False, 138], # bytes 139--140
    [2, 'alias_filter_frequency', False, 140], # bytes 141--142
    [2, 'alias_filter_slope', False, 142], # bytes 143--144
    [2, 'notch_filter_frequency', False, 144], # bytes 145--146
    [2, 'notch_filter_slope', False, 146], # bytes 147--148
    [2, 'low_cut_frequency', False, 148], # bytes 149--150
    [2, 'high_cut_frequency', False, 150], # bytes 151--152
    [2, 'low_cut_slope', False, 152], # bytes 153--154
    [2, 'high_cut_slope', False, 154], # bytes 155--156
    [2, 'year_data_recorded', False, 156], # bytes 157--158
    [2, 'day_of_year', False, 158], # bytes 159--160
    [2, 'hour_of_day', False, 160], # bytes 161--162
    [2, 'minute_of_hour', False, 162], # bytes 163--164
    [2, 'second_of_minute', False, 164], # bytes 165--166
    [2, 'time_basis_code', False, 166], # bytes 167--168
    [2, 'trace_weighting_factor', False, 168], # bytes 169--170
    [2, 'geophone_group_number_of_roll_switch_position_one', False, 170], # bytes 171--172
    [2, 'geophone_group_number_of_trace_number_one', False, 172], # bytes  # bytes 173--174
    [2, 'geophone_group_number_of_last_trace', False, 174], # bytes 175--176
    [2, 'gap_size', False, 176], # bytes 177--178
    [2, 'over_travel_associated_with_taper', False, 178], # bytes 179--180
    [4, 'x_coordinate_of_ensemble_position_of_this_trace', False, 180], # bytes 181--184
    [4, 'y_coordinate_of_ensemble_position_of_this_trace', False, 184], # bytes 185--188
    [4, 'for_3d_poststack_data_this_field_is_for_in_line_number', False, 188], # bytes 189--192
    [4, 'for_3d_poststack_data_this_field_is_for_cross_line_number', False, 192], # bytes 193--196
    [4, 'shotpoint_number', False, 196], # bytes 197--200
    [2, 'scalar_to_be_applied_to_the_shotpoint_number', False, 200], # bytes 201--202
    [2, 'trace_value_measurement_unit', False, 202], # bytes 203--204
    # The transduction constant is encoded with the mantissa and the power of
    # the exponent, e.g.:
    # transduction_constant =
    # transduction_constant_mantissa * 10 ** transduction_constant_exponent
    [4, 'transduction_constant_mantissa', False, 204], # bytes 205--208
    [2, 'transduction_constant_exponent', False, 208], # bytes 209--210
    [2, 'transduction_units', False, 210], # bytes 211--212
    [2, 'device_trace_identifier', False, 212], # bytes 213--214
    [2, 'scalar_to_be_applied_to_times', False, 214], # bytes 215--216
    [2, 'source_type_orientation', False, 216], # bytes 217--218
    # XXX: In the SEGY manual it is unclear how the source energy direction
    # with respect to the source orientation is actually defined. It has 6
    # bytes but it seems like it would just need 4. It is encoded as tenths of
    # degrees, e.g. 347.8 is encoded as 3478.
    # As I am totally unclear how this relates to the 6 byte long field I
    # assume that the source energy direction is also encoded as the mantissa
    # and the power of the exponent, e.g.: source_energy_direction =
    # source_energy_direction_mantissa * 10 ** source_energy_direction_exponent
    # Any clarification on the subject is very welcome.
    [4, 'source_energy_direction_mantissa', False, 218], # bytes 219--222
    [2, 'source_energy_direction_exponent', False, 222], # bytes 223--224
    # The source measurement is encoded with the mantissa and the power of
    # the exponent, e.g.:
    # source_measurement =
    # source_measurement_mantissa * 10 ** source_measurement_exponent
    [4, 'source_measurement_mantissa', False, 224], # bytes 225--228
    [2, 'source_measurement_exponent', False, 228], # bytes 229--230
    [2, 'source_measurement_unit', False, 230], # bytes 231--232
    [8, 'unassigned', False, 232]] # bytes 233--240

TRACE_HEADER_KEYS = [_i[1] for _i in TRACE_HEADER_FORMAT]


#: Functions that unpack the chosen data format. The keys correspond to the
#: number given for each format by the SEG Y format reference.
DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS = {
    1: unpack.unpack_4byte_ibm,
    2: unpack.unpack_4byte_integer,
    3: unpack.unpack_2byte_integer,
    4: unpack.unpack_4byte_fixed_point,
    5: unpack.unpack_4byte_ieee,
    6: unpack.unpack_8byte_ieee,
    7: unpack.unpack_3byte_integer,
    8: unpack.unpack_1byte_integer,
    9: unpack.unpack_8byte_integer,
    10: unpack.unpack_4byte_uint,
    11: unpack.unpack_2byte_uint,
    12: unpack.unpack_8byte_uint,
    15: unpack.unpack_3byte_uint,
    16: unpack.unpack_1byte_uint
}

#: Functions that pack the chosen data format. The keys correspond to the
#: number given for each format by the SEG Y format reference.
DATA_SAMPLE_FORMAT_PACK_FUNCTIONS = {
    1: pack.pack_4byte_ibm,
    2: pack.pack_4byte_integer,
    3: pack.pack_2byte_integer,
    4: pack.pack_4byte_fixed_point,
    5: pack.pack_4byte_ieee,
    8: pack.pack_1byte_integer
}

#: Size of one sample.
DATA_SAMPLE_FORMAT_SAMPLE_SIZE = {
    1: 4,   # 4-byte IBM floating-point
    2: 4,   # 4-byte two's complement
    3: 2,   # 2-byte two's complement
    4: 4,   # obsolete
    5: 4,   # 4-byte IEEE floating-point
    6: 8,   # 8-byte IEEE floating-point
    7: 3,   # 3-byte two's complement
    8: 1,   # 1-byte two's complement
    9: 8,   # 8-byte two's complement
    10: 4,  # 4-byte unsigned integer
    11: 2,  # 2-byte unsigned integer
    12: 8,  # 8-byte unsigned integer
    15: 3,  # 3-byte unsigned integer
    16: 1   # 1-byte unsinged integer
}

#: Map the data format sample code and the corresponding dtype.
DATA_SAMPLE_FORMAT_CODE_DTYPE = {
    1: np.float32,  # 4-byte IBM floating-point to float32
    2: np.int32,    # 4-byte two's complement to int32
    3: np.int16,    # 2-byte two's complement to int16
    5: np.float32,  # 4-byte IEEE floating-point to float32
    6: np.float64,  # 4-byte IEEE floating-point to float64
    7: np.int32,    # 3-byte two's complement to int32
    8: np.int8,     # 1-byte two's complement to int8
    9: np.int64,    # 8-byte two's complement to int64
    10: np.uint32,  # 4-byte unsigned integer to uint32
    11: np.uint16,  # 2-byte unsigned integer to uint16
    12: np.uint64,  # 8-byte unsigned integer to uint64
    15: np.uint32,  # 3-byte unsigned integer to uint32
    16: np.uint8    # 1-byte unsinged integer to uint8
}

#: Map the endianness to bigger/smaller sign.
ENDIAN = {
    'big': '>',
    'little': '<',
    '>': '>',
    '<': '<'}
