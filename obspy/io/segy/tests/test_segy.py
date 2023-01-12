# -*- coding: utf-8 -*-
"""
The obspy.io.segy test suite.
"""
import io
import re
import warnings
from unittest import mock

import numpy as np

import obspy
from obspy.core.compatibility import from_buffer
from obspy.core.util import NamedTemporaryFile, AttribDict
from obspy.core.util.testing import WarningsCapture
from obspy.io.segy.header import (DATA_SAMPLE_FORMAT_PACK_FUNCTIONS,
                                  DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS)
from obspy.io.segy.segy import (SEGYBinaryFileHeader, SEGYFile,
                                SEGYTraceHeader, _read_segy, iread_segy,
                                SEGYInvalidTextualHeaderWarning)
from obspy.io.segy.tests.header import DTYPES, FILES

from . import _patch_header
import pytest


class TestSEGY():
    """
    Test cases for SEG Y reading and writing..
    """
    def test_unpack_segy_data(self, testdata):
        """
        Tests the unpacking of various SEG Y files.
        """
        for file, attribs in FILES.items():
            data_format = attribs['data_sample_enc']
            endian = attribs['endian']
            count = attribs['sample_count']
            file = str(testdata[file])
            # Use the with statement to make sure the file closes.
            with open(file, 'rb') as f:
                # Jump to the beginning of the data.
                f.seek(3840)
                # Unpack the data.
                data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[data_format](
                    f, count, endian)
            # Check the dtype of the data.
            assert data.dtype == DTYPES[data_format]
            # Proven data values, read with Madagascar.
            correct_data = np.load(file + '.npy').ravel()
            # Compare both.
            np.testing.assert_array_equal(correct_data, data)

    def test_pack_segy_data(self, testdata):
        """
        Tests the packing of various SEG Y files.
        """
        # Loop over all files.
        for file, attribs in FILES.items():
            # Get some attributes.
            data_format = attribs['data_sample_enc']
            endian = attribs['endian']
            count = attribs['sample_count']
            size = attribs['sample_size']
            non_normalized_samples = attribs['non_normalized_samples']
            dtype = DTYPES[data_format]
            file = str(testdata[file])
            # Load the data. This data has previously been unpacked by
            # Madagascar.
            data = np.load(file + '.npy').ravel()
            data = np.require(data, dtype)
            # Load the packed data.
            with open(file, 'rb') as f:
                # Jump to the beginning of the data.
                f.seek(3200 + 400 + 240)
                packed_data = f.read(count * size)
            # The pack functions all write to file objects.
            f = io.BytesIO()
            # Pack the data.
            DATA_SAMPLE_FORMAT_PACK_FUNCTIONS[data_format](f, data, endian)
            # Read again.0.
            f.seek(0, 0)
            new_packed_data = f.read()
            # Check the length.
            assert len(packed_data) == len(new_packed_data)
            if len(non_normalized_samples) == 0:
                # The packed data should be totally identical.
                assert packed_data == new_packed_data
            else:
                # Some test files contain non normalized IBM floating point
                # data. These cannot be reproduced exactly.
                # Just a sanity check to be sure it is only IBM floating point
                # data that does not work completely.
                assert data_format == 1

                # Read the data as uint8 to be able to directly access the
                # different bytes.
                # Original data.
                packed_data = from_buffer(packed_data, np.uint8)
                # Newly written.
                new_packed_data = from_buffer(new_packed_data, np.uint8)

                # Figure out the non normalized fractions in the original data
                # because these cannot be compared directly.
                # Get the position of the first byte of the fraction depending
                # on the endianness.
                if endian == '>':
                    start = 1
                else:
                    start = 2
                # The first byte of the fraction.
                first_fraction_byte_old = packed_data[start::4]
                # First get all zeros in the original data because zeros have
                # to be treated differently.
                zeros = np.where(data == 0)[0]
                # Create a copy and set the zeros to a high number to be able
                # to find all non normalized numbers.
                fraction_copy = first_fraction_byte_old.copy()
                fraction_copy[zeros] = 255
                # Normalized numbers will have no zeros in the first 4 bit of
                # the fraction. This means that the most significant byte of
                # the fraction has to be at least 16 for it to be normalized.
                non_normalized = np.where(fraction_copy < 16)[0]

                # Sanity check if the file data and the calculated data are the
                # same.
                np.testing.assert_array_equal(non_normalized,
                                              np.array(non_normalized_samples))

                # Test all other parts of the packed data. Set dtype to int32
                # to get 4 byte numbers.
                packed_data_copy = packed_data.copy()
                new_packed_data_copy = new_packed_data.copy()
                packed_data_copy.dtype = np.int32
                new_packed_data_copy.dtype = np.int32
                # Equalize the non normalized parts.
                packed_data_copy[non_normalized] = \
                    new_packed_data_copy[non_normalized]
                np.testing.assert_array_equal(packed_data_copy,
                                              new_packed_data_copy)

                # Now check the non normalized parts if they are almost the
                # same.
                data = data[non_normalized]
                # Unpack the data again.
                new_packed_data.dtype = np.int32
                new_packed_data = new_packed_data[non_normalized]
                length = len(new_packed_data)
                f = io.BytesIO()
                f.write(new_packed_data.tobytes())
                f.seek(0, 0)
                new_data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[1](
                    f, length, endian)
                f.close()
                packed_data.dtype = np.int32
                packed_data = packed_data[non_normalized]
                length = len(packed_data)
                f = io.BytesIO()
                f.write(packed_data.tobytes())
                f.seek(0, 0)
                old_data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[1](
                    f, length, endian)
                f.close()
                # This works because the normalized and the non normalized IBM
                # floating point numbers will be close enough for the internal
                # IEEE representation to be identical.
                np.testing.assert_array_equal(data, new_data)
                np.testing.assert_array_equal(data, old_data)

    def test_pack_and_unpack_ibm_float(self):
        """
        Packing and unpacking IBM floating points might yield some inaccuracies
        due to floating point rounding errors.
        This test tests a large number of random floating point numbers.
        """
        # Some random seeds.
        seeds = [1234, 592, 459482, 6901, 0, 7083, 68349]
        endians = ['<', '>']
        # Loop over all combinations.
        for seed in seeds:
            # Generate 50000 random floats from -10000 to +10000.
            np.random.seed(seed)
            data = 200000.0 * np.random.ranf(50000) - 100000.0
            # Convert to float64 in case native floats are different to be
            # able to utilize double precision.
            data = np.require(data, np.float64)
            # Loop over little and big endian.
            for endian in endians:
                # Pack.
                f = io.BytesIO()
                DATA_SAMPLE_FORMAT_PACK_FUNCTIONS[1](f, data, endian)
                # Jump to beginning and read again.
                f.seek(0, 0)
                new_data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[1](
                    f, len(data), endian)
                f.close()
                # A relative tolerance of 1E-6 is considered good enough.
                rms1 = rms(data, new_data)
                assert True == (rms1 < 1E-6)

    def test_pack_and_unpack_very_small_ibm_floats(self):
        """
        The same test as test_packAndUnpackIBMFloat just for small numbers
        because they might suffer more from the inaccuracies.
        """
        # Some random seeds.
        seeds = [123, 1592, 4482, 601, 1, 783, 6849]
        endians = ['<', '>']
        # Loop over all combinations.
        for seed in seeds:
            # Generate 50000 random floats from -10000 to +10000.
            np.random.seed(seed)
            data = 1E-5 * np.random.ranf(50000)
            # Convert to float64 in case native floats are different to be
            # able to utilize double precision.
            data = np.require(data, np.float64)
            # Loop over little and big endian.
            for endian in endians:
                # Pack.
                f = io.BytesIO()
                DATA_SAMPLE_FORMAT_PACK_FUNCTIONS[1](f, data, endian)
                # Jump to beginning and read again.
                f.seek(0, 0)
                new_data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[1](
                    f, len(data), endian)
                f.close()
                # A relative tolerance of 1E-6 is considered good enough.
                rms1 = rms(data, new_data)
                assert True == (rms1 < 1E-6)

    def test_pack_and_unpack_ibm_special_cases(self):
        """
        Tests the packing and unpacking of several powers of 16 which are
        problematic because they need separate handling in the algorithm.
        """
        endians = ['>', '<']
        # Create the first 10 powers of 16.
        data = []
        for i in range(10):
            data.append(16 ** i)
            data.append(-16 ** i)
        data = np.array(data)
        # Convert to float64 in case native floats are different to be
        # able to utilize double precision.
        data = np.require(data, np.float64)
        # Loop over little and big endian.
        for endian in endians:
            # Pack.
            f = io.BytesIO()
            DATA_SAMPLE_FORMAT_PACK_FUNCTIONS[1](f, data, endian)
            # Jump to beginning and read again.
            f.seek(0, 0)
            new_data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[1](
                f, len(data), endian)
            f.close()
            # Test both.
            np.testing.assert_array_equal(new_data, data)

    def test_read_and_write_binary_file_header(self, testdata):
        """
        Reading and writing should not change the binary file header.
        """
        for file, attribs in FILES.items():
            endian = attribs['endian']
            file = testdata[file]
            # Read the file.
            with open(file, 'rb') as f:
                f.seek(3200)
                org_header = f.read(400)
            header = SEGYBinaryFileHeader(header=org_header, endian=endian)
            # The header writes to a file like object.
            new_header = io.BytesIO()
            header.write(new_header)
            new_header.seek(0, 0)
            new_header = new_header.read()
            # Assert the correct length.
            assert len(new_header) == 400
            # Assert the actual header.
            assert org_header == new_header

    def test_read_and_write_textual_file_header(self, testdata):
        """
        Reading and writing should not change the textual file header.
        """
        for file, attribs in FILES.items():
            endian = attribs['endian']
            header_enc = attribs['textual_header_enc']
            file = testdata[file]
            # Read the file.
            with open(file, 'rb') as f:
                org_header = f.read(3200)
                f.seek(0, 0)
                # Initialize an empty SEGY object and set certain attributes.
                segy = SEGYFile()
                segy.endian = endian
                segy.file = f
                segy.textual_header_encoding = None
                # Read the textual header.
                segy._read_textual_header()
                # Assert the encoding and compare with known values.
                assert segy.textual_header_encoding == header_enc
            # The header writes to a file like object.
            new_header = io.BytesIO()
            with WarningsCapture():
                segy._write_textual_header(new_header)
            new_header.seek(0, 0)
            new_header = new_header.read()
            # Assert the correct length.
            assert len(new_header) == 3200
            # Patch both headers to not worry about the automatically set
            # values.
            org_header = _patch_header(org_header)
            new_header = _patch_header(new_header)
            # Assert the actual header.
            assert org_header == new_header

    def test_read_and_write_trace_header(self, testdata):
        """
        Reading and writing should not change the trace header.
        """
        for file, attribs in FILES.items():
            endian = attribs['endian']
            file = testdata[file]
            # Read the file.
            with open(file, 'rb') as f:
                f.seek(3600)
                org_header = f.read(240)
            header = SEGYTraceHeader(header=org_header, endian=endian)
            # The header writes to a file like object.
            new_header = io.BytesIO()
            header.write(new_header)
            new_header.seek(0, 0)
            new_header = new_header.read()
            # Assert the correct length.
            assert len(new_header) == 240
            # Assert the actual header.
            assert org_header == new_header

    @pytest.mark.parametrize("headonly", (True, False))
    def test_read_and_write_segy(self, testdata, headonly):
        """
        Reading and writing again should not change a file.
        """
        for file, attribs in FILES.items():
            file = testdata[file]
            non_normalized_samples = attribs['non_normalized_samples']
            # Read the file.
            with open(file, 'rb') as f:
                org_data = f.read()
            segy_file = _read_segy(file, headonly=headonly)
            with NamedTemporaryFile() as tf:
                out_file = tf.name
                with WarningsCapture():
                    segy_file.write(out_file)
                # Read the new file again.
                with open(out_file, 'rb') as f:
                    new_data = f.read()
            # The two files should have the same length.
            assert len(org_data) == len(new_data)
            # Replace the not normalized samples. The not normalized
            # samples are already tested in test_packSEGYData and therefore not
            # tested again here.
            if len(non_normalized_samples) != 0:
                # Convert to 4 byte integers. Any 4 byte numbers work.
                org_data = from_buffer(org_data, np.int32)
                new_data = from_buffer(new_data, np.int32)
                # Skip the header (4*960 bytes) and replace the non normalized
                # data samples.
                org_data[960:][non_normalized_samples] = \
                    new_data[960:][non_normalized_samples]
                # Create strings again.
                org_data = org_data.tobytes()
                new_data = new_data.tobytes()
            # Just patch both headers - this tests something different.
            org_data = _patch_header(org_data)
            new_data = _patch_header(new_data)
            # Always write the SEGY File revision number!
            # org_data[3500:3502] = new_data[3500:3502]
            # Test the identity without the SEGY revision number
            assert org_data[:3500] == new_data[:3500]
            assert org_data[3502:] == new_data[3502:]

    def test_unpack_binary_file_header(self, testdata):
        """
        Compares some values of the binary header with values read with
        SeisView 2 by the DMNG.
        """
        file = testdata['1.sgy_first_trace']
        segy = _read_segy(file)
        header = segy.binary_file_header
        # Compare the values.
        assert header.job_identification_number == 0
        assert header.line_number == 0
        assert header.reel_number == 0
        assert header.number_of_data_traces_per_ensemble == 24
        assert header.number_of_auxiliary_traces_per_ensemble == 0
        assert header.sample_interval_in_microseconds == 250
        assert getattr(
            header,
            "sample_interval_in_microseconds_of_original_field_recording") \
            == 250
        assert header.number_of_samples_per_data_trace == 8000
        assert header. \
            number_of_samples_per_data_trace_for_original_field_recording == \
            8000
        assert header.data_sample_format_code == 2
        assert header.ensemble_fold == 0
        assert header.trace_sorting_code == 1
        assert header.vertical_sum_code == 0
        assert header.sweep_frequency_at_start == 0
        assert header.sweep_frequency_at_end == 0
        assert header.sweep_length == 0
        assert header.sweep_type_code == 0
        assert header.trace_number_of_sweep_channel == 0
        assert header.sweep_trace_taper_length_in_ms_at_start == 0
        assert header.sweep_trace_taper_length_in_ms_at_end == 0
        assert header.taper_type == 0
        assert header.correlated_data_traces == 0
        assert header.binary_gain_recovered == 0
        assert header.amplitude_recovery_method == 0
        assert header.measurement_system == 0
        assert header.impulse_signal_polarity == 0
        assert header.vibratory_polarity_code == 0
        assert header.number_of_3200_byte_ext_file_header_records_following \
            == 0

    def test_unpack_trace_header(self, testdata):
        """
        Compares some values of the first trace header with values read with
        SeisView 2 by the DMNG.
        """
        file = testdata['1.sgy_first_trace']
        segy = _read_segy(file)
        header = segy.traces[0].header
        # Compare the values.
        assert header.trace_sequence_number_within_line == 0
        assert header.trace_sequence_number_within_segy_file == 0
        assert header.original_field_record_number == 1
        assert header.trace_number_within_the_original_field_record == 1
        assert header.energy_source_point_number == 0
        assert header.ensemble_number == 0
        assert header.trace_number_within_the_ensemble == 0
        assert header.trace_identification_code == 1
        assert \
            header.number_of_vertically_summed_traces_yielding_this_trace == \
            5
        assert \
            header.number_of_horizontally_stacked_traces_yielding_this_trace \
            == 0
        assert header.data_use == 0
        assert getattr(
            header, 'distance_from_center_of_the_' +
            'source_point_to_the_center_of_the_receiver_group') == 0
        assert header.receiver_group_elevation == 0
        assert header.surface_elevation_at_source == 0
        assert header.source_depth_below_surface == 0
        assert header.datum_elevation_at_receiver_group == 0
        assert header.datum_elevation_at_source == 0
        assert header.water_depth_at_source == 0
        assert header.water_depth_at_group == 0
        assert header.scalar_to_be_applied_to_all_elevations_and_depths == -100
        assert header.scalar_to_be_applied_to_all_coordinates == -100
        assert header.source_coordinate_x == 0
        assert header.source_coordinate_y == 0
        assert header.group_coordinate_x == 300
        assert header.group_coordinate_y == 0
        assert header.coordinate_units == 0
        assert header.weathering_velocity == 0
        assert header.subweathering_velocity == 0
        assert header.uphole_time_at_source_in_ms == 0
        assert header.uphole_time_at_group_in_ms == 0
        assert header.source_static_correction_in_ms == 0
        assert header.group_static_correction_in_ms == 0
        assert header.total_static_applied_in_ms == 0
        assert header.lag_time_A == 0
        assert header.lag_time_B == 0
        assert header.delay_recording_time == -100
        assert header.mute_time_start_time_in_ms == 0
        assert header.mute_time_end_time_in_ms == 0
        assert header.number_of_samples_in_this_trace == 8000
        assert header.sample_interval_in_ms_for_this_trace == 250
        assert header.gain_type_of_field_instruments == 0
        assert header.instrument_gain_constant == 24
        assert header.instrument_early_or_initial_gain == 0
        assert header.correlated == 0
        assert header.sweep_frequency_at_start == 0
        assert header.sweep_frequency_at_end == 0
        assert header.sweep_length_in_ms == 0
        assert header.sweep_type == 0
        assert header.sweep_trace_taper_length_at_start_in_ms == 0
        assert header.sweep_trace_taper_length_at_end_in_ms == 0
        assert header.taper_type == 0
        assert header.alias_filter_frequency == 1666
        assert header.alias_filter_slope == 0
        assert header.notch_filter_frequency == 0
        assert header.notch_filter_slope == 0
        assert header.low_cut_frequency == 0
        assert header.high_cut_frequency == 0
        assert header.low_cut_slope == 0
        assert header.high_cut_slope == 0
        assert header.year_data_recorded == 2005
        assert header.day_of_year == 353
        assert header.hour_of_day == 15
        assert header.minute_of_hour == 7
        assert header.second_of_minute == 54
        assert header.time_basis_code == 0
        assert header.trace_weighting_factor == 0
        assert header.geophone_group_number_of_roll_switch_position_one == 2
        assert header.geophone_group_number_of_trace_number_one == 2
        assert header.geophone_group_number_of_last_trace == 0
        assert header.gap_size == 0
        assert header.over_travel_associated_with_taper == 0
        assert header.x_coordinate_of_ensemble_position_of_this_trace == 0
        assert header.y_coordinate_of_ensemble_position_of_this_trace == 0
        assert \
            header.for_3d_poststack_data_this_field_is_for_in_line_number == 0
        assert \
            header.for_3d_poststack_data_this_field_is_for_cross_line_number \
            == 0
        assert header.shotpoint_number == 0
        assert header.scalar_to_be_applied_to_the_shotpoint_number == 0
        assert header.trace_value_measurement_unit == 0
        assert header.transduction_constant_mantissa == 0
        assert header.transduction_constant_exponent == 0
        assert header.transduction_units == 0
        assert header.device_trace_identifier == 0
        assert header.scalar_to_be_applied_to_times == 0
        assert header.source_type_orientation == 0
        assert header.source_energy_direction_mantissa == 0
        assert header.source_energy_direction_exponent == 0
        assert header.source_measurement_mantissa == 0
        assert header.source_measurement_exponent == 0
        assert header.source_measurement_unit == 0

    def test_read_bytes_io(self, testdata):
        """
        Tests reading from BytesIO instances.
        """
        # 1
        file = testdata['example.y_first_trace']
        with open(file, 'rb') as f:
            data = f.read()
        st = _read_segy(io.BytesIO(data))
        assert len(st.traces[0].data) == 500
        # 2
        file = testdata['ld0042_file_00018.sgy_first_trace']
        with open(file, 'rb') as f:
            data = f.read()
        st = _read_segy(io.BytesIO(data))
        assert len(st.traces[0].data) == 2050
        # 3
        file = testdata['1.sgy_first_trace']
        with open(file, 'rb') as f:
            data = f.read()
        st = _read_segy(io.BytesIO(data))
        assert len(st.traces[0].data) == 8000
        # 4
        file = testdata['00001034.sgy_first_trace']
        with open(file, 'rb') as f:
            data = f.read()
        st = _read_segy(io.BytesIO(data))
        assert len(st.traces[0].data) == 2001
        # 5
        file = testdata['planes.segy_first_trace']
        with open(file, 'rb') as f:
            data = f.read()
        st = _read_segy(io.BytesIO(data))
        assert len(st.traces[0].data) == 512

    def test_iterative_reading(self, testdata):
        """
        Tests iterative reading.
        """
        # Read normally.
        filename = testdata['example.y_first_trace']
        st = obspy.read(filename, unpack_trace_headers=True)

        # Read iterative.
        ist = [_i for _i in iread_segy(filename, unpack_headers=True)]

        del ist[0].stats.segy.textual_file_header
        del ist[0].stats.segy.binary_file_header
        del ist[0].stats.segy.textual_file_header_encoding
        del ist[0].stats.segy.data_encoding
        del ist[0].stats.segy.endian

        assert st.traces == ist

    def test_revision_number_in_binary_file_header(self):
        """
        It is a bit awkward but it is encoded in a 16 bit number with the
        dot assumed to be between the first and second byte. So the hex 16
        representation is 0x0100.
        """
        tr = obspy.read()[0]
        tr.data = np.float32(tr.data)
        for endian in ("<", ">"):
            _tr = tr.copy()
            with io.BytesIO() as buf:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _tr.write(buf, format="segy", byteorder=endian)
                buf.seek(0, 0)
                data = buf.read()
            # Result differs depending on byte order.
            if endian == "<":
                assert data[3200:3600][-100:-98] == b"\x00\x01"
            else:
                assert data[3200:3600][-100:-98] == b"\x01\x00"

    def test_textual_header_has_the_right_fields_at_the_end(self):
        """
        Needs to have the version number and a magical string at the end.
        """
        tr = obspy.read()[0]
        tr.data = np.float32(tr.data)

        # If both are missing they will be replaced.
        with io.BytesIO() as buf:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                tr.write(buf, format="segy")
            assert len([
                _i for _i in w
                if _i.category is SEGYInvalidTextualHeaderWarning]) == 0
            buf.seek(0, 0)
            data = buf.read()

        # Make sure the textual header has the required fields.
        revision_number = data[:3200][-160:-146].decode()
        end_header_mark = data[:3200][-80:-58].decode()
        assert revision_number == "C39 SEG Y REV1"
        assert end_header_mark == "C40 END TEXTUAL HEADER"

        # An alternate end marker is accepted - no warning will be raised in
        # this case.
        # If both are missing they will be replaced.
        st = obspy.Stream(traces=[tr.copy()])
        st.stats = AttribDict()
        # Write the alternate file header.
        st.stats.textual_file_header = \
            b" " * (3200 - 80) + b"C40 END EBCDIC        " + b" " * 58
        with io.BytesIO() as buf:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                st.write(buf, format="segy", textual_header_encoding="EBCDIC")
            assert len([
                _i for _i in w
                if _i.category is SEGYInvalidTextualHeaderWarning]) == 0
            buf.seek(0, 0)
            data = buf.read()

        # Make sure the textual header has the required fields.
        revision_number = data[:3200][-160:-146].decode("EBCDIC-CP-BE")
        end_header_mark = data[:3200][-80:-58].decode("EBCDIC-CP-BE")
        assert revision_number == "C39 SEG Y REV1"
        assert end_header_mark == "C40 END EBCDIC        "

        # Putting the correct values will raise no warning and leave the
        # values.
        st = obspy.Stream(traces=[tr.copy()])
        st.stats = AttribDict()
        # Write the alternate file header.
        st.stats.textual_file_header =  \
            b" " * (3200 - 160) + b"C39 SEG Y REV1" + b" " * 66 + \
            b"C40 END TEXTUAL HEADER" + b" " * 58
        with io.BytesIO() as buf:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                st.write(buf, format="segy")
            assert len([
                _i for _i in w
                if _i.category is SEGYInvalidTextualHeaderWarning]) == 0
            buf.seek(0, 0)
            data = buf.read()

        # Make sure the textual header has the required fields.
        revision_number = data[:3200][-160:-146].decode()
        end_header_mark = data[:3200][-80:-58].decode()
        assert revision_number == "C39 SEG Y REV1"
        assert end_header_mark == "C40 END TEXTUAL HEADER"

        # Putting a wrong revision number will raise a warning, but it will
        # still be written.
        st = obspy.Stream(traces=[tr.copy()])
        st.stats = AttribDict()
        # Write the alternate file header.
        st.stats.textual_file_header = \
            b" " * (3200 - 160) + b"ABCDEFGHIJKLMN" + b" " * 146
        with io.BytesIO() as buf:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                st.write(buf, format="segy")
            w = [_i for _i in w
                 if _i.category is SEGYInvalidTextualHeaderWarning]
            assert len(w) == 1
            assert w[0].message.args[0] == (
                "The revision number in the textual header should be set as "
                "'C39 SEG Y REV1' for a fully valid SEG-Y file. It is set to "
                "'ABCDEFGHIJKLMN' which will be written to the file. Please "
                "change it if you want a fully valid file.")
            buf.seek(0, 0)
            data = buf.read()

        revision_number = data[:3200][-160:-146].decode()
        end_header_mark = data[:3200][-80:-58].decode()
        # The field is still written.
        assert revision_number == "ABCDEFGHIJKLMN"
        assert end_header_mark == "C40 END TEXTUAL HEADER"

        # Same with the end header mark.
        st = obspy.Stream(traces=[tr.copy()])
        st.stats = AttribDict()
        # Write the alternate file header.
        st.stats.textual_file_header = \
            b" " * (3200 - 80) + b"ABCDEFGHIJKLMNOPQRSTUV" + b" " * 58
        with io.BytesIO() as buf:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                st.write(buf, format="segy")
            w = [_i for _i in w
                 if _i.category is SEGYInvalidTextualHeaderWarning]
            assert len(w) == 1
            assert w[0].message.args[0] == (
                "The end header mark in the textual header should be set as "
                "'C40 END TEXTUAL HEADER' or as 'C40 END EBCDIC        ' for "
                "a fully valid SEG-Y file. It is "
                "set to 'ABCDEFGHIJKLMNOPQRSTUV' which will be written to the "
                "file. Please change it if you want a fully valid file.")
            buf.seek(0, 0)
            data = buf.read()

        revision_number = data[:3200][-160:-146].decode()
        end_header_mark = data[:3200][-80:-58].decode()
        # The field is still written.
        assert revision_number == "C39 SEG Y REV1"
        assert end_header_mark == "ABCDEFGHIJKLMNOPQRSTUV"

    def test_packing_raises_nice_error_messages(self):
        """
        SEG-Y is fairly restrictive in what ranges are allowed for its header
        values. Thus we attempt to raise nice and helpful error messages.
        """
        tr = obspy.read()[0]
        tr.data = np.float32(np.zeros(3))
        # mock the number of traces per ensemble with an invalid number to
        # trigger error message. we have a more meaningful message for number
        # of samples now, so we can't use that anymore for testing any
        # arbitrary invalid binary header value
        msg = re.escape(
            "Failed to pack header value `number_of_data_traces_per_ensemble` "
            "(100000) with format `>h` due to: `'h' format requires -32768 <="
            " number <= 32767`")
        with mock.patch.object(
                SEGYBinaryFileHeader, 'number_of_data_traces_per_ensemble',
                create=True, new_callable=mock.PropertyMock,
                return_value=100000):
            with io.BytesIO() as buf:
                with pytest.raises(ValueError, match=msg):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tr.write(buf, format="segy")


def rms(x, y):
    """
    Normalized RMS

    Taken from the mtspec library:
    https://github.com/krischer/mtspec
    """
    return np.sqrt(((x - y) ** 2).mean() / (x ** 2).mean())
