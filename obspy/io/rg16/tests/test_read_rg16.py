# -*- coding: utf-8 -*-
"""
Tests for reading rg16 format.
"""
import io
import os
import unittest

import numpy as np
import obspy
import obspy.io.rg16.core as rc
from obspy import read
from obspy import UTCDateTime

TEST_FCNT_DIRECTORY = os.path.join(os.path.dirname(__file__), 'data')
ONE_CHAN_FCNT = os.path.join(TEST_FCNT_DIRECTORY,
                             'one_channel_many_traces.fcnt')
THREE_CHAN_FCNT = os.path.join(TEST_FCNT_DIRECTORY,
                               'three_chans_six_traces.fcnt')
BAD_ALIAS_FILTER = os.path.join(TEST_FCNT_DIRECTORY,
                                'channel_set_bad_alias_filter.dat')
HEADER_BLOCK_SAME_RU_CODE = os.path.join(TEST_FCNT_DIRECTORY,
                                         'header_3_chan_one_code.dat')
FCNT_FILES = [ONE_CHAN_FCNT, THREE_CHAN_FCNT]


class TestReadRG16(unittest.TestCase):

    def test_reading_rg16_files(self):
        """
        Ensure that the rg16 files are read by the function
        :func:`~obspy.core.stream.read` with or without specifying the format.
        """
        for fcnt_file in FCNT_FILES:
            st_1 = read(fcnt_file)
            st_2 = read(fcnt_file, format="RG16")
            st_3 = rc._read_rg16(fcnt_file)
            # when the function read is called a key "_format" is introduced
            # in the object stats. This key is not created when the function
            # _read_rg16 is called. In order to check the stream equality, the
            # key "_format" was removed.
            for tr_1, tr_2 in zip(st_1, st_2):
                del tr_1.stats._format
                del tr_2.stats._format
            self.assertTrue(st_1 == st_3)
            self.assertTrue(st_2 == st_3)

    def test_rg16_files_identified(self):
        """
        Ensure the rg16 files are correctly labeled as such.
        """
        for fcnt_file in FCNT_FILES:
            self.assertTrue(rc._is_rg16(fcnt_file))
            with open(fcnt_file, 'rb') as fi:
                self.assertTrue(rc._is_rg16(fi))

    def test_empty_buffer(self):
        """
        Ensure an empty buffer returns false.
        """
        buff = io.BytesIO()
        self.assertFalse(rc._is_rg16(buff))

    def test_headonly_option(self):
        """
        Ensure no data is returned when the option headonly is used.
        """
        st = rc._read_rg16(THREE_CHAN_FCNT, headonly=True)
        for tr in st:
            self.assertEqual(len(tr.data), 0)
            self.assertNotEqual(tr.stats.npts, 0)

    def test_starttime_endtime_option(self):
        """
        Test the options starttime and endtime
        """
        t1 = UTCDateTime(2017, 8, 9, 16, 0, 15)
        t2 = UTCDateTime(2017, 8, 9, 16, 0, 45)

        # read streams for testing. The three channel rg16 file has 6 traces
        # but the streams may have less depending on the starttime/endtime
        st = rc._read_rg16(THREE_CHAN_FCNT)  # no time filtering
        st1 = rc._read_rg16(THREE_CHAN_FCNT, starttime=t1, endtime=t2)

        # test using starttime and endtime
        self.assertEqual(len(st1), len(st))
        for tr, tr1 in zip(st, st1):
            self.assertEqual(tr.stats.starttime, tr1.stats.starttime)
            self.assertEqual(tr.stats.endtime, tr1.stats.endtime)

    def test_intrablock_starttime_endtime(self):
        """
        Test starttime/endtime options when starttime and endtime are comprised
        in a data block.
        """
        t1 = UTCDateTime(2017, 8, 9, 16, 0, 47)
        t2 = UTCDateTime(2017, 8, 9, 16, 0, 58)

        # read streams for testing
        st = rc._read_rg16(THREE_CHAN_FCNT)  # no time filtering
        st1 = rc._read_rg16(THREE_CHAN_FCNT, starttime=t1, endtime=t2)

        # test when starttime and endtime are comprised in a data packet.
        self.assertEqual(len(st1), 3)
        for tr, tr1 in zip(st[1::2], st1):
            self.assertEqual(tr.stats.starttime, tr1.stats.starttime)
            self.assertEqual(tr.stats.endtime, tr1.stats.endtime)

    def test_merge(self):
        """
        Ensure the merge option of read_rg16 merges all contiguous traces
        together.
        """
        for fcnt_file in FCNT_FILES:
            st_merged = rc._read_rg16(fcnt_file, merge=True)
            st = rc._read_rg16(fcnt_file).merge()
            self.assertEqual(len(st), len(st_merged))
            self.assertEqual(st, st_merged)

    def test_contacts_north_and_merge(self):
        """
        Ensure the "contacts_north" and "merge" parameters can be used
        together. See #2198.
        """
        for filename in FCNT_FILES:
            st = rc._read_rg16(filename, contacts_north=True, merge=True)
            assert isinstance(st, obspy.Stream)

    def test_make_stats(self):
        """
        Check function make_stats.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            stats = rc._make_stats(fi, 288, False, False)
        self.assertIsInstance(stats, obspy.core.trace.Stats)
        self.assertAlmostEqual(stats.sampling_rate, 500, delta=1e-5)
        self.assertEqual(stats.network, '1')
        self.assertEqual(stats.station, '1')
        self.assertEqual(stats.location, '1')
        self.assertEqual(stats.channel, 'DP3')
        self.assertEqual(stats.npts, 15000)
        self.assertEqual(stats.starttime,
                         UTCDateTime('2017-08-09T16:00:00.380000Z'))
        self.assertEqual(stats.endtime,
                         UTCDateTime('2017-08-09T16:00:30.378000Z'))
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            stats = rc._make_stats(fi, 288, True, False)
        self.assertEqual(stats.channel, 'DPN')


class TestReadRG16Headers(unittest.TestCase):

    def test_cmp_nbr_headers(self):
        """
        Test to check that the number of headers is correct.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            (nbr_channel_set_headers, nbr_extended_headers,
             nbr_external_headers) = rc._cmp_nbr_headers(fi)
        self.assertEqual(nbr_channel_set_headers, 3)
        self.assertEqual(nbr_extended_headers, 3)
        self.assertEqual(nbr_external_headers, 1)

    def test_cmp_nbr_records(self):
        """
        Check the number of records in the file.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            nbr_records = rc._cmp_nbr_records(fi)
        self.assertEqual(nbr_records, 6)
        with open(ONE_CHAN_FCNT, 'rb') as fi:
            nbr_records = rc._cmp_nbr_records(fi)
        self.assertEqual(nbr_records, 10)

    def test_cmp_jump(self):
        """
        Check the number of bytes to jump to reach the next trace block.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            jump = rc._cmp_jump(fi, 288)
        self.assertEqual(jump, 60340)

    def test_read_trace_header(self):
        """
        Test the reading of the trace header.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            trace_header = rc._read_trace_header(fi, 288)
        self.assertEqual(trace_header['trace_number'], 1)
        self.assertEqual(trace_header['trace_edit_code'], 0)

    def test_read_trace_header_1(self):
        """
        Test the reading of the trace header 1.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            trace_header_1 = rc._read_trace_header_1(fi, 288)
        self.assertEqual(trace_header_1['extended_receiver_line_nbr'], 65536)
        self.assertEqual(trace_header_1['extended_receiver_point_nbr'], 65536)
        self.assertEqual(trace_header_1['sensor_type'], 3)
        self.assertEqual(trace_header_1['trace_count_file'], 1)

    def test_read_trace_header_2(self):
        """
        Test the reading of the trace header 2.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            trace_header_2 = rc._read_trace_header_2(fi, 288)
        self.assertEqual(trace_header_2['shot_line_nbr'], 2240)
        self.assertEqual(trace_header_2['shot_point'], 1)
        self.assertEqual(trace_header_2['shot_point_index'], 0)
        self.assertAlmostEqual(trace_header_2['shot_point_pre_plan_x'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_2['shot_point_pre_plan_y'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_2['shot_point_final_x'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_2['shot_point_final_y'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_2['shot_point_final_depth'], 0,
                               delta=1e-5)
        self.assertEqual(trace_header_2['source_of_final_shot_info'],
                         'undefined')
        self.assertEqual(trace_header_2['energy_source_type'], 'undefined')

    def test_read_trace_header_3(self):
        """
        Test the reading of the trace header 3.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            trace_header_3 = rc._read_trace_header_3(fi, 288)
        self.assertEqual(trace_header_3['epoch_time'],
                         UTCDateTime('2017-08-09T16:00:00.380000Z'))
        self.assertAlmostEqual(trace_header_3['shot_skew_time'], 0, delta=1e-5)
        self.assertAlmostEqual(trace_header_3['time_shift_clock_correction'],
                               0, delta=1e-5)
        self.assertAlmostEqual(trace_header_3['remaining_clock_correction'],
                               0, delta=1e-5)

    def test_read_trace_header_4(self):
        """
        Test the reading of the trace header 4.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            trace_header_4 = rc._read_trace_header_4(fi, 288)
        self.assertAlmostEqual(trace_header_4['pre_shot_guard_band'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_4['post_shot_guard_band'], 0,
                               delta=1e-5)
        self.assertEqual(trace_header_4['preamp_gain'], 24)
        self.assertEqual(trace_header_4['trace_clipped_flag'], 'not clipped')
        self.assertEqual(trace_header_4['record_type_code'],
                         'normal seismic data record')
        self.assertEqual(trace_header_4['shot_status_flag'], 'normal')
        self.assertEqual(trace_header_4['external_shot_id'], 0)
        key = 'post_processed_first_break_pick_time'
        self.assertAlmostEqual(trace_header_4[key], 0, delta=1e-5)
        self.assertAlmostEqual(trace_header_4['post_processed_rms_noise'], 0,
                               delta=1e-5)

    def test_read_trace_header_5(self):
        """
        Test the reading of the trace header 5.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            trace_header_5 = rc._read_trace_header_5(fi, 288)
        self.assertAlmostEqual(trace_header_5['receiver_point_pre_plan_x'],
                               469567.2, delta=1e-5)
        self.assertAlmostEqual(trace_header_5['receiver_point_pre_plan_y'],
                               5280707.8, delta=1e-5)
        self.assertAlmostEqual(trace_header_5['receiver_point_final_x'],
                               469565.2, delta=1e-5)
        self.assertAlmostEqual(trace_header_5['receiver_point_final_y'],
                               5280709.7, delta=1e-5)
        self.assertAlmostEqual(trace_header_5['receiver_point_final_depth'], 0,
                               delta=1e-5)
        self.assertEqual(trace_header_5['source_of_final_receiver_info'],
                         'as laid (no navigation sensor)')

    def test_read_trace_header_6(self):
        """
        Test the reading of the trace header 6.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            trace_header_6 = rc._read_trace_header_6(fi, 288)
        self.assertAlmostEqual(trace_header_6['tilt_matrix_h1x'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_6['tilt_matrix_h2x'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_6['tilt_matrix_vx'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_6['tilt_matrix_h1y'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_6['tilt_matrix_h2y'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_6['tilt_matrix_vy'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_6['tilt_matrix_h1z'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_6['tilt_matrix_h2z'], 0,
                               delta=1e-5)

    def test_read_trace_header_7(self):
        """
        Test the reading of the trace header 7.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            trace_header_7 = rc._read_trace_header_7(fi, 288)
        self.assertAlmostEqual(trace_header_7['tilt_matrix_vz'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_7['azimuth_degree'], 0, delta=1e-5)
        self.assertAlmostEqual(trace_header_7['pitch_degree'], 0, delta=1e-5)
        self.assertAlmostEqual(trace_header_7['roll_degree'], 0, delta=1e-5)
        self.assertAlmostEqual(trace_header_7['remote_unit_temp'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(trace_header_7['remote_unit_humidity'], 0,
                               delta=1e-5)
        self.assertEqual(trace_header_7['orientation_matrix_version_nbr'], 0)
        self.assertEqual(trace_header_7['gimbal_corrections'], 0)

    def test_read_trace_header_8(self):
        """
        Test the reading of the trace header 8.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            trace_header_8 = rc._read_trace_header_8(fi, 288)
        self.assertEqual(trace_header_8['fairfield_test_analysis_code'], 0)
        self.assertEqual(trace_header_8['first_test_oscillator_attenuation'],
                         0)
        self.assertEqual(trace_header_8['second_test_oscillator_attenuation'],
                         0)
        self.assertAlmostEqual(trace_header_8['start_delay'], 0, delta=1e-5)
        self.assertEqual(trace_header_8['dc_filter_flag'], 0)
        self.assertAlmostEqual(trace_header_8['dc_filter_frequency'], 0,
                               delta=1e-5)
        self.assertEqual(trace_header_8['preamp_path'],
                         'external input selected')
        self.assertEqual(trace_header_8['test_oscillator_signal_type'],
                         'test oscillator path open')

    def test_read_trace_header_9(self):
        """
        Test the reading of the trace header 9.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            trace_header_9 = rc._read_trace_header_9(fi, 288)
        self.assertEqual(trace_header_9['test_signal_generator_signal_type'],
                         'pattern is address ramp')
        key = 'test_signal_generator_frequency_1'
        self.assertAlmostEqual(trace_header_9[key], 0, delta=1e-5)
        key = 'test_signal_generator_frequency_2'
        self.assertAlmostEqual(trace_header_9[key], 0, delta=1e-5)
        self.assertEqual(trace_header_9['test_signal_generator_amplitude_1'],
                         0)
        self.assertEqual(trace_header_9['test_signal_generator_amplitude_2'],
                         0)
        key = 'test_signal_generator_duty_cycle_percentage'
        self.assertAlmostEqual(trace_header_9[key], 0, delta=1e-5)
        key = 'test_signal_generator_active_duration'
        self.assertAlmostEqual(trace_header_9[key], 0, delta=1e-5)
        key = 'test_signal_generator_activation_time'
        self.assertAlmostEqual(trace_header_9[key], 0, delta=1e-5)

    def test_read_trace_header_10(self):
        """
        Test the reading of the trace header 10.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            trace_header_10 = rc._read_trace_header_10(fi, 288)
        self.assertEqual(trace_header_10['test_signal_generator_idle_level'],
                         0)
        self.assertEqual(trace_header_10['test_signal_generator_active_level'],
                         0)
        self.assertEqual(trace_header_10['test_signal_generator_pattern_1'], 0)
        self.assertEqual(trace_header_10['test_signal_generator_pattern_2'], 0)

    def test_make_trace(self):
        """
        Test if the ten first samples of the waveform are read correctly
        and if the output is a Trace object.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            trace_3 = rc._make_trace(fi, 288, False, False, False)
            trace_z = rc._make_trace(fi, 241648, False, True, False)
        expected = np.array([-0.18864873, -0.30852857, -0.35189095,
                             -0.22547323, -0.12023376, -0.14336781,
                             -0.11712314, 0.04060567, 0.18024819,
                             0.17769636])
        all_close = np.allclose(trace_3.data[:len(expected)], expected)
        self.assertTrue(all_close)
        self.assertIsInstance(trace_3, obspy.core.trace.Trace)
        expected = np.array([-0.673309, -0.71590775, -0.54966664, -0.33980238,
                             -0.29999766, -0.3031269, -0.12762846, 0.08782373,
                             0.11377038, 0.09888785])
        all_close = np.allclose(trace_z.data[:len(expected)], expected)
        self.assertTrue(all_close)

    def test_can_write(self):
        """
        Ensure the result of _read_rg16 is a stream object and that
        it can be written as mseed.
        """
        st = rc._read_rg16(THREE_CHAN_FCNT)
        self.assertIsInstance(st, obspy.core.stream.Stream)
        bytstr = io.BytesIO()
        # test passes if this doesn't raise
        try:
            st.write(bytstr, 'mseed')
        except Exception:
            self.fail('Failed to write to mseed!')

    def test_read_general_header_1(self):
        """
        Test the reading of the general header 1.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            header_1 = rc._read_general_header_1(fi)
        self.assertEqual(header_1['base_scan_interval'], 32)
        self.assertEqual(header_1['file_number'], 1)
        self.assertEqual(header_1['general_constant'], 0)
        self.assertEqual(header_1['julian_day'], 221)
        self.assertEqual(header_1['manufacturer_code'], 20)
        self.assertEqual(header_1['manufacturer_serial_number'], 0)
        self.assertEqual(header_1['nbr_add_general_header'], 1)
        self.assertEqual(header_1['nbr_channel_set'], 3)
        self.assertEqual(header_1['nbr_skew_block'], 0)
        self.assertEqual(header_1['polarity_code'], 0)
        self.assertEqual(header_1['record_type'], 0)
        self.assertEqual(header_1['sample_format_code'], 8058)
        self.assertEqual(header_1['scan_type_per_record'], 1)
        self.assertEqual(header_1['time_slice'], 160000)
        self.assertEqual(header_1['time_slice_year'], 17)

    def test_read_general_header_2(self):
        """
        Test the reading of the general header 2.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            header_2 = rc._read_general_header_2(fi)
        self.assertEqual(header_2['extended_channel_sets_per_scan_type'], 3)
        self.assertEqual(header_2['extended_file_number'], 1)
        self.assertEqual(header_2['extended_header_blocks'], 3)
        self.assertEqual(header_2['extended_record_length'], 30)
        self.assertEqual(header_2['external_header_blocks'], 1)
        self.assertEqual(header_2['general_header_block_number'], 2)
        self.assertEqual(header_2['version_number'], 262)

    def test_read_channel_sets(self):
        """
        Test that all the channel sets are read.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            channel_sets = rc._read_channel_sets(fi)
        self.assertEqual(len(channel_sets), 3)

    def test_read_channel_set(self):
        """
        Test the reading of the first channel set.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            channel_set_1 = rc._read_channel_set(fi, 64)
        self.assertEqual(channel_set_1['RU_channel_number'], 1)
        self.assertEqual(channel_set_1['alias_filter_frequency'], 207)
        self.assertEqual(channel_set_1['alias_filter_slope'], 320)
        self.assertEqual(channel_set_1['array_forming'], 0)
        self.assertAlmostEqual(channel_set_1['channel_set_end_time'],
                               30, delta=1e-5)
        self.assertEqual(channel_set_1['channel_set_number'], 1)
        self.assertAlmostEqual(channel_set_1['channel_set_start_time'],
                               0, delta=1e-5)
        self.assertEqual(channel_set_1['channel_type_code'], 1)
        self.assertEqual(channel_set_1['extended_channel_set_number'], 1)
        self.assertEqual(channel_set_1['extended_header_flag'], 0)
        self.assertEqual(channel_set_1['gain_control_type'], 3)
        self.assertEqual(channel_set_1['low_cut_filter_freq'], 0)
        self.assertEqual(channel_set_1['low_cut_filter_slope'], 6)
        self.assertEqual(channel_set_1['mp_factor_descaler_multiplier'], 0)
        self.assertEqual(channel_set_1['nbr_32_byte_trace_header_extension'],
                         10)
        self.assertEqual(channel_set_1['nbr_channels_in_channel_set'], 2)
        self.assertAlmostEqual(channel_set_1['notch_2_filter_freq'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(channel_set_1['notch_3_filter_freq'], 0,
                               delta=1e-5)
        self.assertAlmostEqual(channel_set_1['notch_filter_freq'], 0,
                               delta=1e-5)
        self.assertEqual(channel_set_1['optionnal_MP_factor'], 0)
        self.assertEqual(channel_set_1['scan_type_number'], 1)
        self.assertEqual(channel_set_1['vertical_stack_size'], 1)

    def test_read_extended_header_1(self):
        """
        Test the reading of the extended header 1.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            extended_header_1 = rc._read_extended_header_1(fi, 160)
        self.assertEqual(extended_header_1['deployment_time'],
                         UTCDateTime('2017-08-09T15:46:32.230000Z'))
        self.assertEqual(extended_header_1['id_ru'], 1219770716358969536)
        self.assertEqual(extended_header_1['pick_up_time'],
                         UTCDateTime('2017-08-09T20:06:58.120000Z'))
        self.assertEqual(extended_header_1['start_time_ru'],
                         UTCDateTime('2017-08-09T15:52:31.366000Z'))

    def test_read_extended_header_2(self):
        """
        Test the reading of the extended header 2.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            extended_header_2 = rc._read_extended_header_2(fi, 192)
        self.assertAlmostEqual(extended_header_2['acquisition_drift_window'],
                               0, delta=1e-5)
        self.assertAlmostEqual(extended_header_2['clock_drift'], 0, delta=1e-5)
        self.assertEqual(extended_header_2['clock_stop_method'], 'normal')
        self.assertEqual(extended_header_2['data_collection_method'],
                         'continuous')
        self.assertEqual(extended_header_2['data_decimation'], 'not decimated')
        self.assertEqual(extended_header_2['file_number'], 1)
        self.assertEqual(extended_header_2['frequency_drift'],
                         'within specification')
        self.assertEqual(extended_header_2['nbr_files'], 1)
        self.assertEqual(extended_header_2['nbr_time_slices'], 2)
        key = 'nbr_decimation_filter_coef'
        self.assertEqual(extended_header_2[key], 0)
        self.assertEqual(extended_header_2['original_base_scan_interval'], 0)
        self.assertEqual(extended_header_2['oscillator_type'], 'disciplined')

    def test_read_extended_header_3(self):
        """
        Test the reading of the extended header 3.
        """
        with open(THREE_CHAN_FCNT, 'rb') as fi:
            extended_header_3 = rc._read_extended_header_3(fi, 224)
        self.assertEqual(extended_header_3['first_shot_line'], 0)
        self.assertEqual(extended_header_3['first_shot_point'], 0)
        self.assertEqual(extended_header_3['first_shot_point_index'], 0)
        self.assertEqual(extended_header_3['last_shot_line'], 0)
        self.assertEqual(extended_header_3['last_shot_point'], 0)
        self.assertEqual(extended_header_3['last_shot_point_index'], 0)
        self.assertEqual(extended_header_3['receiver_line_number'], 1)
        self.assertEqual(extended_header_3['receiver_point'], 1)
        self.assertEqual(extended_header_3['receiver_point_index'], 1)

    def test_bad_alias_filter(self):
        """
        Tests for when alias filter is written as an int32 rather than BSD in
        the channel descriptor block.
        """
        with open(BAD_ALIAS_FILTER, 'rb') as fi:
            out = rc._read_channel_set(fi, 0)
        alias_freq = out['alias_filter_frequency']
        self.assertEqual(alias_freq, 207)

    def test_3_channel_header(self):
        """
        Tests for header which has three traces but identical RU channel
        number.
        """
        with open(HEADER_BLOCK_SAME_RU_CODE, 'rb') as fi:
            num_records = rc._cmp_nbr_records(fi)
        self.assertEqual(num_records, 2180 * 3)
