# -*- coding: utf-8 -*-
"""
The Quality Control test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import numpy as np

import obspy
from obspy.core.util.base import NamedTemporaryFile
# A bit wild to import a utility function from another test suite ...
from obspy.io.mseed.tests.test_mseed_util import _create_mseed_file
from obspy.signal.quality_control import MSEEDMetadata


class QualityControlTestCase(unittest.TestCase):
    """
    Test cases for Quality Control.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), "data")

    def test_no_files_given(self):
        """
        Tests the raised exception if no file is given.
        """
        with self.assertRaises(ValueError) as e:
            MSEEDMetadata(files=[])

        self.assertEqual(e.exception.args[0],
                         "No data within the temporal constraints.")

    def test_gaps_and_overlaps(self):
        """
        Test gaps and overlaps.
        """
        # Create a file. No gap between 1 and 2, 10 second gap between 2 and
        # 3, 5 second overlap between 3 and 4, and another 10 second gap
        # between 4 and 5.
        tr_1 = obspy.Trace(data=np.arange(10, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(0)})
        tr_2 = obspy.Trace(data=np.arange(10, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(10)})
        tr_3 = obspy.Trace(data=np.arange(10, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(30)})
        tr_4 = obspy.Trace(data=np.arange(10, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(35)})
        tr_5 = obspy.Trace(data=np.arange(10, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(55)})
        st = obspy.Stream(traces=[tr_1, tr_2, tr_3, tr_4, tr_5])

        with NamedTemporaryFile() as tf:
            st.write(tf.name, format="mseed")

            mseed_metadata = MSEEDMetadata(files=[tf.name])
            self.assertEqual(mseed_metadata.meta['num_gaps'], 2)
            self.assertEqual(mseed_metadata.meta['num_overlaps'], 1)
            self.assertEqual(mseed_metadata.meta['sum_overlaps'], 5.0)
            self.assertEqual(mseed_metadata.meta['sum_gaps'], 20.0)
            self.assertEqual(mseed_metadata.meta['percent_availability'],
                             45.0 / 65.0 * 100.0)

            # Same again but this time with start-and end time settings.
            mseed_metadata = MSEEDMetadata(
                files=[tf.name], starttime=obspy.UTCDateTime(5),
                endtime=obspy.UTCDateTime(60))
            self.assertEqual(mseed_metadata.meta['num_gaps'], 2)
            self.assertEqual(mseed_metadata.meta['num_overlaps'], 1)
            self.assertEqual(mseed_metadata.meta['sum_overlaps'], 5.0)
            self.assertEqual(mseed_metadata.meta['sum_gaps'], 20.0)
            self.assertEqual(mseed_metadata.meta['percent_availability'],
                             35.0 / 55.0 * 100.0)

            # Head and tail gaps.
            mseed_metadata = MSEEDMetadata(
                    files=[tf.name], starttime=obspy.UTCDateTime(-10),
                    endtime=obspy.UTCDateTime(80))
            self.assertEqual(mseed_metadata.meta['num_gaps'], 4)
            self.assertEqual(mseed_metadata.meta['num_overlaps'], 1)
            self.assertEqual(mseed_metadata.meta['sum_overlaps'], 5.0)
            self.assertEqual(mseed_metadata.meta['sum_gaps'], 45.0)
            self.assertEqual(mseed_metadata.meta['percent_availability'],
                             45.0 / 90.0 * 100.0)

            # Tail gap must be larger than 1 delta, otherwise it does not
            # count.
            mseed_metadata = MSEEDMetadata(files=[tf.name],
                                           endtime=obspy.UTCDateTime(64))
            self.assertEqual(mseed_metadata.meta['num_gaps'], 2)
            self.assertEqual(mseed_metadata.meta['sum_gaps'], 20.0)
            self.assertEqual(mseed_metadata.meta['percent_availability'],
                             44.0 / 64.0 * 100.0)
            mseed_metadata = MSEEDMetadata(files=[tf.name],
                                           endtime=obspy.UTCDateTime(65))
            self.assertEqual(mseed_metadata.meta['num_gaps'], 2)
            self.assertEqual(mseed_metadata.meta['sum_gaps'], 20.0)
            self.assertEqual(mseed_metadata.meta['percent_availability'],
                             45.0 / 65.0 * 100.0)
            mseed_metadata = MSEEDMetadata(files=[tf.name],
                                           endtime=obspy.UTCDateTime(66))
            self.assertEqual(mseed_metadata.meta['num_gaps'], 3)
            self.assertEqual(mseed_metadata.meta['sum_gaps'], 21.0)
            self.assertEqual(mseed_metadata.meta['percent_availability'],
                             45.0 / 66.0 * 100.0)

    def test_gaps_between_multiple_files(self):
        """
        Test gap counting between multiple files. Simple test but there is
        no effective difference between having multiple files and a single
        one with many Traces as internally it is all parsed to a single
        Stream object.
        """
        with NamedTemporaryFile() as tf1, NamedTemporaryFile() as tf2:
            # Two files, same ids but a gap in-between.
            obspy.Trace(data=np.arange(10, dtype=np.int32),
                        header={"starttime": obspy.UTCDateTime(0)}).write(
                tf1.name, format="mseed")
            obspy.Trace(data=np.arange(10, dtype=np.int32),
                        header={"starttime": obspy.UTCDateTime(100)}).write(
                    tf2.name, format="mseed")
            # Don't calculate statistics on the single segments.
            mseed_metadata = MSEEDMetadata([tf1.name, tf2.name], c_seg=False)
            self.assertEqual(mseed_metadata.meta['num_gaps'], 1)
            self.assertNotIn("c_segments", mseed_metadata.meta)

    def test_file_with_no_timing_quality(self):
        """
        Tests timing quality extraction in files with no timing quality.
        """
        with NamedTemporaryFile() as tf1:
            obspy.Trace(data=np.arange(10, dtype=np.int32),
                        header={"starttime": obspy.UTCDateTime(0)}).write(
                    tf1.name, format="mseed")
            mseed_metadata = MSEEDMetadata([tf1.name], add_flags=True)
            ref = mseed_metadata.meta['miniseed_header_percentages']
            self.assertEqual(ref['timing_quality_max'],
                             None)
            self.assertEqual(ref['timing_quality_min'],
                             None)
            self.assertEqual(ref['timing_quality_mean'],
                             None)

    def test_extraction_of_basic_mseed_headers(self):
        """
        Tests extraction of basic features.
        """
        # Mixed files.
        with NamedTemporaryFile() as tf1, NamedTemporaryFile() as tf2:
            obspy.Trace(data=np.arange(10, dtype=np.int32),
                        header={"starttime": obspy.UTCDateTime(0),
                                "network": "BW", "station": "ALTM",
                                "location": "00", "channel": "EHE"}).write(
                    tf1.name, format="mseed", encoding="STEIM1", reclen=256)
            obspy.Trace(data=np.arange(10, dtype=np.float32),
                        header={"starttime": obspy.UTCDateTime(100),
                                "sampling_rate": 2.0, "network": "BW",
                                "station": "ALTM", "location": "00",
                                "channel": "EHE"}).write(
                    tf2.name, format="mseed", encoding="FLOAT32", reclen=1024)
            md = MSEEDMetadata([tf1.name, tf2.name])
            self.assertEqual(md.meta["network"], "BW")
            self.assertEqual(md.meta["station"], "ALTM")
            self.assertEqual(md.meta["location"], "00")
            self.assertEqual(md.meta["channel"], "EHE")
            self.assertEqual(md.meta["quality"], "D")
            self.assertEqual(md.meta["start_time"], obspy.UTCDateTime(0))
            self.assertEqual(md.meta["end_time"],
                             obspy.UTCDateTime(105))
            self.assertEqual(md.meta["num_records"], 2)
            self.assertEqual(md.meta["num_samples"], 20)
            self.assertEqual(md.meta["sample_rate"], [1.0, 2.0])
            self.assertEqual(md.meta["record_length"], [256, 1024])
            self.assertEqual(md.meta["encoding"], ["FLOAT32", "STEIM1"])

    def test_extraction_fixed_header_flags(self):
        with NamedTemporaryFile() as tf1, NamedTemporaryFile() as tf2:
            _create_mseed_file(tf1.name, record_count=35,
                               starttime=obspy.UTCDateTime(0),
                               seed=12345, flags={
                                'data_quality_flags': {
                                    "amplifier_saturation_detected": 25,
                                    "digitizer_clipping_detected": 12,
                                    "spikes_detected": 30,
                                    "glitches_detected": 6,
                                    "missing_data_present": 15,
                                    "telemetry_sync_error": 16,
                                    "digital_filter_charging": 4,
                                    "time_tag_uncertain": 8},
                                'activity_flags': {
                                    "calibration_signals_present": 10,
                                    "time_correction_applied": 20,
                                    "beginning_event": 33,
                                    "end_event": 33,
                                    "positive_leap": 34,
                                    "negative_leap": 10,
                                    "event_in_progress": 15},
                                'io_and_clock_flags': {
                                    "station_volume_parity_error": 8,
                                    "long_record_read": 33,
                                    "short_record_read": 24,
                                    "start_time_series": 31,
                                    "end_time_series": 24,
                                    "clock_locked": 32}})
            # Previous file ends exactly on 1750, start new file
            # to prevent overlapping records. When records overlap
            # their contributions should NOT be summed
            _create_mseed_file(tf2.name, record_count=23,
                               starttime=obspy.UTCDateTime(1750),
                               seed=12345, flags={
                                'data_quality_flags': {
                                    "amplifier_saturation_detected": 5,
                                    "digitizer_clipping_detected": 7,
                                    "spikes_detected": 5,
                                    "glitches_detected": 3,
                                    "missing_data_present": 5,
                                    "telemetry_sync_error": 3,
                                    "digital_filter_charging": 4,
                                    "time_tag_uncertain": 2},
                                'activity_flags': {
                                    "calibration_signals_present": 1,
                                    "time_correction_applied": 0,
                                    "beginning_event": 3,
                                    "end_event": 3,
                                    "positive_leap": 4,
                                    "negative_leap": 1,
                                    "event_in_progress": 5},
                                'io_and_clock_flags': {
                                    "station_volume_parity_error": 1,
                                    "long_record_read": 3,
                                    "short_record_read": 2,
                                    "start_time_series": 3,
                                    "end_time_series": 4,
                                    "clock_locked": 2}})

            md = MSEEDMetadata([tf1.name, tf2.name], add_flags=True)

            def _assert_float_equal(a, b):
                """
                Supplementary function to test floats to precision of 1E-6
                """
                self.assertTrue(abs(a - b) < 1E-6)

            # Sum up contributions from both files.
            # Check percentages
            meta = md.meta['miniseed_header_percentages']
            meta_dq = meta['data_quality_flags']
            _assert_float_equal(meta_dq['glitches'], 9/0.58)
            _assert_float_equal(meta_dq['amplifier_saturation'], 30/0.58)
            _assert_float_equal(meta_dq['digital_filter_charging'], 8/0.58)
            _assert_float_equal(meta_dq['digitizer_clipping'], 19/0.58)
            _assert_float_equal(meta_dq['missing_padded_data'], 20/0.58)
            _assert_float_equal(meta_dq['spikes'], 35/0.58)
            _assert_float_equal(meta_dq['suspect_time_tag'], 10/0.58)
            _assert_float_equal(meta_dq['telemetry_sync_error'], 19/0.58)

            meta_af = meta['activity_flags']
            _assert_float_equal(meta_af['calibration_signal'], 11/0.58)
            _assert_float_equal(meta_af['event_begin'], 36/0.58)
            _assert_float_equal(meta_af['event_end'], 36/0.58)
            _assert_float_equal(meta_af['event_in_progress'], 20/0.58)
            _assert_float_equal(meta_af['time_correction_applied'], 20/0.58)

            meta_io = meta['io_and_clock_flags']
            _assert_float_equal(meta_io['clock_locked'], 34/0.58)
            _assert_float_equal(meta_io['station_volume'], 9/0.58)
            _assert_float_equal(meta_io['long_record_read'], 36/0.58)
            _assert_float_equal(meta_io['short_record_read'], 26/0.58)
            _assert_float_equal(meta_io['start_time_series'], 34/0.58)
            _assert_float_equal(meta_io['end_time_series'], 28/0.58)

            # Check counts
            meta = md.meta['miniseed_header_counts']

            meta_dq = meta['data_quality_flags']
            self.assertEqual(meta_dq["glitches"], 9)
            self.assertEqual(meta_dq['amplifier_saturation'], 30)
            self.assertEqual(meta_dq['digital_filter_charging'], 8)
            self.assertEqual(meta_dq['digitizer_clipping'], 19)
            self.assertEqual(meta_dq['missing_padded_data'], 20)
            self.assertEqual(meta_dq['spikes'], 35)
            self.assertEqual(meta_dq['suspect_time_tag'], 10)
            self.assertEqual(meta_dq['telemetry_sync_error'], 19)

            meta_af = meta['activity_flags']
            self.assertEqual(meta_af['calibration_signal'], 11)
            self.assertEqual(meta_af['event_begin'], 36)
            self.assertEqual(meta_af['event_end'], 36)
            self.assertEqual(meta_af['event_in_progress'], 20)
            self.assertEqual(meta_af['time_correction_applied'], 20)

            meta_io = meta['io_and_clock_flags']
            self.assertEqual(meta_io['clock_locked'], 34)

            ref = md.meta['miniseed_header_percentages']
            self.assertEqual(ref['timing_quality_mean'], None)
            self.assertEqual(ref['timing_quality_min'], None)
            self.assertEqual(ref['timing_quality_max'], None)

            # Create a new file that is overlapping with the first
            _create_mseed_file(tf2.name, record_count=23,
                               starttime=obspy.UTCDateTime(400),
                               seed=12345, flags={
                                'data_quality_flags': {
                                    "amplifier_saturation_detected": 5,
                                    "digitizer_clipping_detected": 7,
                                    "spikes_detected": 5,
                                    "glitches_detected": 3,
                                    "missing_data_present": 5,
                                    "telemetry_sync_error": 3,
                                    "digital_filter_charging": 4,
                                    "time_tag_uncertain": 2},
                                'activity_flags': {
                                    "calibration_signals_present": 1,
                                    "time_correction_applied": 0,
                                    "beginning_event": 3,
                                    "end_event": 3,
                                    "positive_leap": 4,
                                    "negative_leap": 1,
                                    "event_in_progress": 5},
                                'io_and_clock_flags': {
                                    "station_volume_parity_error": 1,
                                    "long_record_read": 3,
                                    "short_record_read": 2,
                                    "start_time_series": 3,
                                    "end_time_series": 4,
                                    "clock_locked": 2}})

            md = MSEEDMetadata([tf1.name, tf2.name], add_flags=True)

            # Contributions should only come from the first file
            # Now overlap! Thus, length of file should be 35 records
            self.assertEqual(md.meta['end_time'] - md.meta['start_time'], 1750)
            self.assertEqual(35*50, 1750)

            # Check percentages of records in this window
            meta = md.meta['miniseed_header_percentages']
            meta_dq = meta['data_quality_flags']
            _assert_float_equal(meta_dq['glitches'], 6/0.35)
            _assert_float_equal(meta_dq['amplifier_saturation'], 25/0.35)
            _assert_float_equal(meta_dq['digital_filter_charging'], 4/0.35)
            _assert_float_equal(meta_dq['digitizer_clipping'], 12/0.35)
            _assert_float_equal(meta_dq['missing_padded_data'], 15/0.35)
            _assert_float_equal(meta_dq['spikes'], 30/0.35)
            _assert_float_equal(meta_dq['suspect_time_tag'], 8/0.35)
            _assert_float_equal(meta_dq['telemetry_sync_error'], 16/0.35)

            meta_af = meta['activity_flags']
            _assert_float_equal(meta_af['calibration_signal'], 10/0.35)
            _assert_float_equal(meta_af['event_begin'], 33/0.35)
            _assert_float_equal(meta_af['event_end'], 33/0.35)
            _assert_float_equal(meta_af['event_in_progress'], 15/0.35)
            _assert_float_equal(meta_af['time_correction_applied'], 20/0.35)

            meta_io = meta['io_and_clock_flags']
            _assert_float_equal(meta_io['clock_locked'], 32/0.35)
            _assert_float_equal(meta_io['station_volume'], 8/0.35)
            _assert_float_equal(meta_io['long_record_read'], 33/0.35)
            _assert_float_equal(meta_io['short_record_read'], 24/0.35)
            _assert_float_equal(meta_io['start_time_series'], 31/0.35)
            _assert_float_equal(meta_io['end_time_series'], 24/0.35)

    def test_timing_quality(self):
        """
        Test extraction of timing quality with a file that actually has it.
        """
        # Test file is constructed and orignally from the obspy.io.mseed
        # test suite.
        md = MSEEDMetadata(files=[os.path.join(self.path,
                                               "timingquality.mseed")],
                           add_flags=True)
        ref = md.meta['miniseed_header_percentages']
        self.assertEqual(ref['timing_quality_mean'], 50.0)
        self.assertEqual(ref['timing_quality_min'], 0.0)
        self.assertEqual(ref['timing_quality_max'], 100.0)
        self.assertEqual(ref['timing_quality_median'], 50.0)
        self.assertEqual(ref['timing_quality_lower_quartile'], 25.0)
        self.assertEqual(ref['timing_quality_upper_quartile'], 75.0)

    def test_overall_sample_metrics(self):
        """
        Tests the global metrics on the samples.
        """
        with NamedTemporaryFile() as tf:
            obspy.Trace(data=np.arange(10, dtype=np.int32),
                        header={"starttime": obspy.UTCDateTime(0)}).write(
                tf.name, format="mseed")

            md = MSEEDMetadata(files=[tf.name])

        self.assertEqual(md.meta["sample_min"], 0)
        self.assertEqual(md.meta["sample_max"], 9)
        self.assertEqual(md.meta["sample_mean"], 4.5)
        self.assertTrue(md.meta["sample_stdev"] - 2.8722813232 < 1E-6)
        self.assertTrue(md.meta["sample_rms"] - 5.33853912602 < 1E-6)
        self.assertTrue(md.meta["sample_median"], 4.5)

        # Make sure they also work if split up across two arrays.
        d = np.arange(10, dtype=np.int32)
        with NamedTemporaryFile() as tf1, NamedTemporaryFile() as tf2:
            obspy.Trace(data=d[:5],
                        header={"starttime": obspy.UTCDateTime(0)}).write(
                    tf1.name, format="mseed")
            obspy.Trace(data=d[5:],
                        header={"starttime": obspy.UTCDateTime(10)}).write(
                    tf2.name, format="mseed")

            md = MSEEDMetadata(files=[tf1.name, tf2.name])

        self.assertEqual(md.meta["sample_min"], 0)
        self.assertEqual(md.meta["sample_max"], 9)
        self.assertEqual(md.meta["sample_mean"], 4.5)
        self.assertTrue(md.meta["sample_stdev"] - 2.8722813232 < 1E-6)
        self.assertTrue(md.meta["sample_rms"] - 7.14142842854 < 1E-6)
        self.assertEqual(md.meta["sample_median"], 4.5)

    def test_root_mean_square(self):
        """
        Test the RMS calculation on a sine wave with amplitude sqrt(2)
        For this discrete sine wave RMS should be sqrt(2)/sqrt(2) = 1
        Within a certain precision
        """
        d = np.sqrt(2) * np.sin(np.linspace(-np.pi, np.pi, 10000))
        with NamedTemporaryFile() as tf1:
            obspy.Trace(data=d,
                        header={"starttime": obspy.UTCDateTime(10)}).write(
                    tf1.name, format="mseed")
            md = MSEEDMetadata([tf1.name])
        self.assertTrue(np.fabs(md.meta["sample_rms"] - 1) < 1E-3)
        for c_segment in md.meta["c_segments"]:
            self.assertTrue(np.fabs(c_segment["sample_rms"] - 1) < 1E-3)

    def test_int_overflow(self):
        """
        Tests calculations for an array of big numbers that will
        implicitly cause an int32 overflow in the RMS calculation
        For an array of y = a, RMS should a
        """
        d = np.empty(10000, dtype=np.int32)
        d.fill(np.iinfo(np.int32).max)
        with NamedTemporaryFile() as tf1:
            obspy.Trace(data=d,
                        header={"starttime": obspy.UTCDateTime(10)}).write(
                    tf1.name, format="mseed")
            md = MSEEDMetadata([tf1.name])
            self.assertTrue(md.meta["sample_rms"] == np.iinfo(np.int32).max)

    def test_overlap_fire_testing(self):
        """
        Fire tests at a rapid rate to test the overlap function
        Rapid overlap testing. Create the following stream:
        0 -- 1 -- 2 -- 3 -- 4 -- 5 -- 6 --
                       3 -- 4 -- 5 -- 6 -- 7 -- 8 -- 9 -- 10 --
                                              8 -- 9 -- 10 -- 11 --
        And shoot as many strange windows, check if gaps are calculated
        correctly. Add your own!
        """
        tr_1 = obspy.Trace(data=np.arange(7, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(0)})
        tr_2 = obspy.Trace(data=np.arange(8, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(3)})
        tr_3 = obspy.Trace(data=np.arange(4, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(7.5)})
        st = obspy.Stream(traces=[tr_1, tr_2, tr_3])
        with NamedTemporaryFile() as tf:

            st.write(tf.name, format="mseed")

            # Supplementary function to test overlaps rapidly with varying
            # start and endtimes
            def _rapid_overlap_testing(start, end):
                md = MSEEDMetadata(files=[tf.name],
                                   starttime=obspy.UTCDateTime(start),
                                   endtime=obspy.UTCDateTime(end))
                return md.meta['sum_overlaps']

            self.assertTrue(_rapid_overlap_testing(0, 12) == 7.0)
            self.assertTrue(_rapid_overlap_testing(3, 7) == 4.0)
            self.assertTrue(_rapid_overlap_testing(3, 5.5) == 2.5)
            self.assertTrue(_rapid_overlap_testing(4.5, 5.5) == 1.0)
            self.assertTrue(_rapid_overlap_testing(2, 5.25) == 2.25)
            self.assertTrue(_rapid_overlap_testing(2, 3) == 0.0)
            self.assertTrue(_rapid_overlap_testing(2, 3.1) == 0.1)
            self.assertTrue(_rapid_overlap_testing(7, 9) == 1.5)
            self.assertTrue(_rapid_overlap_testing(6.9, 9) == 1.6)
            self.assertTrue(_rapid_overlap_testing(4.30, 9) == 4.2)
            self.assertTrue(_rapid_overlap_testing(5.20, 9000) == 4.8)

    def test_gap_fire_testing(self):
        """
        Fire tests at a rapid rate to test the gap function
        Rapid gap testing. Create the following stream:
        0 -- 1 -- x -- x -- 4 -- x -- x -- 7 -- 8 -- x -- 10 -- 11 --
        And shoot as many strange windows, check if gaps are calculated
        correctly. Add your own!
        """
        tr_1 = obspy.Trace(data=np.arange(2, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(5)})
        tr_2 = obspy.Trace(data=np.arange(1, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(9)})
        tr_3 = obspy.Trace(data=np.arange(2, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(12)})
        tr_4 = obspy.Trace(data=np.arange(2, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(15)})
        st = obspy.Stream(traces=[tr_1, tr_2, tr_3, tr_4])
        with NamedTemporaryFile() as tf:

            st.write(tf.name, format="mseed")

            def _rapid_gap_testing(start, end):
                md = MSEEDMetadata(files=[tf.name],
                                   starttime=obspy.UTCDateTime(start),
                                   endtime=obspy.UTCDateTime(end))
                return md.meta['sum_gaps']

            self.assertTrue(_rapid_gap_testing(5, 17) == 5)
            self.assertTrue(_rapid_gap_testing(5, 10) == 2)
            self.assertTrue(_rapid_gap_testing(8.30, 9.5) == 0.70)
            self.assertTrue(_rapid_gap_testing(9, 12) == 2)
            self.assertTrue(_rapid_gap_testing(12, 17) == 1)
            self.assertTrue(_rapid_gap_testing(10, 13) == 2)
            self.assertTrue(_rapid_gap_testing(10.25, 13) == 1.75)
            self.assertTrue(_rapid_gap_testing(11.75, 17) == 1.25)
            self.assertTrue(_rapid_gap_testing(6, 10.5) == 2.5)
            self.assertTrue(_rapid_gap_testing(11.99, 12.01) == 0.01)
            self.assertTrue(_rapid_gap_testing(10.1, 12.01) == 1.9)
            self.assertTrue(_rapid_gap_testing(7.5, 14.25) == 3.75)
            self.assertTrue(_rapid_gap_testing(5, 17.5) == 5)
            self.assertTrue(_rapid_gap_testing(5, 17.6) == 5.6)
            self.assertTrue(_rapid_gap_testing(5, 18) == 6)
            self.assertTrue(_rapid_gap_testing(0, 5.01) == 5)
            self.assertTrue(_rapid_gap_testing(0, 20) == 13)

    def test_start_gap(self):
        """
        Tests whether a gap in the beginning of the file is interpreted
        A gap at the beginning of the window is ignored if a sample can be
        found before the starttime, and is continuous
        Trace in file runs from [00:01.625000Z to 00:59.300000Z]
        """
        file = os.path.join(self.path, "tiny_quality_file.mseed")

        # first sample is 625000, so we introduce a gap of 0.025
        starttime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 600000)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 300000)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)

        self.assertTrue(md.meta["num_gaps"] == 1)
        self.assertTrue(md.meta["sum_gaps"] == 0.025)

        # Start beyond a sample, but it is padded to the left, no gap
        starttime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 630000)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 300000)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        self.assertTrue(md.meta["num_gaps"] == 0)

        md = MSEEDMetadata([file])
        self.assertTrue(md.meta["num_gaps"] == 0)

    def test_random_window(self):
        """
        Tests a random window within a continuous trace, expect no gaps
        Continuous trace in file runs from [00:01.625000Z to 00:59.300000Z]
        """
        file = os.path.join(self.path, "tiny_quality_file.mseed")

        # Go randomly somewhere between two samples, find no gaps because
        # the trace is continuous everywhere between start-end
        starttime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 22, 646572)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 38, 265749)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        self.assertTrue(md.meta["num_gaps"] == 0)
        self.assertTrue(md.meta["num_overlaps"] == 0)

    def test_end_gap(self):
        """
        Test for the end gap. A gap should be found if the
        endtime exceeds the last sample + delta + time tolerance
        Trace in file runs from [00:01.625000Z to 00:59.300000Z]
        """
        file = os.path.join(self.path, "tiny_quality_file.mseed")

        # Last sample is at 300000, but this sample covers the trace
        # up to 300000 + delta (0.025) => 325000 - no gaps
        starttime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 625000)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 325000)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        self.assertTrue(md.meta["num_gaps"] == 0)
        self.assertTrue(md.meta["sum_gaps"] == 0.0)

        # Add time tolerance (1/2 sampling_rate) - no gaps
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 337500)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        self.assertTrue(md.meta["num_gaps"] == 0)
        self.assertTrue(md.meta["sum_gaps"] == 0.0)

        # Add 1μs; exceed projected sample plus time tolerance - GAP!
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 337501)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        self.assertTrue(md.meta["num_gaps"] == 1)
        self.assertTrue(md.meta["sum_gaps"] == 0.012501)

    def test_clock_locked_percentage(self):
        """
        7/10 records with io_flag clock_locked set to 1 for which we
        count the percentage of record time of total time
        """
        file = os.path.join(self.path, "tiny_quality_file.mseed")

        starttime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 625000)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 300000)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime,
                           add_flags=True)

        # These are the record lengths
        # The final record is cut off by the endtime
        record_lengths = [5.7, 6.05, 5.9, 5.55, 5.475, 5.725, 6.025, 5.775,
                          5.7, 5.775001]

        # Bit 5 is flagged for for these records: [0, 2, 3, 4, 5, 8, 9]
        record_lengths_flagged = [5.7, 5.9, 5.55, 5.475, 5.725, 5.7,
                                  5.775001]

        # Check if the record lenghts matches the total length
        total_time = md.meta["end_time"] - md.meta["start_time"]
        self.assertTrue(abs(sum(record_lengths) - (total_time)) < 1e-6)

        # Calculate the percentage of clock_locked seconds
        percentage = 100*sum(record_lengths_flagged)/sum(record_lengths)
        meta = md.meta["miniseed_header_percentages"]
        meta_io = meta['io_and_clock_flags']
        self.assertTrue(abs(meta_io["clock_locked"] - percentage) < 1e-6)

    def test_endtime_on_sample(self):
        """
        [T0, T1), T0 should be included; T1 excluded
        Test to see whether points on starttime are included, and
        samples on the endtime are excluded.
        Total number of samples for this test = 2308
        Trace in file runs from [00:01.625000Z to 00:59.300000Z]
        """
        file = os.path.join(self.path, "tiny_quality_file.mseed")

        # Set T0 and T1 on sample (N-1)
        starttime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 625000)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 300000)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        self.assertTrue(md.meta["num_samples"] == 2307)
        self.assertTrue(md.meta["end_time"] == endtime)

        # Set T0 on sample and T1 1μ after sample (N)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 300001)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        self.assertTrue(md.meta["num_samples"] == 2308)
        self.assertTrue(md.meta["end_time"] == endtime)

        # Set T0 and T1 1μ after sample (N-1)
        starttime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 625001)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        self.assertTrue(md.meta["num_samples"] == 2307)
        self.assertTrue(md.meta["start_time"] == starttime)

        # Set T0 1μ after sample and T1 on sample (N-2)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 300000)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        self.assertTrue(md.meta["num_samples"] == 2306)
        self.assertTrue(md.meta["start_time"] == starttime)

    def test_continuous_segments_sample_metrics(self):
        """
        Tests the metrics on each segment.
        """
        d = np.arange(10, dtype=np.int32)
        with NamedTemporaryFile() as tf1, NamedTemporaryFile() as tf2, \
                NamedTemporaryFile() as tf3:
            obspy.Trace(data=d[:5],
                        header={"starttime": obspy.UTCDateTime(0)}).write(
                    tf1.name, format="mseed")
            obspy.Trace(data=d[5:],
                        header={"starttime": obspy.UTCDateTime(10)}).write(
                    tf2.name, format="mseed")
            obspy.Trace(data=np.arange(10, dtype=np.int32),
                        header={"starttime": obspy.UTCDateTime(20)}).write(
                    tf3.name, format="mseed")

            md = MSEEDMetadata(files=[tf1.name, tf2.name, tf3.name])

        c_seg = md.meta["c_segments"]
        self.assertEqual(len(c_seg), 3)

        c = c_seg[0]
        self.assertEqual(c["start_time"], obspy.UTCDateTime(0))
        self.assertEqual(c["end_time"], obspy.UTCDateTime(5))
        self.assertEqual(c["sample_min"], 0)
        self.assertEqual(c["sample_max"], 4)
        self.assertEqual(c["sample_mean"], 2.0)
        self.assertTrue(c["sample_rms"] - 2.4494897427831779 < 1E-6)
        self.assertTrue(c["sample_stdev"] - 1.4142135623730951 < 1E-6)
        self.assertEqual(c["num_samples"], 5)
        self.assertEqual(c["seg_len"], 4.0)
        self.assertEqual(c["sample_median"], 2)

        c = c_seg[1]
        self.assertEqual(c["start_time"], obspy.UTCDateTime(10))
        self.assertEqual(c["end_time"], obspy.UTCDateTime(15))
        self.assertEqual(c["sample_min"], 5)
        self.assertEqual(c["sample_max"], 9)
        self.assertEqual(c["sample_mean"], 7.0)
        self.assertTrue(c["sample_rms"] - 7.1414284285428504 < 1E-6)
        self.assertTrue(c["sample_stdev"] - 1.4142135623730951 < 1E-6)
        self.assertEqual(c["num_samples"], 5)
        self.assertEqual(c["sample_median"], 7)

        c = c_seg[2]
        self.assertEqual(c["start_time"], obspy.UTCDateTime(20))
        self.assertEqual(c["end_time"], obspy.UTCDateTime(30))
        self.assertEqual(c["num_samples"], 10)
        self.assertEqual(c["seg_len"], 9.0)
        self.assertEqual(c["sample_min"], 0)
        self.assertEqual(c["sample_max"], 9)
        self.assertEqual(c["sample_mean"], 4.5)
        self.assertTrue(c["sample_stdev"] - 2.8722813232 < 1E-6)
        self.assertTrue(c["sample_rms"] - 5.3385391260156556 < 1E-6)
        self.assertEqual(c["sample_median"], 4.5)

    def test_json_serialization(self):
        """
        Just tests that it actually works and raises no error. We tested the
        dictionaries enough - now we just test the JSON encoder.
        """
        with NamedTemporaryFile() as tf:
            obspy.Trace(data=np.arange(10, dtype=np.int32),
                        header={"starttime": obspy.UTCDateTime(0)}).write(
                tf.name, format="mseed")

            md = MSEEDMetadata(files=[tf.name])

        self.assertTrue(md.get_json_meta())


def suite():
    return unittest.makeSuite(QualityControlTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
