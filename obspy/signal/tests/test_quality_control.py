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

        self.assertEqual(
            e.exception.args[0],
            "Nothing added - no data within the given temporal constraints "
            "found.")

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
            self.assertEqual(mseed_metadata.meta['overlaps_len'], 5.0)
            self.assertEqual(mseed_metadata.meta['gaps_len'], 20.0)
            self.assertEqual(mseed_metadata.meta['percent_availability'],
                             44.0 / 64.0 * 100.0)

            # Same again but this time with start-and end time settings.
            mseed_metadata = MSEEDMetadata(
                files=[tf.name], starttime=obspy.UTCDateTime(5),
                endtime=obspy.UTCDateTime(60))
            self.assertEqual(mseed_metadata.meta['num_gaps'], 2)
            self.assertEqual(mseed_metadata.meta['num_overlaps'], 1)
            self.assertEqual(mseed_metadata.meta['overlaps_len'], 5.0)
            self.assertEqual(mseed_metadata.meta['gaps_len'], 20.0)
            self.assertEqual(mseed_metadata.meta['percent_availability'],
                             35.0 / 55.0 * 100.0)

            # Head and tail gaps.
            mseed_metadata = MSEEDMetadata(
                    files=[tf.name], starttime=obspy.UTCDateTime(-10),
                    endtime=obspy.UTCDateTime(80))
            self.assertEqual(mseed_metadata.meta['num_gaps'], 4)
            self.assertEqual(mseed_metadata.meta['num_overlaps'], 1)
            self.assertEqual(mseed_metadata.meta['overlaps_len'], 5.0)
            self.assertEqual(mseed_metadata.meta['gaps_len'], 45.0)
            self.assertEqual(mseed_metadata.meta['percent_availability'],
                             45.0 / 90.0 * 100.0)

            # Tail gap must be larger than 1 delta, otherwise it does not
            # count.
            mseed_metadata = MSEEDMetadata(files=[tf.name],
                                           endtime=obspy.UTCDateTime(64))
            self.assertEqual(mseed_metadata.meta['num_gaps'], 2)
            self.assertEqual(mseed_metadata.meta['gaps_len'], 20.0)
            self.assertEqual(mseed_metadata.meta['percent_availability'],
                             44.0 / 64.0 * 100.0)
            mseed_metadata = MSEEDMetadata(files=[tf.name],
                                           endtime=obspy.UTCDateTime(65))
            self.assertEqual(mseed_metadata.meta['num_gaps'], 2)
            self.assertEqual(mseed_metadata.meta['gaps_len'], 20.0)
            self.assertEqual(mseed_metadata.meta['percent_availability'],
                             45.0 / 65.0 * 100.0)
            mseed_metadata = MSEEDMetadata(files=[tf.name],
                                           endtime=obspy.UTCDateTime(66))
            self.assertEqual(mseed_metadata.meta['num_gaps'], 3)
            self.assertEqual(mseed_metadata.meta['gaps_len'], 21.0)
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
            mseed_metadata = MSEEDMetadata([tf1.name])
            self.assertEqual(mseed_metadata.meta['timing_quality']['max'],
                             None)
            self.assertEqual(mseed_metadata.meta['timing_quality']['min'],
                             None)
            self.assertEqual(mseed_metadata.meta['timing_quality']['mean'],
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
            self.assertEqual(md.meta['stats']["network"], "BW")
            self.assertEqual(md.meta['stats']["station"], "ALTM")
            self.assertEqual(md.meta['stats']["location"], "00")
            self.assertEqual(md.meta['stats']["channel"], "EHE")
            self.assertEqual(md.meta["quality"], "D")
            self.assertEqual(md.meta["start_time"], obspy.UTCDateTime(0))
            self.assertEqual(md.meta["end_time"],
                             obspy.UTCDateTime(104.5))
            self.assertEqual(md.meta["num_records"], 2)
            self.assertEqual(md.meta["num_samples"], 20)
            self.assertEqual(md.meta["sample_rate"], [1.0, 2.0])
            self.assertEqual(md.meta["record_len"], [256, 1024])
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

            md = MSEEDMetadata([tf1.name, tf2.name])
            # Sum up contributions from both files.
            self.assertEqual(md.meta['miniseed_header_flag_counts']['data_quality_flags']["glitches"], 9)
            self.assertEqual(md.meta['miniseed_header_flag_counts']['data_quality_flags']['amplifier_saturation'], 30)
            self.assertEqual(md.meta['miniseed_header_flag_counts']['data_quality_flags']['digital_filter_charging'], 8)
            self.assertEqual(md.meta['miniseed_header_flag_counts']['data_quality_flags']['digitizer_clipping'], 19)
            self.assertEqual(md.meta['miniseed_header_flag_counts']['data_quality_flags']['missing_padded_data'], 20)
            self.assertEqual(md.meta['miniseed_header_flag_counts']['data_quality_flags']['spikes'], 35)
            self.assertEqual(md.meta['miniseed_header_flag_counts']['data_quality_flags']['suspect_time_tag'], 10)
            self.assertEqual(md.meta['miniseed_header_flag_counts']['data_quality_flags']['telemetry_sync_error'], 19)
            self.assertEqual(md.meta['miniseed_header_flag_counts']['activity_flags']['calibration_signal'], 11)
            self.assertEqual(md.meta['miniseed_header_flag_counts']['activity_flags']['event_begin'], 36)
            self.assertEqual(md.meta['miniseed_header_flag_counts']['activity_flags']['event_end'], 36)
            self.assertEqual(md.meta['miniseed_header_flag_counts']['activity_flags']['event_in_progress'], 20)
            self.assertEqual(md.meta['miniseed_header_flag_counts']['activity_flags']['timing_correction'], 20)
            self.assertEqual(md.meta['miniseed_header_flag_counts']['io_and_clock_flags']['clock_locked'], 34)
            self.assertEqual(md.meta['timing_quality']['mean'], None)
            self.assertEqual(md.meta['timing_quality']['min'], None)
            self.assertEqual(md.meta['timing_quality']['max'], None)

    def test_timing_quality(self):
        """
        Test extraction of timing quality with a file that actually has it.
        """
        # Test file is constructed and orignally from the obspy.io.mseed
        # test suite.
        md = MSEEDMetadata(files=[os.path.join(self.path,
                                               "timingquality.mseed")])
        self.assertEqual(md.meta['timing_quality']['mean'], 50.0)
        self.assertEqual(md.meta['timing_quality']['min'], 0.0)
        self.assertEqual(md.meta['timing_quality']['max'], 100.0)
        self.assertEqual(md.meta['timing_quality']['median'], 50.0)
        self.assertEqual(md.meta['timing_quality']['lower_quartile'], 25.0)
        self.assertEqual(md.meta['timing_quality']['upper_quartile'], 75.0)

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
        self.assertEqual(c["end_time"], obspy.UTCDateTime(4))
        self.assertEqual(c["sample_min"], 0)
        self.assertEqual(c["sample_max"], 4)
        self.assertEqual(c["sample_mean"], 2.0)
        self.assertTrue(c["sample_rms"] - 2.4494897427831779 < 1E-6)
        self.assertTrue(c["sample_stdev"] - 1.4142135623730951 < 1E-6)
        self.assertEqual(c["num_samples"], 5)
        self.assertEqual(c["seg_len"], 4.0)

        c = c_seg[1]
        self.assertEqual(c["start_time"], obspy.UTCDateTime(10))
        self.assertEqual(c["end_time"], obspy.UTCDateTime(14))
        self.assertEqual(c["sample_min"], 5)
        self.assertEqual(c["sample_max"], 9)
        self.assertEqual(c["sample_mean"], 7.0)
        self.assertTrue(c["sample_rms"] - 7.1414284285428504 < 1E-6)
        self.assertTrue(c["sample_stdev"] - 1.4142135623730951 < 1E-6)
        self.assertEqual(c["num_samples"], 5)

        c = c_seg[2]
        self.assertEqual(c["start_time"], obspy.UTCDateTime(20))
        self.assertEqual(c["end_time"], obspy.UTCDateTime(29))
        self.assertEqual(c["num_samples"], 10)
        self.assertEqual(c["seg_len"], 9.0)
        self.assertEqual(c["sample_min"], 0)
        self.assertEqual(c["sample_max"], 9)
        self.assertEqual(c["sample_mean"], 4.5)
        self.assertTrue(c["sample_stdev"] - 2.8722813232 < 1E-6)
        self.assertTrue(c["sample_rms"] - 5.3385391260156556 < 1E-6)

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
