# -*- coding: utf-8 -*-
"""
The Quality Control test suite.
"""
import numpy as np

import obspy
from obspy.core.util.base import NamedTemporaryFile, get_dependency_version
# A bit wild to import a utility function from another test suite ...
from obspy.io.mseed.tests.test_mseed_util import _create_mseed_file
from obspy.signal.quality_control import MSEEDMetadata
import pytest

try:
    import jsonschema  # NOQA
    # 1.0.0 is the first version with full $ref support.
    if get_dependency_version("jsonschema") < [1, 0, 0]:
        HAS_JSONSCHEMA = False
    else:
        HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


class TestQualityControl():
    """
    Test cases for Quality Control.
    """
    def test_no_files_given(self):
        """
        Tests the raised exception if no file is given.
        """
        msg = "No data within the temporal constraints."
        with pytest.raises(ValueError, match=msg):
            MSEEDMetadata(files=[])

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
            assert mseed_metadata.meta['num_gaps'] == 2
            assert mseed_metadata.meta['num_overlaps'] == 1
            assert mseed_metadata.meta['sum_overlaps'] == 5.0
            assert mseed_metadata.meta['sum_gaps'] == 20.0
            assert mseed_metadata.meta['percent_availability'] == \
                45.0 / 65.0 * 100.0

            # Same again but this time with start-and end time settings.
            mseed_metadata = MSEEDMetadata(
                files=[tf.name], starttime=obspy.UTCDateTime(5),
                endtime=obspy.UTCDateTime(60))
            assert mseed_metadata.meta['num_gaps'] == 2
            assert mseed_metadata.meta['num_overlaps'] == 1
            assert mseed_metadata.meta['sum_overlaps'] == 5.0
            assert mseed_metadata.meta['sum_gaps'] == 20.0
            assert mseed_metadata.meta['percent_availability'] == \
                35.0 / 55.0 * 100.0

            # Head and tail gaps.
            mseed_metadata = MSEEDMetadata(
                files=[tf.name], starttime=obspy.UTCDateTime(-10),
                endtime=obspy.UTCDateTime(80))
            assert mseed_metadata.meta['num_gaps'] == 4
            assert mseed_metadata.meta['num_overlaps'] == 1
            assert mseed_metadata.meta['sum_overlaps'] == 5.0
            assert mseed_metadata.meta['sum_gaps'] == 45.0
            assert mseed_metadata.meta['percent_availability'] == \
                45.0 / 90.0 * 100.0

            # Tail gap must be larger than 1 delta, otherwise it does not
            # count.
            mseed_metadata = MSEEDMetadata(files=[tf.name],
                                           endtime=obspy.UTCDateTime(64))
            assert mseed_metadata.meta['num_gaps'] == 2
            assert mseed_metadata.meta['sum_gaps'] == 20.0
            assert mseed_metadata.meta['percent_availability'] == \
                44.0 / 64.0 * 100.0
            mseed_metadata = MSEEDMetadata(files=[tf.name],
                                           endtime=obspy.UTCDateTime(65))
            assert mseed_metadata.meta['num_gaps'] == 2
            assert mseed_metadata.meta['sum_gaps'] == 20.0
            assert mseed_metadata.meta['percent_availability'] == \
                45.0 / 65.0 * 100.0
            mseed_metadata = MSEEDMetadata(files=[tf.name],
                                           endtime=obspy.UTCDateTime(66))
            assert mseed_metadata.meta['num_gaps'] == 3
            assert mseed_metadata.meta['sum_gaps'] == 21.0
            assert mseed_metadata.meta['percent_availability'] == \
                45.0 / 66.0 * 100.0

    def test_raise_unmatching_ids(self):
        """
        Test error raised for multiple stream identifiers
        """
        with NamedTemporaryFile() as tf1, NamedTemporaryFile() as tf2:
            obspy.Trace(data=np.arange(10, dtype=np.int32),
                        header={"starttime": obspy.UTCDateTime(0),
                                "network": "NL", "station": "HGN",
                                "location": "02", "channel": "BHZ"}).write(
                tf1.name, format="mseed", encoding="STEIM1", reclen=256)
            obspy.Trace(data=np.arange(10, dtype=np.float32),
                        header={"starttime": obspy.UTCDateTime(100),
                                "sampling_rate": 2.0, "network": "BW",
                                "station": "ALTM", "location": "00",
                                "channel": "EHE"}).write(
                tf2.name, format="mseed", encoding="FLOAT32", reclen=1024)

            msg = "All traces must have the same SEED id and quality."
            with pytest.raises(ValueError, match=msg):
                MSEEDMetadata([tf1.name, tf2.name])

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
            mseed_metadata = MSEEDMetadata([tf1.name, tf2.name],
                                           add_c_segments=False)
            assert mseed_metadata.meta['num_gaps'] == 1
            assert "c_segments" not in mseed_metadata.meta

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
            assert ref['timing_quality_max'] is None
            assert ref['timing_quality_min'] is None
            assert ref['timing_quality_mean'] is None

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
            md = MSEEDMetadata([tf1.name, tf2.name], add_flags=True)
            assert md.meta["network"] == "BW"
            assert md.meta["station"] == "ALTM"
            assert md.meta["location"] == "00"
            assert md.meta["channel"] == "EHE"
            assert md.meta["quality"] == "D"
            assert md.meta["start_time"] == obspy.UTCDateTime(0)
            assert md.meta["end_time"] == obspy.UTCDateTime(105)
            assert md.meta["num_records"] == 2
            assert md.meta["num_samples"] == 20
            assert md.meta["sample_rate"] == [1.0, 2.0]
            assert md.meta["record_length"] == [256, 1024]
            assert md.meta["encoding"] == ["FLOAT32", "STEIM1"]

    def test_extraction_header_flags_complex(self):
        """
        Tests the flag extraction in a complex record situation
        Three records, with records 2 & 3 two overlapping 50% and
        a 50% record length gap between record 1 and 2.

        Rules for overlaps with different bits are as follows:
        Records are sorted from end to start by endtime and processed
        in this order. Each consecutive record occupies a time-range
        that can no longer be used by another record. In the following
        example, the third record is dominant over the second record
        because it is processed first and occupies the time range.
        Therefore the bit in this entire range is set to 1, despite
        partially overlapping with a record with its bit set to 0

        Bits in the test are set as shown

                        [ ==1== ]
        [ ==1== ]...[ ==0== ]
            |               |
          START            END
           25              125

        [RECORD 1 (1)] = 0 - 50 [clock_locked: 1]
        [RECORD 2 (0)] = 75 - 125 [clock_locked: 0]
        [RECORD 3 (1)] = 100 - 150 [clock_locked: 1]
        With starttime = 25 and endtime = 125

        The clock_locked percentage should thus be exactly 50.0%
        """

        # Couldn't properly break this line following PEP8 so use
        # shorter notation .....
        short = NamedTemporaryFile
        with short() as tf1, short() as tf2, short() as tf3:
            _create_mseed_file(tf1.name, record_count=1,
                               starttime=obspy.UTCDateTime(0),
                               seed=12345, flags={
                                   'io_and_clock_flags': {
                                       "clock_locked": 1}})
            _create_mseed_file(tf2.name, record_count=1,
                               starttime=obspy.UTCDateTime(75),
                               seed=12345, flags={
                                   'io_and_clock_flags': {
                                       "clock_locked": 0}})
            _create_mseed_file(tf3.name, record_count=1,
                               starttime=obspy.UTCDateTime(100),
                               seed=12345, flags={
                                   'io_and_clock_flags': {
                                       "clock_locked": 1}})

            md = MSEEDMetadata([tf1.name, tf2.name, tf3.name],
                               starttime=obspy.UTCDateTime(25),
                               endtime=obspy.UTCDateTime(125), add_flags=True)
            io_f = md.meta["miniseed_header_percentages"]["io_and_clock_flags"]
            assert io_f["clock_locked"] == 50.0

    def test_extraction_fixed_header_flags(self):
        # Had to put positive_leap count to 0 to prevent
        # end time from being wrong
        with NamedTemporaryFile() as tf1, NamedTemporaryFile() as tf2:
            _create_mseed_file(tf1.name, record_count=35,
                               starttime=obspy.UTCDateTime(0),
                               seed=12345, flags={
                                   'data_quality_flags': {
                                       "amplifier_saturation": 25,
                                       "digitizer_clipping": 12,
                                       "spikes": 30,
                                       "glitches": 6,
                                       "missing_padded_data": 15,
                                       "telemetry_sync_error": 16,
                                       "digital_filter_charging": 4,
                                       "suspect_time_tag": 8},
                                   'activity_flags': {
                                       "calibration_signal": 10,
                                       "time_correction_applied": 20,
                                       "event_begin": 33,
                                       "event_end": 33,
                                       "positive_leap": 0,
                                       "negative_leap": 10,
                                       "event_in_progress": 15},
                                   'io_and_clock_flags': {
                                       "station_volume": 8,
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
                                       "amplifier_saturation": 5,
                                       "digitizer_clipping": 7,
                                       "spikes": 5,
                                       "glitches": 3,
                                       "missing_padded_data": 5,
                                       "telemetry_sync_error": 3,
                                       "digital_filter_charging": 4,
                                       "suspect_time_tag": 2},
                                   'activity_flags': {
                                       "calibration_signal": 1,
                                       "time_correction_applied": 0,
                                       "event_begin": 3,
                                       "event_end": 3,
                                       "positive_leap": 0,
                                       "negative_leap": 1,
                                       "event_in_progress": 5},
                                   'io_and_clock_flags': {
                                       "station_volume": 1,
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
                assert abs(a - b) < 1E-6

            # Sum up contributions from both files.
            # Check percentages
            meta = md.meta['miniseed_header_counts']
            meta_dq = meta['data_quality_flags']
            assert meta_dq['glitches'] == 9
            assert meta_dq['amplifier_saturation'] == 30
            assert meta_dq['digital_filter_charging'] == 8
            assert meta_dq['digitizer_clipping'] == 19
            assert meta_dq['missing_padded_data'] == 20
            assert meta_dq['spikes'] == 35
            assert meta_dq['suspect_time_tag'] == 10
            assert meta_dq['telemetry_sync_error'] == 19

            meta_af = meta['activity_flags']
            assert meta_af['calibration_signal'] == 11
            assert meta_af['event_begin'] == 36
            assert meta_af['event_end'] == 36
            assert meta_af['event_in_progress'] == 20
            assert meta_af['time_correction_applied'] == 20

            meta_io = meta['io_and_clock_flags']
            assert meta_io['clock_locked'] == 34
            assert meta_io['station_volume'] == 9
            assert meta_io['long_record_read'] == 36
            assert meta_io['short_record_read'] == 26
            assert meta_io['start_time_series'] == 34
            assert meta_io['end_time_series'] == 28

            meta = md.meta['miniseed_header_percentages']
            meta_dq = meta['data_quality_flags']
            _assert_float_equal(meta_dq['glitches'], 9 / 0.58)
            _assert_float_equal(meta_dq['amplifier_saturation'], 30 / 0.58)
            _assert_float_equal(meta_dq['digital_filter_charging'], 8 / 0.58)
            _assert_float_equal(meta_dq['digitizer_clipping'], 19 / 0.58)
            _assert_float_equal(meta_dq['missing_padded_data'], 20 / 0.58)
            _assert_float_equal(meta_dq['spikes'], 35 / 0.58)
            _assert_float_equal(meta_dq['suspect_time_tag'], 10 / 0.58)
            _assert_float_equal(meta_dq['telemetry_sync_error'], 19 / 0.58)

            meta_af = meta['activity_flags']
            _assert_float_equal(meta_af['calibration_signal'], 11 / 0.58)
            _assert_float_equal(meta_af['event_begin'], 36 / 0.58)
            _assert_float_equal(meta_af['event_end'], 36 / 0.58)
            _assert_float_equal(meta_af['event_in_progress'], 20 / 0.58)
            _assert_float_equal(meta_af['time_correction_applied'], 20 / 0.58)

            meta_io = meta['io_and_clock_flags']
            _assert_float_equal(meta_io['clock_locked'], 34 / 0.58)
            _assert_float_equal(meta_io['station_volume'], 9 / 0.58)
            _assert_float_equal(meta_io['long_record_read'], 36 / 0.58)
            _assert_float_equal(meta_io['short_record_read'], 26 / 0.58)
            _assert_float_equal(meta_io['start_time_series'], 34 / 0.58)
            _assert_float_equal(meta_io['end_time_series'], 28 / 0.58)

            ref = md.meta['miniseed_header_percentages']
            assert ref['timing_quality_mean'] is None
            assert ref['timing_quality_min'] is None
            assert ref['timing_quality_max'] is None

    def test_timing_quality(self, testdata):
        """
        Test extraction of timing quality with a file that actually has it.
        """
        # Test file is constructed and orignally from the obspy.io.mseed
        # test suite.
        md = MSEEDMetadata(files=[testdata["timingquality.mseed"]],
                           add_flags=True)
        ref = md.meta['miniseed_header_percentages']
        assert ref['timing_quality_mean'] == 50.0
        assert ref['timing_quality_min'] == 0.0
        assert ref['timing_quality_max'] == 100.0
        assert ref['timing_quality_median'] == 50.0
        assert ref['timing_quality_lower_quartile'] == 25.0
        assert ref['timing_quality_upper_quartile'] == 75.0

    def test_overall_sample_metrics(self):
        """
        Tests the global metrics on the samples.
        """
        with NamedTemporaryFile() as tf:
            obspy.Trace(data=np.arange(10, dtype=np.int32),
                        header={"starttime": obspy.UTCDateTime(0)}).write(
                tf.name, format="mseed")

            md = MSEEDMetadata(files=[tf.name])

        assert md.meta["sample_min"] == 0
        assert md.meta["sample_max"] == 9
        assert md.meta["sample_mean"] == 4.5
        assert md.meta["sample_stdev"] - 2.8722813232 < 1E-6
        assert md.meta["sample_rms"] - 5.33853912602 < 1E-6
        assert md.meta["sample_median"], 4.5
        assert md.meta["sample_lower_quartile"], 2.25
        assert md.meta["sample_upper_quartile"], 6.75

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

        assert md.meta["sample_min"] == 0
        assert md.meta["sample_max"] == 9
        assert md.meta["sample_mean"] == 4.5
        assert md.meta["sample_stdev"] - 2.8722813232 < 1E-6
        assert md.meta["sample_rms"] - 7.14142842854 < 1E-6
        assert md.meta["sample_median"] == 4.5
        assert md.meta["sample_lower_quartile"], 2.25
        assert md.meta["sample_upper_quartile"], 6.75

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
        assert np.fabs(md.meta["sample_rms"] - 1) < 1E-3
        for c_segment in md.meta["c_segments"]:
            assert np.fabs(c_segment["sample_rms"] - 1) < 1E-3

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
            assert md.meta["sample_rms"] == np.iinfo(np.int32).max

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

            assert _rapid_overlap_testing(0, 12) == 7.5
            assert _rapid_overlap_testing(3, 7) == 4.0
            assert _rapid_overlap_testing(3, 5.5) == 2.5
            assert _rapid_overlap_testing(4.5, 5.5) == 1.0
            assert _rapid_overlap_testing(2, 5.25) == 2.25
            assert _rapid_overlap_testing(2, 3) == 0.0
            assert _rapid_overlap_testing(2, 3.1) == 0.1
            assert _rapid_overlap_testing(7, 9) == 1.5
            assert _rapid_overlap_testing(6.9, 9) == 1.6
            assert _rapid_overlap_testing(4.30, 9) == 4.2
            assert _rapid_overlap_testing(5.20, 9000) == 5.3

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

            assert _rapid_gap_testing(5, 17) == 5
            assert _rapid_gap_testing(5, 10) == 2
            assert _rapid_gap_testing(8.30, 9.5) == 0.70
            assert _rapid_gap_testing(9, 12) == 2
            assert _rapid_gap_testing(12, 17) == 1
            assert _rapid_gap_testing(10, 13) == 2
            assert _rapid_gap_testing(10.25, 13) == 1.75
            assert _rapid_gap_testing(11.75, 17) == 1.25
            assert _rapid_gap_testing(6, 10.5) == 2.5
            assert _rapid_gap_testing(11.99, 12.01) == 0.01
            assert _rapid_gap_testing(10.1, 12.01) == 1.9
            assert _rapid_gap_testing(7.5, 14.25) == 3.75
            assert _rapid_gap_testing(5, 17.5) == 5.5
            assert _rapid_gap_testing(5, 17.6) == 5.6
            assert _rapid_gap_testing(5, 18) == 6
            assert _rapid_gap_testing(0, 5.01) == 5
            assert _rapid_gap_testing(0, 20) == 13

    def test_start_gap(self, testdata):
        """
        Tests whether a gap in the beginning of the file is interpreted
        A gap at the beginning of the window is ignored if a sample can be
        found before the starttime, and is continuous
        Trace in file runs from [00:01.625000Z to 00:59.300000Z]
        """
        file = testdata['tiny_quality_file.mseed']

        # first sample is 625000, so we introduce a gap of 0.025
        starttime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 600000)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 300000)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime,
                           add_c_segments=True)

        assert md.meta["num_gaps"] == 1
        assert md.meta["sum_gaps"] == 0.025
        assert md.meta["start_gap"] == 0.025

        # Test single continuous segments
        # Gap in beginning, cseg starts at first sample
        cseg = md.meta['c_segments'][0]
        assert cseg['start_time'] == \
            obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 625000)
        assert cseg['end_time'] == \
            obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 300000)

        # Start beyond a sample, but it is padded to the left, no gap
        starttime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 630000)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 300000)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime,
                           add_c_segments=True)
        assert md.meta["num_gaps"] == 0
        assert md.meta["start_gap"] is None

        # Test single continuous segments
        # first sample is padded, first cseg starttime is window starttime
        cseg = md.meta['c_segments'][0]
        assert cseg['start_time'] == \
            obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 630000)
        assert cseg['end_time'] == \
            obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 300000)

        md = MSEEDMetadata([file], add_c_segments=True)
        assert md.meta["num_gaps"] == 0
        assert md.meta["start_gap"] is None

        # Test single continuous segments
        # No window, start time is first sample, end time is last sample + dt
        cseg = md.meta['c_segments'][0]
        assert cseg['start_time'] == \
            obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 625000)
        assert cseg['end_time'] == \
            obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 325000)

    def test_random_window(self, testdata):
        """
        Tests a random window within a continuous trace, expect no gaps
        Continuous trace in file runs from [00:01.625000Z to 00:59.300000Z]
        """
        file = testdata['tiny_quality_file.mseed']

        # Go randomly somewhere between two samples, find no gaps because
        # the trace is continuous everywhere between start-end
        starttime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 22, 646572)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 38, 265749)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        assert md.meta["num_gaps"] == 0
        assert md.meta["num_overlaps"] == 0

        # Slicing to a start & end without asking for flags, num_records
        # will be set to None
        assert md.meta["num_records"] is None

    def test_end_gap(self, testdata):
        """
        Test for the end gap. A gap should be found if the
        endtime exceeds the last sample + delta + time tolerance
        Trace in file runs from [00:01.625000Z to 00:59.300000Z]
        """
        file = testdata['tiny_quality_file.mseed']

        # Last sample is at 300000, but this sample covers the trace
        # up to 300000 + delta (0.025) => 325000 - no gaps
        starttime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 625000)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 325000)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        assert md.meta["num_gaps"] == 0
        assert md.meta["sum_gaps"] == 0.0
        assert md.meta["end_gap"] is None

        # Add 1μs; exceed projected sample plus time tolerance - GAP!
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 350001)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        assert md.meta["num_gaps"] == 1
        assert md.meta["sum_gaps"] == 0.025001
        assert md.meta["end_gap"] == 0.025001

    def test_clock_locked_percentage(self, testdata):
        """
        7/10 records with io_flag clock_locked set to 1 for which we
        count the percentage of record time of total time
        """
        file = testdata['tiny_quality_file.mseed']

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
        assert abs(sum(record_lengths) - (total_time)) < 1.001e-6

        # Calculate the percentage of clock_locked seconds
        percentage = 100 * sum(record_lengths_flagged) / sum(record_lengths)
        meta = md.meta["miniseed_header_percentages"]
        meta_io = meta['io_and_clock_flags']
        assert abs(meta_io["clock_locked"] - percentage) < 1e-6

    def test_endtime_on_sample(self, testdata):
        """
        [T0, T1), T0 should be included; T1 excluded
        Test to see whether points on starttime are included, and
        samples on the endtime are excluded.
        Total number of samples for this test = 2308
        Trace in file runs from [00:01.625000Z to 00:59.300000Z]
        """
        file = testdata['tiny_quality_file.mseed']

        # Set T0 and T1 on sample (N-1)
        starttime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 625000)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 300000)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        assert md.meta["num_samples"] == 2307
        assert md.meta["end_time"] == endtime

        # Set T0 on sample and T1 1μ after sample (N)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 300001)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        assert md.meta["num_samples"] == 2308
        assert md.meta["end_time"] == endtime

        # Set T0 and T1 1μ after sample (N-1)
        starttime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 1, 625001)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        assert md.meta["num_samples"] == 2307
        assert md.meta["start_time"] == starttime

        # Set T0 1μ after sample and T1 on sample (N-2)
        endtime = obspy.UTCDateTime(2015, 10, 16, 0, 0, 59, 300000)
        md = MSEEDMetadata([file], starttime=starttime, endtime=endtime)
        assert md.meta["num_samples"] == 2306
        assert md.meta["start_time"] == starttime

    def test_continuous_segments_combined(self):
        """
        Test continuous segments from traces in two files
        that are continuous. Also test a continuous segment
        that is continuous but has a different sampling rate
        """
        tr_1 = obspy.Trace(data=np.arange(10, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(0)})
        tr_2 = obspy.Trace(data=np.arange(10, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(10)})
        tr_3 = obspy.Trace(data=np.arange(10, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(20),
                                   "sampling_rate": 0.5})
        st = obspy.Stream(traces=[tr_1, tr_3])
        st2 = obspy.Stream(traces=[tr_2])
        with NamedTemporaryFile() as tf1, NamedTemporaryFile() as tf2:

            st.write(tf1.name, format="mseed")
            st2.write(tf2.name, format="mseed")
            md = MSEEDMetadata(files=[tf1.name, tf2.name])
            c_seg = md.meta["c_segments"]
            assert len(c_seg) == 2

            c = c_seg[0]
            assert c["start_time"] == obspy.UTCDateTime(0)
            assert c["end_time"] == obspy.UTCDateTime(20)
            assert c["segment_length"] == 20
            assert c["sample_min"] == 0
            assert c["sample_max"] == 9
            assert c["num_samples"] == 20
            assert c["sample_median"] == 4.5
            assert c["sample_lower_quartile"] == 2.0
            assert c["sample_upper_quartile"] == 7.0
            assert c["sample_rate"] == 1.0

            # Not continuous because of different sampling_rate (0.5)
            c = c_seg[1]
            assert c["start_time"] == obspy.UTCDateTime(20)
            assert c["end_time"] == obspy.UTCDateTime(40)
            assert c["segment_length"] == 20
            assert c["sample_min"] == 0
            assert c["sample_max"] == 9
            assert c["num_samples"] == 10
            assert c["sample_median"] == 4.5
            assert c["sample_lower_quartile"] == 2.25
            assert c["sample_upper_quartile"] == 6.75
            assert c["sample_rate"] == 0.5

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
        assert len(c_seg) == 3

        c = c_seg[0]
        assert c["start_time"] == obspy.UTCDateTime(0)
        assert c["end_time"] == obspy.UTCDateTime(5)
        assert c["sample_min"] == 0
        assert c["sample_max"] == 4
        assert c["sample_mean"] == 2.0
        assert c["sample_rms"] - 2.4494897427831779 < 1E-6
        assert c["sample_stdev"] - 1.4142135623730951 < 1E-6
        assert c["num_samples"] == 5
        assert c["segment_length"] == 5.0
        assert c["sample_median"] == 2
        assert c["sample_lower_quartile"], 1.0
        assert c["sample_upper_quartile"], 3.0

        c = c_seg[1]
        assert c["start_time"] == obspy.UTCDateTime(10)
        assert c["end_time"] == obspy.UTCDateTime(15)
        assert c["sample_min"] == 5
        assert c["sample_max"] == 9
        assert c["sample_mean"] == 7.0
        assert c["sample_rms"] - 7.1414284285428504 < 1E-6
        assert c["sample_stdev"] - 1.4142135623730951 < 1E-6
        assert c["num_samples"] == 5
        assert c["segment_length"] == 5.0
        assert c["sample_median"] == 7
        assert c["sample_lower_quartile"], 6.0
        assert c["sample_upper_quartile"], 8.0

        c = c_seg[2]
        assert c["start_time"] == obspy.UTCDateTime(20)
        assert c["end_time"] == obspy.UTCDateTime(30)
        assert c["num_samples"] == 10
        assert c["segment_length"] == 10.0
        assert c["sample_min"] == 0
        assert c["sample_max"] == 9
        assert c["sample_mean"] == 4.5
        assert c["sample_stdev"] - 2.8722813232 < 1E-6
        assert c["sample_rms"] - 5.3385391260156556 < 1E-6
        assert c["sample_median"] == 4.5
        assert c["sample_lower_quartile"], 2.25
        assert c["sample_upper_quartile"], 6.25

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

        assert md.get_json_meta()

    @pytest.mark.skipif(not HAS_JSONSCHEMA,
                        reason="Test requires the jsonschema module")
    def test_schema_validation(self):
        with NamedTemporaryFile() as tf:
            obspy.Trace(data=np.arange(10, dtype=np.int32),
                        header={"starttime": obspy.UTCDateTime(0)}).write(
                tf.name, format="mseed")

            md = MSEEDMetadata(files=[tf.name])

            # One can either directly validate the metrics.
            md.validate_qc_metrics(md.meta)
            # Or do it during the serialization.
            md.get_json_meta(validate=True)

            # Also try with extracting the flags.
            md = MSEEDMetadata(files=[tf.name], add_flags=True)
            md.validate_qc_metrics(md.meta)
            md.get_json_meta(validate=True)
