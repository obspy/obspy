# -*- coding: utf-8 -*-
"""
Quality control module for ObsPy.

Currently requires MiniSEED files as that is the dominant data format in
data centers.

:authors:
    Luca Trani (trani@knmi.nl)
    Lion Krischer (krischer@geophysik.uni-muenchen.de)
    Mathijs Koymans (koymans@knmi.nl)
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import collections
import json

import numpy as np
import obspy
from obspy.io.mseed.util import get_flags


class DataQualityEncoder(json.JSONEncoder):
    """
    Custom encoder capable of dealing with NumPy and ObsPy types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, obspy.UTCDateTime):
            return str(obj)
        else:
            return super(DataQualityEncoder, self).default(obj)


class MSEEDMetadata(object):
    """
    A container for MSEED specific metadata including QC.
    """
    def __init__(self, files, starttime=None, endtime=None, c_seg=True,
                 add_flags=False):
        """
        Reads the MiniSEED files and extracts the data quality metrics.

        :param files: The MiniSEED files.
        :type files: list
        :param starttime: Only use records whose end time is larger then this
            given time. Also specifies the new official start time of the
            metadata object.
        :type starttime: :class:`obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Only use records whose start time is smaller then this
            given time. Also specifies the new official end time of the
            metadata object
        :type endtime: :class:`obspy.core.utcdatetime.UTCDateTime`
        :param c_seg: Calculate metrics for each continuous segment.
        :type c_seg: bool
        :param add_flags: Include miniSEED header statistics in result.
        :type add_flags: bool
        """

        self.data = obspy.Stream()
        self.files = []

        # Allow anything UTCDateTime can parse.
        if starttime is not None:
            starttime = obspy.UTCDateTime(starttime)
        if endtime is not None:
            endtime = obspy.UTCDateTime(endtime)

        self.window_start = starttime
        self.window_end = endtime

        # We are required to exclude samples at T1. Therefore, shift the
        # time window to the left by 1μs and set nearest_sample to False.
        # This will force ObsPy to fall back to the sample left of the endtime
        if endtime is not None:
            endtime_left = endtime - 1e-6
        else:
            endtime_left = None

        # Will raise if not a MiniSEED files.
        for file in files:

            st = obspy.read(file, starttime=starttime, endtime=endtime_left,
                            format="mseed", nearest_sample=False)

            # Empty stream or maybe there is no data in the stream for the
            # requested time span.
            if not st:
                continue

            self.files.append(file)

            # Only extend traces with data (npts > 0)
            for tr in st:
                if(tr.stats.npts != 0):
                    self.data.extend([tr])

        if not self.data:
            raise ValueError("No data within the temporal constraints.")

        # Do some sanity checks. The class only works with data from a
        # single location so we have to make sure that the existing data on
        # this object and the newly added all have the same identifier.
        ids = set(tr.id + "." + tr.stats.mseed.dataquality for tr in self.data)
        if len(ids) != 1:
            raise ValueError("All traces must have the same SEED id and "
                             "quality")

        self.data.sort()

        # Set the metric start and endtime specified by the user.
        # If no start and endtime are given, we pick our own, and the window
        # will start on the first sample and end on the last sample + Δt.
        # This is conform to the definition of [T0, T1).
        end_stats = self.data[-1].stats
        self.starttime = starttime or self.data[0].stats.starttime
        self.endtime = endtime or end_stats.endtime + end_stats.delta
        self.total_time = self.endtime - self.starttime

        self.meta = {}

        # Get sample left of the user specified starttime
        # This will allow us to determine start continuity in our window
        self._get_gaps_and_overlaps()

        # The calculation of all the metrics begins here
        self._extract_mseed_stream_metadata()
        self._compute_sample_metrics()

        if add_flags:
            self._extract_mseed_flags()

        if c_seg:
            self._compute_continuous_seg_sample_metrics()

    def _get_gaps_and_overlaps(self):
        """
        Function to get all gaps and overlaps in the user
        specified (or forced) window.
        """
        self.all_data = obspy.Stream()

        body_gap = []
        body_overlap = []

        # Read all the files entirely and calculate gaps and overlaps
        # for the entire segment. Later we will narrow it to our window.
        for file in self.files:
            self.all_data.extend(obspy.read(file, format="mseed"))

        # Implicitly put a gap at the start of the window if it is not padded
        # with data.
        if self.window_start is not None:
            start_stats = self.all_data[0].stats
            if self.window_start < start_stats.starttime:
                body_gap.append(start_stats.starttime - self.window_start)

        if self.window_end is not None:
            end_stats = self.all_data[-1].stats
            if self.window_end > end_stats.endtime + 1.5*end_stats.delta:
                body_gap.append(self.window_end - (end_stats.endtime +
                                                   end_stats.delta))

        # Let ObsPy determine all the other gaps
        gaps = self.all_data.get_gaps()

        # Handle the logical to determine the actual gaps and overlaps in
        # the user specified window.. this gets pretty tricky because gaps
        # and overlaps may in theory cross the start/end boundaries.
        for _i in (_ for _ in gaps if _[-1] > 0):

            # Only check if the gap is within our window
            if (_i[5] - _i[-2]) >= self.endtime or _i[5] <= self.starttime:
                continue

            # Full gap
            if self.starttime <= (_i[5] - _i[-2]) and self.endtime >= _i[5]:
                body_gap.append(_i[-2])
            elif self.starttime >= (_i[5] - _i[-2]):
                body_gap.append(min(_i[5], self.endtime) - self.starttime)
            elif self.endtime <= _i[5]:
                body_gap.append(self.endtime - max(_i[5] - _i[-2],
                                                   self.starttime))

        # Handle overlaps (negative gaps)
        for _i in (_ for _ in gaps if _[-1] < 0):

            # Only check if the overlap is within our window
            if _i[5] >= self.endtime or (_i[5] - _i[-2]) <= self.starttime:
                continue

            # Full overlap
            if self.starttime <= _i[5] and self.endtime >= (_i[5] - _i[-2]):
                body_overlap.append(abs(_i[-2]))
            elif self.starttime >= _i[5]:
                body_overlap.append(abs(self.starttime - min(_i[5] - _i[-2],
                                                             self.endtime)))
            elif self.endtime <= (_i[5] - _i[-2]):
                body_overlap.append(abs(max(_i[5], self.starttime) -
                                        self.endtime))

        # Set the meta
        self.meta['num_gaps'] = len(body_gap)
        self.meta['sum_gaps'] = sum(body_gap)
        if len(body_gap) != 0:
            self.meta['max_gap'] = max(body_gap)
        else:
            self.meta['max_gap'] = None

        self.meta['num_overlaps'] = len(body_overlap)
        self.meta['sum_overlaps'] = sum(body_overlap)
        if len(body_overlap) != 0:
            self.meta['max_overlap'] = max(body_overlap)
        else:
            self.meta['max_overlap'] = None

    @property
    def number_of_records(self):
        """
        Number of records across files.
        """
        return sum(tr.stats.mseed.number_of_records for tr in self.data)

    @property
    def number_of_samples(self):
        """
        Number of samples across files.
        """
        return sum(tr.stats.npts for tr in self.data)

    def _extract_mseed_stream_stats(self):
        """
        Small function to collects the mSEED stats
        """
        stats = self.data[0].stats
        self.meta['network'] = stats.network
        self.meta['station'] = stats.station
        self.meta['location'] = stats.location
        self.meta['channel'] = stats.channel
        self.meta['quality'] = stats.mseed.dataquality

    def _extract_mseed_stream_metadata(self):
        """
        Collect information from the MiniSEED headers.
        """

        self._extract_mseed_stream_stats()

        meta = self.meta

        # Save first and last sample of the trace
        meta['first_sample'] = self.data[0].stats.starttime
        meta['last_sample'] = self.data[-1].stats.endtime

        # Add some other parameters to the metadata object
        meta['seed_id'] = self.data[0].id
        meta['files'] = self.files
        meta['start_time'] = self.starttime
        meta['end_time'] = self.endtime
        meta['num_records'] = self.number_of_records
        meta['num_samples'] = self.number_of_samples

        # The following are lists and may contain multiple unique entries.
        meta['sample_rate'] = \
            sorted(list(set([tr.stats.sampling_rate for tr in self.data])))
        meta['record_length'] = \
            sorted(list(set([tr.stats.mseed.record_length
                             for tr in self.data])))
        meta['encoding'] = \
            sorted(list(set([tr.stats.mseed.encoding for tr in self.data])))

    def _extract_mseed_flags(self):

        # Setup counters for the MiniSEED header flags.
        data_quality_flags = collections.Counter(
                amplifier_saturation_detected=0,
                digitizer_clipping_detected=0,
                spikes_detected=0,
                glitches_detected=0,
                missing_data_present=0,
                telemetry_sync_error=0,
                digital_filter_charging=0,
                time_tag_uncertain=0)
        activity_flags = collections.Counter(
                calibration_signals_present=0,
                time_correction_applied=0,
                beginning_event=0,
                end_event=0,
                positive_leap=0,
                negative_leap=0)
        io_and_clock_flags = collections.Counter(
                station_volume_parity_error=0,
                long_record_read=0,
                short_record_read=0,
                start_time_series=0,
                end_time_series=0,
                clock_locked=0)
        timing_quality = []

        # Setup counters for the MiniSEED header flags percentages.
        # Counters are supposed to work for integers, but
        # it also appears to work for floats too
        data_quality_flags_seconds = collections.Counter(
                amplifier_saturation_detected=0.0,
                digitizer_clipping_detected=0.0,
                spikes_detected=0.0,
                glitches_detected=0.0,
                missing_data_present=0.0,
                telemetry_sync_error=0.0,
                digital_filter_charging=0.0,
                time_tag_uncertain=0.0)
        activity_flags_seconds = collections.Counter(
                calibration_signals_present=0.0,
                time_correction_applied=0.0,
                beginning_event=0.0,
                end_event=0.0,
                positive_leap=0.0,
                negative_leap=0.0)
        io_and_clock_flags_seconds = collections.Counter(
                station_volume_parity_error=0.0,
                long_record_read=0.0,
                short_record_read=0.0,
                start_time_series=0.0,
                end_time_series=0.0,
                clock_locked=0.0)

        timing_correction = 0.0
        used_segments = []

        for file in self.files:
            flags = get_flags(file, starttime=self.starttime,
                              endtime=self.endtime, used_segments=used_segments)

            used_segments = flags["used_segments"]

            # Update the flag counters
            data_quality_flags.update(flags["data_quality_flags"])
            activity_flags.update(flags["activity_flags"])
            io_and_clock_flags.update(flags["io_and_clock_flags"])

            # Update the percentage counters
            data_quality_flags_seconds.update(
                flags["data_quality_flags_seconds"])
            activity_flags_seconds.update(
                flags["activity_flags_seconds"])
            io_and_clock_flags_seconds.update(
                flags["io_and_clock_flags_seconds"])

            if flags["timing_quality"]:
                timing_quality.append(flags["timing_quality"]["all_values"])

            timing_correction += flags["timing_correction"]

        # Convert second counts to percentages. The total time is the
        # difference between start & end in seconds. The percentage fields
        # are the sum of record lengths for which the respective bits are
        # set in SECONDS
        for key in data_quality_flags_seconds:
            data_quality_flags_seconds[key] /= self.total_time * 1e-2
        for key in activity_flags_seconds:
            activity_flags_seconds[key] /= self.total_time * 1e-2
        for key in io_and_clock_flags_seconds:
            io_and_clock_flags_seconds[key] /= self.total_time * 1e-2

        # Only calculate the timing quality statistics if each files has the
        # timing quality set. This should usually be the case. Otherwise we
        # would created tinted statistics. There is still a chance that some
        # records in a file have timing qualities set and others not but
        # that should be small.
        if len(timing_quality) == len(self.files):
            timing_quality = np.concatenate(timing_quality)
            timing_quality_mean = timing_quality.mean()
            timing_quality_min = timing_quality.min()
            timing_quality_max = timing_quality.max()
            timing_quality_median = np.median(timing_quality)
            timing_quality_lower_quartile = np.percentile(timing_quality, 25)
            timing_quality_upper_quartile = np.percentile(timing_quality, 75)
        else:
            timing_quality_mean = None
            timing_quality_min = None
            timing_quality_max = None
            timing_quality_median = None
            timing_quality_lower_quartile = None
            timing_quality_upper_quartile = None

        meta = self.meta

        # Set miniseed header counts
        meta['miniseed_header_percentages'] = {}
        ref = meta['miniseed_header_percentages']
        ref['timing_correction'] = timing_correction
        ref['timing_quality_mean'] = timing_quality_mean
        ref['timing_quality_min'] = timing_quality_min
        ref['timing_quality_max'] = timing_quality_max
        ref['timing_quality_median'] = timing_quality_median
        ref['timing_quality_lower_quartile'] = timing_quality_lower_quartile
        ref['timing_quality_upper_quartile'] = timing_quality_upper_quartile

        # According to schema @ maybe refactor this to less verbose flag
        # names. Sets miniseed header flag percentages
        ref['activity_flags'] = activity_flags_seconds
        ref['data_quality_flags'] = data_quality_flags_seconds
        ref['io_and_clock_flags'] = io_and_clock_flags_seconds

        # Similarly, set miniseed header flag counts
        meta['miniseed_header_counts'] = {}
        ref = meta['miniseed_header_counts']
        ref['activity_flags'] = activity_flags
        ref['data_quality_flags'] = data_quality_flags
        ref['io_and_clock_flags'] = io_and_clock_flags

        # Small function to change flag names from the get_flags routine
        # to match the schema
        self._fix_flag_names()

    def _fix_flag_names(self):
        """
        Supplementary function to fix flag parameter names
        Parameters with a key in the name_ref will be changed to its value
        """
        name_reference = {
            'amplifier_saturation_detected': 'amplifier_saturation',
            'digitizer_clipping_detected': 'digitizer_clipping',
            'spikes_detected': 'spikes',
            'glitches_detected': 'glitches',
            'missing_data_present': 'missing_padded_data',
            'time_tag_uncertain': 'suspect_time_tag',
            'calibration_signals_present': 'calibration_signal',
            'beginning_event': 'event_begin',
            'end_event': 'event_end',
            'station_volume_parity_error': 'station_volume',
        }

        # Loop over all keys and replace where required according to
        # the name_reference
        prefix = 'miniseed_header'
        for flag_type in ['_percentages', '_counts']:
            for _, flags in self.meta[prefix + flag_type].iteritems():
                if _ not in ["activity_flags", "data_quality_flags",
                             "io_and_clock_flags"]:
                    continue
                for param in flags:
                    if param in name_reference:
                        flags[name_reference[param]] = flags.pop(param)

    def _compute_sample_metrics(self):
        """
        Computes metrics on samples contained in the specified time window
        """

        # Make sure there is no integer division by chance.
        npts = float(self.number_of_samples)

        self.meta['sample_min'] = min([tr.data.min() for tr in self.data])
        self.meta['sample_max'] = max([tr.data.max() for tr in self.data])

        # Manually implement these as they have to work across a list of
        # arrays.
        self.meta['sample_mean'] = \
            sum(tr.data.sum() for tr in self.data) / npts
        self.meta['sample_median'] = \
            np.median([n for n in tr.data for tr in self.data])

        # Might overflow np.int64 so make Python obj. (.astype(object))
        # allows conversion to long int when required (see tests)
        self.meta['sample_rms'] = \
            np.sqrt(sum((tr.data.astype(object) ** 2).sum()
                        for tr in self.data) / npts)

        self.meta['sample_stdev'] = np.sqrt(sum(
            ((tr.data - self.meta["sample_mean"]) ** 2).sum()
            for tr in self.data) / npts)

        # Percentage based availability as a function of total gap length
        # over the full trace duration
        self.meta['percent_availability'] = 100 * (
            (self.total_time - self.meta['sum_gaps']) /
            self.total_time)

    def _compute_continuous_seg_sample_metrics(self):
        """
        Computes metrics on the samples within each continuous segment.
        """
        if not self.data:
            return

        c_segments = []

        # Add metrics for all continuous segments from the start of the trace
        # until the end of the trace + delta
        for tr in self.data:
            seg = {}
            seg['start_time'] = tr.stats.starttime
            seg['end_time'] = tr.stats.endtime + tr.stats.delta
            seg['sample_min'] = tr.data.min()
            seg['sample_max'] = tr.data.max()
            seg['sample_mean'] = tr.data.mean()
            seg['sample_median'] = np.median(tr.data)
            seg['sample_rms'] = np.sqrt((tr.data.astype(object) ** 2).sum() /
                                        tr.stats.npts)
            seg['sample_stdev'] = tr.data.std()
            seg['num_samples'] = tr.stats.npts
            seg['seg_len'] = tr.stats.endtime - tr.stats.starttime
            c_segments.append(seg)

        self.meta['c_segments'] = c_segments

    def get_json_meta(self):
        """
        Serialize the meta dictionary to JSON.

        :return: JSON containing the MSEED metadata
        """
        return json.dumps(self.meta, cls=DataQualityEncoder)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
