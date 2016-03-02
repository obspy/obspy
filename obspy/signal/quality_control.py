# -*- coding: utf-8 -*-
"""
Quality control module for ObsPy.

Currently requires MiniSEED files as that is the dominant data format in
data centers.

:author:
    Luca Trani (trani@knmi.nl)
    Lion Krischer (krischer@geophysik.uni-muenchen.de)
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
    def __init__(self, files, starttime=None, endtime=None, c_seg=True, continuous=False):
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
        :param continuous: Specifies whether start is continuous
            set to True and the class will ignore apparent gaps
            between starttime - startdata
        :type continuous: bool
        """

        self.data = obspy.Stream()
        self.continuous = continuous

        # Allow anything UTCDateTime can parse.
        if(starttime is not None):
            starttime = obspy.UTCDateTime(starttime)
        if(endtime is not None):
            endtime = obspy.UTCDateTime(endtime)

        # Metrics are defined for [T1, T2) thus we
        # subtract 1μs from the endtime to exclude samples at T2
        # by using nearest_sample = False during Stream.read()
        # ObsPy is not allowed to look ahead for samples
        if(endtime is not None):
            read_endtime = obspy.UTCDateTime(endtime) - 1e-6
        else:
            read_endtime = None

        self.files = []

        # Will raise if not a MiniSEED files.
        for file in files:
            st = obspy.read(file, starttime=starttime, endtime=read_endtime,
                            format="mseed", nearest_sample=False)

            # Empty stream or maybe there is no data in the stream for the
            # requested time span.
            if not st:
                continue

            self.files.append(file)

            # Only extend traces with data
            for tr in st:
                if(tr.stats.npts != 0):
                    self.data.extend([tr])

        if not self.data:
            raise ValueError("No data within the temporal constraints.")

        # Do some sanity checks. The class only works with data from a
        # single location so we have to make sure that the existing data on
        # this object and the newly added all have the same identifier.
        ids = set(tr.id + tr.stats.mseed.dataquality for tr in self.data)

        if len(ids) != 1:
            raise ValueError("All traces must have the same SEED id.")

        # Sort so that gaps and what not work in an ok fashion.
        self.data.sort()

        # Set the start/end to user specified or take from ObsPy
        # When ObsPy reads without a given start/end we add delta
        # to the final sample, to get an endtime that is consitent
        # with the QC definitions
        self.starttime = starttime or self.data[0].stats.starttime
        self.endtime = endtime or self.data[-1].stats.endtime + self.data[-1].stats.delta

        # Calculation of all the metrics begins here
        self.meta = {}
        self._extract_mseed_stream_metadata()
        self._compute_sample_metrics()

        if c_seg:
            self._compute_continuous_seg_sample_metrics()

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
        Collects the stats
        """
        stats = self.data[0].stats
        self.meta['network'] = stats.network
        self.meta['station'] = stats.station
        self.meta['location'] = stats.location
        self.meta['channel'] = stats.channel
        self.meta['quality'] = stats.mseed.dataquality
        self.meta['mseed_id'] = self.data[0].id

    def _get_unique_list(self, parameter):
        if(parameter == 'sampling_rate'):
            return sorted(list(set([tr.stats[parameter] for tr in self.data])))
        else:
            return sorted(list(set([tr.stats.mseed[parameter] for tr in self.data])))

    def _extract_mseed_stream_metadata(self):
        """
        Collect information from the MiniSEED headers.
        """

        self._extract_mseed_stream_stats()
        meta = self.meta
        m = self.meta

        # Add other parameters to the metadata object
        meta['files'] = self.files
        meta['start_time'] = self.starttime
        meta['end_time'] = self.endtime
        meta['num_records'] = self.number_of_records
        meta['num_samples'] = self.number_of_samples

        # The following are lists as it might contain multiple entries.
        meta['sample_rate'] = self._get_unique_list('sampling_rate')
        meta['record_len'] = self._get_unique_list('record_length')
        meta['encoding'] = self._get_unique_list('encoding')

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
                negative_leap=0,
                clock_locked=0)
        io_and_clock_flags = collections.Counter(
                station_volume_parity_error=0,
                long_record_read=0,
                short_record_read=0,
                start_time_series=0,
                end_time_series=0,
                clock_locked=0)
        timing_quality = []

        # Setup counters for the MiniSEED header flags percentages.
        data_quality_flags_percentages = collections.Counter(
                amplifier_saturation_detected=0,
                digitizer_clipping_detected=0,
                spikes_detected=0,
                glitches_detected=0,
                missing_data_present=0,
                telemetry_sync_error=0,
                digital_filter_charging=0,
                time_tag_uncertain=0)
        activity_flags_percentages = collections.Counter(
                calibration_signals_present=0,
                time_correction_applied=0,
                beginning_event=0,
                end_event=0,
                positive_leap=0,
                negative_leap=0,
                clock_locked=0)
        io_and_clock_flags_percentages = collections.Counter(
                station_volume_parity_error=0,
                long_record_read=0,
                short_record_read=0,
                start_time_series=0,
                end_time_series=0,
                clock_locked=0)

        for file in self.files:
            flags = get_flags(
                file, starttime=self.starttime, endtime=self.endtime)

            # Update the flag counters
            data_quality_flags.update(flags["data_quality_flags"])
            data_quality_flags_percentages.update(flags["data_quality_flags_percentages"])
            activity_flags.update(flags["activity_flags"])
            activity_flags_percentages.update(flags["activity_flags_percentages"])
            io_and_clock_flags.update(flags["io_and_clock_flags"])
            io_and_clock_flags_percentages.update(flags["io_and_clock_flags_percentages"])

            if flags["timing_quality"]:
                timing_quality.append(flags["timing_quality"]["all_values"])

        #[T1 - T2) - do not include last sample so substract sampling freq from endtime
        self.total_time = self.endtime - self.starttime

        # Set to percentages
        for key in data_quality_flags_percentages:
            data_quality_flags_percentages[key] /= self.total_time * 1e-2
        for key in activity_flags_percentages:
            activity_flags_percentages[key] /= self.total_time * 1e-2
        for key in io_and_clock_flags_percentages:
            io_and_clock_flags_percentages[key] /= self.total_time * 1e-2

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


        # Set miniseed header flag percentages
        m['miniseed_header_flag_percentages'] = {}
        m['miniseed_header_flag_percentages']['activity_flags'] = activity_flags_percentages
        m['miniseed_header_flag_percentages']['data_quality_flags'] = data_quality_flags_percentages
        m['miniseed_header_flag_percentages']['io_and_clock_flags'] = io_and_clock_flags_percentages

        # Set miniseed header flag counts
        m['miniseed_header_flag_counts'] = {}
        m['miniseed_header_flag_counts']['activity_flags'] = activity_flags
        m['miniseed_header_flag_counts']['data_quality_flags'] = data_quality_flags
        m['miniseed_header_flag_counts']['io_and_clock_flags'] = io_and_clock_flags
        
        self._fix_flag_names()

        # Set miniseed header counts
        m['timing_quality'] = {}
        m['timing_quality_mean'] = timing_quality_mean
        m['timing_quality_min'] = timing_quality_min
        m['timing_quality_max'] = timing_quality_max
        m['timing_quality_median'] = timing_quality_median
        m['timing_quality_lower_quartile'] = timing_quality_lower_quartile
        m['timing_quality_upper_quartile'] = timing_quality_upper_quartile

    def _fix_flag_names(self):
        """
        Supplementary function to fix flag parameter names
        Parameters with a key in the name_ref will be changed to its value
        """
        name_ref = {
            "amplifier_saturation_detected": "amplifier_saturation",
            "digitizer_clipping_detected": "digitizer_clipping",
            "spikes_detected": "spikes",
            "glitches_detected": "glitches",
            "missing_data_present": "missing_padded_data",
            "time_tag_uncertain": "suspect_time_tag",
            "calibration_signals_present": "calibration_signal",
            "time_correction_applied": "timing_correction",
            "beginning_event": "event_begin",
            "end_event": "event_end",
            "station_volume_parity_error": "station_volume",
        }

        # Loop over all keys and replace where required according to name_ref
        for flag_type in ["_percentages", "_counts"]:
            for _, flags in self.meta['miniseed_header_flag' + flag_type].iteritems():
                for param in flags:
                    if(param in name_ref):
                        flags[name_ref[param]] = flags.pop(param)

    def _compute_sample_metrics(self):
        """
        Computes metrics on samples contained in the specified time window
        """
        if not self.data:
            return

        # Make sure there is no integer division by chance.
        npts = float(self.number_of_samples)

        self.meta['sample_min'] = min([tr.data.min() for tr in self.data])
        self.meta['sample_max'] = max([tr.data.max() for tr in self.data])

        # Manually implement these as they have to work across a list of
        # arrays.
        self.meta['sample_mean'] = \
            sum(tr.data.sum() for tr in self.data) / npts

        # Might overflow np.int64 so make Python obj. This allows
        # conversion to long int when required
        self.meta['sample_rms'] = \
            np.sqrt(sum((tr.data.astype(object) ** 2).sum()
                        for tr in self.data) / npts)

        self.meta['sample_stdev'] = np.sqrt(sum(
            ((tr.data - self.meta["sample_mean"]) ** 2).sum()
            for tr in self.data) / npts)

        # Get gaps at beginning and end and the stream
        gap_count = 0
        gap_length = 0.0

        # A start gap will be ignored if the user specifies
        # that the trace is continuous (defaults to False)
        #  - x -- x -- x -- x -- x --
        # | <= self.starttime
        if not self.continuous:
            time_difference = self.data[0].stats.starttime - self.starttime
            if(time_difference > 0):
                gap_count += 1
                gap_length += time_difference

        # End of stream
        # We define the endtime as the time of the last sample but the next
        # sample would only start at endtime + delta. Thus the following
        # scenario would not count as a gap at the end:
        # also account for the time tolerance
        # x -- x -- x -- x -- x -- x -- 
        #                               | <= self.endtime
        stats = self.data[-1].stats
        time_tolerance = 0.5*stats.delta
        projected_sample = stats.endtime + stats.delta
        if(self.endtime - projected_sample > time_tolerance):
            gap_count += 1
            gap_length += self.endtime - projected_sample

        # Get the other gaps
        gaps = self.data.get_gaps()
        self.meta["num_gaps"] = \
            len([_i for _i in gaps if _i[-1] > 0]) + gap_count
        self.meta["num_overlaps"] = \
            len([_i for _i in gaps if _i[-1] < 0])
        self.meta["gaps_len"] = \
            sum(abs(_i[-2]) for _i in gaps if _i[-1] > 0) + gap_length
        self.meta["overlaps_len"] = \
            sum(abs(_i[-2]) for _i in gaps if _i[-1] < 0)

        # Percentage based availability as total gap length over trace
        # duration. The trace duration is end - start + dt as the endtime is
        # the time of the last sample but the last simple still "accounts"
        # for one more sample. This could well be defined differently.
        self.meta['percent_availability'] = 100.0 * (
            (self.endtime - self.starttime - self.meta['gaps_len']) /
            (self.endtime - self.starttime))

    def _compute_continuous_seg_sample_metrics(self):
        """
        Computes metrics on the samples within each continuous segment.
        """
        if not self.data:
            return

        c_segments = []

        for tr in self.data:
            seg = {}
            seg['start_time'] = tr.stats.starttime
            seg['end_time'] = tr.stats.endtime
            seg['sample_min'] = tr.data.min()
            seg['sample_max'] = tr.data.max()
            seg['sample_mean'] = tr.data.mean()
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
