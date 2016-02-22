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
    def __init__(self, files, starttime=None, endtime=None, c_seg=True):
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
        """
        self.meta = {}
        self.data = obspy.Stream()
        self.files = []

        # Allow anything UTCDateTime can parse.
        starttime = obspy.UTCDateTime(starttime) \
            if starttime is not None else None
        endtime = obspy.UTCDateTime(endtime) \
            if endtime is not None else None

        for file in files:
            # Will raise if not a MiniSEED files.
            st = obspy.read(file, starttime=starttime, endtime=endtime,
                            format="mseed")
            # Empty stream or maybe there is no data in the stream for the
            # requested time span.
            if not st:
                continue
            
            for tr in st:
                if(tr.stats.npts != 0):
                    self.data.extend([tr])
                    
            self.files.append(file)

        if not self.data:
            raise ValueError("Nothing added - no data within the given "
                             "temporal constraints found.")

        # Do some sanity checks. The class only works with data from a
        # single location so we have to make sure that the existing data on
        # this object and the newly added all have the same identifier.
        ids = set(tr.id + "." + tr.stats.mseed.dataquality
                  for tr in self.data)

        if len(ids) != 1:
            raise ValueError("All traces in all files must have the same SEED "
                             "id.")

        # Sort so that gaps and what not work in an ok fashion.
        self.data.sort()

        if starttime is None:
            self.starttime = self.data[0].stats.starttime
        else:
            self.starttime = starttime
        if endtime is None:
            self.endtime = self.data[-1].stats.endtime
        else:
            self.endtime = endtime

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

    def _extract_mseed_stream_metadata(self):
        """
        Collect information from the MiniSEED headers.
        """
        if not self.data:
            raise ValueError("Object contains no waveform data.")

        self.data.sort()
        m = self.meta

        stats = self.data[0].stats
        m['net'] = stats.network
        m['sta'] = stats.station
        m['loc'] = stats.location
        m['cha'] = stats.channel
        m['mseed_id'] = '.'.join([m['net'], m['sta'], m['loc'], m['cha']])
        m['quality'] = stats.mseed.dataquality
        m['files'] = self.files

        # start time of the requested available stream
        m['start_time'] = self.data[0].stats.starttime
        # end time of the requested available stream
        m['end_time'] = self.data[-1].stats.endtime

        m['num_records'] = self.number_of_records
        m['num_samples'] = self.number_of_samples

        # The following are lists as it might contain multiple entries.
        m['sample_rate'] = \
            sorted(list(set([tr.stats.sampling_rate for tr in self.data])))
        m['record_len'] = \
            sorted(list(set([tr.stats.mseed.record_length
                             for tr in self.data])))
        m['encoding'] = \
            sorted(list(set([tr.stats.mseed.encoding for tr in self.data])))

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

        for file in self.files:
            flags = get_flags(
                file, starttime=self.starttime, endtime=self.endtime)

            data_quality_flags.update(flags["data_quality_flags"])
            activity_flags.update(flags["activity_flags"])
            io_and_clock_flags.update(flags["io_and_clock_flags"])
            if flags["timing_quality"]:
                timing_quality.append(flags["timing_quality"]["all_values"])

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

        m['glitches'] = data_quality_flags["glitches_detected"]
        m['amplifier_saturation'] = \
            data_quality_flags["amplifier_saturation_detected"]
        m['digital_filter_charging'] = \
            data_quality_flags["digital_filter_charging"]
        m['digitizer_clipping'] = \
            data_quality_flags["digitizer_clipping_detected"]
        m['missing_padded_data'] = data_quality_flags["missing_data_present"]
        m['spikes'] = data_quality_flags["spikes_detected"]
        m['suspect_time_tag'] = data_quality_flags["time_tag_uncertain"]
        m['telemetry_sync_error'] = data_quality_flags["telemetry_sync_error"]
        m['calibration_signal'] = activity_flags["calibration_signals_present"]
        m['event_begin'] = activity_flags["beginning_event"]
        m['event_end'] = activity_flags["end_event"]
        m['event_in_progress'] = activity_flags["event_in_progress"]
        m['timing_correction'] = activity_flags["time_correction_applied"]
        m['clock_locked'] = io_and_clock_flags["clock_locked"]
        m['timing_quality_mean'] = timing_quality_mean
        m['timing_quality_min'] = timing_quality_min
        m['timing_quality_max'] = timing_quality_max
        m['timing_quality_median'] = timing_quality_median
        m['timing_quality_lower_quartile'] = timing_quality_lower_quartile
        m['timing_quality_upper_quartile'] = timing_quality_upper_quartile

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

        self.meta['sample_rms'] = \
            np.sqrt(sum((tr.data ** 2).sum() for tr in self.data)) / npts

        self.meta['sample_stdev'] = np.sqrt(sum(
            ((tr.data - self.meta["sample_mean"]) ** 2).sum()
            for tr in self.data) / npts)

        # Get gaps at beginning and end.
        if self.data[0].stats.starttime > self.starttime:
            head_gap_count = 1
            head_gap_length = self.data[0].stats.starttime - self.starttime
        else:
            head_gap_count = 0
            head_gap_length = 0.0
        # We define the endtime as the time of the last sample but the next
        # sample would only start at endtime + delta. Thus the following
        # scenario would not count as a gap at the end:
        #     x -- x -- x -- x -- x -- x
        #                                   | <- self.endtime
        if (self.data[-1].stats.endtime + self.data[-1].stats.delta) < \
                self.endtime:
            tail_gap_count = 1
            tail_gap_length = self.endtime - (
                self.data[-1].stats.endtime + self.data[-1].stats.delta)
        else:
            tail_gap_count = 0.0
            tail_gap_length = 0.0

        gaps = self.data.getGaps()

        self.meta["num_gaps"] = \
            len([_i for _i in gaps if _i[-1] > 0]) + head_gap_count + \
            tail_gap_count
        self.meta["num_overlaps"] = \
            len([_i for _i in gaps if _i[-1] < 0])
        self.meta["gaps_len"] = \
            sum(abs(_i[-2]) for _i in gaps if _i[-1] > 0) + head_gap_length \
            + tail_gap_length
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
            seg['sample_rms'] = np.sqrt((tr.data ** 2).sum() / tr.stats.npts)
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
