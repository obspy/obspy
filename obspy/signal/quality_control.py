# -*- coding: utf-8 -*-
"""
Quality control module for ObsPy.

Currently requires MiniSEED files as that is the dominant data format in
data centers. Please see the documentation of the
:class:`~obspy.signal.quality_control.MSEEDMetadata` class for usage
instructions.

:authors:
    Luca Trani (trani@knmi.nl)
    Lion Krischer (krischer@geophysik.uni-muenchen.de)
    Mathijs Koymans (koymans@knmi.nl)
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import collections.abc
import io
import json
from operator import attrgetter
from pathlib import Path
from uuid import uuid4

import numpy as np

from obspy import Stream, UTCDateTime, read, __version__
from obspy.core.util.base import get_dependency_version
from obspy.io.mseed.util import get_flags


_PRODUCER = "ObsPy %s" % __version__


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
        elif isinstance(obj, UTCDateTime):
            return str(obj)
        else:
            return super(DataQualityEncoder, self).default(obj)


class MSEEDMetadata(object):
    """
    A container for MiniSEED specific metadata, including quality control
    parameters.

    Reads the MiniSEED files and extracts the data quality metrics. All
    MiniSEED files must have a matching stream ID and quality.

    :param files: One ore more MiniSEED files.
    :type files: str or list[str]
    :type id: str, optional
    :param id: A unique identifier of the to be created QC object. It is
        not verified, that it actually is unique. The user has to take care of
        that. If no id is given, uuid.uuid4() will be used to
        create one which assures uniqueness within one Python run.
        If no fixed id is provided, the ID will be built from prefix
        and a random uuid hash.
    :type prefix: str, optional
    :param prefix: An optional identifier that will be put in front of any
        automatically created id. The prefix will only have an effect
        if `id` is not specified (for a fixed ID string).
    :param starttime: Only use records whose end time is larger then this
        given time. Also specifies the new official start time of the
        metadata object.
    :type starttime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param endtime: Only use records whose start time is smaller then this
        given time. Also specifies the new official end time of the
        metadata object
    :type endtime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param add_c_segments: Calculate metrics for each continuous segment.
    :type add_c_segments: bool
    :param add_flags: Include MiniSEED header statistics in result.
    :type add_flags: bool
    :param waveform_type: The type of waveform data, e.g. ``"seismic"``,
        ``"infrasound"``, ...
    :type waveform_type: str

    .. rubric:: Example

    >>> from obspy.signal.quality_control import
    ...     MSEEDMetadata #doctest: +SKIP
    >>> mseedqc = MSEEDMetadata(['path/to/file',
    ...                          'path/to/file2']) # doctest: +SKIP

    The class requires a list of files for calculating metrics.
    Add optional parameters ``starttime="YYYY-MM-DDThh:mm:ss`` and
    ``endtime="YYYY-MM-DDThh:mm:ss"`` or ``obspy.core.utcdatetime.UTCDateTime``
    to limit metric calculation to this window. Continuous segments are
    returned when ``add_c_segments=True`` and MiniSEED header flags information
    is returned when ``add_flags=True``.

    The calculated metrics are then available in the ``.meta`` dictionary.

    >>> mseedqc.meta  # doctest: +SKIP

    This is intended to be serialized as JSON. Retrieve the JSON string (to
    for example store it in a database or save to a file) with:

    >>> mseedqc.get_json_meta() #doctest: +SKIP
    """
    def __init__(self, files, id=None, prefix="smi:local/qc",
                 starttime=None, endtime=None,
                 add_c_segments=True, add_flags=False,
                 waveform_type="seismic"):
        """
        Reads the MiniSEED files and extracts the data quality metrics.
        """
        self.data = Stream()
        self.all_files = files
        self.files = []

        # Allow anything UTCDateTime can parse.
        if starttime is not None:
            starttime = UTCDateTime(starttime)
        if endtime is not None:
            endtime = UTCDateTime(endtime)

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
            st = read(file, starttime=starttime, endtime=endtime_left,
                      format="mseed", nearest_sample=False)

            # Empty stream or maybe there is no data in the stream for the
            # requested time span.
            if not st:
                continue

            self.files.append(file)

            # Only extend traces with data (npts > 0)
            for tr in st:
                if tr.stats.npts != 0:
                    self.data.extend([tr])

        if not self.data:
            raise ValueError("No data within the temporal constraints.")

        # Do some sanity checks. The class only works with data from a
        # single location so we have to make sure that the existing data on
        # this object and the newly added all have the same identifier.
        ids = [tr.id + "." + str(tr.stats.mseed.dataquality) for tr in
               self.data]
        if len(set(ids)) != 1:
            raise ValueError("All traces must have the same SEED id and "
                             "quality.")

        # Get the last sample and add delta
        final_trace = max(self.data, key=attrgetter('stats.endtime')).stats
        self.endtime = endtime or final_trace.endtime + final_trace.delta

        self.data.sort()

        # Set the metric start and endtime specified by the user.
        # If no start and endtime are given, we pick our own, and the window
        # will start on the first sample and end on the last sample + Δt.
        # This is conform to the definition of [T0, T1).
        self.starttime = starttime or self.data[0].stats.starttime
        self.total_time = self.endtime - self.starttime

        if id is None:
            id = prefix + "/" + str(uuid4())

        # Fill with the meta information.
        self.meta = {
            "wfmetadata_id": id,
            "producer": _PRODUCER,
            "waveform_type": waveform_type,
            "waveform_format": "miniSEED",
            "version": "1.0.0"
        }

        # Get sample left of the user specified starttime
        # This will allow us to determine start continuity in our window
        self._get_gaps_and_overlaps()

        # The calculation of all the metrics begins here
        self._extract_mseed_stream_metadata()
        self._compute_sample_metrics()

        if add_flags:
            self._extract_mseed_flags()

        if add_c_segments:
            self._compute_continuous_seg_sample_metrics()

    def _get_gaps_and_overlaps(self):
        """
        Function to get all gaps and overlaps in the user
        specified (or forced) window.
        """
        self.all_data = Stream()

        body_gap = []
        body_overlap = []

        # Read all the files entirely and calculate gaps and overlaps
        # for the entire segment. Later we will narrow it to our window if
        # it has been specified
        for file in self.all_files:
            self.all_data.extend(read(file, format="mseed", headonly=True))

        # Sort the data by so the start times are in order
        self.all_data.sort()

        # Coverage keeps track of the time used
        coverage = None
        for trace in self.all_data:

            # Extend the endtime of a trace with delta
            trace_end = trace.stats.endtime + trace.stats.delta
            trace_start = trace.stats.starttime

            # If a start boundary has been specified
            if self.window_start is not None:
                if trace_end <= self.window_start:
                    continue

                # In case the start time of a trace comes before the window
                # extend the length to the window start
                cut_trace_start = max(trace_start, self.window_start)
            else:
                cut_trace_start = trace_start

            # If a end boundary has been specified
            if self.window_end is not None:
                if trace_start > self.window_end:
                    continue

                # In case the end time of a trace comes after the window
                # reduce the length to the window end
                cut_trace_end = min(trace_end, self.window_end)
            else:
                cut_trace_end = trace_end

            # Calculate the trace time tolerance as 0.5 * delta
            time_tolerance_max = trace_end + 0.5 * trace.stats.delta
            time_tolerance_min = trace_end - 0.5 * trace.stats.delta

            # Set the initial trace coverage and proceed to the next trace
            if coverage is None:
                coverage = {
                    'start': trace_start,
                    'end': trace_end,
                    'end_min': time_tolerance_min,
                    'end_max': time_tolerance_max
                }
                continue

            # Check if the start time of a trace falls
            # beyond covered end max (time tolerance)
            # this must be interpreted as a gap
            if trace_start > coverage['end_max']:
                body_gap.append(cut_trace_start - coverage['end'])

            # Check overlap in coverage
            # the overlap ends at the end of the trace
            # or add the end of the coverage
            if trace_start <= coverage['end_min']:
                min_end = min(cut_trace_end, coverage['end'])
                body_overlap.append(min_end - cut_trace_start)

            # Extend the coverage of the trace
            # the start coverage remains unchanged
            if trace_end > coverage['end']:
                coverage['end'] = trace_end
                coverage['end_min'] = time_tolerance_min
                coverage['end_max'] = time_tolerance_max

        # We have the coverage by the traces
        # check if there is an end or start gap caused by the
        # window forced by the user
        self.meta['start_gap'] = None
        self.meta['end_gap'] = None
        if self.window_start is not None:
            if coverage['start'] > self.window_start:
                self.meta['start_gap'] = coverage['start'] - self.window_start
                body_gap.append(self.meta['start_gap'])
        if self.window_end is not None:
            if coverage['end'] < self.window_end:
                self.meta['end_gap'] = self.window_end - coverage['end']
                body_gap.append(self.meta['end_gap'])

        # Set the gap and overlap information
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
        Number of records across files before slicing.
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
        # Look for the maximum endtime and minimum starttime in case
        # traces are not in order.
        meta['first_sample'] = min(tr.stats.starttime for tr in self.data)
        meta['last_sample'] = max(tr.stats.endtime for tr in self.data)

        # Add some other parameters to the metadata object
        meta['seed_id'] = self.data[0].id
        meta['files'] = self.files
        meta['start_time'] = self.starttime
        meta['end_time'] = self.endtime
        meta['num_samples'] = self.number_of_samples

        # The number records as given by Trace.stats is always
        # the full number of records in a file, regardless of being
        # sliced between a start & endtime
        # If a start/endtime is specified, we cannot be sure of
        # the # records. We will add this parameter later
        # if add_flags is set to true after looping all records
        if self.window_start is None and self.window_end is None:
            meta['num_records'] = self.number_of_records
        else:
            meta['num_records'] = None

        # The following are lists and may contain multiple unique entries.
        meta['sample_rate'] = \
            sorted(list(set([tr.stats.sampling_rate for tr in self.data])))
        meta['record_length'] = \
            sorted(list(set([tr.stats.mseed.record_length
                             for tr in self.data])))
        meta['encoding'] = \
            sorted(list(set([tr.stats.mseed.encoding for tr in self.data])))

    def _extract_mseed_flags(self):
        flags = get_flags(self.files, starttime=self.starttime,
                          endtime=self.endtime)

        data_quality_flags_seconds = flags["data_quality_flags_percentages"]
        activity_flags_seconds = flags["activity_flags_percentages"]
        io_and_clock_flags_seconds = flags["io_and_clock_flags_percentages"]
        timing_correction = flags["timing_correction"]

        # Only calculate the timing quality statistics if each files has the
        # timing quality set. This should usually be the case. Otherwise we
        # would created tinted statistics. There is still a chance that some
        # records in a file have timing qualities set and others not but
        # that should be small.
        if flags["timing_quality"]:
            tq = flags["timing_quality"]
            timing_quality_mean = tq["mean"]
            timing_quality_min = tq["min"]
            timing_quality_max = tq["max"]
            timing_quality_median = tq["median"]
            timing_quality_lower_quartile = tq["lower_quartile"]
            timing_quality_upper_quartile = tq["upper_quartile"]
        else:
            timing_quality_mean = None
            timing_quality_min = None
            timing_quality_max = None
            timing_quality_median = None
            timing_quality_lower_quartile = None
            timing_quality_upper_quartile = None

        meta = self.meta
        meta['num_records'] = flags['record_count']

        # Set MiniSEED header counts
        meta['miniseed_header_counts'] = {}
        ref = meta['miniseed_header_counts']
        ref['timing_correction'] = flags['timing_correction_count']
        ref['activity_flags'] = flags['activity_flags_counts']
        ref['io_and_clock_flags'] = flags['io_and_clock_flags_counts']
        ref['data_quality_flags'] = flags['data_quality_flags_counts']

        # Set MiniSEED header percentages
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
        # names. Sets MiniSEED header flag percentages
        ref['activity_flags'] = activity_flags_seconds
        ref['data_quality_flags'] = data_quality_flags_seconds
        ref['io_and_clock_flags'] = io_and_clock_flags_seconds

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

        full_samples = np.concatenate([tr.data for tr in self.data])
        self.meta['sample_median'] = np.median(full_samples)
        self.meta['sample_lower_quartile'] = np.percentile(full_samples, 25)
        self.meta['sample_upper_quartile'] = np.percentile(full_samples, 75)

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

        if self.meta['start_gap'] is None and self.window_start is not None:
            first_segment_start = self.window_start
        else:
            first_segment_start = self.data[0].stats.starttime

        # Collect data in arrays of continuous segments
        # Manually set the first segment
        c_seg = {
            'start': first_segment_start,
            'data': self.data[0].data
        }

        c_segs = []
        for i in range(len(self.data)):

            trace_end = self.data[i].stats.endtime + self.data[i].stats.delta
            time_tolerance = 0.5 * self.data[i].stats.delta

            c_seg['s_rate'] = self.data[i].stats.sampling_rate
            c_seg['end'] = trace_end

            # Final trace, make sure to append
            if i == len(self.data) - 1:
                c_segs.append(c_seg)
                break

            trace_offset = abs(self.data[i + 1].stats.starttime - trace_end)

            # Check if trace endtime is equal to (trace + 1) starttime
            # and if the sampling_rates match, if so, extend the data with
            # data from trace + 1 and extend the endtime
            # Otherwise the segment stops being continuous and we append it
            # and we create a new data segment
            if (trace_offset < time_tolerance and
                    self.data[i + 1].stats.sampling_rate == c_seg['s_rate']):
                c_seg['data'] = np.concatenate((c_seg['data'],
                                                self.data[i + 1].data))
                c_seg['end'] = self.data[i + 1].stats.endtime + \
                    self.data[i + 1].stats.delta
            else:
                c_segs.append(c_seg)
                c_seg = {
                    'data': self.data[i + 1].data,
                    'start': self.data[i + 1].stats.starttime}

        # Set array of continuous segments from this data
        self.meta['c_segments'] = [self._parse_c_stats(seg) for seg in c_segs]

    def _parse_c_stats(self, tr):
        """
        :param tr: custom dictionary with start, end, data, and sampling_rate
            of a continuous trace
        """
        seg = {}

        # Set continuous segments start & end
        # limit to specified window start/end if set
        if self.window_start is not None:
            seg['start_time'] = max(self.window_start, tr['start'])
        else:
            seg['start_time'] = tr['start']

        if self.window_end is not None:
            seg['end_time'] = min(self.window_end, tr['end'])
        else:
            seg['end_time'] = tr['end']

        seg['sample_rate'] = tr['s_rate']
        seg['sample_min'] = tr['data'].min()
        seg['sample_max'] = tr['data'].max()
        seg['sample_mean'] = tr['data'].mean()
        seg['sample_median'] = np.median(tr['data'])
        seg['sample_rms'] = np.sqrt((tr['data'].astype(object) ** 2).sum() /
                                    len(tr['data']))
        seg['sample_lower_quartile'] = np.percentile(tr['data'], 25)
        seg['sample_upper_quartile'] = np.percentile(tr['data'], 75)
        seg['sample_stdev'] = tr['data'].std()
        seg['num_samples'] = len(tr['data'])
        seg['segment_length'] = seg['end_time'] - seg['start_time']

        return seg

    def get_json_meta(self, validate=False):
        """
        Serialize the meta dictionary to JSON.

        :param validate: Validate the JSON string against the schema before
            returning.
        :type validate: bool

        :return: JSON containing the MSEED metadata
        """
        meta = json.dumps(self.meta, cls=DataQualityEncoder, indent=4)

        if validate:
            self.validate_qc_metrics(meta)

        return meta

    def validate_qc_metrics(self, qc_metrics):
        """
        Validate the passed metrics against the JSON schema.

        :param qc_metrics: The quality metrics to be validated.
        :type qc_metrics: dict, str, or file-like object
        """
        import jsonschema

        # Judging from the changelog 1.0.0 appears to be the first version
        # to have fully working support for references.
        _v = get_dependency_version("jsonschema")
        if _v < [1, 0, 0]:  # pragma: no cover
            msg = ("Validating the QC metrics requires jsonschema >= 1.0.0 "
                   "You have %s. Please update." %
                   get_dependency_version("jsonschema", raw_string=True))
            raise ValueError(msg)

        schema_path = Path(__file__).parent/"data"/"wf_metadata_schema.json"
        with io.open(schema_path, "rt") as fh:
            schema = json.load(fh)

        # If passed as a dictionary, serialize and derialize to get the
        # mapping from Python object to JSON type.
        if isinstance(qc_metrics, collections.abc.Mapping):
            qc_metrics = json.loads(self.get_json_meta(validate=False))
        elif hasattr(qc_metrics, "read"):
            qc_metrics = json.load(qc_metrics)
        else:
            qc_metrics = json.loads(qc_metrics)

        jsonschema.validate(qc_metrics, schema)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
