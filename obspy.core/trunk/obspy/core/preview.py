# -*- coding: utf-8 -*-
"""
Tools for creating and merging previews.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from numpy.ma import is_masked
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime


def createPreview(trace, delta=60):
    """
    Creates a preview trace.
    
    A preview trace consists of maxima minus minima of all samples within
    ``delta`` seconds. The parameter ``delta`` must be a multiple of the
    sampling rate of the ``trace`` object.
    
    Attributes
    ----------
    delta : integer, optional
        Difference between two preview points (default value is 60).

    Notes
    -----
        This method will destroy the data in the original Trace object. This is
        necessary for reduced memory usage. Deepcopy the Trace if you want to
        continue using the original data.

    Returns
    -------
    :class:`~obspy.core.Trace`
        New Trace object.
    """
    if not isinstance(delta, int) or delta < 1:
        msg = 'The delta values need to be an Integer and at least 1.'
        raise TypeError(msg)
    data = trace.data
    start_time = trace.stats.starttime.timestamp
    # number of samples for a single slice of delta seconds
    samples_per_slice = delta * trace.stats.sampling_rate
    # minimum and maximum of samples before a static time marker
    start = (delta - start_time % delta) * trace.stats.sampling_rate
    start_time = start_time - start_time % delta
    if start > (delta / 2) and data[0:start].size:
        first_diff = [data[0:start].max() - data[0:start].min()]
    else:
        # skip starting samples
        first_diff = []
        start_time += delta
    # number of complete slices of data
    number_of_slices = int((len(data) - start) / samples_per_slice)
    # minimum and maximum of remaining samples
    end = samples_per_slice * number_of_slices + start
    if end > (delta / 2) and data[end:].size:
        last_diff = [data[end:].max() - data[end:].min()]
    else:
        # skip tailing samples
        last_diff = []
    # Fill NaN value with -1.
    if np.isnan(last_diff):
        last_diff = -1
    # reshape matrix
    data = trace.data[start:end].reshape([number_of_slices, samples_per_slice])
    # get minimum and maximum for each row
    diff = data.ptp(axis=1)
    # fill masked values with -1 -> means missing data
    data = np.ma.filled(diff, -1) if is_masked(diff) else diff
    data = np.concatenate([first_diff, data, last_diff])
    tr = Trace(data=data, header=trace.stats)
    tr.stats.delta = delta
    tr.stats.npts = len(data)
    tr.stats.starttime = UTCDateTime(start_time)
    tr.stats.preview = True
    return tr


def mergePreviews(stream):
    """
    Merges all preview traces in one Stream object. Does not change the
    original stream because the data needs to be copied anyway.

    Attributes
    ----------
    stream : :class:`~obspy.core.Stream`
        Stream object to be merged

    Returns
    -------
    :class:`~obspy.core.Stream`
        Merged Stream object.
    """
    # Group traces by id.
    traces = {}
    dtypes = []
    for trace in stream:
        # Throw away empty traces.
        if trace.stats.npts == 0:
            continue
        if not hasattr(trace.stats, 'preview') or not trace.stats.preview:
            msg = 'Trace\n%s\n is no preview file.' % str(trace)
            raise Exception(msg)
        traces.setdefault(trace.id, [])
        traces[trace.id].append(trace)
        dtypes.append(trace.data.dtype)
    if len(traces) == 0:
        return Stream()
    # Init new Stream object.
    new_stream = Stream()
    for value in traces.values():
        if len(value) == 1:
            new_stream.append(value)
            continue
        # All traces need to have the same delta value and also be on the same
        # grid spacing. It is enough to only check the sampling rate because the
        # algorithm that creates the preview assures that the grid spacing is
        # correct.
        sampling_rates = set([tr.stats.sampling_rate for tr in value])
        if len(sampling_rates) != 1:
            msg = 'More than one sampling rate for traces with id %s.' % \
                  value[0].id
            raise Exception(msg)
        delta = value[0].stats.delta
        # Check dtype.
        dtypes = set([tr.data.dtype for tr in value])
        if len(dtypes) > 1:
            msg = 'Different dtypes for traces with id %s' % value[0].id
            raise Exception(msg)
        dtype = dtypes.pop()
        # Get the minimum start and maximum endtime for all traces.
        min_starttime = min([tr.stats.starttime for tr in value])
        max_endtime = max([tr.stats.endtime for tr in value])
        samples = (max_endtime - min_starttime) / delta + 1
        data = np.empty(samples, dtype=dtype)
        # Fill with negative one values which corresponds to a gap.
        data[:] = -1
        # Create trace and give starttime.
        new_trace = Trace(data = data, header = value[0].stats)
        new_trace.starttime = min_starttime
        # Loop over all traces in value and add to data.
        for trace in value:
            start_index = int((trace.stats.starttime - min_starttime) / delta)
            end_index = start_index + len(trace.data)
            # Element-by-element comparision.
            data[start_index:end_index] = \
                np.maximum(data[start_index:end_index], trace.data)
        new_stream.append(new_trace)
    return new_stream
