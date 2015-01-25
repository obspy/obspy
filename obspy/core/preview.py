# -*- coding: utf-8 -*-
"""
Tools for creating and merging previews.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

from copy import copy
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from obspy.core.utcdatetime import UTCDateTime
import numpy as np


def createPreview(trace, delta=60):
    """
    Creates a preview trace.

    A preview trace consists of maximum minus minimum of all samples within
    ``delta`` seconds. The parameter ``delta`` must be a multiple of the
    sampling rate of the ``trace`` object.

    :type delta: int, optional
    :param delta: Difference between two preview points. Defaults to ``60``.
    :rtype: :class:`~obspy.core.trace.Trace`
    :return: New Trace object.

    This method will modify the original Trace object. Create a copy of the
    Trace object if you want to continue using the original data.
    """
    if not isinstance(delta, int) or delta < 1:
        msg = 'The delta values need to be an Integer and at least 1.'
        raise TypeError(msg)
    data = trace.data
    start_time = trace.stats.starttime.timestamp
    # number of samples for a single slice of delta seconds
    samples_per_slice = delta * int(trace.stats.sampling_rate)
    if samples_per_slice < 1:
        raise ValueError('samples_per_slice is less than 0 - skipping')
    # minimum and maximum of samples before a static time marker
    start = int((delta - start_time % delta) * int(trace.stats.sampling_rate))
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
    if isinstance(diff, np.ma.masked_array):
        diff = np.ma.filled(diff, -1)
    data = np.concatenate([first_diff, diff, last_diff])
    data = np.require(data, dtype=np.float32)
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

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: Stream object to be merged
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: Merged Stream object.
    """
    copied_traces = copy(stream.traces)
    stream.sort()
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
    # Initialize new Stream object.
    new_stream = Stream()
    for value in traces.values():
        if len(value) == 1:
            new_stream.append(value[0])
            continue
        # All traces need to have the same delta value and also be on the same
        # grid spacing. It is enough to only check the sampling rate because
        # the algorithm that creates the preview assures that the grid spacing
        # is correct.
        sampling_rates = set([tr.stats.sampling_rate for tr in value])
        if len(sampling_rates) != 1:
            msg = 'More than one sampling rate for traces with id %s.' % \
                  value[0].id
            raise Exception(msg)
        delta = value[0].stats.delta
        # Check dtype.
        dtypes = set([native_str(tr.data.dtype) for tr in value])
        if len(dtypes) > 1:
            msg = 'Different dtypes for traces with id %s' % value[0].id
            raise Exception(msg)
        dtype = dtypes.pop()
        # Get the minimum start and maximum end time for all traces.
        min_starttime = min([tr.stats.starttime for tr in value])
        max_endtime = max([tr.stats.endtime for tr in value])
        samples = int(round((max_endtime - min_starttime) / delta)) + 1
        data = np.empty(samples, dtype=dtype)
        # Fill with negative one values which corresponds to a gap.
        data[:] = -1
        # Create trace and give starttime.
        new_trace = Trace(data=data, header=value[0].stats)
        # Loop over all traces in value and add to data.
        for trace in value:
            start_index = int((trace.stats.starttime - min_starttime) / delta)
            end_index = start_index + len(trace.data)
            # Element-by-element comparison.
            data[start_index:end_index] = \
                np.maximum(data[start_index:end_index], trace.data)
        # set npts again, because data is changed in place
        new_trace.stats.npts = len(data)
        new_stream.append(new_trace)
    stream.traces = copied_traces
    return new_stream


def resamplePreview(trace, samples, method='accurate'):
    """
    Resamples a preview Trace to the chosen number of samples.

    :type trace: :class:`~obspy.core.trace.Trace`
    :param trace: Trace object to be resampled.
    :type samples: int
    :param samples: Desired number of samples.
    :type method: str, optional
    :param method: Resample method. Available are ``'fast'`` and
        ``'accurate'``. Defaults to ``'accurate'``.

    .. rubric:: Notes

    This method will destroy the data in the original Trace object.
    Deepcopy the Trace if you want to continue using the original data.

    The fast method works by reshaping the data array to a
    sample x int(npts/samples) matrix (npts are the number of samples in
    the original trace) and taking the maximum of each row. Therefore
    the last npts - int(npts/samples)*samples will be omitted. The worst
    case scenario is resampling a 1999 samples array to 1000 samples. 999
    samples, almost half the data will be omitted.

    The accurate method has no such problems because it will move a window
    over the whole array and take the maximum for each window. It loops
    over each window and is up to 10 times slower than the fast method.
    This of course is highly depended on the number of wished samples and
    the original trace and usually the accurate method is still fast
    enough.
    """
    # Only works for preview traces.
    if not hasattr(trace.stats, 'preview') or not trace.stats.preview:
        msg = 'Trace\n%s\n is no preview file.' % str(trace)
        raise Exception(msg)
    # Save same attributes for later use.
    endtime = trace.stats.endtime
    dtype = trace.data.dtype
    npts = trace.stats.npts
    # XXX: Interpolate?
    if trace.stats.npts < samples:
        msg = 'Can only downsample so far. Interpolation not yet implemented.'
        raise NotImplementedError(msg)
    # Return if no change is necessary. There obviously are no omitted samples.
    elif trace.stats.npts == samples:
        return 0
    # Fast method.
    if method == 'fast':
        data = trace.data[:int(npts / samples) * samples]
        data = data.reshape(samples, len(data) // samples)
        trace.data = data.max(axis=1)
        # Set new sampling rate.
        trace.stats.delta = (endtime - trace.stats.starttime) / \
            float(samples - 1)
        # Return number of omitted samples.
        return npts - int(npts / samples) * samples
    # Slow but accurate method.
    elif method == 'accurate':
        new_data = np.empty(samples, dtype=dtype)
        step = trace.stats.npts / float(samples)
        for _i in range(samples):
            new_data[_i] = trace.data[int(_i * step):
                                      int((_i + 1) * step)].max()
        trace.data = new_data
        # Set new sampling rate.
        trace.stats.delta = (endtime - trace.stats.starttime) / \
            float(samples - 1)
        # Return number of omitted samples. Should be 0 for this method.
        return npts - int(samples * step)
    else:
        raise NotImplementedError('Unknown method')
