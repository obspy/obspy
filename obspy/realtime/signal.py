# -*- coding: utf-8 -*-
"""
Signal processing functions for RtMemory objects.

For sequential packet processing that requires memory (which includes recursive
filtering), each processing function (e.g., :mod:`obspy.realtime.signal`)
needs to manage the initialization and update of
:class:`~obspy.realtime.rtmemory.RtMemory` object(s), and needs to know when
and how to get values from this memory.

For example: Boxcar smoothing: For each new data point available past the end
of the boxcar, the original, un-smoothed data point value at the beginning of
the boxcar has to be subtracted from the running boxcar sum, this value may be
in a previous packet, so has to be retrieved from memory see
:func:`obspy.realtime.signal.boxcar`.

:copyright:
    The ObsPy Development Team (devs@obspy.org), Anthony Lomax & Alessia Maggi
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import math
import sys
import numpy as np
from obspy.core.trace import Trace, UTCDateTime
from obspy.realtime.rtmemory import RtMemory

_PI = math.pi
_TWO_PI = 2.0 * math.pi
_MIN_FLOAT_VAL = 1.0e-20


def offset(trace, offset=0.0, rtmemory_list=None):  # @UnusedVariable
    """
    Add the specified offset to the data.

    :type trace: :class:`~obspy.core.trace.Trace`
    :param trace: :class:`~obspy.core.trace.Trace` object to append to this
        RtTrace
    :type offset: float, optional
    :param offset: offset (default is 0.0)
    :type rtmemory_list: list of :class:`~obspy.realtime.rtmemory.RtMemory`,
        optional
    :param rtmemory_list: Persistent memory used by this process for specified
        trace
    :rtype: NumPy :class:`numpy.ndarray`
    :return: Processed trace data from appended Trace object
    """

    if not isinstance(trace, Trace):
        msg = "Trace parameter must be an obspy.core.trace.Trace object."
        raise ValueError(msg)

    trace.data += offset
    return trace.data


def scale(trace, factor=1.0, rtmemory_list=None):  # @UnusedVariable
    """
    Scale array data samples by specified factor.

    :type trace: :class:`~obspy.core.trace.Trace`
    :param trace:  :class:`~obspy.core.trace.Trace` object to append to this
        RtTrace
    :type factor: float, optional
    :param factor: Scale factor (default is 1.0).
    :type rtmemory_list: list of :class:`~obspy.realtime.rtmemory.RtMemory`,
        optional
    :param rtmemory_list: Persistent memory used by this process for specified
        trace.
    :rtype: NumPy :class:`numpy.ndarray`
    :return: Processed trace data from appended Trace object.
    """
    if not isinstance(trace, Trace):
        msg = "trace parameter must be an obspy.core.trace.Trace object."
        raise ValueError(msg)
    # XXX not sure how this should be for realtime analysis, here
    # I assume, we do not want to change the underlying dtype
    trace.data *= np.array(factor, dtype=trace.data.dtype)
    return trace.data


def integrate(trace, rtmemory_list=None):
    """
    Apply simple rectangular integration to array data.

    :type trace: :class:`~obspy.core.trace.Trace`
    :param trace:  :class:`~obspy.core.trace.Trace` object to append to this
        RtTrace
    :type rtmemory_list: list of :class:`~obspy.realtime.rtmemory.RtMemory`,
        optional
    :param rtmemory_list: Persistent memory used by this process for specified
        trace.
    :rtype: NumPy :class:`numpy.ndarray`
    :return: Processed trace data from appended Trace object.
    """
    if not isinstance(trace, Trace):
        msg = "trace parameter must be an obspy.core.trace.Trace object."
        raise ValueError(msg)

    if not rtmemory_list:
        rtmemory_list = [RtMemory()]

    sample = trace.data
    if np.size(sample) < 1:
        return sample

    delta_time = trace.stats.delta

    rtmemory = rtmemory_list[0]

    # initialize memory object
    if not rtmemory.initialized:
        memory_size_input = 0
        memory_size_output = 1
        rtmemory.initialize(sample.dtype, memory_size_input,
                            memory_size_output, 0, 0)

    sum_ = rtmemory.output[0]

    for i in range(np.size(sample)):
        sum_ += sample[i] * delta_time
        sample[i] = sum_

    rtmemory.output[0] = sum_

    return sample


def differentiate(trace, rtmemory_list=None):
    """
    Apply simple differentiation to array data.

    :type trace: :class:`~obspy.core.trace.Trace`
    :param trace:  :class:`~obspy.core.trace.Trace` object to append to this
        RtTrace
    :type rtmemory_list: list of :class:`~obspy.realtime.rtmemory.RtMemory`,
        optional
    :param rtmemory_list: Persistent memory used by this process for specified
        trace.
    :rtype: NumPy :class:`numpy.ndarray`
    :return: Processed trace data from appended Trace object.
    """
    if not isinstance(trace, Trace):
        msg = "trace parameter must be an obspy.core.trace.Trace object."
        raise ValueError(msg)

    if not rtmemory_list:
        rtmemory_list = [RtMemory()]

    sample = trace.data
    if np.size(sample) < 1:
        return(sample)

    delta_time = trace.stats.delta

    rtmemory = rtmemory_list[0]

    # initialize memory object
    if not rtmemory.initialized:
        memory_size_input = 1
        memory_size_output = 0
        rtmemory.initialize(sample.dtype, memory_size_input,
                            memory_size_output, 0, 0)
        # avoid large diff value for first output sample
        rtmemory.input[0] = sample[0]

    previous_sample = rtmemory.input[0]

    for i in range(np.size(sample)):
        diff = (sample[i] - previous_sample) / delta_time
        previous_sample = sample[i]
        sample[i] = diff

    rtmemory.input[0] = previous_sample

    return sample


def boxcar(trace, width, rtmemory_list=None):
    """
    Apply boxcar smoothing to data in array sample.

    :type trace: :class:`~obspy.core.trace.Trace`
    :param trace:  :class:`~obspy.core.trace.Trace` object to append to this
        RtTrace
    :type width: int
    :param width: Width in number of sample points for filter.
    :type rtmemory_list: list of :class:`~obspy.realtime.rtmemory.RtMemory`,
        optional
    :param rtmemory_list: Persistent memory used by this process for specified
        trace.
    :rtype: NumPy :class:`numpy.ndarray`
    :return: Processed trace data from appended Trace object.
    """
    if not isinstance(trace, Trace):
        msg = "trace parameter must be an obspy.core.trace.Trace object."
        raise ValueError(msg)

    if not width > 0:
        msg = "width parameter not specified or < 1."
        raise ValueError(msg)

    if not rtmemory_list:
        rtmemory_list = [RtMemory()]

    sample = trace.data

    rtmemory = rtmemory_list[0]

    # initialize memory object
    if not rtmemory.initialized:
        memory_size_input = width
        memory_size_output = 0
        rtmemory.initialize(sample.dtype, memory_size_input,
                            memory_size_output, 0, 0)

    # initialize array for time-series results
    new_sample = np.zeros(np.size(sample), sample.dtype)

    i = 0
    i1 = i - width
    i2 = i  # causal boxcar of width width
    sum_ = 0.0
    icount = 0
    for i in range(np.size(sample)):
        value = 0.0
        if (icount == 0):  # first pass, accumulate sum
            for n in range(i1, i2 + 1):
                if (n < 0):
                    value = rtmemory.input[width + n]
                else:
                    value = sample[n]
                sum_ += value
                icount = icount + 1
        else:  # later passes, update sum
            if ((i1 - 1) < 0):
                value = rtmemory.input[width + (i1 - 1)]
            else:
                value = sample[(i1 - 1)]
            sum_ -= value
            if (i2 < 0):
                value = rtmemory.input[width + i2]
            else:
                value = sample[i2]
            sum_ += value
        if (icount > 0):
            new_sample[i] = (float)(sum_ / float(icount))
        else:
            new_sample[i] = 0.0
        i1 = i1 + 1
        i2 = i2 + 1

    rtmemory.updateInput(sample)

    return new_sample


def tauc(trace, width, rtmemory_list=None):
    """
    Calculate instantaneous period in a fixed window (Tau_c).

    .. seealso::

        Implements equations 1-3 in [Allen2003]_ except use a fixed width
        window instead of decay function.

    :type trace: :class:`~obspy.core.trace.Trace`
    :param trace:  :class:`~obspy.core.trace.Trace` object to append to this
        RtTrace
    :type width: int
    :param width: Width in number of sample points for tauc window.
    :type rtmemory_list: list of :class:`~obspy.realtime.rtmemory.RtMemory`,
        optional
    :param rtmemory_list: Persistent memory used by this process for specified
        trace.
    :rtype: NumPy :class:`numpy.ndarray`
    :return: Processed trace data from appended Trace object.
    """
    if not isinstance(trace, Trace):
        msg = "trace parameter must be an obspy.core.trace.Trace object."
        raise ValueError(msg)

    if not width > 0:
        msg = "tauc: width parameter not specified or < 1."
        raise ValueError(msg)

    if not rtmemory_list:
        rtmemory_list = [RtMemory(), RtMemory()]

    sample = trace.data
    delta_time = trace.stats.delta

    rtmemory = rtmemory_list[0]
    rtmemory_dval = rtmemory_list[1]

    sample_last = 0.0

    # initialize memory object
    if not rtmemory.initialized:
        memory_size_input = width
        memory_size_output = 1
        rtmemory.initialize(sample.dtype, memory_size_input,
                            memory_size_output, 0, 0)
        sample_last = sample[0]
    else:
        sample_last = rtmemory.input[width - 1]

    # initialize memory object
    if not rtmemory_dval.initialized:
        memory_size_input = width
        memory_size_output = 1
        rtmemory_dval.initialize(sample.dtype, memory_size_input,
                                 memory_size_output, 0, 0)

    new_sample = np.zeros(np.size(sample), sample.dtype)
    deriv = np.zeros(np.size(sample), sample.dtype)

    # sample_last = rtmemory.input[width - 1]
    sample_d = 0.0
    deriv_d = 0.0
    xval = rtmemory.output[0]
    dval = rtmemory_dval.output[0]

    for i in range(np.size(sample)):

        sample_d = sample[i]
        deriv_d = (sample_d - sample_last) / delta_time
        indexBegin = i - width
        if (indexBegin >= 0):
            xval = xval - (sample[indexBegin]) * (sample[indexBegin]) \
                + sample_d * sample_d
            dval = dval - deriv[indexBegin] * deriv[indexBegin] \
                + deriv_d * deriv_d
        else:
            index = i
            xval = xval - rtmemory.input[index] * rtmemory.input[index] \
                + sample_d * sample_d
            dval = dval \
                - rtmemory_dval.input[index] * rtmemory_dval.input[index] \
                + deriv_d * deriv_d
        deriv[i] = deriv_d
        sample_last = sample_d
        # if (xval > _MIN_FLOAT_VAL &  & dval > _MIN_FLOAT_VAL) {
        if (dval > _MIN_FLOAT_VAL):
            new_sample[i] = _TWO_PI * math.sqrt(xval / dval)
        else:
            new_sample[i] = 0.0

    # update memory
    rtmemory.output[0] = xval
    rtmemory.updateInput(sample)
    rtmemory_dval.output[0] = dval
    rtmemory_dval.updateInput(deriv)

    return new_sample

# memory object indices for storing specific values
_AMP_AT_PICK = 0
_HAVE_USED_MEMORY = 1
_FLAG_COMPETE_MWP = 2
_INT_INT_SUM = 3
_POLARITY = 4
_MEMORY_SIZE_OUTPUT = 5


def mwpIntegral(trace, max_time, ref_time, mem_time=1.0, gain=1.0,
                rtmemory_list=None):
    """
    Calculate Mwp integral on a displacement trace.

    .. seealso:: [Tsuboi1999]_ and [Tsuboi1995]_

    :type trace: :class:`~obspy.core.trace.Trace`
    :param trace:  :class:`~obspy.core.trace.Trace` object to append to this
        RtTrace
    :type max_time: float
    :param max_time: Maximum time in seconds after ref_time to apply Mwp
        integration.
    :type ref_time: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param ref_time: Reference date and time of the data sample
        (e.g. P pick time) at which to begin Mwp integration.
    :type mem_time: float, optional
    :param mem_time: Length in seconds of data memory (must be much larger
        than maximum delay between pick declaration and pick time). Defaults
        to ``1.0``.
    :type gain: float, optional
    :param gain: Nominal gain to convert input displacement trace to meters
        of ground displacement. Defaults to ``1.0``.
    :type rtmemory_list: list of :class:`~obspy.realtime.rtmemory.RtMemory`,
        optional
    :param rtmemory_list: Persistent memory used by this process for specified
        trace.
    :rtype: NumPy :class:`numpy.ndarray`
    :return: Processed trace data from appended Trace object.
    """
    if not isinstance(trace, Trace):
        msg = "trace parameter must be an obspy.core.trace.Trace object."
        raise ValueError(msg)

    if not isinstance(ref_time, UTCDateTime):
        msg = "ref_time must be an obspy.core.utcdatetime.UTCDateTime object."
        raise ValueError(msg)

    if not max_time >= 0:
        msg = "max_time parameter not specified or < 0."
        raise ValueError(msg)

    if not rtmemory_list:
        rtmemory_list = [RtMemory()]

    sample = trace.data
    delta_time = trace.stats.delta

    rtmemory = rtmemory_list[0]

    # initialize memory object
    if not rtmemory.initialized:
        memory_size_input = int(0.5 + mem_time * trace.stats.sampling_rate)
        memory_size_output = _MEMORY_SIZE_OUTPUT
        rtmemory.initialize(sample.dtype, memory_size_input,
                            memory_size_output, 0, 0)

    new_sample = np.zeros(np.size(sample), sample.dtype)

    ioffset_pick = int(round(
                       (ref_time - trace.stats.starttime)
                       * trace.stats.sampling_rate))
    ioffset_mwp_min = ioffset_pick

    # set reference amplitude
    if ioffset_mwp_min >= 0 and ioffset_mwp_min < trace.data.size:
        # value in trace data array
        rtmemory.output[_AMP_AT_PICK] = trace.data[ioffset_mwp_min]
    elif ioffset_mwp_min >= -(np.size(rtmemory.input)) and ioffset_mwp_min < 0:
        # value in memory array
        index = ioffset_mwp_min + np.size(rtmemory.input)
        rtmemory.output[_AMP_AT_PICK] = rtmemory.input[index]
    elif ioffset_mwp_min < -(np.size(rtmemory.input)) \
            and not rtmemory.output[_HAVE_USED_MEMORY]:
        msg = "mem_time not large enough to buffer required input data."
        raise ValueError(msg)
    if ioffset_mwp_min < 0 and rtmemory.output[_HAVE_USED_MEMORY]:
        ioffset_mwp_min = 0
    else:
        rtmemory.output[_HAVE_USED_MEMORY] = 1
    # set Mwp end index corresponding to maximum duration
    mwp_end_index = int(round(max_time / delta_time))
    ioffset_mwp_max = mwp_end_index + ioffset_pick
    if ioffset_mwp_max < trace.data.size:
        rtmemory.output[_FLAG_COMPETE_MWP] = 1  # will complete
    if ioffset_mwp_max > trace.data.size:
        ioffset_mwp_max = trace.data.size
    # apply double integration, check for extrema
    mwp_amp_at_pick = rtmemory.output[_AMP_AT_PICK]
    mwp_int_int_sum = rtmemory.output[_INT_INT_SUM]
    polarity = rtmemory.output[_POLARITY]
    amplitude = 0.0
    for n in range(ioffset_mwp_min, ioffset_mwp_max):
        if n >= 0:
            amplitude = trace.data[n]
        elif n >= -(np.size(rtmemory.input)):
            # value in memory array
            index = n + np.size(rtmemory.input)
            amplitude = rtmemory.input[index]
        else:
            msg = "Error: Mwp: attempt to access rtmemory.input array of " + \
                "size=%d at invalid index=%d: this should not happen!" % \
                (np.size(rtmemory.input), n + np.size(rtmemory.input))
            print(msg)
            continue  # should never reach here
        disp_amp = amplitude - mwp_amp_at_pick
        # check displacement polarity
        if disp_amp >= 0.0:  # pos
            # check if past extremum
            if polarity < 0:  # passed from neg to pos displacement
                mwp_int_int_sum *= -1.0
                mwp_int_int_sum = 0
            polarity = 1
        elif disp_amp < 0.0:  # neg
            # check if past extremum
            if polarity > 0:  # passed from pos to neg displacement
                mwp_int_int_sum = 0
            polarity = -1
        mwp_int_int_sum += (amplitude - mwp_amp_at_pick) * delta_time / gain
        new_sample[n] = mwp_int_int_sum

    rtmemory.output[_INT_INT_SUM] = mwp_int_int_sum
    rtmemory.output[_POLARITY] = polarity

    # update memory
    rtmemory.updateInput(sample)

    return new_sample


MWP_INVALID = -9.9
# 4.213e19 - Tsuboi 1995, 1999
MWP_CONST = 4.0 * _PI  # 4 PI
MWP_CONST *= 3400.0  # rho
MWP_CONST *= 7900.0 * 7900.0 * 7900.0  # Pvel**3
MWP_CONST *= 2.0  # FP average radiation pattern
MWP_CONST *= (10000.0 / 90.0)  # distance deg -> km
MWP_CONST *= 1000.0  # distance km -> meters
# http://mail.python.org/pipermail/python-list/2010-February/1235196.html, ff.
try:
    FLOAT_MIN = sys.float_info.min
except AttributeError:
    FLOAT_MIN = 1.1e-37


def calculateMwpMag(peak, epicentral_distance):
    """
    Calculate Mwp magnitude.

    .. seealso:: [Tsuboi1999]_ and [Tsuboi1995]_

    :type peak: float
    :param peak: Peak value of integral of displacement seismogram.
    :type epicentral_distance: float
    :param epicentral_distance: Great-circle epicentral distance from station
        in degrees.
    :rtype: float
    :returns: Calculated Mwp magnitude.
    """
    moment = MWP_CONST * peak * epicentral_distance
    mwp_mag = MWP_INVALID
    if moment > FLOAT_MIN:
        mwp_mag = (2.0 / 3.0) * (math.log10(moment) - 9.1)
    return mwp_mag


def kurtosis(trace, win=3.0, rtmemory_list=None):
    """
    Apply recursive kurtosis calculation on data.

    Recursive kurtosis is computed using the [ChassandeMottin2002]_
    formulation adjusted to give the kurtosis of a Gaussian distribution = 0.0.

    :type trace: :class:`~obspy.core.trace.Trace`
    :param trace: :class:`~obspy.core.trace.Trace` object to append to this
        RtTrace
    :type win: float, optional
    :param win: window length in seconds for the kurtosis (default is 3.0 s)
    :type rtmemory_list: list of :class:`~obspy.realtime.rtmemory.RtMemory`,
        optional
    :param rtmemory_list: Persistent memory used by this process for specified
        trace
    :rtype: NumPy :class:`numpy.ndarray`
    :return: Processed trace data from appended Trace object
    """
    if not isinstance(trace, Trace):
        msg = "Trace parameter must be an obspy.core.trace.Trace object."
        raise ValueError(msg)

    # if this is the first appended trace, the rtmemory_list will be None
    if not rtmemory_list:
        rtmemory_list = [RtMemory(), RtMemory(), RtMemory()]

    # deal with case of empty trace
    sample = trace.data
    if np.size(sample) < 1:
        return sample

    # get simple info from trace
    npts = len(sample)
    dt = trace.stats.delta

    # set some constants for the kurtosis calculation
    C1 = dt / float(win)
    a1 = 1.0 - C1
    C2 = (1.0 - a1 * a1) / 2.0
    bias = -3 * C1 - 3.0

    # prepare the output array
    kappa4 = np.empty(npts, sample.dtype)

    # initialize the real-time memory needed to store
    # the recursive kurtosis coefficients until the
    # next bloc of data is added
    rtmemory_mu1 = rtmemory_list[0]
    rtmemory_mu2 = rtmemory_list[1]
    rtmemory_k4_bar = rtmemory_list[2]

    # there are three memory objects, one for each "last" coefficient
    # that needs carrying over
    # initialize mu1_last to 0
    if not rtmemory_mu1.initialized:
        memory_size_input = 1
        memory_size_output = 0
        rtmemory_mu1.initialize(sample.dtype, memory_size_input,
                                memory_size_output, 0, 0)

    # initialize mu2_last (sigma) to 1
    if not rtmemory_mu2.initialized:
        memory_size_input = 1
        memory_size_output = 0
        rtmemory_mu2.initialize(sample.dtype, memory_size_input,
                                memory_size_output, 1, 0)

    # initialize k4_bar_last to 0
    if not rtmemory_k4_bar.initialized:
        memory_size_input = 1
        memory_size_output = 0
        rtmemory_k4_bar.initialize(sample.dtype, memory_size_input,
                                   memory_size_output, 0, 0)

    mu1_last = rtmemory_mu1.input[0]
    mu2_last = rtmemory_mu2.input[0]
    k4_bar_last = rtmemory_k4_bar.input[0]

    # do recursive kurtosis
    for i in range(npts):
        mu1 = a1 * mu1_last + C1 * sample[i]
        dx2 = (sample[i] - mu1_last) * (sample[i] - mu1_last)
        mu2 = a1 * mu2_last + C2 * dx2
        dx2 = dx2 / mu2_last
        k4_bar = (1 + C1 - 2 * C1 * dx2) * k4_bar_last + C1 * dx2 * dx2
        kappa4[i] = k4_bar + bias
        mu1_last = mu1
        mu2_last = mu2
        k4_bar_last = k4_bar

    rtmemory_mu1.input[0] = mu1_last
    rtmemory_mu2.input[0] = mu2_last
    rtmemory_k4_bar.input[0] = k4_bar_last

    return kappa4
