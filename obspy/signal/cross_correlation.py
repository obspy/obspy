#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: cross_correlation.py
#   Author: Moritz Beyreuther, Tobias Megies, Tom Eulenfeld
#    Email: megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2016 Moritz Beyreuther, Tobias Megies, Tom Eulenfeld
# ------------------------------------------------------------------
"""
Signal processing routines based on cross correlation techniques.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import ctypes as C
from distutils.version import LooseVersion
import warnings

import numpy as np
import scipy

from obspy import Stream, Trace
from obspy.core.util.misc import MatplotlibBackend
from obspy.signal.headers import clibsignal
from obspy.signal.invsim import cosine_taper


def _pad_zeros(a, num, num2=None):
    """Pad num zeros at both sides of array a"""
    if num2 is None:
        num2 = num
    hstack = [np.zeros(num, dtype=a.dtype), a, np.zeros(num2, dtype=a.dtype)]
    return np.hstack(hstack)


def _call_scipy_correlate(a, b, mode, method):
    """
    Call the correct correlate function depending on Scipy version and method
    """
    if LooseVersion(scipy.__version__) >= LooseVersion('0.19'):
        cc = scipy.signal.correlate(a, b, mode=mode, method=method)
    elif method in ('fft', 'auto'):
        cc = scipy.signal.fftconvolve(a, b[::-1], mode=mode)
    elif method == 'direct':
        cc = scipy.signal.correlate(a, b, mode=mode)
    else:
        msg = "method keyword has to be one of ('auto', 'fft', 'direct')"
        raise ValueError(msg)
    return cc


def _xcorr_padzeros(a, b, shift, method):
    """
    Cross-correlation using SciPy with mode='valid' and precedent zero padding
    """
    if shift is None:
        shift = (len(a) + len(b) - 1) // 2
    dif = len(a) - len(b) - 2 * shift
    if dif > 0:
        b = _pad_zeros(b, dif // 2)
    else:
        a = _pad_zeros(a, -dif // 2)
    return _call_scipy_correlate(a, b, 'valid', method)


def _xcorr_slice(a, b, shift, method):
    """
    Cross-correlation using SciPy with mode='full' and subsequent slicing
    """
    mid = (len(a) + len(b) - 1) // 2
    if shift is None:
        shift = mid
    if shift > mid:
        # Such a large shift is not possible without zero padding
        return _xcorr_padzeros(a, b, shift, method)
    cc = _call_scipy_correlate(a, b, 'full', method)
    return cc[mid - shift:mid + shift + len(cc) % 2]


def correlate(a, b, shift, demean=True, normalize='naive', method='auto',
              domain=None):
    """
    Cross-correlation of two signals up to a specified maximal shift.

    This function only allows 'naive' normalization with the overall
    standard deviations. This is a reasonable approximation for signals of
    similar length and a relatively small shift parameter
    (e.g. noise cross-correlation).
    If you are interested in the full cross-correlation function better use
    :func:`~obspy.signal.cross_correlation.correlate_template` which also
    provides correct normalization.

    :type a: :class:`~numpy.ndarray`, :class:`~obspy.core.trace.Trace`
    :param a: first signal
    :type b: :class:`~numpy.ndarray`, :class:`~obspy.core.trace.Trace`
    :param b: second signal to correlate with first signal
    :param int shift: Number of samples to shift for cross correlation.
        The cross-correlation will consist of ``2*shift+1`` or
        ``2*shift`` samples. The sample with zero shift will be in the middle.
    :param bool demean: Demean data beforehand.
    :param normalize: Method for normalization of cross-correlation.
        One of ``'naive'`` or ``None``
        (``True`` and ``False`` are supported for backwards compatibility).
        ``'naive'`` normalizes by the overall standard deviation.
        ``None`` does not normalize.
    :param str method: Method to use to calculate the correlation.
         ``'direct'``: The correlation is determined directly from sums,
         the definition of correlation.
         ``'fft'`` The Fast Fourier Transform is used to perform the
         correlation more quickly.
         ``'auto'`` Automatically chooses direct or Fourier method based on an
         estimate of which is faster. (Only availlable for SciPy versions >=
         0.19. For older Scipy version method defaults to ``'fft'``.)
    :param str domain: Deprecated. Please use the method argument.

    :return: cross-correlation function.

    To calculate shift and value of the maximum of the returned
    cross-correlation function use
    :func:`~obspy.signal.cross_correlation.xcorr_max`.

    .. note::

        For most input parameters cross-correlation using the FFT is much
        faster.
        Only for small values of ``shift`` (approximately less than 100)
        direct time domain cross-correlation migth save some time.

    .. note::

        If the signals have different length, they will be aligned around
        their middle. The sample with zero shift in the cross-correlation
        function corresponds to this correlation:

        ::

            --aaaa--
            bbbbbbbb

        For odd ``len(a)-len(b)`` the cross-correlation function will
        consist of only ``2*shift`` samples because a shift of 0
        corresponds to the middle between two samples.

    .. rubric:: Example

    >>> from obspy import read
    >>> a = read()[0][450:550]
    >>> b = a[:-2]
    >>> cc = correlate(a, b, 2)
    >>> cc
    array([ 0.62390515,  0.99630851,  0.62187106, -0.05864797, -0.41496995])
    >>> shift, value = xcorr_max(cc)
    >>> shift
    -1
    >>> round(value, 3)
    0.996
    """
    if normalize is False:
        normalize = None
    if normalize is True:
        normalize = 'naive'
    if domain is not None:
        if domain == 'freq':
            method = 'fft'
        elif domain == 'time':
            method = 'direct'
        from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
        msg = ("'domain' keyword of correlate function is deprecated and will "
               "be removed in a subsequent ObsPy release. "
               "Please use the 'method' keyword.")
        warnings.warn(msg, ObsPyDeprecationWarning)
    # if we get Trace objects, use their data arrays
    if isinstance(a, Trace):
        a = a.data
    if isinstance(b, Trace):
        b = b.data
    a = np.asarray(a)
    b = np.asarray(b)
    if demean:
        a = a - np.mean(a)
        b = b - np.mean(b)
    # choose the usually faster xcorr function for each method
    _xcorr = _xcorr_padzeros if method == 'direct' else _xcorr_slice
    cc = _xcorr(a, b, shift, method)
    if normalize == 'naive':
        norm = (np.sum(a ** 2) * np.sum(b ** 2)) ** 0.5
        if norm <= np.finfo(float).eps:
            # norm is zero
            # => cross-correlation function will have only zeros
            cc[:] = 0
        elif cc.dtype == float:
            cc /= norm
        else:
            cc = cc / norm
    elif normalize is not None:
        raise ValueError("normalize has to be one of (None, 'naive'))")
    return cc


def _window_sum(data, window_len):
    """Rolling sum of data"""
    window_sum = np.cumsum(data)
    # in-place equivalent of
    # window_sum = window_sum[window_len:] - window_sum[:-window_len]
    # return window_sum
    np.subtract(window_sum[window_len:], window_sum[:-window_len],
                out=window_sum[:-window_len])
    return window_sum[:-window_len]


def correlate_template(data, template, mode='valid', normalize='full',
                       demean=True, method='auto'):
    """
    Normalized cross-correlation of two signals with specified mode.

    If you are interested only in a part of the cross-correlation function
    around zero shift consider using function
    :func:`~obspy.signal.cross_correlation.correlate` which allows to
    explicetly specify the maximum shift.

    :type data: :class:`~numpy.ndarray`, :class:`~obspy.core.trace.Trace`
    :param data: first signal
    :type template: :class:`~numpy.ndarray`, :class:`~obspy.core.trace.Trace`
    :param template: second signal to correlate with first signal.
        Its length must be smaller or equal to the length of ``data``.
    :param str mode: correlation mode to use.
        It is passed to the used correlation function.
        See :func:`scipy.signal.correlate` for possible options.
        The parameter determines the length of the correlation function.
    :param normalize:
        One of ``'naive'``, ``'full'`` or ``None``.
        ``'full'`` normalizes every correlation properly,
        whereas ``'naive'`` normalizes by the overall standard deviations.
        ``None`` does not normalize.
    :param demean: Demean data beforehand. For ``normalize='full'`` data is
        demeaned in different windows for each correlation value.
    :param str method: Method to use to calculate the correlation.
         ``'direct'``: The correlation is determined directly from sums,
         the definition of correlation.
         ``'fft'`` The Fast Fourier Transform is used to perform the
         correlation more quickly.
         ``'auto'`` Automatically chooses direct or Fourier method based on an
         estimate of which is faster. (Only availlable for SciPy versions >=
         0.19. For older Scipy version method defaults to ``'fft'``.)

    :return: cross-correlation function.

    .. note::
        Calling the function with ``demean=True, normalize='full'`` (default)
        returns the zero-normalized cross-correlation function.
        Calling the function with ``demean=False, normalize='full'``
        returns the normalized cross-correlation function.

    .. rubric:: Example

    >>> from obspy import read
    >>> data = read()[0]
    >>> template = data[450:550]
    >>> cc = correlate_template(data, template)
    >>> index = np.argmax(cc)
    >>> index
    450
    >>> round(cc[index], 9)
    1.0
    """
    # if we get Trace objects, use their data arrays
    if isinstance(data, Trace):
        data = data.data
    if isinstance(template, Trace):
        template = template.data
    data = np.asarray(data)
    template = np.asarray(template)
    lent = len(template)
    if len(data) < lent:
        raise ValueError('Data must not be shorter than template.')
    if demean:
        template = template - np.mean(template)
        if normalize != 'full':
            data = data - np.mean(data)
    cc = _call_scipy_correlate(data, template, mode, method)
    if normalize is not None:
        tnorm = np.sum(template ** 2)
        if normalize == 'naive':
            norm = (tnorm * np.sum(data ** 2)) ** 0.5
            if norm <= np.finfo(float).eps:
                cc[:] = 0
            elif cc.dtype == float:
                cc /= norm
            else:
                cc = cc / norm
        elif normalize == 'full':
            pad = len(cc) - len(data) + lent
            if mode == 'same':
                pad1, pad2 = (pad + 2) // 2, (pad - 1) // 2
            else:
                pad1, pad2 = (pad + 1) // 2, pad // 2
            data = _pad_zeros(data, pad1, pad2)
            # in-place equivalent of
            # if demean:
            #     norm = ((_window_sum(data ** 2, lent) -
            #              _window_sum(data, lent) ** 2 / lent) * tnorm) ** 0.5
            # else:
            #      norm = (_window_sum(data ** 2, lent) * tnorm) ** 0.5
            # cc = cc / norm
            if demean:
                norm = _window_sum(data, lent) ** 2
                if norm.dtype == float:
                    norm /= lent
                else:
                    norm = norm / lent
                np.subtract(_window_sum(data ** 2, lent), norm, out=norm)
            else:
                norm = _window_sum(data ** 2, lent)
            norm *= tnorm
            if norm.dtype == float:
                np.sqrt(norm, out=norm)
            else:
                norm = np.sqrt(norm)
            mask = norm <= np.finfo(float).eps
            if cc.dtype == float:
                cc[~mask] /= norm[~mask]
            else:
                cc = cc / norm
            cc[mask] = 0
        else:
            msg = "normalize has to be one of (None, 'naive', 'full')"
            raise ValueError(msg)
    return cc


def xcorr(tr1, tr2, shift_len, full_xcorr=False):
    """
    Cross correlation of tr1 and tr2 in the time domain using window_len.

    .. note::
       Please use the :func:`~obspy.signal.cross_correlation.correlate`
       function for new code.

    ::

                                    Mid Sample
                                        |
        |AAAAAAAAAAAAAAA|AAAAAAAAAAAAAAA|AAAAAAAAAAAAAAA|AAAAAAAAAAAAAAA|
        |BBBBBBBBBBBBBBB|BBBBBBBBBBBBBBB|BBBBBBBBBBBBBBB|BBBBBBBBBBBBBBB|
        |<-shift_len/2->|   <- region of support ->     |<-shift_len/2->|


    :type tr1: :class:`~numpy.ndarray`, :class:`~obspy.core.trace.Trace`
    :param tr1: Trace 1
    :type tr2: :class:`~numpy.ndarray`, :class:`~obspy.core.trace.Trace`
    :param tr2: Trace 2 to correlate with trace 1
    :type shift_len: int
    :param shift_len: Total length of samples to shift for cross correlation.
    :type full_xcorr: bool
    :param full_xcorr: If ``True``, the complete xcorr function will be
        returned as :class:`~numpy.ndarray`
    :return: **index, value[, fct]** - Index of maximum xcorr value and the
        value itself. The complete xcorr function is returned only if
        ``full_xcorr=True``.

    .. note::
       As shift_len gets higher the window supporting the cross correlation
       actually gets smaller. So with shift_len=0 you get the correlation
       coefficient of both traces as a whole without any shift applied. As the
       xcorr function works in time domain and does not zero pad at all, with
       higher shifts allowed the window of support gets smaller so that the
       moving windows shifted against each other do not run out of the
       timeseries bounds at high time shifts. Of course there are other
       possibilities to do cross correlations e.g. in frequency domain.

    .. seealso::
       `ObsPy-users mailing list
       <http://lists.obspy.org/pipermail/obspy-users/2011-March/000056.html>`_
       and `issue #249 <https://github.com/obspy/obspy/issues/249>`_.
    """
    from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
    msg = ('Call to deprecated function xcorr(). Please use the correlate and '
           'xcorr_max functions.')
    warnings.warn(msg, ObsPyDeprecationWarning)

    # if we get Trace objects, use their data arrays
    for tr in [tr1, tr2]:
        if isinstance(tr, Trace):
            tr = tr.data

    # check if shift_len parameter is in an acceptable range.
    # if not the underlying c code tampers with shift_len and uses shift_len/2
    # instead. we want to avoid this silent automagic and raise an error in the
    # python layer right here.
    # see ticket #249 and src/xcorr.c lines 43-57
    if min(len(tr1), len(tr2)) - 2 * shift_len <= 0:
        msg = "shift_len too large. The underlying C code would silently " + \
              "use shift_len/2 which we want to avoid."
        raise ValueError(msg)
    # be nice and adapt type if necessary
    tr1 = np.ascontiguousarray(tr1, np.float32)
    tr2 = np.ascontiguousarray(tr2, np.float32)
    corp = np.empty(2 * shift_len + 1, dtype=np.float64, order='C')

    shift = C.c_int()
    coe_p = C.c_double()

    res = clibsignal.X_corr(tr1, tr2, corp, shift_len, len(tr1), len(tr2),
                            C.byref(shift), C.byref(coe_p))
    if res:
        raise MemoryError

    if full_xcorr:
        return shift.value, coe_p.value, corp
    else:
        return shift.value, coe_p.value


def xcorr_3c(st1, st2, shift_len, components=["Z", "N", "E"],
             full_xcorr=False, abs_max=True):
    """
    Calculates the cross correlation on each of the specified components
    separately, stacks them together and estimates the maximum and shift of
    maximum on the stack.

    Basically the same as :func:`~obspy.signal.cross_correlation.xcorr` but
    for (normally) three components, please also take a look at the
    documentation of that function. Useful e.g. for estimation of waveform
    similarity on a three component seismogram.

    :type st1: :class:`~obspy.core.stream.Stream`
    :param st1: Stream 1, containing one trace for Z, N, E component (other
        component_id codes are ignored)
    :type st2: :class:`~obspy.core.stream.Stream`
    :param st2: Stream 2, containing one trace for Z, N, E component (other
        component_id codes are ignored)
    :type shift_len: int
    :param shift_len: Total length of samples to shift for cross correlation.
    :type components: list of str
    :param components: List of components to use in cross-correlation, defaults
        to ``['Z', 'N', 'E']``.
    :type full_xcorr: bool
    :param full_xcorr: If ``True``, the complete xcorr function will be
        returned as :class:`~numpy.ndarray`.
    :param bool abs_max: *shift* will be calculated for maximum or
        absolute maximum.
    :return: **index, value[, fct]** - index of maximum xcorr value and the
        value itself. The complete xcorr function is returned only if
        ``full_xcorr=True``.
    """
    streams = [st1, st2]
    # check if we can actually use the provided streams safely
    for st in streams:
        if not isinstance(st, Stream):
            raise TypeError("Expected Stream object but got %s." % type(st))
        for component in components:
            if not len(st.select(component=component)) == 1:
                msg = "Expected exactly one %s trace in stream" % component + \
                      " but got %s." % len(st.select(component=component))
                raise ValueError(msg)
    ndat = len(streams[0].select(component=components[0])[0])
    if False in [len(st.select(component=component)[0]) == ndat
                 for st in streams for component in components]:
        raise ValueError("All traces have to be the same length.")
    # everything should be ok with the input data...
    corp = np.zeros(2 * shift_len + 1, dtype=np.float64, order='C')
    for component in components:
        xx = correlate(streams[0].select(component=component)[0],
                       streams[1].select(component=component)[0],
                       shift_len)
        corp += xx
    corp /= len(components)
    shift, value = xcorr_max(corp, abs_max=abs_max)
    if full_xcorr:
        return shift, value, corp
    else:
        return shift, value


def xcorr_max(fct, abs_max=True):
    """
    Return shift and value of the maximum of the cross-correlation function.

    :type fct: :class:`~numpy.ndarray`
    :param fct: Cross-correlation function e.g. returned by correlate.
    :param bool abs_max: Determines if the absolute maximum should be used.
    :return: **shift, value** - Shift and value of maximum of
        cross-correlation.

    .. rubric:: Example

    >>> fct = np.zeros(101)
    >>> fct[50] = -1.0
    >>> xcorr_max(fct)
    (0, -1.0)
    >>> fct[50], fct[60] = 0.0, 1.0
    >>> xcorr_max(fct)
    (10, 1.0)
    >>> fct[60], fct[40] = 0.0, -1.0
    >>> xcorr_max(fct)
    (-10, -1.0)
    >>> fct[60], fct[40] = 0.5, -1.0
    >>> xcorr_max(fct, abs_max=True)
    (-10, -1.0)
    >>> xcorr_max(fct, abs_max=False)
    (10, 0.5)
    >>> xcorr_max(fct[:-1], abs_max=False)
    (10.5, 0.5)
    """
    mid = (len(fct) - 1) / 2
    if len(fct) % 2 == 1:
        mid = int(mid)
    index = np.argmax(np.abs(fct) if abs_max else fct)
    # float() call is workaround for future package
    # see https://travis-ci.org/obspy/obspy/jobs/174284750
    return index - mid, float(fct[index])


def xcorr_pick_correction(pick1, trace1, pick2, trace2, t_before, t_after,
                          cc_maxlag, filter=None, filter_options={},
                          plot=False, filename=None):
    """
    Calculate the correction for the differential pick time determined by cross
    correlation of the waveforms in narrow windows around the pick times.
    For details on the fitting procedure refer to [Deichmann1992]_.

    The parameters depend on the epicentral distance and magnitude range. For
    small local earthquakes (Ml ~0-2, distance ~3-10 km) with consistent manual
    picks the following can be tried::

        t_before=0.05, t_after=0.2, cc_maxlag=0.10,
        filter="bandpass", filter_options={'freqmin': 1, 'freqmax': 20}

    The appropriate parameter sets can and should be determined/verified
    visually using the option `plot=True` on a representative set of picks.

    To get the corrected differential pick time calculate: ``((pick2 +
    pick2_corr) - pick1)``. To get a corrected differential travel time using
    origin times for both events calculate: ``((pick2 + pick2_corr - ot2) -
    (pick1 - ot1))``

    :type pick1: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param pick1: Time of pick for `trace1`.
    :type trace1: :class:`~obspy.core.trace.Trace`
    :param trace1: Waveform data for `pick1`. Add some time at front/back.
            The appropriate part of the trace is used automatically.
    :type pick2: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param pick2: Time of pick for `trace2`.
    :type trace2: :class:`~obspy.core.trace.Trace`
    :param trace2: Waveform data for `pick2`. Add some time at front/back.
            The appropriate part of the trace is used automatically.
    :type t_before: float
    :param t_before: Time to start cross correlation window before pick times
            in seconds.
    :type t_after: float
    :param t_after: Time to end cross correlation window after pick times in
            seconds.
    :type cc_maxlag: float
    :param cc_maxlag: Maximum lag/shift time tested during cross correlation in
        seconds.
    :type filter: str
    :param filter: `None` for no filtering or name of filter type
            as passed on to :meth:`~obspy.core.Trace.trace.filter` if filter
            should be used. To avoid artifacts in filtering provide
            sufficiently long time series for `trace1` and `trace2`.
    :type filter_options: dict
    :param filter_options: Filter options that get passed on to
            :meth:`~obspy.core.Trace.trace.filter` if filtering is used.
    :type plot: bool
    :param plot: If `True`, a plot window illustrating the alignment of the two
        traces at best cross correlation will be shown. This can and should be
        used to verify the used parameters before running automatedly on large
        data sets.
    :type filename: str
    :param filename: If plot option is selected, specifying a filename here
            (e.g. 'myplot.pdf' or 'myplot.png') will output the plot to a file
            instead of opening a plot window.
    :rtype: (float, float)
    :returns: Correction time `pick2_corr` for `pick2` pick time as a float and
            corresponding correlation coefficient.
    """
    # perform some checks on the traces
    if trace1.stats.sampling_rate != trace2.stats.sampling_rate:
        msg = "Sampling rates do not match: %s != %s" % \
            (trace1.stats.sampling_rate, trace2.stats.sampling_rate)
        raise Exception(msg)
    if trace1.id != trace2.id:
        msg = "Trace ids do not match: %s != %s" % (trace1.id, trace2.id)
        warnings.warn(msg)
    samp_rate = trace1.stats.sampling_rate
    # don't modify existing traces with filters
    if filter:
        trace1 = trace1.copy()
        trace2 = trace2.copy()
    # check data, apply filter and take correct slice of traces
    slices = []
    for _i, (t, tr) in enumerate(((pick1, trace1), (pick2, trace2))):
        start = t - t_before - (cc_maxlag / 2.0)
        end = t + t_after + (cc_maxlag / 2.0)
        duration = end - start
        # check if necessary time spans are present in data
        if tr.stats.starttime > start:
            msg = "Trace %s starts too late." % _i
            raise Exception(msg)
        if tr.stats.endtime < end:
            msg = "Trace %s ends too early." % _i
            raise Exception(msg)
        if filter and start - tr.stats.starttime < duration:
            msg = "Artifacts from signal processing possible. Trace " + \
                  "%s should have more additional data at the start." % _i
            warnings.warn(msg)
        if filter and tr.stats.endtime - end < duration:
            msg = "Artifacts from signal processing possible. Trace " + \
                  "%s should have more additional data at the end." % _i
            warnings.warn(msg)
        # apply signal processing and take correct slice of data
        if filter:
            tr.data = tr.data.astype(np.float64)
            tr.detrend(type='demean')
            tr.data *= cosine_taper(len(tr), 0.1)
            tr.filter(type=filter, **filter_options)
        slices.append(tr.slice(start, end))
    # cross correlate
    shift_len = int(cc_maxlag * samp_rate)
    cc = correlate(slices[0].data, slices[1].data, shift_len, method='direct')
    _cc_shift, cc_max = xcorr_max(cc)
    cc_curvature = np.concatenate((np.zeros(1), np.diff(cc, 2), np.zeros(1)))
    cc_convex = np.ma.masked_where(np.sign(cc_curvature) >= 0, cc)
    cc_concave = np.ma.masked_where(np.sign(cc_curvature) < 0, cc)
    # check results of cross correlation
    if cc_max < 0:
        msg = "Absolute maximum is negative: %.3f. " % cc_max + \
              "Using positive maximum: %.3f" % max(cc)
        warnings.warn(msg)
        cc_max = max(cc)
    if cc_max < 0.8:
        msg = "Maximum of cross correlation lower than 0.8: %s" % cc_max
        warnings.warn(msg)
    # make array with time shifts in seconds corresponding to cc function
    cc_t = np.linspace(-cc_maxlag, cc_maxlag, shift_len * 2 + 1)
    # take the subportion of the cross correlation around the maximum that is
    # convex and fit a parabola.
    # use vertex as subsample resolution best cc fit.
    peak_index = cc.argmax()
    first_sample = peak_index
    # XXX this could be improved..
    while first_sample > 0 and cc_curvature[first_sample - 1] <= 0:
        first_sample -= 1
    last_sample = peak_index
    while last_sample < len(cc) - 1 and cc_curvature[last_sample + 1] <= 0:
        last_sample += 1
    if first_sample == 0 or last_sample == len(cc) - 1:
        msg = "Fitting at maximum lag. Maximum lag time should be increased."
        warnings.warn(msg)
    # work on subarrays
    num_samples = last_sample - first_sample + 1
    if num_samples < 3:
        msg = "Less than 3 samples selected for fit to cross " + \
              "correlation: %s" % num_samples
        raise Exception(msg)
    if num_samples < 5:
        msg = "Less than 5 samples selected for fit to cross " + \
              "correlation: %s" % num_samples
        warnings.warn(msg)
    # quadratic fit for small subwindow
    coeffs, residual = scipy.polyfit(
        cc_t[first_sample:last_sample + 1],
        cc[first_sample:last_sample + 1], deg=2, full=True)[:2]
    # check results of fit
    if coeffs[0] >= 0:
        msg = "Fitted parabola opens upwards!"
        warnings.warn(msg)
    if residual > 0.1:
        msg = "Residual in quadratic fit to cross correlation maximum " + \
              "larger than 0.1: %s" % residual
        warnings.warn(msg)
    # X coordinate of vertex of parabola gives time shift to correct
    # differential pick time. Y coordinate gives maximum correlation
    # coefficient.
    dt = -coeffs[1] / 2.0 / coeffs[0]
    coeff = (4 * coeffs[0] * coeffs[2] - coeffs[1] ** 2) / (4 * coeffs[0])
    # this is the shift to apply on the time axis of `trace2` to align the
    # traces. Actually we do not want to shift the trace to align it but we
    # want to correct the time of `pick2` so that the traces align without
    # shifting. This is the negative of the cross correlation shift.
    dt = -dt
    pick2_corr = dt
    # plot the results if selected
    if plot is True:
        with MatplotlibBackend(filename and "AGG" or None, sloppy=True):
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            tmp_t = np.linspace(0, len(slices[0]) / samp_rate, len(slices[0]))
            ax1.plot(tmp_t, slices[0].data / float(slices[0].data.max()), "k",
                     label="Trace 1")
            ax1.plot(tmp_t, slices[1].data / float(slices[1].data.max()), "r",
                     label="Trace 2")
            ax1.plot(tmp_t - dt, slices[1].data / float(slices[1].data.max()),
                     "g", label="Trace 2 (shifted)")
            ax1.legend(loc="lower right", prop={'size': "small"})
            ax1.set_title("%s" % slices[0].id)
            ax1.set_xlabel("time [s]")
            ax1.set_ylabel("norm. amplitude")
            ax2 = fig.add_subplot(212)
            ax2.plot(cc_t, cc_convex, ls="", marker=".", color="k",
                     label="xcorr (convex)")
            ax2.plot(cc_t, cc_concave, ls="", marker=".", color="0.7",
                     label="xcorr (concave)")
            ax2.plot(cc_t[first_sample:last_sample + 1],
                     cc[first_sample:last_sample + 1], "b.",
                     label="used for fitting")
            tmp_t = np.linspace(cc_t[first_sample], cc_t[last_sample],
                                num_samples * 10)
            ax2.plot(tmp_t, scipy.polyval(coeffs, tmp_t), "b", label="fit")
            ax2.axvline(-dt, color="g", label="vertex")
            ax2.axhline(coeff, color="g")
            ax2.set_xlabel("%.2f at %.3f seconds correction" % (coeff, -dt))
            ax2.set_ylabel("correlation coefficient")
            ax2.set_ylim(-1, 1)
            ax2.set_xlim(cc_t[0], cc_t[-1])
            ax2.legend(loc="lower right", prop={'size': "x-small"})
            # plt.legend(loc="lower left")
            if filename:
                fig.savefig(filename)
            else:
                plt.show()

    return (pick2_corr, coeff)


def templates_max_similarity(st, time, streams_templates):
    """
    Compares all event templates in the streams_templates list of streams
    against the given stream around the time of the suspected event. The stream
    that is being checked has to include all trace ids that are included in
    template events. One component streams can be checked as well as multiple
    components simultaneously. In case of multiple components it is made sure,
    that all three components are shifted together.  The traces in any stream
    need to have a reasonable common starting time.  The stream to check should
    have some additional data to left/right of suspected event, the event
    template streams should be cut to the portion of the event that should be
    compared. Also see :func:`obspy.signal.trigger.coincidence_trigger` and the
    corresponding example in the
    `Trigger/Picker Tutorial
    <https://tutorial.obspy.org/code_snippets/trigger_tutorial.html>`_.

    - computes cross correlation on each component (one stream serves as
      template, one as a longer search stream)
    - stack all three and determine best shift in stack
    - normalization is a bit problematic so compute the correlation coefficient
      afterwards for the best shift to make sure the result is between 0 and 1.

    >>> from obspy import read, UTCDateTime
    >>> import numpy as np
    >>> np.random.seed(123)  # make test reproducible
    >>> st = read()
    >>> t = UTCDateTime(2009, 8, 24, 0, 20, 7, 700000)
    >>> templ = st.copy().slice(t, t+5)
    >>> for tr in templ:
    ...     tr.data += np.random.random(len(tr)) * tr.data.max() * 0.5
    >>> print(templates_max_similarity(st, t, [templ]))
    0.922536411468

    :param time: Time around which is checked for a similarity. Cross
        correlation shifts of around template event length are checked.
    :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param st: One or multi-component Stream to check against event templates.
        Should have additional data left/right of suspected event (around half
        the length of template events).
    :type st: :class:`~obspy.core.stream.Stream`
    :param streams_templates: List of streams with template events to check for
        waveform similarity. Each template has to include data for all
        channels present in stream to check.
    :type streams_templates: list of :class:`~obspy.core.stream.Stream`
    :returns: Best correlation coefficient obtained by the comparison against
        all template events (0 to 1).
    """
    values = []
    for st_tmpl in streams_templates:
        ids = [tr.id for tr in st_tmpl]
        duration = st_tmpl[0].stats.endtime - st_tmpl[0].stats.starttime
        st_ = st.slice(time - (duration * 0.5),
                       time + (duration * 1.5))
        cc = None
        for id_ in reversed(ids):
            if not st_.select(id=id_):
                msg = "Skipping trace %s in template correlation " + \
                      "(not present in stream to check)."
                warnings.warn(msg % id_)
                ids.remove(id_)
        if not ids:
            msg = ("Skipping template(s) for station '{}': No common SEED IDs "
                   "when comparing template ({}) and data streams ({}).")
            warnings.warn(msg.format(
                st_tmpl[0].stats.station,
                ', '.join(sorted(set(tr.id for tr in st_tmpl))),
                ', '.join(sorted(set(tr.id for tr in st_)))))
            continue
        # determine best (combined) shift of multi-component data
        for id_ in ids:
            tr1 = st_.select(id=id_)[0]
            tr2 = st_tmpl.select(id=id_)[0]
            if len(tr1) > len(tr2):
                data_short = tr2.data
                data_long = tr1.data
            else:
                data_short = tr1.data
                data_long = tr2.data
            data_short = (data_short - data_short.mean()) / data_short.std()
            data_long = (data_long - data_long.mean()) / data_long.std()
            tmp = np.correlate(data_long, data_short, native_str("valid"))
            try:
                cc += tmp
            except TypeError:
                cc = tmp
            except ValueError:
                cc = None
                break
        if cc is None:
            msg = "Skipping template(s) for station %s due to problems in " + \
                  "three component correlation (gappy traces?)"
            warnings.warn(msg % st_tmpl[0].stats.station)
            continue
        ind = cc.argmax()
        ind2 = ind + len(data_short)
        coef = 0.0
        # determine correlation coefficient of best shift as the mean of all
        # components
        for id_ in ids:
            tr1 = st_.select(id=id_)[0]
            tr2 = st_tmpl.select(id=id_)[0]
            if len(tr1) > len(tr2):
                data_short = tr2.data
                data_long = tr1.data
            else:
                data_short = tr1.data
                data_long = tr2.data
            coef += np.corrcoef(data_short, data_long[ind:ind2])[0, 1]
        coef /= len(ids)
        values.append(coef)
    if values:
        return max(values)
    else:
        return 0


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
