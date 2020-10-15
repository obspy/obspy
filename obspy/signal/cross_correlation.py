#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: cross_correlation.py
#   Author: Moritz Beyreuther, Tobias Megies, Tom Eulenfeld
#    Email: megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2019 Moritz Beyreuther, Tobias Megies, Tom Eulenfeld
# ------------------------------------------------------------------
"""
Signal processing routines based on cross correlation techniques.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from bisect import bisect_left
from copy import copy
from distutils.version import LooseVersion
import warnings

import numpy as np
import scipy

from obspy import Stream, Trace
from obspy.core.util.misc import MatplotlibBackend
from obspy.signal.invsim import cosine_taper


def _pad_zeros(a, num, num2=None):
    """Pad num zeros at both sides of array a"""
    if num2 is None:
        num2 = num
    hstack = [np.zeros(num, dtype=a.dtype), a, np.zeros(num2, dtype=a.dtype)]
    return np.hstack(hstack)


def _call_scipy_correlate(a, b, mode, method):
    """
    Call the correct correlate function depending on Scipy version and method.
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
    Cross-correlation using SciPy with mode='valid' and precedent zero padding.
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
    Cross-correlation using SciPy with mode='full' and subsequent slicing.
    """
    mid = (len(a) + len(b) - 1) // 2
    if shift is None:
        shift = mid
    if shift > mid:
        # Such a large shift is not possible without zero padding
        return _xcorr_padzeros(a, b, shift, method)
    cc = _call_scipy_correlate(a, b, 'full', method)
    return cc[mid - shift:mid + shift + len(cc) % 2]


def correlate(a, b, shift, demean=True, normalize='naive', method='auto'):
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
    """Rolling sum of data."""
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
    coeffs, residual = np.polyfit(
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
            ax2.plot(tmp_t, np.polyval(coeffs, tmp_t), "b", label="fit")
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
            tmp = np.correlate(data_long, data_short, "valid")
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


def _prep_streams_correlate(stream, template, template_time=None):
    """
    Prepare stream and template for cross-correlation.

    Select traces in stream and template with the same seed id and trim
    stream to correct start and end times.
    """
    if len({tr.stats.sampling_rate for tr in stream + template}) > 1:
        raise ValueError('Traces have different sampling rate')
    ids = {tr.id for tr in stream} & {tr.id for tr in template}
    if len(ids) == 0:
        raise ValueError('No traces with matching ids in template and stream')
    stream = copy(stream)
    template = copy(template)
    stream.traces = [tr for tr in stream if tr.id in ids]
    template.traces = [tr for tr in template if tr.id in ids]
    template.sort()
    stream.sort()
    if len(stream) != len(template):
        msg = ('Length of prepared template stream and data stream are '
               'different. Make sure the data does not contain gaps.')
        raise ValueError(msg)
    starttime = max(tr.stats.starttime for tr in stream)
    endtime = min(tr.stats.endtime for tr in stream)
    starttime_template = min(tr.stats.starttime for tr in template)
    len_templ = max(tr.stats.endtime - tr.stats.starttime for tr in template)
    if template_time is None:
        template_offset = 0
    else:
        template_offset = template_time - starttime_template
    # trim traces
    trim1 = [trt.stats.starttime - starttime_template for trt in template]
    trim2 = [trt.stats.endtime - starttime_template - len_templ
             for trt in template]
    trim1 = [t - min(trim1) for t in trim1]
    trim2 = [t - max(trim2) for t in trim2]
    for i, tr in enumerate(stream):
        tr = tr.slice(starttime + trim1[i], endtime + trim2[i])
        tr.stats.starttime = starttime + template_offset
        stream.traces[i] = tr
    return stream, template


def _correlate_prepared_stream_template(stream, template, **kwargs):
    """
    Calculate cross-correlation of traces in stream with traces in template.

    Operates on prepared streams.
    """
    for tr, trt in zip(stream, template):
        tr.data = correlate_template(tr, trt, mode='valid', **kwargs)
    # make sure xcorrs have the same length, can differ by one sample
    lens = {len(tr) for tr in stream}
    if len(lens) > 1:
        warnings.warn('Samples of traces are slightly misaligned. '
                      'Use Stream.interpolate if this is not intended.')
        if max(lens) - min(lens) > 1:
            msg = 'This should not happen. Please contact the developers.'
            raise RuntimeError(msg)
        for tr in stream:
            tr.data = tr.data[:min(lens)]
    return stream


def correlate_stream_template(stream, template, template_time=None, **kwargs):
    """
    Calculate cross-correlation of traces in stream with traces in template.

    Only matching seed ids are correlated, other traces are silently discarded.
    The template stream and data stream might have traces of different
    length and different start times.
    The data stream must not have gaps and will be sliced as necessary.

    :param stream: Stream with data traces.
    :param template: Stream with template traces (should be shorter than data).
    :param template_time: UTCDateTime associated with template event
        (e.g. origin time, default is the start time of the template stream).
        The start times of the returned Stream will be shifted by the given
        template time minus the template start time.
    :param kwargs: kwargs are passed to
        :func:`~obspy.signal.cross_correlation.correlate_template` function.

    :return: Stream with cross-correlations.

    .. note::

        Use :func:`~obspy.signal.cross_correlation.correlation_detector`
        for detecting events based on their similarity.
        The returned stream of cross-correlations is suitable for
        use with :func:`~obspy.signal.trigger.coincidence_trigger`, though.

    .. rubric:: Example

    >>> from obspy import read, UTCDateTime
    >>> data = read().filter('highpass', freq=5)
    >>> pick = UTCDateTime('2009-08-24T00:20:07.73')
    >>> template = data.slice(pick, pick + 10)
    >>> ccs = correlate_stream_template(data, template)
    >>> print(ccs)  # doctest: +ELLIPSIS
    3 Trace(s) in Stream:
    BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 2000 samples
    BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 2000 samples
    BW.RJOB..EHZ | ... - 2009-08-24T00:20:22.990000Z | 100.0 Hz, 2000 samples
    """
    stream, template = _prep_streams_correlate(stream, template,
                                               template_time=template_time)
    return _correlate_prepared_stream_template(stream, template, **kwargs)


def _calc_mean(stream):
    """
    Return trace with mean of traces in stream.
    """
    if len(stream) == 0:
        return stream
    matrix = np.array([tr.data for tr in stream])
    header = dict(sampling_rate=stream[0].stats.sampling_rate,
                  starttime=stream[0].stats.starttime)
    return Trace(data=np.mean(matrix, axis=0), header=header)


def _find_peaks(data, height, holdon_samples, holdoff_samples):
    """
    Peak finding function used for Scipy versions smaller than 1.1.
    """
    cond = data >= height
    # loop through True values in cond array and guarantee hold time
    similarity_cond = data[cond]
    cindices = np.nonzero(cond)[0]
    detections_index = []
    i = 0
    while True:
        try:
            cindex = cindices[i]
        except IndexError:
            break
        # look for maximum inside holdon time
        j = bisect_left(cindices, cindex + holdon_samples, lo=i)
        k = i + np.argmax(similarity_cond[i:j])
        cindex = cindices[k]
        detections_index.append(cindex)
        # wait holdoff time after detection
        i = bisect_left(cindices, cindex + holdoff_samples, lo=j)
    return detections_index


def _similarity_detector(similarity, height, distance,
                         details=False, cross_correlations=None, **kwargs):
    """
    Detector based on the similarity of waveforms.
    """
    starttime = similarity.stats.starttime
    dt = similarity.stats.delta
    if distance is not None:
        distance = int(round(distance / dt))
    try:
        from scipy.signal import find_peaks
    except ImportError:
        indices = _find_peaks(similarity.data, height, distance, distance)
        properties = {}
    else:
        indices, properties = find_peaks(similarity.data, height,
                                         distance=distance, **kwargs)
    detections = []
    for i, index in enumerate(indices):
        detection = {'time': starttime + index * dt,
                     'similarity': similarity.data[index]}
        if details and cross_correlations is not None:
            detection['cc_values'] = {tr.id: tr.data[index]
                                      for tr in cross_correlations}
        if details:
            for k, v in properties.items():
                if k != 'peak_heights':
                    detection[k[:-1] if k.endswith('s') else k] = v[i]
        detections.append(detection)
    return detections


def _insert_amplitude_ratio(detections, stream, template, template_time=None,
                            template_magnitude=None):
    """
    Insert amplitude ratio and magnitude into detections.
    """
    stream, template = _prep_streams_correlate(stream, template,
                                               template_time=template_time)
    ref_amp = np.mean([np.mean(np.abs(tr.data)) for tr in template])
    for detection in detections:
        t = detection['time']
        ratio = np.mean([np.mean(np.abs(tr.slice(t).data[:len(trt)]))
                         for tr, trt in zip(stream, template)]) / ref_amp
        detection['amplitude_ratio'] = ratio
        if template_magnitude is not None:
            magdiff = 4 / 3 * np.log10(ratio)
            detection['magnitude'] = template_magnitude + magdiff
    return detections


def _get_item(list_, index):
    if isinstance(list_, str):
        return list_
    try:
        return list_[index]
    except TypeError:
        return list_


def _plot_detections(detections, similarities, stream=None, heights=None,
                     template_names=None):
    """
    Plot detections together with similarity traces and data stream.
    """
    import matplotlib.pyplot as plt
    from obspy.imaging.util import _set_xaxis_obspy_dates
    if stream in (True, None):
        stream = []
    akw = dict(xy=(0.02, 0.95), xycoords='axes fraction', va='top')
    num1 = len(stream)
    num2 = len(similarities)
    fig, ax = plt.subplots(num1 + num2, 1, sharex=True)
    if num1 + num2 == 1:
        ax = [ax]
    for detection in detections:
        tid = detection.get('template_id', 0)
        color = 'C{}'.format((tid + 1) % 10)
        for i in list(range(num1)) + [num1 + tid]:
            ax[i].axvline(detection['time'].matplotlib_date, color=color)
    for i, tr in enumerate(stream):
        ax[i].plot(tr.times('matplotlib'), tr.data, 'k')
        ax[i].annotate(tr.id, **akw)
    for i, tr in enumerate(similarities):
        if tr is not None:
            ax[num1 + i].plot(tr.times('matplotlib'), tr.data, 'k')
            height = _get_item(heights, i)
            if isinstance(height, (float, int)):
                ax[num1 + i].axhline(height)
        template_name = _get_item(template_names, i)
        text = ('similarity {}'.format(template_name) if template_name else
                'similarity' if num2 == 1 else
                'similarity template {}'.format(i))
        ax[num1 + i].annotate(text, **akw)
    try:
        _set_xaxis_obspy_dates(ax[-1])
    except ValueError:
        # work-around for python 2.7, minimum dependencies, see
        # https://travis-ci.org/obspy/obspy/jobs/508313177
        # can be safely removed later
        pass
    plt.show()


def correlation_detector(stream, templates, heights, distance,
                         template_times=None, template_magnitudes=None,
                         template_names=None,
                         similarity_func=_calc_mean, details=None,
                         plot=None, **kwargs):
    """
    Detector based on the cross-correlation of waveforms.

    This detector cross-correlates the stream with each of the template
    streams (compare with
    :func:`~obspy.signal.cross_correlation.correlate_stream_template`).
    A similarity is defined, by default it is the mean of all
    cross-correlation functions for each template.
    If the similarity exceeds the `height` threshold a detection is triggered.
    This peak finding utilizes the SciPy function
    :func:`~scipy.signal.find_peaks` with parameters `height` and `distance`.
    For a SciPy version smaller than 1.1 it uses a custom function
    for peak finding.

    :param stream: Stream with data traces.
    :param templates: List of streams with template traces.
        Each template stream should be shorter than the data stream.
        This argument can also be a single template stream.
    :param heights: Similarity values to trigger a detection,
        one for each template. This argument can also be a single value.
    :param distance: The distance in seconds between two detections.
    :param template_times: UTCDateTimes associated with template event
        (e.g. origin times,
        default are the start times of the template streams).
        This argument can also be a single value.
    :param template_magnitudes: Magnitudes of the template events.
        If provided, amplitude ratios between templates and detections will
        be calculated and the magnitude of detections will be estimated.
        This argument can also be a single value.
        This argument can be set to `True`,
        then only amplitude ratios will be calculated.
    :param template_names: List of template names, the corresponding
        template name will be inserted into the detection.
    :param similarity_func: By default, the similarity will be calculated by
        the mean of cross-correlations. If provided, `similarity_func` will be
        called with the stream of cross correlations and the returned trace
        will be used as similarity. See the tutorial for an example.
    :param details: If set to True detections include detailed information.
    :param plot: Plot detections together with the data of the
        supplied stream. The default `plot=None` does not plot
        anything. `plot=True` plots the similarity traces together
        with the detections. If a stream is passed as argument, the traces
        in the stream will be plotted together with the similarity traces and
        detections.
    :param kwargs: Suitable kwargs are passed to
        :func:`~obspy.signal.cross_correlation.correlate_template` function.
        All other kwargs are passed to :func:`~scipy.signal.find_peaks`.

    :return: List of event detections sorted chronologically and
        list of similarity traces - one for each template.
        Each detection is a dictionary with the following keys:
        time, similarity, template_id,
        amplitude_ratio, magnitude (if template_magnitudes is provided),
        template_name (if template_names is provided),
        cross-correlation values, properties returned by find_peaks
        (if details are requested)

    .. rubric:: Example

    >>> from obspy import read, UTCDateTime
    >>> data = read().filter('highpass', freq=5)
    >>> pick = UTCDateTime('2009-08-24T00:20:07.73')
    >>> template = data.slice(pick, pick + 10)
    >>> detections, sims = correlation_detector(data, template, 0.5, 10)
    >>> print(detections)   # doctest: +SKIP
    [{'time': UTCDateTime(2009, 8, 24, 0, 20, 7, 730000),
      'similarity': 0.99999999999999944,
      'template_id': 0}]

    A more advanced :ref:`tutorial <correlation-detector-tutorial>`
    is available.
    """
    if isinstance(templates, Stream):
        templates = [templates]
    cckeys = ('normalize', 'demean', 'method')
    cckwargs = {k: v for k, v in kwargs.items() if k in cckeys}
    pfkwargs = {k: v for k, v in kwargs.items() if k not in cckeys}
    possible_detections = []
    similarities = []
    for template_id, template in enumerate(templates):
        template_time = _get_item(template_times, template_id)
        try:
            ccs = correlate_stream_template(stream, template,
                                            template_time=template_time,
                                            **cckwargs)
        except ValueError as ex:
            msg = '{} -> do not use template {}'.format(ex, template_id)
            warnings.warn(msg)
            similarities.append(None)
            continue
        similarity = similarity_func(ccs)
        height = _get_item(heights, template_id)
        detections_template = _similarity_detector(
            similarity, height, distance, details=details,
            cross_correlations=ccs, **pfkwargs)
        for d in detections_template:
            template_name = _get_item(template_names, template_id)
            if template_name is not None:
                d['template_name'] = template_name
            d['template_id'] = template_id
        if template_magnitudes is True:
            template_magnitude = None
        else:
            template_magnitude = _get_item(template_magnitudes, template_id)
        if template_magnitudes is not None:
            _insert_amplitude_ratio(detections_template, stream, template,
                                    template_time=template_time,
                                    template_magnitude=template_magnitude)
        possible_detections.extend(detections_template)
        similarities.append(similarity)
    # discard detections with small distance, prefer those with high
    # similarity
    if len(templates) == 1:
        detections = possible_detections
    else:
        detections = []
        times = []
        for pd in sorted(possible_detections, key=lambda d: -d['similarity']):
            if all(abs(pd['time'] - t) > distance for t in times):
                times.append(pd['time'])
                detections.append(pd)
        detections = sorted(detections, key=lambda d: d['time'])
    if plot is not None:
        _plot_detections(detections, similarities, stream=plot,
                         heights=heights, template_names=template_names)
    return detections, similarities


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
