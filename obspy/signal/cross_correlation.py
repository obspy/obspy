#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: cross_correlation.py
#   Author: Moritz Beyreuther, Tobias Megies
#    Email: megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2012 Moritz Beyreuther, Tobias Megies
# ------------------------------------------------------------------
"""
Signal processing routines based on cross correlation techniques.

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

import warnings
import numpy as np
import ctypes as C
import scipy
from obspy import Trace, Stream
from obspy.signal.headers import clibsignal
from obspy.signal import cosTaper


def xcorr(tr1, tr2, shift_len, full_xcorr=False):
    """
    Cross correlation of tr1 and tr2 in the time domain using window_len.

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
       and
       `issue #249 <https://github.com/obspy/obspy/issues/249>`_.

    .. rubric:: Example

    >>> tr1 = np.random.randn(10000).astype(np.float32)
    >>> tr2 = tr1.copy()
    >>> a, b = xcorr(tr1, tr2, 1000)
    >>> a
    0
    >>> round(b, 7)
    1.0
    """
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


def xcorr_3C(st1, st2, shift_len, components=["Z", "N", "E"],
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
        xx = xcorr(streams[0].select(component=component)[0],
                   streams[1].select(component=component)[0],
                   shift_len, full_xcorr=True)
        corp += xx[2]

    corp /= len(components)

    shift, value = xcorr_max(corp, abs_max=abs_max)

    if full_xcorr:
        return shift, value, corp
    else:
        return shift, value


def xcorr_max(fct, abs_max=True):
    """
    Return shift and value of maximum xcorr function

    :type fct: :class:`~numpy.ndarray`
    :param fct: xcorr function e.g. returned by xcorr
    :type abs_max: bool
    :param abs_max: determines if the absolute maximum should be used.
    :return: **shift, value** - Shift and value of maximum xcorr.

    .. rubric:: Example

    >>> fct = np.zeros(101)
    >>> fct[50] = -1.0
    >>> xcorr_max(fct)
    (0.0, -1.0)
    >>> fct[50], fct[60] = 0.0, 1.0
    >>> xcorr_max(fct)
    (10.0, 1.0)
    >>> fct[60], fct[40] = 0.0, -1.0
    >>> xcorr_max(fct)
    (-10.0, -1.0)
    >>> fct[60], fct[40] = 0.5, -1.0
    >>> xcorr_max(fct, abs_max=True)
    (-10.0, -1.0)
    >>> xcorr_max(fct, abs_max=False)
    (10.0, 0.5)
    """
    value = fct.max()
    if abs_max:
        _min = fct.min()
        if abs(_min) > abs(value):
            value = _min

    mid = (len(fct) - 1) / 2
    shift = np.where(fct == value)[0][0] - mid
    return float(shift), float(value)


def xcorrPickCorrection(pick1, trace1, pick2, trace2, t_before, t_after,
                        cc_maxlag, filter=None, filter_options={}, plot=False,
                        filename=None):
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
    visually using the option `show=True` on a representative set of picks.

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
    :param cc_maxlag: Maximum lag time tested during cross correlation in
            seconds.
    :type filter: str
    :param filter: None for no filtering or name of filter type
            as passed on to :meth:`~obspy.core.Trace.trace.filter` if filter
            should be used. To avoid artifacts in filtering provide
            sufficiently long time series for `trace1` and `trace2`.
    :type filter_options: dict
    :param filter_options: Filter options that get passed on to
            :meth:`~obspy.core.Trace.trace.filter` if filtering is used.
    :type plot: bool
    :param plot: Determines if pick is refined automatically (default, ""),
            if an informative matplotlib plot is shown ("plot"), or if an
            interactively changeable PyQt Window is opened ("interactive").
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
            tr.data *= cosTaper(len(tr), 0.1)
            tr.filter(type=filter, **filter_options)
        slices.append(tr.slice(start, end))
    # cross correlate
    shift_len = int(cc_maxlag * samp_rate)
    _cc_shift, cc_max, cc = xcorr(slices[0].data, slices[1].data,
                                  shift_len, full_xcorr=True)
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
        import matplotlib
        if filename:
            matplotlib.use('agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        tmp_t = np.linspace(0, len(slices[0]) / samp_rate, len(slices[0]))
        ax1.plot(tmp_t, slices[0].data / float(slices[0].data.max()), "k",
                 label="Trace 1")
        ax1.plot(tmp_t, slices[1].data / float(slices[1].data.max()), "r",
                 label="Trace 2")
        ax1.plot(tmp_t - dt, slices[1].data / float(slices[1].data.max()), "g",
                 label="Trace 2 (shifted)")
        ax1.legend(loc="lower right", prop={'size': "small"})
        ax1.set_title("%s" % slices[0].id)
        ax1.set_xlabel("time [s]")
        ax1.set_ylabel("norm. amplitude")
        ax2 = fig.add_subplot(212)
        ax2.plot(cc_t, cc_convex, ls="", marker=".", c="k",
                 label="xcorr (convex)")
        ax2.plot(cc_t, cc_concave, ls="", marker=".", c="0.7",
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
        ax2.legend(loc="lower right", prop={'size': "x-small"})
        # plt.legend(loc="lower left")
        if filename:
            fig.savefig(fname=filename)
        else:
            plt.show()

    return (pick2_corr, coeff)


def templatesMaxSimilarity(st, time, streams_templates):
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
    compared. Also see :func:`obspy.signal.trigger.coincidenceTrigger` and the
    corresponding example in the
    `Trigger/Picker Tutorial
    <http://tutorial.obspy.org/code_snippets/trigger_tutorial.html>`_.

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
    >>> print(templatesMaxSimilarity(st, t, [templ]))
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
            break
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
