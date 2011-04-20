#!/usr/bin/env python
#--------------------------------------------------------------------
# Filename: cross_correlation.py
#   Author: Moritz Beyreuther, Tobias Megies
#    Email: megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2010 Moritz Beyreuther, Tobias Megies
#-------------------------------------------------------------------
"""
Signal processing routines based on cross correlation techniques.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import ctypes as C
import numpy as np

from obspy.core import Trace, Stream
from obspy.signal.util import clibsignal


def xcorr(tr1, tr2, shift_len, full_xcorr=False):
    """
    Cross correlation of tr1 and tr2 in the time domain using window_len.
    Supports ndarrays (float32) and Trace objects.
    
    >>> tr1 = np.random.randn(10000).astype('float32')
    >>> tr2 = tr1.copy()
    >>> a, b = xcorr(tr1, tr2, 1000)
    >>> a
    0
    >>> round(b, 7)
    1.0

    ::

                                    Mid Sample
                                        |                           
        |AAAAAAAAAAAAAAA|AAAAAAAAAAAAAAA|AAAAAAAAAAAAAAA|AAAAAAAAAAAAAAA|
        |BBBBBBBBBBBBBBB|BBBBBBBBBBBBBBB|BBBBBBBBBBBBBBB|BBBBBBBBBBBBBBB|
        |<-shift_len/2->|   <- region of support ->     |<-shift_len/2->|

    
    :type tr1: numpy ndarray float32 or obspy.core.Trace
    :param tr1: Trace 1
    :type tr2: numpy ndarray float32 or obspy.core.Trace
    :param tr2: Trace 2 to correlate with trace 1
    :type shift_len: Int
    :param shift_len: Total length of samples to shift for cross correlation.
    :type full_xcorr: Bool
    :param full_xcorr: If True, the complete xcorr function will be
        returned as numpy.ndarray
    :return: (index, value, fct) index of maximum xcorr value and the value
        itself. The complete xcorr function is returned only if
        ``full_xcorr=True``.

    :note: As shift_len gets higher the window supporting the cross
           correlation actually gets smaller. So with shift_len=0 you get
           the correlation coefficient of both traces as a whole without
           any shift applied. As the xcorr function works in time domain
           and does not zero pad at all, with higher shifts allowed the
           window of support gets smaller so that the moving windows
           shifted against each other do not run out of the timeseries
           bounds at high time shifts. Of course there are other
           possibilities to do cross correlations e.g. in frequency
           domain.
           Also see exchange in `ObsPy-users mailing list`_ and in this
           `ticket`_.

    .. _`ObsPy-users mailing list`: http://lists.obspy.org/pipermail/obspy-users/2011-March/000056.html
    .. _`ticket`: https://obspy.org/ticket/249
    """
    # if we get Trace objects, use their data arrays
    for tr in [tr1, tr2]:
        if isinstance(tr, Trace):
            tr = tr.data

    # 2009-10-11 Moritz
    clibsignal.X_corr.argtypes = [
        np.ctypeslib.ndpointer(dtype='float32', ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype='float32', ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype='float64', ndim=1, flags='C_CONTIGUOUS'),
        C.c_int, C.c_int, C.c_int,
        C.POINTER(C.c_int), C.POINTER(C.c_double)]
    clibsignal.X_corr.restype = C.c_void_p

    # be nice and adapt type if necessary
    tr1 = np.require(tr1, 'float32', ['C_CONTIGUOUS'])
    tr2 = np.require(tr2, 'float32', ['C_CONTIGUOUS'])
    corp = np.empty(2 * shift_len + 1, dtype='float64', order='C')

    shift = C.c_int()
    coe_p = C.c_double()

    clibsignal.X_corr(tr1, tr2, corp, shift_len, len(tr1), len(tr2),
                      C.byref(shift), C.byref(coe_p))

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
    Basically the same as :func:`~obspy.signal.util.xcorr` but for (normally)
    three components, please also take a look at the documentation of that
    function. Useful e.g. for estimation of waveform similarity on a three
    component seismogram.
    
    :type st1: obspy.core.Stream
    :param st1: Stream 1, containing one trace for Z, N, E component (other
            component_id codes are ignored)
    :type st2: obspy.core.Stream
    :param st2: Stream 2, containing one trace for Z, N, E component (other
            component_id codes are ignored)
    :type shift_len: Int
    :param shift_len: Total length of samples to shift for cross correlation.
    :type components: List of Strings
    :param components: List of components to use in cross-correlation, defaults
            to ["Z", "N", "E"].
    :type full_xcorr: Bool
    :param full_xcorr: If True, the complete xcorr function will be
        returned as numpy.ndarray
    :return: (index, value, fct) index of maximum xcorr value and the value
        itself. The complete xcorr function is returned only if
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
    if False in [len(st.select(component=component)[0]) == ndat \
                 for st in streams for component in components]:
            raise ValueError("All traces have to be the same length.")
    # everything should be ok with the input data...

    corp = np.zeros(2 * shift_len + 1, dtype='float64', order='C')

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
    
    :type fct: numpy.ndarray
    :param fct: xcorr function e.g. returned bei xcorr
    :type abs_max: boolean
    :param abs_max: determines if the absolute maximum should be used.
    :return: (shift, value) Shift and value of maximum xcorr
    """
    value = fct.max()
    if abs_max:
        _min = fct.min()
        if abs(_min) > abs(value):
            value = _min

    mid = (len(fct) - 1) / 2
    shift = np.where(fct == value)[0][0] - mid
    return float(shift), float(value)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
