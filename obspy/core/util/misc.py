# -*- coding: utf-8 -*-
"""
Various additional utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import warnings
import itertools
import numpy as np


# The following dictionary maps the first character of the channel_id to the
# lowest sampling rate this so called Band Code should be used for according
# to: SEED MANUAL p.124
# We use this e.g. in seihub.client.getWaveform to request two samples more on
# both start and end to cut to the samples that really are nearest to requested
# start/endtime afterwards.
BAND_CODE = {'F': 1000.0,
             'G': 1000.0,
             'D': 250.0,
             'C': 250.0,
             'E': 80.0,
             'S': 10.0,
             'H': 80.0,
             'B': 10.0,
             'M': 1.0,
             'L': 1.0,
             'V': 0.1,
             'U': 0.01,
             'R': 0.0001,
             'P': 0.000001,
             'T': 0.0000001,
             'Q': 0.00000001}


def guessDelta(channel):
    """
    Estimate time delta in seconds between each sample from given channel name.

    :type channel: str
    :param channel: Channel name, e.g. ``'BHZ'`` or ``'H'``
    :rtype: float
    :return: Returns ``0`` if band code is not given or unknown.

    .. rubric:: Example

    >>> print(guessDelta('BHZ'))
    0.1

    >>> print(guessDelta('H'))
    0.0125

    >>> print(guessDelta('XZY'))  # doctest: +SKIP
    0
    """
    try:
        return 1. / BAND_CODE[channel[0]]
    except:
        msg = "No or unknown channel id provided. Specifying a channel id " + \
              "could lead to better selection of first/last samples of " + \
              "fetched traces."
        warnings.warn(msg)
    return 0


def scoreatpercentile(a, per, limit=(), issorted=True):
    """
    Calculates the score at the given per percentile of the sequence a.

    For example, the score at ``per=50`` is the median.

    If the desired quantile lies between two data points, we interpolate
    between them.

    If the parameter ``limit`` is provided, it should be a tuple (lower,
    upper) of two values.  Values of ``a`` outside this (closed) interval
    will be ignored.

    .. rubric:: Examples

    >>> a = [1, 2, 3, 4]
    >>> scoreatpercentile(a, 25)
    1.75
    >>> scoreatpercentile(a, 50)
    2.5
    >>> scoreatpercentile(a, 75)
    3.25

    >>> a = [6, 47, 49, 15, 42, 41, 7, 39, 43, 40, 36]
    >>> scoreatpercentile(a, 25)
    25.5
    >>> scoreatpercentile(a, 50)
    40
    >>> scoreatpercentile(a, 75)
    42.5

    This function is taken from :func:`scipy.stats.scoreatpercentile`.

    Copyright (c) Gary Strangman
    """
    if issorted:
        values = sorted(a)
        if limit:
            values = values[(limit[0] < a) & (a < limit[1])]
    else:
        values = a

    def _interpolate(a, b, fraction):
        return a + (b - a) * fraction

    idx = per / 100. * (len(values) - 1)
    if (idx % 1 == 0):
        return values[int(idx)]
    else:
        return _interpolate(values[int(idx)], values[int(idx) + 1], idx % 1)


def flatnotmaskedContiguous(a):
    """
    Find contiguous unmasked data in a masked array along the given axis.

    This function is taken from
    :func:`numpy.ma.extras.flatnotmasked_contiguous`.

    Copyright (c) Pierre Gerard-Marchant
    """
    np.ma.extras.flatnotmasked_contiguous
    m = np.ma.getmask(a)
    if m is np.ma.nomask:
        return slice(0, a.size, None)
    i = 0
    result = []
    for (k, g) in itertools.groupby(m.ravel()):
        n = len(list(g))
        if not k:
            result.append(slice(i, i + n))
        i += n
    return result or None


def complexifyString(line):
    """
    Converts a string in the form "(real, imag)" into a complex type.

    :type line: str
    :param line: String in the form ``"(real, imag)"``.
    :rtype: complex
    :return: Complex number.

    .. rubric:: Example

    >>> complexifyString("(1,2)")
    (1+2j)

    >>> complexifyString(" ( 1 , 2 ) ")
    (1+2j)
    """
    temp = line.split(',')
    return complex(float(temp[0].strip()[1:]), float(temp[1].strip()[:-1]))


def toIntOrZero(value):
    """
    Converts given value to an integer or returns 0 if it fails.

    :param value: Arbitrary data type.
    :rtype: int
numpy.version.version
    .. rubric:: Example

    >>> toIntOrZero("12")
    12

    >>> toIntOrZero("x")
    0
    """
    try:
        return int(value)
    except ValueError:
        return 0


# import numpy loadtxt and check if ndlim parameter is available
try:
    from numpy import loadtxt
    loadtxt(np.array([]), ndlim=1)
except TypeError:
    # otherwise redefine loadtxt
    def loadtxt(*args, **kwargs):
        """
        Replacement for older numpy.loadtxt versions not supporting ndlim
        parameter.
        """
        if not 'ndlim' in kwargs:
            return np.loadtxt(*args, **kwargs)
        # ok we got a ndlim param
        if kwargs['ndlim'] != 1:
            # for now we support only one dimensional arrays
            raise NotImplementedError('Upgrade your NumPy version!')
        del kwargs['ndlim']
        dtype = kwargs.get('dtype', None)
        # lets get the data
        try:
            data = np.loadtxt(*args, **kwargs)
        except IOError, e:
            # raises in older versions if no data could be read
            if 'reached before encountering data' in str(e):
                # return empty array
                return np.array([], dtype=dtype)
            # otherwise just raise
            raise
        # ensures that an array is returned
        return np.atleast_1d(data)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
