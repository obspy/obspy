#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python module containing detrend methods.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np
from scipy.interpolate import LSQUnivariateSpline


def simple(data):
    """
    Detrend signal simply by subtracting a line through the first and last
    point of the trace

    :param data: Data to detrend, type numpy.ndarray.
    :return: Detrended data.
    """
    ndat = len(data)
    x1, x2 = data[0], data[-1]
    return data - (x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1))


def polynomial(data, order):
    """
    Remove a polynomial trend from the data.

    :param data: The data to detrend. Will be modified in-place.
    :type data: :class:`numpy.ndarray`
    :param order: The order of the polynomial to fit.
    :type order: int
    """
    # Convert data if its not a floating point type.
    if not np.issubdtype(data.dtype, float):
        data = np.require(data, dtype=np.float32)

    x = np.arange(len(data))
    coefs = np.polyfit(x, data, deg=order)

    data -= np.polyval(coefs, x)

    return data


def spline(data, order, dspline):
    """
    Remove trend with a spline.

    :param data: The data to detrend. Will be modified in-place.
    :type data: :class:`numpy.ndarray`
    :param order: The order/degree of the smoothing spline to fit.
        Must be 1 <= order <= 5.
    :type order: int
    :param dspline: The distance in samples between two spline nodes.
    :type dspline: int
    """
    # Convert data if its not a floating point type.
    if not np.issubdtype(data.dtype, float):
        data = np.require(data, dtype=np.float32)

    x = np.arange(len(data))
    splknots = np.arange(dspline / 2.0, len(data) - dspline / 2.0 + 2,
                         dspline)

    spl = LSQUnivariateSpline(x=x, y=data, t=splknots)
    data -= spl(x)
    return data


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
