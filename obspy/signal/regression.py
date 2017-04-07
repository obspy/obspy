# -*- coding: utf-8 -*-
"""
Python Module for (Weighted) Linear Regression.

:authors:
    Thomas Lecocq (thomas.lecocq@seismology.be)

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np
import scipy.optimize


def linear_regression(xdata, ydata, weights=None, p0=None,
                      intercept_origin=True, **kwargs):
    """
    Use linear least squares to fit a function, f, to data.
    This method is a generalized version of
    :func:`scipy.optimize.minpack.curve_fit`; allowing for Ordinary Least
    Square and Weighted Least Square regressions:

    * OLS through origin: ``linear_regression(xdata, ydata)``
    * OLS with any intercept: ``linear_regression(xdata, ydata,
      intercept_origin=False)``
    * WLS through origin: ``linear_regression(xdata, ydata, weights)``
    * WLS with any intercept: ``linear_regression(xdata, ydata, weights,
      intercept_origin=False)``

    If the expected values of slope (and intercept) are different from 0.0,
    provide the p0 value(s).

    :param xdata: The independent variable where the data is measured.
    :param ydata: The dependent data - nominally f(xdata, ...)
    :param weights: If not None, the uncertainties in the ydata array. These
        are used as weights in the least-squares problem. If ``None``, the
        uncertainties are assumed to be 1. In SciPy vocabulary, our weights are
        1/sigma.
    :param p0: Initial guess for the parameters. If ``None``, then the initial
        values will all be 0 (Different from SciPy where all are 1)
    :param intercept_origin: If ``True``: solves ``y=a*x`` (default);
        if ``False``: solves ``y=a*x+b``.

    Extra keword arguments will be passed to
    :func:`scipy.optimize.minpack.curve_fit`.

    :rtype: tuple
    :returns: (slope, std_slope) if ``intercept_origin`` is ``True``;
        (slope, intercept, std_slope, std_intercept) if ``False``.
    """
    if weights is not None:
        sigma = 1. / weights
    else:
        sigma = None

    if p0 is None:
        if intercept_origin:
            p0 = 0.0
        else:
            p0 = [0.0, 0.0]

    if intercept_origin:
        p, cov = scipy.optimize.curve_fit(lambda x, a: a * x,
                                          xdata, ydata, p0, sigma=sigma,
                                          **kwargs)
        slope = p[0]
        std_slope = np.sqrt(cov[0, 0])
        return slope, std_slope

    else:
        p, cov = scipy.optimize.curve_fit(lambda x, a, b: a * x + b,
                                          xdata, ydata, p0, sigma=sigma,
                                          **kwargs)
        slope, intercept = p
        std_slope = np.sqrt(cov[0, 0])
        std_intercept = np.sqrt(cov[1, 1])
        return slope, intercept, std_slope, std_intercept


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
