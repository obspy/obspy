#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: regression.py
#  Purpose: Python Module for (Weighted) Linear Regression
#   Author: Thomas Lecocq
#    Email: Thomas.Lecocq@seismology.be
#
# Copyright (C) 2017 Thomas Lecocq
# --------------------------------------------------------------------
"""
Python Module for (Weighted) Linear Regression.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import scipy.optimize
import numpy as np


def linear_regression(xdata, ydata, weights=None, p0=None, intercept=False):
    """ Use non-linear least squares to fit a function, f, to data. This method
    is a generalized version of :meth:`scipy.optimize.minpack.curve_fit`;
    allowing for Ordinary Least Square and Weighted Least Square regressions:

    * OLS without intercept : ``linear_regression(xdata, ydata)``
    * OLS with intercept : ``linear_regression(xdata, ydata, intercept=True)``
    * WLS without intercept : ``linear_regression(xdata, ydata, weights)``
    * WLS with intercept : ``linear_regression(xdata, ydata, weights,
     intercept=True)``

    If the expected values of slope (and intercept) are different from 0.0,
    provide the p0 value(s).

    :param xdata: The independent variable where the data is measured.
    :param ydata: The dependent data - nominally f(xdata, ...)
    :param weights: If not None, the uncertainties in the ydata array. These
     are used as weights in the least-squares problem. If None, the
     uncertainties are assumed to be 1. In SciPy vocabulary, our weights are
     1/sigma.
    :param p0: Initial guess for the parameters. If None, then the initial
     values will all be 0 (Different from SciPy where all are 1)
    :param intercept: If False: solves y=a*x ; if True: solves y=a*x+b.

    :rtype: tuple
    :returns: (slope, intercept, std_slope, std_intercept) if `intercept` is
     `True` or (slope, std_slope) if `False`
    """
    if weights is not None:
        sigma = 1./weights
    else:
        sigma = None

    if p0 is None:
        if intercept:
            p0 = [0.0, 0.0]
        else:
            p0 = 0.0

    if intercept:
        p, cov = scipy.optimize.curve_fit(lambda x, a, b: a * x + b,
                                          xdata, ydata, p0, sigma=sigma,
                                          absolute_sigma=False,
                                          xtol=1e-20)
        slope, intercept = p
        std_slope = np.sqrt(cov[0, 0])
        std_intercept = np.sqrt(cov[1, 1])
        return slope, intercept, std_slope, std_intercept

    else:
        p, cov = scipy.optimize.curve_fit(lambda x, a: a * x,
                                          xdata, ydata, p0, sigma=sigma,
                                          absolute_sigma=False,
                                          xtol=1e-20)
        slope = p[0]
        std_slope = np.sqrt(cov[0, 0])
        return slope, std_slope
