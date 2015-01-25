#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration and differentiation routines.

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
import scipy.interpolate
import scipy.integrate


def integrate_cumtrapz(data, dx, **kwargs):
    """
    Performs first order integration of data using the trapezoidal rule.

    :param data: Data array to integrate.
    :param dx: Sample spacing usually in seconds.
    """
    # Integrate. Set first value to zero to avoid changing the total
    # length of the array.
    return scipy.integrate.cumtrapz(data, dx=dx, initial=0)


def integrate_spline(data, dx, k=3, **kwargs):
    """
    Integrate by generating an interpolating spline and integrating that.

    :param data: The data to integrate.
    :param dx: Sample spacing usually in seconds.
    :param k: Spline order. 1 is linear, 2 quadratic, 3 cubic, Must be
        between 1 and 5.
    """
    time_array = np.linspace(0, (len(data) - 1) * dx, len(data))
    spline = scipy.interpolate.InterpolatedUnivariateSpline(time_array, data,
                                                            k=k)
    return spline.antiderivative(n=1)(time_array)
