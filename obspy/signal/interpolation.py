#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some Seismogram Interpolating Functions.

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

from obspy.signal.headers import clibsignal


def _validate_parameters(data, old_start, old_dt, new_start, new_dt, new_npts):
    """
    Validates the parameters for various interpolation functions.

    Returns the old and the new end.
    """
    if new_dt <= 0.0:
        raise ValueError("The time step must be positive.")

    # Check for 1D array.
    if data.ndim != 1 or not len(data) or not data.shape[-1]:
        raise ValueError("Not a 1D array.")

    old_end = old_start + old_dt * (len(data) - 1)
    new_end = new_start + new_dt * (new_npts - 1)

    if old_start > new_start or old_end < new_end:
        raise ValueError("The new array must be fully contained in the old "
                         "array. No extrapolation can be performed.")

    return old_end, new_end


def interpolate_1d(data, old_start, old_dt, new_start, new_dt, new_npts,
                   type="linear", *args, **kwargs):
    """
    Wrapper around some scipy interpolation functions.

    :type data: array_like
    :param data: Array to interpolate.
    :type old_start: float
    :param old_start: The start of the array as a number.
    :type old_start: float
    :param old_dt: The time delta of the current array.
    :type new_start: float
    :param new_start: The start of the interpolated array. Must be greater
        or equal to the current start of the array.
    :type new_dt: float
    :param new_dt: The desired new time delta.
    :type new_npts: int
    :param new_npts: The new number of samples.
    :type type: str or int
    :param type: Specifies the kind of interpolation as a string (``linear``,
        ``nearest``, ``zero``, ``slinear``, ``quadratic``, ``cubic`` where
        ``slinear``, ``quadratic`` and ``cubic`` refer to a spline
        interpolation of first,  second or third order) or as an integer
        specifying the order of the spline interpolator to use. Default is
        ``linear``.
    """
    old_end, new_end = _validate_parameters(data, old_start, old_dt,
                                            new_start, new_dt, new_npts)

    # In almost all cases the unit will be in time.
    new_time_array = np.linspace(new_start, new_end, new_npts)
    old_time_array = np.linspace(old_start, old_end, len(data))

    s_map = {
        "slinear": 1,
        "quadratic": 2,
        "cubic": 3
    }
    if type in s_map:
        type = s_map[type]

    # InterpolatedUnivariateSpline uses a sane amount of memory for splines.
    # interp1d can easily require 50 GB of memory which is clearly
    # not acceptable.
    if isinstance(type, int):
        new_data = scipy.interpolate.InterpolatedUnivariateSpline(
            old_time_array, data, k=type)(new_time_array)
    # interp1d is used for the "linear", "nearest", and "zero" interpolation
    # methods.
    else:
        new_data = scipy.interpolate.interp1d(old_time_array, data, kind=type)(
            new_time_array)

    return new_data


def weighted_average_slopes(data, old_start, old_dt, new_start, new_dt,
                            new_npts, *args, **kwargs):
    """
    Implements the weighted average slopes interpolation scheme proposed in
    [Wiggins1976]_ for evenly sampled data. The scheme guarantees that there
    will be no additional extrema after the interpolation in contrast to
    spline interpolation.

    The slope :math:`s_i` at each knot is given by a weighted average of the
    adjacent linear slopes :math:`m_i` and :math:`m_{i+j}`:

    .. math::

        s_i = (w_i m_i + w_{i+1} m_{i+1}) / (w_i + w_{i+1})

    where

    .. math::

        w = 1 / max \left\{ \left\| m_i \\right\|, \epsilon \\right\}

    The value at each data point and the slope are then plugged into a
    piecewise continuous cubic polynomial used to evaluate the interpolated
    sample points.

    :type data: array_like
    :param data: Array to interpolate.
    :type old_start: float
    :param old_start: The start of the array as a number.
    :type old_start: float
    :param old_dt: The time delta of the current array.
    :type new_start: float
    :param new_start: The start of the interpolated array. Must be greater
        or equal to the current start of the array.
    :type new_dt: float
    :param new_dt: The desired new time delta.
    :type new_npts: int
    :param new_npts: The new number of samples.
    """
    old_end, new_end = _validate_parameters(data, old_start, old_dt,
                                            new_start, new_dt, new_npts)
    # In almost all cases the unit will be in time.
    new_time_array = np.linspace(new_start, new_end, new_npts)

    m = np.diff(data) / old_dt
    w = np.abs(m)
    w = 1.0 / np.clip(w, np.spacing(1), w.max())

    slope = np.empty(len(data), dtype=np.float64)
    slope[0] = m[0]
    slope[1:-1] = (w[:-1] * m[:-1] + w[1:] * m[1:]) / (w[:-1] + w[1:])
    slope[-1] = m[-1]

    # If m_i and m_{i+1} have opposite signs then set the slope to zero.
    # This forces the curve to have extrema at the sample points and not
    # in-between.
    sign_change = np.diff(np.sign(m)).astype(np.bool)
    slope[1:-1][sign_change] = 0.0

    derivatives = np.empty((len(data), 2), dtype=np.float64)
    derivatives[:, 0] = data
    derivatives[:, 1] = slope

    # Create interpolated value using hermite interpolation. In this case
    # it is directly applicable as the first derivatives are known.
    # Using scipy.interpolate.piecewise_polynomial_interpolate() is to
    # memory intensive
    return_data = np.empty(len(new_time_array), dtype=np.float64)
    clibsignal.hermite_interpolation(data, slope, new_time_array, return_data,
                                     len(data), len(return_data), old_dt,
                                     old_start)
    return return_data


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
