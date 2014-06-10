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
from scipy.interpolate import interp1d


def _validate_parameters(data, old_start, old_dt, new_start, new_dt, new_npts):
    """
    Validates the parameters for various interpolation functions.

    Returns the old and the new end.
    """
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
                   type="linear"):
    """
    Wrapper around scipy.interpolate.interp1d.

    :type data: array like
    :param data: Array to interpolate.
    :type old_start: float
    :param old_start: The start of the array as a number.
    :type old_start: float
    :param old_dt: The time delta of the current array.
    :type new_start: float
    :param new_start: The start of the interpolated array. Must be greater
        or equal to the current start of the array.
    :type new_dt: float
    :param new_dt: The desired ewn time delta.
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

    new_data = interp1d(old_time_array, data, kind=1)(new_time_array)
    return new_data


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
