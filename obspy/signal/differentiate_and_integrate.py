# -*- coding: utf-8 -*-
"""
Integration and differentiation routines.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np
import scipy.integrate
import scipy.interpolate


def integrate_cumtrapz(data, dx, **kwargs):
    """
    Performs first order integration of data using the trapezoidal rule.

    :param data: Data array to integrate.
    :param dx: Sample spacing usually in seconds.
    """
    # Integrate. Set first value to zero to avoid changing the total
    # length of the array.
    # (manually adding the zero and not using `cumtrapz(..., initial=0)` is a
    # backwards compatibility fix for scipy versions < 0.11.
    ret = scipy.integrate.cumtrapz(data, dx=dx)
    return np.concatenate([np.array([0], dtype=ret.dtype), ret])


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

    # Backport of the antiderivative() method for scipy versions < 0.13.0.
    # Can be removed once the minimum supported version is equal or larger
    # to this.
    if not hasattr(spline, "antiderivative"):
        t, c, k = spline._eval_args

        # Compute the multiplier in the antiderivative formula.
        dt = t[k + 1:] - t[:-k - 1]
        # Compute the new coefficients
        c = np.cumsum(c[:-k - 1] * dt) / (k + 1)
        c = np.r_[0, c, [c[-1]] * (k + 2)]
        # New knots
        t = np.r_[t[0], t, t[-1]]
        k += 1

        tmp = scipy.interpolate.InterpolatedUnivariateSpline.__new__(
            scipy.interpolate.InterpolatedUnivariateSpline)
        tmp._eval_args = t, c, k
        tmp._data = (None, None, None, None, None, k, None, len(t), t, c,
                     None, None, None, None)
        tmp.ext = 0
        return tmp(time_array)

    return spline.antiderivative(n=1)(time_array)
