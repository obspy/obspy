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
import matplotlib.pyplot as plt

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
    # Using scipy.interpolate.piecewise_polynomial_interpolate() is too
    # memory intensive
    return_data = np.empty(len(new_time_array), dtype=np.float64)
    clibsignal.hermite_interpolation(data, slope, new_time_array, return_data,
                                     len(data), len(return_data), old_dt,
                                     old_start)
    return return_data


# Map corresponding to the enum on the C side of things.
_LANCZOS_KERNEL_MAP = {
    "lanczos": 0,
    "hanning": 1,
    "blackman": 2
}


def lanczos_interpolation(data, old_start, old_dt, new_start, new_dt, new_npts,
                          a, window="lanczos", *args, **kwargs):
    r"""
    Function performing Lanczos resampling, see
    http://en.wikipedia.org/wiki/Lanczos_resampling for details. Essentially a
    finite support version of sinc resampling (the ideal reconstruction
    filter). For large values of ``a`` it converges towards sinc resampling. If
    used for downsampling, make sure to apply an appropriate anti-aliasing
    lowpass filter first.

    .. note::

        In most cases you do not want to call this method directly but invoke
        it via either the :meth:`obspy.core.stream.Stream.interpolate` or
        :meth:`obspy.core.trace.Trace.interpolate` method. These offer a nicer
        API that naturally integrates with the rest of ObsPy. Use
        ``method="lanczos"`` to use this interpolation method. In that case the
        only additional parameters of interest are ``a`` and ``window``.

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
    :type a: int
    :param a: The width of the window in samples on either side. Runtime
        scales linearly with the value of ``a`` but the interpolation also gets
        better.
    :type window: str
    :param window: The window used to taper the sinc function. One of
        ``"lanczos"``, ``"hanning"``, ``"blackman"``. The window determines
        the trade-off between "sharpness" and the amplitude of the wiggles in
        the pass and stop band. Please use the
        :func:`~obspy.signal.interpolation.plot_lanczos_windows` function to
        judge these for any given application.

    Values of ``a`` >= 20 show good results even for data that has
    energy close to the Nyquist frequency. If your data is extremely
    oversampled you can get away with much smaller ``a``'s.

    To get an idea of the response of the filter and the effect of the
    different windows, please use the
    :func:`~obspy.signal.interpolation.plot_lanczos_windows` function.

    Also be aware of any boundary effects. All values outside the data
    range are assumed to be zero which matters when calculating interpolated
    values at the boundaries. At each side the area with potential boundary
    effects is ``a`` * ``old_dt``. If you want to avoid any boundary effects
    you will have to remove these values.

    **Mathematical Details:**

    The :math:`\operatorname{sinc}` function is defined as

    .. math::

        \operatorname{sinc}(t) = \frac{\sin(\pi t)}{\pi t}.

    The Lanczos kernel is then given by a multiplication of the
    :math:`\operatorname{sinc}` function with an additional window function
    resulting in a finite support kernel.

    .. math::

        \begin{align}
            L(t) =
            \begin{cases}
                \operatorname{sinc}(t)\, \cdot \operatorname{sinc}(t/a)
                    & \text{if } t \in [-a, a]
                    \text{ and } \texttt{window} = \texttt{lanczos}\\
                \operatorname{sinc}(t)\, \cdot \frac{1}{2}
                (1 + \cos(\pi\, t/a))
                    & \text{if } t \in [-a, a]
                    \text{ and } \texttt{window} = \texttt{lanczos}\\
                \operatorname{sinc}(t)\, \cdot \left( \frac{21}{50} +
                \frac{1}{2}
                \cos(\pi\, t/a) + \frac{2}{25} \cos (2\pi\, t/a) \right)
                    & \text{if } t \in [-a, a]
                    \text{ and } \texttt{window} = \texttt{lanczos}\\
                0                     & \text{else}
            \end{cases}
        \end{align}


    Finally interpolation is performed by convolving the discrete signal
    :math:`s_i` with that kernel and evaluating it at the new time samples
    :math:`t_j`:

    .. math::

        \begin{align}
            S(t_j) =
                \sum_{i = \left \lfloor{t_j / \Delta t}\right \rfloor -a + 1}
                    ^{\left \lfloor{t_j / \Delta t}\right \rfloor + a}
            s_i L(t_j/\Delta t - i),
        \end{align}

    where :math:`\lfloor \cdot \rfloor` denotes the floor function. For more
    details and justification please see [Burger2009]_ and [vanDriel2015]_.
    """
    _validate_parameters(data, old_start, old_dt, new_start, new_dt, new_npts)

    # dt and offset in terms of the original sampling interval.
    dt_factor = float(new_dt) / old_dt
    offset = (new_start - old_start) / float(old_dt)

    if offset < 0:
        raise ValueError("Cannot extrapolate. Make sure to only interpolate "
                         "within the time range of the original signal.")

    if a < 1:
        raise ValueError("a must be at least 1.")

    return_data = np.zeros(new_npts, dtype=np.float64)

    clibsignal.lanczos_resample(
        np.require(data, dtype=np.float64), return_data, dt_factor, offset,
        len(data), len(return_data), int(a), 0)
    return return_data


def calculate_lanczos_kernel(x, a, window):
    """
    Helper function to get the actually used kernel for a specific value of
    a. Useful to analyse the behaviour of different tapers and different values
    of a.

    :type x: :class:`numpy.ndarray`
    :param x: The x values at which to calculate the kernel.
    :type a: int
    :param a: The width of the window in samples on either side.
    :type window: str
    :param window: The window used to multiply the sinc function with. One
        of ``"lanczos"``, ``"hanning"``, ``"blackman"``.

    Returns a dictionary of arrays:

    * ``"full_kernel"``: The tapered sinc function evaluated at samples ``x``.
    * ``"only_sinc"``: The sinc function evaluated at samples ``x``.
    * ``"only_taper"``: The taper function evaluated at samples ``x``.
    """
    window = window.lower()
    if window not in _LANCZOS_KERNEL_MAP:
        msg = "Invalid window. Valid windows: %s" % ", ".join(
            sorted(_LANCZOS_KERNEL_MAP.keys()))
        raise ValueError(msg)

    x = np.require(x, dtype=np.float64)
    y0 = np.zeros(x.shape, dtype=np.float64)
    y1 = np.zeros(x.shape, dtype=np.float64)
    y2 = np.zeros(x.shape, dtype=np.float64)

    clibsignal.calculate_kernel(
        x, y0, len(x), a, 0, _LANCZOS_KERNEL_MAP[window])
    clibsignal.calculate_kernel(
        x, y1, len(x), a, 1, _LANCZOS_KERNEL_MAP[window])
    clibsignal.calculate_kernel(
        x, y2, len(x), a, 2, _LANCZOS_KERNEL_MAP[window])

    ret_val = {
        "full_kernel": y0,
        "only_sinc": y1,
        "only_taper": y2
    }

    return ret_val


def plot_lanczos_windows(a):
    """
    Helper function producing a plot of all available tapers of the sinc
    function and their response for the Lanczos interpolation.

    :type a: int
    :param a: The width of the window in samples on either side.

    .. code-block:: python

        from obspy.signal.interpolation import plot_lanczos_windows
        plot_lanczos_windows(a=20)

    .. plot::

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 15))
        from obspy.signal.interpolation import plot_lanczos_windows
        plot_lanczos_windows(a=20)
    """
    x_max = 1024.0 - 0.5
    n = 2 ** 15
    x = np.linspace(-x_max, x_max, n)
    dx = 2 * x_max / (n - 1)

    arrays = {}
    for key in _LANCZOS_KERNEL_MAP.keys():
        arrays[key] = calculate_lanczos_kernel(x, a, key)
        arrays[key]["fft"] = \
            np.abs(np.fft.rfft(arrays[key]["full_kernel"]) * dx)

    height = len(_LANCZOS_KERNEL_MAP) + 1

    plt.subplot(height, 2, 1)
    for key in sorted(arrays.keys()):
        plt.plot(x, arrays[key]["full_kernel"], label=key.capitalize())

    plt.legend()
    plt.xlim(-a, a)
    plt.ylim(-0.3, 1.1)
    plt.title("All Windows")

    plt.subplot(height, 2, 2)
    plt.plot([0.0, 0.5, 0.5, 1000], [1.0, 1.0, 0.0, 0.0], "--",
             color="0.1", label="Ideal")
    for key in sorted(arrays.keys()):
        plt.plot(np.fft.rfftfreq(len(x), dx),
                 arrays[key]["fft"], label=key.capitalize())
    plt.xlim(0.2, 0.8)
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.title("Frequency Response of All Windows")

    for _i, key in enumerate(sorted(_LANCZOS_KERNEL_MAP.keys())):
        plt.subplot(height, 2, 3 + 2 * _i)
        plt.title(key.capitalize())
        plt.plot(x, arrays[key]["full_kernel"], color="black", label="Final")
        plt.plot(x, arrays[key]["only_sinc"], "--", color="gray", label="Sinc")
        plt.plot(x, arrays[key]["only_taper"], color="red", label="Taper")
        plt.legend()
        plt.xlim(-a, a)
        plt.ylim(-0.3, 1.1)

        plt.subplot(height, 2, 3 + 2 * _i + 1)
        plt.title(key.capitalize() + " Response")
        plt.plot([0.0, 0.5, 0.5, 1000], [1.0, 1.0, 0.0, 0.0], "--",
                 color="0.1")
        plt.plot(np.fft.rfftfreq(len(x), dx),
                 arrays[key]["fft"])
        plt.xlim(0.2, 0.8)
        plt.ylim(-0.1, 1.1)

    plt.suptitle("Different windows for sinc interpolation with a=%i"
                 % a, fontsize="large")
    plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
