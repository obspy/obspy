# -*- coding: utf-8 -*-
"""
Wrapper around obspy.taup.tau.

Still around for legacy reasons.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np
import warnings

from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from . import tau


_MODEL_CACHE = {}


def getTravelTimes(delta, depth, model='iasp91', phase_list=["ttall"]):
    """
    Returns travel times.

    :type delta: float
    :param delta: Distance in degrees.
    :type depth: float
    :param depth: Depth in kilometer.
    :type model: str, optional
    :param model: Either ``'iasp91'`` or ``'ak135'`` velocity model. Defaults
        to ``'iasp91'``.
    :param phase_list: List of desired phase names. Will be passed to taupy.
    :type phase_list: list of strings
    :rtype: list of dicts
    :return:
        A list of phase arrivals given in time order. Each phase is represented
        by a dictionary containing phase name, travel time in seconds, take-off
        angle, and the ray parameter.

    .. rubric:: Example

    >>> from obspy.taup.taup import getTravelTimes
    >>> tt = getTravelTimes(delta=52.474, depth=611.0, model='ak135')
    >>> len(tt)
    24
    >>> tt[0]  #doctest: +SKIP
    {'phase_name': 'P', 'dT/dD': 7.1050525, 'take-off angle': 45.169445,
     'time': 497.53741}

    .. versionchanged:: 0.10.0

        Deprecated.

        The backend is no longer the Fortran iasp-tau program but a Python
        port of the Java TauP library. A consequence of this is that the
        ``"dT/dh"`` and ``"d2T/dD2"`` values are no longer returned.

        Furthermore this function now has a ``phase_list`` keyword argument.
    """
    warnings.warn("The getTravelTimes() function is deprecated. Please use "
                  "the obspy.taup.TauPyModel class directly.",
                  ObsPyDeprecationWarning, stacklevel=2)
    model = model.lower()

    # Cache models.
    if model in ("ak135", "iasp91") and model in _MODEL_CACHE:
        tau_model = _MODEL_CACHE[model]
    else:
        tau_model = tau.TauPyModel(model)
        if model in ("ak135", "iasp91"):
            _MODEL_CACHE[model] = tau_model

    tt = tau_model.get_travel_times(source_depth_in_km=float(depth),
                                    distance_in_degree=float(delta),
                                    phase_list=phase_list)
    return [{
        "phase_name": arr.purist_name,
        "time": arr.time,
        "take-off angle": arr.takeoff_angle,
        "dT/dD": arr.ray_param_sec_degree} for arr in tt]


def travelTimePlot(min_degree=0, max_degree=360, npoints=1000,
                   phases=None, depth=100, model='iasp91'):
    """
    Basic travel time plotting function.

    :type min_degree: float, optional
    :param min_degree: Minimum distance in degree used in plot.
        Defaults to ``0``.
    :type max_degree: float, optional
    :param max_degree: Maximum distance in degree used in plot.
        Defaults to ``360``.
    :type npoints: int, optional
    :param npoints: Number of points to plot. Defaults to ``1000``.
    :type phases: list of str, optional
    :param phases: List of phase names which should be used within the plot.
        Defaults to all phases if not explicit set.
    :type depth: float, optional
    :param depth: Depth in kilometer. Defaults to ``100``.
    :type model: str, optional
    :param model: Either ``'iasp91'`` or ``'ak135'`` velocity model.
        Defaults to ``'iasp91'``.
    :return: None

    .. rubric:: Example

    >>> from obspy.taup.taup import travelTimePlot
    >>> travelTimePlot(min_degree=0, max_degree=50, phases=['P', 'S', 'PP'],
    ...                depth=120, model='iasp91')  #doctest: +SKIP

    .. plot::

        from obspy.taup.taup import travelTimePlot
        travelTimePlot(min_degree=0, max_degree=50, phases=['P', 'S', 'PP'],
                       depth=120, model='iasp91')

    .. versionchanged:: 0.10.0

        Deprecated.
    """
    warnings.warn("The travelTimePlot() function is deprecated. Please use "
                  "the obspy.taup.TauPyModel class directly.",
                  ObsPyDeprecationWarning, stacklevel=2)
    import matplotlib.pylab as plt

    data = {}

    if not phases:
        phases = ["ttall"]

    degrees = np.linspace(min_degree, max_degree, npoints)
    # Loop over all degrees.
    for degree in degrees:
        with warnings.catch_warnings(record=True):
            tt = getTravelTimes(degree, depth, model, phase_list=phases)
        # Mirror if necessary.
        if degree > 180:
            degree = 180 - (degree - 180)
        for item in tt:
            phase = item['phase_name']
            if phase not in data:
                data[phase] = [[], []]
            data[phase][1].append(item['time'] / 60.0)
            data[phase][0].append(degree)
    # Plot and some formatting.
    for key, value in data.items():
        plt.plot(value[0], value[1], '.', label=key)
    plt.grid()
    plt.xlabel('Distance (degrees)')
    plt.ylabel('Time (minutes)')
    if max_degree <= 180:
        plt.xlim(min_degree, max_degree)
    else:
        plt.xlim(min_degree, 180)
    plt.legend(numpoints=1)
    plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
