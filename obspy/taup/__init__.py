# -*- coding: utf-8 -*-
"""
obspy.taup - Ray Theoretical Travel Times and Paths
===================================================

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

This package started out as port of the Java TauP Toolkit by [Crotwell1999]_ so
please look there for more details about the used algorithms and further
information. It can be used to calculate theoretical arrival times for
arbitrary seismic phases in a 1D spherically symmetric background model.
Furthermore it can output ray paths for all phases and derive pierce points of
rays with model discontinuities.

Basic Usage
-----------

Start by initializing a :class:`~obspy.taup.tau.TauPyModel` class.

>>> from obspy.taup import TauPyModel
>>> model = TauPyModel(model="iasp91")

Model initialization is a fairly expensive operation so make sure to do it only
if necessary.

Travel Times
^^^^^^^^^^^^
The models' main method is the
:meth:`~obspy.taup.tau.TauPyModel.get_travel_times` method; as the name
suggests it returns travel times for the chosen phases, distance, source depth,
and model. Per default it returns arrivals for a number of phases.

>>> arrivals = model.get_travel_times(source_depth_in_km=55,
...                                   distance_in_degree=67)
>>> print(arrivals)
28 arrivals
    P phase arrival at 647.036 seconds
    pP phase arrival at 662.230 seconds
    sP phase arrival at 668.702 seconds
    PcP phase arrival at 674.868 seconds
    PP phase arrival at 794.975 seconds
    PKiKP phase arrival at 1034.106 seconds
    pPKiKP phase arrival at 1050.535 seconds
    sPKiKP phase arrival at 1056.727 seconds
    S phase arrival at 1176.947 seconds
    pS phase arrival at 1195.500 seconds
    SP phase arrival at 1196.827 seconds
    sS phase arrival at 1203.128 seconds
    PS phase arrival at 1205.418 seconds
    SKS phase arrival at 1239.088 seconds
    SKKS phase arrival at 1239.107 seconds
    ScS phase arrival at 1239.515 seconds
    SKiKP phase arrival at 1242.400 seconds
    pSKS phase arrival at 1260.313 seconds
    sSKS phase arrival at 1266.919 seconds
    SS phase arrival at 1437.417 seconds
    PKIKKIKP phase arrival at 1855.260 seconds
    SKIKKIKP phase arrival at 2063.556 seconds
    PKIKKIKS phase arrival at 2069.749 seconds
    SKIKKIKS phase arrival at 2277.833 seconds
    PKIKPPKIKP phase arrival at 2353.930 seconds
    PKPPKP phase arrival at 2356.420 seconds
    PKPPKP phase arrival at 2358.925 seconds
    SKIKSSKIKS phase arrival at 3208.154 seconds

If you know which phases you are interested in, you can also specify them
directly which speeds up the calculation as unnecessary phases are not
calculated. Please note that it is possible to construct any phases that
adhere to the naming scheme which is detailed later.

>>> arrivals = model.get_travel_times(source_depth_in_km=100,
...                                   distance_in_degree=45,
...                                   phase_list=["P", "PSPSPS"])
>>> print(arrivals)
3 arrivals
    P phase arrival at 485.204 seconds
    PSPSPS phase arrival at 4983.023 seconds
    PSPSPS phase arrival at 5799.225 seconds

Each arrival is represented by an :class:`~obspy.taup.helper_classes.Arrival`
object which can be queried for various attributes.

>>> arr = arrivals[0]
>>> arr.ray_param, arr.time, arr.incident_angle
(453.71881662349625, 485.20416952979105, 24.396848002294515)

Ray Paths
^^^^^^^^^

To also calculate the paths the rays took to the receiver, use the
:meth:`~obspy.taup.tau.TauPyModel.get_ray_paths` method.

>>> arrivals = model.get_ray_paths(500, 130)
>>> arrival = arrivals[0]

The result is a NumPy record array containing ray parameter, time, distance
and depth to use however you see fit.

>>> arrival.path.dtype
dtype([('p', '<f8'), ('time', '<f8'), ('dist', '<f8'), ('depth', '<f8')])


Pierce Points
^^^^^^^^^^^^^

If you only need the pierce points of ray paths with model discontinuities,
use the :meth:`~obspy.taup.tau.TauPyModel.get_pierce_points` method which
results in pierce points being stored as a record array on the arrival object.

>>> arrivals = model.get_pierce_points(500, 130)
>>> arrivals[0].pierce.dtype
dtype([('p', '<f8'), ('time', '<f8'), ('dist', '<f8'), ('depth', '<f8')])


Plotting
--------

If ray paths have been calculated, they can be plotted using the
:meth:`~obspy.taup.tau.Arrivals.plot` method:

>>> arrivals = model.get_ray_paths(source_depth_in_km=500,
                                   distance_in_degree=130)
>>> arrivals.plot()  # doctest: +SKIP

.. plot::
    :width: 50%
    :align: center

    from obspy.taup import TauPyModel
    TauPyModel().get_ray_paths(500, 130).plot()

It will only plot rays for requested phases.


>>> arrivals = model.get_ray_paths(
...     source_depth_in_km=500,
...     distance_in_degree=130,
...     phase_list=["Pdiff", "Sdiff", "pPdiff", "sSdiff"])
>>> arrivals.plot()  # doctest: +SKIP

.. plot::
    :width: 50%
    :align: center

    from obspy.taup import TauPyModel
    TauPyModel().get_ray_paths(
        500, 130,
        phase_list=["Pdiff", "Sdiff", "pPdiff", "sSdiff"]).plot()

Additionally it can also plot on a Cartesian instead of a polar grid.

>>> arrivals = model.get_ray_paths(source_depth_in_km=500,
...                                distance_in_degree=130,
...                                phase_list=["ttbasic"])
>>> arrivals.plot(plot_type="cartesian")  # doctest: +SKIP

.. plot::
    :width: 75%
    :align: center

    from obspy.taup import TauPyModel
    TauPyModel().get_ray_paths(
        500, 130, phase_list=["ttbasic"]).plot(plot_type="cartesian")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import os

# Convenience imports.
from .tau import TauPyModel  # NOQA
from .taup import getTravelTimes, travelTimePlot  # NOQA


# Most generic way to get the data directory.
__DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
