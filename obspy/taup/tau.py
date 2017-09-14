#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
High-level interface to travel-time calculation routines.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import copy
import warnings

import matplotlib.cbook
from matplotlib.cm import get_cmap
import matplotlib.text
import numpy as np

from .helper_classes import Arrival
from .tau_model import TauModel
from .taup_create import TauPCreate
from .taup_path import TauPPath
from .taup_pierce import TauPPierce
from .taup_time import TauPTime
from .taup_geo import calc_dist, add_geo_to_arrivals
import obspy.geodetics.base as geodetics


# Pretty paired colors. Reorder to have saturated colors first and remove
# some colors at the end.
cmap = get_cmap('Paired', lut=12)
COLORS = ['#%02x%02x%02x' % tuple(int(col * 255) for col in cmap(i)[:3])
          for i in range(12)]
COLORS = COLORS[1:][::2][:-1] + COLORS[::2][:-1]


class _SmartPolarText(matplotlib.text.Text):
    """
    Automatically align text on polar plots to be away from axes.

    This class automatically sets the horizontal and vertical alignments
    based on which sides of the spherical axes the text is located.
    """
    def draw(self, renderer, *args, **kwargs):
        fig = self.get_figure()
        midx = fig.get_figwidth() * fig.dpi / 2
        midy = fig.get_figheight() * fig.dpi / 2

        extent = self.get_window_extent(renderer, dpi=fig.dpi)
        points = extent.get_points()

        is_left = points[0, 0] < midx
        is_top = points[0, 1] > midy
        updated = False

        ha = 'right' if is_left else 'left'
        if self.get_horizontalalignment() != ha:
            self.set_horizontalalignment(ha)
            updated = True
        va = 'bottom' if is_top else 'top'
        if self.get_verticalalignment() != va:
            self.set_verticalalignment(va)
            updated = True

        if updated:
            self.update_bbox_position_size(renderer)

        matplotlib.text.Text.draw(self, renderer, *args, **kwargs)


class Arrivals(list):
    """
    List like object of arrivals returned by :class:`TauPyModel` methods.

    :param arrivals: Initial arrivals to store.
    :type arrivals: :class:`list` of
        :class:`~obspy.taup.helper_classes.Arrival`
    :param model: The model used to calculate the arrivals.
    :type model: :class:`~TauPyModel`
    """
    __slots__ = ["model"]

    def __init__(self, arrivals, model):
        super(Arrivals, self).__init__()
        self.model = model
        self.extend(arrivals)

    def __add__(self, other):
        if isinstance(other, Arrival):
            other = Arrivals([other], model=self.model)
        if not isinstance(other, Arrivals):
            raise TypeError
        return self.__class__(super(Arrivals, self).__add__(other),
                              model=self.model)

    def __iadd__(self, other):
        if isinstance(other, Arrival):
            other = Arrivals([other], model=self.model)
        if not isinstance(other, Arrivals):
            raise TypeError
        self.extend(other)
        return self

    def __mul__(self, num):
        if not isinstance(num, int):
            raise TypeError("Integer expected")
        arr = self.copy()
        for _i in range(num - 1):
            arr += self.copy()
        return arr

    def __imul__(self, num):
        if not isinstance(num, int):
            raise TypeError("Integer expected")
        arr = self.copy()
        for _i in range(num - 1):
            self += arr
        return self

    def __setitem__(self, index, arrival):
        if (isinstance(index, slice) and
                all(isinstance(x, Arrival) for x in arrival)):
            super(Arrivals, self).__setitem__(index, arrival)
        elif isinstance(arrival, Arrival):
            super(Arrivals, self).__setitem__(index, arrival)
        else:
            msg = 'Only Arrival objects can be assigned.'
            raise TypeError(msg)

    def __setslice__(self, i, j, seq):
        if all(isinstance(x, Arrival) for x in seq):
            super(Arrivals, self).__setslice__(i, j, seq)
        else:
            msg = 'Only Arrival objects can be assigned.'
            raise TypeError(msg)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(super(Arrivals, self).__getitem__(index),
                                  model=self.model)
        else:
            return super(Arrivals, self).__getitem__(index)

    def __getslice__(self, i, j):
        return self.__class__(super(Arrivals, self).__getslice__(i, j),
                              model=self.model)

    def __str__(self):
        return (
            "{count} arrivals\n\t{arrivals}"
        ).format(
            count=len(self),
            arrivals="\n\t".join([str(_i) for _i in self]))

    def __repr__(self):
        return "[%s]" % (", ".join([repr(_i) for _i in self]))

    def append(self, arrival):
        if isinstance(arrival, Arrival):
            super(Arrivals, self).append(arrival)
        else:
            msg = 'Append only supports a single Arrival object as argument.'
            raise TypeError(msg)

    def copy(self):
        return self.__class__(super(Arrivals, self).copy(),
                              model=self.model)

    def timeplot(self, show=True, fig=None, ax=None):
        """
        Method to plot the travel times of all ray paths in arrivals object.

        :param show: Show the plot.
        :type show: bool
        :param fig: figure instance to plot to. If not given, a new figure
            will be created.
        :type figure: :class:`matplotlib.figure.Figure`
        :param ax: Axes to plot to. If not given, a new figure with an axes
            will be created.
        :type ax: :class:`matplotlib.axes.Axes`

        :returns: The (possibly created) axes instance.
        :rtype: :class:`matplotlib.axes.Axes`

        """
        import matplotlib.pyplot as plt

        if not self:
            raise ValueError("No arrival")

        # create an axis/figure, if there is none, yet:
        if not ax:
            ax = plt.subplot(111)
        if not fig:
            fig = ax.figure

        # extract the time/distance for each phase, and for each receiver:
        for _i, arrival in enumerate(self):
                ax.plot(arrival.distance, arrival.time / 60, '.',
                        label=arrival.name, color=COLORS[_i % len(COLORS)])

        # merge all arrival labels of a certain phase:
        handles, labels = ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]

        ax.legend(handles, labels, loc=2, numpoints=1)

        ax.grid()
        ax.set_xlabel('Distance (degrees)')
        ax.set_ylabel('Time (minutes)')
        if show:
            plt.show()
        return ax

    def plot(self, plot_type="spherical", plot_all=True, legend=True,
             label_arrivals=False, show=True, fig=None, ax=None):
        """
        Method to plot all ray paths in arrivals object.

        :param plot_type: Either ``"spherical"`` or ``"cartesian"``.
            A spherical plot is always global whereas a Cartesian one can
            also be local.
        :type plot_type: str
        :param plot_all: By default all rays, even those travelling in the
            other direction and thus arriving at a distance of *360 - x*
            degrees are shown. Set this to ``False`` to only show rays
            arriving at exactly *x* degrees.
        :type plot_all: bool
        :param legend: If boolean, specify whether or not to show the legend
            (at the default location.) If a str, specify the location of the
            legend. If you are plotting a single phase, you may consider using
            the ``label_arrivals`` argument.
        :type legend: bool or str
        :param label_arrivals: Label the arrivals with their respective phase
            names. This setting is only useful if you are plotting a single
            phase as otherwise the names could be large and possibly overlap
            or clip. Consider using the ``legend`` parameter instead if you
            are plotting multiple phases.
        :type label_arrivals: bool
        :param show: Show the plot.
        :type show: bool
        :param fig: Figure to plot to. If not given, a new figure will be
            created.
        :type fig: :class:`matplotlib.figure.Figure`
        :param ax: Axes to plot to. If not given, a new figure with an axes
            will be created. Must be a polar axes for the spherical plot and
            a regular one for the Cartesian plot.
        :type ax: :class:`matplotlib.axes.Axes`

        :returns: ax
        :rtype: :class:`matplotlib.axes.Axes`
        """
        import matplotlib.pyplot as plt

        if not self:
            raise ValueError("No arrival")
        arrivals = []
        for arrival in self:
            if arrival.path is None:
                continue
            dist = arrival.purist_distance % 360.0
            distance = arrival.distance
            if abs(dist - distance) / dist > 1E-5:
                if plot_all is False:
                    continue
                # Mirror on axis.
                arrival = copy.deepcopy(arrival)
                arrival.path["dist"] *= -1.0
            arrivals.append(arrival)

        discons = self.model.s_mod.v_mod.get_discontinuity_depths()

        if plot_type == "spherical":
            if not ax:
                ax = plt.subplot(111, polar=True)
            if not fig:
                fig = ax.figure

            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_xticks([])
            ax.set_yticks([])

            intp = matplotlib.cbook.simple_linear_interpolation
            radius = self.model.radius_of_planet
            for _i, ray in enumerate(arrivals):
                # Requires interpolation,or diffracted phases look funny.
                ax.plot(intp(ray.path["dist"], 100),
                        radius - intp(ray.path["depth"], 100),
                        color=COLORS[_i % len(COLORS)], label=ray.name,
                        lw=2.0)
            ax.set_yticks(radius - discons)
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())

            if arrivals:
                # Pretty earthquake marker.
                ax.plot([0], [radius - arrivals[0].source_depth],
                        marker="*", color="#FEF215", markersize=20, zorder=10,
                        markeredgewidth=1.5, markeredgecolor="0.3",
                        clip_on=False)

                # Pretty station marker.
                arrowprops = dict(arrowstyle='-|>,head_length=0.8,'
                                  'head_width=0.5',
                                  color='#C95241', lw=1.5)
                station_radius = radius - arrivals[0].receiver_depth
                ax.annotate('',
                            xy=(np.deg2rad(distance), station_radius),
                            xycoords='data',
                            xytext=(np.deg2rad(distance),
                                    station_radius + radius * 0.02),
                            textcoords='data',
                            arrowprops=arrowprops,
                            clip_on=False)
                arrowprops = dict(arrowstyle='-|>,head_length=1.0,'
                                  'head_width=0.6',
                                  color='0.3', lw=1.5, fill=False)
                ax.annotate('',
                            xy=(np.deg2rad(distance), station_radius),
                            xycoords='data',
                            xytext=(np.deg2rad(distance),
                                    station_radius + radius * 0.01),
                            textcoords='data',
                            arrowprops=arrowprops,
                            clip_on=False)
            if label_arrivals:
                name = ','.join(sorted(set(ray.name for ray in arrivals)))
                # We cannot just set the text of the annotations above because
                # it changes the arrow path.
                t = _SmartPolarText(np.deg2rad(distance),
                                    station_radius + radius * 0.07,
                                    name, clip_on=False)
                ax.add_artist(t)

            ax.set_rmax(radius)
            ax.set_rmin(0.0)

            if legend:
                if isinstance(legend, bool):
                    if 0 <= distance <= 180.0:
                        loc = "upper left"
                    else:
                        loc = "upper right"
                else:
                    loc = legend
                ax.legend(loc=loc, prop=dict(size="small"))

        elif plot_type == "cartesian":
            if not ax:
                ax = plt.subplot(111)
                ax.invert_yaxis()
            if not fig:
                fig = ax.figure

            # plot the ray paths:
            for _i, ray in enumerate(arrivals):
                ax.plot(np.rad2deg(ray.path["dist"]), ray.path["depth"],
                        color=COLORS[_i % len(COLORS)], label=ray.name,
                        lw=2.0)

            if arrivals:
                # Pretty station marker.
                ms = 14
                station_marker_transform = matplotlib.transforms.offset_copy(
                    ax.transData,
                    fig=ax.get_figure(),
                    y=ms / 2.0,
                    units="points")
                ax.plot([distance], [arrivals[0].receiver_depth],
                        marker="v", color="#C95241",
                        markersize=ms, zorder=10, markeredgewidth=1.5,
                        markeredgecolor="0.3", clip_on=False,
                        transform=station_marker_transform)
                if label_arrivals:
                    name = ','.join(sorted(set(ray.name for ray in arrivals)))
                    ax.annotate(name,
                                xy=(distance, arrivals[0].receiver_depth),
                                xytext=(0, ms * 1.5),
                                textcoords='offset points',
                                ha='center', annotation_clip=False)

                # Pretty earthquake marker.
                ax.plot([0], [arrivals[0].source_depth], marker="*",
                        color="#FEF215", markersize=20, zorder=10,
                        markeredgewidth=1.5, markeredgecolor="0.3",
                        clip_on=False)

                # lines of major discontinuities:
                x = ax.get_xlim()
                y = ax.get_ylim()
                for depth in discons:
                    if not (y[1] <= depth <= y[0]):
                        continue
                    ax.hlines(depth, x[0], x[1], color="0.5", zorder=-1)

                # Plot some more station markers if necessary.
                possible_distances = [_i * (distance + 360.0)
                                      for _i in range(1, 10)]
                possible_distances += [-_i * (360.0 - distance) for _i in
                                       range(1, 10)]
                possible_distances = [_i for _i in possible_distances
                                      if x[0] <= _i <= x[1]]
                if possible_distances:
                    ax.plot(possible_distances, [arrivals[0].receiver_depth]
                            * len(possible_distances),
                            marker="v", color="#C95241",
                            markersize=ms, zorder=10, markeredgewidth=1.5,
                            markeredgecolor="0.3", clip_on=False, lw=0,
                            transform=station_marker_transform)

            if legend:
                if isinstance(legend, bool):
                    loc = "lower left"
                else:
                    loc = legend
                ax.legend(loc=loc, prop=dict(size="small"))
            ax.set_xlabel("Distance [deg]")
            ax.set_ylabel("Depth [km]")
        else:
            raise NotImplementedError
        if show:
            plt.show()
        return ax


class TauPyModel(object):
    """
    Representation of a seismic model and methods for ray paths through it.
    """

    def __init__(self, model="iasp91", verbose=False, planet_flattening=0.0,
                 cache=None):
        """
        Loads an already created TauPy model.

        :param model: The model name. Either an internal TauPy model or a
            filename in the case of custom models.
        :param planet_flattening: Flattening parameter for the planet's
            ellipsoid (i.e. (a-b)/a, where a is the semimajor equatorial radius
            and b is the semiminor polar radius). A value of 0 (the default)
            gives a spherical planet. Note that this is only used to convert
            from geographical positions (source and receiver latitudes and
            longitudes) to epicentral distances - the actual traveltime and
            raypath calculations are performed on a spherical planet.
        :type planet_flattening: float
        :param cache: An object to use to cache models split at source depths.
            Generating results requires splitting a model at the source depth,
            which may be expensive. The cache allows faster calculation when
            multiple results are requested for the same source depth. The
            dictionary must be ordered, otherwise the LRU cache will not
            behave correctly. If ``False`` is specified, then no cache will be
            used.
        :type cache: :class:`collections.OrderedDict` or bool

        Usage:

        >>> from obspy.taup import tau
        >>> i91 = tau.TauPyModel()
        >>> print(i91.get_travel_times(10, 20)[0].name)
        P
        >>> i91.get_travel_times(10, 20)[0].time  # doctest: +ELLIPSIS
        272.675...
        >>> len(i91.get_travel_times(100, 50, phase_list = ["P", "S"]))
        2
        """
        self.verbose = verbose
        self.model = TauModel.from_file(model, cache=cache)
        self.planet_flattening = planet_flattening

    def get_travel_times(self, source_depth_in_km, distance_in_degree=None,
                         phase_list=("ttall",), receiver_depth_in_km=0.0):
        """
        Return travel times of every given phase.

        :param source_depth_in_km: Source depth in km
        :type source_depth_in_km: float
        :param distance_in_degree: Epicentral distance in degrees.
        :type distance_in_degree: float
        :param phase_list: List of phases for which travel times should be
            calculated. If this is empty, all phases will be used.
        :type phase_list: list of str
        :param receiver_depth_in_km: Receiver depth in km
        :type receiver_depth_in_km: float

        :return: List of ``Arrival`` objects, each of which has the time,
            corresponding phase name, ray parameter, takeoff angle, etc. as
            attributes.
        :rtype: :class:`Arrivals`
        """
        # Accessing the arrivals not just by list indices but by phase name
        # might be useful, but also difficult: several arrivals can have the
        # same phase.
        tt = TauPTime(self.model, phase_list, source_depth_in_km,
                      distance_in_degree, receiver_depth_in_km)
        tt.run()
        return Arrivals(sorted(tt.arrivals, key=lambda x: x.time),
                        model=self.model)

    def get_pierce_points(self, source_depth_in_km, distance_in_degree,
                          phase_list=("ttall",), receiver_depth_in_km=0.0):
        """
        Return pierce points of every given phase.

        :param source_depth_in_km: Source depth in km
        :type source_depth_in_km: float
        :param distance_in_degree: Epicentral distance in degrees.
        :type distance_in_degree: float
        :param phase_list: List of phases for which travel times should be
            calculated. If this is empty, all phases will be used.
        :type phase_list: list of str
        :param receiver_depth_in_km: Receiver depth in km
        :type receiver_depth_in_km: float

        :return: List of ``Arrival`` objects, each of which has the time,
            corresponding phase name, ray parameter, takeoff angle, etc. as
            attributes.
        :rtype: :class:`Arrivals`
        """
        pp = TauPPierce(self.model, phase_list, source_depth_in_km,
                        distance_in_degree, receiver_depth_in_km)
        pp.run()
        return Arrivals(sorted(pp.arrivals, key=lambda x: x.time),
                        model=self.model)

    def get_ray_paths(self, source_depth_in_km, distance_in_degree=None,
                      phase_list=("ttall",), receiver_depth_in_km=0.0):
        """
        Return ray paths of every given phase.

        :param source_depth_in_km: Source depth in km
        :type source_depth_in_km: float
        :param distance_in_degree: Epicentral distance in degrees.
        :type distance_in_degree: float
        :param phase_list: List of phases for which travel times should be
            calculated. If this is empty, all phases will be used.
        :type phase_list: list of str
        :param receiver_depth_in_km: Receiver depth in km
        :type receiver_depth_in_km: float

        :return: List of ``Arrival`` objects, each of which has the time,
            corresponding phase name, ray parameter, takeoff angle, etc. as
            attributes.
        :rtype: :class:`Arrivals`
        """
        rp = TauPPath(self.model, phase_list, source_depth_in_km,
                      distance_in_degree, receiver_depth_in_km)
        rp.run()
        return Arrivals(sorted(rp.arrivals, key=lambda x: x.time),
                        model=self.model)

    def get_travel_times_geo(self, source_depth_in_km, source_latitude_in_deg,
                             source_longitude_in_deg, receiver_latitude_in_deg,
                             receiver_longitude_in_deg, phase_list=("ttall",)):
        """
        Return travel times of every given phase given geographical data.

        .. note::

            Note that the conversion from source and receiver latitudes and
            longitudes to epicentral distances respects the model's flattening
            parameter, so this calculation can be performed for a ellipsoidal
            or spherical planet. However, the actual traveltime and raypath
            calculations are performed on a spherical planet. Ellipticity
            corrections of e.g. [Dziewonski1976]_ are not made.

        :param source_depth_in_km: Source depth in km
        :type source_depth_in_km: float
        :param source_latitude_in_deg: Source latitude in degrees
        :type source_latitude_in_deg: float
        :param source_longitude_in_deg: Source longitude in degrees
        :type source_longitude_in_deg: float
        :param receiver_latitude_in_deg: Receiver latitude in degrees
        :type receiver_latitude_in_deg: float
        :param receiver_longitude_in_deg: Receiver longitude in degrees
        :type receiver_longitude_in_deg: float
        :param phase_list: List of phases for which travel times should be
            calculated. If this is empty, all phases will be used.
        :type phase_list: list of str

        :return: List of ``Arrival`` objects, each of which has the time,
            corresponding phase name, ray parameter, takeoff angle, etc. as
            attributes.
        :rtype: :class:`Arrivals`
        """
        distance_in_deg = calc_dist(source_latitude_in_deg,
                                    source_longitude_in_deg,
                                    receiver_latitude_in_deg,
                                    receiver_longitude_in_deg,
                                    self.model.radius_of_planet,
                                    self.planet_flattening)
        arrivals = self.get_travel_times(source_depth_in_km, distance_in_deg,
                                         phase_list)
        return arrivals

    def get_pierce_points_geo(self, source_depth_in_km, source_latitude_in_deg,
                              source_longitude_in_deg,
                              receiver_latitude_in_deg,
                              receiver_longitude_in_deg,
                              phase_list=("ttall",)):
        """
        Return ray paths of every given phase with geographical info.

        .. note::

            Note that the conversion from source and receiver latitudes and
            longitudes to epicentral distances respects the model's flattening
            parameter, so this calculation can be performed for a ellipsoidal
            or spherical planet. However, the actual traveltime and raypath
            calculations are performed on a spherical planet. Ellipticity
            corrections of e.g. [Dziewonski1976]_ are not made.

        :param source_depth_in_km: Source depth in km
        :type source_depth_in_km: float
        :param source_latitude_in_deg: Source latitude in degrees
        :type source_latitude_in_deg: float
        :param source_longitude_in_deg: Source longitue in degrees
        :type source_longitude_in_deg: float
        :param receiver_latitude_in_deg: Receiver latitude in degrees
        :type receiver_latitude_in_deg: float
        :param receiver_longitude_in_deg: Receiver longitude in degrees
        :type receiver_longitude_in_deg: float
        :param phase_list: List of phases for which travel times should be
            calculated. If this is empty, all phases will be used.
        :type phase_list: list of str
        :return: List of ``Arrival`` objects, each of which has the time,
            corresponding phase name, ray parameter, takeoff angle, etc. as
            attributes.
        :rtype: :class:`Arrivals`
        """
        distance_in_deg = calc_dist(source_latitude_in_deg,
                                    source_longitude_in_deg,
                                    receiver_latitude_in_deg,
                                    receiver_longitude_in_deg,
                                    self.model.radius_of_planet,
                                    self.planet_flattening)

        arrivals = self.get_pierce_points(source_depth_in_km, distance_in_deg,
                                          phase_list)

        if geodetics.HAS_GEOGRAPHICLIB:
            try:
                arrivals = add_geo_to_arrivals(
                    arrivals, source_latitude_in_deg, source_longitude_in_deg,
                    receiver_latitude_in_deg, receiver_longitude_in_deg,
                    self.model.radius_of_planet, self.planet_flattening)
            except ImportError as e:
                warnings.warn("ImportError: " + str(e))
        else:
            msg = "Not able to evaluate positions of pierce points. " + \
                  "Arrivals object will not be modified. " + \
                  "Install the Python module 'geographiclib' to solve " + \
                  "this issue."
            warnings.warn(msg)

        return arrivals

    def get_ray_paths_geo(self, source_depth_in_km, source_latitude_in_deg,
                          source_longitude_in_deg, receiver_latitude_in_deg,
                          receiver_longitude_in_deg, phase_list=("ttall",)):
        """
        Return ray paths of every given phase with geographical info.

        .. note::

            Note that the conversion from source and receiver latitudes and
            longitudes to epicentral distances respects the model's flattening
            parameter, so this calculation can be performed for a ellipsoidal
            or spherical planet. However, the actual traveltime and raypath
            calculations are performed on a spherical planet. Ellipticity
            corrections of e.g. [Dziewonski1976]_ are not made.

        :param source_depth_in_km: Source depth in km
        :type source_depth_in_km: float
        :param source_latitude_in_deg: Source latitude in degrees
        :type source_latitude_in_deg: float
        :param source_longitude_in_deg: Source longitue in degrees
        :type source_longitude_in_deg: float
        :param receiver_latitude_in_deg: Receiver latitude in degrees
        :type receiver_latitude_in_deg: float
        :param receiver_longitude_in_deg: Receiver longitude in degrees
        :type receiver_longitude_in_deg: float
        :param phase_list: List of phases for which travel times should be
            calculated. If this is empty, all phases will be used.
        :type phase_list: list of str
        :return: List of ``Arrival`` objects, each of which has the time,
            corresponding phase name, ray parameter, takeoff angle, etc. as
            attributes.
        :rtype: :class:`Arrivals`
        """
        distance_in_deg = calc_dist(source_latitude_in_deg,
                                    source_longitude_in_deg,
                                    receiver_latitude_in_deg,
                                    receiver_longitude_in_deg,
                                    self.model.radius_of_planet,
                                    self.planet_flattening)

        arrivals = self.get_ray_paths(source_depth_in_km, distance_in_deg,
                                      phase_list)

        if geodetics.HAS_GEOGRAPHICLIB:
            try:
                arrivals = add_geo_to_arrivals(
                    arrivals, source_latitude_in_deg, source_longitude_in_deg,
                    receiver_latitude_in_deg, receiver_longitude_in_deg,
                    self.model.radius_of_planet, self.planet_flattening)
            except ImportError as e:
                warnings.warn("ImportError: " + str(e))
        else:
            msg = "Not able to evaluate positions of points on path. " + \
                  "Arrivals object will not be modified. " + \
                  "Install the Python module 'geographiclib' to solve " + \
                  "this issue."
            warnings.warn(msg)

        return arrivals


def create_taup_model(model_name, output_dir, input_dir):
    """
    Create a .taup model from a .tvel file.

    :param model_name:
    :param output_dir:
    """
    if "." in model_name:
        model_file_name = model_name
    else:
        model_file_name = model_name + ".tvel"
    TauPCreate.main(model_file_name, output_dir, input_dir)


def traveltime_plot(source_depth, min_degree=10, max_degree=90, nrecs=1000,
                    phases=['P', 'S'], model='iasp91', fig=None, ax=None):
        """
        Returns a travel time plot and any created axis instance of this
        plot.

        :param source_depth: Source depth in kilometers.
        :type source_depth: float
        :param min_degree: minimum distance from the source (in degrees) to a
            plot travel times Defaults to ``10``.
        :type min_degree: float
        :param max_degree: maximum distance from the source (in degrees) to
            plot travel times. Defaults to ``90``.
        :type max_degree: float
        :param nrecs: Number of points to plot. Defaults to ``1000``.
        :type nrecs: int
        :param phases: List of phase names which should be used within the
            plot. Defaults to ``['P', 'PKP']``.
        :type phases: list of str, optional
        :param model: string containing the model to use. Defaults to 'iasp91'.
        :type model: str:
        :param fig: Figure to plot into. If not given, a new figure instance
            will be created.
        :type fig: :class:`matplotlib.axes.Axes
        :param ax: Axes to plot into. If not given, a new figure with an axes
            will be created.
        :type ax: :class:`matplotlib.Figure.figure`

        :returns: ax
        :rtype: :class:`matplotlib.axes.Axes`

        .. rubric:: Example

        >>> from obspy.taup import traveltime_plot
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax = traveltime_plot(source_depth=10, ax=ax, fig=fig)
        >>> plt.show()

        .. plot::

        from obspy.taup import traveltime_plot
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax = traveltime_plot(source_depth=10, ax=ax, fig=fig)

        plt.show()

        """

        # compute the requested arrivals:
        model = TauPyModel(model)

        plotted = False  # a flag to check if any arrival exists

        # calculate the arrival times and plot vs. epicentral distance:
        degrees = np.linspace(min_degree, max_degree, nrecs)
        for degree in degrees:
            try:
                arrivals = model.get_ray_paths(source_depth, degree,
                                               phase_list=phases)
                arrivals.timeplot(show=False, ax=ax)
            except ValueError as err:
                print(err)
            else:
                plotted = True

        if not plotted:
            raise ValueError("No travel times to plot")
        return ax


def raypath_plot(source_depth, min_degree=10, max_degree=90, nrecs=20,
                 plot_type='spherical', phases=['P', 'S'], model='iasp91',
                 plot_all=True, legend=False, label_arrivals=False,
                 fig=None, ax=None):
        """
        Returns a ray path plot and any created axis instance of this
        plot.

        :param source_depth: Source depth in kilometers.
        :type source_depth: float
        :param min_degree: minimum distance from the source (in degrees) to a
            plot travel times Defaults to ``10``.
        :type min_degree: float
        :param max_degree: maximum distance from the source (in degrees) to
            plot travel times. Defaults to ``90``.
        :type max_degree: float
        :param nrecs: Number of receivers to plot. Defaults to ``20``.
        :type nrecs: int
        :param plot_type: type of plot to create. Defaults to ``spherical``,
            but can also be ``cartesian``.
        :type plot_type: str
        :param phases: List of phase names which should be used within the
            plot. Defaults to ``['P', 'S']``.
        :type phases: list of str
        :param model: string containing the model to use. Defaults to 'iasp91'.
        :type model: str
        :param plot_all: By default all rays, even those travelling in the
            other direction and thus arriving at a distance of *360 - x*
            degrees are shown. Set this to ``False`` to only show rays
            arriving at exactly *x* degrees.
        :type plot_all: bool
        :param legend: If boolean, specify whether or not to show the legend
            (at the default location.) If a str, specify the location of the
            legend. If you are plotting a single phase, you may consider using
            the ``label_arrivals`` argument.
        :type legend: bool or str
        :param label_arrivals: Label the arrivals with their respective phase
            names. This setting is only useful if you are plotting a single
            phase as otherwise the names could be large and possibly overlap
            or clip. Consider using the ``legend`` parameter instead if you
            are plotting multiple phases.
        :type label_arrivals: bool
        :param fig: Figure to plot into. If not given, a new figure instance
            will be created.
        :type fig: :class:`matplotlib.axes.Axes
        :param ax: Axes to plot to. If not given, a new figure with an axes
            will be created.
        :type ax: :class:`matplotlib.axes.Axes`

        :returns: ax
        :rtype: :class:`matplotlib.axes.Axes`

        .. rubric:: Example

        >>> from obspy.taup import raypath_plot
        >>> import matplotlib.pyplot as plt
        >>> ax = plt.subplot(111, polar=True)
        >>> fig = ax.figure
        >>> ax = raypath_plot(source_depth=100, ax=ax, fig=fig)
        >>> plt.show()

        .. plot::

        from obspy.taup.tau import raypath_plot
        import matplotlib.pyplot as plt

        ax = plt.subplot(111, polar=True)
        fig = ax.figure

        ax = raypath_plot(source_depth=100, ax=ax, fig=fig)

        plt.show()

        """

        plotted = False  # a flag to check if any arrival exists

        # compute the requested arrivals:
        model = TauPyModel(model)

        # calculate the arrival times and plot vs. epicentral distance:
        degrees = np.linspace(min_degree, max_degree, nrecs)
        for degree in degrees:
            try:
                arrivals = model.get_ray_paths(source_depth, degree,
                                               phase_list=phases)
                arrivals.plot(show=False, ax=ax, plot_type=plot_type,
                              plot_all=plot_all,
                              legend=legend)
            except ValueError as err:
                print(err)
            else:
                plotted = True

        if not plotted:
            raise ValueError("No ray paths to plot")
        return ax
