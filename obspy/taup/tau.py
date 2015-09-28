#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
High-level interface to travel-time calculation routines.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import copy

import matplotlib.cbook
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np

from .tau_model import TauModel
from .taup_create import TauP_Create
from .taup_path import TauP_Path
from .taup_pierce import TauP_Pierce
from .taup_time import TauP_Time


# Pretty paired colors. Reorder to have saturated colors first and remove
# some colors at the end.
cmap = plt.get_cmap('Paired', lut=12)
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
    List of arrivals returned by :class:`TauPyModel` methods.

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

    def __str__(self):
        return (
            "{count} arrivals\n\t{arrivals}"
        ).format(
            count=len(self),
            arrivals="\n\t".join([str(_i) for _i in self]))

    def __repr__(self):
        return "[%s]" % (", ".join([repr(_i) for _i in self]))

    def plot(self, plot_type="spherical", plot_all=True, legend=True,
             label_arrivals=False, ax=None, show=True):
        """
        Plot the ray paths if any have been calculated.

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
        :param ax: Axes to plot to. If not given, a new figure with an axes
            will be created. Must be a polar axes for the spherical plot and
            a regular one for the Cartesian plot.
        :type ax: :class:`matplotlib.axes.Axes`
        :param show: Show the plot.
        :type show: bool

        :returns: The (possibly created) axes instance.
        :rtype: :class:`matplotlib.axes.Axes`
        """
        arrivals = []
        for _i in self:
            if _i.path is None:
                continue
            dist = _i.purist_distance % 360.0
            distance = _i.distance
            if abs(dist - distance) / dist > 1E-5:
                if plot_all is False:
                    continue
                # Mirror on axis.
                _i = copy.deepcopy(_i)
                _i.path["dist"] *= -1.0
            arrivals.append(_i)
        if not arrivals:
            raise ValueError("Can only plot arrivals with calculated ray "
                             "paths.")
        discons = self.model.sMod.vMod.getDisconDepths()
        if plot_type == "spherical":
            if not ax:
                plt.figure(figsize=(10, 10))
                ax = plt.subplot(111, polar=True)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_xticks([])
            ax.set_yticks([])
            intp = matplotlib.cbook.simple_linear_interpolation
            radius = self.model.radiusOfEarth
            for _i, ray in enumerate(arrivals):
                # Requires interpolation otherwise diffracted phases look
                # funny.
                ax.plot(intp(ray.path["dist"], 100),
                        radius - intp(ray.path["depth"], 100),
                        color=COLORS[_i % len(COLORS)], label=ray.name,
                        lw=2.0)
            ax.set_yticks(radius - discons)
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            # Pretty earthquake marker.
            ax.plot([0], [radius - arrivals[0].source_depth],
                    marker="*", color="#FEF215", markersize=20, zorder=10,
                    markeredgewidth=1.5, markeredgecolor="0.3", clip_on=False)

            # Pretty station marker.
            arrowprops = dict(arrowstyle='-|>,head_length=0.8,head_width=0.5',
                              color='#C95241',
                              lw=1.5)
            ax.annotate('',
                        xy=(np.deg2rad(distance), radius),
                        xycoords='data',
                        xytext=(np.deg2rad(distance), radius * 1.02),
                        textcoords='data',
                        arrowprops=arrowprops,
                        clip_on=False)
            arrowprops = dict(arrowstyle='-|>,head_length=1.0,head_width=0.6',
                              color='0.3',
                              lw=1.5,
                              fill=False)
            ax.annotate('',
                        xy=(np.deg2rad(distance), radius),
                        xycoords='data',
                        xytext=(np.deg2rad(distance), radius * 1.01),
                        textcoords='data',
                        arrowprops=arrowprops,
                        clip_on=False)
            if label_arrivals:
                name = ','.join(sorted(set(ray.name for ray in arrivals)))
                # We cannot just set the text of the annotations above because
                # it changes the arrow path.
                t = _SmartPolarText(np.deg2rad(distance), radius * 1.07,
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
                plt.legend(loc=loc, prop=dict(size="small"))

        elif plot_type == "cartesian":
            if not ax:
                plt.figure(figsize=(12, 8))
                ax = plt.gca()
            ax.invert_yaxis()
            for _i, ray in enumerate(arrivals):
                ax.plot(np.rad2deg(ray.path["dist"]), ray.path["depth"],
                        color=COLORS[_i % len(COLORS)], label=ray.name,
                        lw=2.0)
            ax.set_ylabel("Depth [km]")
            if legend:
                if isinstance(legend, bool):
                    loc = "lower left"
                else:
                    loc = legend
                ax.legend(loc=loc, prop=dict(size="small"))
            ax.set_xlabel("Distance [deg]")
            # Pretty station marker.
            ms = 14
            station_marker_transform = matplotlib.transforms.offset_copy(
                ax.transData,
                fig=ax.get_figure(),
                y=ms / 2.0,
                units="points")
            ax.plot([distance], [0.0],
                    marker="v", color="#C95241",
                    markersize=ms, zorder=10, markeredgewidth=1.5,
                    markeredgecolor="0.3", clip_on=False,
                    transform=station_marker_transform)
            if label_arrivals:
                name = ','.join(sorted(set(ray.name for ray in arrivals)))
                ax.annotate(name, xy=(distance, 0.0),
                            xytext=(0, ms * 1.5), textcoords='offset points',
                            ha='center', annotation_clip=False)

            # Pretty earthquake marker.
            ax.plot([0], [arrivals[0].source_depth],
                    marker="*", color="#FEF215", markersize=20, zorder=10,
                    markeredgewidth=1.5, markeredgecolor="0.3", clip_on=False)
            x = ax.get_xlim()
            x_range = x[1] - x[0]
            ax.set_xlim(x[0] - x_range * 0.1, x[1] + x_range * 0.1)
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
                ax.plot(possible_distances,  [0.0] * len(possible_distances),
                        marker="v", color="#C95241",
                        markersize=ms, zorder=10, markeredgewidth=1.5,
                        markeredgecolor="0.3", clip_on=False, lw=0,
                        transform=station_marker_transform)

        else:
            raise NotImplementedError
        if show:
            plt.show()
        return ax


class TauPyModel(object):
    """
    Representation of a seismic model and methods for ray paths through it.
    """

    def __init__(self, model="iasp91", verbose=False):
        """
        Loads an already created TauPy model.

        :param model: The model name. Either an internal TauPy model or a
            filename in the case of custom models.

        Usage:

        >>> from obspy.taup import tau
        >>> i91 = tau.TauPyModel()
        >>> print(i91.get_travel_times(10, 20)[0].name)
        P
        >>> i91.get_travel_times(10, 20)[0].time  # doctest: +ELLIPSIS
        272.667...
        >>> len(i91.get_travel_times(100, 50, phase_list = ["P", "S"]))
        2
        """
        self.verbose = verbose
        self.model = TauModel.from_file(model)

    def get_travel_times(self, source_depth_in_km, distance_in_degree=None,
                         phase_list=("ttall",)):
        """
        Return travel times of every given phase.

        :param source_depth_in_km: Source depth in km
        :type source_depth_in_km: float
        :param distance_in_degree: Epicentral distance in degrees.
        :type distance_in_degree: float
        :param phase_list: List of phases for which travel times should be
            calculated. If this is empty, all phases will be used.
        :type phase_list: list of str

        :return: List of ``Arrival`` objects, each of which has the time,
            corresponding phase name, ray parameter, takeoff angle, etc. as
            attributes.
        :rtype: :class:`Arrivals`
        """
        # Accessing the arrivals not just by list indices but by phase name
        # might be useful, but also difficult: several arrivals can have the
        # same phase.
        tt = TauP_Time(self.model, phase_list, source_depth_in_km,
                       distance_in_degree)
        tt.run()
        return Arrivals(sorted(tt.arrivals, key=lambda x: x.time),
                        model=self.model)

    def get_pierce_points(self, source_depth_in_km, distance_in_degree,
                          phase_list=("ttall",)):
        """
        Return pierce points of every given phase.

        :param source_depth_in_km: Source depth in km
        :type source_depth_in_km: float
        :param distance_in_degree: Epicentral distance in degrees.
        :type distance_in_degree: float
        :param phase_list: List of phases for which travel times should be
            calculated. If this is empty, all phases will be used.
        :type phase_list: list of str

        :return: List of ``Arrival`` objects, each of which has the time,
            corresponding phase name, ray parameter, takeoff angle, etc. as
            attributes.
        :rtype: :class:`Arrivals`
        """
        pp = TauP_Pierce(self.model, phase_list, source_depth_in_km,
                         distance_in_degree)
        pp.run()
        return Arrivals(sorted(pp.arrivals, key=lambda x: x.time),
                        model=self.model)

    def get_ray_paths(self, source_depth_in_km, distance_in_degree=None,
                      phase_list=("ttall",)):
        """
        Return ray paths of every given phase.

        :param source_depth_in_km: Source depth in km
        :type source_depth_in_km: float
        :param distance_in_degree: Epicentral distance in degrees.
        :type distance_in_degree: float
        :param phase_list: List of phases for which travel times should be
            calculated. If this is empty, all phases will be used.
        :type phase_list: list of str

        :return: List of ``Arrival`` objects, each of which has the time,
            corresponding phase name, ray parameter, takeoff angle, etc. as
            attributes.
        :rtype: :class:`Arrivals`
        """
        rp = TauP_Path(self.model, phase_list, source_depth_in_km,
                       distance_in_degree)
        rp.run()
        return Arrivals(sorted(rp.arrivals, key=lambda x: x.time),
                        model=self.model)


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
    TauP_Create.main(model_file_name, output_dir, input_dir)
