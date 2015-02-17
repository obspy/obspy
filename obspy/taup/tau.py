#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import matplotlib.pyplot as plt
import numpy as np

from .tau_model import TauModel
from .taup_create import TauP_Create
from .taup_path import TauP_Path
from .taup_pierce import TauP_Pierce
from .taup_time import TauP_Time


# From colorbrewer2.org
COLORS = [
    "#1f78b4",
    "#33a02c",
    "#e31a1c",
    "#ff7f00",
    "#6a3d9a",
    "#a6cee3",
    "#b2df8a",
    "#fb9a99",
    "#fdbf6f",
    "#cab2d6"]


class Arrivals(list):
    """
    List of arrivals returned by the
    methods of the :class:`~obspy.taup.tau.TauPyModel` class.

    :param arrivals: List of arrivals.
    :param model: The model used to calculate the arrivals.
    :param distance: The requested distance.
    """
    __slots__ = ["model", "distance"]

    def __init__(self, arrivals, model, distance):
        super(Arrivals, self).__init__()
        self.model = model
        self.distance = distance
        self.extend(arrivals)

    def __str__(self):
        return (
            "{count} arrivals\n\t{arrivals}"
        ).format(
            count=len(self),
            arrivals="\n\t".join([str(_i) for _i in self]))

    def __repr__(self):
        return "[%s]" % (", ".join([repr(_i) for _i in self]))

    def plot(self, plot_type="spherical", plot_all=False):
        """
        Plot the raypaths if any.

        :param plot_type: Either ``"spherical"`` or ``"cartesian"``.
            A spherical plot is always global whereas a cartesian one can
            also be local.
        :type plot_type: str
        :param plot_all: By default only rays that match the requested
            distance are plotted and rays arriving at 360 - x degrees are
            not shown.
        :type plot_all: bool
        """
        arrivals = []
        for _i in self:
            if _i.path is None:
                continue
            if plot_all is False:
                dist = np.rad2deg(_i.dist)
                if (dist - self.distance) / dist > 1E-5:
                    continue
            arrivals.append(_i)
        if not arrivals:
            raise ValueError("Can only plot arrivals with calculated ray "
                             "paths.")
        discons = self.model.sMod.vMod.getDisconDepths()
        if plot_type == "spherical":
            ax = plt.subplot(111, polar=True)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_xticks([])
            ax.set_yticks([])
            for _i, ray in enumerate(arrivals):
                ax.plot(ray.path["dist"], 6371.0 - ray.path["depth"],
                        color=COLORS[_i % len(COLORS)], label=ray.name,
                        lw=1.5)
            ax.set_xticks(np.pi / 180.0 * np.linspace(180, -180, 8,
                                                      endpoint=False))
            ax.set_yticks(6371 - discons)
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            ax.set_rmax(6371.0)
            ax.set_rmin(0.0)
            plt.legend(loc="upper left")
        elif plot_type == "cartesian":
            plt.gca().invert_yaxis()
            for _i, ray in enumerate(arrivals):
                plt.plot(np.rad2deg(ray.path["dist"]), ray.path["depth"],
                         color=COLORS[_i % len(COLORS)], label=ray.name,
                         lw=1.5)
            plt.ylabel("Depth [km]")
            plt.legend()
            plt.xlabel("Distance [deg]")
            x = plt.xlim()
            y = plt.ylim()
            for depth in discons:
                if not (y[1] <= depth <= y[0]):
                    continue
                plt.hlines(depth, x[0], x[1], color="0.5", zorder=-1)
        else:
            raise NotImplementedError
        plt.show()


class TauPyModel(object):
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
        self.model = TauModel.fromfile(model)

    def get_travel_times(self, source_depth_in_km, distance_in_degree=None,
                         phase_list=("ttall",)):
        """
        Returns travel times of every given phase.

        :param source_depth_in_km: Source depth in km
        :type source_depth_in_km: float
        :param distance_in_degree: Epicentral distance in degrees.
        :type distance_in_degree: float
        :param phase_list: List of phases for which travel times should be
            calculated. If this is empty, all phases will be used.
        :type phase_list: list of str

        :return Arrivals:  List of ``Arrival`` objects, each of which has the
            time, corresponding phase name, ray parameter, takeoff angle, etc.
            as attributes.
        """
        # Accessing the arrivals not just by list indices but by phase name
        # might be useful, but also difficult: several arrivals can have the
        # same phase.
        tt = TauP_Time(self.model, phase_list, source_depth_in_km,
                       distance_in_degree)
        tt.run()
        return Arrivals(sorted(tt.arrivals, key=lambda x: x.time),
                        model=self.model, distance=distance_in_degree)

    def get_pierce_points(self, source_depth_in_km, distance_in_degree,
                          phase_list=("ttall",)):
        """
        Returns pierce points of every given phase.

        :param source_depth_in_km: Source depth in km
        :type source_depth_in_km: float
        :param distance_in_degree: Epicentral distance in degrees.
        :type distance_in_degree: float
        :param phase_list: List of phases for which travel times should be
            calculated. If this is empty, all phases will be used.
        :type phase_list: list of str

        :return Arrivals:  List of ``Arrival`` objects, each of which has the
            time, corresponding phase name, ray parameter, takeoff angle, etc.
            as attributes.
        """
        pp = TauP_Pierce(self.model, phase_list, source_depth_in_km,
                         distance_in_degree)
        pp.run()
        return Arrivals(sorted(pp.arrivals, key=lambda x: x.time),
                        model=self.model, distance=distance_in_degree)

    def get_ray_paths(self, source_depth_in_km, distance_in_degree=None,
                      phase_list=("ttall",)):
        """
        Returns ray paths of every given phase.

        :param source_depth_in_km: Source depth in km
        :type source_depth_in_km: float
        :param distance_in_degree: Epicentral distance in degrees.
        :type distance_in_degree: float
        :param phase_list: List of phases for which travel times should be
            calculated. If this is empty, all phases will be used.
        :type phase_list: list of str

        :return Arrivals:  List of ``Arrival`` objects, each of which has the
            time, corresponding phase name, ray parameter, takeoff angle, etc.
            as attributes.
        """
        rp = TauP_Path(self.model, phase_list, source_depth_in_km,
                       distance_in_degree)
        rp.run()
        return Arrivals(sorted(rp.arrivals, key=lambda x: x.time),
                        model=self.model, distance=distance_in_degree)


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
