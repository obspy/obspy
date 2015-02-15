#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .tau_model import TauModel
from .taup_create import TauP_Create
from .taup_path import TauP_Path
from .taup_pierce import TauP_Pierce
from .taup_time import TauP_Time


class Arrivals(list):
    """
    Class that is returned by the interface methods.
    """
    __slots__ = ["arrivals"]

    def __init__(self, arrivals):
        super(Arrivals, self).__init__()
        self.extend(arrivals)

    def __str__(self):
        return (
            "{count} arrivals\n\t{arrivals}"
        ).format(
            count=len(self),
            arrivals="\n\t".join([str(_i) for _i in self]))

    def __repr__(self):
        return "[%s]" % (", ".join([repr(_i) for _i in self]))


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
        return Arrivals(tt.arrivals)

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
        return Arrivals(pp.arrivals)

    def get_ray_paths(self, source_depth_in_km, distance_in_degree=None,
                      phase_list=("ttall,")):
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
        return Arrivals(rp.arrivals)


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
