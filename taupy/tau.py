#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *
import inspect
import os

from .TauModelLoader import load
from .TauP_Time import TauP_Time
from .TauP_Pierce import TauP_Pierce
from .TauP_Path import TauP_Path
from .TauP_Create import TauP_Create


class Arrivals(list):
    """Class that is returned by the interface methods."""
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
    """
    High level interface to TauPy.
    Example usage:
    >>> from taupy import tau
    >>> i91 = tau.TauPyModel("iasp91")
    >>> tt = i91.get_travel_timess(10, 20, ["P, S"])
    """

    def __init__(self, model="iasp91", verbose=False, taup_model_path=None,
                 velocity_model_path=None):
        """
        Loads or creates a tau model object.
        At the moment the models are by default read from and stored in
        [python script location]/TauPy/taupy/data/.
        :param model: The name of the velocity model which should be used to
            create the tau model or which should be loaded if one has been
            created before.
        :param taup_model_path: Set the path for .taup models here, then it
            will be used for model creation and the get... commands.
        :param velocity_model_path: Set the path to the .tvel input velocity
            files which are to be used in creating the .taup models here.

        Usage:
        >>> from taupy import tau
        >>> i91 = tau.TauPyModel()
        >>> i91.get_travel_timess(10, 20)[0].name
        'Pn'
        >>> i91.get_travel_timess(10, 20)[0].time
        15.60164282924581
        >>> i91.get_travel_timess(100, 50, phase_list = ["P", "S"],
        ...                 print_output=True)

        Model: iasp91
        Distance   Depth   PhaseTravel    Ray Param   Takeoff  Incident  Purist     Purist
           (deg)    (km)   Name Time (s)  p (s/deg)     (deg)     (deg)  Distance   Name
        --------------------------------------------------------------------------------
           50.00   100.0   P     523.92      7.563    33.79     23.23     50.00  = P
           50.00   100.0   S     947.65     13.903    34.80     24.84     50.00  = S

        >>> i91.get_travel_timess(10, phase_list = ["ttall"], coordinate_list =
        ...                     [13,14,50,200], print_output=True)
        """

        # If needed, change where to look for models here in this
        # section.
        # NB the currentframe here is the location of this script!
        default_taup_model_path = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data", "taup_models")
        default_velocity_model_path = os.path.join(os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))), "data")
        if taup_model_path is not None:
            self.taup_model_path = taup_model_path
        else:
            self.taup_model_path = default_taup_model_path
        if velocity_model_path is not None:
            self.velocity_model_path = velocity_model_path
        else:
            self.velocity_model_path = default_velocity_model_path

        # Load or create a .taup model:
        if model.endswith(".tvel"):
            model = model[:-5]
        try:
            self.model = load(model, self.taup_model_path, verbose=verbose)
        except FileNotFoundError:
            print("A {}.taup model file was not found in the {} "
                  "directory, will try to create one. "
                  "This may take a while.".format(model, self.taup_model_path))
            create_taup_model(model, self.taup_model_path,
                              self.velocity_model_path)
            self.model = load(model, self.taup_model_path, verbose=verbose)
        self.verbose = verbose

    def get_travel_times(self, source_depth_in_km, distance_in_degree=None,
                         phase_list=None, coordinate_list=None,
                         print_output=False):
        """
        Returns travel times of every given phase.
        :param source_depth_in_km: Depth of wave path source.
        :param distance_in_degree: Distance between the source and receiver in
            degrees. If this is not given, coordinate_list must be specified.
        :param phase_list: List of phases for which travel times should be
            calculated. If this is empty, all phases will be used ("ttall").
        :param print_output: Whether to print the traveltimes for all phases
            to the screen.
        :param coordinate_list: List of source latitude, source longitude,
            receiver latitude, receiver longitude. Used only to calulate the
            distance in degrees.
        :return Arrivals:  List of 'arrival' objects, each of which has the
            time, corresponding phase name, ray parameter, takeoff angle etc
            as attributes.
        """
        # Accessing the arrivals not just by list indices but by phase name
        # might be useful, but also difficult: several arrivals can have the
        # same phase.
        phase_list = phase_list if phase_list is not None else ["ttall"]
        tt = TauP_Time(phase_list, self.model.sMod.vMod.modelName,
                       source_depth_in_km, distance_in_degree, coordinate_list,
                       self.taup_model_path)
        tt.run(print_output)
        if print_output:
            return
        return Arrivals(tt.arrivals)

    def get_pierce_points(self, source_depth_in_km, distance_in_degree=None,
                          phase_list=None, coordinate_list=None,
                          print_output=False):
        phase_list = phase_list if phase_list is not None else ["ttall"]
        pp = TauP_Pierce(phase_list, self.model.sMod.vMod.modelName,
                         source_depth_in_km, distance_in_degree,
                         coordinate_list, self.taup_model_path)
        pp.run(print_output)
        if print_output:
            return
        return Arrivals(pp.arrivals)

    def get_ray_paths(self, source_depth_in_km, distance_in_degree=None,
                      phase_list=None, coordinate_list=None,
                      print_output=False):
        phase_list = phase_list if phase_list is not None else ["ttall"]
        rp = TauP_Path(phase_list, self.model.sMod.vMod.modelName,
                       source_depth_in_km, distance_in_degree,
                       coordinate_list, self.taup_model_path)
        rp.run(print_output)
        if print_output:
            return
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
