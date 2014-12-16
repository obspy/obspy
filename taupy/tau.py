#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import inspect
import os

from .TauModelLoader import load
from .TauP_Time import TauP_Time
from .TauP_Pierce import TauP_Pierce
from .TauP_Path import TauP_Path
from .TauP_Create import TauP_Create


class Arrivals(list):
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
    >>> i91 = tau.TauPyModel()
    >>> tt = i91.get_travel_time(10, 20, ["P, S"])
    """

    def __init__(self, model="iasp91", verbose=False):
        taup_model_path = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data", "taup_models")
        try:
            self.model = load(model, taup_model_path, verbose=verbose)
        except FileNotFoundError:
            print("A {}.taup model file was not found in the taup_models"
                  "directory, will try to create one. "
                  "This may take a while.".format(model))
            self.create_taup_model(model, taup_model_path)
            self.model = load(model, taup_model_path, verbose=verbose)
        self.verbose = verbose

    def get_travel_time(self, source_depth_in_km, distance_in_degree,
                        phase_list=None, print_output=False):
        phase_list = phase_list if phase_list is not None else ["ttall"]
        tt = TauP_Time(phase_list, self.model.sMod.vMod.modelName,
                       source_depth_in_km, distance_in_degree)
        tt.run(print_output)
        return Arrivals(tt.arrivals)

    def get_pierce_points(self, source_depth_in_km, distance_in_degree,
                          phase_list=None, print_output=False):
        phase_list = phase_list if phase_list is not None else ["ttall"]
        pp = TauP_Pierce(phase_list, self.model.sMod.vMod.modelName,
                         source_depth_in_km, distance_in_degree)
        pp.run(print_output)
        # todo: shouldn't this maybe only return the pierce points?
        return Arrivals(pp.arrivals)

    def get_ray_paths(self, source_depth_in_km, distance_in_degree,
                      phase_list=None, print_output=False):
        phase_list = phase_list if phase_list is not None else ["ttall"]
        rp = TauP_Path(phase_list, self.model.sMod.vMod.modelName,
                       source_depth_in_km, distance_in_degree)
        rp.run(print_output)
        return Arrivals(rp.arrivals)

    @staticmethod
    def create_taup_model(model_name, output_dir):
        if "." in model_name:
            model_file_name = model_name
        else:
            model_file_name = model_name + ".tvel"
        TauP_Create.main(model_file_name, output_dir)
