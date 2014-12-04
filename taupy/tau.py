#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .TauModelLoader import load
from .TauP_Time import TauP_Time, parsePhaseList


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
    """
    def __init__(self, model="iasp91", verbose=False):
        self.model = load(model, ".", verbose=verbose)
        self.verbose = verbose

    # def get_travel_time(self, source_depth_in_km, distance_in_degree,
    #                     phase_list=None):
    #     phase_list = phase_list if phase_list is not None else ["ttall"]
    #     phase_list = parsePhaseList(phase_list)
    #
    #     tt = TauP_Time()
    #     tt.DEBUG = self.verbose
    #     tt.phaseNames = phase_list
    #     tt.phaseFile = None
    #     tt.depth = float(source_depth_in_km)
    #     tt.degrees = float(distance_in_degree)
    #     tt.kilometres = None
    #
    #     tt.tMod = self.model
    #     tt.tModDepth = self.model
    #     tt.modelName = self.model.sMod.vMod.modelName
    #
    #     tt.depthCorrect(source_depth_in_km)
    #     tt.calculate(distance_in_degree)
    #
    #     return Arrivals(tt.arrivals)

    def get_travel_time(self, source_depth_in_km, distance_in_degree,
                         phase_list=None, print_output=False):
        phase_list = phase_list if phase_list is not None else ["ttall"]
        tt = TauP_Time(phase_list, self.model.sMod.vMod.modelName,
                       source_depth_in_km, distance_in_degree)
        tt.run(print_output)
        return Arrivals(tt.arrivals)