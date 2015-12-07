#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pierce point calculations.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .taup_time import TauP_Time


class TauP_Pierce(TauP_Time):
    """
    The methods here allow using TauP_Time to calculate the pierce points
    relating to the different arrivals.
    """
    def __init__(self, model, phase_list, depth, degrees, receiver_depth=0.0):
        super().__init__(model=model, phase_list=phase_list, depth=depth,
                         degrees=degrees, receiver_depth=receiver_depth)
        self.onlyTurnPoints = False
        self.onlyRevPoints = False
        self.onlyUnderPoints = False
        self.onlyAddPoints = False
        self.addDepth = []

    def depth_correct(self, depth, receiver_depth=None):
        """
        Override TauP_Time.depth_correct so that the pierce points may be
        added.
        """
        tModOrig = self.model
        mustRecalc = False
        # First check if depth_corrected_model is correct as it is. Check to
        # make sure source depth is the same, and then check to make sure
        # each addDepth is in the model.
        if self.depth_corrected_model.source_depth == depth:
            if self.addDepth:
                branchDepths = self.depth_corrected_model.getBranchDepths()
                for addDepth in self.addDepth:
                    for branchDepth in branchDepths:
                        if addDepth == branchDepth:
                            # Found it, so break and go to the next addDepth.
                            break
                        # Didn't find the depth as a branch, so must
                        # recalculate.
                        mustRecalc = True
                    if mustRecalc:
                        break
        else:
            # The depth isn't event the same, so must recalculate
            mustRecalc = True
        if not mustRecalc:
            # Won't actually do anything much since depth_corrected_model !=
            #  None.
            TauP_Time.depth_correct(self, depth, receiver_depth)
        else:
            self.depth_corrected_model = None
            if self.addDepth is not None:
                for addDepth in self.addDepth:
                    self.model = self.model.splitBranch(addDepth)
            TauP_Time.depth_correct(self, depth, receiver_depth)
            self.model = tModOrig

    def calculate(self, degrees):
        """
        Call all the necessary calculations to obtain the pierce points.
        """
        self.depth_correct(self.source_depth, self.receiver_depth)
        self.recalc_phases()
        self.arrivals = []
        self.calcPierce(degrees)

    def calcPierce(self, degrees):
        """
        Calculates the pierce points for phases at the given distance by
        calling the calcPierce method of the SeismicPhase class. The results
        are then in self.arrivals.
        """
        for phase in self.phases:
            self.arrivals += phase.calc_pierce(degrees)
