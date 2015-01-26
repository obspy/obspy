#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from math import pi

from .taup_time import TauP_Time


class TauP_Pierce(TauP_Time):
    """
    The methods here allow using TauP_Time to calculate the pierce points
    relating to the different arrivals.
    """

    def __init__(self, model, phase_list, depth, degrees):
        super().__init__(model=model, phase_list=phase_list, depth=depth,
                         degrees=degrees)
        self.onlyTurnPoints = False
        self.onlyRevPoints = False
        self.onlyUnderPoints = False
        self.onlyAddPoints = False
        self.addDepth = []
        self.phaseList = phase_list
        self.depth = depth
        self.degrees = degrees

    def depthCorrect(self, depth):
        """
        Override TauP_Time.depthCorrect so that the pierce points may be added.
        :param depth:
        :return:
        """
        tModOrig = self.model
        mustRecalc = False
        # First check if tModDepth is correct as it is. Check to make sure
        # source depth is the same, and then check to make sure each addDepth
        # is in the model.
        if self.tModDepth.source_depth == depth:
            if self.addDepth:
                branchDepths = self.tModDepth.getBranchDepths()
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
            # Won't actually do anything much since tModDepth != None.
            TauP_Time.depthCorrect(self, depth)
        else:
            self.tModDepth = None
            if self.addDepth is not None:
                for addDepth in self.addDepth:
                    self.model = self.model.splitBranch(addDepth)
            TauP_Time.depthCorrect(self, depth)
            self.model = tModOrig

    def calculate(self, degrees):
        """
        Call all the necessary calculations to obtain the pierce points.
        """
        self.depthCorrect(self.source_depth)
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
            phaseArrivals = phase.calc_pierce(degrees)
            self.arrivals += phaseArrivals

    def printResult(self):
        for currArrival in self.arrivals:
            print(self.getCommentLine(currArrival))
            longWayRound = False
            if (currArrival.dist * 180 / pi) % 360 > 180:
                longWayRound = True
            prevDepth = currArrival.pierce[0].depth
            for j, calcDist in enumerate(
                    [p.get_dist_deg() for p in currArrival.pierce]):
                if longWayRound is True and calcDist != 0:
                    calcDist *= -1
                if j < len(currArrival.pierce) - 1:
                    nextDepth = currArrival.pierce[j + 1].depth
                else:
                    nextDepth = currArrival.pierce[j].depth
                capd = currArrival.pierce[j].depth
                # Beautifully hand-formatted code:
                if ((not any([self.onlyTurnPoints, self.onlyRevPoints,
                              self.onlyUnderPoints, self.onlyAddPoints])) or (
                    self.onlyAddPoints and capd in self.addDepth) or (
                    self.onlyRevPoints
                        and ((prevDepth - capd) * (capd - nextDepth) < 0)) or (
                    self.onlyTurnPoints and j != 0
                        and ((prevDepth-capd) <= 0 <= (capd-nextDepth))) or (
                    self.onlyUnderPoints
                        and ((prevDepth - capd) >= 0 >= (capd - nextDepth)))):
                    print("{:>7.2f} {:>7.1f} {:>7.1f}".format(calcDist,
                          currArrival.pierce[j].depth,
                          currArrival.pierce[j].time))
                    # Optional (only if used in calc?) coords output to follow.

    def getCommentLine(self, currArrival):
        outName = currArrival.name
        if not currArrival.name == currArrival.puristName:
            outName += "(" + currArrival.puristName + ")"
        return ("> " + outName + " at "
                + " {:.2f} seconds at ".format(currArrival.time)
                + " {:.2f} degrees for a ".format(currArrival.get_dist_deg())
                + " {} km deep source in the ".format(currArrival.source_depth)
                + " {} model with ray_param {:.3f} s/deg.".format(
                    self.modelName, currArrival.ray_param * pi / 180))


if __name__ == '__main__':
    # Permits running as script.
    tauPPierce = TauP_Pierce()
    tauPPierce.readcmdLineArgs()
    tauPPierce.run(printOutput=True)
