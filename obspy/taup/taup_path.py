#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import math
import numpy as np

from .taup_pierce import TauP_Pierce
from .helper_classes import TimeDist


class TauP_Path(TauP_Pierce):
    """
    Calcualate the ray paths for each phase, using TauP_Pierce and TauP_Time.
    """
    def __init__(self, model, phase_list, depth, degrees):
        super().__init__(model=model, phase_list=phase_list, depth=depth,
                         degrees=degrees)
        self.maxPathTime = 1e300
        self.maxPathInc = 1
        self.phaseList = phase_list
        self.depth = float(depth)
        self.degrees = float(degrees)

    def calculate(self, degrees):
        """
        Call all the necessary calculations to obtain the ray paths.
        """
        self.depthCorrect(self.source_depth)
        self.recalcPhases()
        self.arrivals = []
        self.calcPath(degrees)

    def calcPath(self, degrees):
        """
        Calculates the ray paths for phases at the given distance by
        calling the calcPath method of the SeismicPhase class. The results
        are then in self.arrivals.
        """
        self.degrees = degrees
        for phase in self.phases:
            phaseArrivals = phase.calc_path(degrees)
            self.arrivals += phaseArrivals

    def printResult(self):
        radiusOfEarth = self.tModDepth.radiusOfEarth
        for currArrival in self.arrivals:
            print(self.getCommentLine(currArrival))
            longWayRound = False
            if currArrival.get_dist_deg() % 360 > 180:
                longWayRound = True
            prevTimeDist = TimeDist(0, 0, 0, 0)
            for j in range(len(currArrival.path)):
                # Reduce the size of these codes:
                capj = currArrival.path[j]
                if j < len(currArrival.path) - 1:
                    capjplus = currArrival.path[j+1]
                if j > 0:
                    capjminus = currArrival.path[j-1]
                if capj.distRadian < prevTimeDist.distRadian:
                    raise RuntimeError("Ray path is backtracking, that's not "
                                       "possible.")
                calcTime = capj.time
                calcDepth = capj.depth
                prevDepth = calcDepth  # only used for interpolating below
                calcDist = capj.get_dist_deg()
                if calcTime > self.maxPathTime:
                    if(j != 0
                       and capjminus.time < self.maxPathTime):
                        # Past max time, so interpolate to maxPathTime.
                        calcDist = np.interp(self.maxPathTime,
                                             (capjminus.time,
                                              capj.time),
                                             (capjminus.get_dist_deg(),
                                              capj.get_dist_deg()))
                        calcDepth = np.interp(
                            self.maxPathTime, (capjminus.time, capj.time),
                            (capjminus.depth, capj.depth))
                        prevDepth = calcDepth
                        calcTime = self.maxPathTime
                    else:
                        break
                if longWayRound and calcDist != 0:
                    calcDist *= -1
                self.printDistRadius(calcDist, radiusOfEarth - calcDepth)
                if calcTime >= self.maxPathTime:
                    break
                if (j < len(currArrival.path) - 1
                        and currArrival.ray_param != 0
                        and capjplus.get_dist_deg()
                        - capj.get_dist_deg() > self.maxPathInc):
                    # Interpolate to steps of at most maxPathInc degrees for
                    # path.
                    maxInterpNum = math.ceil((capjplus.get_dist_deg()
                                              - capj.get_dist_deg())
                                             / self.maxPathInc)
                    for interpNum in range(1, maxInterpNum):
                        calcTime += (capjplus.time - capj.time) / maxInterpNum
                        if calcTime > self.maxPathTime:
                            break
                        if longWayRound:
                            calcDist -= (capjplus.get_dist_deg()
                                         - capj.get_dist_deg()) / maxInterpNum
                        else:
                            calcDist += (capjplus.get_dist_deg()
                                         - capj.get_dist_deg()) / maxInterpNum
                        calcDepth = prevDepth + (interpNum
                                                 * (capjplus.depth - prevDepth)
                                                 / maxInterpNum)
                        self.printDistRadius(calcDist,
                                             radiusOfEarth - calcDepth)

    def printDistRadius(self, calcDist, radius):
        print("{:4.2f}  {:4.1f}".format(calcDist, radius))


if __name__ == '__main__':
    # Permits running as script.
    tauPPath = TauP_Path()
    tauPPath.readcmdLineArgs()
    tauPPath.run(printOutput=True)
