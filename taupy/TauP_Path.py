from taupy.TauP_Pierce import TauP_Pierce
from taupy.helper_classes import TimeDist
import numpy as np
import math


class TauP_Path(TauP_Pierce):
    # In effect, the methods here allow using TauP_Time to calculate the
    # pierce points.

    def __init__(self):
        TauP_Pierce.__init__(self)
        #self.mapWidthUnit = "i"
        #self.mapWidth = 6
        #self.gmtScript = False
        #self.svgOutput =False
        self.maxPathTime = 1e300
        self.maxPathInc = 1

    def calculate(self, degrees):
        self.depthCorrect(self.sourceDepth)
        self.recalcPhases()
        self.arrivals = []
        self.calcPath(degrees)

    def calcPath(self, degrees):
        self.degrees = degrees
        for phase in self.phases:
            phaseArrivals = phase.calcPath(degrees)
            self.arrivals += phaseArrivals

    def printResult(self):
        radiusOfEarth = self.tModDepth.radiusOfEarth
        for currArrival in self.arrivals:
            longWayRound = False
            if currArrival.getDistDeg() % 360 > 180:
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
                calcDist = capj.getDistDeg()
                if calcTime > self.maxPathTime:
                    if(j != 0
                       and capjminus.time < self.maxPathTime):
                        # Past max time, so interpolate to maxPathTime.
                        calcDist = np.interp(self.maxPathTime,
                                             (capjminus.time,
                                              capj.time),
                                             (capjminus.getDistDeg(),
                                              capj.getDistDeg()))
                        calcDepth = np.interp(self.maxPathTime,
                                             (capjminus.time,
                                              capj.time),
                                             (capjminus.depth,
                                              capj.depth))
                        prevDepth = calcDepth
                        calcTime = self.maxPathTime
                    else:
                        break
                if longWayRound and calcDist != 0:
                    calcDist *= -1
                self.printDistRadius(calcDist, radiusOfEarth - calcDepth)
                if calcTime >= self.maxPathTime:
                    break
                if (j < len(currArrival.path) - 1 and currArrival.rayParam != 0
                    and capjplus.getDistDeg()
                        - capj.getDistDeg() > self.maxPathInc):
                    # Interpolate to steps of at most maxPathInc degrees for
                    # path.
                    maxInterpNum = math.ceil((capjplus.getDistDeg()
                                              - capj.getDistDeg())
                                             / self.maxPathInc)
                    for interpNum in range(1, maxInterpNum):
                        calcTime += (capjplus.time - capj.time) / maxInterpNum
                        if calcTime > self.maxPathTime:
                            break
                        if longWayRound:
                            calcDist -= (capjplus.getDistDeg()
                                         - capj.getDistDeg()) / maxInterpNum
                        else:
                            calcDist += (capjplus.getDistDeg()
                                         - capj.getDistDeg()) / maxInterpNum
                        calcDepth = prevDepth + (interpNum
                                                 * (capjplus.depth- prevDepth)
                                                 / maxInterpNum)
                        self.printDistRadius(calcDist,
                                             radiusOfEarth - calcDepth)

    def printDistRadius(self, calcDist, radius):
        print("{}  {}".format(calcDist, radius))




if __name__ == '__main__':
    # Permits running as script.
    tauPPath = TauP_Path()
    tauPPath.readcmdLineArgs()
    tauPPath.run(printOutput=True)