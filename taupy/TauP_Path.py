from taupy.TauP_Pierce import TauP_Pierce
from math import pi


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
        pass


if __name__ == '__main__':
    # Permits running as script.
    tauPPath = TauP_Path()
    tauPPath.readcmdLineArgs()
    tauPPath.start()