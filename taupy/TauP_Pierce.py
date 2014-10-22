from taupy.TauP_Time import TauP_Time
from math import pi


class TauP_Pierce(TauP_Time):
    # In effect, the methods here allow using TauP_Time to calculate the
    # pierce points.

    def __init__(self):
        super().__init__()

    def depthCorrect(self, depth):
        pass

    def calculate(self, degrees):
        self.depthCorrect(self.sourceDepth)
        self.recalcPhases()
        self.arrivals = []
        self.calcPierce(degrees)

    def calcPierce(self, degrees):
        """
        Calculates the pierce points for phases ar the given distance.
        :param degrees:
        :return:
        """
        # This seems stupid: self.degrees = degrees -- see if it's needed.
        for phase in self.phases:
            phaseArrivals = phase.calcPierce(degrees)
            self.arrivals += phaseArrivals

    def printResult(self):
        for currArrival in self.arrivals:
            print(self.getCommentLine(currArrival))
            longWayRound = False
            if (currArrival.dist * 180 / pi) % 360 > 180:
                longWayRound = True
            prevDepth = currArrival.pierce[0].depth
            for j, calcDist in enumerate(
                    [p.getDistDeg() for p in currArrival.pierce]):
                if longWayRound is True and calcDist != 0:
                    calcDist *= -1
                if j < len(currArrival.pierce) - 1:
                    nextDepth = currArrival.pierce[j + 1].depth
                else:
                    nextDepth = currArrival.pierce[j].depth
                #if (not(self.onlyTurnPoints or self.onlyRevPoints or
                #            self.onlyUnderPoints or self.onlyAddPoints) or ((self.onlyAddPoints and self.isAddDepth)))









if __name__ == '__main__':
    # Permits running as executable.
    tauPPierce = TauP_Pierce()
    tauPPierce.readcmdLineArgs()
    tauPPierce.start()