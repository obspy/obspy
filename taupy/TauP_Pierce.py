from taupy.TauP_Time import TauP_Time


class TauP_Pierce(TauP_Time):

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









if __name__ == '__main__':
    # Permits running as executable.
    tauPPierce = TauP_Pierce()
    tauPPierce.readcmdLineArgs()
    tauPPierce.start()