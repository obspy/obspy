class TauBranch(object):
    """ Provides storage and methods for distance, time and tau increments for a
    branch. A branch is a group of layers bounded by discontinuities or reversals
    in slowness gradient."""
    DEBUG = False

    def __init__(self, topDepth, botDepth, isPWave):
        self.topDepth = topDepth
        self.botDepth = botDepth
        self.isPWave = isPWave

    def createBranch(self, sMod, minPSoFar, rayParams):
        """TODO"""
        # TODO: implement
        pass