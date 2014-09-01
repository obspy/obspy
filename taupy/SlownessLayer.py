class SlownessLayer:

    def __init__(self, topP, topDepth, botP, botDepth):
        self.topP = topP
        self.botP=botP
        if topDepth >= 0:
            self.topDepth = topDepth
        else:
            raise ValueError('topDepth must not be negative')
        if botDepth >= 0:
            self.botDepth = botDepth
        else:
            raise ValueError('botDepth must not be negative')

    def create_from_vlayer(vLayer, isPWave, radiusOfEarth=6371, isSpherical=True):
        """ Compute the slowness layer from a velocity layer.

        Note first argument is NOT meant to be self, this
        throws at least my IDE off into flagging wrong class errors."""
        topDepth = vLayer.topDepth
        botDepth = vLayer.botDepth
        waveType = ('p' if isPWave else 's')
        if isSpherical:
            topP = (radiusOfEarth - topDepth) / vLayer.evaluateAtTop(waveType)
            botP = (radiusOfEarth - botDepth) / vLayer.evaluateAtBottom(waveType)
        else:
            raise NotImplementedError("no flat models yet")
            
        return SlownessLayer(topP, topDepth, botP, botDepth)
