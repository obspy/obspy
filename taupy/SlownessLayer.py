from taupy.helper_classes import TimeDist
import math
from taupy.SlownessModel import SlownessModelError


class SlownessLayer:

    def __init__(self, topP, topDepth, botP, botDepth):
        self.topP = topP
        self.botP = botP
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
        throws at least my IDE off into flagging wrong class errors.
        Seriously, it breaks with 'self' for some reason..."""
        topDepth = vLayer.topDepth
        botDepth = vLayer.botDepth
        waveType = ('p' if isPWave else 's')
        if isSpherical:
            topP = (radiusOfEarth - topDepth) / vLayer.evaluateAtTop(waveType)
            botP = (radiusOfEarth - botDepth) / vLayer.evaluateAtBottom(waveType)
        else:
            raise NotImplementedError("no flat models yet")
        return SlownessLayer(topP, topDepth, botP, botDepth)

    def validate(self):
        if math.isnan(self.topDepth) or math.isnan(self.botDepth) or math.isnan(self.topP) or math.isnan(self.botP):
            raise SlownessModelError("Slowness layer has NaN values.")
        if self.topP < 0 or self.botP < 0:
            raise SlownessModelError("Slowness layer has negative slowness.")
        if self.topDepth > self.botDepth:
            raise SlownessModelError("Slowness layer has negative thickness.")
        return True

    def bullenDepthFor(self, rayParam, radiusOfEarth):
        return 5
        # TODO implement the methods here properly

    def bullenRadialSlowness(self, p, radiusOfEarth):
        """Calculates the time and distance (in radians) increments accumulated by a
        ray of spherical ray parameter p when passing through this layer. Note
        that this gives 1/2 of the true range and time increments since there
        will be both an upgoing and a downgoing path. Here we use the
        Mohorovicic or Bullen law: p=A*r^B"""
        timedist = TimeDist(p)
        if (self.botDepth == self.topDepth):
            timedist.distRadian = 0
            timedist.time = 0
            return timedist
        # Only do Bullen radial slowness if the layer is not too thin (e.g. 1 micron).
        # In that case also just return 0.
        if self.botDepth - self.topDepth < 0.000000001:
            return timedist
        B = math.log(self.topP / self.botP) / math.log((radiusOfEarth - self.topDepth)
                                                       / (radiusOfEarth - self.botDepth))
        sqrtTopTopMpp = math.sqrt(self.topP * self.topP - p * p)
        sqrtBotBotMpp = math.sqrt(self.botP * self.botP - p * p)
        timedist.distRadian = (math.atan2(p, sqrtBotBotMpp) - math.atan2(p, sqrtTopTopMpp)) / B
        timedist.time = (sqrtTopTopMpp - sqrtBotBotMpp) / B
        if timedist.distRadian < 0 or timedist.time < 0 or math.isnan(timedist.distRadian) or math.isnan(timedist.time):
            raise SlownessModelError("timedist.time or .distRadian < 0 or Nan")
        return timedist




