from taupy.helper_classes import TimeDist, SlownessModelError
import math


# noinspection PyPep8Naming
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

    def bullenRadialSlowness(self, p, radiusOfEarth):
        """Calculates the time and distance (in radians) increments accumulated by a
        ray of spherical ray parameter p when passing through this layer. Note
        that this gives 1/2 of the true range and time increments since there
        will be both an upgoing and a downgoing path. Here we use the
        Mohorovicic or Bullen law: p=A*r^B"""
        timedist = TimeDist(p)
        if self.botDepth == self.topDepth:
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

    def bullenDepthFor(self, rayParam, radiusOfEarth):
        """Finds the depth for a ray parameter within this layer. Uses a Bullen
        interpolant, Ar^B. Special case for botP == 0 or botDepth == radiusOfEarth
        as these cause div by 0, use linear interpolation in this case."""
        if (self.topP - rayParam) * (rayParam - self.botP) >= 0:
            # Easy cases for 0 thickness layer, or ray parameter found at top or bottom.
            if self.topDepth == self.botDepth:
                return self.botDepth
            if self.topP == rayParam:
                return self.topDepth
            if self.botP == rayParam:
                return self.botDepth
            if self.botP != 0 and self.botDepth != radiusOfEarth:
                B = math.log(self.topP / self.botP) / math.log((radiusOfEarth - self.topDepth)
                                                               / (radiusOfEarth - self.botDepth))
                A = self.topP / math.pow((radiusOfEarth - self.topDepth), B)
                tempDepth = radiusOfEarth - math.exp(1/B * math.log(rayParam/A))
                # or equivalent (maybe better stability?):
                # tempDepth = radiusOfEarth - math.pow(rayParam/A, 1/B)
                # Check if slightly outside layer due to rounding or numerical instability:
                if self.topDepth > tempDepth > self.topDepth - 0.000001:
                    tempDepth = self.topDepth
                if self.botDepth < tempDepth < self.botDepth + 0.000001:
                    tempDepth = self.botDepth
                if (tempDepth < 0 or math.isnan(tempDepth) or math.isinf(tempDepth)
                    or tempDepth < self.topDepth or tempDepth > self.botDepth):
                    # Numerical instability in power law calculation? Try a linear interpolation if
                    # the layer is small (<5km).
                    if self.botDepth - self.topDepth < 5:
                        linear = ((self.botDepth - self.topDepth) / (self.botP - self.topP)
                                  * (rayParam - self.topP ) + self.topDepth)
                        if linear >= 0 and not math.isnan(linear) and not math.isinf(linear):
                            return linear
                    raise SlownessModelError("Calculated depth is outside layer, negative, or NaN.")
                # Check for tempDepth just above topDepth or below bottomDepth.
                if tempDepth < self.topDepth and self.topDepth - tempDepth < 1e-10:
                    return self.topDepth
                if tempDepth > self.botDepth and tempDepth - self.botDepth < 1e-10:
                    return self.botDepth
                return tempDepth
            else:
                # Special case for the centre of the Earth, since Ar^B might blow up at r = 0.
                if self.topP !=self.botP:
                    return (self.botDepth + (rayParam - self.botP)
                            * (self.topDepth - self.botDepth) / (self.topP - self.botP))
                else:
                    # weird case, return botDepth??
                    return self.botDepth
        else:
            raise SlownessModelError("Ray parameter is not contained within this slowness layer.")

