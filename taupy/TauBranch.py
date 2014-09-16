from .helper_classes import TauModelError, TimeDist, SlownessModelError


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
        """Calculates tau for this branch, between slowness layers topLayerNum and
        botLayerNum, inclusive."""
        topLayerNum = sMod.layerNumberBelow(self.topDepth, self.isPWave)
        botLayerNum = sMod.layerNumberAbove(self.botDepth, self.isPWave)
        topSLayer = sMod.getSlownessLayer(topLayerNum, self.isPWave)
        botSLayer = sMod.getSlownessLayer(botLayerNum, self.isPWave)
        if topSLayer.topDepth != self.topDepth or botSLayer.botDepth != self.botDepth:
            if topSLayer.topDepth != self.topDepth and abs(topSLayer.topDepth - self.topDepth) < 0.000001:
                # Really close, so just move the top.
                print("Changing topDepth" + str(self.topDepth) + "-->" + str(topSLayer.topDepth))
                self.topDepth = topSLayer.topDepth
            elif botSLayer.botDepth != self.botDepth and abs(botSLayer.botDepth - self.botDepth) < 0.000001:
                # Really close, so just move the bottom.
                print("Changing botDepth" + str(self.botDepth) + "-->" + str(botSLayer.botDepth))
                self.botDepth = botSLayer.botDepth
            else:
                raise TauModelError("createBranch: TauBranch not compatible with slowness sampling at topDepth"
                                    + str(self.topDepth))
        # Here we set minTurnRayParam to be the ray parameter that turns within
        # the layer, not including total reflections off of the bottom.
        # maxRayParam is the largest ray parameter that can penetrate this
        # branch. minRayParam is the minimum ray parameter that turns or is
        # totally reflected in this branch.
        self.maxRayParam = minPSoFar
        self.minTurnRayParam = sMod.getMinTurnRayParam(self.botDepth, self.isPWave)
        self.minRayParam = sMod.getMinRayParam(self.botDepth, self.isPWave)

        timeDist = [self.calcTimeDist(sMod, topLayerNum, botLayerNum, p) for p in rayParams]
        self.dist = [timeDist.distRadian for timeDist in timeDist]
        self.time = [timeDist.time for timeDist in timeDist]
        # This would maybe be easier using numpy arrays?
        self.tau = [time - p * dist for (time, dist, p) in zip(self.time, self.dist, rayParams)]

    def calcTimeDist(self, sMod, topLayerNum, botLayerNum, p):
        timeDist = TimeDist(p)
        if p <= self.maxRayParam:
            layerNum = topLayerNum
            layer = sMod.getSlownessLayer(layerNum, self.isPWave)
            while layerNum <= botLayerNum and p <= layer.topP and p <= layer.botP:
                timeDist.add(sMod.layerTimeDist(p, layerNum, self.isPWave))
                layerNum += 1
                if layerNum <= botLayerNum:
                    layer = sMod.getSlownessLayer(layerNum, self.isPWave)
            if (layer.topP - p) * (p - layer.botP) > 0:
                raise SlownessModelError("Ray turns in the middle of this layer!")
        return timeDist



