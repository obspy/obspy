from .helper_classes import TauModelError, TimeDist, SlownessModelError


class TauBranch(object):
    """ Provides storage and methods for distance, time and tau increments for a
    branch. A branch is a group of layers bounded by discontinuities or reversals
    in slowness gradient."""
    DEBUG = False

    def __init__(self, topDepth=0, botDepth=0, isPWave=0):
        self.topDepth = topDepth
        self.botDepth = botDepth
        self.isPWave = isPWave

    def __str__(self):
        desc = "Tau Branch\n"
        desc += " topDepth = " + str(self.topDepth) + "\n"
        desc += " botDepth = " + str(self.botDepth) + "\n"
        desc += " maxRayParam=" + str(self.maxRayParam) + " minTurnRayParam=" + str(self.minTurnRayParam)
        desc += " minRayParam=" + str(self.minRayParam) + "\n"
        return desc

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

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

    def insert(self, rayParam, sMod, index):
        """Inserts the distance, time, and tau increment for the slowness sample
        given to the branch. This is used for making the depth correction to a
        tau model for a non-surface source."""
        topLayerNum = sMod.layerNumberBelow(self.topDepth, self.isPWave)
        botLayerNum = sMod.layerNumberAbove(self.botDepth, self.isPWave)
        topSLayer = sMod.getSlownessLayer(topLayerNum, self.isPWave)
        botSLayer = sMod.getSlownessLayer(botLayerNum, self.isPWave)
        if topSLayer.topDepth != self.topDepth or botSLayer.botDepth != self.botDepth:
            raise TauModelError("TauBranch depths not compatible with slowness sampling.")
        td = TimeDist(rayParam, 0, 0)
        if topSLayer.botP >= rayParam and topSLayer.topP >= rayParam:
            for i in range(botLayerNum + 1):
                if sMod.getSlownessLayer(i, self.isPWave).botP < rayParam:
                    # So we don't sum below the turning depth.
                    break
                else:
                    temptd = sMod.layerTimeDist(rayParam, i, self.isPWave)
                    td.distRadian += temptd.distRadian
                    td.time += temptd.time
        self.shiftBranch(index)
        self.dist[index] = td.distRadian
        self.time[index] = td.time
        self.tau[index] = td.time - rayParam * td.distRadian

    def shiftBranch(self, index):
        newDist = self.dist[:index]
        newDist.append(0)
        newDist += self.dist[index:]
        self.dist = newDist
        newTime = self.time[:index]
        newTime.append(0)
        newTime += self.time[index:]
        self.time = newTime
        newTau = self.tau[:index]
        newTau.append(0)
        newTau += self.tau[index:]
        self.tau = newTau

    def difference(self, topBranch, indexP, indexS, sMod, minPSoFar, rayParams):
        """Generates a new tau branch by "subtracting" the given tau branch from
        this tau branch (self). The given tau branch is assumed to by the upper part of this branch.
        indexP  specifies where a new ray corresponding to a P wave sample has been added; it is -1 if no
                ray parameter has been added to topBranch.
        indexS  is similar to indexP except for a S wave sample.
        Note that although the ray parameters for indexP and indexS were for the P and S waves that turned at the
        source depth, both ray parameters need to be added to both P and S branches.
        """
        if topBranch.topDepth != self.topDepth or topBranch.botDepth > self.botDepth:
            if topBranch.topDepth != self.topDepth and abs(topBranch.topDepth - self.topDepth) < 0.000001:
                # Really close, just move top.
                self.topDepth = topBranch.topDepth
            else:
                raise TauModelError("TauBranch not compatible with slowness sampling.")
        if topBranch.isPWave != self.isPWave:
            raise TauModelError("Can't subtract branches is isPWave doesn't agree.")
        # Find the top and bottom slowness layers of the bottom half.
        topLayerNum = sMod.layerNumberBelow(topBranch.botDepth, self.isPWave)
        botLayerNum = sMod.layerNumberBelow(self.botDepth, self.isPWave)
        topSLayer = sMod.getSlownessLayer(topLayerNum, self.isPWave)
        botSLayer = sMod.getSlownessLayer(botLayerNum, self.isPWave)
        if botSLayer.topDepth == self.botDepth and botSLayer.botDepth > self.botDepth:
            # Gone one too far.
            botLayerNum -= 1
            botSLayer = sMod.getSlownessLayer(botLayerNum, self.isPWave)
        if topSLayer.topDepth != topBranch.botDepth or botSLayer.botDepth != self.botDepth:
            raise TauModelError("TauBranch not compatible with slowness sampling.")
        # Make sure indexP and indexS really correspond to new ray parameters at the top of this branch.
        sLayer = sMod.getSlownessLayer(sMod.layerNumberBelow(topBranch.botDepth, True), True)
        if indexP >= 0 and sLayer.topP != rayParams[indexP]:
            raise TauModelError("P wave index doesn't match top layer.")
        sLayer = sMod.getSlownessLayer(sMod.layerNumberBelow(topBranch.botDepth, False), False)
        if indexS >= 0 and sLayer.topP != rayParams[indexS]:
            raise TauModelError("S wave index doesn't match top layer.")
        del sLayer
        # Construct the new TauBranch, going from the bottom of the top half to the bottom of the whole branch.
        botBranch = TauBranch(topBranch.botDepth, self.botDepth, self.isPWave)
        botBranch.maxRayParam = topBranch.minRayParam
        botBranch.minTurnRayParam = self.minTurnRayParam
        botBranch.minRayParam = self.minRayParam
        PRayParam = -1
        SRayParam = -1
        arrayLength = len(self.dist)
        if indexP != -1:
            arrayLength += 1
            PRayParam = rayParams[indexP]
            timeDistP = botBranch.calcTimeDist(sMod, topLayerNum, botLayerNum, PRayParam)
        if indexS != -1 and indexS != indexP:
            arrayLength += 1
            SRayParam = rayParams[indexS]
            timeDistS = botBranch.calcTimeDist(sMod, topLayerNum, botLayerNum, SRayParam)
        else:
            # In case indexS==P then only need one.
            indexS = -1
        # This looks silly, but makes filling these later on easier
        botBranch.dist = [0 for i in range(arrayLength)]
        botBranch.time = [0 for i in range(arrayLength)]
        botBranch.tau = [0 for i in range(arrayLength)]
        if indexP == -1:
            # Then both indices are -1 so no new ray parameters are added.
            botBranch.dist = [a-b for a, b in zip(self.dist, topBranch.dist)]
            botBranch.time = [a-b for a, b in zip(self.time, topBranch.time)]
            botBranch.tau = [a-b for a, b in zip(self.tau, topBranch.tau)]
        else:
            if indexS == -1:
                # Only indexP != -1.
                # Slice to have the 'iteration' go only until indexP (zip ends with the shortest iterator).
                botBranch.dist = [a-b for a, b in zip(self.dist[:indexP], topBranch.dist)]
                botBranch.time = [a-b for a, b in zip(self.time[:indexP], topBranch.time)]
                botBranch.tau = [a-b for a, b in zip(self.tau[:indexP], topBranch.tau)]
                botBranch.dist.append(timeDistP.distRadian)
                botBranch.time.append(timeDistP.time)
                botBranch.tau.append(timeDistP.time - PRayParam * timeDistP.distRadian)
                botBranch.dist[indexP:] = [a-b for a, b in zip(self.dist[indexP:], topBranch.dist[indexP+1:])]
                botBranch.dist[indexP:] = [a-b for a, b in zip(self.dist[indexP:], topBranch.dist[indexP+1:])]
                botBranch.dist[indexP:] = [a-b for a, b in zip(self.dist[indexP:], topBranch.dist[indexP+1:])]
            else:
                # Both indexP and S are != -1 so have two new samples
                # Note the following does the same (in the beginning) as the previous clause.
                # I'm on the fence about which is better.
                for i in range(indexS):
                    botBranch.dist[i] = self.dist[i] - topBranch.dist[i]
                    botBranch.time[i] = self.time[i] - topBranch.time[i]
                    botBranch.tau[i] = self.tau[i] - topBranch.tau[i]
                botBranch.dist[indexS] = timeDistS.distRadian
                botBranch.time[indexS] = timeDistS.time
                botBranch.tau[indexS] = timeDistS.time - SRayParam * timeDistS.distRadian
                for i in range(indexS, indexP):
                    botBranch.dist[i + 1] = self.dist[i] - topBranch.dist[i + 1]
                    botBranch.time[i + 1] = self.time[i] - topBranch.time[i + 1]
                    botBranch.tau[i + 1] = self.tau[i] - topBranch.tau[i + 1]
                # Put in at indexP+1 because we have already shifted by 1 due to indexS.
                botBranch.dist[indexP + 1] = timeDistP.distRadian
                botBranch.time[indexP + 1] = timeDistP.time
                botBranch.tau[indexP + 1] = timeDistP.time - PRayParam * timeDistP.distRadian
                for i in range(indexP, len(self.dist)):
                    botBranch.dist[i + 2] = self.dist[i] - topBranch.dist[i + 2]
                    botBranch.time[i + 2] = self.time[i] - topBranch.time[i + 2]
                    botBranch.tau[i + 2] = self.tau[i] - topBranch.tau[i + 2]
        return botBranch



