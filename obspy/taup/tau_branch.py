#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np

from .helper_classes import TauModelError, TimeDist, SlownessModelError
from .slowness_layer import SlownessLayer, bullenDepthFor, bullenRadialSlowness


class TauBranch(object):
    """
    Provides storage and methods for distance, time and tau increments for a
    branch. A branch is a group of layers bounded by discontinuities or
    reversals in slowness gradient.
    """
    DEBUG = False

    def __init__(self, topDepth=0, botDepth=0, isPWave=0):
        self.topDepth = topDepth
        self.botDepth = botDepth
        self.isPWave = isPWave

    def __str__(self):
        desc = "Tau Branch\n"
        desc += " topDepth = " + str(self.topDepth) + "\n"
        desc += " botDepth = " + str(self.botDepth) + "\n"
        desc += " maxRayParam=" + str(self.maxRayParam) + " minTurnRayParam=" \
            + str(self.minTurnRayParam)
        desc += " minRayParam=" + str(self.minRayParam) + "\n"
        return desc

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def createBranch(self, sMod, minPSoFar, ray_params):
        """
        Calculates tau for this branch, between slowness layers topLayerNum and
        botLayerNum, inclusive.
        """
        topLayerNum = sMod.layerNumberBelow(self.topDepth, self.isPWave)
        botLayerNum = sMod.layerNumberAbove(self.botDepth, self.isPWave)
        topSLayer = sMod.getSlownessLayer(topLayerNum, self.isPWave)
        botSLayer = sMod.getSlownessLayer(botLayerNum, self.isPWave)
        if topSLayer['topDepth'] != self.topDepth \
                or botSLayer['botDepth'] != self.botDepth:
            if topSLayer['topDepth'] != self.topDepth \
                    and abs(topSLayer['topDepth'] - self.topDepth) < 0.000001:
                # Really close, so just move the top.
                print("Changing topDepth" + str(self.topDepth) + "-->" +
                      str(topSLayer.topDepth))
                self.topDepth = topSLayer['topDepth']
            elif botSLayer['botDepth'] != self.botDepth and \
                    abs(botSLayer['botDepth'] - self.botDepth) < 0.000001:
                # Really close, so just move the bottom.
                print("Changing botDepth" + str(self.botDepth) + "-->" +
                      str(botSLayer['botDepth']))
                self.botDepth = botSLayer['botDepth']
            else:
                raise TauModelError("createBranch: TauBranch not compatible "
                                    "with slowness sampling at topDepth"
                                    + str(self.topDepth))
        # Here we set minTurnRayParam to be the ray parameter that turns within
        # the layer, not including total reflections off of the bottom.
        # maxRayParam is the largest ray parameter that can penetrate this
        # branch. minRayParam is the minimum ray parameter that turns or is
        # totally reflected in this branch.
        self.maxRayParam = minPSoFar
        self.minTurnRayParam = sMod.getMinTurnRayParam(self.botDepth,
                                                       self.isPWave)
        self.minRayParam = sMod.getMinRayParam(self.botDepth, self.isPWave)

        timeDist = self.calcTimeDist(sMod, topLayerNum, botLayerNum,
                                     ray_params)
        self.dist = timeDist['dist']
        self.time = timeDist['time']
        self.tau = self.time - ray_params * self.dist

    def calcTimeDist(self, sMod, topLayerNum, botLayerNum, ray_params):
        timeDist = np.zeros_like(ray_params, dtype=TimeDist)
        timeDist['p'] = ray_params

        layerNum = np.arange(topLayerNum, botLayerNum + 1)
        layer = sMod.getSlownessLayer(layerNum, self.isPWave)

        for i, p in enumerate(ray_params):
            if p <= self.maxRayParam:
                mask = np.cumprod((p <= layer['topP']) & (p <= layer['botP']))
                mask = mask.astype(np.bool_)
                layerMask = layer[mask]
                layerNumMask = layerNum[mask]

                if len(layerMask):
                    temptd = sMod.layerTimeDist(p, layerNumMask, self.isPWave)
                    timeDist['dist'][i] = np.sum(temptd['dist'])
                    timeDist['time'][i] = np.sum(temptd['time'])

                    if ((layerMask['topP'][-1] - p) *
                            (p - layerMask['botP'][-1])) > 0:
                        raise SlownessModelError(
                            "Ray turns in the middle of this layer!")

        return timeDist

    def insert(self, ray_param, sMod, index):
        """
        Inserts the distance, time, and tau increment for the slowness sample
        given to the branch. This is used for making the depth correction to a
        tau model for a non-surface source.
        """
        topLayerNum = sMod.layerNumberBelow(self.topDepth, self.isPWave)
        botLayerNum = sMod.layerNumberAbove(self.botDepth, self.isPWave)
        topSLayer = sMod.getSlownessLayer(topLayerNum, self.isPWave)
        botSLayer = sMod.getSlownessLayer(botLayerNum, self.isPWave)
        if topSLayer['topDepth'] != self.topDepth \
                or botSLayer['botDepth'] != self.botDepth:
            raise TauModelError(
                "TauBranch depths not compatible with slowness sampling.")

        new_dist = 0.0
        new_time = 0.0
        if topSLayer['botP'] >= ray_param and topSLayer['topP'] >= ray_param:
            layerNum = np.arange(botLayerNum + 1)
            layers = sMod.getSlownessLayer(layerNum, self.isPWave)
            # So we don't sum below the turning depth.
            mask = np.cumprod(layers['botP'] >= ray_param).astype(np.bool_)
            layerNum = layerNum[mask]
            if len(layerNum):
                temptd = sMod.layerTimeDist(ray_param, layerNum, self.isPWave)
                new_dist = np.sum(temptd['dist'])
                new_time = np.sum(temptd['time'])

        self.shiftBranch(index)
        self.dist[index] = new_dist
        self.time[index] = new_time
        self.tau[index] = new_time - ray_param * new_dist

    def shiftBranch(self, index):
        new_size = len(self.dist) + 1

        self.dist.resize(new_size)
        self.dist[index + 1:] = self.dist[index:-1]
        self.dist[index] = 0

        self.time.resize(new_size)
        self.time[index + 1:] = self.time[index:-1]
        self.time[index] = 0

        self.tau.resize(new_size)
        self.tau[index + 1:] = self.tau[index:-1]
        self.tau[index] = 0

    def difference(self, topBranch, indexP, indexS, sMod, minPSoFar,
                   ray_params):
        """
        Generates a new tau branch by "subtracting" the given tau branch from
        this tau branch (self). The given tau branch is assumed to by the
        upper part of this branch. indexP specifies where a new ray
        corresponding to a P wave sample has been added; it is -1 if no ray
        parameter has been added to topBranch. indexS  is similar to indexP
        except for a S wave sample. Note that although the ray parameters
        for indexP and indexS were for the P and S waves that turned at the
        source depth, both ray parameters need to be added to both P and S
        branches.
        """
        if topBranch.topDepth != self.topDepth \
                or topBranch.botDepth > self.botDepth:
            if topBranch.topDepth != self.topDepth \
                    and abs(topBranch.topDepth - self.topDepth) < 0.000001:
                # Really close, just move top.
                self.topDepth = topBranch.topDepth
            else:
                raise TauModelError(
                    "TauBranch not compatible with slowness sampling.")
        if topBranch.isPWave != self.isPWave:
            raise TauModelError(
                "Can't subtract branches is isPWave doesn't agree.")
        # Find the top and bottom slowness layers of the bottom half.
        topLayerNum = sMod.layerNumberBelow(topBranch.botDepth, self.isPWave)
        botLayerNum = sMod.layerNumberBelow(self.botDepth, self.isPWave)
        topSLayer = sMod.getSlownessLayer(topLayerNum, self.isPWave)
        botSLayer = sMod.getSlownessLayer(botLayerNum, self.isPWave)
        if botSLayer['topDepth'] == self.botDepth \
                and botSLayer['botDepth'] > self.botDepth:
            # Gone one too far.
            botLayerNum -= 1
            botSLayer = sMod.getSlownessLayer(botLayerNum, self.isPWave)
        if topSLayer['topDepth'] != topBranch.botDepth \
                or botSLayer['botDepth'] != self.botDepth:
            raise TauModelError(
                "TauBranch not compatible with slowness sampling.")
        # Make sure indexP and indexS really correspond to new ray
        # parameters at the top of this branch.
        sLayer = sMod.getSlownessLayer(sMod.layerNumberBelow(
            topBranch.botDepth, True), True)
        if indexP >= 0 and sLayer['topP'] != ray_params[indexP]:
            raise TauModelError("P wave index doesn't match top layer.")
        sLayer = sMod.getSlownessLayer(sMod.layerNumberBelow(
            topBranch.botDepth, False), False)
        if indexS >= 0 and sLayer['topP'] != ray_params[indexS]:
            raise TauModelError("S wave index doesn't match top layer.")
        del sLayer
        # Construct the new TauBranch, going from the bottom of the top half
        # to the bottom of the whole branch.
        botBranch = TauBranch(topBranch.botDepth, self.botDepth, self.isPWave)
        botBranch.maxRayParam = topBranch.minRayParam
        botBranch.minTurnRayParam = self.minTurnRayParam
        botBranch.minRayParam = self.minRayParam
        PRayParam = -1
        SRayParam = -1
        arrayLength = len(self.dist)
        if indexP != -1:
            arrayLength += 1
            PRayParam = ray_params[indexP:indexP + 1]
            timeDistP = botBranch.calcTimeDist(sMod, topLayerNum, botLayerNum,
                                               PRayParam)
        if indexS != -1 and indexS != indexP:
            arrayLength += 1
            SRayParam = ray_params[indexS:indexS + 1]
            timeDistS = botBranch.calcTimeDist(sMod, topLayerNum, botLayerNum,
                                               SRayParam)
        else:
            # In case indexS==P then only need one.
            indexS = -1

        ddist = self.dist - topBranch.dist
        dtime = self.time - topBranch.time
        dtau = self.tau - topBranch.tau
        if indexP == -1:
            # Then both indices are -1 so no new ray parameters are added.
            botBranch.dist = ddist
            botBranch.time = dtime
            botBranch.tau = dtau
        else:
            botBranch.dist = np.empty(arrayLength)
            botBranch.time = np.empty(arrayLength)
            botBranch.tau = np.empty(arrayLength)

            if indexS == -1:
                # Only indexP != -1.
                botBranch.dist[:indexP] = ddist[:indexP]
                botBranch.time[:indexP] = dtime[:indexP]
                botBranch.tau[:indexP] = dtau[:indexP]

                botBranch.dist[indexP] = timeDistP['dist']
                botBranch.time[indexP] = timeDistP['time']
                botBranch.tau[indexP] = (timeDistP['time'] -
                                         PRayParam * timeDistP['dist'])

                botBranch.dist[indexP + 1:] = ddist[indexP:]
                botBranch.time[indexP + 1:] = dtime[indexP:]
                botBranch.tau[indexP + 1:] = dtau[indexP:]

            else:
                # Both indexP and S are != -1 so have two new samples
                botBranch.dist[:indexS] = ddist[:indexS]
                botBranch.time[:indexS] = dtime[:indexS]
                botBranch.tau[:indexS] = dtau[:indexS]

                botBranch.dist[indexS] = timeDistS['dist']
                botBranch.time[indexS] = timeDistS['time']
                botBranch.tau[indexS] = (timeDistS['time'] -
                                         SRayParam * timeDistS['dist'])

                botBranch.dist[indexS + 1:indexP + 1] = ddist[indexS:indexP]
                botBranch.time[indexS + 1:indexP + 1] = dtime[indexS:indexP]
                botBranch.tau[indexS + 1:indexP + 1] = dtau[indexS:indexP]

                # Put in at indexP+1 because we have already shifted by 1
                # due to indexS.
                botBranch.dist[indexP + 1] = timeDistP['dist']
                botBranch.time[indexP + 1] = timeDistP['time']
                botBranch.tau[indexP + 1] = (timeDistP['time'] -
                                             PRayParam * timeDistP['dist'])

                botBranch.dist[indexP + 2:] = ddist[indexP:]
                botBranch.time[indexP + 2:] = dtime[indexP:]
                botBranch.tau[indexP + 2:] = dtau[indexP:]

        return botBranch

    def path(self, ray_param, downgoing, sMod):
        """
        Called from TauP_Path to calculate ray paths.
        :param ray_param:
        :param downgoing:
        :param sMod:
        :return:
        """
        if ray_param > self.maxRayParam:
            return np.empty(0, dtype=TimeDist)
        assert ray_param >= 0

        try:
            topLayerNum = sMod.layerNumberBelow(self.topDepth, self.isPWave)
            botLayerNum = sMod.layerNumberAbove(self.botDepth, self.isPWave)
        # except NoSuchLayerError as e:
        except SlownessModelError:
            raise SlownessModelError("SlownessModel and TauModel are likely"
                                     "out of sync.")

        thePath = np.empty(botLayerNum - topLayerNum + 1, dtype=TimeDist)
        pathIndex = 0

        # Check to make sure layers and branches are compatible.
        sLayer = sMod.getSlownessLayer(topLayerNum, self.isPWave)
        if sLayer['topDepth'] != self.topDepth:
            raise SlownessModelError("Branch and slowness model are not in "
                                     "agreement.")
        sLayer = sMod.getSlownessLayer(botLayerNum, self.isPWave)
        if sLayer['botDepth'] != self.botDepth:
            raise SlownessModelError("Branch and slowness model are not in "
                                     "agreement.")

        # Downgoing branches:
        if downgoing:
            sLayerNum = np.arange(topLayerNum, botLayerNum + 1)
            sLayer = sMod.getSlownessLayer(sLayerNum, self.isPWave)

            mask = np.cumprod(sLayer['botP'] >= ray_param).astype(np.bool_)
            mask &= sLayer['topDepth'] != sLayer['botDepth']
            sLayerNum = sLayerNum[mask]
            sLayer = sLayer[mask]

            if len(sLayer):
                pathIndexEnd = pathIndex + len(sLayer)
                thePath[pathIndex:pathIndexEnd] = sMod.layerTimeDist(
                    ray_param,
                    sLayerNum,
                    self.isPWave)
                thePath[pathIndex:pathIndexEnd]['depth'] = sLayer['botDepth']
                pathIndex = pathIndexEnd

            # Apply Bullen laws on last element, if available.
            if len(sLayerNum):
                sLayerNum = sLayerNum[-1] + 1
            else:
                sLayerNum = topLayerNum
            if sLayerNum <= botLayerNum:
                sLayer = sMod.getSlownessLayer(sLayerNum, self.isPWave)
                if sLayer['topDepth'] != sLayer['botDepth']:
                    turnDepth = bullenDepthFor(sLayer, ray_param,
                                               sMod.radiusOfEarth)
                    turnSLayer = np.array([(sLayer['topP'], sLayer['topDepth'],
                                            ray_param, turnDepth)],
                                          dtype=SlownessLayer)
                    thePath[pathIndex] = bullenRadialSlowness(
                        turnSLayer,
                        ray_param,
                        sMod.radiusOfEarth)
                    thePath[pathIndex]['depth'] = turnSLayer['botDepth']
                    pathIndex += 1

        # Upgoing branches:
        else:
            sLayerNum = np.arange(botLayerNum, topLayerNum - 1, -1)
            sLayer = sMod.getSlownessLayer(sLayerNum, self.isPWave)

            mask = np.logical_or(sLayer['topP'] <= ray_param,
                                 sLayer['topDepth'] == sLayer['botDepth'])
            mask = np.cumprod(mask).astype(np.bool_)
            mask[-1] = False  # Always leave one element for Bullen.

            # Apply Bullen laws on first available element, if possible.
            first_unmasked = np.sum(mask)
            sLayer2 = sLayer[first_unmasked]
            if sLayer2['botP'] < ray_param:
                turnDepth = bullenDepthFor(sLayer2, ray_param,
                                           sMod.radiusOfEarth)
                turnSLayer = np.array([(sLayer2['topP'], sLayer2['topDepth'],
                                        ray_param, turnDepth)],
                                      dtype=SlownessLayer)
                thePath[pathIndex] = bullenRadialSlowness(
                    turnSLayer,
                    ray_param,
                    sMod.radiusOfEarth)
                thePath[pathIndex]['depth'] = turnSLayer['topDepth']
                pathIndex += 1
                mask[first_unmasked] = True

            # Apply regular time/distance calculation on all unmasked and
            # non-zero thickness layers.
            mask = (~mask) & (sLayer['topDepth'] != sLayer['botDepth'])
            sLayer = sLayer[mask]
            sLayerNum = sLayerNum[mask]

            if len(sLayer):
                pathIndexEnd = pathIndex + len(sLayer)
                thePath[pathIndex:pathIndexEnd] = sMod.layerTimeDist(
                    ray_param,
                    sLayerNum,
                    self.isPWave)
                thePath[pathIndex:pathIndexEnd]['depth'] = sLayer['topDepth']
                pathIndex = pathIndexEnd

        tempPath = thePath[:pathIndex]
        return tempPath
