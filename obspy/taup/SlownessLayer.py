#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import math
import numpy as np

from .helper_classes import TimeDist, SlownessModelError


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

    def __str__(self):
        desc = ("top p " + str(self.topP) + ", topDepth " + str(self.topDepth)
                + ", bot p " + str(self.botP) + ", botDepth " +
                str(self.botDepth))
        return desc

    def validate(self):
        if math.isnan(self.topDepth) \
                or math.isnan(self.botDepth) \
                or math.isnan(self.topP) \
                or math.isnan(self.botP):
            raise SlownessModelError("Slowness layer has NaN values.")
        if self.topP < 0 or self.botP < 0:
            raise SlownessModelError("Slowness layer has negative slowness.")
        if self.topDepth > self.botDepth:
            raise SlownessModelError("Slowness layer has negative thickness.")
        return True

    def bullenRadialSlowness(self, p, radiusOfEarth):
        """
        Calculates the time and distance (in radians) increments accumulated
        by a ray of spherical ray parameter p when passing through this
        layer. Note that this gives 1/2 of the true range and time
        increments since there will be both an upgoing and a downgoing path.
        Here we use the Mohorovicic or Bullen law: p=A*r^B"""
        timedist = TimeDist(p)
        if self.botDepth == self.topDepth:
            timedist.distRadian = 0
            timedist.time = 0
            return timedist
        # Only do Bullen radial slowness if the layer is not too thin (e.g.
        # 1 micron). In that case also just return 0.
        if self.botDepth - self.topDepth < 0.000000001:
            return timedist
        B = math.log(self.topP / self.botP) / math.log(
            (radiusOfEarth - self.topDepth) / (radiusOfEarth - self.botDepth))
        sqrtTopTopMpp = math.sqrt(self.topP * self.topP - p * p)
        sqrtBotBotMpp = math.sqrt(self.botP * self.botP - p * p)
        timedist.distRadian = (math.atan2(p, sqrtBotBotMpp) -
                               math.atan2(p, sqrtTopTopMpp)) / B
        timedist.time = (sqrtTopTopMpp - sqrtBotBotMpp) / B
        if timedist.distRadian < 0 \
                or timedist.time < 0 \
                or math.isnan(timedist.distRadian) \
                or math.isnan(timedist.time):
            raise SlownessModelError("timedist.time or .distRadian < 0 or Nan")
        return timedist

    def bullenDepthFor(self, rayParam, radiusOfEarth):
        """
        Finds the depth for a ray parameter within this layer. Uses a Bullen
        interpolant, Ar^B. Special case for botP == 0 or
        botDepth == radiusOfEarth as these cause div by 0, use linear
        interpolation in this case.
        """
        if (self.topP - rayParam) * (rayParam - self.botP) >= 0:
            # Easy cases for 0 thickness layer, or ray parameter found at
            # top or bottom.
            if self.topDepth == self.botDepth:
                return self.botDepth
            if self.topP == rayParam:
                return self.topDepth
            if self.botP == rayParam:
                return self.botDepth
            if self.botP != 0 and self.botDepth != radiusOfEarth:
                B = np.divide(math.log(self.topP / self.botP),
                              math.log((radiusOfEarth - self.topDepth)
                              / (radiusOfEarth - self.botDepth)))
                # This is a cludge but it's needed to mimic the Java behaviour.
                try:
                    denom = math.pow((radiusOfEarth - self.topDepth), B)
                except OverflowError:
                    denom = np.inf
                A = np.divide(self.topP, denom)
                with np.errstate(divide='ignore', invalid='ignore'):
                    tempDepth = radiusOfEarth - math.exp(
                        1.0 / B * math.log(np.divide(rayParam, A)))
                # or equivalent (maybe better stability?):
                # tempDepth = radiusOfEarth - math.pow(rayParam/A, 1/B)
                # Check if slightly outside layer due to rounding or
                # numerical instability:
                if self.topDepth > tempDepth > self.topDepth - 0.000001:
                    tempDepth = self.topDepth
                if self.botDepth < tempDepth < self.botDepth + 0.000001:
                    tempDepth = self.botDepth
                if tempDepth < 0 \
                        or math.isnan(tempDepth) \
                        or math.isinf(tempDepth) \
                        or tempDepth < self.topDepth \
                        or tempDepth > self.botDepth:
                    # Numerical instability in power law calculation? Try a
                    # linear interpolation if the layer is small (<5km).
                    if self.botDepth - self.topDepth < 5:
                        linear = ((self.botDepth - self.topDepth) /
                                  (self.botP - self.topP)
                                  * (rayParam - self.topP) + self.topDepth)
                        if linear >= 0 \
                                and not math.isnan(linear) \
                                and not math.isinf(linear):
                            return linear
                    raise SlownessModelError(
                        "Calculated depth is outside layer, negative, or NaN.")
                # Check for tempDepth just above topDepth or below bottomDepth.
                if tempDepth < self.topDepth \
                        and self.topDepth - tempDepth < 1e-10:
                    return self.topDepth
                if tempDepth > self.botDepth \
                        and tempDepth - self.botDepth < 1e-10:
                    return self.botDepth
                return tempDepth
            else:
                # Special case for the centre of the Earth, since Ar^B might
                #  blow up at r = 0.
                if self.topP != self.botP:
                    return (self.botDepth + (rayParam - self.botP)
                            * (self.topDepth - self.botDepth) /
                            (self.topP - self.botP))
                else:
                    # weird case, return botDepth??
                    return self.botDepth
        else:
            raise SlownessModelError(
                "Ray parameter is not contained within this slowness layer.")

    def evaluateAtBullen(self, depth, radiusOfEarth):
        """
        Finds the slowness at the given depth. Note that this method assumes
        a Bullen type of slowness interpolation, ie p(r) = a*r^b. This will
        produce results consistent with a tau model that uses this
        interpolant, but it may differ slightly from going directly to the
        velocity model. Also, if the tau model is generated using another
        interpolant, linear for instance, then the result may not be consistent
        with the tau model.
        """
        topP = self.topP
        botP = self.botP
        topDepth = self.topDepth
        botDepth = self.botDepth
        # Could do some safeguard asserts...
        assert not botDepth > radiusOfEarth
        assert not (topDepth - depth) * (depth - botDepth) < 0
        if depth == topDepth:
            return topP
        elif depth == botDepth:
            return botP
        else:
            B = np.divide(math.log(np.divide(topP, botP)),
                          math.log(np.divide((radiusOfEarth - topDepth),
                                             (radiusOfEarth - botDepth))))
            ADenominator = pow((radiusOfEarth - topDepth), B)
            A = topP / ADenominator
            answer = A * pow((radiusOfEarth - depth), B)
            if answer < 0 or math.isnan(answer) or math.isinf(answer):
                # numerical instability in power law calculation???
                # try a linear interpolation if the layer is small ( <2 km)
                # or if denominator of A is infinity as we probably overflowed
                # the double in that case.
                if botDepth - topDepth < 2 \
                        or math.isinf(ADenominator) \
                        or botP == 0:
                    linear = (botP - topP) / (botDepth - topDepth) * \
                             (depth - topDepth) + topP
                    if linear < 0 \
                            or math.isinf(linear) \
                            or math.isnan(linear):
                        pass
                    else:
                        return linear
                raise SlownessModelError(
                    "Calculated Slowness is NaN or negative!")
        return answer

    def hasZeroThickness(self):
        if self.topDepth == self.botDepth:
            return True
        else:
            return False


def create_from_vlayer(vLayer, isPWave, radiusOfEarth=6371,
                       isSpherical=True):
    """
    Compute the slowness layer from a velocity layer.
    """
    topDepth = vLayer.topDepth
    botDepth = vLayer.botDepth
    waveType = ('p' if isPWave else 's')
    if isSpherical:
        topP = (radiusOfEarth - topDepth) / \
            vLayer.evaluateAtTop(waveType)
        botP = (radiusOfEarth - botDepth) / \
            vLayer.evaluateAtBottom(waveType)
    else:
        raise NotImplementedError("no flat models yet")
    return SlownessLayer(topP, topDepth, botP, botDepth)
