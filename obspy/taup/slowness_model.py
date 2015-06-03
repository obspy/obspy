#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Slowness model class.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import math

import numpy as np

from .helper_classes import (CriticalDepth, DepthRange, SlownessLayer,
                             SlownessModelError, SplitLayerInfo, TimeDist)
from .slowness_layer import (bullenDepthFor,
                             bullenRadialSlowness, create_from_vlayer,
                             evaluateAtBullen)
from .velocity_layer import (DEFAULT_DENSITY, DEFAULT_QP, DEFAULT_QS,
                             VelocityLayer, evaluateVelocityAtBottom,
                             evaluateVelocityAtTop)


def _fixCriticalDepths(criticalDepths, layerNum, isPWave):
    name = 'pLayerNum' if isPWave else 'sLayerNum'

    mask = criticalDepths[name] > layerNum
    criticalDepths[name][mask] += 1


class SlownessModel(object):
    """
    Storage and methods for generating slowness-depth pairs.
    """
    DEBUG = False
    DEFAULT_SLOWNESS_TOLERANCE = 1e-16
    radiusOfEarth = 6371.0

    # NB if the following are actually cleared (lists are mutable) every
    # time createSample is called, maybe it would be better to just put these
    # initialisations into the relevant methods? They do have to be
    # persistent across method calls in createSample though, so don't.

    # Stores the layer number for layers in the velocity model with a critical
    # point at their top. These form the "branches" of slowness sampling.
    criticalDepths = None  # will be list of CriticalDepth objects
    # Store depth ranges that contains a high slowness zone for P/S. Stored as
    # DepthRange objects, containing the top depth and bottom depth.
    highSlownessLayerDepthsP = []  # will be list of DepthRanges
    highSlownessLayerDepthsS = []
    # Stores depth ranges that are fluid, ie S velocity is zero. Stored as
    # DepthRange objects, containing the top depth and bottom depth.
    fluidLayerDepths = []
    PLayers = None
    SLayers = None
    # For methods that have an isPWave parameter
    SWAVE = False
    PWAVE = True

    def __init__(self, vMod, minDeltaP=0.1, maxDeltaP=11, maxDepthInterval=115,
                 maxRangeInterval=2.5 * math.pi / 180, maxInterpError=0.05,
                 allowInnerCoreS=True,
                 slowness_tolerance=DEFAULT_SLOWNESS_TOLERANCE,
                 skip_model_creation=False):

        self.vMod = vMod
        self.minDeltaP = minDeltaP
        self.maxDeltaP = maxDeltaP
        self.maxDepthInterval = maxDepthInterval
        self.maxRangeInterval = maxRangeInterval
        self.maxInterpError = maxInterpError
        self.allowInnerCoreS = allowInnerCoreS
        self.slowness_tolerance = slowness_tolerance
        if skip_model_creation:
            return
        self.createSample()

    def __str__(self):
        desc = "".join([
            "radiusOfEarth=", str(self.radiusOfEarth), "\n maxDeltaP=",
            str(self.maxDeltaP),
            "\n minDeltaP=", str(self.minDeltaP), "\n maxDepthInterval=",
            str(self.maxDepthInterval), "\n maxRangeInterval=",
            str(self.maxRangeInterval),
            "\n allowInnerCoreS=", str(self.allowInnerCoreS),
            "\n slownessTolerance=", str(self.slowness_tolerance),
            "\n getNumLayers('P')=", str(self.getNumLayers(self.PWAVE)),
            "\n getNumLayers('S')=", str(self.getNumLayers(self.SWAVE)),
            "\n fluidLayerDepths.size()=", str(len(self.fluidLayerDepths)),
            "\n highSlownessLayerDepthsP.size()=",
            str(len(self.highSlownessLayerDepthsP)),
            "\n highSlownessLayerDepthsS.size()=",
            str(len(self.highSlownessLayerDepthsS)),
            "\n criticalDepths.size()=",
            (str(len(self.criticalDepths)) if self.criticalDepths else 'N/A'),
            "\n"])
        desc += "**** Critical Depth Layers ************************\n"
        desc += str(self.criticalDepths)
        desc += "\n"
        desc += "\n**** Fluid Layer Depths ************************\n"
        for fl in self.fluidLayerDepths:
            desc += str(fl.topDepth) + "," + str(fl.botDepth) + " "
        desc += "\n"
        desc += "\n**** P High Slowness Layer Depths ****************\n"
        for fl in self.highSlownessLayerDepthsP:
            desc += str(fl.topDepth) + "," + str(fl.botDepth) + " "
        desc += "\n"
        desc += "\n**** S High Slowness Layer Depths ****************\n"
        for fl in self.highSlownessLayerDepthsS:
            desc += str(fl.topDepth) + "," + str(fl.botDepth) + " "
        desc += "\n"
        desc += "\n**** P Layers ****************\n"
        for l in self.PLayers:
            desc += str(l) + "\n"
        return desc

    def createSample(self):
        """
        Create slowness-depth layers from a velocity model.

        This method takes a velocity model and creates a vector containing
        slowness-depth layers that, hopefully, adequately sample both slowness
        and depth so that the travel time as a function of distance can be
        reconstructed from the theta function.
        """
        # Some checks on the velocity model
        self.vMod.validate()
        if self.vMod.getNumLayers() == 0:
            raise SlownessModelError("velModel.getNumLayers()==0")
        if self.vMod.layers[0]['topSVelocity'] == 0:
            raise SlownessModelError(
                "Unable to handle zero S velocity layers at surface. "
                "This should be fixed at some point, but is a limitation of "
                "TauP at this point.")
        if self.DEBUG:
            print("start createSample")

        self.radiusOfEarth = self.vMod.radiusOfEarth

        if self.DEBUG:
            print("findCriticalPoints")
        self.findCriticalPoints()
        if self.DEBUG:
            print("coarseSample")
        self.coarseSample()
        if self.DEBUG:
            self.validate()
        if self.DEBUG:
            print("ray_paramCheck")
        self.ray_paramIncCheck()
        if self.DEBUG:
            print("depthIncCheck")
        self.depthIncCheck()
        if self.DEBUG:
            print("distanceCheck")
        self.distanceCheck()
        if self.DEBUG:
            print("fixCriticalPoints")
        self.fixCriticalPoints()

        self.validate()
        if self.DEBUG:
            print("createSample seems to be done successfully.")

    def findCriticalPoints(self):
        """
        Find all critical points within a velocity model.

        Critical points are first order discontinuities in velocity/slowness,
        local extrema in slowness. A high slowness zone is a low velocity zone,
        but it is possible to have a slightly low velocity zone within a
        spherical Earth that is not a high slowness zone and thus does not
        exhibit any of the pathological behavior of a low velocity zone.
        """
        highSlownessZoneP = DepthRange()
        highSlownessZoneS = DepthRange()
        fluidZone = DepthRange()
        inFluidZone = False
        belowOuterCore = False
        inHighSlownessZoneP = False
        inHighSlownessZoneS = False
        # just some very big values (java had max possible of type,
        # but these should do)
        minPSoFar = 1.1e300
        minSSoFar = 1.1e300
        # First remove any critical points previously stored
        # so these are effectively re-initialised... it's probably silly
        self.criticalDepths = np.empty(len(self.vMod.layers),
                                       dtype=CriticalDepth)
        cd_count = 0
        self.highSlownessLayerDepthsP = []  # lists of DepthRange
        self.highSlownessLayerDepthsS = []
        self.fluidLayerDepths = []

        # Initialize the current velocity layer
        # to be zero thickness layer with values at the surface
        currVLayer = self.vMod.layers[0]
        currVLayer = np.array([(
            currVLayer['topDepth'], currVLayer['topDepth'],
            currVLayer['topPVelocity'], currVLayer['topPVelocity'],
            currVLayer['topSVelocity'], currVLayer['topSVelocity'],
            currVLayer['topDensity'], currVLayer['topDensity'],
            currVLayer['topQp'], currVLayer['topQp'],
            currVLayer['topQs'], currVLayer['topQs'])],
            dtype=VelocityLayer)
        currSLayer = create_from_vlayer(currVLayer, self.SWAVE)
        currPLayer = create_from_vlayer(currVLayer, self.PWAVE)
        # We know that the top is always a critical slowness so add 0
        self.criticalDepths[cd_count] = (0, 0, 0, 0)
        cd_count += 1
        # Check to see if starting in fluid zone.
        if inFluidZone is False and currVLayer['topSVelocity'] == 0:
            inFluidZone = True
            fluidZone = DepthRange(topDepth=currVLayer['topDepth'])
            currSLayer = currPLayer
        if minSSoFar > currSLayer['topP']:
            minSSoFar = currSLayer['topP']
        # P is not a typo, it represents slowness, not P-wave speed.
        if minPSoFar > currPLayer['topP']:
            minPSoFar = currPLayer['topP']

        for layerNum, layer in enumerate(self.vMod.layers):
            prevVLayer = currVLayer
            prevSLayer = currSLayer
            prevPLayer = currPLayer
            # Could make the following a deep copy, but not necessary.
            # Also don't just replace layer here and in the loop
            # control with currVLayer, or the reference to the first
            # zero thickness layer that has been initialised above
            # will break.
            currVLayer = layer
            # Check again if in fluid zone
            if inFluidZone is False and currVLayer['topSVelocity'] == 0:
                inFluidZone = True
                fluidZone = DepthRange(topDepth=currVLayer['topDepth'])
            # If already in fluid zone, check if exited (java line 909)
            if inFluidZone is True and currVLayer['topSVelocity'] != 0:
                if prevVLayer['botDepth'] > self.vMod.iocbDepth:
                    belowOuterCore = True
                inFluidZone = False
                fluidZone.botDepth = prevVLayer['botDepth']
                self.fluidLayerDepths.append(fluidZone)

            currPLayer = create_from_vlayer(currVLayer,
                                            self.PWAVE)
            # If we are in a fluid zone ( S velocity = 0.0 ) or if we are below
            # the outer core and allowInnerCoreS=false then use the P velocity
            # structure to look for critical points.
            if inFluidZone \
                    or (belowOuterCore and self.allowInnerCoreS is False):
                currSLayer = currPLayer
            else:
                currSLayer = create_from_vlayer(currVLayer,
                                                self.SWAVE)

            if prevSLayer['botP'] != currSLayer['topP'] \
                    or prevPLayer['botP'] != currPLayer['topP']:
                # a first order discontinuity
                self.criticalDepths[cd_count] = (
                    currSLayer['topDepth'],
                    layerNum,
                    -1,
                    -1)
                cd_count += 1
                if self.DEBUG:
                    print('First order discontinuity, depth =' +
                          str(currSLayer['topDepth']))
                    print('between' + str(prevPLayer), str(currPLayer))
                if inHighSlownessZoneS and currSLayer['topP'] < minSSoFar:
                    if self.DEBUG:
                        print("Top of current layer is the bottom"
                              " of a high slowness zone.")
                    highSlownessZoneS.botDepth = currSLayer['topDepth']
                    self.highSlownessLayerDepthsS.append(highSlownessZoneS)
                    inHighSlownessZoneS = False
                if inHighSlownessZoneP and currPLayer['topP'] < minPSoFar:
                    if self.DEBUG:
                        print("Top of current layer is the bottom"
                              " of a high slowness zone.")
                    highSlownessZoneP.botDepth = currSLayer['topDepth']
                    self.highSlownessLayerDepthsP.append(highSlownessZoneP)
                    inHighSlownessZoneP = False
                # Update minPSoFar and minSSoFar as all total reflections off
                # of the top of the discontinuity are ok even though below the
                # discontinuity could be the start of a high slowness zone.
                if minPSoFar > currPLayer['topP']:
                    minPSoFar = currPLayer['topP']
                if minSSoFar > currSLayer['topP']:
                    minSSoFar = currSLayer['topP']

                if inHighSlownessZoneS is False and (
                        prevSLayer['botP'] < currSLayer['topP'] or
                        currSLayer['topP'] < currSLayer['botP']):
                    # start of a high slowness zone S
                    if self.DEBUG:
                        print("Found S high slowness at first order " +
                              "discontinuity, layer = " + str(layerNum))
                    inHighSlownessZoneS = True
                    highSlownessZoneS = \
                        DepthRange(topDepth=currSLayer['topDepth'])
                    highSlownessZoneS.ray_param = minSSoFar
                if inHighSlownessZoneP is False and (
                        prevPLayer['botP'] < currPLayer['topP'] or
                        currPLayer['topP'] < currPLayer['botP']):
                    # start of a high slowness zone P
                    if self.DEBUG:
                        print("Found P high slowness at first order " +
                              "discontinuity, layer = " + str(layerNum))
                    inHighSlownessZoneP = True
                    highSlownessZoneP = \
                        DepthRange(topDepth=currPLayer['topDepth'])
                    highSlownessZoneP.ray_param = minPSoFar

            elif ((prevSLayer['topP'] - prevSLayer['botP']) *
                  (prevSLayer['botP'] - currSLayer['botP']) < 0) or (
                      (prevPLayer['topP'] - prevPLayer['botP']) *
                      (prevPLayer['botP'] - currPLayer['botP'])) < 0:
                # local slowness extrema, java l 1005
                self.criticalDepths[cd_count] = (
                    currSLayer['topDepth'],
                    layerNum,
                    -1,
                    -1)
                cd_count += 1
                if self.DEBUG:
                    print("local slowness extrema, depth=" +
                          str(currSLayer['topDepth']))
                # here is line 1014 of the java src!
                if inHighSlownessZoneP is False \
                        and currPLayer['topP'] < currPLayer['botP']:
                    if self.DEBUG:
                        print("start of a P high slowness zone, local "
                              "slowness extrema,minPSoFar= " + str(minPSoFar))
                    inHighSlownessZoneP = True
                    highSlownessZoneP = \
                        DepthRange(topDepth=currPLayer['topDepth'])
                    highSlownessZoneP.ray_param = minPSoFar
                if inHighSlownessZoneS is False \
                        and currSLayer['topP'] < currSLayer['botP']:
                    if self.DEBUG:
                        print("start of a S high slowness zone, local "
                              "slowness extrema, minSSoFar= " +
                              str(minSSoFar))
                    inHighSlownessZoneS = True
                    highSlownessZoneS = \
                        DepthRange(topDepth=currSLayer['topDepth'])
                    highSlownessZoneS.ray_param = minSSoFar

            if inHighSlownessZoneP and currPLayer['botP'] < minPSoFar:
                # P: layer contains the bottom of a high slowness zone. java
                #  l 1043
                if self.DEBUG:
                    print("layer contains the bottom of a P " +
                          "high slowness zone. minPSoFar=" + str(minPSoFar),
                          currPLayer)
                highSlownessZoneP.botDepth = self.findDepth_from_layers(
                    minPSoFar, layerNum, layerNum, self.PWAVE)
                self.highSlownessLayerDepthsP.append(highSlownessZoneP)
                inHighSlownessZoneP = False

            if inHighSlownessZoneS and currSLayer['botP'] < minSSoFar:
                # S: layer contains the bottom of a high slowness zone. java
                #  l 1043
                if self.DEBUG:
                    print("layer contains the bottom of a S " +
                          "high slowness zone. minSSoFar=" + str(minSSoFar),
                          currSLayer)
                # in fluid layers we want to check PWAVE structure
                # when looking for S wave critical points
                porS = (self.PWAVE if currSLayer == currPLayer else self.SWAVE)
                highSlownessZoneS.botDepth = self.findDepth_from_layers(
                    minSSoFar, layerNum, layerNum, porS)
                self.highSlownessLayerDepthsS.append(highSlownessZoneS)
                inHighSlownessZoneS = False
            if minPSoFar > currPLayer['botP']:
                minPSoFar = currPLayer['botP']
            if minPSoFar > currPLayer['topP']:
                minPSoFar = currPLayer['topP']
            if minSSoFar > currSLayer['botP']:
                minSSoFar = currSLayer['botP']
            if minSSoFar > currSLayer['topP']:
                minSSoFar = currSLayer['topP']
            if self.DEBUG and inHighSlownessZoneS:
                print("In S high slowness zone, layerNum = " + str(layerNum) +
                      " minSSoFar=" + str(minSSoFar))
            if self.DEBUG and inHighSlownessZoneP:
                print("In P high slowness zone, layerNum = " + str(layerNum) +
                      " minPSoFar=" + str(minPSoFar))

        # We know that the bottommost depth is always a critical slowness,
        # so we add vMod.getNumLayers()
        # java line 1094
        self.criticalDepths[cd_count] = (self.radiusOfEarth,
                                         self.vMod.getNumLayers(),
                                         -1,
                                         -1)
        cd_count += 1

        # Check if the bottommost depth is contained within a high slowness
        # zone, might happen in a flat non-whole-earth model
        if inHighSlownessZoneS:
            highSlownessZoneS.botDepth = currVLayer['botDepth']
            self.highSlownessLayerDepthsS.append(highSlownessZoneS)
        if inHighSlownessZoneP:
            highSlownessZoneP.botDepth = currVLayer['botDepth']
            self.highSlownessLayerDepthsP.append(highSlownessZoneP)

        # Check if the bottommost depth is contained within a fluid zone, this
        # would be the case if we have a non whole earth model with the bottom
        # in the outer core or if allowInnerCoreS == false and we want to use
        # the P velocity structure in the inner core.
        if inFluidZone:
            fluidZone['botDepth'] = currVLayer['botDepth']
            self.fluidLayerDepths.append(fluidZone)

        self.criticalDepths = self.criticalDepths[:cd_count]

        self.validate()

    def getNumLayers(self, isPWave):
        """
        Number of slowness layers.

        This is meant to return the number of P or S layers.

        :param isPWave: Return P layer count (``True``) or S layer count
            (``False``).
        :type isPWave: bool
        :returns: Number of slowness layers.
        :rtype: int
        """
        if isPWave:
            return len(self.PLayers)
        else:
            return len(self.SLayers)

    def findDepth_from_depths(self, ray_param, topDepth, botDepth, isPWave):
        """
        Find depth corresponding to a slowness between two given depths.

        The given depths are converted to layer numbers before calling
        :meth:`findDepth_from_layers`.

        :param ray_param: Slowness (aka ray parameter) to find, in s/km.
        :type ray_param: float
        :param topDepth: Top depth to search, in km.
        :type topDepth: float
        :param botDepth: Bottom depth to search, in km.
        :type botDepth: float
        :param isPWave: ``True`` if P wave or ``False`` for S wave.
        :type isPWave: bool

        :returns: Depth (in km) corresponding to the desired slowness.
        :rtype: float

        :raises SlownessModelError: If ``topCriticalLayer > botCriticalLayer``
            because there are no layers to search, or if there is an increase
            in slowness, i.e., a negative velocity gradient, that just balances
            the decrease in slowness due to the spherical Earth, or if the ray
            parameter ``p`` is not contained within the specified layer range.
        """
        topLayerNum = self.vMod.layerNumberBelow(topDepth)[0]
        if self.vMod.layers[topLayerNum]['botDepth'] == topDepth:
            topLayerNum += 1
        botLayerNum = self.vMod.layerNumberAbove(botDepth)[0]
        return self.findDepth_from_layers(ray_param, topLayerNum, botLayerNum,
                                          isPWave)

    def findDepth_from_layers(self, p, topCriticalLayer, botCriticalLayer,
                              isPWave):
        """
        Find depth corresponding to a slowness p between two velocity layers.

        Here, slowness is defined as ``(6731-depth) / velocity``, and sometimes
        called ray parameter. Both the top and the bottom velocity layers are
        included. We also check to see if the slowness is less than the bottom
        slowness of these layers but greater than the top slowness of the next
        deeper layer. This corresponds to a total reflection. In this case a
        check needs to be made to see if this is an S wave reflecting off of a
        fluid layer, use P velocity below in this case. We assume that slowness
        is monotonic within these layers and therefore there is only one depth
        with the given slowness. This means we return the first depth that we
        find.

        :param p: Slowness (aka ray parameter) to find, in s/km.
        :type p: float
        :param topCriticalLayer: Top layer number to search.
        :type topCriticalLayer: int
        :param botCriticalLayer: Bottom layer number to search.
        :type botCriticalLayer: int
        :param isPWave: ``True`` if P wave or ``False`` for S wave.
        :type isPWave: bool

        :returns: Depth (in km) corresponding to the desired slowness.
        :rtype: float

        :raises SlownessModelError: If ``topCriticalLayer > botCriticalLayer``
            because there are no layers to search, or if there is an increase
            in slowness, i.e., a negative velocity gradient, that just balances
            the decrease in slowness due to the spherical Earth, or if the ray
            parameter ``p`` is not contained within the specified layer range.
        """
        # topP = 1.1e300  # dummy numbers
        # botP = 1.1e300
        waveType = 'P' if isPWave else 'S'

        if topCriticalLayer > botCriticalLayer:
            raise SlownessModelError(
                "findDepth: no layers to search (wrong layer num?)")
        for layerNum in range(topCriticalLayer, botCriticalLayer + 1):
            velLayer = self.vMod.layers[layerNum]
            topVelocity = evaluateVelocityAtTop(velLayer, waveType)
            botVelocity = evaluateVelocityAtBottom(velLayer, waveType)
            topP = self.toSlowness(topVelocity, velLayer['topDepth'])
            botP = self.toSlowness(botVelocity, velLayer['botDepth'])
            # Check to see if we are within 'chatter level' (numerical
            # error) of the top or bottom and if so then return that depth.
            if abs(topP - p) < self.slowness_tolerance:
                return velLayer['topDepth']
            if abs(p - botP) < self.slowness_tolerance:
                return velLayer['botDepth']

            if (topP - p) * (p - botP) >= 0:
                # Found layer containing p.
                # We interpolate assuming that velocity is linear within
                # this interval. So slope is the slope for velocity versus
                # depth.
                slope = (botVelocity - topVelocity) / (
                    velLayer['botDepth'] - velLayer['topDepth'])
                depth = self.interpolate(p, topVelocity, velLayer['topDepth'],
                                         slope)
                return depth
            elif layerNum == topCriticalLayer \
                    and abs(p - topP) < self.slowness_tolerance:
                # Check to see if p is just outside the topmost layer. If so
                # then return the top depth.
                return velLayer['topDepth']

            # Is p a total reflection? botP is the slowness at the bottom
            # of the last velocity layer from the previous loop, set topP
            # to be the slowness at the top of the next layer.
            if layerNum < self.vMod.getNumLayers() - 1:
                velLayer = self.vMod.layers[layerNum + 1]
                topVelocity = evaluateVelocityAtTop(velLayer, waveType)
                if (isPWave is False and
                        np.any(self.depthInFluid(velLayer['topDepth']))):
                    # Special case for S waves above a fluid. If top next
                    # layer is in a fluid then we should set topVelocity to
                    # be the P velocity at the top of the layer.
                    topVelocity = evaluateVelocityAtTop(velLayer, 'P')

                topP = self.toSlowness(topVelocity, velLayer['topDepth'])
                if botP >= p >= topP:
                    return velLayer['topDepth']

        # noinspection PyUnboundLocalVariable
        if abs(p - botP) < self.slowness_tolerance:
            # java line 1275
            # Check to see if p is just outside the bottommost layer. If so
            # than return the bottom depth.
            print(" p is just outside the bottommost layer. This probably "
                  "shouldn't be allowed to happen!")
            # noinspection PyUnboundLocalVariable
            return velLayer.getBotDepth()

        raise SlownessModelError(
            "slowness p=" + str(p) +
            "is not contained within the specified layers." +
            " topCriticalLayer=" + str(topCriticalLayer) +
            " botCriticalLayer=" + str(botCriticalLayer))

    def toSlowness(self, velocity, depth):
        """
        Convert velocity at some depth to slowness.

        :param velocity: The velocity to convert, in km/s.
        :type velocity: float
        :param depth: The depth (in km) at which to perform the calculation.
            Must be less than the radius of the Earth defined in this model, or
            the result is undefined.
        :type depth: float

        :returns: The slowness, in s/km.
        :rtype: float
        """
        if velocity == 0:
            raise SlownessModelError(
                "toSlowness: velocity can't be zero, at depth" +
                str(depth),
                "Maybe related to using S velocities in outer core?")
        return (self.radiusOfEarth - depth) / velocity

    def interpolate(self, p, topVelocity, topDepth, slope):
        """
        Interpolate slowness to depth within a layer.

        We interpolate assuming that velocity is linear within
        this interval.

        All parameters must be of the same shape.

        :param p: The slowness to interpolate, in s/km.
        :type p: :class:`float` or :class:`~numpy.ndarray`
        :param topVelocity: The velocity (in km/s) at the top of the layer.
        :type topVelocity: :class:`float` or :class:`~numpy.ndarray`
        :param topDepth: The depth (in km) for the top of the layer.
        :type topDepth: :class:`float` or :class:`~numpy.ndarray`
        :param slope: The slope (in (km/s)/km)  for velocity versus depth.
        :type slope: :class:`float` or :class:`~numpy.ndarray`

        :returns: The depth (in km) of the slowness below the layer boundary.
        :rtype: :class:`float` or :class:`~numpy.ndarray`
        """
        denominator = p * slope + 1
        if np.any(denominator == 0):
            raise SlownessModelError(
                "Negative velocity gradient that just balances the slowness "
                "gradient of the spherical slowness, i.e. Earth flattening. "
                "Instructions unclear; explode.")
        else:
            depth = (self.radiusOfEarth +
                     p * (topDepth * slope - topVelocity)) / denominator
            return depth

    def depthInFluid(self, depth):
        """
        Determine if the given depth is contained within a fluid zone.

        The fluid zone includes its upper boundary but not its lower boundary.
        The top and bottom of the fluid zone are not returned as a DepthRange,
        just like in the Java code, despite its claims to the contrary.

        :param depth: The depth to check, in km.
        :type depth: :class:`~numpy.ndarray`, dtype = :class:`float`

        :returns: ``True`` if the depth is within a fluid zone, ``False``
            otherwise.
        :rtype: :class:`~numpy.ndarray` (dtype = :class:`bool`)
        """
        ret = np.zeros(shape=depth.shape, dtype=np.bool_)
        for elem in self.fluidLayerDepths:
            ret |= (elem.topDepth <= depth) & (depth < elem.botDepth)
        return ret

    def coarseSample(self):
        """
        Create a coarse slowness sampling of the velocity model (vMod).

        The resultant slowness layers will satisfy the maximum depth increments
        as well as sampling each point specified within the VelocityModel. The
        P and S sampling will also be compatible.
        """

        self.PLayers = create_from_vlayer(self.vMod.layers, self.PWAVE)

        with np.errstate(divide='ignore'):
            self.SLayers = create_from_vlayer(self.vMod.layers, self.SWAVE)
        mask = self.depthInFluid(self.vMod.layers['topDepth'])
        if not self.allowInnerCoreS:
            mask |= self.vMod.layers['topDepth'] >= self.vMod.iocbDepth
        self.SLayers[mask] = self.PLayers[mask]

        # Check for first order discontinuity. However, we only consider
        # S discontinuities in the inner core if allowInnerCoreS is true.
        above = self.vMod.layers[:-1]
        below = self.vMod.layers[1:]
        mask = np.logical_or(
            above['botPVelocity'] != below['topPVelocity'],
            np.logical_and(
                above['botSVelocity'] != below['topSVelocity'],
                np.logical_or(
                    self.allowInnerCoreS,
                    below['topDepth'] < self.vMod.iocbDepth)))

        index = np.where(mask)[0] + 1
        above = above[mask]
        below = below[mask]

        # If we are going from a fluid to a solid or solid to fluid, e.g., core
        # mantle or outer core to inner core then we need to use the P velocity
        # for determining the S discontinuity.
        topSVel = np.where(above['botSVelocity'] == 0,
                           above['botPVelocity'],
                           above['botSVelocity'])
        botSVel = np.where(below['topSVelocity'] == 0,
                           below['topPVelocity'],
                           below['topSVelocity'])

        # Add the layer, with zero thickness but nonzero slowness step,
        # corresponding to the discontinuity.
        currVLayer = np.empty(shape=above.shape, dtype=VelocityLayer)
        currVLayer['topDepth'] = above['botDepth']
        currVLayer['botDepth'] = above['botDepth']
        currVLayer['topPVelocity'] = above['botPVelocity']
        currVLayer['botPVelocity'] = below['topPVelocity']
        currVLayer['topSVelocity'] = topSVel
        currVLayer['botSVelocity'] = botSVel
        currVLayer['topDensity'].fill(DEFAULT_DENSITY)
        currVLayer['botDensity'].fill(DEFAULT_DENSITY)
        currVLayer['topQp'].fill(DEFAULT_QP)
        currVLayer['botQp'].fill(DEFAULT_QP)
        currVLayer['topQs'].fill(DEFAULT_QS)
        currVLayer['botQs'].fill(DEFAULT_QS)

        currPLayer = create_from_vlayer(currVLayer, self.PWAVE)
        self.PLayers = np.insert(self.PLayers, index, currPLayer)

        currSLayer = create_from_vlayer(currVLayer, self.SWAVE)
        mask2 = (above['botSVelocity'] == 0) & (below['topSVelocity'] == 0)
        if not self.allowInnerCoreS:
            mask2 |= currVLayer['topDepth'] >= self.vMod.iocbDepth
        currSLayer = np.where(mask2,
                              currPLayer,
                              currSLayer)
        self.SLayers = np.insert(self.SLayers, index, currSLayer)

        # Make sure that all high slowness layers are sampled exactly
        # at their bottom
        for highZone in self.highSlownessLayerDepthsS:
            sLayerNum = self.layerNumberAbove(highZone.botDepth, self.SWAVE)
            highSLayer = self.SLayers[sLayerNum]
            while highSLayer['topDepth'] == highSLayer['botDepth'] and (
                    (highSLayer['topP'] - highZone.ray_param) *
                    (highZone.ray_param - highSLayer['botP']) < 0):
                sLayerNum += 1
                highSLayer = self.SLayers[sLayerNum]
            if highZone.ray_param != highSLayer['botP']:
                self.addSlowness(highZone.ray_param, self.SWAVE)
        for highZone in self.highSlownessLayerDepthsP:
            sLayerNum = self.layerNumberAbove(highZone.botDepth, self.PWAVE)
            highSLayer = self.PLayers[sLayerNum]
            while highSLayer['topDepth'] == highSLayer['botDepth'] and (
                    (highSLayer['topP'] - highZone.ray_param) *
                    (highZone.ray_param - highSLayer['botP']) < 0):
                sLayerNum += 1
                highSLayer = self.PLayers[sLayerNum]
            if highZone.ray_param != highSLayer['botP']:
                self.addSlowness(highZone.ray_param, self.PWAVE)

        # Make sure P and S are consistent by adding discontinuities in one to
        # the other.
        # Numpy 1.6 compatibility
        # _tb = self.PLayers[['topP', 'botP']]
        _tb = np.vstack([self.PLayers['topP'], self.PLayers['botP']]).T.ravel()
        uniq = np.unique(_tb)
        for p in uniq:
            self.addSlowness(p, self.SWAVE)

        # Numpy 1.6 compatibility
        # _tb = self.PLayers[['topP', 'botP']]
        _tb = np.vstack([self.SLayers['topP'], self.SLayers['botP']]).T.ravel()
        uniq = np.unique(_tb)
        for p in uniq:
            self.addSlowness(p, self.PWAVE)

    def layerNumberAbove(self, depth, isPWave):
        """
        Find the index of the slowness layer that contains the given depth.

        Note that if the depth is a layer boundary, it returns the shallower
        of the two or possibly more (since total reflections are zero
        thickness layers) layers.

        .. seealso:: :meth:`layerNumberBelow`

        :param depth: The depth to find, in km.
        :type depth: :class:`float` or :class:`~numpy.ndarray`
        :param isPWave: Whether to look at P (``True``) velocity or S
            (``False``) velocity.
        :type isPWave: bool

        :returns: The slowness layer containing the requested depth.
        :rtype: :class:`int` or :class:`~numpy.ndarray` (dtype = :class:`int`,
            shape = ``depth.shape``)

        :raises SlownessModelError: If no layer in the slowness model contains
            the given depth.
        """
        if isPWave:
            layers = self.PLayers
        else:
            layers = self.SLayers

        # Check to make sure depth is within the range available
        if np.any(depth < layers[0]['topDepth']) or \
                np.any(depth > layers[-1]['botDepth']):
            raise SlownessModelError("No layer contains this depth")

        foundLayerNum = np.searchsorted(layers['topDepth'], depth)

        mask = foundLayerNum != 0
        if np.isscalar(foundLayerNum):
            if mask:
                foundLayerNum -= 1
        else:
            foundLayerNum[mask] -= 1

        return foundLayerNum

    def layerNumberBelow(self, depth, isPWave):
        """
        Find the index of the slowness layer that contains the given depth.

        Note that if the depth is a layer boundary, it returns the deeper of
        the two or possibly more (since total reflections are zero thickness
        layers) layers.

        .. seealso:: :meth:`layerNumberAbove`

        :param depth: The depth to find, in km.
        :type depth: :class:`float` or :class:`~numpy.ndarray`
        :param isPWave: Whether to look at P (``True``) velocity or S
            (``False``) velocity.
        :type isPWave: bool

        :returns: The slowness layer containing the requested depth.
        :rtype: :class:`int` or :class:`~numpy.ndarray` (dtype = :class:`int`,
            shape = ``depth.shape``)

        :raises SlownessModelError: If no layer in the slowness model contains
            the given depth.
        """
        if isPWave:
            layers = self.PLayers
        else:
            layers = self.SLayers

        # Check to make sure depth is within the range available
        if np.any(depth < layers[0]['topDepth']) or \
                np.any(depth > layers[-1]['botDepth']):
            raise SlownessModelError("No layer contains this depth")

        foundLayerNum = np.searchsorted(layers['botDepth'], depth,
                                        side='right')

        mask = foundLayerNum == layers.shape[0]
        if np.isscalar(foundLayerNum):
            if mask:
                foundLayerNum -= 1
        else:
            foundLayerNum[mask] -= 1

        return foundLayerNum

    def getSlownessLayer(self, layer, isPWave):
        """
        Return the SlownessLayer of the requested wave type.

        This is not meant to be a clone!

        :param layer: The number of the layer(s) to return.
        :type layer: :class:`int` or :class:`~numpy.ndarray` (dtype =
            :class:`int`)
        :param isPWave: Whether to return the P layer (``True``) or the S
            layer (``False``).
        :type isPWave: bool

        :returns: The slowness layer(s).
        :rtype: :class:`~numpy.ndarray` (dtype = :const:`SlownessLayer`,
            shape = ``layerNum.shape``)
        """
        if isPWave:
            return self.PLayers[layer]
        else:
            return self.SLayers[layer]

    def addSlowness(self, p, isPWave):
        """
        Add a ray parameter to the slowness sampling for the given wave type.

        Slowness layers are split as needed and P and S sampling are kept
        consistent within fluid layers. Note, this makes use of the velocity
        model, so all interpolation is linear in velocity, not in slowness!

        :param p: The slowness value to add, in s/km.
        :type p: float
        :param isPWave: Whether to add to the P wave (``True``) or the S wave
            (``False``) sampling.
        :type isPWave: bool
        """
        if isPWave:
            # NB Unlike Java (unfortunately) these are not modified in place!
            # NumPy arrays cannot have values inserted in place.
            layers = self.PLayers
            otherLayers = self.SLayers
            wave = 'P'
        else:
            layers = self.SLayers
            otherLayers = self.PLayers
            wave = 'S'

        # If depths are the same only need topVelocity, and just to verify we
        # are not in a fluid.
        nonzero = layers['topDepth'] != layers['botDepth']
        above = self.vMod.evaluateAbove(layers['botDepth'], wave)
        below = self.vMod.evaluateBelow(layers['topDepth'], wave)
        topVelocity = np.where(nonzero, below, above)
        botVelocity = np.where(nonzero, above, below)

        mask = ((layers['topP'] - p) * (p - layers['botP'])) > 0
        # Don't need to check for S waves in a fluid or in inner core if
        # allowInnerCoreS is False.
        if not isPWave:
            mask &= topVelocity != 0
            if not self.allowInnerCoreS:
                iocb_mask = layers['botDepth'] > self.vMod.iocbDepth
                mask &= ~iocb_mask

        index = np.where(mask)[0]

        botDepth = np.copy(layers['botDepth'])
        # Not a zero thickness layer, so calculate the depth for
        # the ray parameter.
        slope = ((botVelocity[nonzero] - topVelocity[nonzero]) /
                 (layers['botDepth'][nonzero] - layers['topDepth'][nonzero]))
        botDepth[nonzero] = self.interpolate(p, topVelocity[nonzero],
                                             layers['topDepth'][nonzero],
                                             slope)

        botLayer = np.empty(shape=index.shape, dtype=SlownessLayer)
        botLayer['topP'].fill(p)
        botLayer['topDepth'] = botDepth[mask]
        botLayer['botP'] = layers['botP'][mask]
        botLayer['botDepth'] = layers['botDepth'][mask]

        topLayer = np.empty(shape=index.shape, dtype=SlownessLayer)
        topLayer['topP'] = layers['topP'][mask]
        topLayer['topDepth'] = layers['topDepth'][mask]
        topLayer['botP'].fill(p)
        topLayer['botDepth'] = botDepth[mask]

        # numpy 1.6 compatibility
        otherIndex = np.where(otherLayers.reshape(1, -1) ==
                              layers[mask].reshape(-1, 1))
        layers[index] = botLayer
        layers = np.insert(layers, index, topLayer)
        if len(otherIndex[0]):
            otherLayers[otherIndex[1]] = botLayer[otherIndex[0]]
            otherLayers = np.insert(otherLayers, otherIndex[1],
                                    topLayer[otherIndex[0]])

        if isPWave:
            self.PLayers = layers
            self.SLayers = otherLayers
        else:
            self.SLayers = layers
            self.PLayers = otherLayers

    def ray_paramIncCheck(self):
        """
        Check that no slowness layer's ray parameter interval is too large.

        The limit is determined by ``self.maxDeltaP``.
        """
        for wave in [self.SWAVE, self.PWAVE]:
            # These might change with calls to addSlowness, so be sure we have
            # the correct copy.
            if wave == self.PWAVE:
                layers = self.PLayers
            else:
                layers = self.SLayers

            diff = layers['topP'] - layers['botP']
            absdiff = np.abs(diff)

            mask = absdiff > self.maxDeltaP
            diff = diff[mask]
            absdiff = absdiff[mask]
            topP = layers['topP'][mask]

            new_count = np.ceil(absdiff / self.maxDeltaP).astype(np.int_)
            steps = diff / new_count

            for start, Np, delta in zip(topP, new_count, steps):
                for j in range(1, Np):
                    newp = start + j * delta
                    self.addSlowness(newp, self.PWAVE)
                    self.addSlowness(newp, self.SWAVE)

    def depthIncCheck(self):
        """
        Check that no slowness layer is too thick.

        The maximum is determined by ``self.maxDepthInterval``.
        """
        for wave in [self.SWAVE, self.PWAVE]:
            # These might change with calls to addSlowness, so be sure we have
            # the correct copy.
            if wave == self.PWAVE:
                layers = self.PLayers
            else:
                layers = self.SLayers

            diff = layers['botDepth'] - layers['topDepth']

            mask = diff > self.maxDepthInterval
            diff = diff[mask]
            topDepth = layers['topDepth'][mask]

            new_count = np.ceil(diff / self.maxDepthInterval).astype(np.int_)
            steps = diff / new_count

            for start, Nd, delta in zip(topDepth, new_count, steps):
                new_depth = start + np.arange(1, Nd) * delta
                if wave == self.SWAVE:
                    velocity = self.vMod.evaluateAbove(new_depth, 'S')

                    smask = velocity == 0
                    if not self.allowInnerCoreS:
                        smask |= new_depth >= self.vMod.iocbDepth
                    velocity[smask] = self.vMod.evaluateAbove(new_depth[smask],
                                                              'P')

                    slowness = self.toSlowness(velocity, new_depth)
                else:
                    slowness = self.toSlowness(
                        self.vMod.evaluateAbove(new_depth, 'P'),
                        new_depth)

                for p in slowness:
                    self.addSlowness(p, self.PWAVE)
                    self.addSlowness(p, self.SWAVE)

    def distanceCheck(self):
        """
        Check that no slowness layer is too wide or undersampled.

        The width must be less than ``self.maxRangeInterval`` and the
        (estimated) error due to linear interpolation must be less than
        ``self.maxInterpError``.
        """
        for currWaveType in [self.SWAVE, self.PWAVE]:
            isCurrOK = False
            isPrevOK = False
            prevPrevTD = None
            prevTD = None
            currTD = None
            j = 0
            sLayer = self.getSlownessLayer(j, currWaveType)
            while j < self.getNumLayers(currWaveType):
                prevSLayer = sLayer
                sLayer = self.getSlownessLayer(j, currWaveType)
                if (self.depthInHighSlowness(sLayer['botDepth'],
                                             sLayer['botP'],
                                             currWaveType) is False and
                    self.depthInHighSlowness(sLayer['topDepth'],
                                             sLayer['topP'],
                                             currWaveType) is False):
                    # Don't calculate prevTD if we can avoid it
                    if isCurrOK:
                        if isPrevOK:
                            prevPrevTD = prevTD
                        else:
                            prevPrevTD = None
                        prevTD = currTD
                        isPrevOK = True
                    else:
                        prevTD = self.approxDistance(j - 1, sLayer['topP'],
                                                     currWaveType)
                        isPrevOK = True
                    currTD = self.approxDistance(j, sLayer['botP'],
                                                 currWaveType)
                    isCurrOK = True
                    # Check for jump of too great distance
                    if (abs(prevTD['dist'] - currTD['dist']) >
                            self.maxRangeInterval and
                            abs(sLayer['topP'] - sLayer['botP']) >
                            2 * self.minDeltaP):
                        if self.DEBUG:
                            print("At " + str(j) + " Distance jump too great ("
                                  ">maxRangeInterval " +
                                  str(self.maxRangeInterval) + "), adding "
                                  "slowness. ")
                        p = (sLayer['topP'] + sLayer['botP']) / 2
                        self.addSlowness(p, self.PWAVE)
                        self.addSlowness(p, self.SWAVE)
                        currTD = prevTD
                        prevTD = prevPrevTD
                    else:
                        # Make guess as to error estimate due to linear
                        # interpolation if it is not ok, then we split both
                        # the previous and current slowness layers, this has
                        # the nice, if unintended, consequence of adding
                        # extra samples in the neighborhood of poorly
                        # sampled caustics.
                        splitRayParam = (sLayer['topP'] + sLayer['botP']) / 2
                        allButLayer = self.approxDistance(j - 1, splitRayParam,
                                                          currWaveType)
                        splitLayer = np.array([(
                            sLayer['topP'], sLayer['topDepth'], splitRayParam,
                            bullenDepthFor(sLayer, splitRayParam,
                                           self.radiusOfEarth))],
                            dtype=SlownessLayer)
                        justLayerTime, justLayerDist = bullenRadialSlowness(
                            splitLayer,
                            splitRayParam,
                            self.radiusOfEarth)
                        splitTD = np.array([(
                            splitRayParam,
                            allButLayer['time'] + 2 * justLayerTime,
                            allButLayer['dist'] + 2 * justLayerDist,
                            0)],
                            dtype=TimeDist)
                        # Python standard division is not IEEE compliant,
                        # as “The IEEE 754 standard specifies that every
                        # floating point arithmetic operation, including
                        # division by zero, has a well-defined result”.
                        # Use numpy's division instead by using np.array:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            diff = (currTD['time'] -
                                    ((splitTD['time'] - prevTD['time']) *
                                     ((currTD['dist'] -
                                       prevTD['dist']) /
                                        (splitTD['dist'] -
                                         prevTD['dist'])) + prevTD['time']))
                        if abs(diff) > self.maxInterpError:
                            p1 = (prevSLayer['topP'] + prevSLayer['botP']) / 2
                            p2 = (sLayer['topP'] + sLayer['botP']) / 2
                            self.addSlowness(p1, self.PWAVE)
                            self.addSlowness(p1, self.SWAVE)
                            self.addSlowness(p2, self.PWAVE)
                            self.addSlowness(p2, self.SWAVE)
                            currTD = prevPrevTD
                            isPrevOK = False
                            if j > 0:
                                # Back up one step unless we are at the
                                # beginning, then stay put.
                                j -= 1
                                sLayer = self.getSlownessLayer(
                                    j - 1 if j - 1 >= 0 else 0, currWaveType)
                                # This sLayer will become prevSLayer in the
                                # next loop.
                            else:
                                isPrevOK = False
                                isCurrOK = False
                        else:
                            j += 1
                            if self.DEBUG and j % 10 == 0:
                                print(j)
                else:
                    prevPrevTD = None
                    prevTD = None
                    currTD = None
                    isCurrOK = False
                    isPrevOK = False
                    j += 1
                    if self.DEBUG and j % 100 == 0:
                        print(j)
            if self.DEBUG:
                print("Number of " + ("P" if currWaveType else "S") +
                      " slowness layers: " + str(j))

    def depthInHighSlowness(self, depth, ray_param, isPWave):
        """
        Determine if depth and slowness are within a high slowness zone.

        Whether the high slowness zone includes its upper boundary and its
        lower boundaries depends upon the ray parameter. The slowness at the
        depth is needed because if depth happens to correspond to a
        discontinuity that marks the bottom of the high slowness zone but the
        ray is actually a total reflection then it is not part of the high
        slowness zone. The ray parameter that delimits the zone, i.e., it can
        turn at the top and the bottom, is in the zone at the top, but out of
        the zone at the bottom. (?)

        NOTE: I changed this method a bit by throwing out some seemingly
        useless copying of the values in tempRange, which I think are not used
        anywhere else.

        :param depth: The depth to check, in km.
        :type depth: float
        :param ray_param: The slowness to check, in s/km.
        :type ray_param: float
        :param isPWave: Whether to check the P wave (``True``) or the S wave
            (``False``).
        :type isPWave: bool

        :returns: ``True`` if within a high slowness zone, ``False`` otherwise.
        :rtype: bool
        """
        if isPWave:
            highSlownessLayerDepths = self.highSlownessLayerDepthsP
        else:
            highSlownessLayerDepths = self.highSlownessLayerDepthsS
        for tempRange in highSlownessLayerDepths:
            if tempRange.topDepth <= depth <= tempRange.botDepth:
                if ray_param > tempRange.ray_param \
                        or (ray_param == tempRange.ray_param and
                            depth == tempRange.topDepth):
                    return True
        return False

    def approxDistance(self, slownessTurnLayer, p, isPWave):
        """
        Approximate distance for ray turning at the bottom of a layer.

        Generates approximate distance, in radians, for a ray from a surface
        source that turns at the bottom of the given slowness layer.

        :param slownessTurnLayer: The number of the layer at which the ray
            should turn.
        :type slownessTurnLayer: int
        :param p: The slowness to calculate, in s/km.
        :type p: float
        :param isPWave: Whether to use the P (``True``) or S (``False``) wave.
        :type isPWave: bool

        :returns: The time (in s) and distance (in rad) the ray travels.
        :rtype: :class:`~numpy.ndarray` (dtype = :const:`TimeDist`, shape =
            (``slownessTurnLayer``, ))
        """
        # First, if the slowness model contains less than slownessTurnLayer
        # elements we can't calculate a distance.
        if slownessTurnLayer >= self.getNumLayers(isPWave):
            raise SlownessModelError(
                "Can't calculate a distance when getNumLayers() is smaller "
                "than the given slownessTurnLayer!")
        if p < 0:
            raise SlownessModelError("Ray parameter must not be negative!")
        td = np.zeros(1, dtype=TimeDist)
        td['p'] = p
        layerNum = np.arange(0, slownessTurnLayer + 1)
        if len(layerNum):
            time, dist = self.layerTimeDist(p, layerNum, isPWave)
            # Return 2* distance and time because there is a downgoing as well
            # as an upgoing leg, which are equal since this is for a surface
            # source.
            td['time'] = 2 * np.sum(time)
            td['dist'] = 2 * np.sum(dist)
        return td

    def layerTimeDist(self, sphericalRayParam, layerNum, isPWave, check=True):
        """
        Calculate time and distance for a ray passing through a layer.

        Calculates the time and distance increments accumulated by a ray of
        spherical ray parameter ``p`` when passing through layer ``layerNum``.
        Note that this gives half of the true range and time increments since
        there will be both an upgoing and a downgoing path. It also only does
        the calculation for the simple cases of the centre of the Earth, where
        the ray parameter is zero, or for constant velocity layers. Otherwise,
        it calls :func:`~.bullenRadialSlowness`.

        Either ``sphericalRayParam`` or ``layerNum`` must be 0-D, or they must
        have the same shape.

        :param sphericalRayParam: The spherical ray parameter of the ray(s), in
            s/km.
        :type sphericalRayParam: :class:`float` or :class:`~numpy.ndarray`
        :param layerNum: The layer(s) in which the calculation should be done.
        :type layerNum: :class:`float` or :class:`~numpy.ndarray`
        :param isPWave: Whether to look at the P (``True``) or S (``False``)
            wave.
        :type isPWave: bool
        :param check: Whether to perform checks of input consistency.
        :type check: bool

        :returns: The time (in s) and distance (in rad) increments for the
            specified ray(s) and layer(s).
        :rtype: :class:`~numpy.ndarray` (dtype = :const:`TimeDist`, shape =
            ``sphericalRayParam.shape`` or ``layerNum.shape``)

        :raises SlownessModelError: If the ray with the given spherical ray
            parameter cannot propagate within this layer, or if the ray turns
            within this layer but not at the bottom. These checks may be
            bypassed by specifying ``check=False``.
        """
        sphericalLayer = self.getSlownessLayer(layerNum, isPWave)

        # First make sure that a ray with this ray param can propagate
        # within this layer and doesn't turn in the middle of the layer. If
        # not, raise error.
        if check and sphericalRayParam > max(np.max(sphericalLayer['topP']),
                                             np.max(sphericalLayer['botP'])):
            raise SlownessModelError("Ray cannot propagate within this layer, "
                                     "given ray param too large.")
        if np.any(sphericalRayParam < 0):
            raise SlownessModelError("Ray parameter must not be negative!")
        if check and sphericalRayParam > min(np.min(sphericalLayer['topP']),
                                             np.min(sphericalLayer['botP'])):
            raise SlownessModelError("Ray turns in the middle of this layer! "
                                     "layerNum = " + str(layerNum))

        pdim = np.ndim(sphericalRayParam)
        ldim = np.ndim(layerNum)
        if ldim == 1 and pdim == 0:
            time = np.empty(shape=layerNum.shape, dtype=np.float_)
            dist = np.empty(shape=layerNum.shape, dtype=np.float_)
        elif ldim == 0 and pdim == 1:
            time = np.empty(shape=sphericalRayParam.shape, dtype=np.float_)
            dist = np.empty(shape=sphericalRayParam.shape, dtype=np.float_)
        elif ldim == pdim and (ldim == 0 or
                               layerNum.shape == sphericalRayParam.shape):
            time = np.empty(shape=layerNum.shape, dtype=np.float_)
            dist = np.empty(shape=layerNum.shape, dtype=np.float_)
        else:
            raise TypeError('Either sphericalRayParam or layerNum must be 0D, '
                            'or they must have the same shape.')

        # Check to see if this layer has zero thickness, if so then it is
        # from a critically reflected slowness sample. That means just
        # return 0 for time and distance increments.
        zero_thick = sphericalLayer['topDepth'] == sphericalLayer['botDepth']
        if ldim == 0:
            if zero_thick:
                time.fill(0)
                dist.fill(0)
                return time, dist
            else:
                zero_thick = np.zeros(shape=time.shape, dtype=np.bool_)

        leftover = ~zero_thick
        time[zero_thick] = 0
        dist[zero_thick] = 0

        # Check to see if this layer contains the centre of the Earth. If so
        # then the spherical ray parameter should be 0.0 and we calculate the
        # range and time increments using a constant velocity layer (sphere).
        # See eqns. 43 and 44 of [Buland1983]_, although we implement them
        # slightly differently. Note that the distance and time increments are
        # for just downgoing or just upgoing, i.e. from the top of the layer
        # to the centre of the earth or vice versa but not both. This is in
        # keeping with the convention that these are one way distance and time
        # increments. We will multiply the result by 2 at the end, or if we are
        # doing a 1.5D model, the other direction may be different. The time
        # increment for a ray of zero ray parameter passing half way through a
        # sphere of constant velocity is just the spherical slowness at the top
        # of the sphere. An amazingly simple result!
        centre_layer = np.logical_and(leftover, np.logical_and(
            sphericalRayParam == 0,
            sphericalLayer['botDepth'] == self.radiusOfEarth))
        leftover &= ~centre_layer
        if np.any(layerNum[centre_layer] != self.getNumLayers(isPWave) - 1):
            raise SlownessModelError("There are layers deeper than the "
                                     "centre of the Earth!")
        time[centre_layer] = sphericalLayer['topP'][centre_layer]
        dist[centre_layer] = math.pi / 2

        # Now we check to see if this is a constant velocity layer and if so
        # than we can do a simple triangle calculation to get the range and
        # time increments. To get the time increment we first calculate the
        # path length through the layer using the law of cosines, noting
        # that the angle at the top of the layer can be obtained from the
        # spherical Snell's Law. The time increment is just the path length
        # divided by the velocity. To get the distance we first find the
        # angular distance traveled, using the law of sines.
        topRadius = self.radiusOfEarth - sphericalLayer['topDepth']
        botRadius = self.radiusOfEarth - sphericalLayer['botDepth']
        with np.errstate(invalid='ignore'):
            vel = botRadius / sphericalLayer['botP']
            constant_velocity = np.logical_and(
                leftover,
                np.abs(topRadius / sphericalLayer['topP'] -
                       vel) < self.slowness_tolerance)
        leftover &= ~constant_velocity
        topRadius = topRadius[constant_velocity]
        botRadius = botRadius[constant_velocity]
        vel = vel[constant_velocity]
        if pdim:
            ray_param_const_velocity = sphericalRayParam[constant_velocity]
        else:
            ray_param_const_velocity = sphericalRayParam

        # In cases of a ray turning at the bottom of the layer numerical
        # round-off can cause botTerm to be very small (1e-9) but
        # negative which causes the sqrt to raise an error. We check for
        # values that are within the numerical chatter of zero and just
        # set them to zero.
        topTerm = topRadius ** 2 - (ray_param_const_velocity * vel) ** 2
        topTerm[np.abs(topTerm) < self.slowness_tolerance] = 0

        # In this case the ray turns at the bottom of this layer so
        # sphericalRayParam*vel == botRadius, and botTerm should be
        # zero. We check for this case specifically because
        # numerical chatter can cause small round-off errors that
        # lead to botTerm being negative, causing a sqrt error.
        botTerm = np.zeros(shape=topTerm.shape)
        mask = (ray_param_const_velocity !=
                sphericalLayer['botP'][constant_velocity])
        if pdim:
            botTerm[mask] = botRadius[mask] ** 2 - (
                ray_param_const_velocity[mask] * vel[mask]) ** 2
        else:
            botTerm[mask] = botRadius[mask] ** 2 - (
                ray_param_const_velocity * vel[mask]) ** 2

        b = np.sqrt(topTerm) - np.sqrt(botTerm)
        time[constant_velocity] = b / vel
        dist[constant_velocity] = np.arcsin(
            b * ray_param_const_velocity * vel / (topRadius * botRadius))

        # If the layer is not a constant velocity layer or the centre of the
        #  Earth and p is not zero we have to do it the hard way:
        time[leftover], dist[leftover] = bullenRadialSlowness(
            sphericalLayer[leftover] if ldim else sphericalLayer,
            sphericalRayParam[leftover] if pdim else sphericalRayParam,
            self.radiusOfEarth,
            check=check)

        if check and (np.any(time < 0) or np.any(np.isnan(time)) or
                      np.any(dist < 0) or np.any(np.isnan(dist))):
            raise SlownessModelError(
                "layer time|dist < 0 or NaN.")

        return time, dist

    def fixCriticalPoints(self):
        """
        Reset the slowness layers that correspond to critical points.
        """
        self.criticalDepths['pLayerNum'] = self.layerNumberBelow(
            self.criticalDepths['depth'],
            self.PWAVE)
        sLayer = self.getSlownessLayer(self.criticalDepths['pLayerNum'],
                                       self.PWAVE)

        # We want the last critical point to be the bottom of the last layer.
        mask = ((self.criticalDepths['pLayerNum'] == len(self.PLayers) - 1) &
                (sLayer['botDepth'] == self.criticalDepths['depth']))
        self.criticalDepths['pLayerNum'][mask] += 1

        self.criticalDepths['sLayerNum'] = self.layerNumberBelow(
            self.criticalDepths['depth'],
            self.SWAVE)
        sLayer = self.getSlownessLayer(self.criticalDepths['sLayerNum'],
                                       self.SWAVE)

        # We want the last critical point to be the bottom of the last layer.
        mask = ((self.criticalDepths['sLayerNum'] == len(self.SLayers) - 1) &
                (sLayer['botDepth'] == self.criticalDepths['depth']))
        self.criticalDepths['sLayerNum'][mask] += 1

    def validate(self):
        """
        Perform consistency check on the slowness model.

        In Java, there is a separate validate method defined in the
        SphericalSModel subclass and as such overrides the validate in
        SlownessModel, but it itself calls the super method (by
        super.validate()), i.e. the code above. Both are merged here (in
        fact, it only contained one test).
        """
        if self.radiusOfEarth <= 0:
            raise SlownessModelError("Radius of Earth must be positive.")
        if self.maxDepthInterval <= 0:
            raise SlownessModelError(
                "maxDepthInterval must be positive and non-zero.")
        # Check for inconsistencies in high slowness zones.
        for isPWave in [self.PWAVE, self.SWAVE]:
            if isPWave:
                highSlownessLayerDepths = self.highSlownessLayerDepthsP
            else:
                highSlownessLayerDepths = self.highSlownessLayerDepthsS
            prevDepth = -1e300
            for highSZoneDepth in highSlownessLayerDepths:
                if highSZoneDepth.topDepth >= highSZoneDepth.botDepth:
                    raise SlownessModelError(
                        "High Slowness zone has zero or negative thickness!")
                if highSZoneDepth.topDepth <= prevDepth:
                    raise SlownessModelError(
                        "High Slowness zone overlaps previous zone.")
                prevDepth = highSZoneDepth.botDepth
        # Check for inconsistencies in fluid zones.
        prevDepth = -1e300
        for fluidZone in self.fluidLayerDepths:
            if fluidZone.topDepth >= fluidZone.botDepth:
                raise SlownessModelError(
                    "Fluid zone has zero or negative thickness!")
            if fluidZone.topDepth <= prevDepth:
                raise SlownessModelError("Fluid zone overlaps previous zone.")
            prevDepth = fluidZone.botDepth
        # Check for inconsistencies in slowness layers.
        for layers in [self.PLayers, self.SLayers]:
            if layers is None:
                continue

            if np.any(np.isnan(layers['topP']) | np.isnan(layers['botP'])):
                raise SlownessModelError("Slowness layer has NaN values.")
            if np.any((layers['topP'] < 0) | (layers['botP'] < 0)):
                raise SlownessModelError(
                    "Slowness layer has negative slowness.")
            if np.any(layers['topP'][1:] != layers['botP'][:-1]):
                raise SlownessModelError(
                    "Slowness layer slowness does not agree with "
                    "previous layer (at same depth)!")

            if np.any(layers['topDepth'] > layers['botDepth']):
                raise SlownessModelError(
                    "Slowness layer has negative thickness.")

            if layers['topDepth'][0] > 0:
                raise SlownessModelError("Gap between slowness layers!")
            if np.any(layers['topDepth'][1:] > layers['botDepth'][:-1]):
                raise SlownessModelError("Gap between slowness layers!")

            if layers['topDepth'][0] < 0:
                raise SlownessModelError("Slowness layer overlaps previous!")
            if np.any(layers['topDepth'][1:] < layers['botDepth'][:-1]):
                raise SlownessModelError("Slowness layer overlaps previous!")

            if np.any(np.isnan(layers['topDepth']) |
                      np.isnan(layers['botDepth'])):
                raise SlownessModelError(
                    "Slowness layer depth (top or bottom) is NaN!")

            if np.any(layers['botDepth'] > self.radiusOfEarth):
                raise SlownessModelError(
                    "Slowness layer has a depth larger than radius of the "
                    "Earth.")

        # Everything seems OK.
        return True

    def getMinTurnRayParam(self, depth, isPWave):
        """
        Find minimum slowness, turning but not reflected, at or above a depth.

        Normally this is the slowness sample at the given depth, but if the
        depth is within a high slowness zone, then it may be smaller.

        :param depth: The depth to search for, in km.
        :type depth: float
        :param isPWave: Whether to search the P (``True``) or S (``False``)
            wave.
        :type isPWave: bool

        :returns: The minimum ray parameter, in s/km.
        :rtype: float
        """
        minPSoFar = 1e300
        if self.depthInHighSlowness(depth, 1e300, isPWave):
            for sLayer in (self.PLayers if isPWave else self.SLayers):
                if sLayer['botDepth'] == depth:
                    minPSoFar = min(minPSoFar, sLayer['botP'])
                    return minPSoFar
                elif sLayer['botDepth'] > depth:
                    minPSoFar = min(minPSoFar,
                                    evaluateAtBullen(sLayer, depth,
                                                     self.radiusOfEarth))
                    return minPSoFar
                else:
                    minPSoFar = min(minPSoFar, sLayer['botP'])
        else:
            sLayer = self.getSlownessLayer(
                self.layerNumberAbove(depth, isPWave), isPWave)
            if depth == sLayer['botDepth']:
                minPSoFar = sLayer['botP']
            else:
                minPSoFar = evaluateAtBullen(sLayer, depth, self.radiusOfEarth)
        return minPSoFar

    def getMinRayParam(self, depth, isPWave):
        """
        Find minimum slowness, turning or reflected, at or above a depth.

        Normally this is the slowness sample at the given depth, but if the
        depth is within a high slowness zone, then it may be smaller. Also, at
        first order discontinuities, there may be many slowness samples at the
        same depth.

        :param depth: The depth to search for, in km.
        :type depth: float
        :param isPWave: Whether to search the P (``True``) or S (``False``)
            wave.
        :type isPWave: bool

        :returns: The minimum ray parameter, in s/km.
        :rtype: float
        """
        minPSoFar = self.getMinTurnRayParam(depth, isPWave)
        sLayerAbove = self.getSlownessLayer(
            self.layerNumberAbove(depth, isPWave), isPWave)
        sLayerBelow = self.getSlownessLayer(
            self.layerNumberBelow(depth, isPWave), isPWave)
        if sLayerAbove['botDepth'] == depth:
            minPSoFar = min(minPSoFar, sLayerAbove['botP'],
                            sLayerBelow['topP'])
        return minPSoFar

    def splitLayer(self, depth, isPWave):
        """
        Split a slowness layer into two slowness layers.

        The interpolation for splitting a layer is a Bullen p=Ar^B and so does
        not directly use information from the VelocityModel.

        :param depth: The depth at which attempt a split, in km.
        :type depth: float
        :param isPWave: Whether to split based on P (``True``) or S (``False``)
            wave.
        :type isPWave: bool

        :returns: Information about the split as (or if) it was performed, such
            that:

            * ``neededSplit=True`` if a layer was actually split;
            * ``movedSample=True`` if a layer was very close, and so moving the
              layer's depth is better than making a very thin layer;
            * ``ray_param=...``, the new ray parameter (in s/km), if the layer
              was split.

        :rtype: :class:`~.SplitLayerInfo`
        """
        layerNum = self.layerNumberAbove(depth, isPWave)
        sLayer = self.getSlownessLayer(layerNum, isPWave)
        if sLayer['topDepth'] == depth or sLayer['botDepth'] == depth:
            # Depth is already on a slowness layer boundary so no need to
            # split any slowness layers.
            return SplitLayerInfo(self, False, False, 0)
        elif abs(sLayer['topDepth'] - depth) < 0.000001:
            # Check for very thin layers, just move the layer to hit the
            # boundary.
            outLayers = self.PLayers if isPWave else self.SLayers
            outLayers[layerNum] = np.array([(sLayer['topP'], depth,
                                             sLayer['botP'],
                                             sLayer['botDepth'])],
                                           dtype=SlownessLayer)
            sLayer = self.getSlownessLayer(layerNum - 1, isPWave)
            outLayers[layerNum - 1] = np.array([(sLayer['topP'],
                                                 sLayer['topDepth'],
                                                 sLayer['botP'], depth)],
                                               dtype=SlownessLayer)
            out = self
            out.PLayers = outLayers if isPWave else self.PLayers
            out.SLayers = outLayers if isPWave else self.SLayers
            return SplitLayerInfo(out, False, True, sLayer['botP'])
        elif abs(depth - sLayer['botDepth']) < 0.000001:
            # As above.
            outLayers = self.PLayers if isPWave else self.SLayers
            outLayers[layerNum] = np.array([(sLayer['topP'],
                                             sLayer['topDepth'],
                                             sLayer['botP'], depth)],
                                           dtype=SlownessLayer)
            sLayer = self.getSlownessLayer(layerNum + 1, isPWave)
            outLayers[layerNum + 1] = np.array([(sLayer['topP'], depth,
                                                 sLayer['botP'],
                                                 sLayer['botDepth'])],
                                               dtype=SlownessLayer)
            out = self
            out.PLayers = outLayers if isPWave else self.PLayers
            out.SLayers = outLayers if isPWave else self.SLayers
            return SplitLayerInfo(out, False, True, sLayer['botP'])
        else:
            # Must split properly.
            p = evaluateAtBullen(sLayer, depth, self.radiusOfEarth)
            topLayer = np.array([(sLayer['topP'], sLayer['topDepth'],
                                  p, depth)],
                                dtype=SlownessLayer)
            botLayer = np.array([(p, depth,
                                  sLayer['botP'], sLayer['botDepth'])],
                                dtype=SlownessLayer)
            outLayers = self.PLayers if isPWave else self.SLayers
            outLayers[layerNum] = botLayer
            outLayers = np.insert(outLayers, layerNum, topLayer)
            # Fix critical layers since we added a slowness layer.
            outCriticalDepths = self.criticalDepths
            _fixCriticalDepths(outCriticalDepths, layerNum, isPWave)
            if isPWave:
                outPLayers = outLayers
                outSLayers = self._fixOtherLayers(self.SLayers, p, sLayer,
                                                  topLayer, botLayer,
                                                  outCriticalDepths, False)
            else:
                outPLayers = self._fixOtherLayers(self.PLayers, p, sLayer,
                                                  topLayer, botLayer,
                                                  outCriticalDepths, True)
                outSLayers = outLayers
            out = self
            out.criticalDepths = outCriticalDepths
            out.PLayers = outPLayers
            out.SLayers = outSLayers
            return SplitLayerInfo(out, True, False, p)

    def _fixOtherLayers(self, otherLayers, p, changedLayer, newTopLayer,
                        newBotLayer, criticalDepths, isPWave):
        """
        Fix other wave layers when a split is made.

        This performs the second split of the *other* wave type when a split is
        made by :meth:`splitLayer`.
        """
        out = otherLayers
        # Make sure to keep sampling consistent. If in a fluid, both wave
        # types will share a single slowness layer.

        otherIndex = np.where(otherLayers == changedLayer)
        if len(otherIndex[0]):
            out[otherIndex[0]] = newBotLayer
            out = np.insert(out, otherIndex[0], newTopLayer)

        for otherLayerNum, sLayer in enumerate(out.copy()):
            if (sLayer['topP'] - p) * (p - sLayer['botP']) > 0:
                # Found a slowness layer with the other wave type that
                # contains the new slowness sample.
                topLayer = np.array([(sLayer['topP'], sLayer['topDepth'], p,
                                     bullenDepthFor(sLayer, p,
                                                    self.radiusOfEarth))],
                                    dtype=SlownessLayer)
                botLayer = np.array([(p, topLayer['botDepth'], sLayer['botP'],
                                      sLayer['botDepth'])],
                                    dtype=SlownessLayer)
                out[otherLayerNum] = botLayer
                out = np.insert(out, otherLayerNum, topLayer)
                # Fix critical layers since we have added a slowness layer.
                _fixCriticalDepths(criticalDepths, otherLayerNum, not isPWave)
                # Skip next layer as it was just added: achieved by slicing
                # the list iterator.

        return out
