from math import pi
from taupy.VelocityLayer import VelocityLayer
from taupy.SlownessLayer import SlownessLayer


class SlownessModelError(Exception):
    pass


class CriticalDepth:
    def __init__(self, depth, velLayerNum, pLayerNum, sLayerNum):
        self.depth = depth
        self.velLayerNum = velLayerNum
        self.sLayerNum = pLayerNum
        self.sLayerNum = pLayerNum


class DepthRange:
    def __init__(self, topDepth=None, botDepth=None, rayParam=-1):
        self.topDepth = topDepth
        self.botDepth = botDepth
        self.rayParam = rayParam


class SlownessModel(object):
    """This class provides storage and methods for generating slowness-depth pairs."""
    DEBUG = False
    DEFAULT_SLOWNESS_TOLERANCE = 1e-16
    radiusOfEarth = 6371.0

    # NB if the following are actually cleared (lists are mutable) every
    # time createSample is called, maybe it would be better to just put these
    # initialisations into the relevant methods? They do have to be persistent across
    # method calls in createSample though (maybe??).

    # Stores the layer number for layers in the velocity model with a critical
    # point at their top. These form the "branches" of slowness sampling.
    criticalDepths = []  # will be list of CriticalDepth objects
    # Store depth ranges that contains a high slowness zone for P/S. Stored as
    # DepthRange objects, containing the top depth and bottom depth.
    highSlownessLayerDepthsP = []  # will be list of DepthRanges
    highSlownessLayerDepthsS = []
    # Stores depth ranges that are fluid, ie S velocity is zero. Stored as
    # DepthRange objects, containing the top depth and bottom depth.
    fluidLayerDepths = []

    # For methods that have an isPWave parameter
    SWAVE = False
    PWAVE = True

    def __init__(self, vMod, minDeltaP=0.1, maxDeltaP=11, maxDepthInterval=115, maxRangeInterval=2.5 * pi / 180,
                 maxInterpError=0.05, allowInnerCoreS=True, slowness_tolerance=DEFAULT_SLOWNESS_TOLERANCE):

        self.vMod = vMod
        self.minDeltaP = minDeltaP
        self.maxDeltaP = maxDeltaP
        self.maxDepthInterval = maxDepthInterval
        self.maxRangeInterval = maxRangeInterval
        self.maxInterpError = maxInterpError
        self.allowInnerCoreS = allowInnerCoreS
        self.slowness_tolerance = slowness_tolerance
        self.createSample()

    def createSample(self):
        """ This method takes a velocity model and creates a vector containing
        slowness-depth layers that, hopefully, adequately sample both slowness
        and depth so that the travel time as a function of distance can be
        reconstructed from the theta function."""
        # Some checks on the velocity model
        if self.vMod.validate() is False:
            raise SlownessModelError("Error in velocity model (vMod.validate failed)!")
        if self.vMod.getNumLayers() == 0:
            raise SlownessModelError("velModel.getNumLayers()==0")
        if self.vMod.layers[0].topSVelocity == 0:
            raise SlownessModelError(
                "Unable to handle zero S velocity layers at surface. "
                "This should be fixed at some point, but is a limitation of TauP at this point.")
        if self.DEBUG:
            print("start createSample")

        self.radiusOfEarth = self.vMod.radiusOfEarth

        if self.DEBUG: print("findCriticalPoints")
        self.findCriticalPoints()
        if self.DEBUG: print("coarseSample")
        self.coarseSample()
        if self.DEBUG and self.validate() is False:
            raise (SlownessModelError('validate failed after coarseSample'))
        if self.DEBUG: print("rayParamCheck")
        self.rayParamIncCheck()
        if self.DEBUG: print("depthIncCheck")
        self.depthIncCheck()
        if self.DEBUG: print("distanceCheck")
        self.distanceCheck()
        if self.DEBUG: print("fixCriticalPoints")
        self.fixCriticalPoints()

        if self.validate() is True:
            print("createSample seems to be done successfully.")
        else:
            raise SlownessModelError('SlownessModel.validate failed!')

    # noinspection PyCallByClass
    def findCriticalPoints(self):
        """ Finds all critical points within a velocity model.

         Critical points are first order discontinuities in
        velocity/slowness, local extrema in slowness. A high slowness
        zone is a low velocity zone, but it is possible to have a
        slight low velocity zone within a spherical earth that is not
        a high slowness zone and thus does not exhibit any of the
        pathological behavior of a low velocity zone.  """
        highSlownessZoneP = DepthRange()
        highSlownessZoneS = DepthRange()
        fluidZone = DepthRange()
        inFluidZone = False
        belowOuterCore = False
        inHighSlownessZoneP = False
        inHighSlownessZoneS = False
        # just some very big values (java had max possible of type, but these should do)
        minPSoFar = 1.1e300
        minSSoFar = 1.1e300
        # First remove any critical points previously stored
        # so these are effectively re-initialised... it's probably silly
        self.criticalDepths = []  # list of CriticalDepth
        self.highSlownessLayerDepthsP = []  # lists of DepthRange
        self.highSlownessLayerDepthsS = []
        self.fluidLayerDepths = []

        # Initialize the current velocity layer
        # to be zero thickness layer with values at the surface
        currVLayer = self.vMod.layers[0]
        currVLayer = VelocityLayer(0, currVLayer.topDepth, currVLayer.topDepth,
                                   currVLayer.topPVelocity, currVLayer.topPVelocity,
                                   currVLayer.topSVelocity, currVLayer.topSVelocity,
                                   currVLayer.topDensity, currVLayer.topDensity,
                                   currVLayer.topQp, currVLayer.topQp,
                                   currVLayer.topQs, currVLayer.topQs)
        currSLayer = SlownessLayer.create_from_vlayer(currVLayer, self.SWAVE)
        currPLayer = SlownessLayer.create_from_vlayer(currVLayer, self.PWAVE)
        # We know that the top is always a critical slowness so add 0
        self.criticalDepths.append(CriticalDepth(0, 0, 0, 0))
        # Check to see if starting in fluid zone.
        if inFluidZone is False and currVLayer.topSVelocity == 0:
            inFluidZone = True
            fluidZone = DepthRange(topDepth=currVLayer.topDepth)
            currSLayer = currPLayer
        if minSSoFar > currSLayer.topP:
            minSSoFar = currSLayer.topP
        # P is not a typo, it represents slowness, not P-wave speed.
        if minPSoFar > currPLayer.topP:
            minPSoFar = currPLayer.topP

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
            if inFluidZone is False and currVLayer.topSVelocity == 0:
                inFluidZone = True
                fluidZone = DepthRange(topDepth=currVLayer.topDepth)
            # If already in fluid zone, check if exited (java line 909)
            if inFluidZone is True and currVLayer.topSVelocity != 0:
                if prevVLayer.botDepth > self.vMod.iocbDepth:
                    belowOuterCore = True
                inFluidZone = False
                fluidZone.botDepth = prevVLayer.botDepth
                self.fluidLayerDepths.append(fluidZone)

            currPLayer = SlownessLayer.create_from_vlayer(currVLayer, self.PWAVE)
            # If we are in a fluid zone ( S velocity = 0.0 ) or if we are below
            # the outer core and allowInnerCoreS=false then use the P velocity
            # structure to look for critical points.
            if inFluidZone or (belowOuterCore and self.allowInnerCoreS is False):
                currSLayer = currPLayer
            else:
                currSLayer = SlownessLayer.create_from_vlayer(currVLayer, self.SWAVE)

            if prevSLayer.botP != currSLayer.topP or prevPLayer.botP != currPLayer.topP:
                # a first order discontinuity
                self.criticalDepths.append(CriticalDepth(currSLayer.topDepth,
                                                         layerNum, -1, -1))
                if self.DEBUG:
                    print('First order discontinuity, depth =' + str(currSLayer.topDepth))
                    print('between' + str(prevPLayer), str(currPLayer))
                if inHighSlownessZoneS and currSLayer.topP < minSSoFar:
                    if self.DEBUG:
                        print("Top of current layer is the bottom"
                              + " of a high slowness zone.")
                    highSlownessZoneS.botDepth = currSLayer.topDepth
                    self.highSlownessLayerDepthsS.append(highSlownessZoneS)
                    inHighSlownessZoneS = False
                if inHighSlownessZoneP and currPLayer.topP < minPSoFar:
                    if self.DEBUG:
                        print("Top of current layer is the bottom"
                              + " of a high slowness zone.")
                    highSlownessZoneP.botDepth = currSLayer.topDepth
                    self.highSlownessLayerDepthsP.append(highSlownessZoneP)
                    inHighSlownessZoneP = False
                # Update minPSoFar and minSSoFar as all total reflections off
                # of the top of the discontinuity are ok even though below the
                # discontinuity could be the start of a high slowness zone.
                if minPSoFar > currPLayer.topP:
                    minPSoFar = currPLayer.topP
                if minSSoFar > currSLayer.topP:
                    minSSoFar = currSLayer.topP

                if inHighSlownessZoneS is False and (prevSLayer.botP < currSLayer.topP or
                                                     currSLayer.topP < currSLayer.botP):
                    # start of a high slowness zone S
                    if self.DEBUG:
                        print("Found S high slowness at first order "
                              + "discontinuity, layer = " + str(layerNum))
                    inHighSlownessZoneS = True
                    highSlownessZoneS = DepthRange(topDepth=currSLayer.topDepth)
                    highSlownessZoneS.rayParam = minSSoFar
                if inHighSlownessZoneP is False and (prevPLayer.botP < currPLayer.topP or
                                                     currPLayer.topP < currPLayer.botP):
                    # start of a high slowness zone P
                    if self.DEBUG:
                        print("Found P high slowness at first order "
                              + "discontinuity, layer = " + str(layerNum))
                    inHighSlownessZoneP = True
                    highSlownessZoneP = DepthRange(topDepth=currPLayer.topDepth)
                    highSlownessZoneP.rayParam = minPSoFar

            elif ((prevSLayer.topP - prevSLayer.botP) *
                  (prevSLayer.botP - currSLayer.botP) < 0) or (
                      (prevPLayer.topP - prevPLayer.botP) *
                      (prevPLayer.botP - currPLayer.botP)) < 0:
                # local slowness extrema, java l 1005
                self.criticalDepths.append(CriticalDepth(currSLayer.topDepth, layerNum,
                                                         -1, -1))
                if self.DEBUG:
                    print("local slowness extrema, depth=" + str(currSLayer.topDepth))
                # here is line 1014 of the java src!
                if inHighSlownessZoneP is False and currPLayer.topP < currPLayer.botP:
                    if self.DEBUG:
                        print("start of a P high slowness zone, local slowness extrema,"
                              + "minPSoFar= " + str(minPSoFar))
                    inHighSlownessZoneP = True
                    highSlownessZoneP = DepthRange(topDepth=currPLayer.topDepth)
                    highSlownessZoneP.rayParam = minPSoFar
                if inHighSlownessZoneS is False and currSLayer.topP < currSLayer.botP:
                    if self.DEBUG:
                        print("start of a S high slowness zone, local slowness extrema,"
                              + "minSSoFar= " + str(minSSoFar))
                    inHighSlownessZoneS = True
                    highSlownessZoneS = DepthRange(topDepth=currSLayer.topDepth)
                    highSlownessZoneS.rayParam = minSSoFar

            if inHighSlownessZoneP and currPLayer.botP < minPSoFar:
                # P: layer contains the bottom of a high slowness zone. java l 1043
                if self.DEBUG:
                    print("layer contains the bottom of a P "
                          + "high slowness zone. minPSoFar=" + str(minPSoFar), currPLayer)
                highSlownessZoneP.botDepth = self.findDepth(minPSoFar, layerNum,
                                                            layerNum, self.PWAVE)
                self.highSlownessLayerDepthsP.append(highSlownessZoneP)
                inHighSlownessZoneP = False

            if inHighSlownessZoneS and currSLayer.botP < minSSoFar:
                # S: layer contains the bottom of a high slowness zone. java l 1043
                if self.DEBUG:
                    print("layer contains the bottom of a S "
                          + "high slowness zone. minSSoFar=" + str(minSSoFar), currSLayer)
                # in fluid layers we want to check PWAVE structure
                # when looking for S wave critical points
                porS = (self.PWAVE if currSLayer == currPLayer else self.SWAVE)
                highSlownessZoneS.botDepth = self.findDepth(minSSoFar, layerNum,
                                                            layerNum, porS)
                self.highSlownessLayerDepthsS.append(highSlownessZoneS)
                inHighSlownessZoneS = False
            if minPSoFar > currPLayer.botP:
                minPSoFar = currPLayer.botP
            if minPSoFar > currPLayer.topP:
                minPSoFar = currPLayer.topP
            if minSSoFar > currSLayer.botP:
                minSSoFar = currSLayer.botP
            if minSSoFar > currSLayer.topP:
                minSSoFar = currSLayer.topP
            if self.DEBUG and inHighSlownessZoneS:
                print("In S high slowness zone, layerNum = " + str(layerNum)
                      + " minSSoFar=" + str(minSSoFar))
            if self.DEBUG and inHighSlownessZoneP:
                print("In P high slowness zone, layerNum = " + str(layerNum)
                      + " minPSoFar=" + str(minPSoFar))

        # We know that the bottommost depth is always a critical slowness,
        # so we add vMod.getNumLayers()
        # java line 1094
        self.criticalDepths.append(CriticalDepth(self.radiusOfEarth,
                                                 self.vMod.getNumLayers(), -1, -1))

        # Check if the bottommost depth is contained within a high slowness
        # zone, might happen in a flat non-whole-earth model
        if inHighSlownessZoneS:
            highSlownessZoneS.botDepth = currVLayer.botDepth
            self.highSlownessLayerDepthsS.append(highSlownessZoneS)
        if inHighSlownessZoneP:
            highSlownessZoneP.botDepth = currVLayer.botDepth
            self.highSlownessLayerDepthsP.append(highSlownessZoneP)

        # Check if the bottommost depth is contained within a fluid zone, this
        # would be the case if we have a non whole earth model with the bottom
        # in the outer core or if allowInnerCoreS == false and we want to use
        # the P velocity structure in the inner core.
        if inFluidZone:
            fluidZone.botDepth = currVLayer.botDepth
            self.fluidLayerDepths.append(fluidZone)

        # optionally implement later: print all critical vel layers in debug mode

        if self.validate() is False:
            raise SlownessModelError("Validation failed after findDepth")

    def findDepth_from_depths(self, rayParam, topDepth, botDepth, isPWave):
        """Finds a depth corresponding to a slowness between two given depths in the
        Velocity Model by calling findDepth with layer numbers."""
        topLayerNum = self.vMod.layerNumberBelow(topDepth)
        if self.vMod.layers[topLayerNum].botDepth == topDepth:
            topLayerNum += 1
        botLayerNum = self.vMod.layerNumberAbove(botDepth)
        return self.findDepth(rayParam, topLayerNum, botLayerNum, isPWave)

    def findDepth(self, p, topCriticalLayer, botCriticalLayer, isPWave):
        """Finds a depth corresponding to a slowness between two given velocity
        layers, including the top and the bottom. We also check to see if the
        slowness is less than the bottom slowness of these layers but greater
        than the top slowness of the next deeper layer. This corresponds to a
        total reflection. In this case a check needs to be made to see if this is
        an S wave reflecting off of a fluid layer, use P velocity below in this
        case. We assume that slowness is monotonic within these layers and
        therefore there is only one depth with the given slowness. This means we
        return the first depth that we find.

         SlownessModelError occurs if topCriticalLayer > botCriticalLayer because
                   there are no layers to search, or if there is an increase
                   in slowness, ie a negative velocity gradient, that just
                   balances the decrease in slowness due to the spherical
                   earth, or if the ray parameter p is not contained within
                   the specified layer range."""

        #topP = 1.1e300  # dummy numbers
        #botP = 1.1e300
        waveType = 'P' if isPWave else 'S'

        if topCriticalLayer > botCriticalLayer:
            raise SlownessModelError("findDepth: no layers to search (wrong layer num?)")
        for layerNum in range(topCriticalLayer, botCriticalLayer + 1):
            velLayer = self.vMod.layers[layerNum]
            topVelocity = velLayer.evaluateAtTop(waveType)
            botVelocity = velLayer.evaluateAtBottom(waveType)
            topP = self.toSlowness(topVelocity, velLayer.topDepth)
            botP = self.toSlowness(botVelocity, velLayer.botDepth)
            # check to see if we are within 'chatter level' (numerical error) of the top or
            # bottom and if so then return that depth.
            if abs(topP - p) < self.slowness_tolerance:
                return velLayer.topDepth
            if abs(p - botP) < self.slowness_tolerance:
                return velLayer.botDepth

            if (topP - p) * (p - botP) >= 0:
                # Found layer containing p.
                # We interpolate assuming that velocity is linear within
                # this interval. So slope is the slope for velocity versus depth
                slope = (botVelocity - topVelocity) / (velLayer.botDepth - velLayer.topDepth)
                depth = self.interpolate(p, topVelocity, velLayer.topDepth, slope)
                return depth
            elif layerNum == topCriticalLayer and abs(p - topP) < self.slowness_tolerance:
                # Check to see if p is just outside the topmost layer. If so
                # then return the top depth.
                return velLayer.topDepth

            # Is p a total reflection? botP is the slowness at the bottom
            # of the last velocity layer from the previous loop, set topP
            # to be the slowness at the top of the next layer.
            if layerNum < self.vMod.getNumLayers() - 1:
                velLayer = self.vMod.layers[layerNum + 1]
                topVelocity = velLayer.evaluateAtTop(waveType)
                if isPWave is False and self.depthInFluid(velLayer.topDepth):
                    # Special case for S waves above a fluid. If top next
                    # layer is in a fluid then we should set topVelocity to
                    # be the P velocity at the top of the layer.
                    topVelocity = velLayer.evaluateAtTop('P')

                topP = self.toSlowness(topVelocity, velLayer.topDepth)
                if botP >= p >= topP:
                    return velLayer.topDepth

        # noinspection PyUnboundLocalVariable
        if abs(p - botP) < self.slowness_tolerance:
            # java line 1275
            #Check to see if p is just outside the bottommost layer. If so
            #than return the bottom depth.
            print(" p is just outside the bottommost layer. This probably shouldn't be allowed to happen!")
            # noinspection PyUnboundLocalVariable
            return velLayer.getBotDepth()

        raise SlownessModelError("slowness p=" + str(p) + "is not contained within the specified layers."
                                 + " topCriticalLayer=" + str(topCriticalLayer)
                                 + " botCriticalLayer=" + str(botCriticalLayer))

    def toSlowness(self, velocity, depth):
        if velocity == 0:
            raise SlownessModelError("toSlowness: velocity can't be zero, at depth" +
                                     str(depth),
                                     "Maybe related to using S velocities in outer core?")
        return (self.radiusOfEarth - depth) / velocity

    def interpolate(self, p, topVelocity, topDepth, slope):
        denominator = p * slope + 1
        if denominator == 0:
            raise SlownessModelError("Negative velocity gradient that just balances the slowness gradient "
                                     "of the spherical slowness, i.e. Earth flattening. Instructions unclear; explode.")
        else:
            depth = (self.radiusOfEarth + p * (topDepth * slope - topVelocity)) / denominator
            return depth

    def depthInFluid(self, depth, fluidZoneDepth=DepthRange()):
        pass

    def coarseSample(self):
        pass

    def rayParamIncCheck(self):
        pass

    def depthIncCheck(self):
        pass

    def distanceCheck(self):
        pass

    def fixCriticalPoints(self):
        pass

    def validate(self):
        return True

    def getNumLayers(self, isPWave):
        '''This is meant to return the number of pLayers and sLayers.
        I have not yet been able to find out how these are known in 
        the java code.'''
        # translated Java code:
        # def getNumLayers(self, isPWave):
        # """ generated source for method getNumLayers """
        # if isPWave:
        # return len(self.PLayers)
        # else:
        # return len(self.SLayers)

        # Where
        # self.PLayers = pLayers
        # and the pLayers have been provided in the constructor, but I
        # don't understand from where!

        # dummy code so TauP_Create won't fail:
        if isPWave == True:
            return 'some really dummy number'
        if isPWave == False:
            return 'some other number'


    def __str__(self):
        desc = "This is a dummy SlownessModel so there's nothing here really. Nothing to see. Move on."
        desc += "This might be interesting: slowness_tolerance ought to be 1e-16. It is:" + str(self.slowness_tolerance)
        return desc
