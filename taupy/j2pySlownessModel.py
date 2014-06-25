class SlownessModel(object):
    """
    generated source for class SlownessModel
    """
    def __init___(self, radiusOfEarth, vMod, criticalDepth,
                  highSlownessLayerDepthsP, highSlownessLayerDepthsS,
                  fluidLayerDepths, pLayers, sLayers, minDeltaP, maxDeltaP,
                  maxDepthInterval, maxRangeInterval, maxInterpError,
                  allowInnerCoreS, slownessTolerance):
        """
        generated source for method __init___
        """
        self.radiusOfEarth = radiusOfEarth
        self.vMod = vMod
        self.criticalDepths = criticalDepth
        self.highSlownessLayerDepthsP = highSlownessLayerDepthsP
        self.highSlownessLayerDepthsS = highSlownessLayerDepthsS
        self.fluidLayerDepths = fluidLayerDepths
        self.PLayers = pLayers
        self.SLayers = sLayers
        self.minDeltaP = minDeltaP
        self.maxDeltaP = maxDeltaP
        self.maxDepthInterval = maxDepthInterval
        self.maxRangeInterval = maxRangeInterval
        self.maxInterpError = maxInterpError
        self.allowInnerCoreS = allowInnerCoreS
        self.slownessTolerance = slownessTolerance

    #  True to enable debugging output.
    DEBUG = False

    #  True to enable verbose output.
    verbose = False

    #  Radius of the Earth in km, usually input from the velocity model.
    radiusOfEarth = 6371.0

    #
    #      * Velocity Model used to get slowness model. Usually set in
    #      * createSlowness().
    #
    vMod = VelocityModel()

    #
    #      * Stores the layer number for layers in the velocity model with a critical
    #      * point at their top. These form the "branches" of slowness sampling.
    #      *
    #      * @see edu.sc.seis.TauP.CriticalDepth
    #
    criticalDepths = ArrayList()

    #
    #      * Stores depth ranges that contains a high slowness zone for P. Stored as
    #      * DepthRange objects, containing the top depth and bottom depth.
    #      *
    #      * @see DepthRange
    #
    highSlownessLayerDepthsP = ArrayList()

    #
    #      * Stores depth ranges that contains a high slowness zone for S. Stored as
    #      * DepthRange objects, containing the top depth and bottom depth.
    #      *
    #      * @see DepthRange
    #
    highSlownessLayerDepthsS = ArrayList()

    #
    #      * Stores depth ranges that are fluid, ie S velocity is zero. Stored as
    #      * DepthRange objects, containing the top depth and bottom depth.
    #      *
    #      * @see DepthRange
    #
    fluidLayerDepths = ArrayList()

    #  Initial length of the slowness vectors.
    vectorLength = 256

    #
    #      * Stores the final slowness-depth layers for P waves. Stored as
    #      * SlownessLayer objects.
    #      *
    #      * @see edu.sc.seis.TauP.SlownessLayer
    #
    PLayers = ArrayList(vectorLength)

    #
    #      * Stores the final slowness-depth layers for S waves. Stored as
    #      * SlownessLayer objects. Note that SLayers and PLayers share the same
    #      * SlownessLayer object within fluid layers, so changes made to one will
    #      * affect the other.
    #      *
    #      * @see edu.sc.seis.TauP.SlownessLayer
    #
    SLayers = ArrayList(vectorLength)

    #
    #      * Minimum difference between successive slowness samples. The default is
    #      * 0.1 (km-sec/km or sec/rad for spherical, sec/km for flat models). This
    #      * keeps the sampling from becoming too fine. For example, a strong negative
    #      * S velocity gradient just above the CMB will cause the totally reflected
    #      * ScS too have an extremely large range of distances, over a very small
    #      * range of ray parameters. The distance check would otherwise force a very
    #      * fine sampling of this region. However since in this case time and
    #      * distance are likely to be very close to being linearly related, this sort
    #      * of sampling is overkill. So we ignore the distance check if the ray
    #      * parameter becomes smaller than minDeltaP.
    #
    minDeltaP = 0.1

    #
    #      * Maximum difference between successive slowness samples. The default is
    #      * 11.0 (km-sec/km or sec/rad for spherical, sec/km for flat models). See
    #      * Buland and Chapman p1292
    #
    maxDeltaP = 11.0

    #
    #      * Maximum difference between successive depth samples, default is 115 km.
    #      * See Buland and Chapman p1292
    #
    maxDepthInterval = 115.0

    #
    #      * Maximum difference between successive ranges, in radians. The default is
    #      * 200 km / radiusOfEarth. See Buland and Chapman p1292.
    #      *
    #      * @see radiusOfEarth
    #
    maxRangeInterval = 200.0 / radiusOfEarth
    maxInterpError = 0.5

    #
    #      * Should we allow J phases, S waves in the inner core? If true, then the
    #      * slowness sampling for S will use the S velocity structure for the inner
    #      * core. If false, then we will use the P velocity structure for both the
    #      * inner and outer core for S waves as well as P waves. Disallowing inner
    #      * core S phases reduces the number of slowness samples significantly due to
    #      * the large geometrical spreading of S waves in the inner core. The default
    #      * is false.
    #      *
    #      * @see minInnerCoreDepth
    #
    allowInnerCoreS = True
    DEFAULT_SLOWNESS_TOLERANCE = 1e-16

    #
    #      * Tolerance for slownesses. If two slownesses are closer that this value,
    #      * then we consider them to be identical. Basically this just provides some
    #      * protection against numerical "chatter".
    #
    slownessTolerance = DEFAULT_SLOWNESS_TOLERANCE

    #
    #      * Just useful for calling methods that need to know whether to use P or S
    #      * waves.
    #
    PWAVE = True

    #
    #      * Just useful for calling methods that need to know whether to use P or S
    #      * waves.
    #
    SWAVE = False

    #  METHODS ----------------------------------------------------------------
    #  Accessor methods
    def setRadiusOfEarth(self, radiusOfEarth):
        """ generated source for method setRadiusOfEarth """
        self.radiusOfEarth = radiusOfEarth

    def setMinDeltaP(self, minDeltaP):
        """ generated source for method setMinDeltaP """
        self.minDeltaP = minDeltaP

    def setMaxDeltaP(self, maxDeltaP):
        """ generated source for method setMaxDeltaP """
        self.maxDeltaP = maxDeltaP

    def setMaxDepthInterval(self, maxDepthInterval):
        """ generated source for method setMaxDepthInterval """
        self.maxDepthInterval = maxDepthInterval

    #
    #      * sets the maximum range interval for surface focus turning waves between
    #      * slowness samples, input in degrees.
    #
    def setMaxRangeInterval(self, maxRangeInterval):
        """ generated source for method setMaxRangeInterval """
        self.maxRangeInterval = maxRangeInterval * Math.PI / 180.0

    #
    #      * sets the maximum value of the estimated error due to linear
    #      * interpolation. Care should be taken not to set this too small as a very
    #      * large number of samples may be required. Note also that this is only an
    #      * estimate of the error, and thus the bound is by no means assured.
    #
    def setMaxInterpError(self, maxInterpError):
        """ generated source for method setMaxInterpError """
        self.maxInterpError = maxInterpError

    def setAllowInnerCoreS(self, allowInnerCoreS):
        """ generated source for method setAllowInnerCoreS """
        self.allowInnerCoreS = allowInnerCoreS

    def setSlownessTolerance(self, slownessTolerance):
        """ generated source for method setSlownessTolerance """
        self.slownessTolerance = slownessTolerance

    #  get accessor methods
    def getVelocityModel(self):
        """ generated source for method getVelocityModel """
        return self.vMod

    def getRadiusOfEarth(self):
        """ generated source for method getRadiusOfEarth """
        return self.radiusOfEarth

    def getMinDeltaP(self):
        """ generated source for method getMinDeltaP """
        return self.minDeltaP

    def getMaxDeltaP(self):
        """ generated source for method getMaxDeltaP """
        return self.maxDeltaP

    def getMaxDepthInterval(self):
        """ generated source for method getMaxDepthInterval """
        return self.maxDepthInterval

    #
    #      * @return the maximum range interval for surface focus turning waves
    #      *          between slowness samples output in degrees.
    #
    def getMaxRangeInterval(self):
        """ generated source for method getMaxRangeInterval """
        return 180.0 * self.maxRangeInterval / Math.PI

    #
    #      * gets the maximum value of the estimated error due to linear
    #      * interpolation. Care should be taken not to set this too small as a very
    #      * large number of samples may be required. Note also that this is only an
    #      * estimate of the error, and thus the bound is by no means assured.
    #
    def getMaxInterpError(self):
        """ generated source for method getMaxInterpError """
        return self.maxInterpError

    def isAllowInnerCoreS(self):
        """ generated source for method isAllowInnerCoreS """
        return self.allowInnerCoreS

    def getSlownessTolerance(self):
        """ generated source for method getSlownessTolerance """
        return self.slownessTolerance

    def getNumCriticalDepths(self):
        """ generated source for method getNumCriticalDepths """
        return len(self.criticalDepths)

    def getCriticalDepth(self, i):
        """ generated source for method getCriticalDepth """
        return self.criticalDepths.get(i)

    def getNumLayers(self, isPWave):
        """ generated source for method getNumLayers """
        if isPWave:
            return len(self.PLayers)
        else:
            return len(self.SLayers)

    #
    #      * @return the minimum ray parameter that turns, but is not reflected, at
    #      *          or above the given depth. Normally this is the slowness sample
    #      *          at the given depth, but if the depth is within a high slowness
    #      *          zone, then it may be smaller.
    #
    def getMinTurnRayParam(self, depth, isPWave):
        """ generated source for method getMinTurnRayParam """
        minPSoFar = Double.MAX_VALUE
        sLayer = SlownessLayer()
        layers = List()
        if isPWave:
            layers = self.PLayers
        else:
            layers = self.SLayers
        if depthInHighSlowness(depth, Double.MAX_VALUE, isPWave):
            while i < len(layers):
                sLayer = getSlownessLayer(i, isPWave)
                if sLayer.getBotDepth() == depth:
                    minPSoFar = Math.min(minPSoFar, sLayer.getBotP())
                    return minPSoFar
                elif sLayer.getBotDepth() > depth:
                    minPSoFar = Math.min(minPSoFar, sLayer.evaluateAt_bullen(depth, self.getRadiusOfEarth()))
                    return minPSoFar
                else:
                    minPSoFar = Math.min(minPSoFar, sLayer.getBotP())
                i += 1
        else:
            sLayer = getSlownessLayer(layerNumberAbove(depth, isPWave), isPWave)
            if depth == sLayer.getBotDepth():
                minPSoFar = sLayer.getBotP()
            else:
                minPSoFar = sLayer.evaluateAt_bullen(depth, self.getRadiusOfEarth())
        return minPSoFar

    #
    #      * @return the minimum ray parameter that turns or is reflected at or above
    #      *          the given depth. Normally this is the slowness sample at the
    #      *          given depth, but if the depth is within a high slowness zone,
    #      *          then it may be smaller. Also, at first order discontinuities,
    #      *          there may be many slowness samples at the same depth.
    #
    def getMinRayParam(self, depth, isPWave):
        """ generated source for method getMinRayParam """
        minPSoFar = self.getMinTurnRayParam(depth, isPWave)
        i = layerNumberAbove(depth, isPWave)
        j = layerNumberBelow(depth, isPWave)
        sLayerAbove = getSlownessLayer(i, isPWave)
        sLayerBelow = getSlownessLayer(j, isPWave)
        if sLayerAbove.getBotDepth() == depth:
            minPSoFar = Math.min(Math.min(minPSoFar, sLayerAbove.getBotP()), sLayerBelow.getTopP())
        return minPSoFar

    #
    #      * @return the DepthRange objects for all high slowness zones within the
    #      *          slowness model.
    #
    def getHighSlowness(self, isPWave):
        """ generated source for method getHighSlowness """
        highSlownessLayerDepths = List()
        if isPWave:
            highSlownessLayerDepths = self.highSlownessLayerDepthsP
        else:
            highSlownessLayerDepths = self.highSlownessLayerDepthsS
        hsz = [None]*len(highSlownessLayerDepths)
        i = 0
        while i < len(highSlownessLayerDepths):
            hsz[i] = highSlownessLayerDepths.get(i).clone()
            i += 1
        return hsz

    #
    #      * Returns the SlownessLayer of the requested waveType. This is NOT a clone
    #
    def getSlownessLayer(self, layerNum, isPWave):
        """ generated source for method getSlownessLayer """
        if isPWave:
            return self.PLayers.get(layerNum)
        else:
            return self.SLayers.get(layerNum)

    def getAllSlownessLayers(self, isPWave):
        """ generated source for method getAllSlownessLayers """
        if isPWave:
            return self.PLayers
        else:
            return self.SLayers

    #  Abstract methods
    def toSlowness(self, velocity, depth):
        """ generated source for method toSlowness """

    def toVelocity(self, slowness, depth):
        """ generated source for method toVelocity """

    def layerTimeDist(self, rayParam, layerNum, isPWave):
        """ generated source for method layerTimeDist """

    def toSlownessLayer(self, vLayer, isPWave):
        """ generated source for method toSlownessLayer """

    def interpolate(self, p, topVelocity, topDepth, slope):
        """ generated source for method interpolate """

    #  Defined methods
    #
    #      * generate approximate distance, in radians, for a ray from a surface
    #      * source that turns at the bottom of the given slowness layer.
    #      *
    #      * @exception NoSuchLayerException
    #      *                occurs if no layer in the velocity model contains the
    #      *                given depth.
    #      * @exception SlownessModelException
    #      *                occurs if getNumLayers() == 0 as we cannot compute a
    #      *                distance without a layer.
    #
    def approxDistance(self, slownessTurnLayer, p, isPWave):
        """ generated source for method approxDistance """
        #
        #          * First, if slowness contains less than slownessTurnLayer elements then
        #          * we can't calculate a distance, otherwise we must signal an exception.
        #
        if slownessTurnLayer >= self.getNumLayers(isPWave):
            raise SlownessModelException("Can't calculate a distance when " + "slownessTurnLayer >= getNumLayers(" + isPWave + ")\n" + " slownessTurnLayer=" + slownessTurnLayer + " getNumLayers()=" + self.getNumLayers(isPWave))
        if p < 0.0:
            raise SlownessModelException("approxDistance: Ray parameter is negative!!!" + p + " slownessTurnLayer=" + slownessTurnLayer)
        #
        #          * OK, now we are able to do the calculations for the approximate
        #          * distance, hopefully without errors.
        #
        td = TimeDist(p)
        layerNum = 0
        while layerNum <= slownessTurnLayer:
            td.add(self.layerTimeDist(p, layerNum, isPWave))
            layerNum += 1
        #
        #          * Return 2.0*distance and time because there is a downgoing as well as
        #          * up going leg, which are equal because this is for a surface source.
        #
        td.distRadian *= 2.0
        td.time *= 2.0
        return td

    #
    #      * Determines if the given depth and corresponding slowness is contained
    #      * within a high slowness zone. Whether the high slowness zone includes its
    #      * upper boundary and its lower boundaries depends upon the ray parameter.
    #      * The slowness at the depth is needed because if depth happens to
    #      * correspond to a discontinuity that marks the bottom of the high slowness
    #      * zone but the ray is actually a total reflection then it is not part of
    #      * the high slowness zone. Calls depthInHighSlowness(double, double,
    #      * DepthRange).
    #      *
    #      * @see depthInHighSlowness.
    #
    @overloaded
    def depthInHighSlowness(self, depth, rayParam, isPWave):
        """ generated source for method depthInHighSlowness """
        highSZoneDepth = DepthRange()
        return self.depthInHighSlowness(depth, rayParam, highSZoneDepth, isPWave)

    #
    #      * Determines if the given depth and corresponding slowness is contained
    #      * within a high slowness zone. Whether the high slowness zone includes its
    #      * upper boundary and its lower boundaries depends upon the ray parameter.
    #      * The slowness at the depth is needed because if depth happens to
    #      * correspond to a discontinuity that marks the bottom of the high slowness
    #      * zone but the ray is actually a total reflection then it is not part of
    #      * the high slowness zone. The ray parameter that delimits the zone, ie it
    #      * can turn at the top and the bottom, is in the zone at the top, but out of
    #      * the zone at the bottom.
    #
    @depthInHighSlowness.register(object, float, float, DepthRange, bool)
    def depthInHighSlowness_0(self, depth, rayParam, highSZoneDepth, isPWave):
        """ generated source for method depthInHighSlowness_0 """
        tempRange = DepthRange()
        highSlownessLayerDepths = List()
        if isPWave:
            highSlownessLayerDepths = self.highSlownessLayerDepthsP
        else:
            highSlownessLayerDepths = self.highSlownessLayerDepthsS
        i = 0
        while i < len(highSlownessLayerDepths):
            tempRange = highSlownessLayerDepths.get(i)
            if tempRange.topDepth <= depth and depth <= tempRange.botDepth:
                highSZoneDepth.topDepth = tempRange.topDepth
                highSZoneDepth.botDepth = tempRange.botDepth
                highSZoneDepth.rayParam = tempRange.rayParam
                if rayParam > tempRange.rayParam or (rayParam == tempRange.rayParam and depth == tempRange.topDepth):
                    return True
            i += 1
        return False

    #
    #      * Determines if the given depth is contained within a fluid zone. The fluid
    #      * zone includes its upper boundary but not its lower boundary. Calls
    #      * depthInFluid(double, DepthRange).
    #      *
    #      * @see depthInFluid(double, DepthRange).
    #
    @overloaded
    def depthInFluid(self, depth):
        """ generated source for method depthInFluid """
        fluidZoneDepth = DepthRange()
        return self.depthInFluid(depth, fluidZoneDepth)

    #
    #      * Determines if the given depth is contained within a fluid zone. The fluid
    #      * zone includes its upper boundary but not its lower boundary. The top and
    #      * bottom of the fluid zone are returned in DepthRange.
    #
    @depthInFluid.register(object, float, DepthRange)
    def depthInFluid_0(self, depth, fluidZoneDepth):
        """ generated source for method depthInFluid_0 """
        tempRange = DepthRange()
        i = 0
        while i < len(self.fluidLayerDepths):
            tempRange = self.fluidLayerDepths.get(i)
            if tempRange.topDepth <= depth and depth < tempRange.botDepth:
                fluidZoneDepth.topDepth = tempRange.topDepth
                fluidZoneDepth.botDepth = tempRange.botDepth
                return True
            i += 1
        return False

    #
    #      * Splits a slowness layer into two slowness layers. returns a
    #      * SplitLayerInfo object with neededSplit=true if a layer was actually
    #      * split, false otherwise, movedSample=true if a layer was very close, and
    #      * so moving the layers depth is better than making a very thin layer,
    #      * rayParam= the new ray parameter, if the layer was split. The
    #      * interpolation for splitting a layer is a Bullen p=Ar^B and so does not
    #      * directly use information from the VelocityModel.
    #
    def splitLayer(self, depth, isPWave):
        """ generated source for method splitLayer """
        layerNum = layerNumberAbove(depth, isPWave)
        sLayer = self.getSlownessLayer(layerNum, isPWave)
        if sLayer.getTopDepth() == depth or sLayer.getBotDepth() == depth:
            #
            #              * depth is already on a slowness layer boundary so we don't need to
            #              * split any slowness layers.
            #
            return SplitLayerInfo(self, False, False, 0.0)
        elif Math.abs(sLayer.getTopDepth() - depth) < 0.000001:
            #
            #              * check for very thin layers, just move the layer to hit the
            #              * boundary
            #
            outLayers.addAll(allLayers)
            outLayers.set(layerNum, SlownessLayer(sLayer.getTopP(), depth, sLayer.getBotP(), sLayer.getBotDepth()))
            sLayer = self.getSlownessLayer(layerNum - 1, isPWave)
            outLayers.set(layerNum - 1, SlownessLayer(sLayer.getTopP(), sLayer.getTopDepth(), sLayer.getBotP(), depth))
            if isPWave:
                outPLayers = outLayers
                outSLayers = self.SLayers
            else:
                outPLayers = self.PLayers
                outSLayers = outLayers
            return SplitLayerInfo(out, False, True, sLayer.getBotP())
        elif Math.abs(depth - sLayer.getBotDepth()) < 0.000001:
            #
            #              * check for very thin layers, just move the layer to hit the
            #              * boundary
            #
            outLayers.addAll(allLayers)
            outLayers.set(layerNum, SlownessLayer(sLayer.getTopP(), sLayer.getTopDepth(), sLayer.getBotP(), depth))
            sLayer = self.getSlownessLayer(layerNum + 1, isPWave)
            outLayers.set(layerNum + 1, SlownessLayer(sLayer.getTopP(), depth, sLayer.getBotP(), sLayer.getBotDepth()))
            if isPWave:
                outPLayers = outLayers
                outSLayers = self.SLayers
            else:
                outPLayers = self.PLayers
                outSLayers = outLayers
            return SplitLayerInfo(out, False, True, sLayer.getBotP())
        else:
            topLayer = SlownessLayer(sLayer.getTopP(), sLayer.getTopDepth(), p, depth)
            botLayer = SlownessLayer(p, depth, sLayer.getBotP(), sLayer.getBotDepth())
            outLayers.addAll(allLayers)
            outLayers.remove(layerNum)
            outLayers.add(layerNum, botLayer)
            outLayers.add(layerNum, topLayer)
            #  fix critical layers since we have added a slowness layer
            outCriticalDepths.addAll(self.criticalDepths)
            fixCriticalDepths(outCriticalDepths, layerNum, isPWave)
            if isPWave:
                outPLayers = outLayers
                outSLayers = fixOtherLayers(self.SLayers, p, sLayer, topLayer, botLayer, outCriticalDepths, not isPWave)
            else:
                outPLayers = fixOtherLayers(self.PLayers, p, sLayer, topLayer, botLayer, outCriticalDepths, not isPWave)
                outSLayers = outLayers
            return SplitLayerInfo(out, True, False, p)

    def fixCriticalDepths(self, criticalDepths, layerNum, isPWave):
        """ generated source for method fixCriticalDepths """
        i = 0
        while i < len(criticalDepths):
            if cd.getLayerNum(isPWave) > layerNum:
                if isPWave:
                    criticalDepths.set(i, CriticalDepth(cd.getDepth(), cd.getVelLayerNum(), cd.getPLayerNum() + 1, cd.getSLayerNum()))
                else:
                    criticalDepths.set(i, CriticalDepth(cd.getDepth(), cd.getVelLayerNum(), cd.getPLayerNum(), cd.getSLayerNum() + 1))
            i += 1

    def fixOtherLayers(self, otherLayers, p, changedLayer, newTopLayer, newBotLayer, criticalDepths, isPWave):
        """ generated source for method fixOtherLayers """
        out = ArrayList()
        out.addAll(otherLayers)
        otherIndex = otherLayers.indexOf(changedLayer)
        #  now make sure we keep the sampling consistant
        #  if in a fluid, then both wavetypes will share a single
        #  slowness layer object. Otherwise indexOf returns -1
        if otherIndex != -1:
            out.remove(otherIndex)
            out.add(otherIndex, newBotLayer)
            out.add(otherIndex, newTopLayer)
        otherLayerNum = 0
        while otherLayerNum < len(out):
            if (sLayer.getTopP() - p) * (p - sLayer.getBotP()) > 0.0:
                #  found a slowness layer with the other wave type that
                #  contains the new slowness sample
                topLayer = SlownessLayer(sLayer.getTopP(), sLayer.getTopDepth(), p, sLayer.bullenDepthFor(p, self.radiusOfEarth))
                botLayer = SlownessLayer(p, topLayer.getBotDepth(), sLayer.getBotP(), sLayer.getBotDepth())
                out.remove(otherLayerNum)
                out.add(otherLayerNum, botLayer)
                out.add(otherLayerNum, topLayer)
                #  fix critical layers since we have added a slowness layer
                self.fixCriticalDepths(criticalDepths, otherLayerNum, not isPWave)
                otherLayerNum += 1
                #  skip next layer as it was just added
            otherLayerNum += 1
        return out

    #
    #      * Finds all critical points within a velocity model. Critical points are
    #      * first order discontinuities in velocity/slowness, local extrema in
    #      * slowness. A high slowness zone is a low velocity zone, but it is possible
    #      * to have a slight low velocity zone within a spherical earth that is not a
    #      * high slowness zone and thus does not exhibit any of the pathological
    #      * behavior of a low velocity zone.
    #      *
    #      * @exception NoSuchMatPropException
    #      *                occurs if wavetype is not recognized.
    #      * @exception SlownessModelException
    #      *                occurs if validate() returns false, this indicates a bug
    #      *                in the code.
    #
    def findCriticalPoints(self):
        """ generated source for method findCriticalPoints """
        minPSoFar = Double.MAX_VALUE
        minSSoFar = Double.MAX_VALUE
        highSlownessZoneP = DepthRange()
        highSlownessZoneS = DepthRange()
        inHighSlownessZoneP = False
        inHighSlownessZoneS = False
        fluidZone = DepthRange()
        inFluidZone = False
        belowOuterCore = False
        #
        #                                          * are we in the inner core, see
        #                                          * allowInnerCoreS.
        #
        prevVLayer = VelocityLayer()
        currVLayer = VelocityLayer()
        prevSLayer = SlownessLayer()
        currSLayer = SlownessLayer()
        prevPLayer = SlownessLayer()
        currPLayer = SlownessLayer()
        #  First remove any critical points previously stored
        self.highSlownessLayerDepthsP.clear()
        self.highSlownessLayerDepthsS.clear()
        self.criticalDepths.clear()
        self.fluidLayerDepths.clear()
        #  Initialize the current velocity layer
        #  to be zero thickness layer with values at the surface
        currVLayer = self.vMod.getVelocityLayer(0)
        currVLayer = VelocityLayer(0, currVLayer.getTopDepth(), currVLayer.getTopDepth(), currVLayer.getTopPVelocity(), currVLayer.getTopPVelocity(), currVLayer.getTopSVelocity(), currVLayer.getTopSVelocity(), currVLayer.getTopDensity(), currVLayer.getTopDensity(), currVLayer.getTopQp(), currVLayer.getTopQp(), currVLayer.getTopQs(), currVLayer.getTopQs())
        currSLayer = self.toSlownessLayer(currVLayer, self.SWAVE)
        currPLayer = self.toSlownessLayer(currVLayer, self.PWAVE)
        #  We know that the top is always a critical slowness so add 0
        self.criticalDepths.add(CriticalDepth(0.0, 0, 0, 0))
        #  Check to see if we start in a fluid zone.
        if not inFluidZone and currVLayer.getTopSVelocity() == 0.0:
            inFluidZone = True
            fluidZone = DepthRange()
            fluidZone.topDepth = currVLayer.getTopDepth()
            currSLayer = currPLayer
        if minSSoFar > currSLayer.getTopP():
            minSSoFar = currSLayer.getTopP()
        if minPSoFar > currPLayer.getTopP():
            minPSoFar = currPLayer.getTopP()
        layerNum = 0
        while layerNum < self.vMod.getNumLayers():
            prevVLayer = currVLayer
            prevSLayer = currSLayer
            prevPLayer = currPLayer
            currVLayer = self.vMod.getVelocityLayerClone(layerNum)
            #
            #              * If we are not already in a fluid check to see if we have just
            #              * entered a fluid zone.
            #
            if not inFluidZone and currVLayer.getTopSVelocity() == 0.0:
                inFluidZone = True
                fluidZone = DepthRange()
                fluidZone.topDepth = currVLayer.getTopDepth()
            #
            #              * If we are already in a fluid check to see if we have just exited
            #              * it.
            #
            if inFluidZone and currVLayer.getTopSVelocity() != 0.0:
                if prevVLayer.getBotDepth() > self.vMod.getIocbDepth():
                    belowOuterCore = True
                inFluidZone = False
                fluidZone.botDepth = prevVLayer.getBotDepth()
                self.fluidLayerDepths.add(fluidZone)
            currPLayer = self.toSlownessLayer(currVLayer, self.PWAVE)
            #
            #              * If we are in a fluid zone ( S velocity = 0.0 ) or if we are below
            #              * the outer core and allowInnerCoreS=false then use the P velocity
            #              * structure to look for critical points.
            #
            if inFluidZone or (belowOuterCore and not self.allowInnerCoreS):
                currSLayer = currPLayer
            else:
                currSLayer = self.toSlownessLayer(currVLayer, self.SWAVE)
            if prevSLayer.getBotP() != currSLayer.getTopP() or prevPLayer.getBotP() != currPLayer.getTopP():
                #  first order discontinuity
                self.criticalDepths.add(CriticalDepth(currSLayer.getTopDepth(), layerNum, -1, -1))
                if self.DEBUG:
                    print("first order discontinuity, depth=" + currSLayer.getTopDepth())
                    print(prevSLayer + "\n" + currSLayer)
                    print(prevPLayer + "\n" + currPLayer)
                if inHighSlownessZoneS and (currSLayer.getTopP() < minSSoFar):
                    #  top of current layer is the bottom of a high slowness
                    #  zone.
                    if self.DEBUG:
                        print("top of current layer is the bottom" + " of a high slowness zone.")
                    highSlownessZoneS.botDepth = currSLayer.getTopDepth()
                    self.highSlownessLayerDepthsS.add(highSlownessZoneS)
                    inHighSlownessZoneS = False
                if inHighSlownessZoneP and (currPLayer.getTopP() < minPSoFar):
                    #  top of current layer is the bottom of a high slowness
                    #  zone.
                    if self.DEBUG:
                        print("top of current layer is the bottom" + " of a high slowness zone.")
                    highSlownessZoneP.botDepth = currSLayer.getTopDepth()
                    self.highSlownessLayerDepthsP.add(highSlownessZoneP)
                    inHighSlownessZoneP = False
                #
                #                  * Update minPSoFar and minSSoFar as all total reflections off
                #                  * of the top of the discontinuity are ok even though below the
                #                  * discontinuity could be the start of a high slowness zone.
                #
                if minPSoFar > currPLayer.getTopP():
                    minPSoFar = currPLayer.getTopP()
                if minSSoFar > currSLayer.getTopP():
                    minSSoFar = currSLayer.getTopP()
                if not inHighSlownessZoneS and (prevSLayer.getBotP() < currSLayer.getTopP() or currSLayer.getTopP() < currSLayer.getBotP()):
                    #  start of a high slowness zone
                    if self.DEBUG:
                        print("Found S high slowness at first order " + "discontinuity, layer = " + layerNum)
                    inHighSlownessZoneS = True
                    highSlownessZoneS = DepthRange()
                    highSlownessZoneS.topDepth = currSLayer.getTopDepth()
                    highSlownessZoneS.rayParam = minSSoFar
                if not inHighSlownessZoneP and (prevPLayer.getBotP() < currPLayer.getTopP() or currPLayer.getTopP() < currPLayer.getBotP()):
                    #  start of a high slowness zone
                    if self.DEBUG:
                        print("Found P high slowness at first order " + "discontinuity, layer = " + layerNum)
                    inHighSlownessZoneP = True
                    highSlownessZoneP = DepthRange()
                    highSlownessZoneP.topDepth = currPLayer.getTopDepth()
                    highSlownessZoneP.rayParam = minPSoFar
            else:
                if (prevSLayer.getTopP() - prevSLayer.getBotP()) * (prevSLayer.getBotP() - currSLayer.getBotP()) < 0.0 or (prevPLayer.getTopP() - prevPLayer.getBotP()) * (prevPLayer.getBotP() - currPLayer.getBotP()) < 0.0:
                    #  local slowness extrema
                    self.criticalDepths.add(CriticalDepth(currSLayer.getTopDepth(), layerNum, -1, -1))
                    if self.DEBUG:
                        print("local slowness extrema, depth=" + currSLayer.getTopDepth())
                    if not inHighSlownessZoneP and (currPLayer.getTopP() < currPLayer.getBotP()):
                        #  start of a high slowness zone
                        if self.DEBUG:
                            print("start of a P high slowness zone," + " local slowness extrema, minPSoFar=" + minPSoFar)
                        inHighSlownessZoneP = True
                        highSlownessZoneP = DepthRange()
                        highSlownessZoneP.topDepth = currPLayer.getTopDepth()
                        highSlownessZoneP.rayParam = minPSoFar
                    if not inHighSlownessZoneS and (currSLayer.getTopP() < currSLayer.getBotP()):
                        #  start of a high slowness zone
                        if self.DEBUG:
                            print("start of a S high slowness zone," + " local slowness extrema, minSSoFar=" + minSSoFar)
                        inHighSlownessZoneS = True
                        highSlownessZoneS = DepthRange()
                        highSlownessZoneS.topDepth = currSLayer.getTopDepth()
                        highSlownessZoneS.rayParam = minSSoFar
            if inHighSlownessZoneP and (currPLayer.getBotP() < minPSoFar):
                #  layer contains the bottom of a high slowness zone.
                if self.DEBUG:
                    print("layer contains the bottom of a P " + "high slowness zone. minPSoFar=" + minPSoFar + " " + currPLayer)
                highSlownessZoneP.botDepth = findDepth(minPSoFar, layerNum, layerNum, self.PWAVE)
                self.highSlownessLayerDepthsP.add(highSlownessZoneP)
                inHighSlownessZoneP = False
            if inHighSlownessZoneS and (currSLayer.getBotP() < minSSoFar):
                #  layer contains the bottom of a high slowness zone.
                if self.DEBUG:
                    print("layer contains the bottom of a S " + "high slowness zone. minSSoFar=" + minSSoFar + " " + currSLayer)
                #  in fluid layers we want to check PWAVE structure when looking for S wave critical points
                highSlownessZoneS.botDepth = findDepth(minSSoFar, layerNum, layerNum, self.PWAVE if (currSLayer == currPLayer) else self.SWAVE)
                self.highSlownessLayerDepthsS.add(highSlownessZoneS)
                inHighSlownessZoneS = False
            if minPSoFar > currPLayer.getBotP():
                minPSoFar = currPLayer.getBotP()
            if minPSoFar > currPLayer.getTopP():
                minPSoFar = currPLayer.getTopP()
            if minSSoFar > currSLayer.getBotP():
                minSSoFar = currSLayer.getBotP()
            if minSSoFar > currSLayer.getTopP():
                minSSoFar = currSLayer.getTopP()
            if self.DEBUG and inHighSlownessZoneS:
                print("In S high slowness zone, layerNum = " + layerNum + " minSSoFar=" + minSSoFar)
            if self.DEBUG and inHighSlownessZoneP:
                print("In P high slowness zone, layerNum = " + layerNum + " minPSoFar=" + minPSoFar)
            layerNum += 1
        #  We know that the bottommost depth is always a critical slowness,
        #  so we add vMod.getNumLayers()
        self.criticalDepths.add(CriticalDepth(self.getRadiusOfEarth(), self.vMod.getNumLayers(), -1, -1))
        #  Check if the bottommost depth is contained within a high slowness
        #  zone, might happen in a flat non-whole-earth model
        if inHighSlownessZoneS:
            highSlownessZoneS.botDepth = currVLayer.getBotDepth()
            self.highSlownessLayerDepthsS.add(highSlownessZoneS)
        if inHighSlownessZoneP:
            highSlownessZoneP.botDepth = currVLayer.getBotDepth()
            self.highSlownessLayerDepthsP.add(highSlownessZoneP)
        #
        #          * Check if the bottommost depth is contained within a fluid zone, this
        #          * would be the case if we have a non whole earth model with the bottom
        #          * in the outer core or if allowInnerCoreS == false and we want to use
        #          * the P velocity structure in the inner core.
        #
        if inFluidZone:
            fluidZone.botDepth = currVLayer.getBotDepth()
            self.fluidLayerDepths.add(fluidZone)
        if self.DEBUG and len(self.criticalDepths) != 0:
            botCriticalLayerNum = self.criticalDepths.get(0).getVelLayerNum() - 1
            while criticalNum < len(self.criticalDepths):
                topCriticalLayerNum = botCriticalLayerNum + 1
                botCriticalLayerNum = self.criticalDepths.get(criticalNum).getVelLayerNum() - 1
                desc += " " + topCriticalLayerNum + "," + botCriticalLayerNum
                criticalNum += 1
            print(desc)
        if self.DEBUG and len(self.highSlownessLayerDepthsP) != 0:
            while layerNum < len(self.highSlownessLayerDepthsP):
                print(self.highSlownessLayerDepthsP.get(layerNum))
                layerNum += 1
        if self.DEBUG and len(self.highSlownessLayerDepthsS) != 0:
            while layerNum < len(self.highSlownessLayerDepthsS):
                print(self.highSlownessLayerDepthsS.get(layerNum))
                layerNum += 1
        if not validate():
            raise SlownessModelException("Validation Failed!")

    #
    #      * Finds a depth corresponding to a slowness over the whole VelocityModel.
    #      * Calls findDepth(double, int, int, char).
    #
    @overloaded
    def findDepth(self, rayParam, isPWave):
        """ generated source for method findDepth """
        return self.findDepth(rayParam, 0, self.vMod.getNumLayers() - 1, isPWave)

    #
    #      * Finds a depth corresponding to a slowness between two given depths in the
    #      * Velocity Model. Calls findDepth(double, int, int, char).
    #
    @findDepth.register(object, float, float, float, bool)
    def findDepth_0(self, rayParam, topDepth, botDepth, isPWave):
        """ generated source for method findDepth_0 """
        try:
            if self.vMod.getVelocityLayer(topLayerNum).getBotDepth() == topDepth:
                topLayerNum += 1
            return self.findDepth(rayParam, topLayerNum, botLayerNum, isPWave)
        except NoSuchLayerException as e:
            raise SlownessModelException(e.getMessage())

    #
    #      * Finds a depth corresponding to a slowness between two given velocity
    #      * layers, including the top and the bottom. We also check to see if the
    #      * slowness is less than the bottom slowness of these layers but greater
    #      * than the top slowness of the next deeper layer. This corresponds to a
    #      * total reflection. In this case a check needs to be made to see if this is
    #      * an S wave reflecting off of a fluid layer, use P velocity below in this
    #      * case. We assume that slowness is monotonic within these layers and
    #      * therefore there is only one depth with the given slowness. This means we
    #      * return the first depth that we find.
    #      *
    #      * @exception SlownessModelException
    #      *                occurs if topCriticalLayer > botCriticalLayer because
    #      *                there are no layers to search, or if there is an increase
    #      *                in slowness, ie a negative velocity gradient, that just
    #      *                balances the decrease in slowness due to the spherical
    #      *                earth, or if the ray parameter p is not contained within
    #      *                the specified layer range.
    #
    @findDepth.register(object, float, int, int, bool)
    def findDepth_1(self, p, topCriticalLayer, botCriticalLayer, isPWave):
        """ generated source for method findDepth_1 """
        velLayer = None
        topP = Double.MAX_VALUE
        botP = Double.MAX_VALUE
        topVelocity = float()
        botVelocity = float()
        depth = float()
        slope = float()
        waveType = str()
        if isPWave:
            waveType = 'P'
        else:
            waveType = 'S'
        try:
            if topCriticalLayer > botCriticalLayer:
                raise SlownessModelException("findDepth: no layers to search!: " + "topCriticalLayer = " + topCriticalLayer + "botCriticalLayer = " + botCriticalLayer)
            while layerNum <= botCriticalLayer:
                velLayer = self.vMod.getVelocityLayer(layerNum)
                topVelocity = velLayer.evaluateAtTop(waveType)
                botVelocity = velLayer.evaluateAtBottom(waveType)
                topP = self.toSlowness(topVelocity, velLayer.getTopDepth())
                botP = self.toSlowness(botVelocity, velLayer.getBotDepth())
                #
                #                  * check to see if we are within chatter level of the top or
                #                  * bottom and if so then return that depth.
                #
                if Math.abs(topP - p) < self.slownessTolerance:
                    return velLayer.getTopDepth()
                if Math.abs(p - botP) < self.slownessTolerance:
                    return velLayer.getBotDepth()
                if (topP - p) * (p - botP) >= 0.0:
                    #  found the layer
                    #  containing p
                    #
                    #                      * We interpolate assuming that velocity is linear within
                    #                      * this interval. So slope is the slope for velocity versus
                    #                      * depth.
                    #
                    slope = (botVelocity - topVelocity) / (velLayer.getBotDepth() - velLayer.getTopDepth())
                    depth = self.interpolate(p, topVelocity, velLayer.getTopDepth(), slope)
                    return depth
                elif layerNum == topCriticalLayer and Math.abs(p - topP) < self.slownessTolerance:
                    #
                    #                      * Check to see if p is just outside the topmost layer. If
                    #                      * so than return the top depth.
                    #
                    return velLayer.getTopDepth()
                #
                #                  * Is p a total reflection? botP is the slowness at the bottom
                #                  * of the last velocity layer from the previous loop, set topP
                #                  * to be the slowness at the top of the next layer.
                #
                if layerNum < self.vMod.getNumLayers() - 1:
                    velLayer = self.vMod.getVelocityLayer(layerNum + 1)
                    topVelocity = velLayer.evaluateAtTop(waveType)
                    if not isPWave and self.depthInFluid(velLayer.getTopDepth()):
                        #
                        #                          * Special case for S waves above a fluid. If top next
                        #                          * layer is in a fluid then we should set topVelocity to
                        #                          * be the P velocity at the top of the layer.
                        #
                        topVelocity = velLayer.evaluateAtTop('P')
                    topP = self.toSlowness(topVelocity, velLayer.getTopDepth())
                    if botP >= p and p >= topP:
                        return velLayer.getTopDepth()
                layerNum += 1
            if Math.abs(p - botP) < self.slownessTolerance:
                #
                #                  * Check to see if p is just outside the bottommost layer. If so
                #                  * than return the bottom depth.
                #
                print(" p is just outside the bottommost layer." + " This probably shouldn't be allowed to happen!\n")
                return velLayer.getBotDepth()
        except NoSuchMatPropException as e:
            #  can't happen...
            e.printStackTrace()
        raise SlownessModelException("slowness p=" + p + " is not contained within the specified layers." + "\np=" + p + " topCriticalLayer=" + topCriticalLayer + " botCriticalLayer=" + botCriticalLayer + " isPWave=" + isPWave + " topP=" + topP + " botP=" + botP)

    #
    #      * This method takes a velocity model and creates a vector containing
    #      * slowness-depth layers that, hopefully, adequately sample both slowness
    #      * and depth so that the travel time as a function of distance can be
    #      * reconstructed from the theta function. It catches NoSuchLayerException
    #      * which might be generated in the velocity model. This shouldn't happen
    #      * though.
    #      *
    #      * @see VelocityModel
    #      * @exception SlownessModelException
    #      *                occurs if the validation on the velocity model fails, or
    #      *                if the velocity model has no layers.
    #      * @exception NoSuchMatPropException
    #      *                occurs if wavetype is not recognized.
    #
    def createSample(self):
        """ generated source for method createSample """
        #  First check to make sure velocity model is ok.
        if self.vMod.validate() == False:
            raise SlownessModelException("Error in velocity model!")
        if self.vMod.getNumLayers() == 0:
            raise SlownessModelException("velModel.getNumLayers()==0")
        if self.vMod.getVelocityLayer(0).getTopSVelocity() == 0:
            raise SlownessModelException("Unable to handle zero S velocity layers at surface. This should be fixed at some point, but is a limitation of TauP at this point.")
        if self.DEBUG:
            print("start createSample")
        self.setRadiusOfEarth(self.vMod.getRadiusOfEarth())
        if self.DEBUG:
            print("findCriticalPoints")
        self.findCriticalPoints()
        if self.DEBUG:
            print("coarseSample")
        coarseSample()
        isOK = False
        if self.DEBUG:
            isOK = validate()
            print("rayParamCheck")
        rayParamIncCheck()
        if self.DEBUG:
            isOK &= validate()
            print("depthIncCheck")
        depthIncCheck()
        if self.DEBUG:
            isOK &= validate()
            print("distanceCheck")
        distanceCheck()
        if self.DEBUG:
            isOK &= validate()
            print("fixCriticalPoints")
        fixCriticalPoints()
        if self.DEBUG:
            print("done createSample")

    #
    #      * Creates a coarse slowness sampling of the velocity model (vMod). The
    #      * resultant slowness layers will satisfy the maximum depth increments as
    #      * well as sampling each point specified within the VelocityModel. The P and
    #      * S sampling will also be compatible.
    #
    def coarseSample(self):
        """ generated source for method coarseSample """
        prevVLayer = VelocityLayer()
        origVLayer = VelocityLayer()
        currVLayer = VelocityLayer()
        currPLayer = SlownessLayer()
        currSLayer = SlownessLayer()
        self.PLayers.clear()
        self.SLayers.clear()
        #  to initialize prevVLayer
        origVLayer = self.vMod.getVelocityLayer(0)
        origVLayer = VelocityLayer(0, origVLayer.getTopDepth(), origVLayer.getTopDepth(), origVLayer.getTopPVelocity(), origVLayer.getTopPVelocity(), origVLayer.getTopSVelocity(), origVLayer.getTopSVelocity(), origVLayer.getTopDensity(), origVLayer.getTopDensity(), origVLayer.getTopQp(), origVLayer.getTopQp(), origVLayer.getTopQs(), origVLayer.getTopQs())
        layerNum = 0
        while layerNum < self.vMod.getNumLayers():
            prevVLayer = origVLayer
            origVLayer = self.vMod.getVelocityLayer(layerNum)
            #
            #              * Check for first order discontinuity. However, we only
            #              * consider S discontinuities in the inner core if
            #              * allowInnerCoreS is true.
            #
            if prevVLayer.getBotPVelocity() != origVLayer.getTopPVelocity() or (prevVLayer.getBotSVelocity() != origVLayer.getTopSVelocity() and (self.allowInnerCoreS or origVLayer.getTopDepth() < self.vMod.getIocbDepth())):
                #
                #                  * if we are going from a fluid to a solid or solid to
                #                  * fluid, ex core mantle or outer core to inner core then we
                #                  * need to use the P velocity for determining the S
                #                  * discontinuity.
                #
                if prevVLayer.getBotSVelocity() == 0.0:
                    topSVel = prevVLayer.getBotPVelocity()
                else:
                    topSVel = prevVLayer.getBotSVelocity()
                if origVLayer.getTopSVelocity() == 0.0:
                    botSVel = origVLayer.getTopPVelocity()
                else:
                    botSVel = origVLayer.getTopSVelocity()
                currVLayer = VelocityLayer(layerNum, prevVLayer.getBotDepth(), prevVLayer.getBotDepth(), prevVLayer.getBotPVelocity(), origVLayer.getTopPVelocity(), topSVel, botSVel)
                #
                #                  * Add the zero thickness, but with nonzero slowness step,
                #                  * layer corresponding to the discontinuity.
                #
                currPLayer = self.toSlownessLayer(currVLayer, self.PWAVE)
                self.PLayers.add(currPLayer)
                if (prevVLayer.getBotSVelocity() == 0.0 and origVLayer.getTopSVelocity() == 0.0) or (not self.allowInnerCoreS and currVLayer.getTopDepth() >= self.vMod.getIocbDepth()):
                    currSLayer = currPLayer
                else:
                    currSLayer = self.toSlownessLayer(currVLayer, self.SWAVE)
                self.SLayers.add(currSLayer)
            currPLayer = self.toSlownessLayer(origVLayer, self.PWAVE)
            self.PLayers.add(currPLayer)
            if self.depthInFluid(origVLayer.getTopDepth()) or (not self.allowInnerCoreS and origVLayer.getTopDepth() >= self.vMod.getIocbDepth()):
                currSLayer = currPLayer
            else:
                currSLayer = self.toSlownessLayer(origVLayer, self.SWAVE)
            self.SLayers.add(currSLayer)
            layerNum += 1
        #  make sure that all high slowness layers are sampled exactly
        #  at their bottom
        highZoneNum = int()
        SLayerNum = int()
        highSLayer = SlownessLayer()
        highZone = DepthRange()
        while highZoneNum < len(self.highSlownessLayerDepthsS):
            highZone = self.highSlownessLayerDepthsS.get(highZoneNum)
            SLayerNum = layerNumberAbove(highZone.botDepth, self.SWAVE)
            highSLayer = self.getSlownessLayer(SLayerNum, self.SWAVE)
            while highSLayer.getTopDepth() == highSLayer.getBotDepth() and (highSLayer.getTopP() - highZone.rayParam) * (highZone.rayParam - highSLayer.getBotP()) < 0:
                SLayerNum += 1
                highSLayer = self.getSlownessLayer(SLayerNum, self.SWAVE)
            if highZone.rayParam != highSLayer.getBotP():
                addSlowness(highZone.rayParam, self.SWAVE)
            highZoneNum += 1
        while highZoneNum < len(self.highSlownessLayerDepthsP):
            highZone = self.highSlownessLayerDepthsP.get(highZoneNum)
            SLayerNum = layerNumberAbove(highZone.botDepth, self.PWAVE)
            highSLayer = self.getSlownessLayer(SLayerNum, self.PWAVE)
            while highSLayer.getTopDepth() == highSLayer.getBotDepth() and (highSLayer.getTopP() - highZone.rayParam) * (highZone.rayParam - highSLayer.getBotP()) < 0:
                SLayerNum += 1
                highSLayer = self.getSlownessLayer(SLayerNum, self.PWAVE)
            if highZone.rayParam != highSLayer.getBotP():
                addSlowness(highZone.rayParam, self.PWAVE)
            highZoneNum += 1
        #  make sure P and S sampling are consistant
        botP = -1
        topP = -1
        j = 0
        while j < len(self.PLayers):
            topP = self.PLayers.get(j).getTopP()
            if topP != botP:
                addSlowness(topP, self.SWAVE)
            botP = self.PLayers.get(j).getBotP()
            addSlowness(botP, self.SWAVE)
            j += 1
        botP = -1
        j = 0
        while j < len(self.SLayers):
            topP = self.SLayers.get(j).getTopP()
            if topP != botP:
                addSlowness(topP, self.PWAVE)
            botP = self.SLayers.get(j).getBotP()
            addSlowness(botP, self.PWAVE)
            j += 1

    #
    #      * Checks to make sure that no slowness layer spans more than maxDeltaP.
    #
    def rayParamIncCheck(self):
        """ generated source for method rayParamIncCheck """
        sLayer = SlownessLayer()
        numNewP = float()
        deltaP = float()
        j = 0
        while j < len(self.SLayers):
            sLayer = self.SLayers.get(j)
            if Math.abs(sLayer.getTopP() - sLayer.getBotP()) > self.maxDeltaP:
                numNewP = Math.ceil(Math.abs(sLayer.getTopP() - sLayer.getBotP()) / self.maxDeltaP)
                deltaP = (sLayer.getTopP() - sLayer.getBotP()) / numNewP
                while rayNum < numNewP:
                    addSlowness(sLayer.getTopP() + rayNum * deltaP, self.PWAVE)
                    addSlowness(sLayer.getTopP() + rayNum * deltaP, self.SWAVE)
                    rayNum += 1
            j += 1
        j = 0
        while j < len(self.PLayers):
            sLayer = self.PLayers.get(j)
            if Math.abs(sLayer.getTopP() - sLayer.getBotP()) > self.maxDeltaP:
                numNewP = Math.ceil(Math.abs(sLayer.getTopP() - sLayer.getBotP()) / self.maxDeltaP)
                deltaP = (sLayer.getTopP() - sLayer.getBotP()) / numNewP
                while rayNum < numNewP:
                    addSlowness(sLayer.getTopP() + rayNum * deltaP, self.PWAVE)
                    addSlowness(sLayer.getTopP() + rayNum * deltaP, self.SWAVE)
                    rayNum += 1
            j += 1

    #
    #      * Checks to make sure no slowness layer spans more than maxDepthInterval.
    #
    def depthIncCheck(self):
        """ generated source for method depthIncCheck """
        sLayer = SlownessLayer()
        numNewDepths = int()
        deltaDepth = float()
        velocity = float()
        p = float()
        try:
            while j < len(self.SLayers):
                sLayer = self.SLayers.get(j)
                if (sLayer.getBotDepth() - sLayer.getTopDepth()) > self.maxDepthInterval:
                    numNewDepths = int(Math.ceil((sLayer.getBotDepth() - sLayer.getTopDepth()) / self.maxDepthInterval))
                    deltaDepth = (sLayer.getBotDepth() - sLayer.getTopDepth()) / numNewDepths
                    while depthNum < numNewDepths:
                        velocity = self.vMod.evaluateAbove(sLayer.getTopDepth() + depthNum * deltaDepth, 'S')
                        if velocity == 0.0 or (not self.allowInnerCoreS and sLayer.getTopDepth() + depthNum * deltaDepth >= self.vMod.getIocbDepth()):
                            velocity = self.vMod.evaluateAbove(sLayer.getTopDepth() + depthNum * deltaDepth, 'P')
                        p = self.toSlowness(velocity, sLayer.getTopDepth() + depthNum * deltaDepth)
                        addSlowness(p, self.PWAVE)
                        addSlowness(p, self.SWAVE)
                        depthNum += 1
                j += 1
            while j < len(self.PLayers):
                sLayer = self.PLayers.get(j)
                if (sLayer.getBotDepth() - sLayer.getTopDepth()) > self.maxDepthInterval:
                    numNewDepths = int(Math.ceil((sLayer.getBotDepth() - sLayer.getTopDepth()) / self.maxDepthInterval))
                    deltaDepth = (sLayer.getBotDepth() - sLayer.getTopDepth()) / numNewDepths
                    while depthNum < numNewDepths:
                        p = self.toSlowness(self.vMod.evaluateAbove(sLayer.getTopDepth() + depthNum * deltaDepth, 'P'), sLayer.getTopDepth() + depthNum * deltaDepth)
                        addSlowness(p, self.PWAVE)
                        addSlowness(p, self.SWAVE)
                        depthNum += 1
                j += 1
        except NoSuchMatPropException as e:
            raise RuntimeException("can't happen", e)

    #
    #      * Checks to make sure no slowness layer spans more than maxRangeInterval
    #      * and that the (estimated) error due to linear interpolation is less than
    #      * maxInterpError.
    #
    def distanceCheck(self):
        """ generated source for method distanceCheck """
        sLayer = SlownessLayer()
        prevSLayer = SlownessLayer()
        j = int()
        prevTD = TimeDist()
        currTD = TimeDist()
        prevPrevTD = TimeDist()
        isCurrOK = bool()
        isPrevOK = bool()
        currWaveType = bool()
        #  TRUE=P and FALSE=S
        #  do SWAVE and then PWAVE, waveN is ONLY used on the next 2 lines
        waveN = 0
        while waveN < 2:
            currWaveType = self.SWAVE if waveN == 0 else self.PWAVE
            prevPrevTD = None
            prevTD = None
            currTD = None
            isCurrOK = False
            isPrevOK = False
            j = 0
            sLayer = self.getSlownessLayer(0, currWaveType)
            #  preset sLayer so
            #  prevSLayer is ok
            while j < self.getNumLayers(currWaveType):
                prevSLayer = sLayer
                sLayer = self.getSlownessLayer(j, currWaveType)
                if not self.depthInHighSlowness(sLayer.getBotDepth(), sLayer.getBotP(), currWaveType) and not self.depthInHighSlowness(sLayer.getTopDepth(), sLayer.getTopP(), currWaveType):
                    #  Don't calculate prevTD if we can avoid it
                    if isCurrOK:
                        if isPrevOK:
                            prevPrevTD = prevTD
                        else:
                            prevPrevTD = None
                        prevTD = currTD
                        isPrevOK = True
                    else:
                        prevTD = self.approxDistance(j - 1, sLayer.getTopP(), currWaveType)
                        isPrevOK = True
                    currTD = self.approxDistance(j, sLayer.getBotP(), currWaveType)
                    isCurrOK = True
                    #  check for too great of distance jump
                    if Math.abs(prevTD.distRadian - currTD.distRadian) > self.maxRangeInterval and Math.abs(sLayer.getTopP() - sLayer.getBotP()) > 2 * self.minDeltaP:
                        if self.DEBUG:
                            print(" " + j + "Dist jump too great: " + Math.abs(prevTD.distRadian - currTD.distRadian) + " > " + self.maxRangeInterval + "  adding slowness: " + (sLayer.getTopP() + sLayer.getBotP()) / 2.0)
                        addSlowness((sLayer.getTopP() + sLayer.getBotP()) / 2.0, self.PWAVE)
                        addSlowness((sLayer.getTopP() + sLayer.getBotP()) / 2.0, self.SWAVE)
                        currTD = prevTD
                        prevTD = prevPrevTD
                    else:
                        #  make guess as to error estimate due to linear
                        #  interpolation
                        #  if it is not ok, then we split both the previous and
                        #  current
                        #  slowness layers, this has the nice, if unintended,
                        #  consequense
                        #  of adding extra samples in the neighborhood of poorly
                        #  sampled
                        #  caustics
                        #                         if(Math.abs(prevTD.time
                        #                                     - ((currTD.time - prevPrevTD.time)
                        #                                             * (prevTD.dist - prevPrevTD.dist)
                        #                                             / (currTD.dist - prevPrevTD.dist) + prevPrevTD.time)) > maxInterpError) {
                        if Math.abs(currTD.time - ((splitTD.time - prevTD.time) * (currTD.distRadian - prevTD.distRadian) / (splitTD.distRadian - prevTD.distRadian) + prevTD.time)) > self.maxInterpError:
                            if self.DEBUG:
                                print(" " + j + " add slowness " + Math.abs(currTD.time - ((splitTD.time - prevTD.time, * (currTD.distRadian - prevTD.distRadian) / (splitTD.distRadian - prevTD.distRadian) + prevTD.time)) + " > " + self.maxInterpError))
                            addSlowness((prevSLayer.getTopP() + prevSLayer.getBotP()) / 2.0, self.PWAVE)
                            addSlowness((prevSLayer.getTopP() + prevSLayer.getBotP()) / 2.0, self.SWAVE)
                            addSlowness((sLayer.getTopP() + sLayer.getBotP()) / 2.0, self.PWAVE)
                            addSlowness((sLayer.getTopP() + sLayer.getBotP()) / 2.0, self.SWAVE)
                            currTD = prevPrevTD
                            isPrevOK = False
                            if j > 0:
                                #  back up one step unless we are at beginning, then stay put
                                j -= 1
                                sLayer = self.getSlownessLayer((j - 1 if (j - 1 >= 0) else 0), currWaveType)
                                #  ^^^ make sure j != 0
                                #  this sLayer will become prevSLayer in next loop
                            else:
                                isPrevOK = False
                                isCurrOK = False
                        else:
                            j += 1
                            if self.DEBUG and (j % 10 == 0):
                                print(j)
                else:
                    prevPrevTD = None
                    prevTD = None
                    currTD = None
                    isCurrOK = False
                    isPrevOK = False
                    j += 1
                    if self.DEBUG and (j % 100 == 0):
                        print(" " + j),
            if self.DEBUG:
                print( "\nNumber of " + ('P' if currWaveType else 'S') + " slowness layers: " + j)
            waveN += 1

    #
    #      * Adds the given ray parameter, p, to the slowness sampling for the given
    #      * waveType. It splits slowness layers as needed and keeps P and S sampling
    #      * consistant within fluid layers. Note, this makes use of the velocity
    #      * model, so all interpolation is linear in velocity, not in slowness!
    #      *
    #
    def addSlowness(self, p, isPWave):
        """ generated source for method addSlowness """
        layers = List()
        otherLayers = List()
        sLayer = SlownessLayer()
        topLayer = SlownessLayer()
        botLayer = SlownessLayer()
        slope = float()
        topVelocity = float()
        botVelocity = float()
        otherIndex = int()
        if isPWave:
            layers = self.PLayers
            otherLayers = self.SLayers
        else:
            layers = self.SLayers
            otherLayers = self.PLayers
        i = 0
        while i < len(layers):
            sLayer = layers.get(i)
            try:
                if sLayer.getTopDepth() != sLayer.getBotDepth():
                    topVelocity = self.vMod.evaluateBelow(sLayer.getTopDepth(), ('P' if isPWave else 'S'))
                    botVelocity = self.vMod.evaluateAbove(sLayer.getBotDepth(), ('P' if isPWave else 'S'))
                else:
                    #  if depths are same we really only need topVelocity,
                    #  and just to verify that we are not in a fluid.
                    topVelocity = self.vMod.evaluateAbove(sLayer.getBotDepth(), ('P' if isPWave else 'S'))
                    botVelocity = self.vMod.evaluateBelow(sLayer.getTopDepth(), ('P' if isPWave else 'S'))
            except NoSuchMatPropException as e:
                #  Can't happen but...
                raise SlownessModelException("Caught NoSuchMatPropException: " + e.getMessage())
            #  We don't need to check for S waves in a fluid or
            #  in inner core if allowInnerCoreS==false.
            if not isPWave:
                if not self.allowInnerCoreS and sLayer.getBotDepth() > self.vMod.getIocbDepth():
                    break
                elif topVelocity == 0.0:
                    continue
            if (sLayer.getTopP() - p) * (p - sLayer.getBotP()) > 0:
                if sLayer.getBotDepth() != sLayer.getTopDepth():
                    #
                    #                      * not a zero thickness layer, so calculate the depth for
                    #                      * the ray parameter.
                    #
                    slope = (botVelocity - topVelocity) / (sLayer.getBotDepth() - sLayer.getTopDepth())
                    botDepth = self.interpolate(p, topVelocity, sLayer.getTopDepth(), slope)
                botLayer = SlownessLayer(p, botDepth, sLayer.getBotP(), sLayer.getBotDepth())
                topLayer = SlownessLayer(sLayer.getTopP(), sLayer.getTopDepth(), p, botDepth)
                layers.remove(i)
                layers.add(i, botLayer)
                layers.add(i, topLayer)
                otherIndex = otherLayers.indexOf(sLayer)
                if otherIndex != -1:
                    otherLayers.remove(otherIndex)
                    otherLayers.add(otherIndex, botLayer)
                    otherLayers.add(otherIndex, topLayer)
            i += 1

    #
    #      * Resets the slowness layers that correspond to critical points.
    #
    def fixCriticalPoints(self):
        """ generated source for method fixCriticalPoints """
        cd = CriticalDepth()
        sLayer = SlownessLayer()
        i = 0
        while i < len(self.criticalDepths):
            cd = self.criticalDepths.get(i)
            cd.setPLayerNum(layerNumberBelow(cd.getDepth(), self.PWAVE))
            sLayer = self.getSlownessLayer(cd.getPLayerNum(), self.PWAVE)
            if cd.getPLayerNum() == len(self.PLayers) - 1 and sLayer.getBotDepth() == cd.getDepth():
                cd.setPLayerNum(cd.getPLayerNum() + 1)
                #  want the last
                #  critical point to be
                #  the bottom of the
                #  last layer
            cd.setSLayerNum(layerNumberBelow(cd.getDepth(), self.SWAVE))
            sLayer = self.getSlownessLayer(cd.getSLayerNum(), self.SWAVE)
            if cd.getSLayerNum() == len(self.SLayers) - 1 and sLayer.getBotDepth() == cd.getDepth():
                cd.setSLayerNum(cd.getSLayerNum() + 1)
                #  want the last
                #  critical point to be
                #  the bottom of the
                #  last layer
            i += 1

    #  finds a layer that contains the depth. This may not be unique in the case of a depth on
    #      * a boundary in the velocity model due to zero thickness layers. If the uppermost or
    #      * lowermost layer containing the depth is needed, use layerNumberAbove() or layerNumberBelow().
    #
    def layerNumForDepth(self, depth, isPWave):
        """ generated source for method layerNumForDepth """
        tempLayer = SlownessLayer()
        layers = List()
        if isPWave:
            layers = self.PLayers
        else:
            layers = self.SLayers
        #  check to make sure depth is within the range available
        if depth < layers.get(0).getTopDepth() or layers.get(len(layers) - 1).getBotDepth() < depth:
            raise NoSuchLayerException(depth)
        tooSmallNum = 0
        tooLargeNum = len(layers) - 1
        currentNum = 0
        while True:
            if tooLargeNum - tooSmallNum < 3:
                #  end of Newton, just check
                while currentNum <= tooLargeNum:
                    tempLayer = self.getSlownessLayer(currentNum, isPWave)
                    if tempLayer.containsDepth(depth):
                        return currentNum
                    currentNum += 1
            else:
                currentNum = Math.round((tooSmallNum + tooLargeNum) / 2.0)
            tempLayer = self.getSlownessLayer(currentNum, isPWave)
            if tempLayer.getTopDepth() > depth:
                tooLargeNum = currentNum - 1
            elif tempLayer.getBotDepth() < depth:
                tooSmallNum = currentNum + 1
            else:
                return currentNum
            if tooSmallNum > tooLargeNum:
                raise RuntimeException("tooSmallNum (" + tooSmallNum + ") >= tooLargeNum (" + tooLargeNum + ")")

    #
    #      * Finds the index of the slowness layer that contains the given depth Note
    #      * that if the depth is a layer boundary, it returns the shallower of the
    #      * two or possibly more (since total reflections are zero thickness layers)
    #      * layers.
    #      *
    #      * @return the layer number.
    #      * @exception NoSuchLayerException
    #      *                occurs if no layer in the slowness model contains the
    #      *                given depth.
    #
    def layerNumberAbove(self, depth, isPWave):
        """ generated source for method layerNumberAbove """
        foundLayerNum = self.layerNumForDepth(depth, isPWave)
        tempLayer = self.getSlownessLayer(foundLayerNum, isPWave)
        while tempLayer.getTopDepth() == depth and foundLayerNum > 0:
            foundLayerNum -= 1
            tempLayer = self.getSlownessLayer(foundLayerNum, isPWave)
        return foundLayerNum

    #
    #      * Finds the index of the slowness layer that contains the given depth Note
    #      * that if the depth is a layer boundary, it returns the deeper of the two
    #      * or possibly more (since total reflections are zero thickness layers)
    #      * layers.
    #      *
    #      * @return the layer number.
    #      * @exception NoSuchLayerException
    #      *                occurs if no layer in the slowness model contains the
    #      *                given depth.
    #
    def layerNumberBelow(self, depth, isPWave):
        """ generated source for method layerNumberBelow """
        foundLayerNum = self.layerNumForDepth(depth, isPWave)
        layers = List()
        if isPWave:
            layers = self.PLayers
        else:
            layers = self.SLayers
        tempLayer = self.getSlownessLayer(foundLayerNum, isPWave)
        while tempLayer.getBotDepth() == depth and foundLayerNum < len(layers) - 1:
            foundLayerNum += 1
            tempLayer = self.getSlownessLayer(foundLayerNum, isPWave)
        return foundLayerNum

    #
    #      * Performs consistency check on the slowness model.
    #      *
    #      * @return true if successful, throws SlownessModelException otherwise.
    #      * @exception SlownessModelException
    #      *                if any check fails
    #
    def validate(self):
        """ generated source for method validate """
        isOK = True
        prevDepth = float()
        highSZoneDepth = DepthRange()
        fluidZone = DepthRange()
        #  is radiusOfEarth positive?
        if self.radiusOfEarth <= 0.0:
            raise SlownessModelException("Radius of earth is not positive. radiusOfEarth = " + self.radiusOfEarth)
        #  is maxDepthInterval positive?
        if self.maxDepthInterval <= 0.0:
            raise SlownessModelException("maxDepthInterval is not positive. maxDepthInterval = " + self.maxDepthInterval)
        #  Check for inconsistencies in high slowness zones.
        highSlownessLayerDepths = self.highSlownessLayerDepthsP
        isPWave = self.PWAVE
        j = 0
        while j < 2:
            if isPWave:
                highSlownessLayerDepths = self.highSlownessLayerDepthsP
            else:
                highSlownessLayerDepths = self.highSlownessLayerDepthsS
            prevDepth = -1 * Double.MAX_VALUE
            while i < len(highSlownessLayerDepths):
                highSZoneDepth = highSlownessLayerDepths.get(i)
                if highSZoneDepth.topDepth >= highSZoneDepth.botDepth:
                    raise SlownessModelException("High slowness zone has zero or negative thickness. Num " + i + " isPWave=" + isPWave + " top depth " + highSZoneDepth.topDepth + " bottom depth " + highSZoneDepth.botDepth)
                if highSZoneDepth.topDepth <= prevDepth:
                    raise SlownessModelException("High slowness zone overlaps previous zone. Num " + i + " isPWave=" + isPWave + " top depth " + highSZoneDepth.topDepth + " bottom depth " + highSZoneDepth.botDepth)
                prevDepth = highSZoneDepth.botDepth
                i += 1
            isPWave = self.SWAVE
        #  Check for inconsistencies in fluid zones.
        prevDepth = -1 * Double.MAX_VALUE
        i = 0
        while i < len(self.fluidLayerDepths):
            fluidZone = self.fluidLayerDepths.get(i)
            if fluidZone.topDepth >= fluidZone.botDepth:
                raise SlownessModelException("Fluid zone has zero or negative thickness. Num " + i + " top depth " + fluidZone.topDepth + " bottom depth " + fluidZone.botDepth)
            if fluidZone.topDepth <= prevDepth:
                raise SlownessModelException("Fluid zone overlaps previous zone. Num " + i + " top depth " + fluidZone.topDepth + " bottom depth " + fluidZone.botDepth)
            prevDepth = fluidZone.botDepth
            i += 1
        #  Check for inconsistencies in slowness layers.
        isPWave = self.PWAVE
        prevBotP = float()
        j = 0
        while j < 2:
            prevDepth = 0.0
            if self.getNumLayers(isPWave) > 0:
                sLayer = self.getSlownessLayer(0, isPWave)
                prevBotP = sLayer.getTopP()
            else:
                prevBotP = -1
            while i < self.getNumLayers(isPWave):
                sLayer = self.getSlownessLayer(i, isPWave)
                isOK &= sLayer.validate()
                if sLayer.getTopDepth() > prevDepth:
                    raise SlownessModelException("Gap of " + (sLayer.getTopDepth() - prevDepth) + " between slowness layers. Num " + i + " isPWave=" + isPWave + "  top " + prevSLayer + " bottom " + sLayer)
                if sLayer.getTopDepth() < prevDepth:
                    raise SlownessModelException("Slowness layer overlaps previous layer by " + (prevDepth - sLayer.getTopDepth()) + ". Num " + i + " isPWave=" + isPWave + " top depth " + sLayer.getTopDepth() + " bottom depth " + sLayer.getBotDepth())
                if sLayer.getTopP() != prevBotP:
                    raise SlownessModelException("Slowness layer gap/overlaps previous layer in slowness " + ". Num " + i + " isPWave=" + isPWave + " prevBotP= " + prevBotP + " prevSLayer= " + prevSLayer + " sLayer= " + sLayer)
                if Double.isNaN(sLayer.getTopDepth()):
                    raise SlownessModelException("Top depth is NaN, layerNum=" + i + " waveType=" + ('P' if isPWave else 'S'))
                if Double.isNaN(sLayer.getBotDepth()):
                    raise SlownessModelException("Top depth is NaN, layerNum=" + i + " waveType=" + ('P' if isPWave else 'S'))
                prevSLayer = sLayer
                prevBotP = sLayer.getBotP()
                prevDepth = sLayer.getBotDepth()
                i += 1
            isPWave = self.SWAVE
        #  Everything checks out OK so return true.
        return isOK

    def __str__(self):
        """ generated source for method toString """
        topCriticalLayerNum = int()
        botCriticalLayerNum = int()
        desc = ""
        desc = "radiusOfEarth=" + self.radiusOfEarth + "\n maxDeltaP=" + self.maxDeltaP + "\n minDeltaP=" + self.minDeltaP + "\n maxDepthInterval=" + self.maxDepthInterval + "\n maxRangeInterval=" + self.maxRangeInterval + "\n allowInnerCoreS=" + self.allowInnerCoreS + "\n slownessTolerance=" + self.slownessTolerance + "\n getNumLayers('P')=" + self.getNumLayers(self.PWAVE) + "\n getNumLayers('S')=" + self.getNumLayers(self.SWAVE) + "\n len(fluidLayerDepths)=" + len(self.fluidLayerDepths) + "\n len(highSlownessLayerDepthsP)=" + len(self.highSlownessLayerDepthsP) + "\n len(highSlownessLayerDepthsS)=" + len(self.highSlownessLayerDepthsS) + "\n len(criticalDepths)=" + len(self.criticalDepths) + "\n"
        if len(self.criticalDepths) != 0:
            desc += ("**** Critical Depth Layers ************************\n")
            botCriticalLayerNum = self.criticalDepths.get(0).getVelLayerNum() - 1
            while criticalNum < len(self.criticalDepths):
                topCriticalLayerNum = botCriticalLayerNum + 1
                botCriticalLayerNum = self.criticalDepths.get(criticalNum).getVelLayerNum() - 1
                desc += " " + topCriticalLayerNum + "," + botCriticalLayerNum
                criticalNum += 1
        desc += "\n"
        if len(self.fluidLayerDepths) != 0:
            desc += "\n**** Fluid Layer Depths ************************\n"
            while i < len(self.fluidLayerDepths):
                desc += self.fluidLayerDepths.get(i).topDepth + "," + self.fluidLayerDepths.get(i).botDepth + " "
                i += 1
        desc += "\n"
        if len(self.highSlownessLayerDepthsP) != 0:
            desc += "\n**** P High Slowness Layer Depths ****************\n"
            while i < len(self.highSlownessLayerDepthsP):
                desc += self.highSlownessLayerDepthsP.get(i).topDepth + "," + self.highSlownessLayerDepthsP.get(i).botDepth + " "
                i += 1
        desc += "\n"
        if len(self.highSlownessLayerDepthsS) != 0:
            desc += "\n**** S High Slowness Layer Depths ****************\n"
            while i < len(self.highSlownessLayerDepthsS):
                desc += self.highSlownessLayerDepthsS.get(i).topDepth + "," + self.highSlownessLayerDepthsS.get(i).botDepth + " "
                i += 1
        desc += "\n"
        desc += "\n**** P Layers ****************\n"
        for l in PLayers:
            desc += l.__str__() + "\n"
        return desc


