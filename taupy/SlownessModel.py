from math import pi
class SlownessModelError(Exception):
    pass
class SlownessModel(object):
    """Dummy for testing."""
    DEBUG = False
    DEFAULT_SLOWNESS_TOLERANCE = 500
    
    def __init__(self, vMod, minDeltaP=0.1, maxDeltaP=11, maxDepthInterval=115, maxRangeInterval=2.5*pi/180, maxInterpError=0.05, allowInnerCoreS=True, slowness_tolerance=500):
        
        self.vMod = vMod
        self.minDeltaP = minDeltaP
        self.maxDeltaP = maxDeltaP
        self.maxDepthInterval = maxDepthInterval
        self.maxRangeInterval = maxRangeInterval
        self.maxInterpError = maxInterpError
        self.isAllowInnerCoreS = allowInnerCoreS
        self.slowness_tolerance = slowness_tolerance

    def createSample(self):
        ''' Creates slowmess samples?'''
        # Some checks on the velocity model
        if self.vMod.validate() == False:
            raise SlownessModelError("Error in velocity model (vMod.validate failed)!")
        if self.vMod.getNumLayers() == 0:
            raise SlownessModelError("velModel.getNumLayers()==0")
        if self.vMod.layers[0].topSVelocity == 0:
            raise SlownessModelError("Unable to handle zero S velocity layers at surface. This should be fixed at some point, but is a limitation of TauP at this point.")
        if self.DEBUG:
            print("start createSample")

        self.radiusOfEarth = self.vMod.radiusOfEarth

        if self.DEBUG: print("findCriticalPoints")
        self.findCriticalPoints()
        if self.DEBUG: print("coarseSample")
        self.coarseSample()
        if self.DEBUG and self.validate() != True: 
            raise(SlownessModelError('validate failed after coarseSample'))
        if self.DEBUG: print("rayParamCheck")
        self.rayParamIncCheck()
        if self.DEBUG: print("depthIncCheck")
        self.depthIncCheck()
        if self.DEBUG: print("distanceCheck")
        self.distanceCheck()
        if self.DEBUG: print("fixCriticalPoints")
        self.fixCriticalPoints()
        
        if self.validate() == True: 
            print("createSample seems to be done successfully.")
        else:
            raise SlownessModelError('SlownessModel.validate failed!')

    def findCriticalPoints(self):
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
        #     return len(self.PLayers)
        # else:
        #     return len(self.SLayers)

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
        desc += "This might be interesting: slowness_tolerance ought to be 500. It is:" + str(self.slowness_tolerance)
        return desc
