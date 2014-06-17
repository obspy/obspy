class SlownessModel(object):
    """Dummy for testing."""
    DEBUG = False
    DEFAULT_SLOWNESS_TOLERANCE = 500
    
    def __init__(self, vMod, minDeltaP, maxDeltaP, maxDepthInterval, maxRangeInterval, maxInterpError, allowInnerCoreS, slowness_tolerance):
        
        self.vMod = vMod
        self.minDeltaP = minDeltaP
        self.maxDeltaP = maxDeltaP
        self.maxDepthInterval = maxDepthInterval
        self.maxRangeInterval = maxRangeInterval
        self.maxInterpError = maxInterpError
        self.isAllowInnerCoreS = allowInnerCoreS
        self.slowness_tolerance = slowness_tolerance

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
        desc = "This is a dummy SphericalSModel so there's nothing here really. Nothing to see. Move on."
        desc += "This might be interesting: slowness_tolerance ought to be 500. It is:" + str(self.slowness_tolerance)
        return desc
