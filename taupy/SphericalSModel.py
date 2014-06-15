
class SphericalSModel(object):
    """ Dummy class for testing."""

    # You should try extending the SlownessModel
    # and inheriting the value of the DEBUG flag!

    def __init__(self, vMod, minDeltaP, maxDeltaP, maxDepthInterval, maxRangeInterval, maxInterpError, allowInnerCoreS, slowness_tolerance):
        
        self.vMod = vMod
        self.minDeltaP = minDeltaP
        self.maxDeltaP = maxDeltaP
        self.maxDepthInterval = maxDeltaP
        self.maxRangeInterval = maxRangeInterval
        self.maxInterpError = maxInterpError
        self.isAllowInnerCoreS = allowInnerCoreS
        self.slowness_tolerance = slowness_tolerance
        
    def getNumLayers(self, pors):
        if pors == True:
            return 'some dummy number'
        if pors == False:
            return 'some ther number'

    def __str__(self):
        desc = "This is a dummy SphericalSModel so there's nothing here really. Nothing to see. Move on."
        desc += "This might be interesting: slowness_tolerance ought to be 500. It is:" + str(self.slowness_tolerance)
        return desc
              
    
    
