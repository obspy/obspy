import unittest
from math import pi
from taupy.VelocityModel import VelocityModel
from taupy.SlownessModel import SlownessModel



# noinspection PyPep8Naming
class SlownessmodelSimpleInit(SlownessModel):
    """Copy the definition of slownessModel class attributes and the init,
    BUT without the call to createSample in the end. Inherit all other methods."""
    # Inherits class attributes, but somehow can't use them in the init?
    def __init__(self, vMod, minDeltaP=0.1, maxDeltaP=11, maxDepthInterval=115, maxRangeInterval=2.5 * pi / 180,
                 maxInterpError=0.05, allowInnerCoreS=True, slowness_tolerance=1e-16, pLayers=[],
                 sLayers=[]):
        # Don't add a call to the super class init!
        self.vMod = vMod
        self.minDeltaP = minDeltaP
        self.maxDeltaP = maxDeltaP
        self.maxDepthInterval = maxDepthInterval
        self.maxRangeInterval = maxRangeInterval
        self.maxInterpError = maxInterpError
        self.allowInnerCoreS = allowInnerCoreS
        self.slowness_tolerance = slowness_tolerance
        self.PLayers = pLayers
        self.SLayers = sLayers
        # Extra, from createSample:
        self.radiusOfEarth = self.vMod.radiusOfEarth


# noinspection PyPep8Naming
class TestSlownessmodelSlowly(unittest.TestCase):
    vMod = VelocityModel.readVelocityFile('./data/iasp91.tvel')
    tm = SlownessmodelSimpleInit(vMod)

    def testSlownessModelSlowly(self):
        # Now do what createSample does, but step by step:
        # Check values in between by comparing with variable states from Java debugger.
        self.tm.findCriticalPoints()
        self.assertEqual(len(self.tm.criticalDepths), 9)
        self.assertEqual(self.tm.criticalDepths[1].sLayerNum, -1)
        self.assertEqual(self.tm.criticalDepths[3].velLayerNum, 6)
        self.assertEqual(self.tm.criticalDepths[6].depth, 2889.0)
        self.assertEqual(self.tm.fluidLayerDepths[0].botDepth, 5153.9)
        # Can't put these into separate test methods because they're not called in order!
        self.tm.coarseSample()
        self.assertEqual(len(self.tm.PLayers), 307)
        self.assertEqual(len(self.tm.SLayers), 286)
        self.assertEqual(self.tm.PLayers[40].botDepth, 410)
        self.assertEqual(self.tm.PLayers[40].topDepth, 410)  # Zero thickness layer, but p changes.
        self.assertEqual(self.tm.PLayers[305].botP, 4.511284884393324)  # Same as top of deepest layer.
        self.assertEqual(self.tm.SLayers[0].topP, 1896.1309523809525)
        self.assertEqual(self.tm.SLayers[284].botDepth, 6354.919811136119)  # Zero thickness layer, but p changes.
        self.assertEqual(self.tm.SLayers[284].botP, 4.511284884393324)

        self.tm.rayParamIncCheck()
        self.assertEqual(len(self.tm.PLayers), 343)
        self.assertEqual(len(self.tm.SLayers), 366)
        self.assertEqual(self.tm.PLayers[0].topP, 1098.448275862069)
        self.assertEqual(self.tm.PLayers[40].topDepth, 35)
        self.assertEqual(self.tm.PLayers[341].botP, 0.9955173285740244)
        self.assertEqual(self.tm.SLayers[100].topP, 821.5956109380346)
        self.assertEqual(self.tm.SLayers[200].topDepth, 3697.8839336701108)
        self.assertEqual(self.tm.SLayers[364].botP, 0.9955173285740244)

        self.tm.depthIncCheck()
        # No change to PLayers, SLayers, apparently.
        # Not ideal, because hard to check correct execution.

        #print(self.tm)
        self.tm.distanceCheck()
        self.tm.fixCriticalPoints()
        self.tm.criticalDepths

        self.assertTrue(self.tm.validate())


if __name__ == '__main__':
    unittest.main(buffer=True)
