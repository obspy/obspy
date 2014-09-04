import unittest

from taupy.SlownessModel import SlownessModel
from taupy.SlownessModel import SlownessModelError
from taupy.VelocityModel import VelocityModel
from taupy.SlownessLayer import SlownessLayer
from taupy.VelocityLayer import VelocityLayer


class TestSlownessModel(unittest.TestCase):
    """"WARNING: The values I'm testing can't be right. Half of the methods needed by createSample aren't implemented
    yet! However, as that is needed in the constructor of the SlownessModel, the other methods can't be tested
    independently of it. So I can test some (probably) always true boundary, but the intermediate testing values should
    at some point, as work progresses, start to throw errors. I could true unit tests, but the effort doesn't seem
    worth it at the moment.
    """

    # def test_createSample(self):
    #     testmod = SlownessModel(VelocityModel.readVelocityFile('./data/iasp91.tvel'))
    #     testmod.DEBUG = True
    #
    #     # An error should be raised if vMod.validate fails, or vMod
    #     # has 0 layers, or the top layer has a 0 S wave velocity. Only
    #     # the first is testable (I think?); setting the topSVelocity
    #     # to 0 makes vMod.validate fail.
    #     with self.assertRaises(SlownessModelError):
    #         testmod.vMod.layers[0].topSVelocity = 0
    #         testmod.createSample()
    #
    #     # test normal execution:
    #     testmod = SlownessModel(VelocityModel.readVelocityFile('./data/iasp91.tvel'))
    #     #testmod.DEBUG = True
    #     testmod.createSample()
    #     self.assertEqual(testmod.radiusOfEarth, 6371)
    #     self.assertTrue(testmod.validate())

    # def test_slownesslayer(self):
    #     vLayer=VelocityLayer(1, 10, 31, 3, 5, 2, 4)
    #     a = SlownessLayer.create_from_vlayer(vLayer, True)
    #     self.assertEqual(a.botP, 1268.0)
    #     self.assertEqual(a.botDepth, 31.0)
    #     b = SlownessLayer.create_from_vlayer(vLayer, False)
    #     self.assertEqual(b.topP, 3180.5)
    #
    # def test_findDepth(self):
    #     testmod = SlownessModel(VelocityModel.readVelocityFile('./data/iasp91.tvel'))
    #     self.assertEqual(testmod.findDepth(0, 0, 138, True), 6371.0)
    #     I am not quite sure if this is in fact the true value, I should try and get the java program to calculate it too.
    #     If it's necessary.
    #     self.assertEqual(testmod.findDepth(600, 0, 138, True), 524.3899204244029)


if __name__ == '__main__':
    unittest.main(buffer=True)
