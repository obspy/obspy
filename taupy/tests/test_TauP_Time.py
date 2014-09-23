import unittest
from taupy.TauP_Time import TauP_Time


class TestTauPTime(unittest.TestCase):
    #def setUp(self):
        # Runs before the tests!
    #    self.tauPTime = TauP_Time()

    def test_main(self):
        tt = TauP_Time()
        # Pseudo cmd line args:
        tt.phaseNames = ["S", "P"]
        tt.modelName = "iasp91"
        tt.degrees = 57.4
        tt.depth = 200

        tt.init()
        self.assertEqual(tt.tMod, tt.tModDepth)
        self.assertEqual(tt.tMod.tauBranches[0][3].maxRayParam, 742.2891566265059)
        # Of course tMod should be correct, if TauP_Create works as it should.
        # As far as I could tell, it does, even though it's a bit difficult to compare to the java all at once
        # and there may be some rounding differences some 10 digits behind the comma.
        self.assertEqual(len(tt.phases), 0)

        # tt.start()
        # Calls, given that tt.degrees isn't None:
        tt.depthCorrect(tt.depth)
        self.assertEqual(tt.tModDepth.sourceDepth, 200)
        # Checking the corected tModDepth will be difficult again...
        # Also calls recalcPhases, so check phases here...


if __name__ == '__main__':
    unittest.main(buffer=True)