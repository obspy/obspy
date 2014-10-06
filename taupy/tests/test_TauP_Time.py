#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from taupy.TauP_Time import TauP_Time


class TestTauPTime(unittest.TestCase):
    # def setUp(self):
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
        self.assertEqual(tt.tMod.tauBranches[0][3].maxRayParam,
                         742.2891566265059)
        # Of course tMod should be correct, if TauP_Create works as it should.
        # As far as I could tell, it does, even though it's a bit difficult
        # to compare to the java all at once and there may be some rounding
        # differences some 10 digits behind the comma.
        self.assertEqual(len(tt.phases), 0)

        # tt.start()
        # Calls, given that tt.degrees isn't None:
        tt.depthCorrect(tt.depth)
        self.assertEqual(tt.tModDepth.sourceDepth, 200)
        # Checking the corected tModDepth will be difficult again...
        # Also calls recalcPhases, so check phases here...
        self.assertEqual(tt.tMod.tauBranches[0][7],
                         tt.tModDepth.tauBranches[0][8])
        self.assertEqual(tt.tModDepth.tauBranches[0][8].maxRayParam,
                         109.7336675261915)
        self.assertEqual(tt.tMod.rayParams, tt.tModDepth.rayParams)
        self.assertEqual(tt.tModDepth.noDisconDepths[0], 200)
        self.assertEqual(len(tt.tModDepth.sMod.PLayers), 562)
        self.assertEqual(tt.tModDepth.sMod.PLayers[105].topP,
                         745.9964237713114)
        self.assertEqual(tt.tModDepth, tt.phases[0].tMod)
        self.assertEqual(len(tt.tModDepth.sMod.criticalDepths), 9)
        self.assertTrue(all(len(x) == 234 for x in (
            tt.phases[0].dist, tt.phases[0].time, tt.phases[0].rayParams)))
        self.assertTrue(all(len(x) == 179 for x in (
            tt.phases[1].dist, tt.phases[1].time, tt.phases[1].rayParams)))
        self.assertEqual(tt.phases[1].puristName, "P")
        # ought to be ...02 but close enough
        self.assertEqual(tt.phases[0].dist[232], 1.7004304436716804)
        self.assertEqual(tt.phases[1].dist[121], 0.9912173304550589)
        tt.calculate(tt.degrees)
        self.assertEqual(tt.arrivals[1].time, 1028.9304953527787)


if __name__ == '__main__':
    unittest.main(buffer=True)
