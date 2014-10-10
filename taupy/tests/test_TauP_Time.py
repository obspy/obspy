import unittest, sys, subprocess, os
from taupy.TauP_Time import TauP_Time


class TestTauPTime(unittest.TestCase):

    def test_main(self):
        tt = TauP_Time(["S", "P"], "iasp91", 200, 57.4)
        tt.start()
        self.assertEqual(tt.tMod.tauBranches[0][3].maxRayParam,
                         742.2891566265059)
        # Of course tMod should be correct, if TauP_Create works as it should.
        # As far as I could tell, it does, even though it's a bit difficult
        # to compare to the java all at once and there may be some rounding
        # differences some 10 digits behind the comma.

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

    def test_range(self):
        """
        Check taup_time output for a range of inputs against the Java output.
        """
        if not os.path.isfile("data/java_tauptime_testoutput"):
            subprocess.call("./generate_tauptime_output.sh", shell=True)
        stdout = sys.stdout
        with open('taup_time_test_output', 'wt') as sys.stdout:
            for degree in [0, 45, 90, 180, 360, 560]:
                for depth in [0, 100, 1000, 2889]:
                    tauptime = TauP_Time(degrees=degree, depth=depth,
                                         modelName="iasp91",
                                         phaseList=["ttall"])
                    tauptime.start()
        sys.stdout = stdout
        # Using ttall need to sort; or lines with same arrival times are in
        # different order. With explicit names of all the phases might not be
        # a problem.
        subprocess.check_call("./compare_tauptime_outputs.sh", shell=True)
        # Use this if lines are in same order:
        #subprocess.check_call("diff -wB data/java_tauptime_testoutput "
        #                     "taup_time_test_output", shell=True)
        os.remove("taup_time_test_output")

if __name__ == '__main__':
    unittest.main(buffer=True)
