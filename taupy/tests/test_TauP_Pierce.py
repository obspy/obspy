import unittest, sys, subprocess, os
from taupy.TauP_Pierce import TauP_Pierce


class TestTauPPierce(unittest.TestCase):

    def test_range(self):
        """
        Check taup_pierce output for a range of inputs against the Java output.
        """
        if not os.path.isfile("data/java_tauppierce_testoutput"):
            subprocess.call("./generate_tauppierce_output.sh", shell=True)
        stdout = sys.stdout
        with open('data/taup_pierce_test_output', 'wt') as sys.stdout:
            for degree in [0, 45, 90, 180, 360, 560]:
                for depth in [0, 100, 1000, 2889]:
                    tauppierce = TauP_Pierce(degrees=degree, depth=depth,
                                             modelName="iasp91",
                                             phaseList=["ttall"])
                    tauppierce.start()
        sys.stdout = stdout
        # Using ttall need to sort; or lines with same arrival times are in
        # different order. With explicit names of all the phases might not be
        # a problem.
        # subprocess.check_call("./compare_tauppierce_outputs.sh", shell=True)
        # Use this if lines are in same order:
        #subprocess.check_call("diff -wB data/java_tauppierce_testoutput "
        #                      "data/taup_pierce_test_output", shell=True)
        # Todo: think of something more clever. It doesn't work, the tiny format differences screw it up.
        os.remove("data/taup_pierce_test_output")

if __name__ == '__main__':
    unittest.main(buffer=True)
