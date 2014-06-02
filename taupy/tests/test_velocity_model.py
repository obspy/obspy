import inspect
import os
import unittest

from taupy.VelocityModel import VelocityModel


data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")


class TestVelocityModel(unittest.TestCase):
    def test_read_velocity_model(self):
        velocity_model = os.path.join(data_dir, "iasp91.tvel")

        # test_file.tvel is shorter
        test2 = VelocityModel.readVelocityFile(velocity_model)

        self.assertEqual(len(test2.layers), 129)
        self.assertEqual(len(test2), 129)

        self.assertEqual(
            test2.getDisconDepths(),
            [0.0, 20.0, 35.0, 210.0, 410.0, 660.0, 2889.0, 5153.9, 6371.0])

        self.assertEqual(test2.layerNumberAbove(30), 1)
        self.assertEqual(test2.layerNumberBelow(0), 0)

        #from IPython.core.debugger import Tracer; Tracer(colors="Linux")()


if __name__ == '__main__':
    unittest.main()
