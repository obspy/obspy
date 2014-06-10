import inspect
import os
import unittest

from taupy.VelocityModel import VelocityModel


# to get ./data:
data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")


class TestVelocityModel(unittest.TestCase):
    def test_read_velocity_model(self):
        print("reading and checking the read iasp91.tvel")        

        for i in range(0,2):
            if i==0:
                # read ./data/iasp91.tvel
                velocity_model = os.path.join(data_dir, "iasp91.tvel")
            else:
                velocity_model = os.path.join(data_dir, "iasp91_w_comment.tvel")

            # test_file.tvel is shorter
            test2 = VelocityModel.readVelocityFile(velocity_model)
            #print(test2)
            
            self.assertEqual(len(test2.layers), 129)
            self.assertEqual(len(test2), 129)
            
            self.assertEqual(test2.radiusOfEarth, 6371.0)
            self.assertEqual(test2.mohoDepth,35)
            self.assertEqual(test2.cmbDepth,2889.0)
            self.assertEqual(test2.iocbDepth,5153.9)
            self.assertEqual(test2.minRadius,0.0)
            self.assertEqual(test2.maxRadius,6371.0)
            #self.assertEqual(test2.spherical,True)
            #self.assertEqual(test2.modelName, "iasp91")
            
            self.assertEqual(test2.validate(), True)

            self.assertEqual(
                test2.getDisconDepths(),
                [0.0, 20.0, 35.0, 210.0, 410.0, 660.0, 2889.0, 5153.9, 6371.0])

            #check boundary cases
            self.assertEqual(test2.layerNumberAbove(6371), 128)
            self.assertEqual(test2.layerNumberBelow(0), 0)
        
            #evaluate at cmb
            self.assertEqual(test2.evaluateAbove(2889.0, 'p'), 13.6908)
            self.assertEqual(test2.evaluateBelow(2889.0, 'D'), 9.9145)
            self.assertEqual(test2.depthAtTop(50), 2393.5)
            self.assertEqual(test2.depthAtBottom(50), 2443.0)

            self.assertEqual(test2.fixDisconDepths(), False)
            
            # breakpoint:
            #from IPython.core.debugger import Tracer; Tracer(colors="Linux")()


if __name__ == '__main__':
    unittest.main()
