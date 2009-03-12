# -*- coding: utf-8 -*-
"""
The obspy.imaging.beachball test suite.
"""

from obspy.imaging.beachball import Mij2SDR, Beachball, AuxPlane, StrikeDip, \
    TDL
import inspect
import os
import unittest


class BeachballTestCase(unittest.TestCase):
    """
    Test cases for beachball generation.
    """
    def setUp(self):
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'output')
    
    def tearDown(self):
        pass
    
    def test_Beachball(self):
        """
        Creates beachball examples in tests/ouput directory.
        """
        # Moment Tensor
        # @see: http://en.wikipedia.org/wiki/File:USGS_sumatra_mts.gif
        fm = [0.01, -0.89, -0.02, 1.78, -1.55, 0.47]
        Beachball(fm, file=os.path.join(self.path, 'sumatra.png'))
        
        # Plane: [Strike, Dip, Rake] 
        # @see: http://serc.carleton.edu/files/NAGTWorkshops/structure04/Focal_mechanism_primer.pdf
        fm = [115, 35, 50]
        Beachball(fm, file=os.path.join(self.path, 'primer.png'))
        
        # Explosion
        fm = [1, 1, 1, 0, 0, 0]
        Beachball(fm, file=os.path.join(self.path, 'explosion.png'))
        # Implosion
        fm = [-1, -1, -1, 0, 0, 0]
        Beachball(fm, file=os.path.join(self.path, 'implosion.png'))
        # Double Couple
        fm = [-1, -1, 0, 0, 0, 0]
        Beachball(fm, file=os.path.join(self.path, 'double-couple.png'))
        # CLVD - Compensate Linear Vector Dipole
        # XXX: not working
        fm = [1, -2, 1, 0, 0, 0]
        Beachball(fm, file=os.path.join(self.path, 'clvd-not-working.png'))
        
        # Plane: [Strike, Dip, Rake] 
        # @see: http://www.eas.slu.edu/Earthquake_Center/MECH.NA/19950128062621/index.html
        np1 = [264.98, 45.00, -159.99]
        Beachball(np1, file=os.path.join(self.path, '19950128062621-np1.png'))
        np2 = [160.55, 76.00, -46.78]
        Beachball(np2, file=os.path.join(self.path, '19950128062621-np2.png'))
        
        # Moment Tensor + Plane
        # @see: http://www.eas.slu.edu/Earthquake_Center/MECH.NA/20090102141713/index.html
        fm = [1.45, -6.60, 5.14, -2.67, -3.16, 1.36]
        Beachball(fm, file=os.path.join(self.path, '20090102141713-mt.png'))
        np1 = [235, 80, 35]
        Beachball(np1, file=os.path.join(self.path, '20090102141713-np1.png'))
        np2 = [138, 56, 168]
        Beachball(np2, file=os.path.join(self.path, '20090102141713-np2.png'))
        
        # XXX: not working
        fm = [1,-1,0,0,0,-1]
        Beachball(fm, file=os.path.join(self.path, 'lars-not-working.png'))
    
    def test_BeachBallOutputFormats(self):
        """
        Tests various output formats.
        """
        fm = [115, 35, 50]
        # PDF
        data = Beachball(fm, format='pdf')
        self.assertEquals(data[0:4], "%PDF")
        # PS
        data = Beachball(fm, format='ps')
        self.assertEquals(data[0:4], "%!PS")
        # PNG
        data = Beachball(fm, format='png')
        self.assertEquals(data[1:4], "PNG")
        # SVG
        data = Beachball(fm, format='svg')
        self.assertEquals(data[0:5], "<?xml")
    
    def test_StrikeDip(self):
        """
        Test StrikeDip function - all values are taken from MatLab.
        """
        sl1 = -0.048901208623019
        sl2 = 0.178067035725425
        sl3 = 0.982802524713469
        (strike, dip) = StrikeDip(sl2, sl1, sl3)
        self.assertAlmostEqual(strike, 254.64386091007400)
        self.assertAlmostEqual(dip, 10.641291652406172)
    
    def test_AuxPlane(self):
        """
        Test AuxPlane function - all values are taken from MatLab.
        """
        # @see: http://en.wikipedia.org/wiki/File:USGS_sumatra_mts.gif
        s1 = 132.18005257215460
        d1 = 84.240987194376590
        r1 = 98.963372641038790
        (s2, d2, r2) = AuxPlane(s1, d1, r1)
        self.assertAlmostEqual(s2, 254.64386091007400)
        self.assertAlmostEqual(d2, 10.641291652406172)
        self.assertAlmostEqual(r2, 32.915578422454380)
        # @see: http://www.eas.slu.edu/Earthquake_Center/MECH.NA/19950128062621/index.html
        s1 = 160.55
        d1 = 76.00
        r1 = -46.78
        (s2, d2, r2) = AuxPlane(s1, d1, r1)
        self.assertAlmostEqual(s2, 264.98676854650216)
        self.assertAlmostEqual(d2, 45.001906942415623)
        self.assertAlmostEqual(r2, -159.99404307049076)
    
    def test_Mij2SDR(self):
        """
        Test Mij2SDR function - all values are taken from MatLab.
        """
        [s1, d1, r1] = Mij2SDR(0.01, -0.89, -0.02, 1.78, -1.55, 0.47)
        self.assertAlmostEqual(s1, 132.18005257215460)
        self.assertAlmostEqual(d1, 84.240987194376590)
        self.assertAlmostEqual(r1, 98.963372641038790)
    
    def test_TDL(self):
        """
        Test TDL function - all values are taken from MatLab.
        """
        AN = [0.737298200871146,-0.668073596186761,-0.100344571703004]
        BN = [-0.178067035261159,-0.048901208638715,-0.982802524796805]
        (FT, FD, FL) = TDL(AN, BN)
        self.assertAlmostEqual(FT, 227.81994742784540)
        self.assertAlmostEqual(FD, 84.240987194376590)
        self.assertAlmostEqual(FL, 81.036627358961210)


def suite():
    return unittest.makeSuite(BeachballTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
