# -*- coding: utf-8 -*-
"""
The obspy.imaging.beachball test suite.
"""

from numpy import array
from obspy.imaging.beachball import Mij2SDR, Beachball, AuxPlane, StrikeDip, \
    TDL
import unittest


class BeachballTestCase(unittest.TestCase):
    """
    Test cases for beachball generation.
    """
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_Beachball(self):
        """
        """
        fm = array([0.01, -0.89, -0.02, 1.78, -1.55, 0.47])
        Beachball(fm)
    
    def test_StrikeDip(self):
        """
        """
        sl1 = -0.048901208623019
        sl2 = 0.178067035725425
        sl3 = 0.982802524713469
        (strike, dip) = StrikeDip(sl2, sl1, sl3)
        self.assertAlmostEqual(strike, 254.64386091007400)
        self.assertAlmostEqual(dip, 10.641291652406172)
    
    def test_AuxPlane(self):
        """
        """
        s1 = 132.18005257215460
        d1 = 84.240987194376590
        r1 = 98.963372641038790
        (s2, d2, r2) = AuxPlane(s1, d1, r1)
        self.assertAlmostEqual(s2, 254.64386091007400)
        self.assertAlmostEqual(d2, 10.641291652406172)
        self.assertAlmostEqual(r2, 32.915578422454380)
    
    def test_Mij2SDR(self):
        """
        """
        [s1, d1, r1] = Mij2SDR(0.01, -0.89, -0.02, 1.78, -1.55, 0.47)
        self.assertAlmostEqual(s1, 132.18005257215460)
        self.assertAlmostEqual(d1, 84.240987194376590)
        self.assertAlmostEqual(r1, 98.963372641038790)
    
    def test_TDL(self):
        """
        """
        AN = array([0.737298200871146,-0.668073596186761,-0.100344571703004])
        BN = array([-0.178067035261159,-0.048901208638715,-0.982802524796805])
        (FT, FD, FL) = TDL(AN, BN)
        self.assertAlmostEqual(FT, 227.81994742784540)
        self.assertAlmostEqual(FD, 84.240987194376590)
        self.assertAlmostEqual(FL, 81.036627358961210)


def suite():
    return unittest.makeSuite(BeachballTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
