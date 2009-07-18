# -*- coding: utf-8 -*-
"""
The obspy.imaging.beachball test suite.
"""

from obspy.imaging.beachball import Beachball, AuxPlane, StrikeDip, TDL, \
    MomentTensor, MT2Plane, MT2Axes
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
        Create beachball examples in tests/output directory.
        """
        # @see: http://en.wikipedia.org/wiki/File:USGS_sumatra_mts.gif
        mt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]
        Beachball(mt, file=os.path.join(self.path,
                                        'beachball-sumatra-mt.png'))
        np1 = [274, 13, 55]
        Beachball(np1, file=os.path.join(self.path,
                                         'beachball-sumatra-np1.png'))
        np2 = [130, 79, 98]
        Beachball(np2, file=os.path.join(self.path,
                                         'beachball-sumatra-np2.png'))

        # @see: http://www.eas.slu.edu/Earthquake_Center/MECH.NA/19950128062621/index.html
        np1 = [264.98, 45.00, -159.99]
        Beachball(np1, file=os.path.join(self.path,
                                         'beachball-19950128062621-np1.png'))
        np2 = [160.55, 76.00, -46.78]
        Beachball(np2, file=os.path.join(self.path,
                                         'beachball-19950128062621-np2.png'))

        # @see: http://www.eas.slu.edu/Earthquake_Center/MECH.NA/20090102141713/index.html
        mt = [1.45, -6.60, 5.14, -2.67, -3.16, 1.36]
        Beachball(mt, file=os.path.join(self.path,
                                        'beachball-20090102141713-mt.png'))
        np1 = [235, 80, 35]
        Beachball(np1, file=os.path.join(self.path,
                                         'beachball-20090102141713-np1.png'))
        np2 = [138, 56, 168]
        Beachball(np2, file=os.path.join(self.path,
                                         'beachball-20090102141713-np2.png'))

        # Explosion
        mt = [1, 1, 1, 0, 0, 0]
        Beachball(mt, file=os.path.join(self.path,
                                        'beachball-explosion.png'))
        # Implosion
        mt = [-1, -1, -1, 0, 0, 0]
        Beachball(mt, file=os.path.join(self.path,
                                        'beachball-implosion.png'))
        # CLVD - Compensate Linear Vector Dipole
        mt = [1, -2, 1, 0, 0, 0]
        Beachball(mt, file=os.path.join(self.path,
                                        'beachball-clvd.png'))
        # Double Couple
        mt = [1, -1, 0, 0, 0, 0]
        Beachball(mt, file=os.path.join(self.path,
                                        'beachball-double-couple.png'))

        # Lars
        mt = [1, -1, 0, 0, 0, -1]
        Beachball(mt, file=os.path.join(self.path, 'beachball-lars.png'))

        # @see: http://wwweic.eri.u-tokyo.ac.jp/yuji/Aki-nada/
        np1 = [179, 55, -78]
        Beachball(np1, file=os.path.join(self.path, 'beachball-geiyo-np1.png'))

        # @see: http://iisee.kenken.go.jp/staff/yagi/eq/east_honshu20030726/east_honshu20030726.html
        np1 = [10, 42.5, 90]
        Beachball(np1, file=os.path.join(self.path,
                                         'beachball-honshu-np1.png'))
        np2 = [10, 42.5, 92]
        Beachball(np2, file=os.path.join(self.path,
                                         'beachball-honshu-np2.png'))

        # @see: http://wwweic.eri.u-tokyo.ac.jp/yuji/tottori/
        np1 = [150, 87, 1]
        Beachball(np1, file=os.path.join(self.path,
                                         'beachball-tottori-np1.png'))

        # @see: http://iisee.kenken.go.jp/staff/thara/2004/09/20040905_1/2nd.html
        mt = [0.99, -2.00, 1.01, 0.92, 0.48, 0.15]
        Beachball(mt, file=os.path.join(self.path,
                                        'beachball-20040905_1-mt.png'))

        # @see: http://iisee.kenken.go.jp/staff/thara/2004/09/20040905_0/1st.html
        mt = [5.24, -6.77, 1.53, 0.81, 1.49, -0.05]
        Beachball(mt, file=os.path.join(self.path,
                                        'beachball-20040905_0-mt.png'))

        # @see: http://iisee.kenken.go.jp/staff/thara/miyagi.htm
        mt = [16.578, -7.987, -8.592, -5.515, -29.732, 7.517]
        Beachball(mt, file=os.path.join(self.path, 'beachball-miyagi-mt.png'))

        # @see: http://iisee.kenken.go.jp/staff/thara/20050613/chile.html
        mt = [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94]
        Beachball(mt, file=os.path.join(self.path, 'beachball-chile-mt.png'))

    def test_BeachBallOutputFormats(self):
        """
        Tests various output formats.
        """
        fm = [115, 35, 50]
        # PDF
        data = Beachball(fm, format='pdf')
        self.assertEquals(data[0:4], "%PDF")
        # as file
        Beachball(fm, format='pdf', file=os.path.join(self.path,
                                                      'beachball.pdf'))
        # PS
        data = Beachball(fm, format='ps')
        self.assertEquals(data[0:4], "%!PS")
        # as file
        Beachball(fm, format='ps', file=os.path.join(self.path,
                                                     'beachball.ps'))
        # PNG
        data = Beachball(fm, format='png')
        self.assertEquals(data[1:4], "PNG")
        # as file
        Beachball(fm, format='png', file=os.path.join(self.path,
                                                      'beachball.png'))
        # SVG
        data = Beachball(fm, format='svg')
        self.assertEquals(data[0:5], "<?xml")
        # as file
        Beachball(fm, format='svg', file=os.path.join(self.path,
                                                      'beachball.svg'))

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

    def test_TDL(self):
        """
        Test TDL function - all values are taken from MatLab.
        """
        AN = [0.737298200871146, -0.668073596186761, -0.100344571703004]
        BN = [-0.178067035261159, -0.048901208638715, -0.982802524796805]
        (FT, FD, FL) = TDL(AN, BN)
        self.assertAlmostEqual(FT, 227.81994742784540)
        self.assertAlmostEqual(FD, 84.240987194376590)
        self.assertAlmostEqual(FL, 81.036627358961210)

    def test_MT2Plane(self):
        """
        Tests MT2Plane. 
        """
        mt = MomentTensor((0.91, -0.89, -0.02, 1.78, -1.55, 0.47), 0)
        np = MT2Plane(mt)
        self.assertAlmostEqual(np.strike, 129.86262672080011)
        self.assertAlmostEqual(np.dip, 79.022700906654734)
        self.assertAlmostEqual(np.rake, 97.769255185515192)

    def test_MT2Axes(self):
        """
        Tests MT2Axes.
        """
        # @see: http://en.wikipedia.org/wiki/File:USGS_sumatra_mts.gif
        mt = MomentTensor((0.91, -0.89, -0.02, 1.78, -1.55, 0.47), 0)
        (T, N, P) = MT2Axes(mt)
        self.assertAlmostEqual(T.val, 2.52461359)
        self.assertAlmostEqual(T.dip, 55.33018576)
        self.assertAlmostEqual(T.strike, 49.53656116)
        self.assertAlmostEqual(N.val, 0.08745048)
        self.assertAlmostEqual(N.dip, 7.62624529)
        self.assertAlmostEqual(N.strike, 308.37440488)
        self.assertAlmostEqual(P.val, -2.61206406)
        self.assertAlmostEqual(P.dip, 33.5833323)
        self.assertAlmostEqual(P.strike, 213.273886)


def suite():
    return unittest.makeSuite(BeachballTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
