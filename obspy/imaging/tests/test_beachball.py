# -*- coding: utf-8 -*-
"""
The obspy.imaging.beachball test suite.
"""

from obspy.core.util.decorator import skipIf
from obspy.imaging.beachball import Beachball, AuxPlane, StrikeDip, TDL, \
    MomentTensor, MT2Plane, MT2Axes, Beach
import matplotlib.pyplot as plt
import os
import unittest


class BeachballTestCase(unittest.TestCase):
    """
    Test cases for beachball generation.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'output')

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_Beachball(self):
        """
        Create beachball examples in tests/output directory.
        """
        # http://en.wikipedia.org/wiki/File:USGS_sumatra_mts.gif
        mt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]
        Beachball(mt, outfile=os.path.join(self.path, 'bb_sumatra_mt.png'))
        np1 = [274, 13, 55]
        Beachball(np1, outfile=os.path.join(self.path, 'bb_sumatra_np1.png'))
        np2 = [130, 79, 98]
        Beachball(np2, outfile=os.path.join(self.path, 'bb_sumatra_np2.png'))
        #
        np1 = [264.98, 45.00, -159.99]
        Beachball(np1, outfile=os.path.join(self.path, 'bb_19950128_np1.png'))
        np2 = [160.55, 76.00, -46.78]
        Beachball(np2, outfile=os.path.join(self.path, 'bb_19950128_np2.png'))
        #
        mt = [1.45, -6.60, 5.14, -2.67, -3.16, 1.36]
        Beachball(mt, outfile=os.path.join(self.path, 'bb_20090102_mt.png'))
        np1 = [235, 80, 35]
        Beachball(np1, outfile=os.path.join(self.path, 'bb_20090102_np1.png'))
        np2 = [138, 56, 168]
        Beachball(np2, outfile=os.path.join(self.path, 'bb-20090102-np2.png'))
        # Explosion
        mt = [1, 1, 1, 0, 0, 0]
        Beachball(mt, outfile=os.path.join(self.path, 'bb_explosion.png'))
        # Implosion
        mt = [-1, -1, -1, 0, 0, 0]
        Beachball(mt, outfile=os.path.join(self.path, 'bb_implosion.png'))
        # CLVD - Compensate Linear Vector Dipole
        mt = [1, -2, 1, 0, 0, 0]
        Beachball(mt, outfile=os.path.join(self.path, 'bb_clvd.png'))
        # Double Couple
        mt = [1, -1, 0, 0, 0, 0]
        Beachball(mt, outfile=os.path.join(self.path, 'bb_double_couple.png'))
        # Lars
        mt = [1, -1, 0, 0, 0, -1]
        Beachball(mt, outfile=os.path.join(self.path, 'bb_lars.png'))
        # http://wwweic.eri.u-tokyo.ac.jp/yuji/Aki-nada/
        np1 = [179, 55, -78]
        Beachball(np1, outfile=os.path.join(self.path, 'bb_geiyo_np1.png'))
        #
        np1 = [10, 42.5, 90]
        Beachball(np1, outfile=os.path.join(self.path, 'bb_honshu_np1.png'))
        np2 = [10, 42.5, 92]
        Beachball(np2, outfile=os.path.join(self.path, 'bb_honshu_np2.png'))
        # http://wwweic.eri.u-tokyo.ac.jp/yuji/tottori/
        np1 = [150, 87, 1]
        Beachball(np1, outfile=os.path.join(self.path, 'bb_tottori_np1.png'))
        # http://iisee.kenken.go.jp/staff/thara/2004/09/20040905_1/2nd.html
        mt = [0.99, -2.00, 1.01, 0.92, 0.48, 0.15]
        Beachball(mt, outfile=os.path.join(self.path, 'bb_20040905_1_mt.png'))
        # http://iisee.kenken.go.jp/staff/thara/2004/09/20040905_0/1st.html
        mt = [5.24, -6.77, 1.53, 0.81, 1.49, -0.05]
        Beachball(mt, outfile=os.path.join(self.path, 'bb_20040905_0_mt.png'))
        # http://iisee.kenken.go.jp/staff/thara/miyagi.htm
        mt = [16.578, -7.987, -8.592, -5.515, -29.732, 7.517]
        Beachball(mt, outfile=os.path.join(self.path, 'bb_miyagi_mt.png'))
        # http://iisee.kenken.go.jp/staff/thara/20050613/chile.html
        mt = [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94]
        Beachball(mt, outfile=os.path.join(self.path, 'bb_chile_mt.png'))

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_BeachBallOutputFormats(self):
        """
        Tests various output formats.
        """
        fm = [115, 35, 50]
        # PDF
        data = Beachball(fm, format='pdf')
        self.assertEquals(data[0:4], "%PDF")
        # as file
        Beachball(fm, format='pdf', outfile=os.path.join(self.path, 'bb.pdf'))
        # PS
        data = Beachball(fm, format='ps')
        self.assertEquals(data[0:4], "%!PS")
        # as file
        Beachball(fm, format='ps', outfile=os.path.join(self.path, 'bb.ps'))
        # PNG
        data = Beachball(fm, format='png')
        self.assertEquals(data[1:4], "PNG")
        # as file
        Beachball(fm, format='png', outfile=os.path.join(self.path, 'bb.png'))
        # SVG
        data = Beachball(fm, format='svg')
        self.assertEquals(data[0:5], "<?xml")
        # as file
        Beachball(fm, format='svg', outfile=os.path.join(self.path, 'bb.svg'))

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
        # http://en.wikipedia.org/wiki/File:USGS_sumatra_mts.gif
        s1 = 132.18005257215460
        d1 = 84.240987194376590
        r1 = 98.963372641038790
        (s2, d2, r2) = AuxPlane(s1, d1, r1)
        self.assertAlmostEqual(s2, 254.64386091007400)
        self.assertAlmostEqual(d2, 10.641291652406172)
        self.assertAlmostEqual(r2, 32.915578422454380)
        #
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
        # http://en.wikipedia.org/wiki/File:USGS_sumatra_mts.gif
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

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_Beach(self):
        """
        Tests to plot beachballs as collection into an existing axis
        object. The moment tensor values are taken form the
        test_Beachball unit test. See that test for more information about
        the parameters.
        """
        mt = [[0.91, -0.89, -0.02, 1.78, -1.55, 0.47],
              [274, 13, 55],
              [130, 79, 98],
              [264.98, 45.00, -159.99],
              [160.55, 76.00, -46.78],
              [1.45, -6.60, 5.14, -2.67, -3.16, 1.36],
              [235, 80, 35],
              [138, 56, 168],
              [1, 1, 1, 0, 0, 0],
              [-1, -1, -1, 0, 0, 0],
              [1, -2, 1, 0, 0, 0],
              [1, -1, 0, 0, 0, 0],
              [1, -1, 0, 0, 0, -1],
              [179, 55, -78],
              [10, 42.5, 90],
              [10, 42.5, 92],
              [150, 87, 1],
              [0.99, -2.00, 1.01, 0.92, 0.48, 0.15],
              [5.24, -6.77, 1.53, 0.81, 1.49, -0.05],
              [16.578, -7.987, -8.592, -5.515, -29.732, 7.517],
              [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94],
              [150, 87, 1]]

        # Initialize figure
        fig = plt.figure(1, figsize=(6, 6), dpi=300)
        ax = fig.add_subplot(111, aspect='equal')

        # Plot the stations or borders
        ax.plot([-100, -100, 100, 100], [-100, 100, -100, 100], 'rv')

        x = -100
        y = -100
        for i, t in enumerate(mt):
            # add the beachball (a collection of two patches) to the axis
            ax.add_collection(Beach(t, width=30, xy=(x, y), linewidth=.6))
            x += 50
            if (i + 1) % 5 == 0:
                x = -100
                y += 50

        # Set the x and y limits and save the output
        ax.axis([-120, 120, -120, 120])
        fig.savefig(os.path.join(self.path, 'bb_collection.png'))


def suite():
    return unittest.makeSuite(BeachballTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
