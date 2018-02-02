# -*- coding: utf-8 -*-
"""
The obspy.imaging.beachball test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest
import warnings

import matplotlib.pyplot as plt

from obspy.core.util.base import NamedTemporaryFile
from obspy.core.util.testing import ImageComparison
from obspy.imaging.beachball import (tdl, aux_plane, beach, beachball,
                                     MomentTensor, mt2axes, mt2plane,
                                     strike_dip)


class BeachballTestCase(unittest.TestCase):
    """
    Test cases for beachball generation.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'images')

    def test_beachball(self):
        """
        Create beachball examples in tests/output directory.
        """
        # https://en.wikipedia.org/wiki/File:USGS_sumatra_mts.gif
        data = [[0.91, -0.89, -0.02, 1.78, -1.55, 0.47],
                [274, 13, 55],
                [130, 79, 98],
                [264.98, 45.00, -159.99],
                [160.55, 76.00, -46.78],
                [1.45, -6.60, 5.14, -2.67, -3.16, 1.36],
                [235, 80, 35],
                [138, 56, 168],
                # Explosion
                [1, 1, 1, 0, 0, 0],
                # Implosion
                [-1, -1, -1, 0, 0, 0],
                # CLVD - Compensate Linear Vector Dipole
                [1, -2, 1, 0, 0, 0],
                # Double Couple
                [1, -1, 0, 0, 0, 0],
                # Lars
                [1, -1, 0, 0, 0, -1],
                # https://wwweic.eri.u-tokyo.ac.jp/yuji/Aki-nada/
                [179, 55, -78],
                [10, 42.5, 90],
                [10, 42.5, 92],
                # https://wwweic.eri.u-tokyo.ac.jp/yuji/tottori/
                [150, 87, 1],
                # http://iisee.kenken.go.jp/staff/thara/2004/09/20040905_1/
                # 2nd.html
                [0.99, -2.00, 1.01, 0.92, 0.48, 0.15],
                # http://iisee.kenken.go.jp/staff/thara/2004/09/20040905_0/
                # 1st.html
                [5.24, -6.77, 1.53, 0.81, 1.49, -0.05],
                # http://iisee.kenken.go.jp/staff/thara/miyagi.htm
                [16.578, -7.987, -8.592, -5.515, -29.732, 7.517],
                # http://iisee.kenken.go.jp/staff/thara/20050613/chile.html
                [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94],
                ]
        filenames = ['bb_sumatra_mt.png', 'bb_sumatra_np1.png',
                     'bb_sumatra_np2.png', 'bb_19950128_np1.png',
                     'bb_19950128_np2.png', 'bb_20090102_mt.png',
                     'bb_20090102_np1.png', 'bb-20090102-np2.png',
                     'bb_explosion.png', 'bb_implosion.png', 'bb_clvd.png',
                     'bb_double_couple.png', 'bb_lars.png', 'bb_geiyo_np1.png',
                     'bb_honshu_np1.png', 'bb_honshu_np2.png',
                     'bb_tottori_np1.png', 'bb_20040905_1_mt.png',
                     'bb_20040905_0_mt.png', 'bb_miyagi_mt.png',
                     'bb_chile_mt.png',
                     ]
        for data_, filename in zip(data, filenames):
            with ImageComparison(self.path, filename) as ic:
                beachball(data_, outfile=ic.name)

    def test_beachball_output_format(self):
        """
        Tests various output formats.
        """
        fm = [115, 35, 50]
        # PDF - Some matplotlib versions internally raise some warnings here
        # which we don't want to see in the tests.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = beachball(fm, format='pdf')
            self.assertEqual(data[0:4], b"%PDF")
            # as file
            # create and compare image
            with NamedTemporaryFile(suffix='.pdf') as tf:
                beachball(fm, format='pdf', outfile=tf.name)
        # PS
        data = beachball(fm, format='ps')
        self.assertEqual(data[0:4], b"%!PS")
        # as file
        with NamedTemporaryFile(suffix='.ps') as tf:
            beachball(fm, format='ps', outfile=tf.name)
        # PNG
        data = beachball(fm, format='png')
        self.assertEqual(data[1:4], b"PNG")
        # as file
        with NamedTemporaryFile(suffix='.png') as tf:
            beachball(fm, format='png', outfile=tf.name)
        # SVG
        data = beachball(fm, format='svg')
        self.assertEqual(data[0:5], b"<?xml")
        # as file
        with NamedTemporaryFile(suffix='.svg') as tf:
            beachball(fm, format='svg', outfile=tf.name)

    def test_strike_dip(self):
        """
        Test strike_dip function - all values are taken from MatLab.
        """
        sl1 = -0.048901208623019
        sl2 = 0.178067035725425
        sl3 = 0.982802524713469
        (strike, dip) = strike_dip(sl2, sl1, sl3)
        self.assertAlmostEqual(strike, 254.64386091007400)
        self.assertAlmostEqual(dip, 10.641291652406172)

    def test_aux_plane(self):
        """
        Test aux_plane function - all values are taken from MatLab.
        """
        # https://en.wikipedia.org/wiki/File:USGS_sumatra_mts.gif
        s1 = 132.18005257215460
        d1 = 84.240987194376590
        r1 = 98.963372641038790
        (s2, d2, r2) = aux_plane(s1, d1, r1)
        self.assertAlmostEqual(s2, 254.64386091007400)
        self.assertAlmostEqual(d2, 10.641291652406172)
        self.assertAlmostEqual(r2, 32.915578422454380)
        #
        s1 = 160.55
        d1 = 76.00
        r1 = -46.78
        (s2, d2, r2) = aux_plane(s1, d1, r1)
        self.assertAlmostEqual(s2, 264.98676854650216)
        self.assertAlmostEqual(d2, 45.001906942415623)
        self.assertAlmostEqual(r2, -159.99404307049076)

    def test_aux_plane_735(self):
        """
        Test aux_plane precision issue #735
        """
        s, d, r = aux_plane(164, 90, -32)
        self.assertAlmostEqual(s, 254.)
        self.assertAlmostEqual(d, 58.)
        self.assertAlmostEqual(r, -180.)

    def test_tdl(self):
        """
        Test tdl function - all values are taken from MatLab.
        """
        an = [0.737298200871146, -0.668073596186761, -0.100344571703004]
        bn = [-0.178067035261159, -0.048901208638715, -0.982802524796805]
        (ft, fd, fl) = tdl(an, bn)
        self.assertAlmostEqual(ft, 227.81994742784540)
        self.assertAlmostEqual(fd, 84.240987194376590)
        self.assertAlmostEqual(fl, 81.036627358961210)

    def test_mt2plane(self):
        """
        Tests mt2plane.
        """
        mt = MomentTensor((0.91, -0.89, -0.02, 1.78, -1.55, 0.47), 0)
        np = mt2plane(mt)
        self.assertAlmostEqual(np.strike, 129.86262672080011)
        self.assertAlmostEqual(np.dip, 79.022700906654734)
        self.assertAlmostEqual(np.rake, 97.769255185515192)

    def test_mt2axes(self):
        """
        Tests mt2axes.
        """
        # https://en.wikipedia.org/wiki/File:USGS_sumatra_mts.gif
        mt = MomentTensor((0.91, -0.89, -0.02, 1.78, -1.55, 0.47), 0)
        (T, N, P) = mt2axes(mt)
        self.assertAlmostEqual(T.val, 2.52461359)
        self.assertAlmostEqual(T.dip, 55.33018576)
        self.assertAlmostEqual(T.strike, 49.53656116)
        self.assertAlmostEqual(N.val, 0.08745048)
        self.assertAlmostEqual(N.dip, 7.62624529)
        self.assertAlmostEqual(N.strike, 308.37440488)
        self.assertAlmostEqual(P.val, -2.61206406)
        self.assertAlmostEqual(P.dip, 33.5833323)
        self.assertAlmostEqual(P.strike, 213.273886)

    def test_collection(self):
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

        with ImageComparison(self.path, 'bb_collection.png') as ic:
            # Initialize figure
            fig = plt.figure(figsize=(6, 6), dpi=300)
            ax = fig.add_subplot(111, aspect='equal')

            # Plot the stations or borders
            ax.plot([-100, -100, 100, 100], [-100, 100, -100, 100], 'rv')

            x = -100
            y = -100
            for i, t in enumerate(mt):
                # add the beachball (a collection of two patches) to the axis
                ax.add_collection(beach(t, width=30, xy=(x, y), linewidth=.6))
                x += 50
                if (i + 1) % 5 == 0:
                    x = -100
                    y += 50

            # set the x and y limits
            ax.axis([-120, 120, -120, 120])

            # save the output
            fig.savefig(ic.name)

    def collection_aspect(self, axis, filename_width, filename_width_height):
        """
        Common part of the test_collection_aspect_[xy] tests.
        """
        mt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]

        # Test passing only a width
        with ImageComparison(self.path, filename_width) as ic:
            # Initialize figure
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # add the beachball (a collection of two patches) to the axis
            # give it an axes to keep make the beachballs circular
            # even though axes are not scaled
            ax.add_collection(beach(mt, width=400, xy=(0, 0), linewidth=.6,
                                    axes=ax))
            # set the x and y limits
            ax.axis(axis)
            fig.savefig(ic.name)

        # Test passing a width and a height
        with ImageComparison(self.path, filename_width_height) as ic:
            # Initialize figure
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # add the beachball (a collection of two patches) to the axis
            # give it an axes to keep make the beachballs circular
            # even though axes are not scaled
            ax.add_collection(beach(mt, width=(400, 200), xy=(0, 0),
                                    linewidth=.6, axes=ax))
            # set the x and y limits
            ax.axis(axis)
            # save the output
            fig.savefig(ic.name)

    def test_collection_aspect_x(self):
        """
        Tests to plot beachball into a non-scaled axes with an x-axis larger
        than y-axis. Use the 'axes' kwarg to make beachballs circular.
        """
        self.collection_aspect(axis=[-10000, 10000, -100, 100],
                               filename_width='bb_aspect_x.png',
                               filename_width_height='bb_aspect_x_height.png')

    def test_collection_aspect_y(self):
        """
        Tests to plot beachball into a non-scaled axes with a y-axis larger
        than x-axis. Use the 'axes' kwarg to make beachballs circular.
        """
        self.collection_aspect(axis=[-100, 100, -10000, 10000],
                               filename_width='bb_aspect_y.png',
                               filename_width_height='bb_aspect_y_height.png')

    def test_mopad_fallback(self):
        """
        Test the fallback to mopad.
        """
        mt = [0.000, -1.232e25, 1.233e25, 0.141e25, -0.421e25, 2.531e25]

        with warnings.catch_warnings(record=True) as w:
            # Always raise warning.
            warnings.simplefilter("always")
            with ImageComparison(self.path, 'mopad_fallback.png') as ic:
                beachball(mt, outfile=ic.name)

        # Make sure the appropriate warnings has been raised.
        self.assertTrue(w)
        # Filter
        w = [_i.message.args[0] for _i in w]
        w = [_i for _i in w
             if "falling back to the mopad wrapper" in _i.lower()]
        self.assertTrue(w)


def suite():
    return unittest.makeSuite(BeachballTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
