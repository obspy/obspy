#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import numpy as np

from obspy.signal.array_analysis import (array_rotation_strain, get_geometry,
                                         get_timeshift)


class ArrayTestCase(unittest.TestCase):
    """
    Test cases for array_analysis functions.
    """
    def setUp(self):
        self.array_coords = np.array([[0.0, 0.0, 0.0],
                                      [-5.0, 7.0, 0.0],
                                      [5.0, 7.0, 0.0],
                                      [10.0, 0.0, 0.0],
                                      [5.0, -7.0, 0.0],
                                      [-5.0, -7.0, 0.0],
                                      [-10.0, 0.0, 0.0]])
        self.subarray = np.array([0, 1, 2, 3, 4, 5, 6])
        self.ts1 = np.empty((1000, 7))
        self.ts2 = np.empty((1000, 7))
        self.ts3 = np.empty((1000, 7))
        self.ts1.fill(np.NaN)
        self.ts2.fill(np.NaN)
        self.ts3.fill(np.NaN)
        self.sigmau = 0.0001
        self.Vp = 1.93
        self.Vs = 0.326

    def tearDown(self):
        pass

    def test_array_rotation(self):
        """
        Tests function array_rotation_strain with synthetic data with pure
        rotation and no strain
        """
        array_coords = self.array_coords
        subarray = self.subarray
        ts1 = self.ts1
        ts2 = self.ts2
        ts3 = self.ts3
        sigmau = self.sigmau
        vp = self.Vp
        vs = self.Vs

        rotx = 0.00001 * np.exp(-1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-30 * np.pi, 30 * np.pi, 1000))
        roty = 0.00001 * np.exp(-1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-20 * np.pi, 20 * np.pi, 1000))
        rotz = 0.00001 * np.exp(-1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-10 * np.pi, 10 * np.pi, 1000))

        for stat in range(7):
            for t in range(1000):
                ts1[t, stat] = -1. * array_coords[stat, 1] * rotz[t]
                ts2[t, stat] = array_coords[stat, 0] * rotz[t]
                ts3[t, stat] = array_coords[stat, 1] * rotx[t] - \
                    array_coords[stat, 0] * roty[t]

        out = array_rotation_strain(subarray, ts1, ts2, ts3, vp, vs,
                                    array_coords, sigmau)

        np.testing.assert_array_almost_equal(rotx, out['ts_w1'], decimal=12)
        np.testing.assert_array_almost_equal(roty, out['ts_w2'], decimal=12)
        np.testing.assert_array_almost_equal(rotz, out['ts_w3'], decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_s'],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_d'],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_m'],
                                             decimal=12)

    def test_array_dilation(self):
        """
        Tests function array_rotation_strain with synthetic data with pure
        dilation and no rotation or shear strain
        """
        array_coords = self.array_coords
        subarray = self.subarray
        ts1 = self.ts1
        ts2 = self.ts2
        ts3 = self.ts3
        sigmau = self.sigmau
        vp = self.Vp
        vs = self.Vs

        eta = 1 - 2 * vs ** 2 / vp ** 2

        dilation = .00001 * np.exp(
            -1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-40 * np.pi, 40 * np.pi, 1000))

        for stat in range(7):
            for t in range(1000):
                ts1[t, stat] = array_coords[stat, 0] * dilation[t]
                ts2[t, stat] = array_coords[stat, 1] * dilation[t]
                ts3[t, stat] = array_coords[stat, 2] * dilation[t]

        out = array_rotation_strain(subarray, ts1, ts2, ts3, vp, vs,
                                    array_coords, sigmau)

        # remember free surface boundary conditions!
        # see Spudich et al, 1995, (A2)
        np.testing.assert_array_almost_equal(dilation * (2 - 2 * eta),
                                             out['ts_d'], decimal=12)
        np.testing.assert_array_almost_equal(dilation * 2, out['ts_dh'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(
            abs(dilation * .5 * (1 + 2 * eta)), out['ts_s'], decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_sh'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_w1'],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_w2'],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_w3'],
                                             decimal=15)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_m'],
                                             decimal=12)

    def test_array_horizontal_shear(self):
        """
        Tests function array_rotation_strain with synthetic data with pure
        horizontal shear strain, no rotation or dilation.
        """
        array_coords = self.array_coords
        subarray = self.subarray
        ts1 = self.ts1
        ts2 = self.ts2
        sigmau = self.sigmau
        vp = self.Vp
        vs = self.Vs

        shear_strainh = .00001 * np.exp(
            -1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-10 * np.pi, 10 * np.pi, 1000))

        ts3 = np.zeros((1000, 7))

        for stat in range(7):
            for t in range(1000):
                ts1[t, stat] = array_coords[stat, 1] * shear_strainh[t]
                ts2[t, stat] = array_coords[stat, 0] * shear_strainh[t]

        out = array_rotation_strain(subarray, ts1, ts2, ts3, vp, vs,
                                    array_coords, sigmau)

        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_d'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_dh'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(abs(shear_strainh), out['ts_s'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(abs(shear_strainh), out['ts_sh'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_w1'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_w2'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_w3'],
                                             decimal=12)
        np.testing.assert_array_almost_equal(np.zeros(1000), out['ts_m'],
                                             decimal=12)

    def test_get_geometry(self):
        """
        Test get_geometry() in array_analysis.py
        """
        ll = np.array([[24.5797167, 121.4842444, 385.106],
                       [24.5797611, 121.4842333, 384.893],
                       [24.5796694, 121.4842556, 385.106]])

        la = get_geometry(ll)

        np.testing.assert_almost_equal(la[:, 0].sum(), 0., decimal=8)
        np.testing.assert_almost_equal(la[:, 1].sum(), 0., decimal=8)
        np.testing.assert_almost_equal(la[:, 2].sum(), 0., decimal=8)

        ll = np.array([[10., 10., 10.],
                       [0., 5., 5.],
                       [0., 0., 0.]])

        la = get_geometry(ll, coordsys='xy')

        np.testing.assert_almost_equal(la[:, 0].sum(), 0., decimal=8)
        np.testing.assert_almost_equal(la[:, 1].sum(), 0., decimal=8)
        np.testing.assert_almost_equal(la[:, 2].sum(), 0., decimal=8)

    def test_get_timeshift(self):
        """
        Tests the get_timeshift function.
        """
        geometry = np.array(
            [[-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 1, 0], [2, 2, 0]])

        t = get_timeshift(geometry=geometry, sll_x=1, sll_y=1, sl_s=2,
                          grdpts_x=2, grdpts_y=2)

        np.testing.assert_allclose(t, np.array([
            # (x_s, y_s) = 1, 1;  1, 3; 3, 1; 3, 3
            #
            # The timeshift is not a geometric distance but sums up x + y
            # axis.
            #
            # Station at index 0.
            [[-2, -2], [-6, -6]],
            # Station at index 1.
            [[-1, -1], [-3, -3]],
            # Station at index 2.
            [[0, 0], [0, 0]],
            # Station at index 3.
            [[2, 4], [4, 6]],
            # Station at index 4.
            [[4, 8], [8, 12]]
        ]), rtol=1E-5)


def suite():
    return unittest.makeSuite(ArrayTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
