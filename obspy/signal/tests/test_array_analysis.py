#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The array_analysis test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest
import numpy as np

from obspy.signal.array_analysis import SeismicArray
from obspy.core import inventory
from obspy import read


class ArrayTestCase(unittest.TestCase):
    """
    Test cases for array_analysis functions.
    """
    def setUp(self):
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 'data'))
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
        self.pfield = SeismicArray('pfield', inventory=inventory.read_inventory(
                os.path.join(self.path, 'pfield_inv_for_instaseis.xml'),
                format='stationxml'))
        self.vel = read(os.path.join(self.path, 'pfield_instaseis.mseed'))

    def test_array_rotation(self):
        # tests function array_rotation_strain with synthetic data with pure
        # rotation and no strain

        rotx = 0.00001 * np.exp(-1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-30 * np.pi, 30 * np.pi, 1000))
        roty = 0.00001 * np.exp(-1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-20 * np.pi, 20 * np.pi, 1000))
        rotz = 0.00001 * np.exp(-1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-10 * np.pi, 10 * np.pi, 1000))

        for stat in range(7):
            for t in range(1000):
                self.ts1[t, stat] = -1. * self.array_coords[stat, 1] * rotz[t]
                self.ts2[t, stat] = self.array_coords[stat, 0] * rotz[t]
                self.ts3[t, stat] = self.array_coords[stat, 1] * rotx[t] - \
                    self.array_coords[stat, 0] * roty[t]

        out = SeismicArray.array_rotation_strain(self.subarray, self.ts1,
                                                 self.ts2, self.ts3, self.Vp,
                                                 self.Vs, self.array_coords,
                                                 self.sigmau)

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
        # tests function array_rotation_strain with synthetic data with pure
        # dilation and no rotation or shear strain
        eta = 1 - 2 * self.Vs ** 2 / self.Vp ** 2

        dilation = .00001 * np.exp(
            -1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-40 * np.pi, 40 * np.pi, 1000))

        for stat in range(7):
            for t in range(1000):
                self.ts1[t, stat] = self.array_coords[stat, 0] * dilation[t]
                self.ts2[t, stat] = self.array_coords[stat, 1] * dilation[t]
                self.ts3[t, stat] = self.array_coords[stat, 2] * dilation[t]

        out = SeismicArray.array_rotation_strain(self.subarray, self.ts1,
                                                 self.ts2, self.ts3, self.Vp,
                                                 self.Vs, self.array_coords,
                                                 self.sigmau)

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
        # tests function array_rotation_strain with synthetic data with pure
        # horizontal shear strain, no rotation or dilation
        shear_strainh = .00001 * np.exp(
            -1 * np.square(np.linspace(-2, 2, 1000))) * \
            np.sin(np.linspace(-10 * np.pi, 10 * np.pi, 1000))

        ts3 = np.zeros((1000, 7))

        for stat in range(7):
            for t in range(1000):
                self.ts1[t, stat] = self.array_coords[stat, 1] * \
                    shear_strainh[t]
                self.ts2[t, stat] = self.array_coords[stat, 0] * \
                    shear_strainh[t]

        out = SeismicArray.array_rotation_strain(self.subarray, self.ts1,
                                                 self.ts2, ts3, self.Vp,
                                                 self.Vs, self.array_coords,
                                                 self.sigmau)

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

    def test_three_component_beamforming(self):
        """
        Integration test for 3cbf. Parameter values are fairly arbitrary.
        """
        out = self.pfield.three_component_beamforming(self.vel.select(
            channel='BXN'), self.vel.select(channel='BXE'),
            self.vel.select(channel='BXZ'), 64, 0, 0.6, 0.03, wavetype='P',
            freq_range=[0.1, .3], whiten=True, coherency=False)
        self.assertEqual(out.max_pow_baz, 246)
        self.assertEqual(out.max_pow_slow, 0.3)
        np.testing.assert_array_almost_equal(out.max_rel_power, 1.22923997,
                                             decimal=8)

    def test_fk_analysis(self):
        """
        Integration test for FK-analysis.
        """
        out = self.pfield.fk_analysis(self.vel.select(channel='BXZ'), 0.1, 0.3,
                                      wlen=64, wfrac=0.2, slx=(-0.5, 0.5),
                                      sly=(-0.5, 0.5), sls=0.03)
        self.assertEqual(list(out.max_pow_baz), [250.34617594194668,
                         250.34617594194668, 252.34987578006991,
                                                 254.35775354279127])
        self.assertEqual(list(out.max_pow_slow), [0.29732137494637012,
                                                  0.29732137494637012,
                                                  0.23086792761230387,
                                                  0.25961509971494334])
        self.assertEqual(list(out.max_abs_power), [14422.833620665502,
                                                   13015.583243478792,
                                                   93.883822211713934,
                                                   1.9298134533374038])
        self.assertEqual(list(out.max_rel_power), [0.96002866762132411,
                                                   0.96293146139421071,
                                                   0.93713525165346112,
                                                   0.95433426076472017])


def suite():
    return unittest.makeSuite(ArrayTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
