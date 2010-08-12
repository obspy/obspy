#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Array test suite.
"""

import unittest
import numpy as np
from obspy.signal.array import array_rotation_strain

class ArrayTestCase(unittest.TestCase):
    """
    Test cases for Array functions.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_array(self):
        #    REL OFFSET (M)
        #    x       y      z     
        array_coords = np.array([[   0.0,    0.0,   0.0],
                                 [  -5.0,    7.0,   0.0],
                                 [   5.0,    7.0,   0.0],
                                 [  10.0,    0.0,   0.0],
                                 [   5.0,   -7.0,   0.0],
                                 [  -5.0,   -7.0,   0.0],
                                 [ -10.0,    0.0,   0.0]])
        subarray = np.array([0, 1, 2, 3, 4, 5, 6])
        sigmau = 0.0001
        Vp = 1.93  # km/s
        Vs = 0.326 # km/s

        rotx = 0.00001 * np.exp(-1*np.square(np.linspace(-2, 2, 1000))) * \
                np.sin(np.linspace(-30*np.pi, 30*np.pi, 1000))
        roty = 0.00001 * np.exp(-1*np.square(np.linspace(-2, 2, 1000))) * \
                np.sin(np.linspace(-20*np.pi, 20*np.pi, 1000))
        rotz = 0.00001 * np.exp(-1*np.square(np.linspace(-2, 2, 1000))) * \
                np.sin(np.linspace(-10*np.pi, 10*np.pi, 1000))

        ts1 = np.ones((1000, 7)) * np.NaN
        ts2 = np.ones((1000, 7)) * np.NaN
        ts3 = np.ones((1000, 7)) * np.NaN

        for stat in np.arange(7):
            for t in np.arange(1000):
                ts1[t, stat] = -1. * array_coords[stat, 1] * rotz[t]
                ts2[t, stat] = array_coords[stat, 0] * rotz[t]
                ts3[t, stat] = array_coords[stat, 1] * rotx[t] - \
                        array_coords[stat, 0] * roty[t] 

        out = array_rotation_strain(subarray, ts1, ts2, ts3, Vp, Vs,
                                    array_coords, sigmau)

        np.testing.assert_array_almost_equal(rotx, out['ts_w1'])
        np.testing.assert_array_almost_equal(roty, out['ts_w2'])
        np.testing.assert_array_almost_equal(rotz, out['ts_w3'])


def suite():
    return unittest.makeSuite(ArrayTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
