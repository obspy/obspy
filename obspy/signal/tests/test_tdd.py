# -*- coding: utf-8 -*-
"""
The obspy.signal.tdd test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import numpy as np

from obspy.core.util import AttribDict
from obspy.signal.tdd import deconvolve_volt_to_velocity


class TDDTestCase(unittest.TestCase):
    """
    Test cases for obspy.tdd
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_deconvolve_volt_to_velocity(self):
        """
        Test case for ctypes version of recursive_STALTA
        """
        # guralp CMG-3T, sampling interval 0.01s
        dpz = AttribDict()
        dpz.poles = [-0.037+0.037j, -0.037-0.037j, -503.000+0j, -1010.000+0j,
                     -1130.000+0j]
        dpz.zeros = [0j, 0j]
        dpz.norm = 571507692
        dpz.sensitivity = 1500
        dpz.delta = 0.01
        dpz.zpg = AttribDict()
        dpz.zpg.zero = [1+0j, 1+0j]
        dpz.zpg.pole = [0.999628385-0.0003683j, 0.999628385+0.0003683j,
                        0.049564709+0j, 0.003482788-0j, 0.001962318+0j]
        dpz.zpg.gain = 1408.607
        dpz.zpg.fmax = 25.61493
        dpz.zpg.method = "FD"

        raw_in_volts = np.load(
            os.path.join(self.path, "tdd_raw_in_volts.npy"))
        deconvolved_expected = np.load(
            os.path.join(self.path, "tdd_deconvolved.npy"))
        deconvolved_got = deconvolve_volt_to_velocity(
            raw_in_volts, dpz, filter_low=0.05, filter_high=None,
            bitweight=None, dec=None, demean=True)
        # we need to add absolute tolerance, otherwise we get a fail because of
        # a zero crossing of the expected values (expected value array is
        # -1e-10 at index 11480 in the array). standard deviation of expected
        # array is 7.9e-07, so 1e-9 is really small.
        self.assertTrue(np.allclose(
            deconvolved_got, deconvolved_expected, atol=1e-9, rtol=1e-4))


def suite():
    return unittest.makeSuite(TDDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
