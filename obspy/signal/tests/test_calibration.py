#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The calibration test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import numpy as np

from obspy import read
from obspy.signal.calibration import rel_calib_stack
from obspy.core.util.misc import TemporaryWorkingDirectory


class CalibrationTestCase(unittest.TestCase):
    """
    Calibration test case
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 'data'))

    def test_relcal_sts2_vs_unknown(self):
        """
        Test relative calibration of unknown instrument vs STS2 in the same
        time range. Window length is set to 20 s, smoothing rate to 10.
        """
        st1 = read(os.path.join(self.path, 'ref_STS2'))
        st2 = read(os.path.join(self.path, 'ref_unknown'))
        calfile = os.path.join(self.path, 'STS2_simp.cal')

        with TemporaryWorkingDirectory():
            freq, amp, phase = rel_calib_stack(st1, st2, calfile, 20,
                                               smooth=10, save_data=True)
            self.assertTrue(os.path.isfile("0438.EHZ.20.resp"))
            self.assertTrue(os.path.isfile("STS2.refResp"))
            freq_, amp_, phase_ = np.loadtxt("0438.EHZ.20.resp", unpack=True)
            self.assertTrue(np.allclose(freq, freq_, rtol=1e-8, atol=1e-8))
            self.assertTrue(np.allclose(amp, amp_, rtol=1e-8, atol=1e-8))
            self.assertTrue(np.allclose(phase, phase_, rtol=1e-8, atol=1e-8))

        # read in the reference responses
        un_resp = np.loadtxt(os.path.join(self.path, 'unknown.resp'))
        kn_resp = np.loadtxt(os.path.join(self.path, 'STS2.refResp'))

        # bug resolved with 2f9876d, arctan was used which maps to
        # [-pi/2, pi/2]. arctan2 or np.angle shall be used instead
        # correct the test data by hand
        un_resp[:, 2] = np.unwrap(un_resp[:, 2] * 2) / 2
        if False:
            import matplotlib.pyplot as plt
            plt.plot(freq, un_resp[:, 2], 'b', label='reference', alpha=.8)
            plt.plot(freq, phase, 'r', label='new', alpha=.8)
            plt.xlim(-10, None)
            plt.legend()
            plt.show()

        # test if freq, amp and phase match the reference values
        np.testing.assert_array_almost_equal(freq, un_resp[:, 0],
                                             decimal=4)
        np.testing.assert_array_almost_equal(freq, kn_resp[:, 0],
                                             decimal=4)
        np.testing.assert_array_almost_equal(amp, un_resp[:, 1],
                                             decimal=4)
        # TODO: unknown why the first frequency mismatches so much
        np.testing.assert_array_almost_equal(phase[1:], un_resp[1:, 2],
                                             decimal=4)

    def test_relcalUsingTraces(self):
        """
        Tests using traces instead of stream objects as input parameters.
        """
        st1 = read(os.path.join(self.path, 'ref_STS2'))
        st2 = read(os.path.join(self.path, 'ref_unknown'))
        calfile = os.path.join(self.path, 'STS2_simp.cal')
        # stream
        freq, amp, phase = rel_calib_stack(st1, st2, calfile, 20, smooth=10,
                                           save_data=False)
        # traces
        freq2, amp2, phase2 = rel_calib_stack(st1[0], st2[0], calfile, 20,
                                              smooth=10, save_data=False)
        np.testing.assert_array_almost_equal(freq, freq2, decimal=4)
        np.testing.assert_array_almost_equal(amp, amp2, decimal=4)
        np.testing.assert_array_almost_equal(phase, phase2, decimal=4)


def suite():
    return unittest.makeSuite(CalibrationTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
