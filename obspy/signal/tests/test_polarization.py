#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The polarization.core test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.signal import polarization, util
from scipy import signal
import numpy as np
from os.path import join, dirname
import unittest


class PolarizationTestCase(unittest.TestCase):
    """
    Test cases for polarization analysis
    """
    def setUp(self):
        path = join(dirname(__file__), 'data')
        # setting up sliding window data
        data_z = np.loadtxt(join(path, 'MBGA_Z.ASC'))
        data_e = np.loadtxt(join(path, 'MBGA_E.ASC'))
        data_n = np.loadtxt(join(path, 'MBGA_N.ASC'))
        n = 256
        fs = 75
        inc = int(0.05 * fs)
        self.data_win_z, self.nwin, self.no_win = \
            util.enframe(data_z, signal.hamming(n), inc)
        self.data_win_e, self.nwin, self.no_win = \
            util.enframe(data_e, signal.hamming(n), inc)
        self.data_win_n, self.nwin, self.no_win = \
            util.enframe(data_n, signal.hamming(n), inc)
        # global test input
        self.fk = [2, 1, 0, -1, -2]
        self.norm = pow(np.max(data_z), 2)
        self.res = np.loadtxt(join(path, '3cssan.hy.1.MBGA_Z'))

    def tearDown(self):
        pass

    def test_polarization(self):
        """
        windowed data
        """
        pol = polarization.eigval(self.data_win_e, self.data_win_n,
                                  self.data_win_z, self.fk, self.norm)
        rms = np.sqrt(np.sum((pol[0] - self.res[:, 34]) ** 2) /
                      np.sum(self.res[:, 34] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[1] - self.res[:, 35]) ** 2) /
                      np.sum(self.res[:, 35] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[2] - self.res[:, 36]) ** 2) /
                      np.sum(self.res[:, 36] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[3] - self.res[:, 40]) ** 2) /
                      np.sum(self.res[:, 40] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[4] - self.res[:, 42]) ** 2) /
                      np.sum(self.res[:, 42] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[5][:, 0] - self.res[:, 37]) ** 2) /
                      np.sum(self.res[:, 37] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[5][:, 1] - self.res[:, 38]) ** 2) /
                      np.sum(self.res[:, 38] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[5][:, 2] - self.res[:, 39]) ** 2) /
                      np.sum(self.res[:, 39] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[6] - self.res[:, 41]) ** 2) /
                      np.sum(self.res[:, 41] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[7] - self.res[:, 43]) ** 2) /
                      np.sum(self.res[:, 43] ** 2))
        self.assertEqual(rms < 1.0e-5, True)

    def test_polarization1D(self):
        """
        1 dimenstional input --- regression test case for bug #919
        """
        pol = polarization.eigval(self.data_win_e[100, :],
                                  self.data_win_n[100, :],
                                  self.data_win_z[100, :],
                                  self.fk, self.norm)
        pol_5_ref = [2.81387533e-04, 3.18409580e-04, 6.74030846e-04,
                     5.55067015e-01, 4.32938188e-01]
        self.assertTrue(np.allclose(np.concatenate(pol[:5]), pol_5_ref))


def suite():
    return unittest.makeSuite(PolarizationTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
