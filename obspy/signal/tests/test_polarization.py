#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The polarization.core test suite.
"""

from obspy.signal import polarization, util
from scipy import signal
import numpy as np
import os
import unittest


# only tests for windowed data are implemented currently

class PolarizationTestCase(unittest.TestCase):
    """
    Test cases for polarization analysis
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        file = os.path.join(self.path, '3cssan.hy.1.MBGA_Z')
        f = open(file)
        self.res = np.loadtxt(f)
        f.close()
        file = os.path.join(self.path, 'MBGA_Z.ASC')
        f = open(file)
        data_z = np.loadtxt(f)
        f.close()
        file = os.path.join(self.path, 'MBGA_E.ASC')
        f = open(file)
        data_e = np.loadtxt(f)
        f.close()
        file = os.path.join(self.path, 'MBGA_N.ASC')
        f = open(file)
        data_n = np.loadtxt(f)
        f.close()
        #self.path = os.path.dirname(__file__)
        #self.res = N.loadtxt("3cssan.hy.1.MBGA_Z")
        #data = N.loadtxt("MBGA_Z.ASC")
        self.n = 256
        self.fs = 75
        self.smoothie = 3
        self.fk = [2, 1, 0, -1, -2]
        self.inc = int(0.05 * self.fs)
        self.norm = pow(np.max(data_z), 2)
        #[0] Time (k*inc)
        #[1] A_norm
        #[2] dA_norm
        #[3] dAsum
        #[4] dA2sum
        #[5] ct
        #[6] dct
        #[7] omega
        #[8] domega
        #[9] sigma
        #[10] dsigma
        #[11] logcep
        #[12] logcep
        #[13] logcep
        #[14] dperiod
        #[15] ddperiod
        #[16] bwith
        #[17] dbwith
        #[18] cfreq
        #[19] dcfreq
        #[20] hob1
        #[21] hob2
        #[22] hob3
        #[23] hob4
        #[24] hob5
        #[25] hob6
        #[26] hob7
        #[27] hob8
        #[28] phi12
        #[29] dphi12
        #[30] phi13
        #[31] dphi13
        #[32] phi23
        #[33] dphi23
        #[34] lv_h1
        #[35] lv_h2
        #[36] lv_h3
        #[37] dlv_h1
        #[38] dlv_h2
        #[39] dlv_h3
        #[40] rect
        #[41] drect
        #[42] plan
        #[43] dplan
        self.data_win_z, self.nwin, self.no_win = \
            util.enframe(data_z, signal.hamming(self.n), self.inc)
        self.data_win_e, self.nwin, self.no_win = \
            util.enframe(data_e, signal.hamming(self.n), self.inc)
        self.data_win_n, self.nwin, self.no_win = \
            util.enframe(data_n, signal.hamming(self.n), self.inc)

    def tearDown(self):
        pass

    def test_polarization(self):
        """
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


def suite():
    return unittest.makeSuite(PolarizationTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
