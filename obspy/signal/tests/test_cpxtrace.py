#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The cpxtrace.core test suite.
"""
import os
import unittest

import numpy as np
from scipy import signal

from obspy.signal import cpxtrace, util


# only tests for windowed data are implemented currently

class CpxTraceTestCase(unittest.TestCase):
    """
    Test cases for complex trace analysis
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
        self.data = np.loadtxt(f)
        f.close()
        # self.path = os.path.dirname(__file__)
        # self.res = np.loadtxt("3cssan.hy.1.MBGA_Z")
        # data = np.loadtxt("MBGA_Z.ASC")
        self.n = 256
        self.fs = 75
        self.smoothie = 3
        self.fk = [2, 1, 0, -1, -2]
        self.inc = int(0.05 * self.fs)
        # [0] Time (k*inc)
        # [1] A_norm
        # [2] dA_norm
        # [3] dAsum
        # [4] dA2sum
        # [5] ct
        # [6] dct
        # [7] omega
        # [8] domega
        # [9] sigma
        # [10] dsigma
        # [11] log_cepstrum
        # [12] log_cepstrum
        # [13] log_cepstrum
        # [14] dperiod
        # [15] ddperiod
        # [16] bandwidth
        # [17] dbwith
        # [18] cfreq
        # [19] dcfreq
        # [20] hob1
        # [21] hob2
        # [22] hob3
        # [23] hob4
        # [24] hob5
        # [25] hob6
        # [26] hob7
        # [27] hob8
        # [28] phi12
        # [29] dphi12
        # [30] phi13
        # [31] dphi13
        # [32] phi23
        # [33] dphi23
        # [34] lv_h1
        # [35] lv_h2
        # [36] lv_h3
        # [37] dlv_h1
        # [38] dlv_h2
        # [39] dlv_h3
        # [40] rect
        # [41] drect
        # [42] plan
        # [43] dplan
        self.data_win, self.nwin, self.no_win = \
            util.enframe(self.data, signal.hamming(self.n), self.inc)
        # self.data_win = data

    def tearDown(self):
        pass

    def test_normenvelope(self):
        """
        """
        # A_cpx,A_real = cpxtrace.envelope(self.data_win)
        anorm = cpxtrace.normalized_envelope(self.data_win, self.fs,
                                             self.smoothie, self.fk)
        rms = np.sqrt(np.sum((anorm[0] - self.res[:, 1]) ** 2) /
                      np.sum(self.res[:, 1] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((anorm[1] - self.res[:, 2]) ** 2) /
                      np.sum(self.res[:, 2] ** 2))
        self.assertEqual(rms < 1.0e-5, True)

    def test_centroid(self):
        """
        """
        centroid = cpxtrace.centroid(self.data_win, self.fk)
        rms = np.sqrt(np.sum((centroid[0] - self.res[:, 5]) ** 2) /
                      np.sum(self.res[:, 5] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((centroid[1] - self.res[:, 6]) ** 2) /
                      np.sum(self.res[:, 6] ** 2))
        self.assertEqual(rms < 1.0e-5, True)

    def test_inst_freq(self):
        """
        """
        omega = cpxtrace.instantaneous_frequency(self.data_win, self.fs,
                                                 self.fk)
        rms = np.sqrt(np.sum((omega[0] - self.res[:, 7]) ** 2) /
                      np.sum(self.res[:, 7] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((omega[1] - self.res[:, 8]) ** 2) /
                      np.sum(self.res[:, 8] ** 2))
        self.assertEqual(rms < 1.0e-5, True)

    def test_inst_bwith(self):
        """
        """
        sigma = cpxtrace.instantaneous_bandwidth(self.data_win, self.fs,
                                                 self.fk)
        rms = np.sqrt(np.sum((sigma[0] - self.res[:, 9]) ** 2) /
                      np.sum(self.res[:, 9] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((sigma[1] - self.res[:, 10]) ** 2) /
                      np.sum(self.res[:, 10] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
