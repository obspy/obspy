#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The hoctavbands.core test suite.
"""
import os

import numpy as np
from scipy import signal

from obspy.signal import hoctavbands, util


# only tests for windowed data are implemented currently

class TestHoctavbands():
    """
    Test cases for half octav bands
    """
    @classmethod
    def setup_class(cls):
        # directory where the test files are located
        cls.path = os.path.join(os.path.dirname(__file__), 'data')
        file = os.path.join(cls.path, '3cssan.hy.1.MBGA_Z')
        f = open(file)
        cls.res = np.loadtxt(f)
        f.close()
        file = os.path.join(cls.path, 'MBGA_Z.ASC')
        f = open(file)
        data = np.loadtxt(f)
        f.close()
        # cls.path = os.path.dirname(__file__)
        # cls.res = np.loadtxt("3cssan.hy.1.MBGA_Z")
        # data = np.loadtxt("MBGA_Z.ASC")
        cls.n = 256
        cls.fs = 75
        cls.smoothie = 3
        cls.fk = [2, 1, 0, -1, -2]
        cls.inc = int(0.05 * cls.fs)
        cls.fc1 = 0.68
        cls.nofb = 8
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
        cls.data_win, cls.nwin, cls.no_win = \
            util.enframe(data, signal.hamming(cls.n), cls.inc)

    def test_hoctavbands(self):
        """
        """
        hob = hoctavbands.sonogram(self.data_win, self.fs, self.fc1,
                                   self.nofb, self.no_win)
        rms = np.sqrt(np.sum((hob[:, 0] - self.res[:, 20]) ** 2) /
                      np.sum(self.res[:, 20] ** 2))
        assert rms < 1.0e-5
        rms = np.sqrt(np.sum((hob[:, 1] - self.res[:, 21]) ** 2) /
                      np.sum(self.res[:, 21] ** 2))
        assert rms < 1.0e-5
        rms = np.sqrt(np.sum((hob[:, 2] - self.res[:, 22]) ** 2) /
                      np.sum(self.res[:, 22] ** 2))
        assert rms < 1.0e-5
        rms = np.sqrt(np.sum((hob[:, 3] - self.res[:, 23]) ** 2) /
                      np.sum(self.res[:, 23] ** 2))
        assert rms < 1.0e-5
        rms = np.sqrt(np.sum((hob[:, 4] - self.res[:, 24]) ** 2) /
                      np.sum(self.res[:, 24] ** 2))
        assert rms < 1.0e-5
        rms = np.sqrt(np.sum((hob[:, 5] - self.res[:, 25]) ** 2) /
                      np.sum(self.res[:, 25] ** 2))
        assert rms < 1.0e-5
        rms = np.sqrt(np.sum((hob[:, 6] - self.res[:, 26]) ** 2) /
                      np.sum(self.res[:, 26] ** 2))
        assert rms < 1.0e-5
        rms = np.sqrt(np.sum((hob[:, 7] - self.res[:, 27]) ** 2) /
                      np.sum(self.res[:, 27] ** 2))
        assert rms < 1.0e-5
