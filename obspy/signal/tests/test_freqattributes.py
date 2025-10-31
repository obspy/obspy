#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The freqattributes.core test suite.
"""
from math import pi

import numpy as np
import pytest
from scipy import signal

from obspy.signal import freqattributes, util


# only tests for windowed data are implemented currently

class TestFreqTrace():
    """
    Test cases for frequency attributes
    """
    @pytest.fixture(scope="function", autouse=True)
    def setup_data(self, testdata):
        self.res = np.loadtxt(testdata['3cssan.hy.1.MBGA_Z'])
        self.data = np.loadtxt(testdata['MBGA_Z.ASC'])
        # self.path = os.path.dirname(__file__)
        # self.res = np.loadtxt("3cssan.hy.1.MBGA_Z")
        # data = np.loadtxt("MBGA_Z.ASC")
        self.n = 256
        self.fs = 75
        self.smoothie = 3
        self.fk = [2, 1, 0, -1, -2]
        self.inc = int(0.05 * self.fs)
        self.nc = 12
        self.p = np.floor(3 * np.log(self.fs))
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
            util.enframe(self.data, signal.windows.hamming(self.n), self.inc)
        self.data_win_bc, self.nwin_, self.no_win_ = \
            util.enframe(self.data, np.ones(self.n), self.inc)
        # self.data_win = data

    def test_cfrequency(self):
        """
        """
        cfreq = freqattributes.central_frequency(self.data_win_bc, self.fs,
                                                 self.smoothie, self.fk)
        rms = np.sqrt(np.sum((cfreq[0] - self.res[:, 18]) ** 2) /
                      np.sum(self.res[:, 18] ** 2))
        assert rms < 1.0e-5
        rms = np.sqrt(np.sum((cfreq[1] - self.res[:, 19]) ** 2) /
                      np.sum(self.res[:, 19] ** 2))
        assert rms < 1.0e-5

    def test_cfrequency_no_win(self):
        cfreq = freqattributes.central_frequency(self.data_win_bc[0], self.fs,
                                                 self.smoothie, self.fk)
        rms = (cfreq - self.res[0, 18]) / self.res[0, 18]
        assert rms < 1.0e-5

    def test_bwith(self):
        """
        """
        bwith = freqattributes.bandwidth(self.data_win, self.fs, self.smoothie,
                                         self.fk)
        rms = np.sqrt(np.sum((bwith[0] - self.res[:, 16]) ** 2) /
                      np.sum(self.res[:, 16] ** 2))
        assert rms < 1.0e-5
        rms = np.sqrt(np.sum((bwith[1] - self.res[:, 17]) ** 2) /
                      np.sum(self.res[:, 17] ** 2))
        assert rms < 1.0e-5

    def test_domper(self):
        """
        """
        dperiod = freqattributes.dominant_period(self.data_win, self.fs,
                                                 self.smoothie, self.fk)
        rms = np.sqrt(np.sum((dperiod[0] - self.res[:, 14]) ** 2) /
                      np.sum(self.res[:, 14] ** 2))
        assert rms < 1.0e-5
        rms = np.sqrt(np.sum((dperiod[1] - self.res[:, 15]) ** 2) /
                      np.sum(self.res[:, 15] ** 2))
        assert rms < 1.0e-5

    def test_logcep(self):
        """
        """
        cep = freqattributes.log_cepstrum(self.data_win, self.fs, self.nc,
                                          self.p, self.n, 'Hamming')
        rms = np.sqrt(np.sum((cep[0] - self.res[:, 11]) ** 2) /
                      np.sum(self.res[:, 11] ** 2))
        assert rms < 1.0e-5
        rms = np.sqrt(np.sum((cep[1] - self.res[:, 12]) ** 2) /
                      np.sum(self.res[:, 12] ** 2))
        assert rms < 1.0e-5
        rms = np.sqrt(np.sum((cep[2] - self.res[:, 13]) ** 2) /
                      np.sum(self.res[:, 13] ** 2))
        assert rms < 1.0e-5

    def test_pgm(self):
        """
        """
        # flat array of zeros
        data = np.zeros(100)
        pgm = freqattributes.peak_ground_motion(data, 1.0, 1.0)
        assert pgm == (0.0, 0.0, 0.0, 0.0)
        # spike in middle of signal
        data[50] = 1.0
        (pg, m_dis, m_vel, m_acc) = freqattributes.peak_ground_motion(
            data, 1.0, 1.0)
        assert round(abs(pg-0.537443503597), 6) == 0
        assert m_dis == 1.0
        assert m_vel == 0.5
        assert m_acc == 0.5
        # flat array with one circle of sin (degree)
        data = np.zeros(400)
        for i in range(360):
            data[i + 20] = np.sin(i * pi / 180)
        (pg, m_dis, m_vel, m_acc) = freqattributes.peak_ground_motion(
            data, 1.0, 1.0)
        assert round(abs(pg-0.00902065171505), 6) == 0
        assert m_dis == 1.0
        assert round(abs(m_vel-0.0174524064373), 6) == 0
        assert round(abs(m_acc-0.00872487417563), 6) == 0
