#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The cpxtrace.core test suite.
"""

from obspy.signal import cpxtrace, util
from scipy import signal
import inspect
import numpy as N
import os
import unittest


# only tests for windowed data are implemented currently

class CpxTraceTestCase(unittest.TestCase):
    """
    Test cases for complex trace analysis
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'data')
        file = os.path.join(self.path, '3cssan.hy.1.MBGA_Z')
        f = open(file)
        self.res = N.loadtxt(f)
        f.close()
        file = os.path.join(self.path, 'MBGA_Z.ASC')
        f = open(file)
        data = N.loadtxt(f)
        f.close()
        #self.path = os.path.dirname(inspect.getsourcefile(self.__class__))
        #self.res = N.loadtxt("3cssan.hy.1.MBGA_Z")
        #data = N.loadtxt("MBGA_Z.ASC")
        self.n = 256
        self.fs = 75
        self.smoothie = 3
        self.fk = [2, 1, 0, -1, -2]
        self.inc = int(0.05 * self.fs)
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
        self.data_win, self.nwin, self.no_win = \
            util.enframe(data, signal.hamming(self.n), self.inc)
        #self.data_win = data

    def tearDown(self):
        pass

    def test_normenvelope(self):
        """
        Read files via L{obspy.Trace}
        """
        #A_cpx,A_real = cpxtrace.envelope(self.data_win)
        Anorm = cpxtrace.normEnvelope(self.data_win, self.fs, self.smoothie,
                                      self.fk)
        rms = N.sqrt(N.sum((Anorm[0] - self.res[:, 1]) ** 2) /
                     N.sum(self.res[:, 1] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = N.sqrt(N.sum((Anorm[1] - self.res[:, 2]) ** 2) /
                     N.sum(self.res[:, 2] ** 2))
        self.assertEqual(rms < 1.0e-5, True)

    def test_centroid(self):
        """
        Read files via L{obspy.Trace}
        """
        centroid = cpxtrace.centroid(self.data_win, self.fk)
        rms = N.sqrt(N.sum((centroid[0] - self.res[:, 5]) ** 2) /
                     N.sum(self.res[:, 5] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = N.sqrt(N.sum((centroid[1] - self.res[:, 6]) ** 2) /
                     N.sum(self.res[:, 6] ** 2))
        self.assertEqual(rms < 1.0e-5, True)

    def test_instFreq(self):
        """
        Read files via L{obspy.Trace}
        """
        omega = cpxtrace.instFreq(self.data_win, self.fs, self.fk)
        rms = N.sqrt(N.sum((omega[0] - self.res[:, 7]) ** 2) /
                     N.sum(self.res[:, 7] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = N.sqrt(N.sum((omega[1] - self.res[:, 8]) ** 2) /
                     N.sum(self.res[:, 8] ** 2))
        self.assertEqual(rms < 1.0e-5, True)

    def test_instBwith(self):
        """
        Read files via L{obspy.Trace}
        """
        sigma = cpxtrace.instBwith(self.data_win, self.fs, self.fk)
        rms = N.sqrt(N.sum((sigma[0] - self.res[:, 9]) ** 2) /
                     N.sum(self.res[:, 9] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = N.sqrt(N.sum((sigma[1] - self.res[:, 10]) ** 2) /
                     N.sum(self.res[:, 10] ** 2))
        self.assertEqual(rms < 1.0e-5, True)


def suite():
    return unittest.makeSuite(CpxTraceTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
