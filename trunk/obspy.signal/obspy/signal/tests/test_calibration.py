#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The calibration test suite.
"""

import os
import unittest
import numpy as np
from obspy.core import read
from obspy.signal.calibration import relcalstack

class CalibrationTestCase(unittest.TestCase):
    """
    Calibration test case
    """
    def test_relcal_sts2_vs_unknown(self):
        """
        Test relative calibration of unknow instrument vs STS2 in the same time
        range. Window length is set to 20 s, smoothing rate to 10.
        """

        cwd = os.getcwd()
        st1 = read('%s/data/ref_STS2'%(cwd))
        st2 = read('%s/data/ref_unknown'%(cwd))
        calfile = "%s/data/STS2_simp.cal"%(cwd)
        
        freq, amp, phase = relcalstack(st1, st2, calfile, 20, smooth=10)

        # read in the reference responses
        un_resp = np.loadtxt("data/unknown.resp")
        kn_resp = np.loadtxt("data/STS2.refResp")

        # test if freq, amp and phase match the reference values
        np.testing.assert_array_almost_equal(freq, un_resp[:,0],
                                             decimal=4)
        np.testing.assert_array_almost_equal(freq, kn_resp[:,0],
                                             decimal=4)
        np.testing.assert_array_almost_equal(amp, un_resp[:,1],
                                             decimal=4)
        np.testing.assert_array_almost_equal(phase, un_resp[:,2],
                                             decimal=4)

        os.remove("0438.20.resp")
        os.remove("STS2.refResp")

def suite():
    return unittest.makeSuite(RelcalTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')

