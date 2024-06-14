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
from obspy.signal.transmatrix import transMatrix
from obspy.core.util.misc import TemporaryWorkingDirectory


class transmatrixTestCase(unittest.TestCase):
    """
    Transmatrix test case
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 'data'))

    def test_transmatrix_kown_vs_unknown(self):
        """
        Test transformation matrix of unknown instrument vs known one in the
        same time range. Filter is set between 0.125 and 0.5Hz
        """
        st1 = read(os.path.join(self.path, 'unknown_sensor.mseed'))
        st2 = read(os.path.join(self.path, 'known_sensor.mseed'))
        ref_matrix = np.array([[2.5112e-01, -1.2244e-02, -2.7012e-04],
                               [1.1542e-02, 2.5021e-01, -6.4375e-04],
                               [3.7274e-04, 1.7176e-04, 2.5122e-01]])
        ref_angle = np.arctan2(ref_matrix[1][0], ref_matrix[0][0])

        with TemporaryWorkingDirectory():
            matrix = transMatrix(st1, st2, fmin=0.125, fmax=0.5)
            angle = np.arctan2(-1*matrix[0][1], matrix[1][1])

        # test if matrix and orientation error matche the reference values
        np.testing.assert_array_almost_equal(matrix, ref_matrix, decimal=3)
        np.testing.assert_almost_equal(angle, ref_angle, decimal=2)


def suite():
    return unittest.makeSuite(transmatrixTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
