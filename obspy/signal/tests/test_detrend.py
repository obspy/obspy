#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for data detrending.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import numpy as np

from obspy.signal.detrend import polynomial, spline


class DetrendTestCase(unittest.TestCase):
    """
    Test cases for Rotate.
    """
    def test_polynomial_detrend(self):
        """
        Simple test removing polynomial detrends.
        """
        coeffs = [(1, 2, 3), (2, -4), (-3, 2, -5, 15), (-10, 20, -1, 2, 15)]
        data = np.linspace(-5, 5, 100)

        for c in coeffs:
            # Create data.
            d = np.polyval(c, data)
            original_ptp = np.ptp(d)
            # Detrend with polynomial of same order.
            detrended = polynomial(d, order=len(c) - 1)
            # Make sure the maximum amplitude is reduced by some orders of
            # magnitude. It should almost be reduced to zero as we detrend a
            # polynomial with a polynomial...
            self.assertTrue(np.ptp(detrended) * 1E10 < original_ptp)

    def test_spline_detrend(self):
        """
        Simple test for the spline detrending.
        """
        coeffs = [(1, 2, 3), (2, -4), (-3, 2, -5, 15), (-10, 20, -1, 2, 15)]
        data = np.linspace(-5, 5, 100)

        for c in coeffs:
            # Create data.
            d = np.polyval(c, data)
            original_ptp = np.ptp(d)
            # Detrend with a spline of the same order as the polynomial.
            # This should be very very similar.
            detrended = spline(d, order=len(c) - 1, dspline=10)
            # Not as good as for the polynomial detrending.
            self.assertTrue(np.ptp(detrended) * 1E4 < original_ptp)


def suite():
    return unittest.makeSuite(DetrendTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
