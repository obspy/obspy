#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for data detrending.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import numpy as np

import obspy
from obspy.core.util.testing import ImageComparison
from obspy.signal.detrend import polynomial, spline


class DetrendTestCase(unittest.TestCase):
    """
    Test cases for the detrend methods.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        self.path_images = os.path.join(os.path.dirname(__file__), 'images')

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
            self.assertLess(np.ptp(detrended) * 1E10, original_ptp)

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
            self.assertLess(np.ptp(detrended) * 1E4, original_ptp)

    def test_polynomial_detrend_plotting(self):
        """
        Tests the plotting of the polynomial detrend operation.
        """
        tr = obspy.read()[0].filter("highpass", freq=2)
        tr.data += 6000 + 4 * tr.times() ** 2 - 0.1 * tr.times() ** 3 - \
            0.00001 * tr.times() ** 5

        with ImageComparison(self.path_images, 'polynomial_detrend.png') as ic:
            polynomial(tr.data, order=3, plot=ic.name)

    def test_spline_detrend_plotting(self):
        """
        Tests the plotting of the spline detrend operation.
        """
        tr = obspy.read()[0].filter("highpass", freq=2)
        tr.data += 6000 + 4 * tr.times() ** 2 - 0.1 * tr.times() ** 3 - \
            0.00001 * tr.times() ** 5

        # Use an first order spline to see a difference to the polynomial
        # picture.
        with ImageComparison(self.path_images,
                             'degree_1_spline_detrend.png') as ic:
            spline(tr.data, order=1, dspline=1500, plot=ic.name)


def suite():
    return unittest.makeSuite(DetrendTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
