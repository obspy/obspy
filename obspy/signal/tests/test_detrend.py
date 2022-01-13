#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for data detrending.
"""
import os

import numpy as np

import obspy
from obspy.signal.detrend import polynomial, spline


class TestDetrend:
    """
    Test cases for the detrend methods.
    """
    path = os.path.join(os.path.dirname(__file__), 'data')
    path_images = os.path.join(os.path.dirname(__file__), 'images')

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
            assert np.ptp(detrended) * 1E10 < original_ptp

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
            assert np.ptp(detrended) * 1E4 < original_ptp

    def test_polynomial_detrend_plotting(self, image_path):
        """
        Tests the plotting of the polynomial detrend operation.
        """
        tr = obspy.read()[0].filter("highpass", freq=2)
        tr.data += 6000 + 4 * tr.times() ** 2 - 0.1 * tr.times() ** 3 - \
            0.00001 * tr.times() ** 5
        polynomial(tr.data, order=3, plot=image_path)

    def test_spline_detrend_plotting(self, image_path):
        """
        Tests the plotting of the spline detrend operation.
        """
        tr = obspy.read()[0].filter("highpass", freq=2)
        tr.data += 6000 + 4 * tr.times() ** 2 - 0.1 * tr.times() ** 3 - \
            0.00001 * tr.times() ** 5
        spline(tr.data, order=1, dspline=1500, plot=image_path)
