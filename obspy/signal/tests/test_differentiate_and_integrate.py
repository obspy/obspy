#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for differentiation and integration functions.
"""
import unittest

import numpy as np

from obspy.signal.differentiate_and_integrate import (
    integrate_cumtrapz, integrate_spline)


class IntegrateTestCase(unittest.TestCase):
    """
    Test cases for the integration methods.
    """
    def test_cumtrapz_integration(self):
        """
        Test the basic and obvious cases. We are using external methods which
        are tested extensively elsewhere.
        """
        np.testing.assert_allclose(
            integrate_cumtrapz(np.ones(10), dx=1.0),
            np.arange(10))

        np.testing.assert_allclose(
            integrate_cumtrapz(np.ones(10), dx=2.0),
            np.arange(10) * 2.0)

        np.testing.assert_allclose(
            integrate_cumtrapz(np.ones(10), dx=0.5),
            np.arange(10) * 0.5)

        np.testing.assert_allclose(
            integrate_cumtrapz(np.zeros(10), dx=0.5),
            np.zeros(10))

    def test_spline_integration(self):
        """
        Test the basic and obvious cases. We are using external methods which
        are tested extensively elsewhere.
        """
        for k in (1, 2, 3):
            np.testing.assert_allclose(
                integrate_spline(np.ones(10), dx=1.0, k=k),
                np.arange(10))

            np.testing.assert_allclose(
                integrate_spline(np.ones(10), dx=2.0, k=k),
                np.arange(10) * 2.0)

            np.testing.assert_allclose(
                integrate_spline(np.ones(10), dx=0.5, k=k),
                np.arange(10) * 0.5)

            np.testing.assert_allclose(
                integrate_spline(np.zeros(10), dx=0.5, k=k),
                np.zeros(10))
