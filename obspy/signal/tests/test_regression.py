#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for linear regression.
"""
import unittest

import numpy as np

from obspy.signal.regression import linear_regression


class RegressionTestCase(unittest.TestCase):
    """
    Test cases for the regression method.
    """
    def setUp(self):
        # seed a random sequence and add some noise
        np.random.seed(0)
        n = 10
        self.x = np.arange(n, dtype=float)
        self.y = np.arange(n, dtype=float) + np.random.random(n) + 5 + \
            np.random.random(n)
        self.weights = np.linspace(1.0, 10.0, n)

    def test_weight_intercept(self):
        """
        Test for regression with weight and any intercept
        """
        ref_result = [1.103354807117634, 5.5337860049233898,
                      0.076215945728448836, 0.54474791667622224]
        result = linear_regression(self.x, self.y, self.weights,
                                   intercept_origin=False)
        np.testing.assert_equal(len(result), 4)
        np.testing.assert_allclose(result, ref_result)

    def test_weight_nointercept(self):
        """
        Test for regression with weight and intercept at origin
        """
        ref_result = [1.8461448748925857, 0.075572144774706959]
        result = linear_regression(self.x, self.y, self.weights,
                                   intercept_origin=True)
        np.testing.assert_equal(len(result), 2)
        np.testing.assert_allclose(result, ref_result)

    def test_noweight_intercept(self):
        """
        Test for regression with no weight and any intercept
        """
        ref_result = [1.0161734373705231, 6.090329180877835,
                      0.053630481620317534, 0.28630842447712868]
        result = linear_regression(self.x, self.y, intercept_origin=False)
        np.testing.assert_equal(len(result), 4)
        np.testing.assert_allclose(result, ref_result)

    def test_noweight_nointercept(self):
        """
        Test for regression with no weight and intercept at origin
        """
        ref_result = [1.9778043606670241, 0.20639881421917514]
        result = linear_regression(self.x, self.y, intercept_origin=True)
        np.testing.assert_equal(len(result), 2)
        np.testing.assert_allclose(result, ref_result)
