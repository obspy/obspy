#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The polarization.core test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.signal import konnoOhmachiSmoothing
from obspy.signal.konnoohmachismoothing import konnoOhmachiSmoothingWindow, \
    calculateSmoothingMatrix
import numpy as np
import unittest
import warnings


class KonnoOhmachiTestCase(unittest.TestCase):
    """
    Test cases for the Konno Ohmachi Smoothing.
    """
    def setUp(self):
        self.frequencies = np.logspace(-3.0, 2.0, 100)

    def tearDown(self):
        pass

    def test_smoothingWindow(self):
        """
        Tests the creation of the smoothing window.
        """
        # Disable div by zero errors.
        temp = np.geterr()
        np.seterr(all='ignore')
        # Frequency of zero results in a delta peak at zero (there usually
        # should be just one zero in the frequency array.
        window = konnoOhmachiSmoothingWindow(np.array([0, 1, 0, 3],
                                                      dtype=np.float32), 0)
        np.testing.assert_array_equal(window, np.array([1, 0, 1, 0],
                                                       dtype=np.float32))
        # Wrong dtypes raises.
        self.assertRaises(ValueError, konnoOhmachiSmoothingWindow,
                          np.arange(10, dtype=np.int32), 10)
        # If frequency=center frequency, log results in infinity. Limit of
        # whole formulae is 1.
        window = konnoOhmachiSmoothingWindow(np.array([5.0, 1.0, 5.0, 2.0],
                                                      dtype=np.float32), 5)
        np.testing.assert_array_equal(
            window[[0, 2]], np.array([1.0, 1.0], dtype=np.float32))
        # Output dtype should be the dtype of frequencies.
        self.assertEqual(konnoOhmachiSmoothingWindow(np.array([1, 6, 12],
                         dtype=np.float32), 5).dtype, np.float32)
        self.assertEqual(konnoOhmachiSmoothingWindow(np.array([1, 6, 12],
                         dtype=np.float64), 5).dtype, np.float64)
        # Check if normalizing works.
        window = konnoOhmachiSmoothingWindow(self.frequencies, 20)
        self.assertTrue(window.sum() > 1.0)
        window = konnoOhmachiSmoothingWindow(self.frequencies, 20,
                                             normalize=True)
        self.assertAlmostEqual(window.sum(), 1.0, 5)
        # Just one more to test if there are no invalid values and the range if
        # ok.
        window = konnoOhmachiSmoothingWindow(self.frequencies, 20)
        self.assertEqual(np.any(np.isnan(window)), False)
        self.assertEqual(np.any(np.isinf(window)), False)
        self.assertTrue(np.all(window <= 1.0))
        self.assertTrue(np.all(window >= 0.0))
        np.seterr(**temp)

    def test_smoothingMatrix(self):
        """
        Tests some aspects of the matrix.
        """
        # Disable div by zero errors.
        temp = np.geterr()
        np.seterr(all='ignore')
        frequencies = np.array([0.0, 1.0, 2.0, 10.0, 25.0, 50.0, 100.0],
                               dtype=np.float32)
        matrix = calculateSmoothingMatrix(frequencies, 20.0)
        self.assertEqual(matrix.dtype, np.float32)
        for _i, freq in enumerate(frequencies):
            np.testing.assert_array_equal(
                matrix[_i],
                konnoOhmachiSmoothingWindow(frequencies, freq, 20.0))
            # Should not be normalized. Test only for larger frequencies
            # because smaller ones have a smaller window.
            if freq >= 10.0:
                self.assertTrue(matrix[_i].sum() > 1.0)
        # Input should be output dtype.
        frequencies = np.array(
            [0.0, 1.0, 2.0, 10.0, 25.0, 50.0, 100.0],
            dtype=np.float64)
        matrix = calculateSmoothingMatrix(frequencies, 20.0)
        self.assertEqual(matrix.dtype, np.float64)
        # Check normalization.
        frequencies = np.array(
            [0.0, 1.0, 2.0, 10.0, 25.0, 50.0, 100.0],
            dtype=np.float32)
        matrix = calculateSmoothingMatrix(frequencies, 20.0, normalize=True)
        self.assertEqual(matrix.dtype, np.float32)
        for _i, freq in enumerate(frequencies):
            np.testing.assert_array_equal(
                matrix[_i],
                konnoOhmachiSmoothingWindow(frequencies, freq, 20.0,
                                            normalize=True))
            # Should not be normalized. Test only for larger frequencies
            # because smaller ones have a smaller window.
            self.assertAlmostEqual(matrix[_i].sum(), 1.0, 5)
        np.seterr(**temp)

    def test_konnoOhmachiSmoothing(self):
        """
        Tests the actual smoothing matrix.
        """
        # Create some random spectra.
        np.random.seed(1111)
        spectra = np.random.ranf((5, 200)) * 50
        frequencies = np.logspace(-3.0, 2.0, 200)
        spectra = np.require(spectra, dtype=np.float32)
        frequencies = np.require(frequencies, dtype=np.float64)
        # Wrong dtype raises.
        self.assertRaises(ValueError, konnoOhmachiSmoothing, spectra,
                          np.arange(200))
        # Differing float dtypes raise a warning.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            self.assertRaises(UserWarning, konnoOhmachiSmoothing, spectra,
                              frequencies)
        # Correct the dtype.
        frequencies = np.require(frequencies, dtype=np.float32)
        # The first one uses the matrix method, the second one the non matrix
        # method.
        smoothed_1 = konnoOhmachiSmoothing(spectra, frequencies, count=3)
        smoothed_2 = konnoOhmachiSmoothing(spectra, frequencies, count=3,
                                           max_memory_usage=0)
        # XXX: Why are the numerical inaccuracies quite large?
        np.testing.assert_almost_equal(smoothed_1, smoothed_2, 3)
        # Test the non-matrix mode for single spectra.
        smoothed_3 = konnoOhmachiSmoothing(
            np.require(spectra[0], dtype=np.float64),
            np.require(frequencies, dtype=np.float64))
        smoothed_4 = konnoOhmachiSmoothing(
            np.require(spectra[0], dtype=np.float64),
            np.require(frequencies, dtype=np.float64),
            normalize=True)
        # The normalized and not normalized should not be the same. That the
        # normalizing works has been tested before.
        self.assertFalse(np.all(smoothed_3 == smoothed_4))
        # Input dtype should be output dtype.
        self.assertEqual(smoothed_3.dtype, np.float64)


def suite():
    return unittest.makeSuite(KonnoOhmachiTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
