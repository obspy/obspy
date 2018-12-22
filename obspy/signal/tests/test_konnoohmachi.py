#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The polarization.core test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import itertools
import unittest
import warnings

import numpy as np

import obspy
from obspy.signal.konnoohmachismoothing import (calculate_smoothing_matrix,
                                                apply_smoothing_matrix,
                                                konno_ohmachi_smoothing_window,
                                                konno_ohmachi_smoothing)


class KonnoOhmachiTestCase(unittest.TestCase):
    """
    Test cases for the Konno Ohmachi Smoothing.
    """
    def setUp(self):
        self.frequencies = np.logspace(-3.0, 2.0, 100)

    def tearDown(self):
        pass

    def get_default_spectra(self, n=1):
        """
        Return the n spectra of the default stream and corresponding
        frequencies. If n > 3 repeat streams.
        """
        st = obspy.read()
        st.detrend('linear')
        data_len = len(st[0].data)
        sampling_period = 1. / st[0].stats.sampling_rate
        # Create spectra.
        cycle = itertools.cycle([tr.data for tr in st])
        data = np.array([x for _, x in zip(range(n), cycle)])
        spectrum = np.abs(np.fft.rfft(data, axis=-1))
        # Determine corresponding frequencies and return.
        freq = np.fft.rfftfreq(data_len, sampling_period)
        assert spectrum.shape[-1] == freq.shape[-1]
        return spectrum, freq

    def test_smoothing_window(self):
        """
        Tests the creation of the smoothing window.
        """
        # Frequency of zero results in a delta peak at zero (there usually
        # should be just one zero in the frequency array.
        window = konno_ohmachi_smoothing_window(
            np.array([0, 1, 0, 3], dtype=np.float32), 0)
        np.testing.assert_array_equal(window, np.array([1, 0, 1, 0],
                                                       dtype=np.float32))
        # Wrong dtypes raises.
        self.assertRaises(ValueError, konno_ohmachi_smoothing_window,
                          np.arange(10, dtype=np.int32), 10)
        # If frequency=center frequency, log results in infinity. Limit of
        # whole formulae is 1.
        window = konno_ohmachi_smoothing_window(
            np.array([5.0, 1.0, 5.0, 2.0], dtype=np.float32), 5)
        np.testing.assert_array_equal(
            window[[0, 2]], np.array([1.0, 1.0], dtype=np.float32))
        # Output dtype should be the dtype of frequencies.
        self.assertEqual(konno_ohmachi_smoothing_window(
            np.array([1, 6, 12], dtype=np.float32), 5).dtype, np.float32)
        self.assertEqual(konno_ohmachi_smoothing_window(
            np.array([1, 6, 12], dtype=np.float64), 5).dtype, np.float64)
        # Check if normalizing works.
        window = konno_ohmachi_smoothing_window(self.frequencies, 20)
        self.assertGreater(window.sum(), 1.0)
        window = konno_ohmachi_smoothing_window(self.frequencies, 20,
                                                normalize=True)
        self.assertAlmostEqual(window.sum(), 1.0, 5)
        # Just one more to test if there are no invalid values and the
        # range if ok.
        window = konno_ohmachi_smoothing_window(self.frequencies, 20)
        self.assertEqual(np.any(np.isnan(window)), False)
        self.assertEqual(np.any(np.isinf(window)), False)
        self.assertTrue(np.all(window <= 1.0))
        self.assertTrue(np.all(window >= 0.0))

    def test_smoothing_matrix(self):
        """
        Tests some aspects of the matrix.
        """
        frequencies = np.array([0.0, 1.0, 2.0, 10.0, 25.0, 50.0, 100.0],
                               dtype=np.float32)
        matrix = calculate_smoothing_matrix(frequencies, 20.0)
        self.assertEqual(matrix.dtype, np.float32)
        for _i, freq in enumerate(frequencies):
            np.testing.assert_array_equal(
                matrix[_i],
                konno_ohmachi_smoothing_window(frequencies, freq, 20.0))
            # Should not be normalized. Test only for larger frequencies
            # because smaller ones have a smaller window.
            if freq >= 10.0:
                self.assertGreater(matrix[_i].sum(), 1.0)
        # Input should be output dtype.
        frequencies = np.array(
            [0.0, 1.0, 2.0, 10.0, 25.0, 50.0, 100.0],
            dtype=np.float64)
        matrix = calculate_smoothing_matrix(frequencies, 20.0)
        self.assertEqual(matrix.dtype, np.float64)
        # Check normalization.
        frequencies = np.array(
            [0.0, 1.0, 2.0, 10.0, 25.0, 50.0, 100.0],
            dtype=np.float32)
        matrix = calculate_smoothing_matrix(frequencies, 20.0,
                                            normalize=True)
        self.assertEqual(matrix.dtype, np.float32)
        for _i, freq in enumerate(frequencies):
            np.testing.assert_array_equal(
                matrix[_i],
                konno_ohmachi_smoothing_window(
                    frequencies, freq, 20.0, normalize=True))
            # Should not be normalized. Test only for larger frequencies
            # because smaller ones have a smaller window.
            self.assertAlmostEqual(matrix[_i].sum(), 1.0, 5)

    def test_konno_ohmachi_smoothing(self):
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
        self.assertRaises(ValueError, konno_ohmachi_smoothing, spectra,
                          np.arange(200))
        # Differing float dtypes raise a warning.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            self.assertRaises(UserWarning, konno_ohmachi_smoothing, spectra,
                              frequencies)
        # Correct the dtype.
        frequencies = np.require(frequencies, dtype=np.float32)
        # The first one uses the matrix method, the second one the non matrix
        # method.
        smoothed_1 = konno_ohmachi_smoothing(spectra, frequencies, count=3)
        smoothed_2 = konno_ohmachi_smoothing(spectra, frequencies, count=3,
                                             max_memory_usage=0)
        # XXX: Why are the numerical inaccuracies quite large?
        np.testing.assert_almost_equal(smoothed_1, smoothed_2, 3)
        # Test using a pre-computed smoothing matrix
        smoothing_matrix = calculate_smoothing_matrix(frequencies)
        smoothed_3 = apply_smoothing_matrix(spectra, smoothing_matrix, count=3)
        np.testing.assert_almost_equal(smoothed_1, smoothed_3, 3)
        # Test the non-matrix mode for single spectra.
        smoothed_4 = konno_ohmachi_smoothing(
            np.require(spectra[0], dtype=np.float64),
            np.require(frequencies, dtype=np.float64),
            enforce_no_matrix=True)
        smoothed_5 = konno_ohmachi_smoothing(
            np.require(spectra[0], dtype=np.float64),
            np.require(frequencies, dtype=np.float64),
            enforce_no_matrix=True,
            normalize=True)
        self.assertFalse(np.all(smoothed_4 == smoothed_5))
        # Input dtype should be output dtype.
        self.assertEqual(smoothed_4.dtype, np.float64)

    def test_smoothing_with_and_without_smoothing_matrix(self):
        """
        The results of smoothing wih enforce_no_matrix should be the same
        as without.
        """
        spectra, frequencies = self.get_default_spectra(3)

        # # test normalization with disabled matrix and enabled ones
        smoothed_1 = konno_ohmachi_smoothing(spectra, frequencies,
                                             normalize=True)
        smoothed_2 = konno_ohmachi_smoothing(spectra, frequencies,
                                             normalize=True,
                                             enforce_no_matrix=True)
        np.testing.assert_almost_equal(smoothed_1, smoothed_2, 5)
        # # test normalization with disabled matrix and enabled ones
        smoothed_1 = konno_ohmachi_smoothing(spectra, frequencies)
        smoothed_2 = konno_ohmachi_smoothing(spectra, frequencies,
                                             enforce_no_matrix=True)
        np.testing.assert_almost_equal(smoothed_1, smoothed_2, 5)

    def test_downsample_with_center_frequencies(self):
        """
        Test the smoothing with specified center frequencies for downsampling.
        """
        spectra, frequencies = self.get_default_spectra(4)
        # Use a linear extrapolation to get center frequencies.
        center_frequencies = np.linspace(min(frequencies), max(frequencies))
        out = konno_ohmachi_smoothing(spectra, frequencies,
                                      center_frequencies=center_frequencies)
        self.assertEqual(out.shape[-1], center_frequencies.shape[-1])
        self.assertEqual(out.shape[0], 4)
        # Test with one 1D array.
        spectrum = spectra[0, :]
        out = konno_ohmachi_smoothing(spectrum, frequencies,
                                      center_frequencies=center_frequencies)
        self.assertEqual(out.shape[-1], center_frequencies.shape[-1])
        # Test count != 0 still works.
        out = konno_ohmachi_smoothing(spectrum, frequencies, count=5,
                                      center_frequencies=center_frequencies)
        self.assertEqual(out.shape[-1], center_frequencies.shape[-1])
        # It should also work if smoothing matrix is disabled.
        spectrum = spectra[0, :]
        out = konno_ohmachi_smoothing(spectrum, frequencies,
                                      center_frequencies=center_frequencies,
                                      enforce_no_matrix=True, count=2)
        self.assertEqual(out.shape[-1], center_frequencies.shape[-1])

    def test_upsampling_with_center_frequencies(self):
        """
        Smoothing should also work when the number of center frequencies
        is greater than the original frequencies.
        """
        spectra, frequencies = self.get_default_spectra(4)
        num = int(spectra.shape[-1] * 1.2)
        center_frequencies = np.linspace(min(frequencies), max(frequencies),
                                         num=num)
        out = konno_ohmachi_smoothing(spectra, frequencies,
                                      center_frequencies=center_frequencies)
        self.assertEqual(out.shape[-1], center_frequencies.shape[-1])

    def test_center_frequencies_outside_frequencies_raises(self):
        """
        Center frequencies must be contained in the range of frequencies
        or a ValueError should be raised.
        """
        spectra, frequencies = self.get_default_spectra(2)
        # First test a value which is too low.
        center_frequencies = np.copy(frequencies)
        center_frequencies[0] = -1
        self.assertRaises(ValueError, konno_ohmachi_smoothing, spectra,
                          frequencies, center_frequencies=center_frequencies)
        # Then test a value which is too high
        center_frequencies = np.copy(frequencies)
        center_frequencies[-1] = center_frequencies[-1] * 10
        self.assertRaises(ValueError, konno_ohmachi_smoothing, spectra,
                          frequencies, center_frequencies=center_frequencies)


def suite():
    return unittest.makeSuite(KonnoOhmachiTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
