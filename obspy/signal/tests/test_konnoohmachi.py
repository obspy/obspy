#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The polarization.core test suite.
"""
import warnings

import numpy as np

from obspy.signal.konnoohmachismoothing import (calculate_smoothing_matrix,
                                                apply_smoothing_matrix,
                                                konno_ohmachi_smoothing_window,
                                                konno_ohmachi_smoothing)
import pytest


class TestKonnoOhmachi():
    """
    Test cases for the Konno Ohmachi Smoothing.
    """
    @classmethod
    def setup_class(cls):
        cls.frequencies = np.logspace(-3.0, 2.0, 100)

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
        with pytest.raises(ValueError):
            konno_ohmachi_smoothing_window(np.arange(10, dtype=np.int32), 10)
        # If frequency=center frequency, log results in infinity. Limit of
        # whole formulae is 1.
        window = konno_ohmachi_smoothing_window(
            np.array([5.0, 1.0, 5.0, 2.0], dtype=np.float32), 5)
        np.testing.assert_array_equal(
            window[[0, 2]], np.array([1.0, 1.0], dtype=np.float32))
        # Output dtype should be the dtype of frequencies.
        assert konno_ohmachi_smoothing_window(
            np.array([1, 6, 12], dtype=np.float32), 5).dtype == np.float32
        assert konno_ohmachi_smoothing_window(
            np.array([1, 6, 12], dtype=np.float64), 5).dtype == np.float64
        # Check if normalizing works.
        window = konno_ohmachi_smoothing_window(self.frequencies, 20)
        assert window.sum() > 1.0
        window = konno_ohmachi_smoothing_window(self.frequencies, 20,
                                                normalize=True)
        assert round(abs(window.sum()-1.0), 5) == 0
        # Just one more to test if there are no invalid values and the
        # range if ok.
        window = konno_ohmachi_smoothing_window(self.frequencies, 20)
        assert not np.any(np.isnan(window))
        assert not np.any(np.isinf(window))
        assert np.all(window <= 1.0)
        assert np.all(window >= 0.0)

    def test_smoothing_matrix(self):
        """
        Tests some aspects of the matrix.
        """
        frequencies = np.array([0.0, 1.0, 2.0, 10.0, 25.0, 50.0, 100.0],
                               dtype=np.float32)
        matrix = calculate_smoothing_matrix(frequencies, 20.0)
        assert matrix.dtype == np.float32
        for _i, freq in enumerate(frequencies):
            np.testing.assert_array_equal(
                matrix[_i],
                konno_ohmachi_smoothing_window(frequencies, freq, 20.0))
            # Should not be normalized. Test only for larger frequencies
            # because smaller ones have a smaller window.
            if freq >= 10.0:
                assert matrix[_i].sum() > 1.0
        # Input should be output dtype.
        frequencies = np.array(
            [0.0, 1.0, 2.0, 10.0, 25.0, 50.0, 100.0],
            dtype=np.float64)
        matrix = calculate_smoothing_matrix(frequencies, 20.0)
        assert matrix.dtype == np.float64
        # Check normalization.
        frequencies = np.array(
            [0.0, 1.0, 2.0, 10.0, 25.0, 50.0, 100.0],
            dtype=np.float32)
        matrix = calculate_smoothing_matrix(frequencies, 20.0,
                                            normalize=True)
        assert matrix.dtype == np.float32
        for _i, freq in enumerate(frequencies):
            np.testing.assert_array_equal(
                matrix[_i],
                konno_ohmachi_smoothing_window(
                    frequencies, freq, 20.0, normalize=True))
            # Should not be normalized. Test only for larger frequencies
            # because smaller ones have a smaller window.
            assert round(abs(matrix[_i].sum()-1.0), 5) == 0

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
        with pytest.raises(ValueError):
            konno_ohmachi_smoothing(spectra, np.arange(200))
        # Differing float dtypes raise a warning.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            with pytest.raises(UserWarning):
                konno_ohmachi_smoothing(spectra, frequencies)
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
            np.require(frequencies, dtype=np.float64))
        smoothed_5 = konno_ohmachi_smoothing(
            np.require(spectra[0], dtype=np.float64),
            np.require(frequencies, dtype=np.float64),
            normalize=True)
        # The normalized and not normalized should not be the same. That the
        # normalizing works has been tested before.
        assert not np.all(smoothed_4 == smoothed_5)
        # Input dtype should be output dtype.
        assert smoothed_4.dtype == np.float64
