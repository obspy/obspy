# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: konnoohmachismoothing.py
#  Purpose: Small module to smooth spectra with the so called Konno & Ohmachi
#           method.
#   Author: Lion Krischer, Derrick Chambers
#    Email: krischer@geophysik.uni-muenchen.de
#  License: GPLv2
#
# Copyright (C) 2011 Lion Krischer
# --------------------------------------------------------------------
"""
Functions to smooth spectra with the so called Konno & Ohmachi method.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import warnings

import numpy as np


def konno_ohmachi_smoothing_window(frequencies, center_frequency,
                                   bandwidth=40.0, normalize=False):
    """
    Calculate the Konno & Ohmachi smoothing window(s).

    Returns the smoothing window around the center frequency with one value per
    input frequency defined as follows (see [Konno1998]_)::

        [sin(b * log_10(f/f_c)) / (b * log_10(f/f_c)]^4
            b   = bandwidth
            f   = frequency
            f_c = center frequency

    If an array is provided for center_frequency then a 2D array is returned
    where each row is a smoothing window for the corresponding center
    frequency.

    The bandwidth of the smoothing function is constant on a logarithmic scale.
    A small value will lead to a strong smoothing, while a large value of will
    lead to a low smoothing of the Fourier spectra.
    The default (and generally used) value for the bandwidth is 40. (From the
    `Geopsy documentation <http://www.geopsy.org>`_)

    All parameters need to be positive. This is not checked due to performance
    reasons and therefore any negative parameters might have unexpected
    results.

    :type frequencies: :class:`numpy.ndarray` (float32 or float64)
    :param frequencies:
        All frequencies for which the smoothing window will be returned.
    :type center_frequency: float or :class:`numpy.ndarray`
    :param center_frequency:
        The frequency or frequencies around which the smoothing is performed.
        Must be greater or equal to 0.
    :type bandwidth: float
    :param bandwidth:
        Determines the width of the smoothing peak. Lower values result in a
        broader peak. Must be greater than 0. Defaults to 40.
    :type normalize: bool, optional
    :param normalize:
        The Konno-Ohmachi smoothing window is normalized on a logarithmic
        scale. Set this parameter to True to normalize it on a normal scale.
        Default to False.
    """
    if frequencies.dtype != np.float32 and frequencies.dtype != np.float64:
        msg = 'frequencies needs to have a dtype of float32/64.'
        raise ValueError(msg)
    # Cast center freq. to dtype of frequencies if it is not an array
    if not isinstance(center_frequency, np.ndarray):
        center_frequency = np.array(center_frequency, dtype=frequencies.dtype)
    # Disable div by zero errors and return zero instead
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate outer product
        outer_division = np.divide.outer(frequencies, center_frequency).T
        # Calculate the bandwidth*log10(f/f_c)
        product = bandwidth * np.log10(outer_division)
        # The Konno-Ohmachi formulae.
        window = (np.sin(product) / product) ** 4
    # Check if the center frequency is exactly part of the provided
    # frequencies. This will result in a division by 0. The limit of f->f_c is
    # one.
    window[np.equal.outer(center_frequency, frequencies)] = 1.0
    # For center_frequency == 0 ensure the row is 0 for every freq. except 0,
    # which should be one.
    window[center_frequency == 0.0] = 0.0
    window[np.logical_and.outer(center_frequency == 0, frequencies == 0)] = 1
    # A frequency of zero will result in a logarithm of -inf. The limit of
    # f->0 with f_c!=0 is zero. All remaining NaNs will be due to this.
    window[np.isnan(window)] = 0.
    # Normalize to one if wished. Transposes are to make division work in 2D.
    if normalize:
        window = (window.T / window.sum(axis=-1)).T
    return window


def calculate_smoothing_matrix(frequencies, bandwidth=40.0, normalize=False):
    """
    Calculates a len(frequencies) x len(frequencies) matrix with the Konno &
    Ohmachi window for each frequency as the center frequency.

    Any spectrum with the same frequency bins as this matrix can later be
    smoothed by using
    :func:`~obspy.signal.konnoohmachismoothing.apply_smoothing_matrix`.

    This also works for many spectra stored in one large matrix and is even
    more efficient.

    This makes it very efficient for smoothing the same spectra again and again
    but it comes with a high memory consumption for larger frequency arrays!

    :type frequencies: :class:`numpy.ndarray` (float32 or float64)
    :param frequencies:
        The input frequencies.
    :type bandwidth: float
    :param bandwidth:
        Determines the width of the smoothing peak. Lower values result in a
        broader peak. Must be greater than 0. Defaults to 40.
    :type normalize: bool, optional
    :param normalize:
        The Konno-Ohmachi smoothing window is normalized on a logarithmic
        scale. Set this parameter to True to normalize it on a normal scale.
        Default to False.
    """
    return konno_ohmachi_smoothing_window(frequencies, frequencies,
                                          bandwidth, normalize=normalize)


def _smooth_no_matrix(spectra, frequencies, bandwidth, normalize, count):
    """
    Apply the smoothing algorithm without using matrices.
    This is much slower but more memory efficient than using matricies.
    """
    if count == 0:
        return spectra
    new_spec = np.empty(spectra.shape, spectra.dtype)
    # Separate case for just one spectrum.
    if len(spectra.shape) == 1:
        for _i, freq in enumerate(frequencies):
            window = konno_ohmachi_smoothing_window(
                frequencies, freq, bandwidth, normalize=normalize)
            new_spec[_i] = (window * spectra).sum()
    else:  # Reuse smoothing window if more than one spectrum.
        for _i, freq in enumerate(frequencies):
            window = konno_ohmachi_smoothing_window(
                frequencies, freq, bandwidth, normalize=normalize)

            for _j, spec in enumerate(spectra):
                new_spec[_j, _i] = (window * spec).sum()
    # Call recursively to ensure spectra was smoothed count times
    return _smooth_no_matrix(new_spec, frequencies, bandwidth, normalize,
                             count - 1)


def apply_smoothing_matrix(spectra, smoothing_matrix, count=1):
    """
    Smooths a matrix containing one spectra per row with the Konno-Ohmachi
    smoothing window, using a smoothing matrix pre-computed through the
    :func:`~obspy.signal.konnoohmachismoothing.calculate_smoothing_matrix`
    function.
    This function is useful if one needs to smooth the same type of spectrum
    (same shape) through different function calls.

    All spectra need to have frequency bins corresponding to the same
    frequencies.
    """
    if spectra.dtype not in (np.float32, np.float64):
        msg = '`spectra` needs to have a dtype of float32/64.'
        raise ValueError(msg)
    new_spec = np.dot(spectra, smoothing_matrix)
    # Eventually apply more than once.
    for _i in range(count - 1):
        new_spec = np.dot(new_spec, smoothing_matrix)
    return new_spec


def konno_ohmachi_smoothing(spectra, frequencies, bandwidth=40, count=1,
                            enforce_no_matrix=False, max_memory_usage=512,
                            normalize=False):
    """
    Smooths a matrix containing one spectra per row with the Konno-Ohmachi
    smoothing window.

    All spectra need to have frequency bins corresponding to the same
    frequencies.

    This method first will estimate the memory usage and then either use a fast
    and memory intensive method or a slow one with a better memory usage.

    :type spectra: :class:`numpy.ndarray` (float32 or float64)
    :param spectra:
        One or more spectra per row. If more than one the first spectrum has to
        be accessible via spectra[0], the next via spectra[1], ...
    :type frequencies: :class:`numpy.ndarray` (float32 or float64)
    :param frequencies:
        Contains the frequencies for the spectra.
    :type bandwidth: float
    :param bandwidth:
        Determines the width of the smoothing peak. Lower values result in a
        broader peak. Must be greater than 0. Defaults to 40.
    :type count: int, optional
    :param count:
        How often the apply the filter. For very noisy spectra it is useful to
        apply is more than once. Defaults to 1.
    :type enforce_no_matrix: bool, optional
    :param enforce_no_matrix:
        An efficient but memory intensive matrix-multiplication algorithm is
        used in case more than one spectra is to be smoothed or one spectrum is
        to be smoothed more than once if enough memory is available. This flag
        disables the matrix algorithm altogether. Defaults to False
    :type max_memory_usage: int, optional
    :param max_memory_usage:
        Set the maximum amount of extra memory in MB for this method. Decides
        whether or not the matrix multiplication method is used. Defaults to
        512 MB.
    :type normalize: bool, optional
    :param normalize:
        The Konno-Ohmachi smoothing window is normalized on a logarithmic
        scale. Set this parameter to True to normalize it on a normal scale.
        Default to False.
    """
    if spectra.dtype not in (np.float32, np.float64):
        msg = '`spectra` needs to have a dtype of float32/64.'
        raise ValueError(msg)
    if frequencies.dtype not in (np.float32, np.float64):
        msg = '`frequencies` needs to have a dtype of float32/64.'
        raise ValueError(msg)
    # Spectra and frequencies should have the same dtype.
    if frequencies.dtype != spectra.dtype:
        frequencies = np.require(frequencies, np.float64)
        spectra = np.require(spectra, np.float64)
        msg = '`frequencies` and `spectra` should have the same dtype. It ' + \
              'will be changed to np.float64 for both.'
        warnings.warn(msg)
    # Check the dtype to get the correct size.
    if frequencies.dtype == np.float32:
        size = 4.0
    elif frequencies.dtype == np.float64:
        size = 8.0
    # Calculate the approximate memory required for the smoothing matrix.
    length = len(frequencies)
    element_count = (length * length + 2 * len(spectra) + length)
    mem_required = (element_count * size) / 1048576.
    # If smaller than the allowed maximum memory consumption build a smoothing
    # matrix and apply to each spectrum.
    if not enforce_no_matrix and mem_required < max_memory_usage:
        smoothing_matrix = calculate_smoothing_matrix(
            frequencies, bandwidth, normalize=normalize)
        return apply_smoothing_matrix(spectra, smoothing_matrix, count=count)
    # If memory is limited calculate/apply the smoothing window for each freq.
    else:
        return _smooth_no_matrix(spectra, frequencies, bandwidth, normalize,
                                 count)
