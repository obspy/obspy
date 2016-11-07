#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: tf_misfit.py
#  Purpose: Various Time Frequency Misfit Functions
#   Author: Martin van Driel
#    Email: vandriel@sed.ethz.ch
#
# Copyright (C) 2012 Martin van Driel
# --------------------------------------------------------------------
"""
Various Time Frequency Misfit Functions based on [Kristekova2006]_ and
[Kristekova2009]_.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np

from obspy.imaging.cm import obspy_sequential, obspy_divergent
from obspy.signal import util


def cwt(st, dt, w0, fmin, fmax, nf=100, wl='morlet'):
    """
    Continuous Wavelet Transformation in the Frequency Domain.

    .. seealso:: [Kristekova2006]_, eq. (4)

    :param st: time dependent signal.
    :param dt: time step between two samples in st (in seconds)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param fmin: minimum frequency (in Hz)
    :param fmax: maximum frequency (in Hz)
    :param nf: number of logarithmically spaced frequencies between fmin and
        fmax
    :param wl: wavelet to use, for now only 'morlet' is implemented

    :return: time frequency representation of st, type numpy.ndarray of complex
        values, shape = (nf, len(st)).
    """
    npts = len(st) * 2
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts)
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    cwt = np.zeros((npts // 2, nf), dtype=np.complex)

    if wl == 'morlet':

        def psi(t):
            return np.pi ** (-.25) * np.exp(1j * w0 * t) * \
                np.exp(-t ** 2 / 2.)

        def scale(f):
            return w0 / (2 * np.pi * f)
    else:
        raise ValueError('wavelet type "' + wl + '" not defined!')

    nfft = util.next_pow_2(npts) * 2
    sf = np.fft.fft(st, n=nfft)

    # Ignore underflows.
    with np.errstate(under="ignore"):
        for n, _f in enumerate(f):
            a = scale(_f)
            # time shift necessary, because wavelet is defined around t = 0
            psih = psi(-1 * (t - t[-1] / 2.) / a).conjugate() / np.abs(a) ** .5
            psihf = np.fft.fft(psih, n=nfft)
            tminin = int(t[-1] / 2. / (t[1] - t[0]))
            cwt[:, n] = np.fft.ifft(psihf * sf)[tminin:tminin + npts // 2] * \
                (t[1] - t[0])

    return cwt.T


def tfem(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
         st2_isref=True):
    """
    Time Frequency Envelope Misfit

    .. seealso:: [Kristekova2009]_, Table 1. and 2.

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference

    :return: time frequency representation of Envelope Misfit,
        type numpy.ndarray with shape (nf, len(st1)) for single component data
        and (number of components, nf, len(st1)) for multicomponent data
    """
    if len(st1.shape) == 1:
        w_1 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)
        w_2 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)

        w_1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        w_2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        w_1 = np.zeros((st1.shape[0], nf, st1.shape[1]), dtype=np.complex)
        w_2 = np.zeros((st2.shape[0], nf, st2.shape[1]), dtype=np.complex)

        for i in np.arange(st1.shape[0]):
            w_1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            w_2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        ar = np.abs(w_2)
    else:
        if np.abs(w_1).max() > np.abs(w_2).max():
            ar = np.abs(w_1)
        else:
            ar = np.abs(w_2)

    _tfem = (np.abs(w_1) - np.abs(w_2))

    if norm == 'global':
        if len(st1.shape) == 1:
            return _tfem[0] / np.max(ar)
        else:
            return _tfem / np.max(ar)
    elif norm == 'local':
        if len(st1.shape) == 1:
            return _tfem[0] / ar[0]
        else:
            return _tfem / ar
    else:
        raise ValueError('norm "' + norm + '" not defined!')


def tfpm(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
         st2_isref=True):
    """
    Time Frequency Phase Misfit

    .. seealso:: [Kristekova2009]_, Table 1. and 2.

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference

    :return: time frequency representation of Phase Misfit,
        type numpy.ndarray with shape (nf, len(st1)) for single component data
        and (number of components, nf, len(st1)) for multicomponent data
    """
    if len(st1.shape) == 1:
        w_1 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)
        w_2 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)

        w_1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        w_2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        w_1 = np.zeros((st1.shape[0], nf, st1.shape[1]), dtype=np.complex)
        w_2 = np.zeros((st2.shape[0], nf, st2.shape[1]), dtype=np.complex)

        for i in np.arange(st1.shape[0]):
            w_1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            w_2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        _ar = np.abs(w_2)
    else:
        if np.abs(w_1).max() > np.abs(w_2).max():
            _ar = np.abs(w_1)
        else:
            _ar = np.abs(w_2)

    _tfpm = np.angle(w_1 / w_2) / np.pi

    if norm == 'global':
        if len(st1.shape) == 1:
            return _ar[0] * _tfpm[0] / np.max(_ar)
        else:
            return _ar * _tfpm / np.max(_ar)
    elif norm == 'local':
        if len(st1.shape) == 1:
            return _tfpm[0]
        else:
            return _tfpm
    else:
        raise ValueError('norm "' + norm + '" not defined!')


def tem(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
        st2_isref=True):
    """
    Time-dependent Envelope Misfit

    .. seealso:: [Kristekova2009]_, Table 1. and 2.

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference

    :return: Time-dependent Envelope Misfit, type numpy.ndarray with shape
        (len(st1),) for single component data and (number of components,
        len(st1)) for multicomponent data
    """
    if len(st1.shape) == 1:
        w_1 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)
        w_2 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)

        w_1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        w_2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        w_1 = np.zeros((st1.shape[0], nf, st1.shape[1]), dtype=np.complex)
        w_2 = np.zeros((st2.shape[0], nf, st2.shape[1]), dtype=np.complex)

        for i in np.arange(st1.shape[0]):
            w_1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            w_2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        _ar = np.abs(w_2)
    else:
        if np.abs(w_1).max() > np.abs(w_2).max():
            _ar = np.abs(w_1)
        else:
            _ar = np.abs(w_2)

    _tem = np.sum((np.abs(w_1) - np.abs(w_2)), axis=1)

    if norm == 'global':
        if len(st1.shape) == 1:
            return _tem[0] / np.max(np.sum(_ar, axis=1))
        else:
            return _tem / np.max(np.sum(_ar, axis=1))
    elif norm == 'local':
        if len(st1.shape) == 1:
            return _tem[0] / np.sum(_ar, axis=1)[0]
        else:
            return _tem / np.sum(_ar, axis=1)
    else:
        raise ValueError('norm "' + norm + '" not defined!')


def tpm(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
        st2_isref=True):
    """
    Time-dependent Phase Misfit

    .. seealso:: [Kristekova2009]_, Table 1. and 2.

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference

    :return: Time-dependent Phase Misfit, type numpy.ndarray with shape
        (len(st1),) for single component data and (number of components,
        len(st1)) for multicomponent data
    """
    if len(st1.shape) == 1:
        w_1 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)
        w_2 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)

        w_1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        w_2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        w_1 = np.zeros((st1.shape[0], nf, st1.shape[1]), dtype=np.complex)
        w_2 = np.zeros((st2.shape[0], nf, st2.shape[1]), dtype=np.complex)

        for i in np.arange(st1.shape[0]):
            w_1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            w_2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        _ar = np.abs(w_2)
    else:
        if np.abs(w_1).max() > np.abs(w_2).max():
            _ar = np.abs(w_2)
        else:
            _ar = np.abs(w_1)

    _tpm = np.angle(w_1 / w_2) / np.pi
    _tpm = np.sum(_ar * _tpm, axis=1)

    if norm == 'global':
        if len(st1.shape) == 1:
            return _tpm[0] / np.max(np.sum(_ar, axis=1))
        else:
            return _tpm / np.max(np.sum(_ar, axis=1))
    elif norm == 'local':
        if len(st1.shape) == 1:
            return _tpm[0] / np.sum(_ar, axis=1)[0]
        else:
            return _tpm / np.sum(_ar, axis=1)
    else:
        raise ValueError('norm "' + norm + '" not defined!')


def fem(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
        st2_isref=True):
    """
    Frequency-dependent Envelope Misfit

    .. seealso:: [Kristekova2009]_, Table 1. and 2.

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference

    :return: Frequency-dependent Envelope Misfit, type numpy.ndarray with shape
        (nf,) for single component data and (number of components, nf) for
        multicomponent data
    """
    if len(st1.shape) == 1:
        w_1 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)
        w_2 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)

        w_1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        w_2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        w_1 = np.zeros((st1.shape[0], nf, st1.shape[1]), dtype=np.complex)
        w_2 = np.zeros((st2.shape[0], nf, st2.shape[1]), dtype=np.complex)

        for i in np.arange(st1.shape[0]):
            w_1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            w_2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        _ar = np.abs(w_2)
    else:
        if np.abs(w_1).max() > np.abs(w_2).max():
            _ar = np.abs(w_1)
        else:
            _ar = np.abs(w_2)

    _tem = np.abs(w_1) - np.abs(w_2)
    _tem = np.sum(_tem, axis=2)

    if norm == 'global':
        if len(st1.shape) == 1:
            return _tem[0] / np.max(np.sum(_ar, axis=2))
        else:
            return _tem / np.max(np.sum(_ar, axis=2))
    elif norm == 'local':
        if len(st1.shape) == 1:
            return _tem[0] / np.sum(_ar, axis=2)[0]
        else:
            return _tem / np.sum(_ar, axis=2)
    else:
        raise ValueError('norm "' + norm + '" not defined!')


def fpm(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
        st2_isref=True):
    """
    Frequency-dependent Phase Misfit

    .. seealso:: [Kristekova2009]_, Table 1. and 2.

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference

    :return: Frequency-dependent Phase Misfit, type numpy.ndarray with shape
        (nf,) for single component data and (number of components, nf) for
        multicomponent data
    """
    if len(st1.shape) == 1:
        w_1 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)
        w_2 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)

        w_1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        w_2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        w_1 = np.zeros((st1.shape[0], nf, st1.shape[1]), dtype=np.complex)
        w_2 = np.zeros((st2.shape[0], nf, st2.shape[1]), dtype=np.complex)

        for i in np.arange(st1.shape[0]):
            w_1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            w_2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        _ar = np.abs(w_2)
    else:
        if np.abs(w_1).max() > np.abs(w_2).max():
            _ar = np.abs(w_1)
        else:
            _ar = np.abs(w_2)

    _tpm = np.angle(w_1 / w_2) / np.pi
    _tpm = np.sum(_ar * _tpm, axis=2)

    if norm == 'global':
        if len(st1.shape) == 1:
            return _tpm[0] / np.max(np.sum(_ar, axis=2))
        else:
            return _tpm / np.max(np.sum(_ar, axis=2))
    elif norm == 'local':
        if len(st1.shape) == 1:
            return _tpm[0] / np.sum(_ar, axis=2)[0]
        else:
            return _tpm / np.sum(_ar, axis=2)
    else:
        raise ValueError('norm "' + norm + '" not defined!')


def em(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
       st2_isref=True):
    """
    Single Valued Envelope Misfit

    .. seealso:: [Kristekova2009]_, Table 1. and 2.

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference

    :return: Single Valued Envelope Misfit
    """
    if len(st1.shape) == 1:
        w_1 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)
        w_2 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)

        w_1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        w_2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        w_1 = np.zeros((st1.shape[0], nf, st1.shape[1]), dtype=np.complex)
        w_2 = np.zeros((st2.shape[0], nf, st2.shape[1]), dtype=np.complex)

        for i in np.arange(st1.shape[0]):
            w_1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            w_2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        _ar = np.abs(w_2)
    else:
        if np.abs(w_1).max() > np.abs(w_2).max():
            _ar = np.abs(w_1)
        else:
            _ar = np.abs(w_2)

    _em = (np.sum(np.sum((np.abs(w_1) - np.abs(w_2)) ** 2, axis=2),
                  axis=1)) ** .5

    if norm == 'global':
        if len(st1.shape) == 1:
            return _em[0] / (np.sum(_ar ** 2)) ** .5
        else:
            return _em / ((np.sum(np.sum(_ar ** 2, axis=2),
                                  axis=1)) ** .5).max()
    elif norm == 'local':
        if len(st1.shape) == 1:
            return _em[0] / (np.sum(_ar ** 2)) ** .5
        else:
            return _em / (np.sum(np.sum(_ar ** 2, axis=2), axis=1)) ** .5
    else:
        raise ValueError('norm "' + norm + '" not defined!')


def pm(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
       st2_isref=True):
    """
    Single Valued Phase Misfit

    .. seealso:: [Kristekova2009]_, Table 1. and 2.

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference

    :return: Single Valued Phase Misfit
    """
    if len(st1.shape) == 1:
        w_1 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)
        w_2 = np.zeros((1, nf, st1.shape[0]), dtype=np.complex)

        w_1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        w_2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        w_1 = np.zeros((st1.shape[0], nf, st1.shape[1]), dtype=np.complex)
        w_2 = np.zeros((st2.shape[0], nf, st2.shape[1]), dtype=np.complex)

        for i in np.arange(st1.shape[0]):
            w_1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            w_2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        _ar = np.abs(w_2)
    else:
        if np.abs(w_1).max() > np.abs(w_2).max():
            _ar = np.abs(w_1)
        else:
            _ar = np.abs(w_2)

    _pm = np.angle(w_1 / w_2) / np.pi

    _pm = (np.sum(np.sum((_ar * _pm) ** 2, axis=2), axis=1)) ** .5

    if norm == 'global':
        if len(st1.shape) == 1:
            return _pm[0] / (np.sum(_ar ** 2)) ** .5
        else:
            return _pm / ((np.sum(np.sum(_ar ** 2, axis=2),
                                  axis=1)) ** .5).max()
    elif norm == 'local':
        if len(st1.shape) == 1:
            return _pm[0] / (np.sum(_ar ** 2)) ** .5
        else:
            return _pm / (np.sum(np.sum(_ar ** 2, axis=2), axis=1)) ** .5
    else:
        raise ValueError('norm "' + norm + '" not defined!')


def tfeg(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
         st2_isref=True, a=10., k=1.):
    """
    Time Frequency Envelope Goodness-of-Fit

    .. seealso:: [Kristekova2009]_, Eq.(15)

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference
    :param a: Maximum value of Goodness-of-Fit for perfect agreement
    :param k: sensitivity of Goodness-of-Fit to the misfit

    :return: time frequency representation of Envelope Goodness-of-Fit,
        type numpy.ndarray with shape (nf, len(st1)) for single component data
        and (number of components, nf, len(st1)) for multicomponent data
    """
    _tfem = tfem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0,
                 norm=norm, st2_isref=st2_isref)
    return a * np.exp(-np.abs(_tfem) ** k)


def tfpg(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
         st2_isref=True, a=10., k=1.):
    """
    Time Frequency Phase Goodness-of-Fit

    .. seealso:: [Kristekova2009]_, Eq.(16)

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference
    :param a: Maximum value of Goodness-of-Fit for perfect agreement
    :param k: sensitivity of Goodness-of-Fit to the misfit

    :return: time frequency representation of Phase Goodness-of-Fit,
        type numpy.ndarray with shape (nf, len(st1)) for single component data
        and (number of components, nf, len(st1)) for multicomponent data
    """
    _tfpm = tfpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0,
                 norm=norm, st2_isref=st2_isref)
    return a * (1 - np.abs(_tfpm) ** k)


def teg(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
        st2_isref=True, a=10., k=1.):
    """
    Time-dependent Envelope Goodness-of-Fit

    .. seealso:: [Kristekova2009]_, Eq.(15)

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference
    :param a: Maximum value of Goodness-of-Fit for perfect agreement
    :param k: sensitivity of Goodness-of-Fit to the misfit

    :return: time dependent Envelope Goodness-of-Fit, type numpy.ndarray with
        shape (len(st1),) for single component data and (number of components,
        len(st1)) for multicomponent data
    """
    _tem = tem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return a * np.exp(-np.abs(_tem) ** k)


def tpg(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
        st2_isref=True, a=10., k=1.):
    """
    Time-dependent Phase Goodness-of-Fit

    .. seealso:: [Kristekova2009]_, Eq.(16)

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference
    :param a: Maximum value of Goodness-of-Fit for perfect agreement
    :param k: sensitivity of Goodness-of-Fit to the misfit

    :return: time dependent Phase Goodness-of-Fit, type numpy.ndarray with
        shape (len(st1),) for single component data and (number of components,
        len(st1)) for multicomponent data
    """
    _tpm = tpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return a * (1 - np.abs(_tpm) ** k)


def feg(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
        st2_isref=True, a=10., k=1.):
    """
    Frequency-dependent Envelope Goodness-of-Fit

    .. seealso:: [Kristekova2009]_, Eq.(15)

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference
    :param a: Maximum value of Goodness-of-Fit for perfect agreement
    :param k: sensitivity of Goodness-of-Fit to the misfit

    :return: frequency dependent Envelope Goodness-of-Fit, type numpy.ndarray
        with shape (nf,) for single component data and (number of components,
        nf) for multicomponent data
    """
    _fem = fem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return a * np.exp(-np.abs(_fem) ** k)


def fpg(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
        st2_isref=True, a=10., k=1.):
    """
    Frequency-dependent Phase Goodness-of-Fit

    .. seealso:: [Kristekova2009]_, Eq.(16)

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference
    :param a: Maximum value of Goodness-of-Fit for perfect agreement
    :param k: sensitivity of Goodness-of-Fit to the misfit

    :return: frequency dependent Phase Goodness-of-Fit, type numpy.ndarray
        with shape (nf,) for single component data and (number of components,
        nf) for multicomponent data
    """
    _fpm = fpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return a * (1 - np.abs(_fpm) ** k)


def eg(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
       st2_isref=True, a=10., k=1.):
    """
    Single Valued Envelope Goodness-of-Fit

    .. seealso:: [Kristekova2009]_, Eq.(15)

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference
    :param a: Maximum value of Goodness-of-Fit for perfect agreement
    :param k: sensitivity of Goodness-of-Fit to the misfit

    :return: Single Valued Envelope Goodness-of-Fit
    """
    _em = em(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
             st2_isref=st2_isref)
    return a * np.exp(-np.abs(_em) ** k)


def pg(st1, st2, dt=0.01, fmin=1., fmax=10., nf=100, w0=6, norm='global',
       st2_isref=True, a=10., k=1.):
    """
    Single Valued Phase Goodness-of-Fit

    .. seealso:: [Kristekova2009]_, Eq.(16)

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference
    :param a: Maximum value of Goodness-of-Fit for perfect agreement
    :param k: sensitivity of Goodness-of-Fit to the misfit

    :return: Single Valued Phase Goodness-of-Fit
    """
    _pm = pm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
             st2_isref=st2_isref)
    return a * (1 - np.abs(_pm) ** k)


def plot_tf_misfits(st1, st2, dt=0.01, t0=0., fmin=1., fmax=10., nf=100, w0=6,
                    norm='global', st2_isref=True, left=0.1, bottom=0.1,
                    h_1=0.2, h_2=0.125, h_3=0.2, w_1=0.2, w_2=0.6, w_cb=0.01,
                    d_cb=0.0, show=True, plot_args=['k', 'r', 'b'], ylim=0.,
                    clim=0., cmap=obspy_divergent):
    """
    Plot all time frequency misfits and the time series in one plot (per
    component).

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param t0: starting time for plotting
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference
    :param left: plot distance from the left of the figure
    :param bottom: plot distance from the bottom of the figure
    :param h_1: height of the signal axes
    :param h_2: height of the TEM and TPM axes
    :param h_3: height of the TFEM and TFPM axes
    :param w_1: width of the FEM and FPM axes
    :param w_2: width of the TFEM, TFPM, signal etc. axes
    :param w_cb: width of the colorbar axes
    :param d_cb: distance of the colorbar axes to the other axes
    :param show: show figure or return
    :param plot_args: list of plot arguments passed to the signal 1/2 and
        TEM/TPM/FEM/FPM plots
    :param ylim: limits in misfit for TEM/TPM/FEM/FPM
    :param clim: limits of the colorbars
    :param cmap: colormap for TFEM/TFPM, either a string or
        matplotlib.cm.Colormap instance

    :return: If show is False, returns a matplotlib.pyplot.figure object
        (single component data) or a list of figure objects (multi component
        data)

    .. rubric:: Example

    For a signal with pure phase error

    .. seealso:: [Kristekova2006]_, Fig.(4)

    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> tmax = 6.
    >>> dt = 0.01
    >>> npts = int(tmax / dt + 1)
    >>> t = np.linspace(0., tmax, npts)
    >>> A1 = 4.
    >>> t1 = 2.
    >>> f1 = 2.
    >>> phi1 = 0.
    >>> phase_shift = 0.1
    >>> H1 = (np.sign(t - t1) + 1)/ 2
    >>> st1 = (A1 * (t - t1) * np.exp(-2*(t - t1)) *
    ...       np.cos(2. * np.pi * f1 * (t - t1) + phi1 * np.pi) * H1)
    >>> # Reference signal
    >>> st2 = st1.copy()
    >>> # Distorted signal:
    >>> # generate analytical signal (hilbert transform) and add phase shift
    >>> st1 = hilbert(st1)
    >>> st1 = np.real(np.abs(st1) * np.exp((np.angle(st1) +
    ...                                     phase_shift * np.pi) * 1j))
    >>> plot_tf_misfits(st1, st2, dt=dt, fmin=1., fmax=10.) # doctest: +SKIP

    .. plot::

        import numpy as np
        from scipy.signal import hilbert
        from obspy.signal.tf_misfit import plot_tf_misfits
        tmax = 6.
        dt = 0.01
        npts = int(tmax / dt + 1)
        t = np.linspace(0., tmax, npts)
        A1 = 4.
        t1 = 2.
        f1 = 2.
        phi1 = 0.
        phase_shift = 0.1
        H1 = (np.sign(t - t1) + 1)/ 2
        st1 = (A1 * (t - t1) * np.exp(-2*(t - t1)) *
              np.cos(2. * np.pi * f1 * (t - t1) + phi1 * np.pi) * H1)
        # Reference signal
        st2 = st1.copy()
        # Distorted signal:
        # generate analytical signal (hilbert transform) and add phase shift
        st1 = hilbert(st1)
        st1 = np.real(np.abs(st1) * np.exp((np.angle(st1) +
                                            phase_shift * np.pi) * 1j))
        plot_tf_misfits(st1, st2, dt=dt, fmin=1., fmax=10.)
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    npts = st1.shape[-1]
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts) + t0
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    # compute time frequency misfits
    _tfem = tfem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0,
                 norm=norm, st2_isref=st2_isref)
    _tem = tem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    _fem = fem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    _em = em(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
             st2_isref=st2_isref)
    _tfpm = tfpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0,
                 norm=norm, st2_isref=st2_isref)
    _tpm = tpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    _fpm = fpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    _pm = pm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
             st2_isref=st2_isref)

    if len(st1.shape) == 1:
        _tfem = _tfem.reshape((1, nf, npts))
        _tem = _tem.reshape((1, npts))
        _fem = _fem.reshape((1, nf))
        _em = _em.reshape((1, 1))
        _tfpm = _tfpm.reshape((1, nf, npts))
        _tpm = _tpm.reshape((1, npts))
        _fpm = _fpm.reshape((1, nf))
        _pm = _pm.reshape((1, 1))
        st1 = st1.reshape((1, npts))
        st2 = st2.reshape((1, npts))
        ntr = 1
    else:
        ntr = st1.shape[0]

    figs = []

    for itr in np.arange(ntr):
        fig = plt.figure()

        # plot signals
        ax_sig = fig.add_axes([left + w_1, bottom + h_2 + h_3, w_2, h_1])
        ax_sig.plot(t, st1[itr], plot_args[0])
        ax_sig.plot(t, st2[itr], plot_args[1])

        # plot TEM
        ax_tem = fig.add_axes([left + w_1, bottom + h_1 + h_2 + h_3, w_2, h_2])
        ax_tem.plot(t, _tem[itr], plot_args[2])

        # plot TFEM
        ax_tfem = fig.add_axes([left + w_1, bottom + h_1 + 2 * h_2 + h_3, w_2,
                                h_3])

        x, y = np.meshgrid(
            t, np.logspace(np.log10(fmin), np.log10(fmax),
                           _tfem[itr].shape[0]))
        img_tfem = ax_tfem.pcolormesh(x, y, _tfem[itr], cmap=cmap)
        img_tfem.set_rasterized(True)
        ax_tfem.set_yscale("log")
        ax_tfem.set_ylim(fmin, fmax)

        # plot FEM
        ax_fem = fig.add_axes([left, bottom + h_1 + 2 * h_2 + h_3, w_1, h_3])
        ax_fem.semilogy(_fem[itr], f, plot_args[2])
        ax_fem.set_ylim(fmin, fmax)

        # plot TPM
        ax_tpm = fig.add_axes([left + w_1, bottom, w_2, h_2])
        ax_tpm.plot(t, _tpm[itr], plot_args[2])

        # plot TFPM
        ax_tfpm = fig.add_axes([left + w_1, bottom + h_2, w_2, h_3])

        x, y = np.meshgrid(t, f)
        img_tfpm = ax_tfpm.pcolormesh(x, y, _tfpm[itr], cmap=cmap)
        img_tfpm.set_rasterized(True)
        ax_tfpm.set_yscale("log")
        ax_tfpm.set_ylim(f[0], f[-1])

        # add colorbars
        ax_cb_tfpm = fig.add_axes([left + w_1 + w_2 + d_cb + w_cb, bottom,
                                   w_cb, h_2 + h_3])
        fig.colorbar(img_tfpm, cax=ax_cb_tfpm)

        # plot FPM
        ax_fpm = fig.add_axes([left, bottom + h_2, w_1, h_3])
        ax_fpm.semilogy(_fpm[itr], f, plot_args[2])
        ax_fpm.set_ylim(fmin, fmax)

        # set limits
        ylim_sig = np.max([np.abs(st1).max(), np.abs(st2).max()]) * 1.1
        ax_sig.set_ylim(-ylim_sig, ylim_sig)

        if ylim == 0.:
            ylim = np.max([np.abs(_tem).max(), np.abs(_tpm).max(),
                           np.abs(_fem).max(), np.abs(_fpm).max()]) * 1.1

        ax_tem.set_ylim(-ylim, ylim)
        ax_fem.set_xlim(-ylim, ylim)
        ax_tpm.set_ylim(-ylim, ylim)
        ax_fpm.set_xlim(-ylim, ylim)

        ax_sig.set_xlim(t[0], t[-1])
        ax_tem.set_xlim(t[0], t[-1])
        ax_tpm.set_xlim(t[0], t[-1])

        if clim == 0.:
            clim = np.max([np.abs(_tfem).max(), np.abs(_tfpm).max()])

        img_tfpm.set_clim(-clim, clim)
        img_tfem.set_clim(-clim, clim)

        # add text box for EM + PM
        textstr = 'EM = %.2f\nPM = %.2f' % (_em[itr], _pm[itr])
        props = dict(boxstyle='round', facecolor='white')
        ax_sig.text(-0.3, 0.5, textstr, transform=ax_sig.transAxes,
                    verticalalignment='center', horizontalalignment='left',
                    bbox=props)

        ax_tpm.set_xlabel('time')
        ax_fem.set_ylabel('frequency')
        ax_fpm.set_ylabel('frequency')

        # add text boxes
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax_tfem.text(0.95, 0.85, 'TFEM', transform=ax_tfem.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=props)
        ax_tfpm.text(0.95, 0.85, 'TFPM', transform=ax_tfpm.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=props)
        ax_tem.text(0.95, 0.75, 'TEM', transform=ax_tem.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)
        ax_tpm.text(0.95, 0.75, 'TPM', transform=ax_tpm.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)
        ax_fem.text(0.9, 0.85, 'FEM', transform=ax_fem.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)
        ax_fpm.text(0.9, 0.85, 'FPM', transform=ax_fpm.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)

        # remove axis labels
        ax_tfpm.xaxis.set_major_formatter(NullFormatter())
        ax_tfem.xaxis.set_major_formatter(NullFormatter())
        ax_tem.xaxis.set_major_formatter(NullFormatter())
        ax_sig.xaxis.set_major_formatter(NullFormatter())
        ax_tfpm.yaxis.set_major_formatter(NullFormatter())
        ax_tfem.yaxis.set_major_formatter(NullFormatter())

        figs.append(fig)

    if show:
        plt.show()
    else:
        if ntr == 1:
            return figs[0]
        else:
            return figs


def plot_tf_gofs(st1, st2, dt=0.01, t0=0., fmin=1., fmax=10., nf=100, w0=6,
                 norm='global', st2_isref=True, a=10., k=1., left=0.1,
                 bottom=0.1, h_1=0.2, h_2=0.125, h_3=0.2, w_1=0.2, w_2=0.6,
                 w_cb=0.01, d_cb=0.0, show=True, plot_args=['k', 'r', 'b'],
                 ylim=0., clim=0., cmap=obspy_sequential):
    """
    Plot all time frequency Goodness-of-Fits and the time series in one plot
    (per component).

    :param st1: signal 1 of two signals to compare, type numpy.ndarray with
        shape (number of components, number of time samples) or (number of
        timesamples, ) for single component data
    :param st2: signal 2 of two signals to compare, type and shape as st1
    :param dt: time step between two samples in st1 and st2
    :param t0: starting time for plotting
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param norm: 'global' or 'local' normalization of the misfit
    :type st2_isref: bool
    :param st2_isref: True if st2 is a reference signal, False if none is a
        reference
    :param A: Maximum value of Goodness-of-Fit for perfect agreement
    :param k: sensitivity of Goodness-of-Fit to the misfit
    :param left: plot distance from the left of the figure
    :param bottom: plot distance from the bottom of the figure
    :param h_1: height of the signal axes
    :param h_2: height of the TEM and TPM axes
    :param h_3: height of the TFEM and TFPM axes
    :param w_1: width of the FEM and FPM axes
    :param w_2: width of the TFEM, TFPM, signal etc. axes
    :param w_cb: width of the colorbar axes
    :param d_cb: distance of the colorbar axes to the other axes
    :param show: show figure or return
    :param plot_args: list of plot arguments passed to the signal 1/2 and
        TEM/TPM/FEM/FPM plots
    :param ylim: limits in misfit for TEM/TPM/FEM/FPM
    :param clim: limits of the colorbars
    :param cmap: colormap for TFEM/TFPM, either a string or
        matplotlib.cm.Colormap instance

    :return: If show is False, returns a matplotlib.pyplot.figure object
        (single component data) or a list of figure objects (multi component
        data)

    .. rubric:: Example

    For a signal with pure amplitude error

    >>> import numpy as np
    >>> tmax = 6.
    >>> dt = 0.01
    >>> npts = int(tmax / dt + 1)
    >>> t = np.linspace(0., tmax, npts)
    >>> A1 = 4.
    >>> t1 = 2.
    >>> f1 = 2.
    >>> phi1 = 0.
    >>> phase_shift = 0.1
    >>> H1 = (np.sign(t - t1) + 1)/ 2
    >>> st1 = (A1 * (t - t1) * np.exp(-2*(t - t1)) *
    ...       np.cos(2. * np.pi * f1 * (t - t1) + phi1 * np.pi) * H1)
    >>> # Reference signal
    >>> st2 = st1.copy()
    >>> # Distorted signal:
    >>> st1 = st1 * 3.
    >>> plot_tf_gofs(st1, st2, dt=dt, fmin=1., fmax=10.) # doctest: +SKIP

    .. plot::

        import numpy as np
        from obspy.signal.tf_misfit import plot_tf_gofs
        tmax = 6.
        dt = 0.01
        npts = int(tmax / dt + 1)
        t = np.linspace(0., tmax, npts)
        A1 = 4.
        t1 = 2.
        f1 = 2.
        phi1 = 0.
        phase_shift = 0.1
        H1 = (np.sign(t - t1) + 1)/ 2
        st1 = (A1 * (t - t1) * np.exp(-2*(t - t1)) *
              np.cos(2. * np.pi * f1 * (t - t1) + phi1 * np.pi) * H1)
        # Reference signal
        st2 = st1.copy()
        # Distorted signal:
        st1 = st1 * 3.
        plot_tf_gofs(st1, st2, dt=dt, fmin=1., fmax=10.)
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    npts = st1.shape[-1]
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts) + t0
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    # compute time frequency misfits
    _tfeg = tfeg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0,
                 norm=norm, st2_isref=st2_isref, a=a, k=k)
    _teg = teg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref, a=a, k=k)
    _feg = feg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref, a=a, k=k)
    _eg = eg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
             st2_isref=st2_isref, a=a, k=k)
    _tfpg = tfpg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0,
                 norm=norm, st2_isref=st2_isref, a=a, k=k)
    _tpg = tpg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref, a=a, k=k)
    _fpg = fpg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref, a=a, k=k)
    _pg = pg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
             st2_isref=st2_isref, a=a, k=k)

    if len(st1.shape) == 1:
        _tfeg = _tfeg.reshape((1, nf, npts))
        _teg = _teg.reshape((1, npts))
        _feg = _feg.reshape((1, nf))
        _eg = _eg.reshape((1, 1))
        _tfpg = _tfpg.reshape((1, nf, npts))
        _tpg = _tpg.reshape((1, npts))
        _fpg = _fpg.reshape((1, nf))
        _pg = _pg.reshape((1, 1))
        st1 = st1.reshape((1, npts))
        st2 = st2.reshape((1, npts))
        ntr = 1
    else:
        ntr = st1.shape[0]

    figs = []

    for itr in np.arange(ntr):
        fig = plt.figure()

        # plot signals
        ax_sig = fig.add_axes([left + w_1, bottom + h_2 + h_3, w_2, h_1])
        ax_sig.plot(t, st1[itr], plot_args[0])
        ax_sig.plot(t, st2[itr], plot_args[1])

        # plot TEG
        ax_teg = fig.add_axes([left + w_1, bottom + h_1 + h_2 + h_3, w_2, h_2])
        ax_teg.plot(t, _teg[itr], plot_args[2])

        # plot TFEG
        ax_tfeg = fig.add_axes([left + w_1, bottom + h_1 + 2 * h_2 + h_3, w_2,
                                h_3])

        x, y = np.meshgrid(
            t, np.logspace(np.log10(fmin), np.log10(fmax),
                           _tfeg[itr].shape[0]))
        img_tfeg = ax_tfeg.pcolormesh(x, y, _tfeg[itr], cmap=cmap)
        img_tfeg.set_rasterized(True)
        ax_tfeg.set_yscale("log")
        ax_tfeg.set_ylim(fmin, fmax)

        # plot FEG
        ax_feg = fig.add_axes([left, bottom + h_1 + 2 * h_2 + h_3, w_1, h_3])
        ax_feg.semilogy(_feg[itr], f, plot_args[2])
        ax_feg.set_ylim(fmin, fmax)

        # plot TPG
        ax_tpg = fig.add_axes([left + w_1, bottom, w_2, h_2])
        ax_tpg.plot(t, _tpg[itr], plot_args[2])

        # plot TFPG
        ax_tfpg = fig.add_axes([left + w_1, bottom + h_2, w_2, h_3])

        x, y = np.meshgrid(t, f)
        img_tfpg = ax_tfpg.pcolormesh(x, y, _tfpg[itr], cmap=cmap)
        img_tfpg.set_rasterized(True)
        ax_tfpg.set_yscale("log")
        ax_tfpg.set_ylim(f[0], f[-1])

        # add colorbars
        ax_cb_tfpg = fig.add_axes([left + w_1 + w_2 + d_cb + w_cb, bottom,
                                   w_cb, h_2 + h_3])
        fig.colorbar(img_tfpg, cax=ax_cb_tfpg)

        # plot FPG
        ax_fpg = fig.add_axes([left, bottom + h_2, w_1, h_3])
        ax_fpg.semilogy(_fpg[itr], f, plot_args[2])
        ax_fpg.set_ylim(fmin, fmax)

        # set limits
        ylim_sig = np.max([np.abs(st1).max(), np.abs(st2).max()]) * 1.1
        ax_sig.set_ylim(-ylim_sig, ylim_sig)

        if ylim == 0.:
            ylim = np.max([np.abs(_teg).max(), np.abs(_tpg).max(),
                           np.abs(_feg).max(), np.abs(_fpg).max()]) * 1.1

        ax_teg.set_ylim(0., ylim)
        ax_feg.set_xlim(0., ylim)
        ax_tpg.set_ylim(0., ylim)
        ax_fpg.set_xlim(0., ylim)

        ax_sig.set_xlim(t[0], t[-1])
        ax_teg.set_xlim(t[0], t[-1])
        ax_tpg.set_xlim(t[0], t[-1])

        if clim == 0.:
            clim = np.max([np.abs(_tfeg).max(), np.abs(_tfpg).max()])

        img_tfpg.set_clim(0., clim)
        img_tfeg.set_clim(0., clim)

        # add text box for EG + PG
        textstr = 'EG = %2.2f\nPG = %2.2f' % (_eg[itr], _pg[itr])
        props = dict(boxstyle='round', facecolor='white')
        ax_sig.text(-0.3, 0.5, textstr, transform=ax_sig.transAxes,
                    verticalalignment='center', horizontalalignment='left',
                    bbox=props)

        ax_tpg.set_xlabel('time')
        ax_feg.set_ylabel('frequency')
        ax_fpg.set_ylabel('frequency')

        # add text boxes
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax_tfeg.text(0.95, 0.85, 'TFEG', transform=ax_tfeg.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=props)
        ax_tfpg.text(0.95, 0.85, 'TFPG', transform=ax_tfpg.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=props)
        ax_teg.text(0.95, 0.75, 'TEG', transform=ax_teg.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)
        ax_tpg.text(0.95, 0.75, 'TPG', transform=ax_tpg.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)
        ax_feg.text(0.9, 0.85, 'FEG', transform=ax_feg.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)
        ax_fpg.text(0.9, 0.85, 'FPG', transform=ax_fpg.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)

        # remove axis labels
        ax_tfpg.xaxis.set_major_formatter(NullFormatter())
        ax_tfeg.xaxis.set_major_formatter(NullFormatter())
        ax_teg.xaxis.set_major_formatter(NullFormatter())
        ax_sig.xaxis.set_major_formatter(NullFormatter())
        ax_tfpg.yaxis.set_major_formatter(NullFormatter())
        ax_tfeg.yaxis.set_major_formatter(NullFormatter())

        figs.append(fig)

    if show:
        plt.show()
    else:
        if ntr == 1:
            return figs[0]
        else:
            return figs


def plot_tfr(st, dt=0.01, t0=0., fmin=1., fmax=10., nf=100, w0=6, left=0.1,
             bottom=0.1, h_1=0.2, h_2=0.6, w_1=0.2, w_2=0.6, w_cb=0.01,
             d_cb=0.0, show=True, plot_args=['k', 'k'], clim=0.0,
             cmap=obspy_sequential, mode='absolute', fft_zero_pad_fac=0):
    """
    Plot time frequency representation, spectrum and time series of the signal.

    :param st: signal, type numpy.ndarray with shape (number of components,
        number of time samples) or (number of timesamples, ) for single
        component data
    :param dt: time step between two samples in st
    :param t0: starting time for plotting
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param left: plot distance from the left of the figure
    :param bottom: plot distance from the bottom of the figure
    :param h_1: height of the signal axis
    :param h_2: height of the TFR/spectrum axis
    :param w_1: width of the spectrum axis
    :param w_2: width of the TFR/signal axes
    :param w_cb: width of the colorbar axes
    :param d_cb: distance of the colorbar axes to the other axes
    :param show: show figure or return
    :param plot_args: list of plot arguments passed to the signal and spectrum
        plots
    :param clim: limits of the colorbars
    :param cmap: colormap for TFEM/TFPM, either a string or
        matplotlib.cm.Colormap instance
    :param mode: 'absolute' for absolute value of TFR, 'power' for ``|TFR|^2``
    :param fft_zero_pad_fac: integer, if > 0, the signal is zero padded to
        ``nfft = next_pow_2(len(st)) * fft_zero_pad_fac`` to get smoother
        spectrum in the low frequencies (has no effect on the TFR and might
        make demeaning/tapering necessary to avoid artifacts)

    :return: If show is False, returns a matplotlib.pyplot.figure object
        (single component data) or a list of figure objects (multi component
        data)

    .. rubric:: Example

    >>> from obspy import read
    >>> tr = read("https://examples.obspy.org/a02i.2008.240.mseed")[0]
    >>> plot_tfr(tr.data, dt=tr.stats.delta, fmin=.01, # doctest: +SKIP
    ...         fmax=50., w0=8., nf=64, fft_zero_pad_fac=4)

    .. plot::

        from obspy.signal.tf_misfit import plot_tfr
        from obspy import read
        tr = read("https://examples.obspy.org/a02i.2008.240.mseed")[0]
        plot_tfr(tr.data, dt=tr.stats.delta, fmin=.01,
                fmax=50., w0=8., nf=64, fft_zero_pad_fac=4)
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    npts = st.shape[-1]
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts) + t0

    if fft_zero_pad_fac == 0:
        nfft = npts
    else:
        nfft = util.next_pow_2(npts) * fft_zero_pad_fac

    f_lin = np.linspace(0, 0.5 / dt, nfft // 2 + 1)

    if len(st.shape) == 1:
        _w = np.zeros((1, nf, npts), dtype=np.complex)
        _w[0] = cwt(st, dt, w0, fmin, fmax, nf)
        ntr = 1

        spec = np.zeros((1, nfft // 2 + 1), dtype=np.complex)
        spec[0] = np.fft.rfft(st, n=nfft) * dt

        st = st.reshape((1, npts))
    else:
        _w = np.zeros((st.shape[0], nf, npts), dtype=np.complex)
        spec = np.zeros((st.shape[0], nfft // 2 + 1), dtype=np.complex)

        for i in np.arange(st.shape[0]):
            _w[i] = cwt(st[i], dt, w0, fmin, fmax, nf)
            spec[i] = np.fft.rfft(st[i], n=nfft) * dt

        ntr = st.shape[0]

    if mode == 'absolute':
        _tfr = np.abs(_w)
        spec = np.abs(spec)
    elif mode == 'power':
        _tfr = np.abs(_w) ** 2
        spec = np.abs(spec) ** 2
    else:
        raise ValueError('mode "' + mode + '" not defined!')

    figs = []

    for itr in np.arange(ntr):
        fig = plt.figure()

        # plot signals
        ax_sig = fig.add_axes([left + w_1, bottom, w_2, h_1])
        ax_sig.plot(t, st[itr], plot_args[0])

        # plot TFR
        ax_tfr = fig.add_axes([left + w_1, bottom + h_1, w_2, h_2])

        x, y = np.meshgrid(
            t, np.logspace(np.log10(fmin), np.log10(fmax),
                           _tfr[itr].shape[0]))
        img_tfr = ax_tfr.pcolormesh(x, y, _tfr[itr], cmap=cmap)
        img_tfr.set_rasterized(True)
        ax_tfr.set_yscale("log")
        ax_tfr.set_ylim(fmin, fmax)
        ax_tfr.set_xlim(t[0], t[-1])

        # plot spectrum
        ax_spec = fig.add_axes([left, bottom + h_1, w_1, h_2])
        ax_spec.semilogy(spec[itr], f_lin, plot_args[1])

        # add colorbars
        ax_cb_tfr = fig.add_axes([left + w_1 + w_2 + d_cb + w_cb, bottom +
                                  h_1, w_cb, h_2])
        fig.colorbar(img_tfr, cax=ax_cb_tfr)

        # set limits
        ax_sig.set_ylim(st.min() * 1.1, st.max() * 1.1)
        ax_sig.set_xlim(t[0], t[-1])

        xlim = spec.max() * 1.1

        ax_spec.set_xlim(xlim, 0.)
        ax_spec.set_ylim(fmin, fmax)

        if clim == 0.:
            clim = _tfr.max()

        img_tfr.set_clim(0., clim)

        ax_sig.set_xlabel('time')
        ax_spec.set_ylabel('frequency')

        # remove axis labels
        ax_tfr.xaxis.set_major_formatter(NullFormatter())
        ax_tfr.yaxis.set_major_formatter(NullFormatter())

        figs.append(fig)

    if show:
        plt.show()
    else:
        if ntr == 1:
            return figs[0]
        else:
            return figs


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
