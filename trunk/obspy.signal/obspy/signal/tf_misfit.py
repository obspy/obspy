#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: tf_misfit.py
#  Purpose: Various Time Frequency Misfit Functions
#   Author: Martin van Driel
#    Email: vandriel@sed.ethz.ch
#
# Copyright (C) 2012 Martin van Driel
#---------------------------------------------------------------------
"""
Various Time Frequency Misfit Functions based on Kristekova et. al. (2006) and
Kristekova et. al. (2009).

.. seealso:: [Kristekova2006]_
.. seealso:: [Kristekova2009]_

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from obspy.signal import util
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.colors import LinearSegmentedColormap


def cwt(st, dt, w0, fmin, fmax, nf=100., wl='morlet'):
    """
    Continuous Wavelet Transformation in the Frequency Domain.

    .. seealso:: [Kristekova2006]_, eq. (4)

    :param st: time dependent signal.
    :param dt: time step between two samples in st
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param f: frequency discretization, type numpy.ndarray.
    :param wl: wavelet to use, for now only 'morlet' is implemented

    :return: time frequency representation of st, type numpy.ndarray of complex
        values.
    """
    npts = len(st)
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts)
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    cwt = np.empty((npts, nf)) * 0j

    if wl == 'morlet':
        psi = lambda t: np.pi ** (-.25) * np.exp(1j * w0 * t) * \
            np.exp(-t ** 2 / 2.)
        scale = lambda f: w0 / (2 * np.pi * f)
    else:
        raise ValueError('wavelet type "' + wl + '" not defined!')

    nfft = util.nextpow2(len(st)) * 2
    sf = np.fft.fft(st, n=nfft)

    for n, _f in enumerate(f):
        a = scale(_f)
        # time shift necessary, because wavelet is defined around t = 0
        psih = psi(-1 * (t - t[-1] / 2.) / a).conjugate() / np.abs(a) ** .5
        psihf = np.fft.fft(psih, n=nfft)
        tminin = int(t[-1] / 2. / (t[1] - t[0]))
        cwt[:, n] = np.fft.ifft(psihf * sf)[tminin:tminin + t.shape[0]] * \
            (t[1] - t[0])
    return cwt.T


def tfem(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference

    :return: time frequency representation of Envelope Misfit,
        type numpy.ndarray with shape (nf, len(st1)) for single component data
        and (number of components, nf, len(st1)) for multicomponent data
    """
    if len(st1.shape) == 1:
        W1 = np.empty((1, nf, st1.shape[0])) * 0j
        W2 = np.empty((1, nf, st1.shape[0])) * 0j

        W1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        W2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        W1 = np.empty((st1.shape[0], nf, st1.shape[1])) * 0j
        W2 = np.empty((st2.shape[0], nf, st2.shape[1])) * 0j

        for i in np.arange(st1.shape[0]):
            W1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            W2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        Ar = np.abs(W2)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    TFEM = (np.abs(W1) - np.abs(W2))

    if norm == 'global':
        if len(st1.shape) == 1:
            return  TFEM[0] / np.max(Ar)
        else:
            return  TFEM / np.max(Ar)
    elif norm == 'local':
        if len(st1.shape) == 1:
            return  TFEM[0] / Ar[0]
        else:
            return  TFEM / Ar


def tfpm(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference

    :return: time frequency representation of Phase Misfit,
        type numpy.ndarray with shape (nf, len(st1)) for single component data
        and (number of components, nf, len(st1)) for multicomponent data
    """
    if len(st1.shape) == 1:
        W1 = np.empty((1, nf, st1.shape[0])) * 0j
        W2 = np.empty((1, nf, st1.shape[0])) * 0j

        W1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        W2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        W1 = np.empty((st1.shape[0], nf, st1.shape[1])) * 0j
        W2 = np.empty((st2.shape[0], nf, st2.shape[1])) * 0j

        for i in np.arange(st1.shape[0]):
            W1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            W2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        Ar = np.abs(W2)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    TFPM = np.angle(W1 / W2) / np.pi

    if norm == 'global':
        if len(st1.shape) == 1:
            return Ar[0] * TFPM[0] / np.max(Ar)
        else:
            return Ar * TFPM / np.max(Ar)
    elif norm == 'local':
        if len(st1.shape) == 1:
            return TFPM[0]
        else:
            return TFPM


def tem(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference

    :return: Time-dependent Envelope Misfit, type numpy.ndarray with shape
        (len(st1),) for single component data and (number of components,
        len(st1)) for multicomponent data
    """
    if len(st1.shape) == 1:
        W1 = np.empty((1, nf, st1.shape[0])) * 0j
        W2 = np.empty((1, nf, st1.shape[0])) * 0j

        W1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        W2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        W1 = np.empty((st1.shape[0], nf, st1.shape[1])) * 0j
        W2 = np.empty((st2.shape[0], nf, st2.shape[1])) * 0j

        for i in np.arange(st1.shape[0]):
            W1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            W2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        Ar = np.abs(W2)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    TEM = np.sum((np.abs(W1) - np.abs(W2)), axis=1)

    if norm == 'global':
        if len(st1.shape) == 1:
            return TEM[0] / np.max(np.sum(Ar, axis=1))
        else:
            return TEM / np.max(np.sum(Ar, axis=1))
    elif norm == 'local':
        if len(st1.shape) == 1:
            return TEM[0] / np.sum(Ar, axis=1)[0]
        else:
            return TEM / np.sum(Ar, axis=1)


def tpm(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference

    :return: Time-dependent Phase Misfit, type numpy.ndarray with shape
        (len(st1),) for single component data and (number of components,
        len(st1)) for multicomponent data
    """
    if len(st1.shape) == 1:
        W1 = np.empty((1, nf, st1.shape[0])) * 0j
        W2 = np.empty((1, nf, st1.shape[0])) * 0j

        W1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        W2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        W1 = np.empty((st1.shape[0], nf, st1.shape[1])) * 0j
        W2 = np.empty((st2.shape[0], nf, st2.shape[1])) * 0j

        for i in np.arange(st1.shape[0]):
            W1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            W2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        Ar = np.abs(W2)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W2)
        else:
            Ar = np.abs(W1)

    TPM = np.angle(W1 / W2) / np.pi
    TPM = np.sum(Ar * TPM, axis=1)

    if norm == 'global':
        if len(st1.shape) == 1:
            return TPM[0] / np.max(np.sum(Ar, axis=1))
        else:
            return TPM / np.max(np.sum(Ar, axis=1))
    elif norm == 'local':
        if len(st1.shape) == 1:
            return TPM[0] / np.sum(Ar, axis=1)[0]
        else:
            return TPM / np.sum(Ar, axis=1)


def fem(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference

    :return: Frequency-dependent Envelope Misfit, type numpy.ndarray with shape
        (nf,) for single component data and (number of components, nf) for
        multicomponent data
    """
    if len(st1.shape) == 1:
        W1 = np.empty((1, nf, st1.shape[0])) * 0j
        W2 = np.empty((1, nf, st1.shape[0])) * 0j

        W1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        W2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        W1 = np.empty((st1.shape[0], nf, st1.shape[1])) * 0j
        W2 = np.empty((st2.shape[0], nf, st2.shape[1])) * 0j

        for i in np.arange(st1.shape[0]):
            W1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            W2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        Ar = np.abs(W2)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    TEM = np.abs(W1) - np.abs(W2)
    TEM = np.sum(TEM, axis=2)

    if norm == 'global':
        if len(st1.shape) == 1:
            return TEM[0] / np.max(np.sum(Ar, axis=2))
        else:
            return TEM / np.max(np.sum(Ar, axis=2))
    elif norm == 'local':
        if len(st1.shape) == 1:
            return TEM[0] / np.sum(Ar, axis=2)[0]
        else:
            return TEM / np.sum(Ar, axis=2)


def fpm(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference

    :return: Frequency-dependent Phase Misfit, type numpy.ndarray with shape
        (nf,) for single component data and (number of components, nf) for
        multicomponent data
    """
    if len(st1.shape) == 1:
        W1 = np.empty((1, nf, st1.shape[0])) * 0j
        W2 = np.empty((1, nf, st1.shape[0])) * 0j

        W1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        W2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        W1 = np.empty((st1.shape[0], nf, st1.shape[1])) * 0j
        W2 = np.empty((st2.shape[0], nf, st2.shape[1])) * 0j

        for i in np.arange(st1.shape[0]):
            W1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            W2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        Ar = np.abs(W2)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    TPM = np.angle(W1 / W2) / np.pi
    TPM = np.sum(Ar * TPM, axis=2)

    if norm == 'global':
        if len(st1.shape) == 1:
            return TPM[0] / np.max(np.sum(Ar, axis=2))
        else:
            return TPM / np.max(np.sum(Ar, axis=2))
    elif norm == 'local':
        if len(st1.shape) == 1:
            return TPM[0] / np.sum(Ar, axis=2)[0]
        else:
            return TPM / np.sum(Ar, axis=2)


def em(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference

    :return: Single Valued Envelope Misfit
    """
    if len(st1.shape) == 1:
        W1 = np.empty((1, nf, st1.shape[0])) * 0j
        W2 = np.empty((1, nf, st1.shape[0])) * 0j

        W1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        W2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        W1 = np.empty((st1.shape[0], nf, st1.shape[1])) * 0j
        W2 = np.empty((st2.shape[0], nf, st2.shape[1])) * 0j

        for i in np.arange(st1.shape[0]):
            W1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            W2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        Ar = np.abs(W2)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    EM = (np.sum(np.sum((np.abs(W1) - np.abs(W2)) ** 2, axis=2), axis=1)) ** .5

    if norm == 'global':
        if len(st1.shape) == 1:
            return EM[0] / (np.sum(Ar ** 2)) ** .5
        else:
            return EM / ((np.sum(np.sum(Ar ** 2, axis=2), axis=1)) ** .5).max()
    elif norm == 'local':
        if len(st1.shape) == 1:
            return EM[0] / (np.sum(Ar ** 2)) ** .5
        else:
            return EM / (np.sum(np.sum(Ar ** 2, axis=2), axis=1)) ** .5


def pm(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference

    :return: Single Valued Phase Misfit
    """
    if len(st1.shape) == 1:
        W1 = np.empty((1, nf, st1.shape[0])) * 0j
        W2 = np.empty((1, nf, st1.shape[0])) * 0j

        W1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        W2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        W1 = np.empty((st1.shape[0], nf, st1.shape[1])) * 0j
        W2 = np.empty((st2.shape[0], nf, st2.shape[1])) * 0j

        for i in np.arange(st1.shape[0]):
            W1[i] = cwt(st1[i], dt, w0, fmin, fmax, nf)
            W2[i] = cwt(st2[i], dt, w0, fmin, fmax, nf)

    if st2_isref:
        Ar = np.abs(W2)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    PM = np.angle(W1 / W2) / np.pi

    PM = (np.sum(np.sum((Ar * PM) ** 2, axis=2), axis=1)) ** .5

    if norm == 'global':
        if len(st1.shape) == 1:
            return PM[0] / (np.sum(Ar ** 2)) ** .5
        else:
            return PM / ((np.sum(np.sum(Ar ** 2, axis=2), axis=1)) ** .5).max()
    elif norm == 'local':
        if len(st1.shape) == 1:
            return PM[0] / (np.sum(Ar ** 2)) ** .5
        else:
            return PM / (np.sum(np.sum(Ar ** 2, axis=2), axis=1)) ** .5


def tfeg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
         st2_isref=True, A=10., k=1.):
    """
    Time Frequency Envelope Goodness-Of-Fit

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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference
    :param A: Maximum value of Goodness-Of-Fit for perfect agreement
    :param k: sensitivity of Goodness-Of-Fit to the misfit

    :return: time frequency representation of Envelope Goodness-Of-Fit,
        type numpy.ndarray with shape (nf, len(st1)) for single component data
        and (number of components, nf, len(st1)) for multicomponent data
    """
    TFEM = tfem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
                st2_isref=st2_isref)
    return A * np.exp(-np.abs(TFEM) ** k)


def tfpg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
         st2_isref=True, A=10., k=1.):
    """
    Time Frequency Phase Goodness-Of-Fit

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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference
    :param A: Maximum value of Goodness-Of-Fit for perfect agreement
    :param k: sensitivity of Goodness-Of-Fit to the misfit

    :return: time frequency representation of Phase Goodness-Of-Fit,
        type numpy.ndarray with shape (nf, len(st1)) for single component data
        and (number of components, nf, len(st1)) for multicomponent data
    """
    TFPM = tfpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
                st2_isref=st2_isref)
    return A * (1 - np.abs(TFPM) ** k)


def teg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
        st2_isref=True, A=10., k=1.):
    """
    Time Dependent Envelope Goodness-Of-Fit

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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference
    :param A: Maximum value of Goodness-Of-Fit for perfect agreement
    :param k: sensitivity of Goodness-Of-Fit to the misfit

    :return: time dependent Envelope Goodness-Of-Fit, type numpy.ndarray with
        shape (len(st1),) for single component data and (number of components,
        len(st1)) for multicomponent data
    """
    TEM = tem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return A * np.exp(-np.abs(TEM) ** k)


def tpg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
        st2_isref=True, A=10., k=1.):
    """
    Time Dependent Phase Goodness-Of-Fit

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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference
    :param A: Maximum value of Goodness-Of-Fit for perfect agreement
    :param k: sensitivity of Goodness-Of-Fit to the misfit

    :return: time dependent Phase Goodness-Of-Fit, type numpy.ndarray with
        shape (len(st1),) for single component data and (number of components,
        len(st1)) for multicomponent data
    """
    TPM = tpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return A * (1 - np.abs(TPM) ** k)


def feg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
        st2_isref=True, A=10., k=1.):
    """
    Frequency Dependent Envelope Goodness-Of-Fit

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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference
    :param A: Maximum value of Goodness-Of-Fit for perfect agreement
    :param k: sensitivity of Goodness-Of-Fit to the misfit

    :return: frequency dependent Envelope Goodness-Of-Fit, type numpy.ndarray
        with shape (nf,) for single component data and (number of components,
        nf) for multicomponent data
    """
    FEM = fem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return A * np.exp(-np.abs(FEM) ** k)


def fpg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
        st2_isref=True, A=10., k=1.):
    """
    Frequency Dependent Phase Goodness-Of-Fit

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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference
    :param A: Maximum value of Goodness-Of-Fit for perfect agreement
    :param k: sensitivity of Goodness-Of-Fit to the misfit

    :return: frequency dependent Phase Goodness-Of-Fit, type numpy.ndarray
        with shape (nf,) for single component data and (number of components,
        nf) for multicomponent data
    """
    FPM = fpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return A * (1 - np.abs(FPM) ** k)


def eg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
       st2_isref=True, A=10., k=1.):
    """
    Single Valued Envelope Goodness-Of-Fit

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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference
    :param A: Maximum value of Goodness-Of-Fit for perfect agreement
    :param k: sensitivity of Goodness-Of-Fit to the misfit

    :return: Single Valued Envelope Goodness-Of-Fit
    """
    EM = em(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return A * np.exp(-np.abs(EM) ** k)


def pg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
       st2_isref=True, A=10., k=1.):
    """
    Single Valued Phase Goodness-Of-Fit

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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference
    :param A: Maximum value of Goodness-Of-Fit for perfect agreement
    :param k: sensitivity of Goodness-Of-Fit to the misfit

    :return: Single Valued Phase Goodness-Of-Fit
    """
    PM = pm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return A * (1 - np.abs(PM) ** k)


def plot_tf_misfits(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6,
                    norm='global', st2_isref=True, left=0.1, bottom=0.1,
                    h_1=0.2, h_2=0.125, h_3=0.2, w_1=0.2, w_2=0.6, w_cb=0.01,
                    d_cb=0.0, show=True, plot_args=['k', 'r', 'b'], ylim=0.,
                    clim=0., cmap=None):
    """
    Plot all timefrequency misfits in one plot.

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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference
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

    :return: If show is False, returns a maplotlib.pyplot.figure object (single
        component data) or a list of figure objects (multi component data)
    """
    npts = st1.shape[-1]
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts)
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    if cmap == None:
        CDICT_TFM = {'red': ((0.0, 0.0, 0.0),
                             (0.2, 0.0, 0.0),
                             (0.4, 0.0, 0.0),
                             (0.5, 1.0, 1.0),
                             (0.6, 1.0, 1.0),
                             (0.8, 1.0, 1.0),
                             (1.0, 0.2, 0.2)),
                     'green': ((0.0, 0.0, 0.0),
                               (0.2, 0.0, 0.0),
                               (0.4, 1.0, 1.0),
                               (0.5, 1.0, 1.0),
                               (0.6, 1.0, 1.0),
                               (0.8, 0.0, 0.0),
                               (1.0, 0.0, 0.0)),
                     'blue': ((0.0, 0.2, 0.2),
                              (0.2, 1.0, 1.0),
                              (0.4, 1.0, 1.0),
                              (0.5, 1.0, 1.0),
                              (0.6, 0.0, 0.0),
                              (0.8, 0.0, 0.0),
                              (1.0, 0.0, 0.0))}

        cmap = LinearSegmentedColormap('cmap_tfm', CDICT_TFM, 1024)

    # compute time frequency misfits
    TFEM = tfem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
                st2_isref=st2_isref)
    TEM = tem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref)
    FEM = fem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref)
    EM = em(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
            st2_isref=st2_isref)
    TFPM = tfpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref)
    TPM = tpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref)
    FPM = fpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref)
    PM = pm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
            st2_isref=st2_isref)

    if len(st1.shape) == 1:
        TFEM = TFEM.reshape((1, nf, npts))
        TEM = TEM.reshape((1, npts))
        FEM = FEM.reshape((1, nf))
        EM = EM.reshape((1, 1))
        TFPM = TFPM.reshape((1, nf, npts))
        TPM = TPM.reshape((1, npts))
        FPM = FPM.reshape((1, nf))
        PM = PM.reshape((1, 1))
        st1 = st1.reshape((1, npts))
        st2 = st2.reshape((1, npts))
        ntr = 1
    else:
        ntr = st1.shape[0]

    figs = []

    for itr in np.arange(ntr):
        # Plot S1 and S1t and TFEM + TFPM misfits
        fig = plt.figure()

        # plot signals
        ax_sig = fig.add_axes([left + w_1, bottom + h_2 + h_3, w_2, h_1])
        ax_sig.plot(t, st1[itr], plot_args[0])
        ax_sig.plot(t, st2[itr], plot_args[1])

        # plot TEM
        ax_TEM = fig.add_axes([left + w_1, bottom + h_1 + h_2 + h_3, w_2, h_2])
        ax_TEM.plot(t, TEM[itr], plot_args[2])

        # plot TFEM
        ax_TFEM = fig.add_axes([left + w_1, bottom + h_1 + 2 * h_2 + h_3, w_2,
                                h_3])
        img_TFEM = ax_TFEM.imshow(TFEM[itr], interpolation='nearest',
            cmap=cmap, extent=[t[0], t[-1], fmin, fmax], aspect='auto',
            origin='lower')
        ax_TFEM.set_yscale('log')

        # plot FEM
        ax_FEM = fig.add_axes([left, bottom + h_1 + 2 * h_2 + h_3, w_1, h_3])
        ax_FEM.semilogy(FEM[itr], f, plot_args[2])
        ax_FEM.set_ylim(fmin, fmax)

        # plot TPM
        ax_TPM = fig.add_axes([left + w_1, bottom, w_2, h_2])
        ax_TPM.plot(t, TPM[itr], plot_args[2])

        # plot TFPM
        ax_TFPM = fig.add_axes([left + w_1, bottom + h_2, w_2, h_3])
        img_TFPM = ax_TFPM.imshow(TFPM[itr], interpolation='nearest',
            cmap=cmap, extent=[t[0], t[-1], f[0], f[-1]], aspect='auto',
            origin='lower')
        ax_TFPM.set_yscale('log')

        # add colorbars
        ax_cb_TFPM = fig.add_axes([left + w_1 + w_2 + d_cb + w_cb, bottom,
                                   w_cb, h_2 + h_3])
        fig.colorbar(img_TFPM, cax=ax_cb_TFPM)

        # plot FPM
        ax_FPM = fig.add_axes([left, bottom + h_2, w_1, h_3])
        ax_FPM.semilogy(FPM[itr], f, plot_args[2])
        ax_FPM.set_ylim(fmin, fmax)

        # set limits
        ylim_sig = np.max([np.abs(st1).max(), np.abs(st2).max()]) * 1.1
        ax_sig.set_ylim(-ylim_sig, ylim_sig)

        if ylim == 0.:
            ylim = np.max([np.abs(TEM).max(), np.abs(TPM).max(),
                           np.abs(FEM).max(), np.abs(FPM).max()]) * 1.1

        ax_TEM.set_ylim(-ylim, ylim)
        ax_FEM.set_xlim(-ylim, ylim)
        ax_TPM.set_ylim(-ylim, ylim)
        ax_FPM.set_xlim(-ylim, ylim)

        if clim == 0.:
            clim = np.max([np.abs(TFEM).max(), np.abs(TFPM).max()])

        img_TFPM.set_clim(-clim, clim)
        img_TFEM.set_clim(-clim, clim)

        # add text box for EM + PM
        textstr = 'EM = %.2f\nPM = %.2f' % (EM[itr], PM[itr])
        props = dict(boxstyle='round', facecolor='white')
        ax_sig.text(-0.3, 0.5, textstr, transform=ax_sig.transAxes,
                verticalalignment='center', horizontalalignment='left',
                bbox=props)

        ax_TPM.set_xlabel('time')
        ax_FEM.set_ylabel('frequency')
        ax_FPM.set_ylabel('frequency')

        # add text boxes
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax_TFEM.text(0.95, 0.85, 'TFEM', transform=ax_TFEM.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)
        ax_TFPM.text(0.95, 0.85, 'TFPM', transform=ax_TFPM.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)
        ax_TEM.text(0.95, 0.75, 'TEM', transform=ax_TEM.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)
        ax_TPM.text(0.95, 0.75, 'TPM', transform=ax_TPM.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)
        ax_FEM.text(0.9, 0.85, 'FEM', transform=ax_FEM.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)
        ax_FPM.text(0.9, 0.85, 'FPM', transform=ax_FPM.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)

        # remove axis labels
        ax_TFPM.xaxis.set_major_formatter(NullFormatter())
        ax_TFEM.xaxis.set_major_formatter(NullFormatter())
        ax_TEM.xaxis.set_major_formatter(NullFormatter())
        ax_sig.xaxis.set_major_formatter(NullFormatter())
        ax_TFPM.yaxis.set_major_formatter(NullFormatter())
        ax_TFEM.yaxis.set_major_formatter(NullFormatter())

        figs.append(fig)

    if show:
        plt.show()
    else:
        if len(st1.shape) == 1:
            return figs[0]
        else:
            return figs


def plot_tf_gofs(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6,
                    norm='global', st2_isref=True, A=10., k=1., left=0.1,
                    bottom=0.1, h_1=0.2, h_2=0.125, h_3=0.2, w_1=0.2, w_2=0.6,
                    w_cb=0.01, d_cb=0.0, show=True, plot_args=['k', 'r', 'b'],
                    ylim=0., clim=0., cmap=None):
    """
    Plot all timefrequency Goodnes-Of-Fits its in one plot.

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
    :param st2_isref: Boolean, True if st2 is a reference signal, False if none
        is a reference
    :param A: Maximum value of Goodness-Of-Fit for perfect agreement
    :param k: sensitivity of Goodness-Of-Fit to the misfit
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

    :return: If show is False, returns a maplotlib.pyplot.figure object (single
        component data) or a list of figure objects (multi component data)
    """
    npts = st1.shape[-1]
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts)
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    if cmap == None:
        CDICT_GOF = {'red': ((0.0, 0.6, 0.6),
                             (0.4, 0.6, 1.0),
                             (0.6, 1.0, 1.0),
                             (0.8, 1.0, 1.0),
                             (1.0, 1.0, 1.0)),
                     'green': ((0.0, 0.0, 0.0),
                               (0.4, 0.0, 0.5),
                               (0.6, 0.5, 1.0),
                               (0.8, 1.0, 1.0),
                               (1.0, 1.0, 1.0)),
                     'blue': ((0.0, 0.0, 0.0),
                              (0.4, 0.0, 0.0),
                              (0.6, 0.0, 0.0),
                              (0.8, 0.0, 1.0),
                              (1.0, 1.0, 1.0))}

        cmap = LinearSegmentedColormap('cmap_gof', CDICT_GOF, 1024)

    # compute time frequency misfits
    TFEG = tfeg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
                st2_isref=st2_isref, A=A, k=k)
    TEG = teg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref, A=A, k=k)
    FEG = feg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref, A=A, k=k)
    EG = eg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
            st2_isref=st2_isref, A=A, k=k)
    TFPG = tfpg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref, A=A, k=k)
    TPG = tpg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref, A=A, k=k)
    FPG = fpg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref, A=A, k=k)
    PG = pg(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
            st2_isref=st2_isref, A=A, k=k)

    if len(st1.shape) == 1:
        TFEG = TFEG.reshape((1, nf, npts))
        TEG = TEG.reshape((1, npts))
        FEG = FEG.reshape((1, nf))
        EG = EG.reshape((1, 1))
        TFPG = TFPG.reshape((1, nf, npts))
        TPG = TPG.reshape((1, npts))
        FPG = FPG.reshape((1, nf))
        PG = PG.reshape((1, 1))
        st1 = st1.reshape((1, npts))
        st2 = st2.reshape((1, npts))
        ntr = 1
    else:
        ntr = st1.shape[0]

    figs = []

    for itr in np.arange(ntr):
        # Plot S1 and S1t and TFEG + TFPG misfits
        fig = plt.figure()

        # plot signals
        ax_sig = fig.add_axes([left + w_1, bottom + h_2 + h_3, w_2, h_1])
        ax_sig.plot(t, st1[itr], plot_args[0])
        ax_sig.plot(t, st2[itr], plot_args[1])

        # plot TEG
        ax_TEG = fig.add_axes([left + w_1, bottom + h_1 + h_2 + h_3, w_2, h_2])
        ax_TEG.plot(t, TEG[itr], plot_args[2])

        # plot TFEG
        ax_TFEG = fig.add_axes([left + w_1, bottom + h_1 + 2 * h_2 + h_3, w_2,
                                h_3])
        img_TFEG = ax_TFEG.imshow(TFEG[itr], interpolation='nearest',
            cmap=cmap, extent=[t[0], t[-1], fmin, fmax], aspect='auto',
            origin='lower')
        ax_TFEG.set_yscale('log')

        # plot FEG
        ax_FEG = fig.add_axes([left, bottom + h_1 + 2 * h_2 + h_3, w_1, h_3])
        ax_FEG.semilogy(FEG[itr], f, plot_args[2])
        ax_FEG.set_ylim(fmin, fmax)

        # plot TPG
        ax_TPG = fig.add_axes([left + w_1, bottom, w_2, h_2])
        ax_TPG.plot(t, TPG[itr], plot_args[2])

        # plot TFPG
        ax_TFPG = fig.add_axes([left + w_1, bottom + h_2, w_2, h_3])
        img_TFPG = ax_TFPG.imshow(TFPG[itr], interpolation='nearest',
            cmap=cmap, extent=[t[0], t[-1], f[0], f[-1]], aspect='auto',
            origin='lower')
        ax_TFPG.set_yscale('log')

        # add colorbars
        ax_cb_TFPG = fig.add_axes([left + w_1 + w_2 + d_cb + w_cb, bottom,
                                   w_cb, h_2 + h_3])
        fig.colorbar(img_TFPG, cax=ax_cb_TFPG)

        # plot FPG
        ax_FPG = fig.add_axes([left, bottom + h_2, w_1, h_3])
        ax_FPG.semilogy(FPG[itr], f, plot_args[2])
        ax_FPG.set_ylim(fmin, fmax)

        # set limits
        ylim_sig = np.max([np.abs(st1).max(), np.abs(st2).max()]) * 1.1
        ax_sig.set_ylim(-ylim_sig, ylim_sig)

        if ylim == 0.:
            ylim = np.max([np.abs(TEG).max(), np.abs(TPG).max(),
                           np.abs(FEG).max(), np.abs(FPG).max()]) * 1.1

        ax_TEG.set_ylim(0., ylim)
        ax_FEG.set_xlim(0., ylim)
        ax_TPG.set_ylim(0., ylim)
        ax_FPG.set_xlim(0., ylim)

        if clim == 0.:
            clim = np.max([np.abs(TFEG).max(), np.abs(TFPG).max()])

        img_TFPG.set_clim(0., clim)
        img_TFEG.set_clim(0., clim)

        # add text box for EG + PG
        textstr = 'EG = %2.2f\nPG = %2.2f' % (EG[itr], PG[itr])
        props = dict(boxstyle='round', facecolor='white')
        ax_sig.text(-0.3, 0.5, textstr, transform=ax_sig.transAxes,
                verticalalignment='center', horizontalalignment='left',
                bbox=props)

        ax_TPG.set_xlabel('time')
        ax_FEG.set_ylabel('frequency')
        ax_FPG.set_ylabel('frequency')

        # add text boxes
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax_TFEG.text(0.95, 0.85, 'TFEG', transform=ax_TFEG.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)
        ax_TFPG.text(0.95, 0.85, 'TFPG', transform=ax_TFPG.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)
        ax_TEG.text(0.95, 0.75, 'TEG', transform=ax_TEG.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)
        ax_TPG.text(0.95, 0.75, 'TPG', transform=ax_TPG.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)
        ax_FEG.text(0.9, 0.85, 'FEG', transform=ax_FEG.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)
        ax_FPG.text(0.9, 0.85, 'FPG', transform=ax_FPG.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props)

        # remove axis labels
        ax_TFPG.xaxis.set_major_formatter(NullFormatter())
        ax_TFEG.xaxis.set_major_formatter(NullFormatter())
        ax_TEG.xaxis.set_major_formatter(NullFormatter())
        ax_sig.xaxis.set_major_formatter(NullFormatter())
        ax_TFPG.yaxis.set_major_formatter(NullFormatter())
        ax_TFEG.yaxis.set_major_formatter(NullFormatter())

        figs.append(fig)

    if show:
        plt.show()
    else:
        if len(st1.shape) == 1:
            return figs[0]
        else:
            return figs
