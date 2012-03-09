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
from obspy.signal import util, cosTaper
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, MultipleLocator, NullLocator


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
        psi = lambda t : np.pi**(-.25) * np.exp(1j * w0 * t) * \
            np.exp(-t**2 / 2.)
        scale = lambda f : w0 / (2 * np.pi * f)
    else:
        raise ValueError('wavelet type "' + wl + '" not defined!')

    nfft = util.nextpow2(len(st)) * 2
    sf = np.fft.fft(st, n=nfft)

    for n, _f in enumerate(f):
        a = scale(_f)
        # time shift necessary, because wavelet is defined around t = 0
        psih = psi(-1*(t - t[-1]/2.)/a).conjugate() / np.abs(a)**.5
        psihf = np.fft.fft(psih, n=nfft)
        tminin = int(t[-1]/2. / (t[1] - t[0]))
        cwt[:, n] = np.fft.ifft(psihf * sf)[tminin:tminin+t.shape[0]] * \
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
        type numpy.ndarray.
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
        type numpy.ndarray.
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

    :return: Time-dependent Envelope Misfit, type numpy.ndarray.
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

    :return: Time-dependent Phase Misfit, type numpy.ndarray.
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

    :return: Frequency-dependent Envelope Misfit, type numpy.ndarray.
    """
    if len(st1.shape) == 1:
        npts = st1.shape[0]
        W1 = np.empty((1, nf, st1.shape[0])) * 0j
        W2 = np.empty((1, nf, st1.shape[0])) * 0j
        
        W1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        W2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        npts = st1.shape[1]
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

    :return: Frequency-dependent Phase Misfit, type numpy.ndarray.
    """
    if len(st1.shape) == 1:
        npts = st1.shape[0]
        W1 = np.empty((1, nf, st1.shape[0])) * 0j
        W2 = np.empty((1, nf, st1.shape[0])) * 0j
        
        W1[0] = cwt(st1, dt, w0, fmin, fmax, nf)
        W2[0] = cwt(st2, dt, w0, fmin, fmax, nf)
    else:
        npts = st1.shape[1]
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

    EM = (np.sum(np.sum((np.abs(W1) - np.abs(W2))**2, axis=2), axis=1))**.5
   
    if norm == 'global':
        if len(st1.shape) == 1:
            return EM[0] / (np.sum(Ar**2))**.5
        else:
            return EM / ((np.sum(np.sum(Ar**2, axis=2), axis=1))**.5).max()
    elif norm == 'local':
        if len(st1.shape) == 1:
            return EM[0] / (np.sum(Ar**2))**.5
        else:
            return EM / (np.sum(Ar**2))**.5


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

    PM = (np.sum(np.sum((Ar * PM)**2, axis=2), axis=1))**.5

    if norm == 'global':
        if len(st1.shape) == 1:
            return PM[0] / (np.sum(Ar**2))**.5
        else:
            return PM / ((np.sum(np.sum(Ar**2, axis=1), axis=2))**.5).max()
    elif norm == 'local':
        if len(st1.shape) == 1:
            return PM[0] / (np.sum(Ar**2))**.5
        else:
            return PM / (np.sum(np.sum(Ar**2, axis=2), axis=1))**.5



def tfeg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st2_isref=True, A=10., k=1.):
    """
    Time Frequency Envelope Goodness-Of_Fit

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
        type numpy.ndarray.
    """
    TFEM = tfem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
                st2_isref=st2_isref)
    return A * np.exp(-np.abs(TFEM)**k)


def tfpg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st2_isref=True, A=10., k=1.):
    """
    Time Frequency Phase Goodness-Of_Fit

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
        type numpy.ndarray.
    """
    TFPM = tfpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
                st2_isref=st2_isref)
    return A * (1 - np.abs(TFPM)**k)


def teg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st2_isref=True, A=10., k=1.):
    """
    Time Dependent Envelope Goodness-Of_Fit

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

    :return: time dependent Envelope Goodness-Of-Fit, type numpy.ndarray.
    """
    TEM = tem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return A * np.exp(-np.abs(TEM)**k)


def tpg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st2_isref=True, A=10., k=1.):
    """
    Time Dependent Phase Goodness-Of_Fit

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

    :return: time dependent Phase Goodness-Of-Fit, type numpy.ndarray.
    """
    TPM = tpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return A * (1 - np.abs(TPM)**k)


def feg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st2_isref=True, A=10., k=1.):
    """
    Frequency Dependent Envelope Goodness-Of_Fit

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

    :return: frequency dependent Envelope Goodness-Of-Fit, type numpy.ndarray.
    """
    FEM = fem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return A * np.exp(-np.abs(FEM)**k)


def fpg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st2_isref=True, A=10., k=1.):
    """
    Frequency Dependent Phase Goodness-Of_Fit

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

    :return: frequency dependent Phase Goodness-Of-Fit, type numpy.ndarray.
    """
    FPM = fpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return A * (1 - np.abs(FPM)**k)


def eg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st2_isref=True, A=10., k=1.):
    """
    Single Valued Envelope Goodness-Of_Fit

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

    :return: Single Valued Envelope Goodness-Of-Fit,
    """
    EM = em(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return A * np.exp(-np.abs(EM)**k)


def pg(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st2_isref=True, A=10., k=1.):
    """
    Single Valued Phase Goodness-Of_Fit

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

    :return: Single Valued Phase Goodness-Of-Fit,
    """
    PM = pm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)
    return A * (1 - np.abs(PM)**k)



def plot_tf_misfits(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6,
    norm='global', st2_isref=True, left=0.1, bottom=0.1, h_1=0.2, h_2=0.125,
    h_3=0.2, w_1=0.2, w_2=0.6, w_cb=0.01, d_cb=0.0, show=True, plot_args=['k',
    'r', 'b'], ylim=0., clim=0., cmap='RdBu'):
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
    :param cmap: colormap for TFEM/TFPM

    :return: If show is False, returns a maplotlib.pyplot.figure object
    """

    npts = len(st1)
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts)
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    # Plot S1 and S1t and TFEM + TFPM misfits
    fig = plt.figure()
    
    # plot signals
    ax_sig = fig.add_axes([left + w_1, bottom + h_2 + h_3, w_2, h_1])
    ax_sig.plot(t, st1, plot_args[0])
    ax_sig.plot(t, st2, plot_args[1])
    
    # plot TEM
    TEM = tem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref)
    ax_TEM = fig.add_axes([left + w_1, bottom + h_1 + h_2 + h_3, w_2, h_2])
    ax_TEM.plot(t, TEM, plot_args[2])
    
    # plot TFEM
    TFEM = tfem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
                st2_isref=st2_isref)
    ax_TFEM = fig.add_axes([left + w_1, bottom + h_1 + 2*h_2 + h_3, w_2, h_3])
    img_TFEM = ax_TFEM.imshow(TFEM, interpolation='nearest', cmap=cmap, extent=[t[0],
                              t[-1], fmin, fmax], aspect='auto', origin='lower')
    ax_TFEM.set_yscale('log')
    
    # plot FEM
    FEM = fem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref)
    ax_FEM = fig.add_axes([left, bottom + h_1 + 2*h_2 + h_3, w_1, h_3])
    ax_FEM.semilogy(FEM, f, plot_args[2])
    ax_FEM.set_ylim(fmin, fmax)
    
    # plot TPM
    TPM = tpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref)
    ax_TPM = fig.add_axes([left + w_1, bottom, w_2, h_2])
    ax_TPM.plot(t, TPM, plot_args[2])
    
    # plot TFPM
    TFPM = tfpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref)
    ax_TFPM = fig.add_axes([left + w_1, bottom + h_2, w_2, h_3])
    img_TFPM = ax_TFPM.imshow(TFPM, interpolation='nearest', cmap=cmap, extent=[t[0],
                         t[-1], f[0], f[-1]], aspect='auto', origin='lower')
    ax_TFPM.set_yscale('log')
    
    # add colorbars
    #ax_cb_TFEM = fig.add_axes([left + w_1 + w_2 + d_cb + w_cb, 
    #                           bottom + h_1 + 2*h_2 + h_3, w_cb, h_3])
    #fig.colorbar(img_TFEM, cax=ax_cb_TFEM)

    ax_cb_TFPM = fig.add_axes([left + w_1 + w_2 + d_cb + w_cb, bottom,
                               w_cb, h_2 + h_3])
    fig.colorbar(img_TFPM, cax=ax_cb_TFPM)
    
    # plot FPM
    FPM = fpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
              st2_isref=st2_isref)
    ax_FPM = fig.add_axes([left, bottom + h_2, w_1, h_3])
    ax_FPM.semilogy(FPM, f, plot_args[2])
    ax_FPM.set_ylim(fmin, fmax)
   
    
    # set limits
    ylim_sig = np.max([np.abs(st1).max(), np.abs(st2).max()]) * 1.1
    ax_sig.set_ylim(-ylim_sig, ylim_sig)

    if ylim == 0.:
        ylim = np.max([np.abs(TEM).max(), np.abs(TPM).max(), np.abs(FEM).max(),
                       np.abs(FPM).max()]) * 1.1

    ax_TEM.set_ylim(-ylim, ylim)
    ax_FEM.set_xlim(-ylim, ylim)
    ax_TPM.set_ylim(-ylim, ylim)
    ax_FPM.set_xlim(-ylim, ylim)

    if clim == 0.:
        clim = np.max([np.abs(TFEM).max(), np.abs(TFPM).max()])

    img_TFPM.set_clim(-clim, clim)
    img_TFEM.set_clim(-clim, clim)


    # add text box for EM + PM
    PM = pm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
            st2_isref=st2_isref)
    EM = em(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
            st2_isref=st2_isref)

    textstr = 'EM = %.2f\nPM = %.2f' % (EM, PM)
    props = dict(boxstyle='round', facecolor='white')
    ax_sig.text(-0.15, 0.5, textstr, transform=ax_sig.transAxes,
            verticalalignment='center', horizontalalignment='right',
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
    
    if show:
        plt.show()
    else:
        return fig
