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

    :param st1: signal 1 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
    :param st2: signal 2 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
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

    :param st1: signal 1 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
    :param st2: signal 2 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
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

    :param st1: signal 1 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
    :param st2: signal 2 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
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

    :param st1: signal 1 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
    :param st2: signal 2 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
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

    :param st1: signal 1 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
    :param st2: signal 2 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
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

    :param st1: signal 1 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
    :param st2: signal 2 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
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

    :param st1: signal 1 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
    :param st2: signal 2 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
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

    :param st1: signal 1 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
    :param st2: signal 2 of two signals to compare, will be demeaned and
        tapered before FFT in CWT, type numpy.ndarray.
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
