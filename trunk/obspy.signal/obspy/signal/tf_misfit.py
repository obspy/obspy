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
Various Time Frequency Misfit Functions based on Kristekova et. al. (2006).

.. seealso:: [Kristekova2006]_

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

    :param st: time dependent signal. Will be demeaned and tapered before FFT,
        type numpy.ndarray.
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
    st1_isref=True):
    """
    Time Frequency Envelope Misfit

    .. seealso:: [Kristekova2006]_, eq. (9)

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

    :return: time frequency representation of Envelope Misfit,
        type numpy.ndarray.
    """
    W1 = cwt(st1, dt, w0, fmin, fmax, nf)
    W2 = cwt(st2, dt, w0, fmin, fmax, nf)

    if st1_isref:
        Ar = np.abs(W1)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    TFEM = (np.abs(W2) - np.abs(W1))

    if norm == 'global':
        return  TFEM / np.max(Ar)
    elif norm == 'local':
        return  TFEM / Ar


def tfpm(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st1_isref=True):
    """
    Time Frequency Phase Misfit

    .. seealso:: [Kristekova2006]_, eq. (10)

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

    :return: time frequency representation of Phase Misfit,
        type numpy.ndarray.
    """
    W1 = cwt(st1, dt, w0, fmin, fmax, nf)
    W2 = cwt(st2, dt, w0, fmin, fmax, nf)

    if st1_isref:
        Ar = np.abs(W1)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    TFPM = np.angle(W2 / W1) / np.pi

    if norm == 'global':
        return Ar * TFPM / np.max(Ar)
    elif norm == 'local':
        return TFPM


def tem(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st1_isref=True):
    """
    Time-dependent Envelope Misfit

    .. seealso:: [Kristekova2006]_, eq. (11)

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

    :return: Time-dependent Envelope Misfit, type numpy.ndarray.
    """
    W1 = cwt(st1, dt, w0, fmin, fmax, nf)
    W2 = cwt(st2, dt, w0, fmin, fmax, nf)

    if st1_isref:
        Ar = np.abs(W1)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    TEM = np.sum((np.abs(W2) - np.abs(W1)), axis=0)

    if norm == 'global':
        return TEM / np.max(np.sum(Ar, axis=0))
    elif norm == 'local':
        return TEM / np.sum(Ar, axis=0)


def tpm(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st1_isref=True):
    """
    Time-dependent Phase Misfit

    .. seealso:: [Kristekova2006]_, eq. (12)

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

    :return: Time-dependent Phase Misfit, type numpy.ndarray.
    """
    W1 = cwt(st1, dt, w0, fmin, fmax, nf)
    W2 = cwt(st2, dt, w0, fmin, fmax, nf)

    if st1_isref:
        Ar = np.abs(W1)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    TPM = np.angle(W2 / W1) / np.pi
    TPM = np.sum(Ar * TPM, axis=0)

    if norm == 'global':
        return TPM / np.max(np.sum(Ar, axis=0))
    elif norm == 'local':
        return TPM / np.sum(Ar, axis=0)


def fem(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st1_isref=True):
    """
    Frequency-dependent Envelope Misfit

    .. seealso:: [Kristekova2006]_, eq. (14)

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

    :return: Frequency-dependent Envelope Misfit, type numpy.ndarray.
    """
    npts = len(st1)

    W1 = cwt(st1, dt, w0, fmin, fmax, nf)
    W2 = cwt(st2, dt, w0, fmin, fmax, nf)

    if st1_isref:
        Ar = np.abs(W1)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    TEM = np.abs(W2) - np.abs(W1)
    TEM = np.sum(TEM, axis=1)
   
    if norm == 'global':
        return TEM / np.max(np.sum(Ar, axis=1))
    elif norm == 'local':
        return TEM / np.sum(Ar, axis=1)


def fpm(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st1_isref=True):
    """
    Frequency-dependent Phase Misfit
    
    .. seealso:: [Kristekova2006]_, eq. (15)

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

    :return: Frequency-dependent Phase Misfit, type numpy.ndarray.
    """
    npts = len(st1)

    W1 = cwt(st1, dt, w0, fmin, fmax, nf)
    W2 = cwt(st2, dt, w0, fmin, fmax, nf)

    if st1_isref:
        Ar = np.abs(W1)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    TPM = np.angle(W2 / W1) / np.pi
    TPM = np.sum(Ar * TPM, axis=1)

    if norm == 'global':
        return TPM / np.max(np.sum(Ar, axis=1))
    elif norm == 'local':
        return TPM / np.sum(Ar, axis=1)


def em(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st1_isref=True):
    """
    Single Valued Envelope Misfit

    .. seealso:: [Kristekova2006]_, eq. (17)

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

    :return: Single Valued Envelope Misfit
    """
    W1 = cwt(st1, dt, w0, fmin, fmax, nf)
    W2 = cwt(st2, dt, w0, fmin, fmax, nf)

    if st1_isref:
        Ar = np.abs(W1)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    EM = (np.sum((np.abs(W2) - np.abs(W1))**2))**.5
   
    if norm == 'global':
        return EM / (np.sum(Ar**2))**.5
    elif norm == 'local':
        return EM / (np.sum(Ar**2))**.5


def pm(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, norm='global',
    st1_isref=True):
    """
    Single Valued Phase Misfit

    .. seealso:: [Kristekova2006]_, eq. (18)

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

    :return: Single Valued Phase Misfit
    """
    W1 = cwt(st1, dt, w0, fmin, fmax, nf)
    W2 = cwt(st2, dt, w0, fmin, fmax, nf)

    if st1_isref:
        Ar = np.abs(W1)
    else:
        if np.abs(W1).max() > np.abs(W2).max():
            Ar = np.abs(W1)
        else:
            Ar = np.abs(W2)

    PM = np.angle(W2 / W1) / np.pi

    PM = (np.sum((Ar * PM)**2))**.5

    if norm == 'global':
        return PM / (np.sum(Ar**2))**.5
    elif norm == 'local':
        return PM / (np.sum(Ar**2))**.5
