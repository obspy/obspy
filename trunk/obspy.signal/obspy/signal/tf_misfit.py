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
Various Time Frequency Misfit Functions

Based on:

M. Kristekova, J. Kristek, P. Moczo, and S. M. Day,
Misfit Criteria for Quantitative Comparison of Seismograms
Bulletin of the Seismological Society of America, Vol. 96, No. 5,
pp. 1836-1850, October 2006, doi: 10.1785/0120060012

Will be refered to as Kristekova et. al. (2006) below.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
            
import numpy as np
from obspy.signal import util, cosTaper


def cwt(st, dt, w0, f, wl='morlet'):
    '''
    Continuous Wavelet Transformation
    
    Continuous Wavelet Transformation in the Frequency Domain. Compare to
    Kristekova et. al. (2006) eq. (4)

    :param st: time dependent signal. Will be demeaned and tapered before FFT,
        type numpy.ndarray.
    :param dt: time step between two samples in st
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param f: frequency discretization, type numpy.ndarray.  
    :param wl: wavelet to use, for now only 'morlet' is implemented

    :return: time frequency representation of st, type numpy.ndarray of complex
        values.
    '''
    npts = len(st)
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts)
    
    cwt = np.zeros((t.shape[0], f.shape[0])) * 0j

    if wl == 'morlet':
        psi = lambda t : np.pi**(-.25) * np.exp(1j * w0 * t) * np.exp(-t**2 / 2.)
        scale = lambda f : w0 / (2 * np.pi * f)
    else:
        raise ValueError('wavelet type "' + wl + '" not defined!')

    st -= st.mean()
    st *= cosTaper(len(st), p=0.05)
    nfft = util.nextpow2(len(st)) * 2
    sf = np.fft.fft(st, n=nfft)

    for n, _f in enumerate(f):
        a = scale(_f)
        # time shift necesarry, because wavelet is defined around t = 0
        psih = psi(-1*(t - t[-1]/2.)/a).conjugate() / np.abs(a)**.5
        psihf = np.fft.fft(psih, n=nfft)
        tminin = int(t[-1]/2. / (t[1] - t[0]))
        cwt[:,n] = np.fft.ifft(psihf * sf)[tminin:tminin+t.shape[0]] * (t[1] - t[0])

    return cwt.T



def tfem(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, wl='morlet'):
    '''
    Time Frequency Envelope Misfit
    
    Time Frequency Envelope Misfit as defined in Kristekova et. al. (2006) eq.(9)

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
    :param wl: wavelet to use in continuous wavelet transform

    :return: time frequency representation of Envelope Misfit,
        type numpy.ndarray.
    '''
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    W1 = cwt(st1, dt, w0, f)
    W2 = cwt(st2, dt, w0, f)

    return (np.abs(W2) - np.abs(W1)) / np.max(np.abs(W1))



def tfpm(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, wl='morlet'):
    '''
    Time Frequency Phase Misfit
    
    Time Frequency Phase Misfit as defined in Kristekova et. al. (2006) eq.(10)

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
    :param wl: wavelet to use in continuous wavelet transform

    :return: time frequency representation of Phase Misfit,
        type numpy.ndarray.
    '''
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    W1 = cwt(st1, dt, w0, f)
    W2 = cwt(st2, dt, w0, f)

    TFPMl = np.angle(W2) - np.angle(W1)
    TFPMl[TFPMl > np.pi] -= 2*np.pi
    TFPMl[TFPMl < -np.pi] += 2*np.pi

    return np.abs(W1) * TFPMl / np.pi / np.max(np.abs(W1))



def tem(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, wl='morlet'):
    '''
    Time-dependent Envelope Misfit
    
    Time-dependent Envelope Misfit Misfit as defined in Kristekova et. al.
        (2006) eq.(11)

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
    :param wl: wavelet to use in continuous wavelet transform

    :return: Time-dependent Envelope Misfit, type numpy.ndarray.
    '''
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    W1 = cwt(st1, dt, w0, f)
    W2 = cwt(st2, dt, w0, f)

    TEMl = np.sum((np.abs(W2) - np.abs(W1)), axis=0) / nf
    TEMl /=  np.max(np.sum(np.abs(W1), axis=0))  / nf
   
    return TEMl



def tpm(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, wl='morlet'):
    '''
    Time-dependent Phase Misfit
    
    Time-dependent Phase Misfit Misfit as defined in Kristekova et. al.
        (2006) eq.(12)

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
    :param wl: wavelet to use in continuous wavelet transform

    :return: Time-dependent Phase Misfit, type numpy.ndarray.
    '''
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    W1 = cwt(st1, dt, w0, f)
    W2 = cwt(st2, dt, w0, f)

    TPMl = np.angle(W2) - np.angle(W1)
    TPMl[TPMl > np.pi] -= 2*np.pi
    TPMl[TPMl < -np.pi] += 2*np.pi

    TPMl = np.abs(W1) * TPMl / np.pi

    TPMl = np.sum(TPMl, axis=0) / nf
    TPMl /= np.max(np.sum(np.abs(W1), axis=0)) / nf

    return TPMl



def fem(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, wl='morlet'):
    '''
    Frequency-dependent Envelope Misfit
    
    Frequency-dependent Envelope Misfit as defined in Kristekova et. al.
        (2006) eq.(14)

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
    :param wl: wavelet to use in continuous wavelet transform

    :return: Frequency-dependent Envelope Misfit, type numpy.ndarray.
    '''
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)
    npts = len(st1)

    W1 = cwt(st1, dt, w0, f)
    W2 = cwt(st2, dt, w0, f)

    TEMl = np.sum((np.abs(W2) - np.abs(W1)), axis=1) / npts
    TEMl /=  np.max(np.sum(np.abs(W1), axis=1))  / npts
   
    return TEMl



def fpm(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, wl='morlet'):
    '''
    Frequency-dependent Phase Misfit
    
    Frequency-dependent Phase Misfit as defined in Kristekova et. al.
        (2006) eq.(15)

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
    :param wl: wavelet to use in continuous wavelet transform

    :return: Frequency-dependent Phase Misfit, type numpy.ndarray.
    '''
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)
    npts = len(st1)

    W1 = cwt(st1, dt, w0, f)
    W2 = cwt(st2, dt, w0, f)

    TPMl = np.angle(W2) - np.angle(W1)
    TPMl[TPMl > np.pi] -= 2*np.pi
    TPMl[TPMl < -np.pi] += 2*np.pi

    TPMl = np.abs(W1) * TPMl / np.pi

    TPMl = np.sum(TPMl, axis=1) / npts
    TPMl /= np.max(np.sum(np.abs(W1), axis=1)) / npts

    return TPMl



def em(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, wl='morlet'):
    '''
    Single Valued Envelope Misfit

    Single Valued Envelope Misfit as defined in Kristekova et. al.
        (2006) eq.(17)

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
    :param wl: wavelet to use in continuous wavelet transform

    :return: Single Valued Envelope Misfit
    '''
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    W1 = cwt(st1, dt, w0, f)
    W2 = cwt(st2, dt, w0, f)

    EMl = (np.sum((np.abs(W2) - np.abs(W1))**2))**.5
    EMl /=  (np.sum(np.abs(W1)**2))**.5
   
    return EMl



def pm(st1, st2, dt=1., fmin=1., fmax=10., nf=100, w0=6, wl='morlet'):
    '''
    Single Valued Phase Misfit

    Single Valued Phase Misfit as defined in Kristekova et. al.
        (2006) eq.(18)

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
    :param wl: wavelet to use in continuous wavelet transform

    :return: Single Valued Phase Misfit
    '''
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    W1 = cwt(st1, dt, w0, f)
    W2 = cwt(st2, dt, w0, f)

    PMl = np.angle(W2) - np.angle(W1)
    PMl[PMl > np.pi] -= 2 * np.pi
    PMl[PMl < -np.pi] += 2 * np.pi

    PMl = np.abs(W1) * PMl / np.pi

    PMl = (np.sum(PMl**2))**.5
    PMl /= (np.sum(np.abs(W1)**2))**.5

    return PMl


