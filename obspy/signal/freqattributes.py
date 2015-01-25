# -*- coding: utf-8 -*-
# -----------------------------------------------------------------
# Filename: freqattributes.py
#   Author: Conny Hammer
#    Email: conny.hammer@geo.uni-potsdam.de
#
# Copyright (C) 2008-2012 Conny Hammer
# -----------------------------------------------------------------
"""
Frequency Attributes

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from operator import itemgetter
from scipy import fftpack, signal, sparse
from obspy.signal.invsim import seisSim, cornFreq2Paz
import numpy as np
from obspy.signal import util


def mper(data, win, Nfft, n1=0, n2=0):
    """
    Spectrum of a signal.

    Computes the spectrum of the given data which can be windowed or not. The
    spectrum is estimated using the modified periodogram. If n1 and n2 are not
    specified the periodogram of the entire sequence is returned.

    The modified periodogram of the given signal is returned.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to make spectrum of.
    :param win: Window to multiply with given signal.
    :param Nfft: Number of points for FFT.
    :type n1: int, optional
    :param n1: Starting index, defaults to ``0``.
    :type n2: int, optional
    :param n2: Ending index, defaults to ``0``.
    :return: Spectrum.
    """
    if (n2 == 0):
        n2 = len(data)
    n = n2 - n1
    U = pow(np.linalg.norm([win]), 2) / n
    xw = data * win
    Px = pow(abs(fftpack.fft(xw, Nfft)), 2) / (n * U)
    Px[0] = Px[1]
    return Px


def welch(data, win, Nfft, L=0, over=0):
    """
    Spectrum of a signal.

    Computes the spectrum of the given data which can be windowed or not.
    The spectrum is estimated using Welch's method of averaging modified
    periodograms.

    Welch's estimate of the power spectrum is returned using a linear scale.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to make spectrum of.
    :param win: Window to multiply with given signal.
    :param Nfft: Number of points for FFT.
    :type L: int, optional
    :param L: Length of windows to be averaged, defaults to ``0``.
    :type over: float, optional
    :param over: Overlap of windows to be averaged 0<over<1, defaults to ``0``.
    :return: Spectrum.
    """
    if (L == 0):
        L = len(data)
    n1 = 0
    n2 = L
    n0 = (1. - float(over)) * L
    nsect = 1 + int(np.floor((len(data) - L) / (n0)))
    Px = 0
    for _i in range(nsect):
        Px = Px + mper(data, win, Nfft, n1, n2) / nsect
        n1 = n1 + n0
        n2 = n2 + n0
    return Px


def cfrequency(data, fs, smoothie, fk):
    """
    Central frequency of a signal.

    Computes the central frequency of the given data which can be windowed or
    not. The central frequency is a measure of the frequency where the
    power is concentrated. It corresponds to the second moment of the power
    spectral density function.

    The central frequency is returned.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to estimate central frequency from.
    :param fs: Sampling frequency in Hz.
    :param smoothie: Factor for smoothing the result.
    :param fk: Coefficients for calculating time derivatives
        (calculated via central difference).
    :return: **cfreq[, dcfreq]** - Central frequency, Time derivative of center
        frequency (windowed only).
    """
    # for windowed data
    if np.size(data.shape) > 1:
        cfreq = np.zeros(data.shape[0])
        i = 0
        for row in data:
            cfreq[i] = cfrequency_unwindowed(row, fs)
            i = i + 1
        cfreq = util.smooth(cfreq, smoothie)
        # cfreq_add = \
        #        np.append(np.append([cfreq[0]] * (np.size(fk) // 2), cfreq),
        #        [cfreq[np.size(cfreq) - 1]] * (np.size(fk) // 2))
        # faster alternative
        cfreq_add = np.hstack(
            ([cfreq[0]] * (np.size(fk) // 2), cfreq,
             [cfreq[np.size(cfreq) - 1]] * (np.size(fk) // 2)))
        dcfreq = signal.lfilter(fk, 1, cfreq_add)
        # dcfreq = dcfreq[np.size(fk) // 2:(np.size(dcfreq) -
        #         np.size(fk) // 2)]
        # correct start and end values of time derivative
        dcfreq = dcfreq[np.size(fk) - 1:np.size(dcfreq)]
        return cfreq, dcfreq
    # for unwindowed data
    else:
        cfreq = cfrequency_unwindowed(data, fs)
        return cfreq


def cfrequency_unwindowed(data, fs):
    """
    Central frequency of a signal.

    Computes the central frequency of the given data (a single waveform).
    The central frequency is a measure of the frequency where the
    power is concentrated. It corresponds to the second moment of the power
    spectral density function.

    The central frequency is returned in Hz.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to estimate central frequency from.
    :param fs: Sampling frequency in Hz.
    :return: **cfreq** - Central frequency in Hz
    """
    nfft = util.nextpow2(len(data))
    freq = np.linspace(0, fs, nfft + 1)
    freqaxis = freq[0:nfft // 2]
    Px_wm = welch(data, np.hamming(len(data)), nfft)
    Px = Px_wm[0:len(Px_wm) // 2]
    cfreq = np.sqrt(np.sum(freqaxis ** 2 * Px) / (sum(Px)))
    return cfreq


def bwith(data, fs, smoothie, fk):
    """
    Bandwidth of a signal.

    Computes the bandwidth of the given data which can be windowed or not.
    The bandwidth corresponds to the level where the power of the spectrum is
    half its maximum value. It is determined as the level of 1/sqrt(2) times
    the maximum Fourier amplitude.

    If data are windowed the bandwidth of each window is returned.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to make envelope of.
    :param fs: Sampling frequency in Hz.
    :param smoothie: Factor for smoothing the result.
    :param fk: Coefficients for calculating time derivatives
        (calculated via central difference).
    :return: **bwith[, dbwithd]** - Bandwidth, Time derivative of predominant
        period (windowed only).
    """
    new_dtype = np.float32 if data.dtype.itemsize == 4 else np.float64
    data = np.require(data, dtype=new_dtype)

    nfft = util.nextpow2(data.shape[1])
    freqaxis = np.linspace(0, fs, nfft + 1)
    bwith = np.zeros(data.shape[0])
    f = fftpack.fft(data, nfft)
    f_sm = util.smooth(abs(f[:, 0:nfft // 2]), 10)
    if np.size(data.shape) > 1:
        i = 0
        for row in f_sm:
            minfc = abs(row - max(abs(row * (1 / np.sqrt(2)))))
            [mdist_ind, _mindist] = min(enumerate(minfc), key=itemgetter(1))
            bwith[i] = freqaxis[mdist_ind]
            i = i + 1
        # bwith_add = \
        #        np.append(np.append([bwith[0]] * (np.size(fk) // 2), bwith),
        #        [bwith[np.size(bwith) - 1]] * (np.size(fk) // 2))
        # faster alternative
        bwith_add = np.hstack(
            ([bwith[0]] * (np.size(fk) // 2), bwith,
             [bwith[np.size(bwith) - 1]] * (np.size(fk) // 2)))
        dbwith = signal.lfilter(fk, 1, bwith_add)
        # dbwith = dbwith[np.size(fk) // 2:(np.size(dbwith) -
        #         np.size(fk) // 2)]
        # correct start and end values of time derivative
        dbwith = dbwith[np.size(fk) - 1:]
        bwith = util.smooth(bwith, smoothie)
        dbwith = util.smooth(dbwith, smoothie)
        return bwith, dbwith
    else:
        minfc = abs(data - max(abs(data * (1 / np.sqrt(2)))))
        [mdist_ind, _mindist] = min(enumerate(minfc), key=itemgetter(1))
        bwith = freqaxis[mdist_ind]
        return bwith


def domperiod(data, fs, smoothie, fk):
    """
    Predominant period of a signal.

    Computes the predominant period of the given data which can be windowed or
    not. The period is determined as the period of the maximum value of the
    Fourier amplitude spectrum.

    If data are windowed the predominant period of each window is returned.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to determine predominant period of.
    :param fs: Sampling frequency in Hz.
    :param smoothie: Factor for smoothing the result.
    :param fk: Coefficients for calculating time derivatives
        (calculated via central difference).
    :return: **dperiod[, ddperiod]** - Predominant period, Time derivative of
        predominant period (windowed only).
    """
    new_dtype = np.float32 if data.dtype.itemsize == 4 else np.float64
    data = np.require(data, dtype=new_dtype)

    nfft = 1024
    # nfft = util.nextpow2(data.shape[1])
    freqaxis = np.linspace(0, fs, nfft + 1)
    dperiod = np.zeros(data.shape[0])
    f = fftpack.fft(data, nfft)
    # f_sm = util.smooth(abs(f[:,0:nfft // 2]),1)
    f_sm = f[:, 0:nfft // 2]
    if np.size(data.shape) > 1:
        i = 0
        for row in f_sm:
            [mdist_ind, _mindist] = max(enumerate(abs(row)), key=itemgetter(1))
            dperiod[i] = freqaxis[mdist_ind]
            i = i + 1
        # dperiod_add = np.append(np.append([dperiod[0]] * \
        #        (np.size(fk) // 2), dperiod),
        #        [dperiod[np.size(dperiod) - 1]] * (np.size(fk) // 2))
        # faster alternative
        dperiod_add = np.hstack(
            ([dperiod[0]] * (np.size(fk) // 2), dperiod,
             [dperiod[np.size(dperiod) - 1]] * (np.size(fk) // 2)))
        ddperiod = signal.lfilter(fk, 1, dperiod_add)
        # ddperiod = ddperiod[np.size(fk) / \
        #    2:(np.size(ddperiod) - np.size(fk) // 2)]
        # correct start and end values of time derivative
        ddperiod = ddperiod[np.size(fk) - 1:]
        dperiod = util.smooth(dperiod, smoothie)
        ddperiod = util.smooth(ddperiod, smoothie)
        return dperiod, ddperiod
    else:
        [mdist_ind, _mindist] = max(enumerate(abs(data)), key=itemgetter(1))
        dperiod = freqaxis[mdist_ind]
        return dperiod


def logbankm(p, n, fs, w):
    """
    Matrix for a log-spaced filterbank.

    Computes a matrix containing the filterbank amplitudes for a log-spaced
    filterbank.

    :param p: Number of filters in filterbank.
    :param n: Length of fft.
    :param fs: Sampling frequency in Hz.
    :param w: Window function.
    :return: **xx, yy, zz** - Matrix containing the filterbank amplitudes,
        Lowest fft bin with a non-zero coefficient, Highest fft bin with a
        non-zero coefficient.
    """
    # alternative to avoid above problems: low end of the lowest filter
    # corresponds to maximum frequency resolution
    fn2 = np.floor(n / 2)
    fl = np.floor(fs) / np.floor(n)
    fh = np.floor(fs / 2)
    lr = np.log((fh) / (fl)) / (p + 1)
    bl = n * ((fl) *
              np.exp(np.array([0, 1, p, p + 1]) * float(lr)) / float(fs))
    b2 = int(np.ceil(bl[1]))
    b3 = int(np.floor(bl[2]))
    b1 = int(np.floor(bl[0])) + 1
    b4 = int(min(fn2, np.ceil(bl[3]))) - 1
    pf = np.log(np.arange(b1 - 1, b4 + 1, dtype=np.float64) / n * fs / fl) / lr
    fp = np.floor(pf)
    pm = pf - fp
    k2 = b2 - b1 + 1
    k3 = b3 - b1 + 1
    k4 = b4 - b1 + 1
    r = np.append(fp[k2:k4 + 2], 1 + fp[1:k3 + 1]) - 1
    c = np.append(np.arange(k2, k4 + 1), np.arange(1, k3 + 1)) - 1
    v = 2 * np.append([1 - pm[k2:k4 + 1]], [pm[1:k3 + 1]])
    mn = b1 + 1
    mx = b4 + 1
    # x = np.array([[c],[r]], dtype=[('x', np.float), ('y', np.float)])
    # ind=np.argsort(x, order=('x','y'))
    if (w == 'Hann'):
        v = 1. - [np.cos([v * float(np.pi / 2.)])]
    elif (w == 'Hamming'):
        v = 1. - 0.92 / 1.08 * np.cos(v * float(np.pi / 2))
    # bugfix for #70 - scipy.sparse.csr_matrix() delivers sometimes a
    # transposed matrix depending on the installed NumPy version - using
    # scipy.sparse.coo_matrix() ensures compatibility with old NumPy versions
    xx = sparse.coo_matrix((v, (c, r))).transpose().todense()
    return xx, mn - 1, mx - 1


def logcep(data, fs, nc, p, n, w):  # @UnusedVariable: n is never used!!!
    """
    Cepstrum of a signal.

    Computes the cepstral coefficient on a logarithmic scale of the given data
    which can be windowed or not.

    If data are windowed the analytic signal and the envelope of each window is
    returned.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to make envelope of.
    :param fs: Sampling frequency in Hz.
    :param nc: number of cepstral coefficients.
    :param p: Number of filters in filterbank.
    :param n: Number of data windows.
    :return: Cepstral coefficients.
    """
    new_dtype = np.float32 if data.dtype.itemsize == 4 else np.float64
    data = np.require(data, dtype=new_dtype)

    dataT = np.transpose(data)
    nfft = util.nextpow2(dataT.shape[0])
    fc = fftpack.fft(dataT, nfft, 0)
    f = fc[1:len(fc) // 2 + 1, :]
    m, a, b = logbankm(p, nfft, fs, w)
    pw = np.real(np.multiply(f[a:b, :], np.conj(f[a:b, :])))
    pth = np.max(pw) * 1E-20
    ath = np.sqrt(pth)
    # h1 = np.transpose(np.array([[ath] * int(b + 1 - a)]))
    # h2 = m * abs(f[a - 1:b, :])
    y = np.log(np.maximum(m * abs(f[a - 1:b, :]), ath))
    z = util.rdct(y)
    z = z[1:, :]
    # nc = nc + 1
    nf = np.size(z, 1)
    if (p > nc):
        z = z[:nc, :]
    elif (p < nc):
        z = np.vstack([z, np.zeros(nf, nc - p)])
    return z


def pgm(data, delta, freq, damp=0.1):
    """
    Peak ground motion parameters

    Compute the maximal displacement, velocity, acceleration and the peak
    ground acceleration at a certain frequency (standard frequencies for
    ShakeMaps are 0.3/1.0/3.0 Hz).

    Data must be displacement

    :type data: :class:`~numpy.ndarray`
    :param data: Data in displacement to convolve with pendulum at freq.
    :type delta: float
    :param delta: Sampling interval
    :type freq: float
    :param freq: Frequency in Hz.
    :type damp: float
    :param damp: damping factor. Default is set to 0.1
    :rtype: (float, float, float, float)
    :return: Peak Ground Acceleration, maximal displacement, velocity,
        acceleration
    """
    data = data.copy()

    # Displacement
    if abs(max(data)) >= abs(min(data)):
        m_dis = abs(max(data))
    else:
        m_dis = abs(min(data))

    # Velocity
    data = np.gradient(data, delta)
    if abs(max(data)) >= abs(min(data)):
        m_vel = abs(max(data))
    else:
        m_vel = abs(min(data))

    # Acceleration
    data = np.gradient(data, delta)
    if abs(max(data)) >= abs(min(data)):
        m_acc = abs(max(data))
    else:
        m_acc = abs(min(data))

    samp_rate = 1.0 / delta
    T = freq * 1.0
    D = damp
    omega = (2 * 3.14159 * T) ** 2

    paz_sa = cornFreq2Paz(T, damp=D)
    paz_sa['sensitivity'] = omega
    paz_sa['zeros'] = []
    data = seisSim(data, samp_rate, paz_remove=None, paz_simulate=paz_sa,
                   taper=True, simulate_sensitivity=True, taper_fraction=0.05)

    if abs(max(data)) >= abs(min(data)):
        pga = abs(max(data))
    else:
        pga = abs(min(data))

    return (pga, m_dis, m_vel, m_acc)
