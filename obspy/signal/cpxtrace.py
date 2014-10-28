# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: cpxtrace.py
#   Author: Conny Hammer
#    Email: conny.hammer@geo.uni-potsdam.de
#
# Copyright (C) 2008-2012 Conny Hammer
# ------------------------------------------------------------------
"""
Complex Trace Analysis

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from numpy import size, pi
from scipy import signal
from scipy.integrate import cumtrapz
import numpy as np
from . import util


def envelope(data):
    """
    Envelope of a signal.

    Computes the envelope of the given data which can be windowed or
    not. The envelope is determined by the absolute value of the analytic
    signal of the given data.

    If data are windowed the analytic signal and the envelope of each
    window is returned.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to make envelope of.
    :return: **A_cpx, A_abs** - Analytic signal of input data, Envelope of
        input data.
    """
    nfft = util.nextpow2(data.shape[size(data.shape) - 1])
    A_cpx = np.zeros((data.shape), dtype=np.complex64)
    A_abs = np.zeros((data.shape), dtype=np.float64)
    if (np.size(data.shape) > 1):
        i = 0
        for row in data:
            A_cpx[i, :] = signal.hilbert(row, nfft)
            A_abs[i, :] = abs(signal.hilbert(row, nfft))
            i = i + 1
    else:
        A_cpx = signal.hilbert(data, nfft)
        A_abs = abs(signal.hilbert(data, nfft))
    return A_cpx, A_abs


def normEnvelope(data, fs, smoothie, fk):
    """
    Normalized envelope of a signal.

    Computes the normalized envelope of the given data which can be windowed
    or not. In order to obtain a normalized measure of the signal envelope
    the instantaneous bandwidth of the smoothed envelope is normalized by the
    Nyquist frequency and is integrated afterwards.

    The time derivative of the normalized envelope is returned if input data
    are windowed only.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to make normalized envelope of.
    :param fs: Sampling frequency.
    :param smoothie: Window length for moving average.
    :param fk: Coefficients for calculating time derivatives
        (calculated via central difference).
    :return: **Anorm[, dAnorm]** - Normalized envelope of input data, Time
        derivative of normalized envelope (windowed only).
    """
    x = envelope(data)
    fs = float(fs)
    if (size(x[1].shape) > 1):
        i = 0
        Anorm = np.zeros(x[1].shape[0], dtype=np.float64)
        for row in x[1]:
            A_win_smooth = util.smooth(row, int(np.floor(len(row) / 3)))
            # Differentiation of original signal, dA/dt
            # Better, because faster, calculation of A_win_add
            A_win_add = np.hstack(([A_win_smooth[0]] * (size(fk) // 2),
                                   A_win_smooth,
                                   [A_win_smooth[size(A_win_smooth) - 1]] *
                                   (size(fk) // 2)))
            t = signal.lfilter(fk, 1, A_win_add)
            # correct start and end values of time derivative
            t = t[size(fk) - 1:size(t)]
            A_win_smooth[A_win_smooth < 1] = 1
            # (dA/dt) / 2*PI*smooth(A)*fs/2
            t_ = t / (2. * pi * (A_win_smooth) * (fs / 2.0))
            # Integral within window
            t_ = cumtrapz(t_, dx=(1. / fs))
            t_ = np.concatenate((t_[0:1], t_))
            Anorm[i] = ((np.exp(np.mean(t_))) - 1) * 100
            i = i + 1
        # faster alternative to calculate Anorm_add
        Anorm_add = np.hstack(
            ([Anorm[0]] * (np.size(fk) // 2), Anorm,
             [Anorm[np.size(Anorm) - 1]] * (np.size(fk) // 2)))
        dAnorm = signal.lfilter(fk, 1, Anorm_add)
        # correct start and end values of time derivative
        dAnorm = dAnorm[size(fk) - 1:size(dAnorm)]
        # dAnorm = dAnorm[size(fk) // 2:(size(dAnorm) - size(fk) // 2)]
        return Anorm, dAnorm
    else:
        Anorm = np.zeros(1, dtype=np.float64)
        A_win_smooth = util.smooth(x[1], smoothie)
        # Differentiation of original signal, dA/dt
        # Better, because faster, calculation of A_win_add
        A_win_add = np.hstack(
            ([A_win_smooth[0]] * (size(fk) // 2),
             A_win_smooth, [A_win_smooth[size(A_win_smooth) - 1]] *
             (size(fk) // 2)))
        t = signal.lfilter(fk, 1, A_win_add)
        # correct start and end values of time derivative
        t = t[size(fk) - 1:size(t)]
        A_win_smooth[A_win_smooth < 1] = 1
        t_ = t / (2. * pi * (A_win_smooth) * (fs / 2.0))
        # Integral within window
        t_ = cumtrapz(t_, dx=(1.0 / fs))
        t_ = np.concatenate((t_[0:1], t_))
        Anorm = ((np.exp(np.mean(t_))) - 1) * 100
        return Anorm


def centroid(data, fk):
    """
    Centroid time of a signal.

    Computes the centroid time of the given data which can be windowed or
    not. The centroid time is determined as the time in the processed
    window where 50 per cent of the area below the envelope is reached.

    The time derivative of the centroid time is returned if input data are
    windowed only.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to determine centroid time of.
    :param fk: Coefficients for calculating time derivatives
        (calculated via central difference).
    :return: **centroid[, dcentroid]** - Centroid time input data, Time
        derivative of centroid time (windowed only).
    """
    x = envelope(data)
    if (size(x[1].shape) > 1):
        centroid = np.zeros(x[1].shape[0], dtype=np.float64)
        i = 0
        for row in x[1]:
            # Integral within window
            half = 0.5 * sum(row)
            # Estimate energy centroid
            for k in range(2, size(row)):
                t = sum(row[0:k])
                if (t >= half):
                    frac = (half - (t - sum(row[0:k - 1]))) / \
                        (t - (t - sum(row[0:k - 1])))
                    centroid[i] = \
                        (float(k - 1) + float(frac)) / float(size(row))
                    break
            i = i + 1
        centroid_add = np.hstack(
            ([centroid[0]] * (np.size(fk) // 2),
             centroid, [centroid[np.size(centroid) - 1]] *
             (np.size(fk) // 2)))
        dcentroid = signal.lfilter(fk, 1, centroid_add)
        dcentroid = dcentroid[size(fk) - 1:size(dcentroid)]
        return centroid, dcentroid
    else:
        centroid = np.zeros(1, dtype=np.float64)
        # Integral within window
        half = 0.5 * sum(x[1])
        # Estimate energy centroid
        for k in range(2, size(x[1])):
            t = sum(x[1][0:k])
            if (t >= half):
                frac = (half - (t - sum(x[1][0:k - 1]))) / \
                    (t - (t - sum(x[1][0:k - 1])))
                centroid = (float(k) + float(frac)) / float(size(x[1]))
                break
        return centroid


def instFreq(data, fs, fk):
    """
    Instantaneous frequency of a signal.

    Computes the instantaneous frequency of the given data which can be
    windowed or not. The instantaneous frequency is determined by the time
    derivative of the analytic signal of the input data.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to determine instantaneous frequency of.
    :param fs: Sampling frequency.
    :param fk: Coefficients for calculating time derivatives
        (calculated via central difference).
    :return: **omega[, domega]** - Instantaneous frequency of input data, Time
        derivative of instantaneous frequency (windowed only).
    """
    x = envelope(data)
    if (size(x[0].shape) > 1):
        omega = np.zeros(x[0].shape[0], dtype=np.float64)
        i = 0
        for row in x[0]:
            f = np.real(row)
            h = np.imag(row)
            # faster alternative to calculate f_add
            f_add = np.hstack(
                ([f[0]] * (np.size(fk) // 2), f,
                 [f[np.size(f) - 1]] * (np.size(fk) // 2)))
            fd = signal.lfilter(fk, 1, f_add)
            # correct start and end values of time derivative
            fd = fd[size(fk) - 1:size(fd)]
            # faster alternative to calculate h_add
            h_add = np.hstack(
                ([h[0]] * (np.size(fk) // 2), h,
                 [h[np.size(h) - 1]] * (np.size(fk) // 2)))
            hd = signal.lfilter(fk, 1, h_add)
            # correct start and end values of time derivative
            hd = hd[size(fk) - 1:size(hd)]
            omega_win = abs(((f * hd - fd * h) / (f * f + h * h)) *
                            fs / 2 / pi)
            omega[i] = np.median(omega_win)
            i = i + 1
        # faster alternative to calculate omega_add
        omega_add = np.hstack(
            ([omega[0]] * (np.size(fk) // 2), omega,
             [omega[np.size(omega) - 1]] * (np.size(fk) // 2)))
        domega = signal.lfilter(fk, 1, omega_add)
        # correct start and end values of time derivative
        domega = domega[size(fk) - 1:size(domega)]
        return omega, domega
    else:
        omega = np.zeros(size(x[0]), dtype=np.float64)
        f = np.real(x[0])
        h = np.imag(x[0])
        # faster alternative to calculate f_add
        f_add = np.hstack(
            ([f[0]] * (np.size(fk) // 2), f,
             [f[np.size(f) - 1]] * (np.size(fk) // 2)))
        fd = signal.lfilter(fk, 1, f_add)
        # correct start and end values of time derivative
        fd = fd[size(fk) - 1:size(fd)]
        # faster alternative to calculate h_add
        h_add = np.hstack(
            ([h[0]] * (np.size(fk) // 2), h,
             [h[np.size(h) - 1]] * (np.size(fk) // 2)))
        hd = signal.lfilter(fk, 1, h_add)
        # correct start and end values of time derivative
        hd = hd[size(fk) - 1:size(hd)]
        omega = abs(((f * hd - fd * h) / (f * f + h * h)) * fs / 2 / pi)
        return omega


def instBwith(data, fs, fk):
    """
    Instantaneous bandwidth of a signal.

    Computes the instantaneous bandwidth of the given data which can be
    windowed or not. The instantaneous bandwidth is determined by the time
    derivative of the envelope normalized by the envelope of the input data.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to determine instantaneous bandwidth of.
    :param fs: Sampling frequency.
    :param fk: Filter coefficients for computing time derivative.
    :return: **sigma[, dsigma]** - Instantaneous bandwidth of input data, Time
        derivative of instantaneous bandwidth (windowed only).
    """
    x = envelope(data)
    if (size(x[1].shape) > 1):
        sigma = np.zeros(x[1].shape[0], dtype=np.float64)
        i = 0
        for row in x[1]:
            # faster alternative to calculate A_win_add
            A_win_add = np.hstack(
                ([row[0]] * (np.size(fk) // 2), row,
                 [row[np.size(row) - 1]] * (np.size(fk) // 2)))
            t = signal.lfilter(fk, 1, A_win_add)
            # t = t[size(fk) // 2:(size(t) - size(fk) // 2)]
            # correct start and end values
            t = t[size(fk) - 1:size(t)]
            sigma_win = abs((t * fs) / (row * 2 * pi))
            sigma[i] = np.median(sigma_win)
            i = i + 1
        # faster alternative to calculate sigma_add
        sigma_add = np.hstack(
            ([sigma[0]] * (np.size(fk) // 2), sigma,
             [sigma[np.size(sigma) - 1]] * (np.size(fk) // 2)))
        dsigma = signal.lfilter(fk, 1, sigma_add)
        # dsigma = dsigma[size(fk) // 2:(size(dsigma) - size(fk) // 2)]
        # correct start and end values
        dsigma = dsigma[size(fk) - 1:size(dsigma)]
        return sigma, dsigma
    else:
        row = x[1]
        sigma = np.zeros(size(x[0]), dtype=np.float64)
        # faster alternative to calculate A_win_add
        A_win_add = np.hstack(
            ([row[0]] * (np.size(fk) // 2), row,
             [row[np.size(row) - 1]] * (np.size(fk) // 2)))
        t = signal.lfilter(fk, 1, A_win_add)
        # correct start and end values
        t = t[size(fk) - 1:size(t)]
        sigma = abs((t * fs) / (x[1] * 2 * pi))
        return sigma
