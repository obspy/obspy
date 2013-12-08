#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: hoctavbands.py
#   Author: Conny Hammer
#    Email: conny.hammer@geo.uni-potsdam.de
#
# Copyright (C) 2008-2012 Conny Hammer
#-------------------------------------------------------------------
"""
Half Octave Bands

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from scipy import fftpack
import numpy as np
import util


def sonogram(data, fs, fc1, nofb, no_win):
    """
    Sonogram of a signal.

    Computes the sonogram of the given data which can be windowed or not.
    The sonogram is determined by the power in half octave bands of the given
    data.

    If data are windowed the analytic signal and the envelope of each window
    is returned.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to make envelope of.
    :param fs: Sampling frequency in Hz.
    :param fc1: Center frequency of lowest half octave band.
    :param nofb: Number of half octave bands.
    :param no_win: Number of data windows.
    :return: Half octave bands.
    """
    fc = np.zeros([nofb])
    fmin = np.zeros([nofb])
    fmax = np.zeros([nofb])

    fc[0] = float(fc1)
    fmin[0] = fc[0] / np.sqrt(float(5. / 3.))
    fmax[0] = fc[0] * np.sqrt(float(5. / 3.))
    for i in range(1, nofb):
        fc[i] = fc[i - 1] * 1.5
        fmin[i] = fc[i] / np.sqrt(float(5. / 3.))
        fmax[i] = fc[i] * np.sqrt(float(5. / 3.))
    nfft = util.nextpow2(data.shape[np.size(data.shape) - 1])
    #c = np.zeros((data.shape), dtype='complex64')
    c = fftpack.fft(data, nfft)
    z = np.zeros([len(c[:, 1]), nofb])
    z_tot = np.zeros(len(c[:, 1]))
    hob = np.zeros([no_win, nofb])
    for k in xrange(no_win):
        for j in xrange(len(c[1, :])):
            z_tot[k] = z_tot[k] + pow(np.abs(c[k, j]), 2)
        for i in xrange(nofb):
            start = int(round(fmin[i] * nfft * 1. / float(fs), 0))
            end = int(round(fmax[i] * nfft * 1. / float(fs), 0)) + 1
            for j in xrange(start, end):
                z[k, i] = z[k, i] + pow(np.abs(c[k, j - 1]), 2)
            hob[k, i] = np.log(z[k, i] / z_tot[k])
    return hob
