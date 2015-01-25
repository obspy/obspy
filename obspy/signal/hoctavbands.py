# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: hoctavbands.py
#   Author: Conny Hammer
#    Email: conny.hammer@geo.uni-potsdam.de
#
# Copyright (C) 2008-2012 Conny Hammer
# ------------------------------------------------------------------
"""
Half Octave Bands

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from scipy import fftpack
import numpy as np
from . import util


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
    fc = float(fc1) * 1.5**np.arange(nofb)
    fmin = fc / np.sqrt(5. / 3.)
    fmax = fc * np.sqrt(5. / 3.)

    nfft = util.nextpow2(data.shape[-1])
    new_dtype = np.float32 if data.dtype.itemsize == 4 else np.float64
    data = np.require(data, dtype=new_dtype)
    c = fftpack.fft(data, nfft)
    z_tot = np.sum(np.abs(c)**2, axis=1)

    start = np.around(fmin * nfft / fs, 0).astype(int) - 1
    end = np.around(fmax * nfft / fs, 0).astype(int)
    z = np.zeros([c.shape[0], nofb])
    for i in range(nofb):
        z[:, i] = np.sum(np.abs(c[:, start[i]:end[i]])**2, axis=1)

    hob = np.log(z / z_tot[:, np.newaxis])
    return hob
