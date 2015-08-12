# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: calibration.py
#  Purpose: Functions for relative calibration (e.g. Huddle test calibration)
#   Author: Felix Bernauer, Simon Kremers
#    Email: bernauer@geophysik.uni-muenchen.de
#
# Copyright (C) 2011 Felix Bernauer, Simon Kremers
# --------------------------------------------------------------------
"""
Functions for relative calibration.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import numpy as np

from obspy.core.stream import Stream
from obspy.core.trace import Trace
from obspy.io.gse2.paz import read_paz
from obspy.signal.invsim import paz_to_freq_resp
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing
from obspy.signal.util import next_pow_2


def rel_calib_stack(st1, st2, calib_file, window_len, overlap_frac=0.5,
                    smooth=0, save_data=True):
    """
    Method for relative calibration of sensors using a sensor with known
    transfer function

    :param st1: Stream or Trace object, (known)
    :param st2: Stream or Trace object, (unknown)
    :type calib_file: str
    :param calib_file: file name of calibration file containing the PAZ of the
        known instrument in GSE2 standard.
    :type window_len: float
    :param window_len: length of sliding window in seconds
    :type overlap_frac: float
    :param overlap_frac: fraction of overlap, defaults to fifty percent (0.5)
    :type smooth: float
    :param smooth: variable that defines if the Konno-Ohmachi taper is used or
        not. default = 0 -> no taper generally used in geopsy: smooth = 40
    :type save_data: bool
    :param save_data: Whether or not to save the result to a file. If True, two
        output files will be created:
        * The new response in station_name.window_length.resp
        * The ref response in station_name.refResp
        Defaults to True
    :returns: frequency, amplitude and phase spectrum

    implemented after rel_calib_stack.c by M.Ohrnberger and J.Wassermann.
    """
    # transform given trace objects to streams
    if isinstance(st1, Trace):
        st1 = Stream([st1])
    if isinstance(st2, Trace):
        st2 = Stream([st2])
    # check if sampling rate and trace length is the same
    if st1[0].stats.npts != st2[0].stats.npts:
        msg = "Traces don't have the same length!"
        raise ValueError(msg)
    elif st1[0].stats.sampling_rate != st2[0].stats.sampling_rate:
        msg = "Traces don't have the same sampling rate!"
        raise ValueError(msg)
    else:
        ndat1 = st1[0].stats.npts
        sampfreq = st1[0].stats.sampling_rate

    # read waveforms
    tr1 = st1[0].data.astype(np.float64)
    tr2 = st2[0].data.astype(np.float64)

    # get window length, nfft and frequency step
    ndat = int(window_len * sampfreq)
    nfft = next_pow_2(ndat)

    # read calib file and calculate response function
    gg, _freq = _calc_resp(calib_file, nfft, sampfreq)

    # calculate number of windows and overlap
    nwin = int(np.floor((ndat1 - nfft) / (nfft / 2)) + 1)
    noverlap = nfft * overlap_frac

    auto, _freq, _t = \
        spectral_helper(tr1, tr1, NFFT=nfft, Fs=sampfreq, noverlap=noverlap)
    cross, freq, _t = \
        spectral_helper(tr2, tr1, NFFT=nfft, Fs=sampfreq, noverlap=noverlap)

    res = (cross / auto).sum(axis=1) * gg

    # The first item might be zero. Problems with phase calculations.
    res = res[1:]
    freq = freq[1:]
    gg = gg[1:]

    res /= nwin
    # apply Konno-Ohmachi smoothing taper if chosen
    if smooth > 0:
        # Write in one matrix for performance reasons.
        spectra = np.empty((2, len(res.real)))
        spectra[0] = res.real
        spectra[1] = res.imag
        new_spectra = \
            konno_ohmachi_smoothing(spectra, freq, bandwidth=smooth, count=1,
                                    max_memory_usage=1024, normalize=True)
        res.real = new_spectra[0]
        res.imag = new_spectra[1]

    amp = np.abs(res)
    # include phase unwrapping
    phase = np.unwrap(np.angle(res))  # + 2.0 * np.pi
    ra = np.abs(gg)
    rpha = np.unwrap(np.angle(gg))

    if save_data:
        trans_new = (st2[0].stats.station + "." + st2[0].stats.channel +
                     "." + str(window_len) + ".resp")
        trans_ref = st1[0].stats.station + ".refResp"
        # Create empty array for easy saving
        temp = np.empty((len(freq), 3))
        temp[:, 0] = freq
        temp[:, 1] = amp
        temp[:, 2] = phase
        np.savetxt(trans_new, temp, fmt=native_str('%.10f'))
        temp[:, 1] = ra
        temp[:, 2] = rpha
        np.savetxt(trans_ref, temp, fmt=native_str('%.10f'))

    return freq, amp, phase


def _calc_resp(calfile, nfft, sampfreq):
    """
    Calculate transfer function of known system.

    :type calfile: str
    :param calfile: file containing poles, zeros and scale factor for known
        system
    :returns: complex transfer function, array of frequencies
    """
    # calculate transfer function
    poles, zeros, scale_fac = read_paz(calfile)
    h, f = paz_to_freq_resp(poles, zeros, scale_fac, 1.0 / sampfreq,
                            nfft, freq=True)
    return h, f


# A modified copy of the Matplotlib 0.99.1.1 method spectral_helper found in
# .../matlab/mlab.py.
# Some function were changed to avoid additional dependencies. Included here as
# it is essential for the above rel_calib_stack function and only present in
#  recent matplotlib versions.

# This is a helper function that implements the commonality between the
# psd, csd, and spectrogram.  It is *NOT* meant to be used outside of mlab
def spectral_helper(x, y, NFFT=256, Fs=2, noverlap=0, pad_to=None,
                    sides='default', scale_by_freq=None):
    # The checks for if y is x are so that we can use the same function to
    # implement the core of psd(), csd(), and spectrogram() without doing
    # extra calculations.  We return the unaveraged Pxy, freqs, and t.
    same_data = y is x

    # Make sure we're dealing with a NumPy array. If y and x were the same
    # object to start with, keep them that way

    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)

    # zero pad x and y up to NFFT if they are shorter than NFFT
    if len(x) < NFFT:
        n = len(x)
        x = np.resize(x, (NFFT,))
        x[n:] = 0

    if not same_data and len(y) < NFFT:
        n = len(y)
        y = np.resize(y, (NFFT,))
        y[n:] = 0

    if pad_to is None:
        pad_to = NFFT

    if scale_by_freq is None:
        scale_by_freq = True

    # For real x, ignore the negative frequencies unless told otherwise
    if (sides == 'default' and np.iscomplexobj(x)) or sides == 'twosided':
        numFreqs = pad_to
        scaling_factor = 1.
    elif sides in ('default', 'onesided'):
        numFreqs = pad_to // 2 + 1
        scaling_factor = 2.
    else:
        raise ValueError("sides must be one of: 'default', 'onesided', or "
                         "'twosided'")

    # Matlab divides by the sampling frequency so that density function
    # has units of dB/Hz and can be integrated by the plotted frequency
    # values. Perform the same scaling here.
    if scale_by_freq:
        scaling_factor /= Fs

    windowVals = np.hanning(NFFT)

    step = int(NFFT) - int(noverlap)
    ind = np.arange(0, len(x) - NFFT + 1, step, dtype=np.int32)
    n = len(ind)
    Pxy = np.zeros((numFreqs, n), np.complex_)

    # do the ffts of the slices
    for i in range(n):
        thisX = x[ind[i]:ind[i] + NFFT]
        thisX = windowVals * thisX
        fx = np.fft.fft(thisX, n=pad_to)

        if same_data:
            fy = fx
        else:
            th_is_y = y[ind[i]:ind[i] + NFFT]
            th_is_y = windowVals * th_is_y
            fy = np.fft.fft(th_is_y, n=pad_to)
        Pxy[:, i] = np.conjugate(fx[:numFreqs]) * fy[:numFreqs]

    # Scale the spectrum by the norm of the window to compensate for
    # windowing loss; see Bendat & Piersol Sec 11.5.2.  Also include
    # scaling factors for one-sided densities and dividing by the sampling
    # frequency, if desired.
    Pxy *= scaling_factor / (np.abs(windowVals) ** 2).sum()
    t = 1. / Fs * (ind + NFFT / 2.)
    freqs = float(Fs) / pad_to * np.arange(numFreqs)

    if (np.iscomplexobj(x) and sides == 'default') or sides == 'twosided':
        # center the frequency range at zero
        freqs = np.concatenate((freqs[numFreqs // 2:] - Fs,
                                freqs[:numFreqs // 2]))
        Pxy = np.concatenate((Pxy[numFreqs // 2:, :],
                              Pxy[:numFreqs // 2, :]), 0)

    return Pxy, freqs, t
