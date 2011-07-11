#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: calibration.py
#  Purpose: Functions for relative calibration (e.g. Huddle test calibration)
#   Author: Felix Bernauer, Simon Kremers
#    Email: bernauer@geophysik.uni-muenchen.de
#
# Copyright (C) 2011 Felix Bernauer, Simon Kremers
#---------------------------------------------------------------------
"""
Functions for relative calibration

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from obspy.signal.util import nextpow2
from obspy.signal import konnoOhmachiSmoothing
from matplotlib.mlab import _spectral_helper as spectral_helper


def relcalstack(st1, st2, calib_file, window_len, OverlapFrac=0.5, smooth=0):
    """
    Method for relative calibration of sensors using a sensor with known
    transfer function

    :param st1: Stream object, (known) the trace.stats dict like class must
        contain the parameters "sampling_rate", "npts" and "station"
    :param st2: Stream object, (unknown) the trace.stats dict like class must
        contain the parameters "sampling_rate", "npts" and "station"
    :type calib_file: String
    :param calib_file: name of calib file containing the known PAZ of known
        instrument in GSE2 standard.
    :type window_len: Float
    :param window_len: length of sliding window in seconds
    :type OverlapFrac: float
    :param OverlapFrac: fraction of overlap, defaults to fifty percent (0.5)
    :type smooth: Float
    :param smooth: variable that defines if the Konno-Ohmachi taper is used or
        not. default = 0 -> no taper generally used in geopsy: smooth = 40
    :returns: frequency, amplitude and phase spectrum

    implemented after relcalstack.c by M.Ohrnberger and J.Wassermann.
    """
    # check Konno-Ohmachi
    if smooth < 0:
        smooth = 0

    # check if sampling rate and trace length is the same
    if st1[0].stats.npts != st2[0].stats.npts:
        msg = 'Traces dont have the same length!'
        raise ValueError(msg)
    elif st1[0].stats.sampling_rate != st2[0].stats.sampling_rate:
        msg = 'Traces dont have the same sampling rate!'
        raise ValueError(msg)
    else:
        ndat1 = st1[0].stats.npts
        sampfreq = st1[0].stats.sampling_rate

    # read waveforms
    tr1 = st1[0].data.astype(np.float64)
    tr2 = st2[0].data.astype(np.float64)

    # get window length, nfft and frequency step
    ndat = int(window_len * sampfreq)
    nfft = nextpow2(ndat)

    # initialize array for response function
    res = np.zeros(nfft / 2 + 1, dtype='complex128')

    # read calib file and calculate response function
    gg, _freq = calcresp(calib_file, nfft, sampfreq)

    # calculate number of windows and overlap
    nwin = int(np.floor((ndat1 - nfft) / (nfft / 2)) + 1)
    noverlap = nfft * OverlapFrac

    auto, _freq, _t = \
        spectral_helper(tr1, tr1, NFFT=nfft, Fs=sampfreq, noverlap=noverlap)
    cross, freq, _t = \
        spectral_helper(tr1, tr2, NFFT=nfft, Fs=sampfreq, noverlap=noverlap)

    # 180 Grad Phasenverschiebung
    cross.imag = -cross.imag

    for i in range(nwin):
        res += (cross[:, i] / auto[:, i]) * gg

    # The first item might be zero. Problems with phase calculations.
    res = res[1:]
    freq = freq[1:]
    gg = gg[1:]

    # apply Konno-Ohmachi smoothing taper if chosen
    if smooth > 0:
        res /= nwin

        # Write in one matrix for performance reasons.
        spectra = np.empty((2, len(res.real)))
        spectra[0] = res.real
        spectra[1] = res.imag

        new_spectra = konnoOhmachiSmoothing(spectra, freq, bandwidth=smooth,
                count=1, max_memory_usage=1024, normalize=True)

        res.real = new_spectra[0]
        res.imag = new_spectra[1]

    else:
        res /= nwin

    trans_new = st2[0].stats.station + "." + str(window_len) + ".resp"
    trans_ref = st1[0].stats.station + ".refResp"

    amp = np.abs(res)
    phase = np.arctan(res.imag / res.real)
    ra = np.abs(gg)
    rpha = np.arctan(gg.imag / gg.real)

    # Create empty array for easy saving
    temp = np.empty((len(freq), 3))
    temp[:, 0] = freq
    temp[:, 1] = amp
    temp[:, 2] = phase
    np.savetxt(trans_new, temp, fmt="%.10f")
    temp[:, 1] = ra
    temp[:, 2] = rpha
    np.savetxt(trans_ref, temp, fmt="%.10f")

    return freq, amp, phase


def calcresp(calfile, nfft, sampfreq):
    """
    calculate transfer function of known system
    
    :type calfile: String
    :param calfile: file containing poles, zeros and scale factor for known
        system
    :returns: complex transfer function, array of frequencies
    """
    buffer = np.empty(nfft / 2 + 1, dtype='complex128')
    poles = []
    zeros = []
    file = open(str(calfile), 'r')

    # read file until calibration section is found
    print '...reading calibration file'
    text = ' '
    while text != 'CAL1':
        textln = file.readline()
        text = textln.split(' ')[0]
    if not text == 'CAL1':
        msg = 'could not find calibration section!'
        raise NameError(msg)
    else:
        cal = textln[31:34]
    if cal == 'PAZ':
        # read poles
        npoles = int(file.readline())
        for i in xrange(npoles):
            pole = file.readline()
            pole_r = float(pole.split(" ")[0])
            pole_i = float(pole.split(" ")[1])
            pole_c = pole_r + pole_i * 1.j
            poles.append(pole_c)
        # read zeros
        nzeros = int(file.readline())
        for i in xrange(nzeros):
            zero = file.readline()
            zero_r = float(zero.split(" ")[0])
            zero_i = float(zero.split(" ")[1])
            zero_c = zero_r + zero_i * 1.j
            zeros.append(zero_c)
        # read scale factor
        scale_fac = float(file.readline())
        file.close

        # calculate transfer function
        delta_f = sampfreq / nfft
        F = np.empty(nfft / 2 + 1)
        for i in xrange(nfft / 2 + 1):
            fr = i * delta_f
            F[i] = fr
            om = 2 * np.pi * fr
            num = 1. + 0.j

            for ii in xrange(nzeros):
                s = 0. + om * 1.j
                dif = s - zeros[ii]
                num = dif * num

            denom = 1. + 0.j
            for ii in xrange(npoles):
                s = 0. + om * 1.j
                dif = s - poles[ii]
                denom = dif * denom

            t_om = 1. + 0.j
            if denom.real != 0. or denom.imag != 0.:
                t_om = num / denom

            t_om *= scale_fac

            if i < nfft / 2 and i > 0:
                buffer[i] = t_om

            if i == 0:
                buffer[i] = t_om + 0.j

            if i == nfft / 2:
                buffer[i] = t_om + 0.j

        return buffer, F

    else:
        msg = '%s type not known!' % (cal)
        raise NameError(msg)
