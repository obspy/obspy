#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: invsim.py
#  Purpose: Python Module for Instrument Correction (Seismology)
#   Author: Moritz Beyreuther, Yannik Behr
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2012 Moritz Beyreuther, Yannik Behr
#---------------------------------------------------------------------
"""
Python Module for Instrument Correction (Seismology).
PAZ (Poles and zeros) information must be given in SEED convention, correction
to m/s.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.util.base import NamedTemporaryFile
from obspy.core.util.decorator import deprecated_keywords
from obspy.signal.detrend import simple as simpleDetrend
from obspy.signal.headers import clibevresp
import ctypes as C
import math as M
import numpy as np
import os
import scipy.signal
import util
import warnings


# Sensitivity is 2080 according to:
# P. Bormann: New Manual of Seismological Observatory Practice
# IASPEI Chapter 3, page 24
# (PITSA has 2800)
WOODANDERSON = {'poles': [-6.283 + 4.7124j, -6.283 - 4.7124j],
                'zeros': [0 + 0j], 'gain': 1.0, 'sensitivity': 2080}


def cosTaper(npts, p=0.1, freqs=None, flimit=None, halfcosine=True,
             sactaper=False):
    """
    Cosine Taper.

    :type npts: Int
    :param npts: Number of points of cosine taper.
    :type p: Float
    :param p: Decimal percentage of cosine taper (ranging from 0 to 1). Default
        is 0.1 (10%) which tapers 5% from the beginning and 5% form the end.
    :rtype: float NumPy ndarray
    :return: Cosine taper array/vector of length npts.
    :type freqs: NumPy ndarray
    :param freqs: Frequencies as, for example, returned by fftfreq
    :type flimit: List or tuple of floats
    :param flimit: The list or tuple defines the four corner frequencies
        (f1, f2, f3, f4) of the cosine taper which is one between f2 and f3 and
        tapers to zero for f1 < f < f2 and f3 < f < f4.
    :type halfcosine: Boolean
    :param halfcosine: If True the taper is a half cosine function. If False it
        is a quarter cosine function.
    :type sactaper: Boolean
    :param sactaper: If set to True the cosine taper already tapers at the
        corner frequency (SAC behaviour). By default, the taper has a value
        of 1.0 at the corner frequencies.

    .. rubric:: Example

    >>> tap = cosTaper(100, 1.0)
    >>> tap2 = 0.5 * (1 + np.cos(np.linspace(np.pi, 2 * np.pi, 50)))
    >>> np.allclose(tap[0:50], tap2)
    True
    >>> npts = 100
    >>> p = 0.1
    >>> tap3 = cosTaper(npts, p)
    >>> ( tap3[npts*p/2.:npts*(1-p/2.)]==np.ones(npts*(1-p)) ).all()
    True
    """
    if p < 0 or p > 1:
        msg = "Decimal taper percentage must be between 0 and 1."
        raise ValueError(msg)
    if p == 0.0 or p == 1.0:
        frac = int(npts * p / 2.0)
    else:
        frac = int(npts * p / 2.0 + 0.5)

    if freqs is not None and flimit is not None:
        fl1, fl2, fl3, fl4 = flimit
        idx1 = np.argmin(abs(freqs - fl1))
        idx2 = np.argmin(abs(freqs - fl2))
        idx3 = np.argmin(abs(freqs - fl3))
        idx4 = np.argmin(abs(freqs - fl4))
    else:
        idx1 = 0
        idx2 = frac - 1
        idx3 = npts - frac
        idx4 = npts - 1
    if sactaper:
        # in SAC the second and third
        # index are already tapered
        idx2 += 1
        idx3 -= 1

    # Very small data lengths or small decimal taper percentages can result in
    # idx1 == idx2 and idx3 == idx4. This breaks the following calculations.
    if idx1 == idx2:
        idx2 += 1
    if idx3 == idx4:
        idx3 -= 1

    # the taper at idx1 and idx4 equals zero and
    # at idx2 and idx3 equals one
    cos_win = np.zeros(npts)
    if halfcosine:
        #cos_win[idx1:idx2+1] =  0.5 * (1.0 + np.cos((np.pi * \
        #    (idx2 - np.arange(idx1, idx2+1)) / (idx2 - idx1))))
        cos_win[idx1:idx2 + 1] = 0.5 * (1.0 - np.cos((np.pi * \
            (np.arange(idx1, idx2 + 1) - idx1) / (idx2 - idx1))))
        cos_win[idx2 + 1:idx3] = 1.0
        cos_win[idx3:idx4 + 1] = 0.5 * (1.0 + np.cos((np.pi * \
            (idx3 - np.arange(idx3, idx4 + 1)) / (idx4 - idx3))))
    else:
        cos_win[idx1:idx2 + 1] = np.cos(-(np.pi / 2.0 * \
               (idx2 - np.arange(idx1, idx2 + 1)) / (idx2 - idx1)))
        cos_win[idx2 + 1:idx3] = 1.0
        cos_win[idx3:idx4 + 1] = np.cos((np.pi / 2.0 * \
            (idx3 - np.arange(idx3, idx4 + 1)) / (idx4 - idx3)))

    # if indices are identical division by zero
    # causes NaN values in cos_win
    if idx1 == idx2:
        cos_win[idx1] = 0.0
    if idx3 == idx4:
        cos_win[idx3] = 0.0
    return cos_win


def c_sac_taper(npts, p=0.1, freqs=None, flimit=None, pitsa=False):
    twopi = 6.283185307179586
    dblepi = 0.5 * twopi
    fl1, fl2, fl3, fl4 = flimit
    taper = []
    for freq in freqs:
        if freq < fl3 and freq > fl2:
            taper_v = 1.0
        if freq >= fl3 and freq <= fl4:
            taper_v = 0.5 * (1.0 + M.cos(dblepi * (freq - fl3) / (fl4 - fl3)))
        if freq > fl4 or freq < fl1:
            taper_v = 0.0
        if freq >= fl1 and freq <= fl2:
            taper_v = 0.5 * (1.0 - M.cos(dblepi * (freq - fl1) / (fl2 - fl1)))
        taper.append(taper_v)
    return np.array(taper)


@deprecated_keywords({'pitsa': None})
def evalresp(t_samp, nfft, filename, date, station='*', channel='*',
             network='*', locid='*', units="VEL", freq=False,
             debug=False):
    """
    Use the evalresp library to extract instrument response
    information from a SEED RESP-file.

    :type t_samp: float
    :param t_samp: Sampling interval in seconds
    :type nfft: int
    :param nfft: Number of FFT points of signal which needs correction
    :type filename: str
    :param filename: SEED RESP-filename or content of RESP file
    :type date: UTCDateTime
    :param date: Date of interest
    :type station: str
    :param station: Station id
    :type channel: str
    :param channel: Channel id
    :type network: str
    :param network: Network id
    :type locid: str
    :param locid: Location id
    :type units: str
    :param units: Units to return response in. Can be either DIS, VEL or ACC
    :type debug: bool
    :param debug: Verbose output to stdout. Disabled by default.
    :rtype: numpy.ndarray complex128
    :return: Frequency response from SEED RESP-file of length nfft
    """
    # evalresp needs files with correct line separators depending on OS
    data = open(filename, 'rb').read()
    fh = NamedTemporaryFile()
    tempfile = fh.name
    fh.write(os.linesep.join(data.splitlines()))
    fh.close()

    fy = 1 / (t_samp * 2.0)
    # start at zero to get zero for offset/ DC of fft
    freqs = np.linspace(0, fy, nfft // 2 + 1)
    start_stage = C.c_int(-1)
    stop_stage = C.c_int(0)
    stdio_flag = C.c_int(0)
    sta = C.create_string_buffer(station)
    cha = C.create_string_buffer(channel)
    net = C.create_string_buffer(network)
    locid = C.create_string_buffer(locid)
    unts = C.create_string_buffer(units)
    if debug:
        vbs = C.create_string_buffer("-v")
    else:
        vbs = C.create_string_buffer("")
    rtyp = C.create_string_buffer("CS")
    datime = C.create_string_buffer("%d,%3d" % (date.year, date.julday))
    fn = C.create_string_buffer(tempfile)
    nfreqs = C.c_int(freqs.shape[0])
    res = clibevresp.evresp(sta, cha, net, locid, datime, unts, fn,
                            freqs, nfreqs, rtyp, vbs, start_stage,
                            stop_stage, stdio_flag, C.c_int(0))
    # optimizing performance, see
    # http://wiki.python.org/moin/PythonSpeed/PerformanceTips
    nfreqs, rfreqs, rvec = res[0].nfreqs, res[0].freqs, res[0].rvec
    h = np.empty(nfreqs, dtype='complex128')
    f = np.empty(nfreqs, dtype='float64')
    for i in xrange(nfreqs):
        h[i] = rvec[i].real + rvec[i].imag * 1j
        f[i] = rfreqs[i]
    clibevresp.free_response(res)
    del nfreqs, rfreqs, rvec, res
    # delete temporary file
    try:
        os.remove(tempfile)
    except:
        pass
    if freq:
        return h, f
    return h


def cornFreq2Paz(fc, damp=0.707):
    """
    Convert corner frequency and damping to poles and zeros. 2 zeros at
    position (0j, 0j) are given as output  (m/s).

    :param fc: Corner frequency
    :param damping: Corner frequency
    :return: Dictionary containing poles, zeros and gain
    """
    poles = [-(damp + M.sqrt(1 - damp ** 2) * 1j) * 2 * np.pi * fc]
    poles.append(-(damp - M.sqrt(1 - damp ** 2) * 1j) * 2 * np.pi * fc)
    return {'poles': poles, 'zeros': [0j, 0j], 'gain': 1, 'sensitivity': 1.0}


@deprecated_keywords({'pitsa': None})
def pazToFreqResp(poles, zeros, scale_fac, t_samp, nfft, freq=False):
    """
    Convert Poles and Zeros (PAZ) to frequency response. The output
    contains the frequency zero which is the offset of the trace.

    :type poles: List of complex numbers
    :param poles: The poles of the transfer function
    :type zeros: List of complex numbers
    :param zeros: The zeros of the transfer function
    :type scale_fac: Float
    :param scale_fac: Gain factor
    :type t_samp: Float
    :param t_samp: Sampling interval in seconds
    :type nfft: Integer
    :param nfft: Number of FFT points of signal which needs correction
    :rtype: numpy.ndarray complex128
    :return: Frequency response of PAZ of length nfft

    .. note::
        In order to plot/calculate the phase you need to multiply the
        complex part by -1. This results from the different definition of
        the Fourier transform and the phase. The numpy.fft is defined as
        A(jw) = \int_{-\inf}^{+\inf} a(t) e^{-jwt}; where as the analytic
        signal is defined A(jw) = | A(jw) | e^{j\phi}. That is in order to
        calculate the phase the complex conjugate of the signal needs to be
        taken. E.g. phi = angle(f,conj(h),deg=True)
        As the range of phi is from -pi to pi you could add 2*pi to the
        negative values in order to get a plot from [0, 2pi]:
        where(phi<0,phi+2*pi,phi); plot(f,phi)
    """
    n = nfft // 2
    b, a = scipy.signal.ltisys.zpk2tf(zeros, poles, scale_fac)
    # a has to be a list for the scipy.signal.freqs() call later but zpk2tf()
    # strangely returns it as an integer.
    if not isinstance(a, np.ndarray) and a == 1.0:
        a = [1.0]
    fy = 1 / (t_samp * 2.0)
    # start at zero to get zero for offset / DC of fft
    f = np.linspace(0, fy, n + 1)
    _w, h = scipy.signal.freqs(b, a, f * 2 * np.pi)
    if freq:
        return h, f
    return h


def waterlevel(spec, wlev):
    """
    Return the absolute spectral value corresponding
    to dB wlev in spectrum spec.

    :param spec: The spectrum
    :param wlev: The water level
    """
    return np.abs(spec).max() * 10.0 ** (-wlev / 20.0)


def specInv(spec, wlev):
    """
    Invert Spectrum and shrink values under water-level of max spec
    amplitude. The water-level is given in db scale.

    :note: In place operations on spec, translated from PITSA spr_sinv.c
    :param spec: Spectrum as returned by numpy.fft.rfft
    :param wlev: Water level to use
    """
    # Calculated waterlevel in the scale of spec
    swamp = waterlevel(spec, wlev)

    # Find length in real fft frequency domain, spec is complex
    sqrt_len = np.abs(spec)
    # Set/scale length to swamp, but leave phase untouched
    # 0 sqrt_len will transform in np.nans when dividing by it
    idx = np.where((sqrt_len < swamp) & (sqrt_len > 0.0))
    spec[idx] *= swamp / sqrt_len[idx]
    found = len(idx[0])
    # Now invert the spectrum for values where sqrt_len is greater than
    # 0.0, see PITSA spr_sinv.c for details
    sqrt_len = np.abs(spec)  # Find length of new scaled spec
    inn = np.where(sqrt_len > 0.0)
    spec[inn] = 1.0 / spec[inn]
    # For numerical stability, set all zero length to zero, do not invert
    spec[sqrt_len == 0.0] = complex(0.0, 0.0)
    return found


def seisSim(data, samp_rate, paz_remove=None, paz_simulate=None,
            remove_sensitivity=True, simulate_sensitivity=True,
            water_level=600.0, zero_mean=True, taper=True,
            taper_fraction=0.05, pre_filt=None, seedresp=None,
            nfft_pow2=False, pitsasim=True, sacsim=False, shsim=False,
            **_kwargs):
    """
    Simulate/Correct seismometer.

    :type data: NumPy ndarray
    :param data: Seismogram, detrend before hand (e.g. zero mean)
    :type samp_rate: Float
    :param samp_rate: Sample Rate of Seismogram
    :type paz_remove: Dictionary, None
    :param paz_remove: Dictionary containing keys 'poles', 'zeros', 'gain'
        (A0 normalization factor). poles and zeros must be a list of complex
        floating point numbers, gain must be of type float. Poles and Zeros are
        assumed to correct to m/s, SEED convention. Use None for no inverse
        filtering.
    :type paz_simulate: Dictionary, None
    :param paz_simulate: Dictionary containing keys 'poles', 'zeros', 'gain'.
        Poles and zeros must be a list of complex floating point numbers, gain
        must be of type float. Or None for no simulation.
    :type remove_sensitivity: Boolean
    :param remove_sensitivity: Determines if data is divided by
        `paz_remove['sensitivity']` to correct for overall sensitivity of
        recording instrument (seismometer/digitizer) during instrument
        correction.
    :type simulate_sensitivity: Boolean
    :param simulate_sensitivity: Determines if data is multiplied with
        `paz_simulate['sensitivity']` to simulate overall sensitivity of
        new instrument (seismometer/digitizer) during instrument simulation.
    :type water_level: Float
    :param water_level: Water_Level for spectrum to simulate
    :type zero_mean: Boolean
    :param zero_mean: If true the mean of the data is subtracted
    :type taper: Boolean
    :param taper: If true a cosine taper is applied.
    :type taper_fraction: Float
    :param taper_fraction: Taper fraction of cosine taper to use
    :type pre_filt: List or tuple of floats
    :param pre_filt: Apply a bandpass filter to the data trace before
        deconvolution. The list or tuple defines the four corner frequencies
        (f1,f2,f3,f4) of a cosine taper which is one between f2 and f3 and
        tapers to zero for f1 < f < f2 and f3 < f < f4.
    :type seedresp: Dictionary, None
    :param seedresp: Dictionary contains keys 'filename', 'date', 'units'.
        'filename' is the path to a RESP-file generated from a dataless SEED
        volume;
        'date' is a `~obspy.core.utcdatetime.UTCDateTime` object for the date
        that the response function should be extracted for;
        'units' defines the units of the response function.
        Can be either 'DIS', 'VEL' or 'ACC'.
    :type nfft_pow2: Boolean
    :param nfft_pow2: Number of frequency points to use for FFT. If True,
        the exact power of two is taken (default in PITSA). If False the
        data are not zeropadded to the next power of two which makes a
        slower FFT but is then much faster for e.g. evalresp which scales
        with the FFT points.
    :type pitsasim: Boolean
    :param pitsasim: Choose parameters to match
        instrument correction as done by PITSA.
    :type sacsim: Boolean
    :param sacsim: Choose parameters to match
        instrument correction as done by SAC.
    :type shsim: Boolean
    :param shsim: Choose parameters to match
        instrument correction as done by Seismic Handler.
    :return: The corrected data are returned as numpy.ndarray float64
        array. float64 is chosen to avoid numerical instabilities.

    This function works in the frequency domain, where nfft is the next power
    of len(data) to avoid wrap around effects during convolution. The inverse
    of the frequency response of the seismometer (``paz_remove``) is
    convolved with the spectrum of the data and with the frequency response
    of the seismometer to simulate (``paz_simulate``). A 5% cosine taper is
    taken before simulation. The data must be detrended (e.g.) zero mean
    beforehand. If paz_simulate=None only the instrument correction is done.
    In the latter case, a broadband filter can be applied to the data trace
    using pre_filt. This restricts the signal to the valid frequency band and
    thereby avoids artefacts due to amplification of frequencies outside of the
    instrument's passband (for a detailed discussion see
    *Of Poles and Zeros*, F. Scherbaum, Kluwer Academic Publishers).

    .. versionchanged:: 0.5.1
        The default for `remove_sensitivity` and `simulate_sensitivity` has
        been changed to ``True``. Old deprecated keyword arguments `paz`,
        `inst_sim`, `no_inverse_filtering` have been removed.
    """
    # Checking the types
    if not paz_remove and not paz_simulate and not seedresp:
        msg = "Neither inverse nor forward instrument simulation specified."
        raise TypeError(msg)

    for d in [paz_remove, paz_simulate]:
        if d is None:
            continue
        for key in ['poles', 'zeros', 'gain']:
            if key not in d:
                raise KeyError("Missing key: %s" % key)
    # Translated from PITSA: spr_resg.c
    delta = 1.0 / samp_rate
    #
    ndat = len(data)
    data = data.astype("float64")
    if zero_mean:
        data -= data.mean()
    if taper:
        if sacsim:
            data *= cosTaper(ndat, taper_fraction,
                             sactaper=sacsim, halfcosine=False)
        else:
            data *= cosTaper(ndat, taper_fraction)
    # The number of points for the FFT has to be at least 2 * ndat (in
    # order to prohibit wrap around effects during convolution) cf.
    # Numerical Recipes p. 429 calculate next power of 2.
    if nfft_pow2:
        nfft = util.nextpow2(2 * ndat)
    # evalresp scales directly with nfft, therefor taking the next power of
    # two has a greater negative performance impact than the slow down of a
    # not power of two in the FFT
    elif ndat & 0x1:  # check if uneven
        nfft = 2 * (ndat + 1)
    else:
        nfft = 2 * ndat
    # Transform data in Fourier domain
    data = np.fft.rfft(data, n=nfft)
    # Inverse filtering = Instrument correction
    if paz_remove:
        freq_response, freqs = pazToFreqResp(paz_remove['poles'],
                                             paz_remove['zeros'],
                                             paz_remove['gain'], delta, nfft,
                                             freq=True)
    if seedresp:
        freq_response, freqs = evalresp(delta, nfft, seedresp['filename'],
                                        seedresp['date'],
                                        units=seedresp['units'], freq=True)
        if not remove_sensitivity:
            msg = "remove_sensitivity is set to False, but since seedresp " + \
                  "is selected the overall sensitivity will be corrected " + \
                  " for anyway!"
            warnings.warn(msg)
    if paz_remove or seedresp:
        if pre_filt:
            # make cosine taper
            fl1, fl2, fl3, fl4 = pre_filt
            if sacsim:
                cos_win = c_sac_taper(freqs.size, freqs=freqs,
                                      flimit=(fl1, fl2, fl3, fl4))
            else:
                cos_win = cosTaper(freqs.size, freqs=freqs,
                                   flimit=(fl1, fl2, fl3, fl4))
            data *= cos_win
        specInv(freq_response, water_level)
        data *= freq_response
        del freq_response
    # Forward filtering = Instrument simulation
    if paz_simulate:
        data *= pazToFreqResp(paz_simulate['poles'],
                paz_simulate['zeros'], paz_simulate['gain'], delta, nfft)

    data[-1] = abs(data[-1]) + 0.0j
    # transform data back into the time domain
    data = np.fft.irfft(data)[0:ndat]
    if pitsasim:
        # linear detrend
        data = simpleDetrend(data)
    if shsim:
        # detrend using least squares
        data = scipy.signal.detrend(data, type="linear")
    # correct for involved overall sensitivities
    if paz_remove and remove_sensitivity and not seedresp:
        data /= paz_remove['sensitivity']
    if paz_simulate and simulate_sensitivity:
        data *= paz_simulate['sensitivity']
    return data


def paz2AmpValueOfFreqResp(paz, freq):
    """
    Returns Amplitude at one frequency for the given poles and zeros

    :param paz: Given poles and zeros
    :param freq: Given frequency

    The amplitude of the freq is estimated according to "Of Poles and
    Zeros", Frank Scherbaum, p 43.

    .. rubric:: Example

    >>> paz = {'poles': [-4.44 + 4.44j, -4.44 - 4.44j],
    ...        'zeros': [0 + 0j, 0 + 0j],
    ...        'gain': 0.4}
    >>> amp = paz2AmpValueOfFreqResp(paz, 1)
    >>> print(round(amp, 7))
    0.2830262
    """
    jw = complex(0, 2 * np.pi * freq)  # angular frequency
    fac = complex(1, 0)
    for zero in paz['zeros']:  # numerator
        fac *= jw - zero
    for pole in paz['poles']:  # denominator
        fac /= jw - pole
    return abs(fac) * paz['gain']


def estimateMagnitude(paz, amplitude, timespan, h_dist):
    """
    Estimates local magnitude from poles and zeros of given instrument, the
    peak to peak amplitude and the time span from peak to peak.
    Readings on two components can be used in magnitude estimation by providing
    lists for ``paz``, ``amplitude`` and ``timespan``.

    :param paz: PAZ of the instrument [m/s] or list of the same
    :param amplitude: Peak to peak amplitude [counts] or list of the same
    :param timespan: Timespan of peak to peak amplitude [s] or list of the same
    :param h_dist: Hypocentral distance [km]
    :returns: Estimated local magnitude Ml

    .. note::
        Magnitude estimation according to Bakun & Joyner, 1984, Eq. (3) page
        1835. Bakun, W. H. and W. B. Joyner: The Ml scale in central
        California, Bull. Seismol. Soc. Am., 74, 1827-1843, 1984

    .. rubric:: Example

    >>> paz = {'poles': [-4.444+4.444j, -4.444-4.444j, -1.083+0j],
    ...        'zeros': [0+0j, 0+0j, 0+0j],
    ...        'gain': 1.0, 'sensitivity': 671140000.0}
    >>> mag = estimateMagnitude(paz, 3.34e6, 0.065, 0.255)
    >>> print(round(mag, 6))
    2.132873
    >>> mag = estimateMagnitude([paz, paz], [3.34e6, 5e6], [0.065, 0.1], 0.255)
    >>> print(round(mag, 6))
    2.347618
    """
    # convert input to lists
    if not isinstance(paz, list) and not isinstance(paz, tuple):
        paz = [paz]
    if not isinstance(amplitude, list) and not isinstance(amplitude, tuple):
        amplitude = [amplitude]
    if not isinstance(timespan, list) and not isinstance(timespan, tuple):
        timespan = [timespan]
    # convert every input amplitude to Wood Anderson and calculate the mean
    wa_ampl_mean = 0.0
    count = 0
    for paz, amplitude, timespan in zip(paz, amplitude, timespan):
        wa_ampl_mean += estimateWoodAndersonAmplitude(paz, amplitude, timespan)
        count += 1
    wa_ampl_mean /= count
    # mean of input amplitudes (if more than one) should be used in final
    # magnitude estimation (usually N and E components)
    magnitude = np.log10(wa_ampl_mean) + np.log10(h_dist / 100.0) + \
                0.00301 * (h_dist - 100.0) + 3.0
    return magnitude


def estimateWoodAndersonAmplitude(paz, amplitude, timespan):
    """
    Convert amplitude in counts measured of instrument with given Poles and
    Zeros information for use in :func:`estimateMagnitude`.
    Amplitude should be measured as full peak to peak amplitude, timespan as
    difference of the two readings.

    :param paz: PAZ of the instrument [m/s] or list of the same
    :param amplitude: Peak to peak amplitude [counts] or list of the same
    :param timespan: Timespan of peak to peak amplitude [s] or list of the same
    :returns: Simulated zero to peak displacement amplitude on Wood Anderson
        seismometer [mm] for use in local magnitude estimation.
    """
    # analog to pitsa/plt/RCS/plt_wave.c,v, lines 4881-4891
    freq = 1.0 / (2 * timespan)
    wa_ampl = amplitude / 2.0  # half peak to peak amplitude
    wa_ampl /= (paz2AmpValueOfFreqResp(paz, freq) * paz['sensitivity'])
    wa_ampl *= paz2AmpValueOfFreqResp(WOODANDERSON, freq) * \
            WOODANDERSON['sensitivity']
    wa_ampl *= 1000  # convert to mm
    return wa_ampl


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
