#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: invsim.py
#  Purpose: Python Module for Instrument Correction (Seismology)
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2010 Moritz Beyreuther
#---------------------------------------------------------------------
""" 
Python Module for Instrument Correction (Seismology), PAZ
Poles and zeros information must be given in SEED convention, correction to
m/s.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import math as M
import numpy as np
import scipy.signal
import util


def cosTaper(npts, p=0.1):
    """
    Cosine Taper.

    >>> tap = cosTaper(100, 1.0)
    >>> tap2 = 0.5*(1+np.cos(np.linspace(np.pi,2*np.pi,50)))
    >>> (tap[0:50]==tap2).all()
    True
    >>> npts = 100
    >>> p = 0.1
    >>> tap3 = cosTaper(npts, p)
    >>> ( tap3[npts*p/2.:npts*(1-p/2.)]==np.ones(npts*(1-p)) ).all()
    True

    :type npts: Int
    :param npts: Number of points of cosine taper.
    :type p: Float
    :param p: Percent of cosine taper. Default is 10% which tapers 5% from
              the beginning and 5% form the end
    :rtype: float numpy ndarray
    :return: Cosine taper array/vector of length npts.
    """
    if p == 0.0 or p == 1.0:
        frac = int(npts * p / 2.0)
    else:
        frac = int(npts * p / 2.0) + 1
    # return concatenated beginning, middle and end
    return np.concatenate((
        0.5 * (1 + np.cos(np.linspace(np.pi, 2 * np.pi, frac))),
        np.ones(npts - 2 * frac),
        0.5 * (1 + np.cos(np.linspace(0, np.pi, frac)))
        ), axis=0)

def detrend(trace):
    """
    Inplace detrend signal simply by subtracting a line through the first
    and last point of the trace

    :param trace: Data to detrend
    """
    ndat = len(trace)
    x1, x2 = trace[0], trace[-1]
    trace -= (x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1))


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
    return {'poles':poles, 'zeros':[0j, 0j], 'gain':1}


def pazToFreqResp(poles, zeros, scale_fac, t_samp, nfft, freq=False):
    """
    Convert Poles and Zeros (PAZ) to frequency response. The output
    contains the frequency zero which is the offset of the trace.

    :note: In order to plot/calculate the phase you need to multiply the
        complex part by -1. This results from the different definition of
        the Fourier transform and the phase. The numpy.fft is defined as
        A(jw) = \int_{-\inf}^{+\inf} a(t) e^{-jwt}; where as the analytic
        signal is defined A(jw) = | A(jw) | e^{j\phi}. That is in order to
        calculate the phase the complex conjugate of the signal needs to be
        taken. E.g. phi = angle(f,conj(h),deg=True)
        As the range of phi is from -pi to pi you could add 2*pi to the
        negative values in order to get a plot from [0, 2pi]:
        where(phi<0,phi+2*pi,phi); plot(f,phi)

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
    """
    n = nfft // 2
    a, b = scipy.signal.ltisys.zpk2tf(zeros, poles, scale_fac)
    fy = 1 / (t_samp * 2.0)
    # start at zero to get zero for offset/ DC of fft
    f = np.arange(0, fy + fy / n, fy / n) #arange should includes fy/n
    _w, h = scipy.signal.freqs(a, b, f * 2 * np.pi)
    h = np.conj(h) # like in PITSA paz2Freq (insdeconv.c) last line
    h[-1] = h[-1].real + 0.0j
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
    # 0 sqrt_len will transform in np.nans when deviding by it
    idx = np.where((sqrt_len < swamp) & (sqrt_len > 0.0))
    spec[idx] *= swamp / sqrt_len[idx]
    found = len(idx[0])
    # Now invert the spectrum for values where sqrt_len is greater than
    # 0.0, see PITSA spr_sinv.c for details
    sqrt_len = np.abs(spec) # Find length of new scaled spec
    inn = np.where(sqrt_len > 0.0)
    spec[inn] = 1.0 / spec[inn]
    # For numerical stability, set all zero length to zero, do not invert
    spec[sqrt_len == 0.0] = complex(0.0, 0.0)
    return found


def seisSim(data, samp_rate, paz, inst_sim=None, water_level=600.0,
            no_inverse_filtering=False, zero_mean=True,
            taper_fraction=0.05):
    """
    Simulate/Correct seismometer.

    This function works in the frequency domain, where nfft is the next power 
    of len(data) to avoid warp around effects during convolution. The inverse 
    of the frequency response of the seismometer is convolved with the spectrum 
    of the data and with the frequency response of the seismometer to
    simulate. A 5% cosine taper is taken before simulation. The data must
    be detrended (e.g.) zero mean before hand. If inst_sim=None only the
    instrument correction is done.

    :type data: Numpy Ndarray
    :param data: Seismogram, detrend before hand (e.g. zero mean)
    :type samp_rate: Float
    :param samp_rate: Sample Rate of Seismogram
    :type paz: Dictionary
    :param paz: Dictionary containing keys 'poles', 'zeros',
                'gain' (A0 normalization factor). poles and zeros must be a
                list of complex floating point numbers, gain must be of
                type float. Poles and Zeros are assumed to correct to m/s,
                SEED convention.
    :type water_level: Float
    :param water_level: Water_Level for spectrum to simulate
    :type inst_sim: Dictionary, None
    :param inst_sim: Dictionary containing keys 'poles', 'zeros',
                     'gain'. Poles and zeros must be a list of complex
                     floating point numbers, gain must be of type float. Or
                     None for no simulation.
    :type no_inverse_filtering: Boolean
    :param no_inverse_filtering: Specifies if the removal of the prior
            frequency response should be skipped, i.e. only applying the
            given new frequency response.
    :type zero_mean: Boolean
    :param zero_mean: If true the mean of the data is subtracted
    :type taper_fraction: Float
    :param taper_fraction: Typer fraction of cosinus taper to use
    
    Pre-defined poles, zeros, gain dictionaries for instruments to simulate
    can be imported from obspy.signal.seismometer
    """
    # Translated from PITSA: spr_resg.c
    error = """
    %s must be either of type None or of type dictionary. The dictionary
    must contain poles, zeros and gain as keys, values of poles and zeros
    are iterables of complex entries, the value of gain is a float.
    """
    samp_int = 1 / float(samp_rate)
    try:
        poles = paz['poles']
        zeros = paz['zeros']
        gain = paz['gain']
    except:
        raise TypeError(error % 'paz')
    #
    ndat = len(data)
    # find next power of 2 in order to prohibit wrap around effects
    # during convolution, the number of points for the FFT has to be at
    # least 2 *ndat cf. Numerical Recipes p. 429 calculate next power
    # of 2
    nfft = util.nextpow2(2 * ndat)
    # explicitly copy, else input data will be modified (astype copies the
    # data). Avoid numerical problems by using float64
    tr = data.astype("float64")
    if zero_mean:
        tr -= tr.mean()
    tr *= cosTaper(ndat, taper_fraction)
    freq_response = pazToFreqResp(poles, zeros, gain, samp_int, nfft)
    specInv(freq_response, water_level)
    # transform trace in fourier domain
    tr = np.fft.rfft(tr, n=nfft)
    if not no_inverse_filtering:
        tr *= np.conj(freq_response)
    del freq_response
    #
    # now depending on inst_sim, simulate the seismometer
    if inst_sim is None:
        pass
    elif isinstance(inst_sim, dict):
        try:
            poles = inst_sim['poles']
            zeros = inst_sim['zeros']
            gain = inst_sim['gain']
        except:
            raise KeyError(error % 'inst_sim')
        tr *= np.conj(pazToFreqResp(poles, zeros, gain, samp_int, nfft))
    else:
        raise TypeError(error % 'inst_sim')
    # transfrom trace back into the time domain
    tr = np.fft.irfft(tr)[0:ndat]
    # linear detrend, 
    detrend(tr)
    return tr


def paz2AmpValueOfFreqResp(paz, freq):
    """
    Returns Amplitude at one frequency for the given poles and zeros

    The amplitude of the freq is estimated according to "Of Poles and
    Zeros", Frank Scherbaum, p 43.

    :param paz: Given poles and zeros
    :param freq: Given frequency

    >>> paz = {'poles': [-4.44 + 4.44j, -4.44 - 4.44j], \
               'zeros': [0 + 0j, 0 + 0j], \
               'gain': 0.4}
    >>> amp = paz2AmpValueOfFreqResp(paz, 1)
    >>> print round(amp, 7)
    0.2830262
    """
    jw = complex(0, 2 * np.pi * freq) #angular frequency
    fac = complex(1, 0)
    for zero in paz['zeros']: #numerator
        fac *= (jw - zero)
    for pole in paz['poles']: #denumerator
        fac /= (jw - pole)
    return abs(fac) * paz['gain']


def estimateMagnitude(paz, amplitude, timespan, h_dist):
    """
    Estimates local magnitude from poles and zeros of given instrument, the
    peak to peak amplitude and the period in which it is measured

    Magnitude estimation according to Bakun & Joyner, 1984, Eq. (3) page 1835.
    Bakun, W. H. and W. B. Joyner: The Ml scale in central California,
    Bull. Seismol. Soc. Am., 74, 1827-1843, 1984

    :param paz: PAZ of the instrument [m/s]
    :param amplitude: Peak to peak amplitude [counts]
    :param timespan: Timespan of peak to peak amplitude [s]
    :param h_dist: Hypocentral distance [km]

    >>> paz = {'poles': [-4.444 + 4.444j, -4.444 - 4.444j, -1.083 + 0j], \
               'zeros': [0 + 0j, 0 + 0j, 0 + 0j], \
               'gain': 1.0, \
               'sensitivity': 671140000.0}
    >>> mag = estimateMagnitude(paz, 3.34e6, 0.065, 0.255)
    >>> print round(mag, 6)
    2.165345
    """
    # Sensitivity is 2080 according to:
    # P. Bormann: New Manual of Seismological Observatory Practice
    # IASPEI Chapter 3, page 24
    woodander = {'poles': [-6.283 + 4.7124j, -6.283 - 4.7124],
                 'zeros': [0 + 0j],
                 'gain': 1.0,
                 'sensitivity': 2080} #iaspei, pitsa has 2800
    # analog to pitsa/plt/RCS/plt_wave.c,v, lines 4881-4891
    freq = 1 / (2 * timespan)
    wa_ampl = amplitude / 2.0 #half peak to peak amplitude
    wa_ampl /= (paz2AmpValueOfFreqResp(paz, freq) * paz['sensitivity'])
    wa_ampl *= paz2AmpValueOfFreqResp(woodander, freq) * woodander['sensitivity']
    wa_ampl *= 1000 #convert to mm
    magnitude = np.log10(wa_ampl) + np.log10(h_dist / 100.0) + \
                0.00301 * (h_dist - 100.0) + 3.0
    return magnitude


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
