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


GNU General Public License (GPL)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
USA.
"""

import math as M
import numpy as np
import scipy as S
import scipy.signal
import util


def cosTaper(npts, p):
    """
    Cosinus Taper.

    >>> tap = cosTaper(100,1.0)
    >>> tap2 = 0.5*(1+np.cos(np.linspace(np.pi,2*np.pi,50)))
    >>> (tap[0:50]==tap2).all()
    True
    >>> npts = 100
    >>> p = .1
    >>> tap3 = cosTaper(npts,p)
    >>> ( tap3[npts*p/2.:npts*(1-p/2.)]==np.ones(npts*(1-p)) ).all()
    True

    @type npts: Int
    @param npts: Number of points of cosinus taper.
    @type p: Float
    @param p: Percent of cosinus taper.
    @rtype: float numpy ndarray
    @return: Cosine taper array/vector of length npts.
    """
    #
    if p == 0.0 or p == 1.0:
        frac = int(npts * p / 2.0)
    else:
        frac = int(npts * p / 2.0) + 1
    return np.concatenate((
        0.5 * (1 + np.cos(np.linspace(np.pi, 2 * np.pi, frac))),
        np.ones(npts - 2 * frac),
        0.5 * (1 + np.cos(np.linspace(0, np.pi, frac)))
        ), axis=0)

def detrend(trace):
    """
    Inplace detrend signal simply by subtracting a line through the first
    and last point of the trace

    @param trace: Data to detrend
    """
    ndat = len(trace)
    x1, x2 = trace[0], trace[-1]
    trace -= (x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1))


def cornFreq2Paz(fc, damp=0.707):
    """
    Convert corner frequency and damping to poles and zeros. 2 zeros at
    postion (0j, 0j) are given as output  (m/s).

    @param fc: Corner frequency
    @param damping: Corner frequency
    @return: Dictionary containing poles, zeros and gain
    """
    poles = [-(damp + M.sqrt(1 - damp ** 2) * 1j) * 2 * np.pi * fc]
    poles.append(-(damp - M.sqrt(1 - damp ** 2) * 1j) * 2 * np.pi * fc)
    return {'poles':poles, 'zeros':[0j, 0j], 'gain':1}


def pazToFreqResp(poles, zeros, scale_fac, t_samp, nfft, freq=False):
    """
    Convert Poles and Zeros (PAZ) to frequency response. The output
    contains the frequency zero which is the offset of the trace.

    @note: In order to plot/calculate the phase you need to multiply the
        complex part by -1. This results from the different definition of
        the fourier transform and the phase. The numpy.fft is defined as
        A(jw) = \int_{-\inf}^{+\inf} a(t) e^{-jwt}; where as the analytic
        signal is defined A(jw) = | A(jw) | e^{j\phi}. That is in order to
        calculate the phase the complex conjugate of the signal needs to be
        taken. E.g. phi = angle(f,conj(h),deg=True)
        As the range of phi is from -pi to pi you could add 2*pi to the
        negative values in order to get a plot from [0, 2pi]:
        where(phi<0,phi+2*pi,phi); plot(f,phi)
    
    @type poles: List of complex numbers
    @param poles: The poles of the transfer function
    @type zeros: List of complex numbers
    @param zeros: The zeros of the transfer function
    @type scale_fac: Float
    @param scale_fac: Gain factor
    @type t_samp: Float
    @param t_samp: Sampling interval in seconds
    @type nfft: Integer
    @param nfft: Number of FFT points of signal which needs correction
    @rtype: numpy.ndarray complex128
    @return: Frequency response of PAZ of length nfft 
    """
    n = nfft // 2
    a, b = S.signal.ltisys.zpk2tf(zeros, poles, scale_fac)
    fy = 1 / (t_samp * 2.0)
    # start at zero to get zero for offset/ DC of fft
    f = np.arange(0, fy + fy / n, fy / n) #arange should includes fy/n
    _w, h = S.signal.freqs(a, b, f * 2 * np.pi)
    h = np.conj(h) # like in PITSA paz2Freq (insdeconv.c) last line
    h[-1] = h[-1].real + 0.0j
    if freq:
        return h, f
    return h


def specInv(spec, wlev):
    """
    Invert Spectrum and shrink values under WaterLevel. Fast version which
    uses array computation.

    @note: In place opertions on spec
    @param spec: real spectrum as returned by numpy.fft.rfft
    @param wlev: water level to use 
    """
    # Translated from PITSA: spr_sinv.c

    max_spec_amp = np.abs(spec).max()

    # swamp is the amplitude spectral value corresponding
    # to wlev dB below the maximum spectral value
    swamp = max_spec_amp * 10.0 ** (-wlev / 20.0)
    found = 0

    # Manually correct offset (index 0) and nyquist frequency (index -1)
    # replaced > 0 with > 1e-50
    for _i in (0, -1):
        if np.abs(spec[_i].real) < swamp:
            if np.abs(spec[_i].real) > 1e-50:
                real = 1. / swamp;
            else:
                real = 0.0;
            found += 1
        else:
            real = 1. / spec[_i].real
        spec[_i] = complex(real, 0.0)
    #
    # Do it for the rest by array computation in a slice (specs), i.e.
    # pointer to the orig.
    # Attention: Do not copy specs, else the pointer is lost
    specs = spec[1:-1]
    sqrt_len = abs(specs)
    # set/scale length to swamp, but leave phase untouched
    idx = np.where(sqrt_len < swamp)
    specs[idx] *= swamp / sqrt_len[idx]
    found += len(idx[0])
    # now do the inversion of the spectrum, this part is most probably
    # unnecessary but coded exqual to spr_sinv.c
    sqrt_len = abs(specs) # take new sqrt_len
    # replace > 0.0 with > 1e-50
    inn = np.where(sqrt_len > 1e-50)
    specs[inn] = 1.0 / specs[inn]
    specs[sqrt_len < 1e-50] = complex(0.0, 0.0)
    # spec is/was modified inplace => only found need to be returned
    return found


def seisSim(data, samp_rate, paz, inst_sim=None, water_level=600.0):
    """
    Simulate seismometer. 
    
    This function works in the frequency domain, where nfft is the next power 
    of len(data) to avoid warp around effects during convolution. The inverse 
    of the frequency response of the seismometer is convelved by the spectrum 
    of the data and convolved by the frequency response of the seismometer to 
    simulate.
    
    @type data: Numpy Ndarray
    @param data: Seismogram, (zero mean?)
    @type samp_rate: Float
    @param samp_rate: Sample Rate of Seismogram
    @type paz: Dictionary
    @param paz: Dictionary containing keys 'poles', 'zeros',
    'gain'. poles and zeros must be a list of complex floating point
    numbers, gain must be of type float. Poles and Zeros are assumed to
    correct to m/s, SEED convention.
    @type water_level: Float
    @param water_level: Water_Level for spectrum to simulate
    @type inst_sim: Dictionary, None
    @param inst_sim: Dictionary containing keys 'poles', 'zeros',
        'gain'. Poles and zeros must be a list of complex floating point
        numbers, gain must be of type float. Or None for no simulation.
    
    Ready to go poles, zeros, gain dictionaries for instruments to simulate
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
    # explicitly copy, else input data will be modified
    tr = data * cosTaper(ndat, 0.05)
    freq_response = pazToFreqResp(poles, zeros, gain, samp_int, nfft)
    found = specInv(freq_response, water_level)
    # transform trace in fourier domain
    tr = np.fft.rfft(tr, n=nfft)
    tr *= np.conj(freq_response)
    del freq_response
    #
    # now depending on inst_sim, simulate the seismometer
    if isinstance(inst_sim, type(None)):
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
