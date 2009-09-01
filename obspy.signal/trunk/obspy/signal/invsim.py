#!/usr/bin/python
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
import numpy as N
import scipy as S


def cosTaper(npts, p):
    """
    Cosinus Taper.

    >>> tap = cosTaper(100,1.0)
    >>> tap2 = 0.5*(1+N.cos(N.linspace(N.pi,2*N.pi,50)))
    >>> (tap[0:50]==tap2).all()
    True
    >>> npts = 100
    >>> p = .1
    >>> tap3 = cosTaper(npts,p)
    >>> ( tap3[npts*p/2.:npts*(1-p/2.)]==N.ones(npts*(1-p)) ).all()
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
    return N.concatenate((
        0.5 * (1 + N.cos(N.linspace(N.pi, 2 * N.pi, frac))),
        N.ones(npts - 2 * frac),
        0.5 * (1 + N.cos(N.linspace(0, N.pi, frac)))
        ), axis=0)

def detrend(trace):
    """
    Inplace detrend signal simply by subtracting a line through the first
    and last point of the trace

    @param trace: Data to detrend
    """
    ndat = len(trace)
    x1, x2 = trace[0], trace[-1]
    trace -= (x1 + N.arange(ndat) * (x2 - x1) / float(ndat - 1))


def cornFreq2Paz(fc, damp=0.707):
    """
    Convert corner frequency and damping to poles and zeros. 2 zeros at
    postion (0j, 0j) are given as output  (m/s).

    @param fc: Corner frequency
    @param damping: Corner frequency
    @return: Dictionary containing poles, zeros and gain
    """
    poles = [-(damp + M.sqrt(1 - damp ** 2) * 1j) * 2 * N.pi * fc]
    poles.append(-(damp - M.sqrt(1 - damp ** 2) * 1j) * 2 * N.pi * fc)
    return {'poles':poles, 'zeros':[0j, 0j], 'gain':1}


def pazToFreqResp(poles, zeros, scale_fac, t_samp, nfft, freq=False):
    """
    Convert Poles and Zeros (PAZ) to frequency response. Fast version which
    uses scipy. For understanding the source code, take a look at 
    pazToFreqResp3. The output contains the frequency zero which is the offset 
    of the trace.
    
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
    n = nfft / 2
    a, b = S.signal.ltisys.zpk2tf(zeros, poles, scale_fac)
    fy = 1 / (t_samp * 2.0)
    # start at zero to get zero for offset/ DC of fft
    f = N.arange(0, fy + fy / n, fy / n) #arange should includes fy/n
    _w, h = S.signal.freqs(a, b, f * 2 * N.pi)
    h = N.conj(h)
    h[-1] = h[-1].real + 0.0j
    if freq:
        return h, f
    return h


def specInv(spec, wlev, nfft):
    """
    Invert Spectrum and shrink values under WaterLevel. Fast version which
    uses array computation. For understanding the source code, take a look
    at specInv2.

    @note: In place opertions on spec
    @type: Complex numpy ndarray
    @param: spectrum to invert, in place opertaion 
    """

    max_spec_amp = N.abs(spec).max()

    swamp = max_spec_amp * 10.0 ** (-wlev / 20.0)
    swamp2 = swamp ** 2
    found = 0

    if N.abs(spec[0].real) < swamp:
        if N.abs(spec[0].real) > 0.0:
            real = 1. / swamp2;
        else:
            real = 0.0;
        found += 1
    else:
        real = 1. / spec[0].real ** 2
    spec[0] = complex(real, 0.0)

    if N.abs(spec[nfft / 2].real) < swamp:
        if N.abs(spec[nfft / 2].real) > 0.0:
            real = 1. / swamp2
        else:
            real = 0.0;
        found += 1
    else:
        real = 1. / spec[nfft / 2].real ** 2
    spec[nfft / 2] = complex(real, 0.0)

    spec0 = spec[0]
    specn = spec[nfft / 2]
    spec = spec[1:-1]
    sqr_len = abs(spec) ** 2
    idx = N.where(sqr_len < swamp2)
    spec[idx] *= N.sqrt(swamp2 / sqr_len[idx])
    sqr_len[idx] = abs(spec[idx]) ** 2
    found += len(idx[0])

    inn = N.where(sqr_len > 0.0)
    spec[inn] = 1 / spec[inn]
    spec[sqr_len <= 1.0e-10] = complex(0.0, 0.0) #where sqr_len == 0.0
    spec = N.concatenate(([spec0], spec, [specn]))

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
    numbers, gain must be of type float.
    @type water_level: Float
    @param water_level: Water_Level for spectrum to simulate
    @type inst_sim: Dictionary, None
    @param inst_sim: Dictionary containing keys 'poles', 'zeros',
        'gain'. Poles and zeros must be a list of complex floating point
        numbers, gain must be of type float. Or None for no simulation.
    
    Ready to go poles, zeros, gain dictionaries for instruments to simulate
    can be imported from obspy.signal.seismometer
    """
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
    nfft = int(M.pow(2, M.ceil(N.log2(2 * ndat))))
    # explicitly copy, else input data will be modified
    tr = data * cosTaper(ndat, 0.05)
    freq_response = pazToFreqResp(poles, zeros, gain, samp_int, nfft)
    found = specInv(freq_response, water_level, nfft)
    spec = N.fft.rfft(tr, n=nfft)
    spec *= N.conj(freq_response)
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
        spec *= N.conj(pazToFreqResp(poles, zeros, gain, samp_int, nfft))
    else:
        raise TypeError(error % 'inst_sim')
    tr2 = N.fft.irfft(spec)[0:ndat]
    # linear detrend, 
    x1 = tr2[0]
    x2 = tr2[-1]
    tr2 -= (x1 + N.arange(ndat) * (x2 - x1) / float(ndat - 1))
    #
    return tr2
