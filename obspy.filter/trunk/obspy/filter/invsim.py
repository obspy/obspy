#!/usr/bin/python

import numpy as N
import math as M
PI = N.pi


def cosTaper(npts,p):
    """
    Cosinus Taper.

    >>> tap = cosTaper(100,1.0)
    >>> tap2 = 0.5*(1+N.cos(N.linspace(PI,2*PI,50)))
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
    frac = int(npts*p/2+.5)
    return N.concatenate((
        0.5*(1+N.cos(N.linspace(PI,2*PI,frac))),
        N.ones(npts-2*frac),
        0.5*(1+N.cos(N.linspace(0,PI,frac)))
        ),axis=0)


def pazToFreqResp2(poles,zeroes,scale_fac,t_samp,nfft):
    """
    Convert Poles and Zeros (PAZ) to frequency response
    
    @type poles: List of complex numbers
    @param poles: The poles of the transfer function
    @type zeros: List of complex numbers
    @param zeros: The zeros of the transfer function
    @type scale_fac: Float
    @param scale_fac: Gain factor
    @type t_samp: Float
    @param t_samp: Sampling interval in seconds
    @type nfft: Int
    @param nfft: Number of FFT points of signal which needs correction
    @rtype: numpy.ndarray complex128
    @return: Frequency response of PAZ of length nfft 
    """
    tr = N.zeros(nfft/2+1,dtype=N.complex128)
    npole = len(poles)
    nzero = len(zeroes)
    # io_agsec.c, line 455
    #/* calculate transfer function */
    delta_f = 1.0/(t_samp*nfft)
    for i in xrange(0,nfft/2+1):
        s = complex(0.0, i*2*PI*delta_f)
        num = 1.0 + 0.0j
        for ii in xrange(nzero):
            num *= (s - zeroes[ii])
    
        denom = 1.0 + 0.0j
        for ii in xrange(npole):
            denom *= (s - poles[ii])
    
        t_om = 1.0 + 0.0j
        if (denom.real != 0.0) or (denom.imag != 0.0):
            t_om = num/denom
    
        t_om *= scale_fac
    
        if (i < nfft/2) and (i > 0):
            # complex conjugate!!!
            tr[i] = complex(t_om.real,-t_om.imag)
        elif (i == 0):
            tr[i] = complex(t_om.real,0.0)
        elif (i == nfft/2):
            tr[i] = complex(t_om.real,0.0)
    return tr


def pazToFreqResp(poles,zeroes,scale_fac,t_samp,nfft):
    """
    Convert Poles and Zeros (PAZ) to frequency response. Fast version which
    uses array computation. For understanding the source code, take a look
    at pazToFreqResp2.
    
    @type poles: List of complex numbers
    @param poles: The poles of the transfer function
    @type zeros: List of complex numbers
    @param zeros: The zeros of the transfer function
    @type scale_fac: Float
    @param scale_fac: Gain factor
    @type t_samp: Float
    @param t_samp: Sampling interval in seconds
    @type nfft: Int
    @param nfft: Number of FFT points of signal which needs correction
    @rtype: numpy.ndarray complex128
    @return: Frequency response of PAZ of length nfft 
    """
    npole = len(poles)
    nzero = len(zeroes)
    delta_f = 1.0/(t_samp*nfft)
    s = N.arange(0,nfft/2+1,dtype=N.complex128)
    s.imag = s.real*delta_f*2*PI
    s.real = 0.0
    num = 1.0 + 0.0j
    for ii in xrange(nzero):
        num = (s - zeroes[ii]) * num
    denom = 1.0 + 0.0j
    for ii in xrange(npole):
        denom = (s - poles[ii]) * denom
    del s
    tr = num/denom
    tr[denom==0.0+0.0j] = 1.0 + 0.0j
    del num, denom
    tr *= scale_fac
    tr = N.conj(tr)
    #
    tr[0] = complex(tr[0].real,0.0)
    tr[nfft/2] = complex(tr[nfft/2].real,0.0)
    #
    return tr


def specInv2(spec,wlev,nfft):
    """
    Invert Spectrum and shrink values under WaterLevel. Fast version which
    uses array computation. For understanding the source code, take a look
    at specInv2.

    @note: In place opertions on spec
    @type spec: Complex numpy ndarray
    @param spec: spectrum to invert, in place opertaion 
    @type wlev: Float
    @param spec: WaterLevel in dB und which values are shrinked
    @rtype: Int
    @return: Number of values below water level
    """
    
    max_spec_amp=N.abs(spec).max()
    
    swamp = max_spec_amp*10.**(-wlev/20.0)
    found = 0
    
    if N.abs(spec[0].real) < swamp:
        if N.abs(spec[0].real) > 0.0:
            real = 1./(swamp*swamp);
        else:
            real = 0.0;
        found += 1
    else:
        real = 1./(spec[0].real*spec[0].real)
    spec[0] = complex(real,0.0)
    
    if N.abs(spec[nfft/2].real) < swamp:
        if N.abs(spec[nfft/2].real) > 0.0:
            real = 1./(swamp*swamp);
        else:
            real = 0.0;
        found += 1
    else:
        real = 1./ (spec[nfft/2].real*spec[nfft/2].real)
    spec[nfft/2] = complex(real,0.0)
    
    for i in xrange(1,nfft/2):
        real = spec[i].real
        imag = spec[i].imag
    
        sqr_len = float(real*real +imag*imag)
        # scale length to swamp but leave phase untouched
        if sqr_len < swamp*swamp: 
            real *= N.sqrt(swamp*swamp/sqr_len)
            imag *= N.sqrt(swamp*swamp/sqr_len)
            sqr_len = real*real +imag*imag
            found += 1
    
        #/* Division of a complex number 1/(a + i b) = */
        #/*  a/(a*a + b*b) - i b/(a*a + b*b) */
        if sqr_len > 0:
            spec[i] = complex(real/sqr_len,-imag/sqr_len)
        elif sqr_len == 0:
            spec[i] = complex(0.0,0.0)
    
    return found 


def specInv(spec,wlev,nfft):
    """
    @note: In place opertions on spec
    @type: Complex numpy ndarray
    @param: spectrum to invert, in place opertaion 
    """
    
    max_spec_amp=N.abs(spec).max()
    
    swamp = max_spec_amp*10.**(-wlev/20.0)
    swamp2 = swamp**2
    found = 0
    
    if N.abs(spec[0].real) < swamp:
        if N.abs(spec[0].real) > 0.0:
            real = 1./swamp2;
        else:
            real = 0.0;
        found += 1
    else:
        real = 1./spec[0].real**2
    spec[0] = complex(real,0.0)
    
    if N.abs(spec[nfft/2].real) < swamp:
        if N.abs(spec[nfft/2].real) > 0.0:
            real = 1./swamp2
        else:
            real = 0.0;
        found += 1
    else:
        real = 1./ spec[nfft/2].real**2
    spec[nfft/2] = complex(real,0.0)
    
    spec0 = spec[0]
    specn = spec[nfft/2]
    spec = spec[1:-1]
    sqr_len = abs(spec)**2
    idx = N.where(sqr_len<swamp2)
    spec[idx] *= N.sqrt(swamp2/sqr_len[idx])
    sqr_len[idx] = abs(spec[idx])**2
    found += len(idx[0])

    inn = N.where(sqr_len>0.0)
    spec[inn] = 1/spec[inn]
    spec[sqr_len==0.0] = complex(0.0,0.0)
    spec = N.concatenate(([spec0],spec,[specn]))

    return found 


def simFilt(tr2,tsa,fc0,fc1,fc2,h0,h1,h2,do_int):
    """Filter Simulation. Not used any more far."""
    # boolean galvo = true; /* galvanometer  */
    # double oms0,oms1,oms2;  /* filter coefficients */
    # double xk0,xk1,xk2;     /* k sub i dash */
    # /* integration */
    # float offset;
    # float buff;
    # float buff0;
    # int ndat;                   /* number of points in trace */
    # float tsa;                  /* sampling interval */
    ndat = len(tr2)
    galvo = True
    
    if fc2 <= 0:
            galvo = False
            fc2=1./(2.*tsa)
            h2=.69
    
    # /*     calculate filter coefficients  omega dash sub i */
    if fc0 <= 0.0:
        oms0 = 0.0;
    else:
        oms0 = M.tan(tsa*PI*fc0)
    
    oms1 = M.tan(tsa*PI*fc1)
    oms2 = M.tan(tsa*PI*fc2)
    
    # /*   k sub i dash */
    xk0 = 2.0*h0*oms0
    xk1 = 2.0*h1*oms1
    xk2 = 2.0*h2*oms2
    
    b0 = oms1*oms1 + xk1 +1.0
    a0 = (oms0*oms0 + xk0 +1.0)/b0
    a1 = 2.0*(oms0*oms0 -1.0)/b0
    a2 = (oms0*oms0 - xk0 +1.0)/b0
    b1 = 2.0*(oms1*oms1 -1.0)/b0
    b2 = (oms1*oms1 -xk1 +1.0)/b0
    d0 = oms2*oms2 + xk2 +1.0
    c0 = oms2*oms2/d0
    d1 = 2.0*(oms2*oms2 -1.0)/d0
    d2 = (oms2*oms2 - xk2 +1.0)/d0
    
    #/* filtering step 1  */
    u0 = 0.0
    u1 = 0.0
    u2 = 0.0
    v0 = 0.0
    v1 = 0.0
    v2 = 0.0
    
    for i in xrange(0,ndat):
        u2 = u1
        u1 = u0
        u0 = tr2[i]
        v2 = v1
        v1 = v0
        tr2[i] = a0*u0 + a1*u1 +a2*u2 -b1*v1 -b2*v2
        v0 =tr2[i]
    
    if galvo == True:
        #/*   filtering step 2   */
        v0 = 0.0
        v1 = 0.0
        v2 = 0.0
        w0 = 0.0
        w1 = 0.0
        w2 = 0.0
        for i in xrange(0,ndat):
            v2 = v1
            v1 = v0
            v0 = tr2[i]
            w2 = w1
            w1 = w0
            tr2[i] = c0*(v0 + 2.0*v1 +v2) -d1*w1 -d2*w2
            w0 =tr2[i]
    
    if do_int == 1:
        # /* remove offset */
        offset = 0.0
        for i in xrange(0,ndat):
            offset += tr2[i]
        offset /= ndat
        for i in xrange(0,ndat):
            tr2[i] = tr2[i] - offset;
        
        #/* integrate using trapezoidal rule */
        buff0 = (tsa * (tr2[0] + tr2[1])/2.0)
        
        tr2[0] = buff0
        buff = 0.0
        for i in xrange(2,ndat):
            buff = buff0 + (tsa *  ( tr2[i] + tr2[i-1])/2.0)
            tr2[i-1] = buff0
            buff0 = buff
        tr2[ndat-1] = tr2[ndat-2]
    #
    elif do_int == -1:
        # /* differentiate by first differences */
        buff0 = tr2[0]
        tr2[0] *= 2/float(tsa)
        for i in xrange(1,ndat):
            buff = tr2[i]
            tr2[i] = (tr2[i] - buff0)/float(tsa)
            buff0 = buff

