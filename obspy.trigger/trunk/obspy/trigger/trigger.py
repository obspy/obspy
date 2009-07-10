#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: trigger.py
#  Purpose: Python trigger routines for seismology.
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2010 Moritz Beyreuther
#-------------------------------------------------------------------
"""
Python trigger routines for seismology.

Module implementing the Recursive STA/LTA (see Withers et al. 1998 p. 98)
Two versions, a fast ctypes one and a slow python one. In this doctest the
ctypes version is tested against the python implementation. Further the
classic and delayed STA/LTA, the carlstatrig and the zdecect are
implemented. (see Withers et al. 1998 p. 98).
      

GNU General Public License (GPL)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301, USA.


>>> N.random.seed(815)
>>> a = N.random.randn(1000)
>>> nsta, nlta = 5, 10
>>> c1 = recStalta(a,nsta,nlta)
>>> c2 = recStaltaPy(a.copy(),nsta,nlta)
>>> c2[99:103]
array([ 0.80810165,  0.75939449,  0.91763978,  0.97465004])
>>> err = sum(abs(c1-c2))
>>> err <= 1e-10
True
>>> recStalta([1],nsta,nlta)
Traceback (most recent call last):
...
AssertionError: Error, data need to be a numpy ndarray
>>> recStalta(N.array([1],dtype='int32'),nsta,nlta)
Traceback (most recent call last):
...
AssertionError: Error, data need to be float64 numpy ndarray
"""

import os, random, copy, numpy as N
import ctypes as C

def recStalta(a,nsta,nlta):
    """Recursive STA/LTA (see Withers et al. 1998 p. 98)
    Fast version written in C.

    @note: This version directly uses a C version via CTypes
    @type a: Numpy ndarray
    @param a: Seismic Trace
    @type nsta: Int
    @param nsta: Length of short time average window in samples
    @type nlta: Int
    @param nlta: Length of long time average window in samples
    @rtype: Numpy ndarray
    @return: Charactristic function of recursive STA/LTA
    """

    assert type(a) == N.ndarray, "Error, data need to be a numpy ndarray"
    assert a.dtype == 'float64', "Error, data need to be float64 numpy ndarray"
    ndat = len(a)

    lib = C.CDLL(os.path.join(os.path.dirname(__file__),'recstalta.so'))

    lib.recstalta.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int]
    lib.recstalta.restype=C.POINTER(C.c_double)

    charfct = C.pointer(lib.recstalta(a.ctypes.data_as(C.c_void_p),ndat,nsta,nlta))
    # old method using interable a
    #c_a = (C.c_double*ndat)()
    #c_a[0:ndat] = a
    #charfct = C.pointer(lib.recstalta(c_a,ndat,nsta,nlta))
    return N.array(charfct.contents[0:ndat])

def recStaltaPy(a,nsta,nlta):
    """
    Recursive STA/LTA (see Withers et al. 1998 p. 98)
    Slow version written in Python.
    
    @note: There exists a faster version of this trigger wrapped in C
    called recstalta in this module!
    @type a: Numpy ndarray
    @param a: Seismic Trace
    @type nsta: Int
    @param nsta: Length of short time average window in samples
    @type nlta: Int
    @param nlta: Length of long time average window in samples
    @rtype: Numpy ndarray
    @return: Charactristic function of recursive STA/LTA
    """

    ndat = len(a)
    charfct = copy.deepcopy(a)
    #
    # compute the short time average (STA) and long time average (LTA)
    # given by Evans and Allen
    #Csta = 1-exp(-S/Nsta); Clta = 1-exp(-S/Nlta)
    csta = 1./nsta; clta = 1./nlta
    sta = 0.; lta = 0.; charfct[0] = 0.
    for i in range(1,ndat):
        # THERE IS A SQUARED MISSING IN THE FORMULA, I ADDED IT
        sta=csta*a[i]**2 + (1-csta)*sta
        lta=clta*a[i]**2 + (1-clta)*lta
        charfct[i] = sta/lta
        if i < nlta:
            charfct[i] = 0.
    return charfct

def carlStaTrig(a,Nsta,Nlta,ratio,quiet):
    """Computes the carlStaTrig characteristic function

    eta = star - (ratio * ltar) - abs(sta - lta) - quiet

    @type a: Numpy ndarray
    @param a: Seismic Trace
    @type Nsta: Int
    @param Nsta: Length of short time average window in samples
    @type Nlta: Int
    @param Nlta: Length of long time average window in samples
    @type ration: Float
    @param ratio: as ratio gets smaller, carlstatrig gets more sensitive
    @type quiet: Float
    @param quiet: as quiet gets smaller, carlstatrig gets more sensitive
    @rtype: Numpy ndarray
    @return: Charactristic function of CarlStaTrig
    """
    m=len(a)
    #
    sta=zeros(len(a),dtype=float)
    lta=zeros(len(a),dtype=float)
    star =zeros(len(a),dtype=float)
    ltar =zeros(len(a),dtype=float)
    pad_sta=zeros(Nsta)
    pad_lta=zeros(Nlta) # avoid for 0 division 0/1=0
    #
    # compute the short time average (STA)
    for i in xrange(Nsta): # window size to smooth over
        sta += concatenate((pad_sta,a[i:m-Nsta+i]))
    sta /= Nsta
    #
    # compute the long time average (LTA), 8 sec average over sta
    for i in xrange(Nlta): # window size to smooth over
        lta += concatenate((pad_lta,sta[i:m-Nlta+i]))
    lta /= Nlta
    lta = concatenate((zeros(1),lta))[:m] #XXX ???
    #
    # compute star, average of abs diff between trace and lta
    for i in xrange(Nsta): # window size to smooth over
        star += concatenate((pad_sta,abs(a[i:m-Nsta+i]-lta[i:m-Nsta+i])))
    star /= Nsta
    #
    # compute ltar, 8 sec average over star
    for i in xrange(Nlta): # window size to smooth over
        ltar += concatenate((pad_lta,star[i:m-Nlta+i]))
    ltar /= Nlta
    #
    eta =  star - (ratio * ltar) - abs(sta - lta) - quiet
    eta[:Nlta] = -1.0
    return eta

def classicStaLta(a,Nsta,Nlta):
    """Computes the standard STA/LTA from a given imput array a. The length of
    the STA is given by Nsta in samples, respectively is the length of the
    LTA given by Nlta in samples.

    @type a: Numpy ndarray
    @param a: Seismic Trace
    @type Nsta: Int
    @param Nsta: Length of short time average window in samples
    @type Nlta: Int
    @param Nlta: Length of long time average window in samples
    @rtype: Numpy ndarray
    @return: Charactristic function of classic STA/LTA
    """
    m=len(a)
    stalta=zeros(m,dtype=float)
    start = 0
    stop = 0
    #
    # compute the short time average (STA)
    sta=zeros(len(a),dtype=float)
    pad_sta=zeros(Nsta)
    for i in range(Nsta): # window size to smooth over
        sta=sta+concatenate((pad_sta,a[i:m-Nsta+i]**2))
    sta=sta/Nsta
    #
    # compute the long time average (LTA)
    lta=zeros(len(a),dtype=float)
    pad_lta=ones(Nlta) # avoid for 0 division 0/1=0
    for i in range(Nlta): # window size to smooth over
        lta=lta+concatenate((pad_lta,a[i:m-Nlta+i]**2))
    lta=lta/Nlta
    #
    # pad zeros of length Nlta to avoid overfit and
    # return STA/LTA ratio
    sta[0:Nlta]=0
    return sta/lta

def delayedStaLta(a,Nsta,Nlta):
    """Delayed STA/LTA, (see Withers et al. 1998 p. 97)

    @type a: Numpy ndarray
    @param a: Seismic Trace
    @type Nsta: Int
    @param Nsta: Length of short time average window in samples
    @type Nlta: Int
    @param Nlta: Length of long time average window in samples
    @rtype: Numpy ndarray
    @return: Charactristic function of delayed STA/LTA
    """
    m=len(a)
    stalta=zeros(m,dtype=float)
    on = 0;
    start = 0;
    stop = 0;
    #
    # compute the short time average (STA) and long time average (LTA)
    # don't start for STA at Nsta because it's muted later anyway
    sta=zeros(len(a),dtype=float)
    lta=zeros(len(a),dtype=float)
    for i in range(Nlta+Nsta+1,m):
        sta[i]=(a[i]**2 + a[i-Nsta]**2)/Nsta + sta[i-1]
        lta[i]=(a[i-Nsta-1]**2 + a[i-Nsta-Nlta-1]**2)/Nlta + lta[i-1]
        sta[0:Nlta+Nsta+50]=0
    return sta/lta

def zdetect(a,Nsta):
    """Z-detector, (see Withers et al. 1998 p. 99)"""
    m=len(a)
    #
    # Z-detector given by Swindell and Snell (1977)
    Z=zeros(len(a),dtype=float)
    sta=zeros(len(a),dtype=float)
    # Standard Sta
    pad_sta=zeros(Nsta)
    for i in range(Nsta): # window size to smooth over
        sta=sta+concatenate((pad_sta,a[i:m-Nsta+i]**2))
    a_mean=mean(sta)
    a_std=std(sta)
    Z=(sta-a_mean)/a_std
    return Z


def triggerOnset(charfct,thres1,thres2,samp_int=1):
    """
    Given thres1 and thres2 calculate trigger on and off times from
    characteristic function.

    @type charfct: Numpy ndarray
    @param charfct: Characteristic function of e.g. STA/LTA trigger
    @type thres1: Float
    @param thres1: Value above which trigger (of characteristic function)
        is activated
    @type thres2: Float
    @param thres2: Value below which trigger (of characteristic function)
        is deactivated
    @type samp_int: Float
    @param samp_int: Sample interval, needed as return times are in second
    @rtype: List
    @return: Nested List of trigger on and of times in samples
    """
    try: 
        on = where(charfct>thres1)[0]
        ind = on.min()
        of = ind + where(charfct[ind:]<thres2)[0]
    except ValueError:
        return True
    #
    pick = []
    start = stop = indstart = indstop = 0
    while True:
        try: 
            buf = start
            indstart = where(on[indstart:]>stop)[0].min() + indstart
            start = on[indstart]
            #
            indstop = where(of[indstop:]>start)[0].min() + indstop
            stop = of[indstop]
        except ValueError: # only activated if out of indices
            if not (start == buf) and not (start == 0):
                pick.add([start*samp_int,len(charfct)-1*samp_int])
            break
        #
        pick.add([start*samp_int,stop*samp_int])


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
