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
Two versions, a fast ctypes one and a bit slower python one. Further the
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
"""

import ctypes as C
import numpy as np
import os
import platform
import copy


if platform.system() == 'Windows':
    lib_name = 'signal.win32.dll'
elif platform.system() == 'Darwin':
    # 32 bit OSX, tested with 10.5.6
    lib_name = 'signal.dylib'
else:
    if platform.architecture()[0] == '64bit':
        lib_name = 'signal.lin64.so'
    else:
        lib_name = 'signal.so'

lib = C.CDLL(os.path.join(os.path.dirname(__file__), 'lib', lib_name))


def recStalta(a, nsta, nlta):
    """
    Recursive STA/LTA (see Withers et al. 1998 p. 98)
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
    lib.recstalta.argtypes = [np.ctypeslib.ndpointer(dtype='float64',
                                                    ndim=1,
                                                    flags='C_CONTIGUOUS'),
                              np.ctypeslib.ndpointer(dtype='float64',
                                                    ndim=1,
                                                    flags='C_CONTIGUOUS'),
                              C.c_int, C.c_int, C.c_int]
    lib.recstalta.restype = C.c_void_p
    ndat = len(a)
    charfct = np.ndarray(ndat, dtype='float64')
    lib.recstalta(a, charfct, ndat, nsta, nlta) # do not use pointer here
    return charfct

#
# Old but fancy version of recStalta, keep it for the moment
#lib.recstalta.restype = C.POINTER(C.c_double)
# reading C memory into buffer which can be converted to numpy array
#C.pythonapi.PyBuffer_FromMemory.argtypes = [C.c_void_p, C.c_int]
#C.pythonapi.PyBuffer_FromMemory.restype = C.py_object
#charfct = lib.recstalta(a, ndat, nsta, nlta # do not use pointer here
#size = struct.calcsize('d') # calculate size of float64
#return np.frombuffer(C.pythonapi.PyBuffer_FromMemory(charfct,ndat*size),
#                    dtype='float64',count=ndat)


def recStaltaPy(a, nsta, nlta):
    """
    Recursive STA/LTA (see Withers et al. 1998 p. 98)
    Bit slower version written in Python.
    
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
    try:
        a = a.tolist()
    except:
        pass
    ndat = len(a)
    # compute the short time average (STA) and long time average (LTA)
    # given by Evans and Allen
    csta = 1. / nsta
    clta = 1. / nlta
    sta = 0.
    lta = 0.
    charfct = [0.0]*len(a)
    icsta = 1 - csta
    iclta = 1 - clta
    for i in xrange(1, ndat):
        sq = a[i] ** 2
        sta = csta * sq + icsta * sta
        lta = clta * sq + iclta * lta
        charfct[i] = sta / lta
        if i < nlta:
            charfct[i] = 0.
    return np.array(charfct)


def carlStaTrig(a, Nsta, Nlta, ratio, quiet):
    """
    Computes the carlStaTrig characteristic function

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
    m = len(a)
    #
    sta = np.zeros(len(a), dtype='float64')
    lta = np.zeros(len(a), dtype='float64')
    star = np.zeros(len(a), dtype='float64')
    ltar = np.zeros(len(a), dtype='float64')
    pad_sta = np.zeros(Nsta)
    pad_lta = np.zeros(Nlta) # avoid for 0 division 0/1=0
    #
    # compute the short time average (STA)
    for i in xrange(Nsta): # window size to smooth over
        sta += np.concatenate((pad_sta, a[i:m - Nsta + i]))
    sta /= Nsta
    #
    # compute the long time average (LTA), 8 sec average over sta
    for i in xrange(Nlta): # window size to smooth over
        lta += np.concatenate((pad_lta, sta[i:m - Nlta + i]))
    lta /= Nlta
    lta = np.concatenate((np.zeros(1), lta))[:m] #XXX ???
    #
    # compute star, average of abs diff between trace and lta
    for i in xrange(Nsta): # window size to smooth over
        star += np.concatenate((pad_sta,
                               abs(a[i:m - Nsta + i] - lta[i:m - Nsta + i])))
    star /= Nsta
    #
    # compute ltar, 8 sec average over star
    for i in xrange(Nlta): # window size to smooth over
        ltar += np.concatenate((pad_lta, star[i:m - Nlta + i]))
    ltar /= Nlta
    #
    eta = star - (ratio * ltar) - abs(sta - lta) - quiet
    eta[:Nlta] = -1.0
    return eta

def classicStaLta(a, Nsta, Nlta):
    """
    Computes the standard STA/LTA from a given imput array a. The length of
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
    m = len(a)
    #
    # compute the short time average (STA)
    sta = np.zeros(len(a), dtype='float64')
    pad_sta = np.zeros(Nsta)
    for i in range(Nsta): # window size to smooth over
        sta = sta + np.concatenate((pad_sta, a[i:m - Nsta + i] ** 2))
    sta = sta / Nsta
    #
    # compute the long time average (LTA)
    lta = np.zeros(len(a), dtype='float64')
    pad_lta = np.ones(Nlta) # avoid for 0 division 0/1=0
    for i in range(Nlta): # window size to smooth over
        lta = lta + np.concatenate((pad_lta, a[i:m - Nlta + i] ** 2))
    lta = lta / Nlta
    #
    # pad zeros of length Nlta to avoid overfit and
    # return STA/LTA ratio
    sta[0:Nlta] = 0
    return sta / lta

def delayedStaLta(a, Nsta, Nlta):
    """
    Delayed STA/LTA, (see Withers et al. 1998 p. 97)

    @type a: Numpy ndarray
    @param a: Seismic Trace
    @type Nsta: Int
    @param Nsta: Length of short time average window in samples
    @type Nlta: Int
    @param Nlta: Length of long time average window in samples
    @rtype: Numpy ndarray
    @return: Charactristic function of delayed STA/LTA
    """
    m = len(a)
    #
    # compute the short time average (STA) and long time average (LTA)
    # don't start for STA at Nsta because it's muted later anyway
    sta = np.zeros(len(a), dtype='float64')
    lta = np.zeros(len(a), dtype='float64')
    for i in range(Nlta + Nsta + 1, m):
        sta[i] = (a[i] ** 2 + a[i - Nsta] ** 2) / Nsta + sta[i - 1]
        lta[i] = (a[i - Nsta - 1] ** 2 + a[i - Nsta - Nlta - 1] ** 2) / \
                 Nlta + lta[i - 1]
        sta[0:Nlta + Nsta + 50] = 0
    return sta / lta


def zdetect(a, Nsta):
    """
    Z-detector, (see Withers et al. 1998 p. 99)

    @param Nsta: Window length in Samples.
    """
    m = len(a)
    #
    # Z-detector given by Swindell and Snell (1977)
    sta = np.zeros(len(a), dtype='float64')
    # Standard Sta
    pad_sta = np.zeros(Nsta)
    for i in range(Nsta): # window size to smooth over
        sta = sta + np.concatenate((pad_sta, a[i:m - Nsta + i] ** 2))
    a_mean = np.mean(sta)
    a_std = np.std(sta)
    Z = (sta - a_mean) / a_std
    return Z


def triggerOnset(charfct, thres1, thres2, max_len=9e99):
    """
    Calculate trigger on and off times.

    Given thres1 and thres2 calculate trigger on and off times from
    characteristic function. 
    
    This method is written in pure Python and gets slow as soon as there
    are more then 1e6 triggerings ("on" AND "of") in charfct --- normally
    this does not happen.

    @type charfct: Numpy ndarray
    @param charfct: Characteristic function of e.g. STA/LTA trigger
    @type thres1: Float
    @param thres1: Value above which trigger (of characteristic function)
                   is activated (higher threshold)
    @type thres2: Float
    @param thres2: Value below which trigger (of characteristic function)
        is deactivated (lower threshold)
    @type max_len: Int
    @param max_len: Maximum length of triggered event in samples. A new
                    event will be triggered as soon as the signal reaches
                    again above thres1.
    @rtype: List
    @return: Nested List of trigger on and of times in samples
    """
    # 1) find indices of samples greater than threshold
    # 2) calculate trigger "of" times by the gap in trigger indices
    #    above the threshold i.e. the difference of two following indices
    #    in ind is greater than 1
    # 3) in principle the same as for "of" just add one to the index to get
    #    start times, this opperation is not supported on the compact
    #    syntax
    # 4) as long as there is a on time greater than the actual of time find
    #    trigger on states which are greater than last of state an the
    #    corresponding of state which is greater than current on state
    # 5) if the signal stays above thres2 longer than max_len an event
    #    is triggered and following a new event can be triggered as soon as
    #    the signal is above thres1
    ind1 = np.where(charfct > thres1)[0]
    if len(ind1) == 0:
        return []
    ind2 = np.where(charfct > thres2)[0]
    #
    of = [-1]
    of.extend(ind2[np.diff(ind2) > 1].tolist())
    of.extend([ind2[-1]])
    on = [ind1[0]]
    on.extend(ind1[np.where(np.diff(ind1) > 1)[0] + 1].tolist())
    #
    pick = []
    while on[-1] > of[0]:
        while on[0] <= of[0]:
            on.pop(0)
        while of[0] < on[0]:
            of.pop(0)
        if of[0] - on[0] > max_len:
            of.insert(0, on[0] + max_len)
        pick.append([on[0], of[0]])
    return np.array(pick)


def pkBaer(reltrc,samp_int,tdownmax,tupevent,thr1,thr2,preset_len,p_dur):
    """
    Wrapper for P-picker routine by m. baer, schweizer. erdbebendienst

    See paper by m. baer and u. kradolfer: an automatic phase picker for
    local and teleseismic events bssa vol. 77,4 pp1437-1445

    @param reltrc    : timeseries as floating data, possibly filtered
    @param samp_int  : number of samples per second
    @param tdownmax  : if dtime exceeds tdownmax, the trigger is examined
                       for validity
    @param tupevent  : min nr of samples for itrm to be accepted as a pick
    @param thr1      : threshold to trigger for pick (c.f. paper)
    @param thr2      : threshold for updating sigma  (c.f. paper)
    @param preset_len: no of points taken for the estimation of variance
                       of SF(t) on preset()
    @param p_dur     : p_dur defines the time interval for which the
                       maximum amplitude is evaluated Originally set to 6 secs
    @return          : (pptime, pfm) pptime sample number of parrival; pfm direction
                         of first motion (U or D)
    """
    pptime = C.c_int()
    # c_chcar_p strings are immutable, use string_buffer for pointers
    pfm = C.create_string_buffer("     ", 5)
    lib.ppick.argtypes = [np.ctypeslib.ndpointer(dtype='float32',
                                                 ndim=1,
                                                 flags='C_CONTIGUOUS'),
                          C.c_int, C.POINTER(C.c_int), C.c_char_p, C.c_float,
                          C.c_int, C.c_int, C.c_float, C.c_float, C.c_int,
                          C.c_int]
    lib.ppick.restype = C.c_void_p
    lib.ppick(reltrc, len(reltrc), C.byref(pptime), pfm, samp_int, 
              tdownmax, tupevent, thr1, thr2, preset_len,
              p_dur)
    return pptime.value, pfm.value


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
