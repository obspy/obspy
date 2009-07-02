#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: ctypes_recstalta.py
#  Purpose: stalta in extern C function, interfaced via ctypes
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2009 Moritz Beyreuther
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#---------------------------------------------------------------------

"""
Module implementing the Recursive STA/LTA (see Withers et al. 1998 p. 98)
Two versions, a fast ctypes one and a slow python one. In this doctest the
ctypes version is tested against the python implementation
      
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
    @type lsta: Int
    @param lsta: Length of long time average window in samples
    @rtype: Numpy ndarray
    @return: Charactristic function of STA/LTA
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
    @type lsta: Int
    @param lsta: Length of long time average window in samples
    @rtype: Numpy ndarray
    @return: Charactristic function of STA/LTA
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
    #import pdb
    #pdb.set_trace()
