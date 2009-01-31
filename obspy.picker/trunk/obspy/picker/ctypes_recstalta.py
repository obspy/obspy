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
      
>>> random.seed(815)
>>> a = [random.gauss(0.,1.) for i in range(1000)]
>>> nsta=5
>>> nlta=10
>>> a1 = copy.deepcopy(a)
>>> c1 = recstalta(a,nsta,nlta)
>>> c2 = recstalta_py(a1,nsta,nlta)
>>> c1[99:103]
[1.0616963616768496, 1.3318426875849496, 1.248654066654898, 1.1828839719916469]
>>> c2[99:103]
[1.0616963616768493, 1.3318426875849498, 1.2486540666548982, 1.1828839719916469]
>>> err = sum([abs(i-j) for i,j in zip(c1,c2)])
>>> err <= 1e-10
True
"""

import os, random, copy
import ctypes as C

def recstalta(a,nsta,nlta):
    """Recursive STA/LTA (see Withers et al. 1998 p. 98)

    a    -- seismic trace
    nsta -- short time average window in samples
    nlta -- long time average window in samples
    This version directly uses a C version via CTypes"""

    ndat = len(a)
    lib = C.CDLL(os.path.join(os.path.dirname(__file__),'recstalta.so'))

    lib.recstalta.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int]
    lib.recstalta.restype=C.POINTER(C.c_double)

    # would be much easier using numpy, but we want was less dependencies as possible
    #import numpy; a = array(a)
    #charfct = C.pointer(lib.recstalta(a.ctypes.data_as(C.c_void_p),ndat,nsta,nlta))
    c_a = (C.c_double*ndat)()
    c_a[0:ndat] = a
    charfct = C.pointer(lib.recstalta(c_a,ndat,nsta,nlta))
    return charfct.contents[0:ndat]

def recstalta_py(a,nsta,nlta):
    """Recursive STA/LTA (see Withers et al. 1998 p. 98)
    
    NOTE: There exists a version of this trigger wrapped in C called
    recstalta in this module!"""
    
    #import numpy
    #assert type(a) == numpy.ndarray, "Error, data need to be a numpy array"

    ndat = len(a)
    charfunct = copy.deepcopy(a)
    #
    # compute the short time average (STA) and long time average (LTA)
    # given by Evans and Allen
    #Csta = 1-exp(-S/Nsta); Clta = 1-exp(-S/Nlta)
    csta = 1./nsta; clta = 1./nlta
    sta = 0.; lta = 0.; charfunct[0] = 0.
    for i in range(1,ndat):
        # THERE IS A SQUARED MISSING IN THE FORMULA, I ADDED IT
        sta=csta*a[i]**2 + (1-csta)*sta
        lta=clta*a[i]**2 + (1-clta)*lta
        charfunct[i] = sta/lta
        if i < nlta:
            charfunct[i] = 0.

    return charfunct

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
    #import pdb
    #pdb.set_trace()
