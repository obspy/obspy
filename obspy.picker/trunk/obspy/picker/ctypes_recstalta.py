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


import os
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

if __name__ == '__main__':
    def test():
      """Test Ctypes version against an explicitly wrapped numpy C code"""
      from numpy import array,random
      from ext_recstalta import rec_stalta as recstalta_numpy
      
      a = random.random(1000)
      nsta=10
      nlta=100
      
      a1 = a.copy()
      
      c1 = array(recstalta(a,nsta,nlta))
      c2 = array(recstalta_numpy(a1,nsta,nlta))
      
      print "c1=...",c1[99:101],"...",c1[-3:-1]
      print "c2=...",c2[99:101],"...",c2[-3:-1]
      print "sum(abs(c1)-abs(c2)):",sum(abs(c1)-abs(c2))
    
    test()
    #import pdb
    #pdb.set_trace()
