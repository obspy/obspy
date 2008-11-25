#!/usr/bin/python

def recstalta(a,nsta,nlta):
  """Recursive STA/LTA (see Withers et al. 1998 p. 98)

  a    -- seismic trace
  nsta -- short time average window in samples
  nlta -- long time average window in samples
  This version directly uses a C version via CTypes"""

  ndat = len(a)
  import ctypes as C
  lib = C.CDLL('./recstalta.so')

  lib.recstalta.restype=C.POINTER(C.c_double)

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
