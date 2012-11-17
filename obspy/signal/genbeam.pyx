#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: genbeam.py
#  Purpose: more general beamforming
#   Author: Joachim Wassermann  
#    Email: j.wassermann@lmu.de
#
# Copyright (C) 2012 J. Wassermann
#---------------------------------------------------------------------
"""
More Generalized beamforming (currently fk and capon supported)

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import warnings
import ctypes as C
import numpy as np
cimport numpy as np
cimport cython
import pylab as pl
from time import time
from obspy.core import Stream

@cython.boundscheck(False)


def generalized_beamformer(np.ndarray[np.float64_t,ndim=2] trace, np.ndarray[np.complex128_t,ndim=4] steer, np.ndarray[np.complex128_t,ndim=4] nsteer, np.float flow, np.float fhigh,
         np.float digfreq, np.int nsamp, np.int nstat, np.int prewhiten, np.int grdpts_x, np.int grdpts_y, np.int nfft, np.int nf ,np.str method):
    """

    """
    # start the code -------------------------------------------------
    # This assumes that all stations and components have the same number of
    # time samples, nt

    cdef int nlow,x,y,i,j,n,ix,iy
    cdef float df
    cdef complex xxx
    cdef np.ndarray[np.complex128_t, ndim=1] xx = np.zeros((nfft),dtype=complex)
    cdef np.ndarray[np.complex128_t,ndim=3] R = np.zeros((nstat, nstat,nf),dtype=complex)
    cdef np.ndarray[np.complex128_t,ndim=3] R_inv = np.zeros((nstat, nstat,nf),dtype=complex)
    cdef double dpow
    cdef np.ndarray[np.float64_t,ndim=3] p = np.zeros((grdpts_x,grdpts_y,nf),dtype=float)
    cdef np.ndarray[np.float64_t,ndim=2] abspow = np.zeros((grdpts_x,grdpts_y),dtype=float)
    cdef np.ndarray[np.float64_t,ndim=2] relpow = np.zeros((grdpts_x,grdpts_y),dtype=float)
    cdef np.ndarray[np.float64_t,ndim=1] white = np.zeros((nf),dtype=float)
    cdef np.ndarray[np.float64_t,ndim=1] tap = np.zeros((nsamp),dtype=float)
    cdef np.ndarray[np.complex128_t,ndim=1] e = np.zeros((nstat),dtype=complex)
    cdef np.ndarray[np.complex128_t,ndim=1] eH = np.zeros((nstat),dtype=complex)
    cdef np.ndarray[np.complex128_t,ndim=1] Ce = np.zeros((nstat),dtype=complex)
    cdef np.ndarray[np.complex128_t,ndim=1] ICe = np.zeros((nstat),dtype=complex)
    cdef np.ndarray[np.complex128_t,ndim=2] C = np.zeros((nstat,nstat),dtype=complex)
    cdef np.ndarray[np.complex128_t,ndim=2] IC = np.zeros((nstat,nstat),dtype=complex)
    cdef extern from "math.h":
        float sqrt "sqrtf" (float dummy)

    df = digfreq/float(nfft)
    nlow = int(flow/df)

    # in general, beamforming is done by simply computing the co-variances
    # of the signal at different receivers and than stear the matrix R with
    # "weights" which are the trial-DOAs e.g., Kirlin & Done, 1999
    dpow = 0.

    # fill up R
    for i in xrange(nstat):
       for j in xrange(i,nstat):
            xx = np.fft.rfft(trace[i],nfft) * np.fft.rfft(trace[j],nfft).conjugate()
            if method == 'capon':
                 R[i,j,0:nf] = xx[nlow:nlow+nf]/np.abs(xx[nlow:nlow+nf].sum())
                 if i != j:
                     R[j,i,0:nf] = xx[nlow:nlow+nf].conjugate()/np.abs(xx[nlow:nlow+nf].sum())
            else :
                 R[i,j,0:nf] = xx[nlow:nlow+nf]
                 if i != j:
                     R[j,i,0:nf] = xx[nlow:nlow+nf].conjugate()
                 else:
                     dpow += np.abs(R[i,j,:].sum())

    dpow *= nstat

    if method == "bf":
    # P(f) = e.H R(f) e
        for x from 0 <= x < grdpts_x:
            for y from 0 <= y < grdpts_y:
              for n from 0 <= n < nf:
                 xxx = np.inner(nsteer[0:nstat,x,y,n],np.dot(R[0:nstat,0:nstat,n],steer[0:nstat,x,y,n]))
                 if prewhiten == 0:
                    abspow[x,y] += sqrt(xxx.real * xxx.real + xxx.imag*xxx.imag)
                 if prewhiten == 1:
                    p[x,y,n] = sqrt(xxx.real * xxx.real + xxx.imag*xxx.imag)
              if prewhiten == 0:
                 relpow[x,y] = abspow[x,y]/dpow
              if prewhiten == 1:
                 for n from 0 <= n < nf: 
                   if p[x][y][n] > white[n]:
                      white[n] = p[x,y,n]
              
        if prewhiten == 1:
            for x from 0 <= x < grdpts_x:
               for y from 0 <= y < grdpts_y:
                   abspow[x,y] = p[x,y,0:nf].sum()
                   relpow[x,y] = (p[x,y,0:nf]/(white[0:nf]*nf*nstat)).sum()

    elif method == "capon":
    # P(f) = 1/(e.H R(f)^-1 e)
        for n from 0 <= n < nf:
            R_inv[:, :, n] = np.linalg.pinv(R[0:nstat,0:nstat,n])
        for x from 0 <= x < grdpts_x:
            for y from 0 <= y < grdpts_y:
              for n from 0 <= n < nf:
                  #C = R[0:nstat,0:nstat,n]
                  #IC = np.linalg.pinv(C)
                  #e = steer[0:nstat,x,y,n]
                  #eH = nsteer[0:nstat,x,y,n]
                  #ICe = np.dot(IC,e)
                  xxx = (1./np.inner(nsteer[0:nstat,x,y,n],np.dot(R_inv[:, :, n],steer[0:nstat,x,y,n])))
                  if prewhiten == 0:
                      abspow[x,y] += sqrt(xxx.real * xxx.real + xxx.imag*xxx.imag)
                  if prewhiten == 1:
                      p[x,y,n] = sqrt(xxx.real * xxx.real + xxx.imag*xxx.imag)
              if prewhiten == 0:
                 relpow[x,y] = abspow[x,y]
              if prewhiten == 1:
                 for n from 0 <= n < nf:
                   if p[x][y][n] > white[n]:
                      white[n] = p[x,y,n]
        if prewhiten == 1:
            for x from 0 <= x < grdpts_x:
               for y from 0 <= y < grdpts_y:
                  relpow[x,y] = np.sum(p[x,y,0:nf]/(white[0:nf]*nf*nstat))

    # find the maximum in the map and return its value and the indices
    ix,iy = pl.unravel_index(relpow.argmax(), np.shape(relpow))

    return abspow.max(), relpow.max(), ix, iy

