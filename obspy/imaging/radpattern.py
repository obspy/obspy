# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: beachball.py
#  Purpose: Computes and plots radiation patterns
# ---------------------------------------------------------------------

"""
This script contains the following functions to compute and plot radiation patterns:
    farfield_p
    farfield_s

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport


import numpy as np
import matplotlib.pyplot as plt


D2R = np.pi / 180
R2D = 180 / np.pi
EPSILON = 0.00001

def farfieldP(mt):
    """
    This function is based on Aki & Richards Eq 4.29
    """
    ndim = 3
    Mpq = fullmt(mt)

    #---- make spherical grid ----
    nlats  = 30
    colats = np.linspace(0.,np.pi,nlats)
    norms = np.sin(colats)
    nlons = (nlats*norms+1).astype(int)
    colatgrid,longrid = [],[]
    for ilat in range(nlats):
        nlon = nlons[ilat]
        dlon = 2.*np.pi/nlon
        lons = np.arange(0.,2.*np.pi,dlon)
        for ilon in range(nlon):
            colatgrid.append(colats[ilat])
            longrid.append(lons[ilon])
    npoints = len(longrid)

    #---- get cartesian coordinates of spherical grid ----
    points = np.empty( (ndim,npoints) )
    points[0] = np.sin(colatgrid)*np.cos(longrid)
    points[1] = np.sin(colatgrid)*np.sin(longrid)
    points[2] = np.cos(colatgrid)

    #---- precompute directional cosine array ----
    dists = np.sqrt(points[0]*points[0]+points[1]*points[1]+points[2]*points[2])
    gammas = np.empty( (ndim,npoints) )
    gammas[0] = points[0]/dists
    gammas[1] = points[1]/dists
    gammas[2] = points[2]/dists

    #---- initialize displacement array ----
    disp   = np.empty( (ndim,npoints) )

    #---- loop through points ----
    for ipoint in range(npoints):
      #---- loop through displacement component [n index] ----
      gamma = gammas[:,ipoint]
      gammapq = np.outer(gamma,gamma)
      gammatimesmt = gammapq*Mpq
      for n in range(ndim):
          disp[n,ipoint] = gamma[n]*np.sum(gammatimesmt.flatten())

    return points,disp

def farfieldS(mt):
    """
    This function is based on Aki & Richards Eq 4.29
    """
    ndim = 3
    Mpq = fullmt(mt)

    #---- make spherical grid ----
    nlats  = 30
    colats = np.linspace(0.,np.pi,nlats)
    norms = np.sin(colats)
    nlons = (nlats*norms+1).astype(int)
    colatgrid,longrid = [],[]
    for ilat in range(nlats):
        nlon = nlons[ilat]
        dlon = 2.*np.pi/nlon
        lons = np.arange(0.,2.*np.pi,dlon)
        for ilon in range(nlon):
            colatgrid.append(colats[ilat])
            longrid.append(lons[ilon])
    npoints = len(longrid)

    #---- get cartesian coordinates of spherical grid ----
    points = np.empty( (ndim,npoints) )
    points[0] = np.sin(colatgrid)*np.cos(longrid)
    points[1] = np.sin(colatgrid)*np.sin(longrid)
    points[2] = np.cos(colatgrid)

    #---- precompute directional cosine array ----
    dists = np.sqrt(points[0]*points[0]+points[1]*points[1]+points[2]*points[2])
    gammas = np.empty( (ndim,npoints) )
    gammas[0] = points[0]/dists
    gammas[1] = points[1]/dists
    gammas[2] = points[2]/dists

    #---- initialize displacement array ----
    disp   = np.empty( (ndim,npoints) )

    #---- loop through points ----
    for ipoint in range(npoints):
      #---- loop through displacement component [n index] ----
      gamma = gammas[:,ipoint]
      Mp = np.dot(Mpq,gamma)
      for n in range(ndim):
          psum = 0.0
          for p in range(ndim):
              deltanp = int(n==p)
              psum += (gamma[n]*gamma[p] - deltanp)*Mp[p]
          disp[n,ipoint] = psum

    return points,disp

def fullmt(mt):
    mt_full = np.array( ([[mt[0],mt[3],mt[4]],
                          [mt[3],mt[1],mt[5]],
                          [mt[4],mt[5],mt[2]]]) )
    return mt_full

