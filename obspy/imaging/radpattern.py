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

def farfield_p_unitsphere(mt,nlats=30):
    """
    Returns the P farfield radiation pattern on a unit sphere grid

    :param mt: Focal mechanism NM x 6 (M11, M22, M33, M12, M13, M23 - the
        six independent components of the moment tensor, where the coordinate
        system is 1,2,3 = Up,South,East which equals r,theta,phi -
        Harvard/Global CMT convention). The relation to Aki and Richards
        x,y,z equals North,East,Down convention is as follows: Mrr=Mzz,
        Mtt=Mxx, Mpp=Myy, Mrt=Mxz, Mrp=-Myz, Mtp=-Mxy.

    :return: 3D grid point array with shape [3,npts] that contains
             the sperical grid points

             3D vector array with shape [3,npts] that contains the
             displacement vector for each grid point

    based on Aki & Richards Eq 4.29
    """
    ndim = 3
    Mpq = fullmt(mt)

    #---- make spherical grid ----
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

def farfield_s_unitsphere(mt,nlats=30):
    """
    Returns the S farfield radiation pattern on a unit sphere grid

    :param mt: Focal mechanism NM x 6 (M11, M22, M33, M12, M13, M23 - the
        six independent components of the moment tensor, where the coordinate
        system is 1,2,3 = Up,South,East which equals r,theta,phi -
        Harvard/Global CMT convention). The relation to Aki and Richards
        x,y,z equals North,East,Down convention is as follows: Mrr=Mzz,
        Mtt=Mxx, Mpp=Myy, Mrt=Mxz, Mrp=-Myz, Mtp=-Mxy.

    :return: 3D grid point array with shape [3,npts] that contains
             the sperical grid points

             3D vector array with shape [3,npts] that contains the
             displacement vector for each grid point

    based on Aki & Richards Eq 4.29
    """
    ndim = 3
    Mpq = fullmt(mt)

    #---- make spherical grid ----
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
