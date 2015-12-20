# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: radpattern.py
#  Purpose: Computes and plots radiation patterns
# ---------------------------------------------------------------------

"""
This script contains the following functions to compute and plot radiation
patterns:
    farfield_p
    farfield_s
    plot_3drpattern

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
from obspy.imaging.scripts.mopad import MomentTensor, BeachBall
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import PathPatch


def plot_3drpattern(mt, kind='both_sphere', vtk_fname='radpattern.vtk'):
    """
    Returns the P farfield radiation pattern on a unit sphere grid

    :param mt: Focal mechanism NM x 6 (M11, M22, M33, M12, M13, M23 - the
        six independent components of the moment tensor, where the coordinate
        system is 1,2,3 = Up,South,East which equals r,theta,phi -
        Harvard/Global CMT convention). The relation to Aki and Richards
        x,y,z equals North,East,Down convention is as follows: Mrr=Mzz,
        Mtt=Mxx, Mpp=Myy, Mrt=Mxz, Mrp=-Myz, Mtp=-Mxy.

    :param kind: can be 'p_quiver, s_quiver, both_quiver, p_sphere, s_sphere'

    :return: 3D grid point array with shape [3,npts] that contains
             the sperical grid points

             3D vector array with shape [3,npts] that contains the
             displacement vector for each grid point

    based on Aki & Richards Eq 4.29
    """

    vlength = 0.1  # length of vectors
    nlat = 30      # points for quiver sphere

    if kind == 'p_quiver':
        #precompute even spherical grid and directional cosine array
        points = spherical_grid(nlat=nlat)

        #get radiation pattern
        disp = farfield_p(mt, points)

        #plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(points[0], points[1], points[2],
                  disp[0], disp[1], disp[2], length=vlength)
        plt.show()

    elif kind == 'p_sphere':
        #generate spherical mesh that is aligned with the moment tensor null
        #axis
        mtensor = mopad_MomentTensor(mt, system='NED')
        null = np.ravel(mtensor._null_axis)

        #make rotation matrix (after numpy mailing list)
        #(bugs for zero rotation?)
        zaxis = np.array([0., 0., 1.])
        raxis = np.cross(null, zaxis)  # rotate z axis to null
        raxis /= np.linalg.norm(raxis)
        angle = np.arccos(np.dot(zaxis, null))  # angle between z and null

        eye = np.eye(3, dtype=np.float64)
        raxis2 = np.outer(raxis, raxis)
        skew = np.array([[0, raxis[2], -raxis[1]],
                         [-raxis[2], 0, raxis[0]],
                         [raxis[1], -raxis[0], 0]])

        rotmtx = raxis2 + np.cos(angle) * (eye - raxis2) + np.sin(angle) * skew

        #make uv sphere that is aligned with z-axis
        ntheta, nphi = 100, 100
        sshape = (ntheta, nphi)
        u = np.linspace(0, 2 * np.pi, nphi)
        v = np.linspace(0, np.pi, ntheta)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        #ravel point array and rotate them to the null axis
        points = np.vstack((x.flatten(), y.flatten(), z.flatten()))
        points = np.dot(rotmtx, points)

        #get radiation pattern
        disp = farfield_p(mt, points)
        magn = np.sum(disp * points, axis=0)
        magn /= np.max(np.abs(magn))

        #compute colours and displace points along normal
        norm = plt.Normalize(-1., 1.)
        cmap = plt.get_cmap('bwr')
        points *= (1. + np.abs(magn) / 2.)
        colors = np.array([cmap(norm(val)) for val in magn])
        colors = colors.reshape(ntheta, nphi, 4)

        x = points[0].reshape(sshape)
        y = points[1].reshape(sshape)
        z = points[2].reshape(sshape)

        #plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=colors)
        ax.plot([0, null[0]], [0, null[1]], [0, null[2]])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    elif kind == 's_quiver':
        #precompute even spherical grid and directional cosine array
        points = spherical_grid(nlat=nlat)

        #get radiation pattern
        disp = farfield_s(mt, points)

        #plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(points[0], points[1], points[2],
                  disp[0], disp[1], disp[2], length=vlength)
        plt.show()

    elif kind == 's_sphere':
        #lat/lon sphere
        u = np.linspace(0, 2 * np.pi, 200)
        v = np.linspace(0, np.pi, 200)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        points = np.vstack((x.flatten(), y.flatten(), z.flatten()))

        #get radiation pattern
        disp = farfield_s(mt, points)
        magn = np.sum(disp * disp, axis=0)
        magn /= np.max(np.abs(magn))

        #compute colours and displace points for normalized vectors
        norm = plt.Normalize(-1., 1.)
        cmap = plt.get_cmap('bwr')
        x *= (1. + np.abs(magn.reshape(x.shape)) / 2.)
        y *= (1. + np.abs(magn.reshape(x.shape)) / 2.)
        z *= (1. + np.abs(magn.reshape(x.shape)) / 2.)
        colors = np.array([cmap(norm(val)) for val in magn])
        colors = colors.reshape(x.shape[0], x.shape[1], 4)

        #plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=colors)
        plt.show()

    elif kind == 'both_quiver':
        #precompute even spherical grid and directional cosine array
        points = spherical_grid(nlat=nlat)

        #get radiation pattern
        dispp = farfield_p(mt, points)
        disps = farfield_s(mt, points)

        #plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        qp = ax.quiver(points[0], points[1], points[2],
                       dispp[0], dispp[1], dispp[2], length=vlength)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        qs = ax.quiver(points[0], points[1], points[2],
                       disps[0], disps[1], disps[2], length=vlength)
        qs.set_array(normp)
        plt.show()

    elif kind == 'mayavi':
        #use mayavi if possible. Otherwise fall back to output a vtkfile
        from mayavi import mlab
        mopad_mt = MomentTensor(mt, system='NED')
        bb = BeachBall(mopad_mt, npoints=200)
        bb._setup_BB(unit_circle=False)

        # extract the coordinates of the nodal lines
        neg_nodalline = bb._nodalline_negative
        pos_nodalline = bb._nodalline_positive

        #plot radiation pattern and nodal lines
        points = spherical_grid(nlat=nlat)
        dispp = farfield_p(mt, points)
        disps = farfield_s(mt, points)

        fig1 = mlab.figure(size=(800, 800), bgcolor=(0, 0, 0))
        pts1 = mlab.quiver3d(points[0], points[1], points[2],
                             dispp[0], dispp[1], dispp[2],
                             scalars=normp, vmin=-rangep, vmax=rangep)
        pts1.glyph.color_mode = 'color_by_scalar'
        mlab.plot3d(*neg_nodalline, color=(0,0.5,0), tube_radius=0.01)
        mlab.plot3d(*pos_nodalline, color=(0,0.5,0), tube_radius=0.01)
        plot_sphere(0.7)

        fig2 = mlab.figure(size=(800, 800), bgcolor=(0, 0, 0))
        mlab.quiver3d(points[0], points[1], points[2],
                      disps[0], disps[1], disps[2],
                      vmin=-ranges, vmax=ranges)
        mlab.plot3d(*neg_nodalline, color=(0, 0.5, 0), tube_radius=0.01)
        mlab.plot3d(*pos_nodalline, color=(0, 0.5, 0), tube_radius=0.01)
        plot_sphere(0.7)

        mlab.show()

    elif kind == 'vtk':
        fname_vtkrpattern = 'rpattern.vtk'
        fname_vtkbeachlines = 'beachlines.vtk'
        #output a vtkfile that can be read e.g. by paraview
        mopad_mt = MomentTensor(mt, system='NED')
        bb = BeachBall(mopad_mt, npoints=200)
        bb._setup_BB(unit_circle=False)

        # extract the coordinates of the nodal lines
        neg_nodalline = bb._nodalline_negative
        pos_nodalline = bb._nodalline_positive

        #plot radiation pattern and nodal lines
        points  = spherical_grid(nlat=nlat)
        ndim,npoints = points.shape
        dispp = farfield_p(mt, points)
        disps = farfield_s(mt, points)

        #output to file
        with open(fname_vtkrpattern,'w') as vtk_file:
            vtk_header = '# vtk DataFile Version 2.0\n'+\
                         'radiation pattern vector field\n'+\
                         'ASCII\n'+\
                         'DATASET UNSTRUCTURED_GRID\n'+\
                         'POINTS {:d} float\n'.format(npoints)

            vtk_file.write(vtk_header)
            #write point locations
            for x,y,z in np.transpose(points):
                vtk_file.write('{:.3e} {:.3e} {:.3e}\n'.format(x,y,z))
            #write vector field
            vtk_file.write('POINT_DATA {:d}\n'.format(npoints))
            vtk_file.write('VECTORS s_radiation float\n')
            for x,y,z in np.transpose(disps):
                vtk_file.write('{:.3e} {:.3e} {:.3e}\n'.format(x,y,z))
            vtk_file.write('VECTORS p_radiation float\n'.format(npoints))
            for x,y,z in np.transpose(dispp):
                vtk_file.write('{:.3e} {:.3e} {:.3e}\n'.format(x,y,z))

        with open(fname_vtkbeachlines,'w') as vtk_file:
            npoints_neg = neg_nodalline.shape[1]
            npoints_pos = neg_nodalline.shape[1]
            vtk_header = '# vtk DataFile Version 2.0\n'+\
                         'beachball nodal lines\n'+\
                         'ASCII\n'+\
                         'DATASET UNSTRUCTURED_GRID\n'+\
                         'POINTS {:d} float\n'.format(npoints_neg+npoints_pos)

            vtk_file.write(vtk_header)
            #write point locations
            for x,y,z in np.transpose(neg_nodalline):
                vtk_file.write('{:.3e} {:.3e} {:.3e}\n'.format(x,y,z))
            for x,y,z in np.transpose(pos_nodalline):
                vtk_file.write('{:.3e} {:.3e} {:.3e}\n'.format(x,y,z))

            #LINE CONNECTIVITY (SHOULD BECOME A TUBE LATER)

    else:
        raise NotImplementedError('{:s} not implemented yet'.format(kind))



def spherical_grid(nlat=30):
    """
    generates a simple equal area spherical grid
    """

    ndim = 3
    colats = np.linspace(0., np.pi, nlat)
    norms = np.sin(colats)
    nlons = (nlat * norms + 1).astype(int)  # scale number of point with lat

    #---- make colat/lon grid ----
    colatgrid, longrid = [], []
    for ilat in range(nlat):
        nlon = nlons[ilat]
        dlon = 2.*np.pi / nlon
        lons = np.arange(0., 2.*np.pi, dlon)
        for ilon in range(nlon):
            colatgrid.append(colats[ilat])
            longrid.append(lons[ilon])
    npoints = len(longrid)

    #---- get cartesian coordinates of spherical grid ----
    points = np.empty((ndim, npoints))
    points[0] = np.sin(colatgrid) * np.cos(longrid)
    points[1] = np.sin(colatgrid) * np.sin(longrid)
    points[2] = np.cos(colatgrid)

    return points


def farfield_p(mt, points):
    """
    Returns the P farfield radiation pattern

    :param mt: Focal mechanism NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - the
        six independent components of the moment tensor)

    :param points: 3D vector array with shape [3,npts] (x,y,z) or [2,npts]
                   (theta,phi) The normalized displacement of the moment
                   tensor source is computed at these points.

    :return: 3D vector array with shape [3,npts] that contains the
             displacement vector for each grid point

    based on Aki & Richards Eq 4.29
    """
    ndim, npoints = points.shape
    if ndim == 2:
        #points are given as theta,phi
        points = np.empty((3, npoints))
        points[0] = np.sin(points[0]) * np.cos(points[1])
        points[1] = np.sin(points[0]) * np.sin(points[1])
        points[2] = np.cos(points[0])
    elif ndim == 3:
        #points are given as x,y,z, (same system as the moment tensor)
        pass
    else:
        raise ValueError('points should have shape 2 x npoints or 3 x npoints')
    Mpq = fullmt(mt)

    #---- precompute directional cosine array ----
    dists = np.sqrt(points[0] * points[0] + points[1] * points[1] +
                    points[2] * points[2])
    gammas = points / dists

    #---- initialize displacement array ----
    disp = np.empty((ndim, npoints))

    #---- loop through points ----
    for ipoint in range(npoints):
        #loop through displacement component [n index]
        gamma = gammas[:, ipoint]
        gammapq = np.outer(gamma, gamma)
        gammatimesmt = gammapq * Mpq
        for n in range(ndim):
            disp[n, ipoint] = gamma[n] * np.sum(gammatimesmt.flatten())

    return disp


def farfield_s(mt, points):
    """
    Returns the S farfield radiation pattern

    :param mt: Focal mechanism NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - the
        six independent components of the moment tensor)

    :param points: 3D vector array with shape [3,npts] (x,y,z) or [2,npts]
                   (theta,phi) The normalized displacement of the moment
                   tensor source is computed at these points.

    :return: 3D vector array with shape [3,npts] that contains the
             displacement vector for each grid point

    based on Aki & Richards Eq 4.29
    """
    ndim, npoints = points.shape
    if ndim == 2:
        #points are given as theta,phi
        points = np.empty((3, npoints))
        points[0] = np.sin(points[0]) * np.cos(points[1])
        points[1] = np.sin(points[0]) * np.sin(points[1])
        points[2] = np.cos(points[0])
    elif ndim == 3:
        #points are given as x,y,z, (same system as the moment tensor)
        pass
    else:
        raise ValueError('points should have shape 2 x npoints or 3 x npoints')
    Mpq = fullmt(mt)

    #---- precompute directional cosine array ----
    dists = np.sqrt(points[0] * points[0] + points[1] * points[1] +
                    points[2] * points[2])
    gammas = points / dists

    #---- initialize displacement array ----
    disp = np.empty((ndim, npoints))

    #---- loop through points ----
    for ipoint in range(npoints):
        #---- loop through displacement component [n index] ----
        gamma = gammas[:, ipoint]
        Mp = np.dot(Mpq, gamma)
        for n in range(ndim):
            psum = 0.0
            for p in range(ndim):
                deltanp = int(n == p)
                psum += (gamma[n] * gamma[p] - deltanp) * Mp[p]
            disp[n, ipoint] = psum

    return disp


#---- get full moment tensor ----
def fullmt(mt):
    mt_full = np.array(([[mt[0], mt[3], mt[4]],
                         [mt[3], mt[1], mt[5]],
                         [mt[4], mt[5], mt[2]]]))
    return mt_full
