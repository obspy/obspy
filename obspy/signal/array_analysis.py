#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: array.py
#  Purpose: Functions for Array Analysis
#   Author: Martin van Driel, Moritz Beyreuther
#    Email: driel@geophysik.uni-muenchen.de
#
# Copyright (C) 2010 Martin van Driel, Moritz Beyreuther
#---------------------------------------------------------------------
"""
Functions for Array Analysis

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import math
import warnings
import ctypes as C
import numpy as np
import pylab as pl
from obspy.signal.util import utlGeoKm, nextpow2
from obspy.signal.headers import clibsignal
from obspy.core import Stream
from scipy.integrate import cumtrapz
from obspy.signal.invsim import cosTaper
from scipy.signal import detrend
import cProfile
import genbeam


def array_rotation_strain(subarray, ts1, ts2, ts3, vp, vs, array_coords,
                          sigmau):
    """
    This routine calculates the best-fitting rigid body rotation and
    uniform strain as functions of time, and their formal errors, given
    three-component ground motion time series recorded on a seismic array.
    The theory implemented herein is presented in the papers [Spudich1995]_,
    (abbreviated S95 herein) [Spudich2008]_ (SF08) and [Spudich2009]_ (SF09).

    This is a translation of the Matlab Code presented in (SF09) with
    small changes in details only. Output has been checked to be the same
    as the original Matlab Code.

    .. note::
        ts\_ below means "time series"

    :type vp: Float
    :param vp: P wave speed in the soil under the array (km/s)
    :type vs: Float
    :param vs: S wave speed in the soil under the array Note - vp and vs may be
        any unit (e.g. miles/week), and this unit need not be related to the
        units of the station coordinates or ground motions, but the units of vp
        and vs must be the SAME because only their ratio is used.
    :type array_coords: numpy.ndarray
    :param array_coords: array of dimension Na x 3, where Na is the number of
        stations in the array.  array_coords[i,j], i in arange(Na), j in
        arange(3) is j coordinate of station i.  units of array_coords may be
        anything, but see the "Discussion of input and output units" above.
        The origin of coordinates is arbitrary and does not affect the
        calculated strains and rotations.  Stations may be entered in any
        order.
    :type ts1: numpy.ndarray
    :param ts1: array of x1-component seismograms, dimension nt x Na.
        ts1[j,k], j in arange(nt), k in arange(Na) contains the kth time sample
        of the x1 component ground motion at station k. NOTE that the
        seismogram in column k must correspond to the station whos coordinates
        are in row k of in.array_coords. nt is the number of time samples in
        the seismograms.  Seismograms may be displacement, velocity,
        acceleration, jerk, etc.  See the "Discussion of input and output
        units" below.
    :type ts2: numpy.ndarray
    :param ts2: same as ts1, but for the x2 component of motion.
    :type ts3: numpy.ndarray
    :param ts3: same as ts1, but for the x3 (UP or DOWN) component of motion.
    :type sigmau: Float or numpy.ndarray
    :param sigmau: standard deviation (NOT VARIANCE) of ground noise,
        corresponds to sigma-sub-u in S95 lines above eqn (A5).
        NOTE: This may be entered as a scalar, vector, or matrix!

        * If sigmau is a scalar, it will be used for all components of all
          stations.
        * If sigmau is a 1D array of length Na, sigmau[i] will be the noise
          assigned to all components of the station corresponding to
          array_coords[i,:]
        * If sigmau is a 2D array of dimension  Na x 3, then sigmau[i,j] is
          used as the noise of station i, component j.

        In all cases, this routine assumes that the noise covariance between
        different stations and/or components is zero.
    :type subarray: numpy.ndarray
    :param subarray: NumPy array of subarray stations to use. I.e. if subarray
        = array([1, 4, 10]), then only rows 1, 4, and 10 of array_coords will
        be used, and only ground motion time series in the first, fourth, and
        tenth columns of ts1 will be used. Nplus1 is the number of elements in
        the subarray vector, and N is set to Nplus1 - 1. To use all stations in
        the array, set in.subarray = arange(Na), where Na is the total number
        of stations in the array (equal to the number of rows of
        in.array_coords. Sequence of stations in the subarray vector is
        unimportant; i.e.  subarray = array([1, 4, 10]) will yield essentially
        the same rotations and strains as subarray = array([10, 4, 1]).
        "Essentially" because permuting subarray sequence changes the d vector,
        yielding a slightly different numerical result.
    :return: Dictionary with fields:
        | **A:** (array, dimension 3N x 6) - data mapping matrix 'A' of
        |     S95(A4)
        | **g:** (array, dimension 6 x 3N) - generalized inverse matrix
        |     relating ptilde and data vector, in S95(A5)
        | **Ce:** (4 x 4) covariance matrix of the 4 independent strain
        |     tensor elements e11, e21, e22, e33
        | **ts_d:** (array, length nt) - dilatation
        |     (trace of the 3x3 strain tensor) as a function of time
        | **sigmad:** scalar, standard deviation of dilatation
        | **ts_dh:** (array, length nt) - horizontal dilatation (also
        |     known as areal strain) (eEE+eNN) as a function of time
        | **sigmadh:** scalar, standard deviation of horizontal dilatation
        |     (areal strain)
        | **ts_e:** (array, dimension nt x 3 x 3) - strain tensor
        | **ts_s:** (array, length nt) -  maximum strain
        |     ( .5*(max eigval of e - min eigval of e) as a
        |     function of time, where e is the 3x3 strain tensor
        | **Cgamma:** (4 x 4) covariance matrix of the 4 independent shear
        |     strain tensor elements g11, g12, g22, g33 (includes full
        |     covariance effects). gamma is traceless part of e.
        | **ts_sh:** (array, length nt) - maximum horizontal strain
        |     ( .5*(max eigval of eh - min eigval of eh)
        |     as a function of time, where eh is e(1:2,1:2)
        | **Cgammah:** (3 x 3) covariance matrix of the 3 independent
        |     horizontal shear strain tensor elements gamma11, gamma12,
        |     gamma22 gamma is traceless part of e.
        | **ts_wmag:** (array, length nt) -  total rotation
        |     angle (radians) as a function of time.  I.e. if the
        |     rotation vector at the j'th time step is
        |     w = array([w1, w2, w3]), then ts_wmag[j] = sqrt(sum(w**2))
        |     positive for right-handed rotation
        | **Cw:** (3 x 3) covariance matrix of the 3 independent
        |     rotation tensor elements w21, w31, w32
        | **ts_w1:** (array, length nt) - rotation
        |     (rad) about the x1 axis, positive for right-handed rotation
        | **sigmaw1:** scalar, standard deviation of the ts_w1
        |     (sigma-omega-1 in SF08)
        | **ts_w2:** (array, length nt) - rotation
        |     (rad) about the x2 axis, positive for right-handed rotation
        | **sigmaw2:** scalar, standard deviation of ts_w2
        |     (sigma-omega-2 in SF08)
        | **ts_w3:** (array, length nt) - "torsion", rotation
        |     (rad) about a vertical up or down axis, i.e. x3, positive
        |     for right-handed rotation
        | **sigmaw3:** scalar, standard deviation of the torsion
        |     (sigma-omega-3 in SF08)
        | **ts_tilt:** (array, length nt) - tilt (rad)
        |     (rotation about a horizontal axis, positive for right
        |     handed rotation)
        |     as a function of time.  tilt = sqrt( w1^2 + w2^2)
        | **sigmat:** scalar, standard deviation of the tilt
        |     (not defined in SF08, From Papoulis (1965, p. 195,
        |     example 7.8))
        | **ts_data:** (array, shape (nt x 3N)). time series of
        |     the observed displacement
        |     differences, which are the di in S95 eqn A1.
        | **ts_pred:** (array, shape (nt x 3N)) time series of
        |     the fitted model's predicted displacement difference
        |     Note that the fitted model displacement
        |     differences correspond to linalg.dot(A, ptilde), where A
        |     is the big matrix in S95 eqn A4 and ptilde is S95 eqn A5.
        | **ts_misfit:** (array, shape (nt x 3N)) time series of the
        |     residuals (fitted model displacement differences minus
        |     observed displacement differences). Note that the fitted
        |     model displacement differences correspond to
        |     linalg.dot(A, ptilde), where A is the big
        |     matrix in S95 eqn A4 and ptilde is S95 eqn A5.
        | **ts_M:** (array, length nt) Time series of M, misfit
        |     ratio of S95, p. 688.
        | **ts_ptilde:** (array, shape (nt x 6)) - solution
        |     vector p-tilde (from S95 eqn A5) as a function of time
        | **Cp:** 6x6 solution covariance matrix defined in SF08.

    .. rubric:: Warnings

    This routine does not check to verify that your array is small
    enough to conform to the assumption that the array aperture is less
    than 1/4 of the shortest seismic wavelength in the data. See SF08
    for a discussion of this assumption.

    This code assumes that ts1[j,:], ts2[j,:], and ts3[j,:] are all sampled
    SIMULTANEOUSLY.

    .. rubric:: Notes

    (1) Note On Specifying Input Array And Selecting Subarrays

        This routine allows the user to input the coordinates and ground
        motion time series of all stations in a seismic array having Na
        stations and the user may select for analysis a subarray of Nplus1
        <= Na stations.

    (2) Discussion Of Physical Units Of Input And Output

        If the input seismograms are in units of displacement, the output
        strains and rotations will be in units of strain (unitless) and
        angle (radians).  If the input seismograms are in units of
        velocity, the output will be strain rate (units = 1/s) and rotation
        rate (rad/s).  Higher temporal derivative inputs yield higher
        temporal derivative outputs.

        Input units of the array station coordinates must match the spatial
        units of the seismograms.  For example, if the input seismograms
        are in units of m/s^2, array coordinates must be entered in m.

    (3) Note On Coordinate System

        This routine assumes x1-x2-x3 is a RIGHT handed orthogonal
        coordinate system. x3 must point either UP or DOWN.
    """
    # start the code -------------------------------------------------
    # This assumes that all stations and components have the same number of
    # time samples, nt
    [nt, Na] = np.shape(ts1)

    # check to ensure all components have same duration
    if ts1.shape != ts2.shape:
        raise ValueError('ts1 and ts2 have different sizes')
    if ts1.shape != ts3.shape:
        raise ValueError('ts1 and ts3 have different sizes')

    # check to verify that the number of stations in ts1 agrees with the number
    # of stations in array_coords
    [nrac, _ncac] = array_coords.shape
    if nrac != Na:
        msg = 'ts1 has %s columns(stations) but array_coords has ' % Na + \
              '%s rows(stations)' % nrac
        raise ValueError(msg)

    # check stations in subarray exist
    if min(subarray) < 0:
        raise ValueError('Station number < 0 in subarray')
    if max(subarray) > Na:
        raise ValueError('Station number > Na in subarray')

    # extract the stations of the subarray to be used
    subarraycoords = array_coords[subarray, :]

    # count number of subarray stations: Nplus1 and number of station
    # offsets: N
    Nplus1 = subarray.size
    N = Nplus1 - 1

    if Nplus1 < 3:
        msg = 'The problem is underdetermined for fewer than 3 stations'
        raise ValueError(msg)
    elif Nplus1 == 3:
        msg = 'For a 3-station array the problem is even-determined'
        warnings.warn(msg)

    # ------------------- NOW SOME SEISMOLOGY!! --------------------------
    # constants
    eta = 1 - 2 * vs ** 2 / vp ** 2

    # form A matrix, which relates model vector of 6 displacement derivatives
    # to vector of observed displacement differences. S95(A3)
    # dim(A) = (3*N) * 6
    # model vector is [ u1,1 u1,2 u1,3 u2,1 u2,2 u2,3 ] (free surface boundary
    # conditions applied, S95(A2))
    # first initialize A to the null matrix
    A = np.zeros((N * 3, 6))
    z3t = np.zeros(3)
    # fill up A
    for i in xrange(N):
        ss = subarraycoords[(i + 1), :] - subarraycoords[0, :]
        A[(3 * i):(3 * i + 3), :] = np.c_[np.r_[ss, z3t], np.r_[z3t, ss], \
            np.array([-eta * ss[2], \
            0., -ss[0], 0., -eta * ss[2], -ss[1]])].transpose()

    #------------------------------------------------------
    # define data covariance matrix Cd.
    # step 1 - define data differencing matrix D
    # dimension of D is (3*N) * (3*Nplus1)
    I3 = np.eye(3)
    II = np.eye(3 * N)
    D = -I3

    for i in xrange(N - 1):
        D = np.c_[D, -I3]
    D = np.r_[D, II].T

    # step 2 - define displacement u covariance matrix Cu
    # This assembles a covariance matrix Cu that reflects actual data errors.
    # populate Cu depending on the size of sigmau
    if np.size(sigmau) == 1:
        # sigmau is a scalar.  Make all diag elements of Cu the same
        Cu = sigmau ** 2 * np.eye(3 * Nplus1)
    elif np.shape(sigmau) == (np.size(sigmau),):
        # sigmau is a row or column vector
        # check dimension is okay
        if np.size(sigmau) != Na:
            raise ValueError('sigmau must have %s elements' % Na)
        junk = (np.c_[sigmau, sigmau, sigmau]) ** 2  # matrix of variances
        Cu = np.diag(np.reshape(junk[subarray, :], (3 * Nplus1)))
    elif sigmau.shape == (Na, 3):
        Cu = np.diag(np.reshape(((sigmau[subarray, :]) ** 2).transpose(), \
                (3 * Nplus1)))
    else:
        raise ValueError('sigmau has the wrong dimensions')

    # Cd is the covariance matrix of the displ differences
    # dim(Cd) is (3*N) * (3*N)
    Cd = np.dot(np.dot(D, Cu), D.T)

    #---------------------------------------------------------
    # form generalized inverse matrix g.  dim(g) is 6 x (3*N)
    Cdi = np.linalg.inv(Cd)
    AtCdiA = np.dot(np.dot(A.T, Cdi), A)
    g = np.dot(np.dot(np.linalg.inv(AtCdiA), A.T), Cdi)

    condition_number = np.linalg.cond(AtCdiA)

    if condition_number > 100:
        msg = 'Condition number is %s' % condition_number
        warnings.warn(msg)

    # set up storage for vectors that will contain time series
    ts_wmag = np.empty(nt)
    ts_w1 = np.empty(nt)
    ts_w2 = np.empty(nt)
    ts_w3 = np.empty(nt)
    ts_tilt = np.empty(nt)
    ts_dh = np.empty(nt)
    ts_sh = np.empty(nt)
    ts_s = np.empty(nt)
    ts_pred = np.empty((nt, 3 * N))
    ts_misfit = np.empty((nt, 3 * N))
    ts_M = np.empty(nt)
    ts_data = np.empty((nt, 3 * N))
    ts_ptilde = np.empty((nt, 6))
    for array in (ts_wmag, ts_w1, ts_w2, ts_w3, ts_tilt, ts_dh, ts_sh, ts_s,
                  ts_pred, ts_misfit, ts_M, ts_data, ts_ptilde):
        array.fill(np.NaN)
    ts_e = np.NaN * np.empty((nt, 3, 3))

    # other matrices
    udif = np.empty((3, N))
    udif.fill(np.NaN)

    #---------------------------------------------------------------
    # here we define 4x6 Be and 3x6 Bw matrices.  these map the solution
    # ptilde to strain or to rotation.  These matrices will be used
    # in the calculation of the covariances of strain and rotation.
    # Columns of both matrices correspond to the model solution vector
    # containing elements [u1,1 u1,2 u1,3 u2,1 u2,2 u2,3 ]'
    #
    # the rows of Be correspond to e11 e21 e22 and e33
    Be = np.zeros((4, 6))
    Be[0, 0] = 2.
    Be[1, 1] = 1.
    Be[1, 3] = 1.
    Be[2, 4] = 2.
    Be[3, 0] = -2 * eta
    Be[3, 4] = -2 * eta
    Be = Be * .5
    #
    # the rows of Bw correspond to w21 w31 and w32
    Bw = np.zeros((3, 6))
    Bw[0, 1] = 1.
    Bw[0, 3] = -1.
    Bw[1, 2] = 2.
    Bw[2, 5] = 2.
    Bw = Bw * .5
    #
    # this is the 4x6 matrix mapping solution to total shear strain gamma
    # where gamma = strain - tr(strain)/3 * eye(3)
    # the four elements of shear are 11, 12, 22, and 33.  It is symmetric.
    aa = (2 + eta) / 3
    b = (1 - eta) / 3
    c = (1 + 2 * eta) / 3
    Bgamma = np.zeros((4, 6))
    Bgamma[0, 0] = aa
    Bgamma[0, 4] = -b
    Bgamma[2, 2] = .5
    Bgamma[1, 3] = .5
    Bgamma[2, 0] = -b
    Bgamma[2, 4] = aa
    Bgamma[3, 0] = -c
    Bgamma[3, 4] = -c
    #
    # this is the 3x6 matrix mapping solution to horizontal shear strain
    # gamma
    # the four elements of horiz shear are 11, 12, and 22.  It is symmetric.
    Bgammah = np.zeros((3, 6))
    Bgammah[0, 0] = .5
    Bgammah[0, 4] = -.5
    Bgammah[1, 1] = .5
    Bgammah[1, 3] = .5
    Bgammah[2, 0] = -.5
    Bgammah[2, 4] = .5

    # solution covariance matrix.  dim(Cp) = 6 * 6
    # corresponding to solution elements [u1,1 u1,2 u1,3 u2,1 u2,2 u2,3 ]
    Cp = np.dot(np.dot(g, Cd), g.T)

    # Covariance of strain tensor elements
    # Ce should be 4x4, correspond to e11, e21, e22, e33
    Ce = np.dot(np.dot(Be, Cp), Be.T)
    # Cw should be 3x3 correspond to w21, w31, w32
    Cw = np.dot(np.dot(Bw, Cp), Bw.T)

    # Cgamma is 4x4 correspond to 11, 12, 22, and 33.
    Cgamma = np.dot(np.dot(Bgamma, Cp), Bgamma.T)
    #
    #  Cgammah is 3x3 correspond to 11, 12, and 22
    Cgammah = np.dot(np.dot(Bgammah, Cp), Bgammah.T)
    #
    #
    # covariance of the horizontal dilatation and the total dilatation
    # both are 1x1, i.e. scalars
    Cdh = Cp[0, 0] + 2 * Cp[0, 4] + Cp[4, 4]
    sigmadh = np.sqrt(Cdh)

    # covariance of the (total) dilatation, ts_dd
    sigmadsq = (1 - eta) ** 2 * Cdh
    sigmad = np.sqrt(sigmadsq)
    #
    # Cw3, covariance of w3 rotation, i.e. torsion, is 1x1, i.e. scalar
    Cw3 = (Cp[1, 1] - 2 * Cp[1, 3] + Cp[3, 3]) / 4
    sigmaw3 = np.sqrt(Cw3)

    # For tilt cannot use same approach because tilt is not a linear function
    # of the solution.  Here is an approximation :
    # For tilt use conservative estimate from
    # Papoulis (1965, p. 195, example 7.8)
    sigmaw1 = np.sqrt(Cp[5, 5])
    sigmaw2 = np.sqrt(Cp[2, 2])
    sigmat = max(sigmaw1, sigmaw2) * np.sqrt(2 - np.pi / 2)

    #
    # BEGIN LOOP OVER DATA POINTS IN TIME SERIES==============================
    #
    for itime in xrange(nt):
        #
        # data vector is differences of stn i displ from stn 1 displ
        # sum the lengths of the displ difference vectors
        sumlen = 0
        for i in xrange(N):
            udif[0, i] = ts1[itime, subarray[i + 1]] - ts1[itime, subarray[0]]
            udif[1, i] = ts2[itime, subarray[i + 1]] - ts2[itime, subarray[0]]
            udif[2, i] = ts3[itime, subarray[i + 1]] - ts3[itime, subarray[0]]
            sumlen = sumlen + np.sqrt(np.sum(udif[:, i].T ** 2))

        data = udif.T.reshape(udif.size)
        #
        # form solution
        # ptilde is (u1,1 u1,2 u1,3 u2,1 u2,2 u2,3).T
        ptilde = np.dot(g, data)
        #
        # place in uij_vector the full 9 elements of the displacement gradients
        # uij_vector is (u1,1 u1,2 u1,3 u2,1 u2,2 u2,3 u3,1 u3,2 u3,3).T
        # The following implements the free surface boundary condition
        u31 = -ptilde[2]
        u32 = -ptilde[5]
        u33 = -eta * (ptilde[0] + ptilde[4])
        uij_vector = np.r_[ptilde, u31, u32, u33]
        #
        # calculate predicted data
        pred = np.dot(A, ptilde)  # 9/8/92.I.3(9) and 8/26/92.I.3.T bottom
        #
        # calculate  residuals (misfits concatenated for all stations)
        misfit = pred - data

        # Calculate ts_M, misfit ratio.
        # calculate summed length of misfits (residual displacements)
        misfit_sq = misfit ** 2
        misfit_sq = np.reshape(misfit_sq, (N, 3)).T
        misfit_sumsq = np.empty(N)
        misfit_sumsq.fill(np.NaN)
        for i in xrange(N):
            misfit_sumsq[i] = misfit_sq[:, i].sum()
        misfit_len = np.sum(np.sqrt(misfit_sumsq))
        ts_M[itime] = misfit_len / sumlen
        #
        ts_data[itime, 0:3 * N] = data.T
        ts_pred[itime, 0:3 * N] = pred.T
        ts_misfit[itime, 0:3 * N] = misfit.T
        ts_ptilde[itime, :] = ptilde.T
        #
        #---------------------------------------------------------------
        #populate the displacement gradient matrix U
        U = np.zeros(9)
        U[:] = uij_vector
        U = U.reshape((3, 3))
        #
        # calculate strain tensors
        # Fung eqn 5.1 p 97 gives dui = (eij-wij)*dxj
        e = .5 * (U + U.T)
        ts_e[itime] = e

        # Three components of the rotation vector omega (=w here)
        w = np.empty(3)
        w.fill(np.NaN)
        w[0] = -ptilde[5]
        w[1] = ptilde[2]
        w[2] = .5 * (ptilde[3] - ptilde[1])

        # amount of total rotation is length of rotation vector
        ts_wmag[itime] = np.sqrt(np.sum(w ** 2))
        #
        # Calculate tilt and torsion
        ts_w1[itime] = w[0]
        ts_w2[itime] = w[1]
        ts_w3[itime] = w[2]  # torsion in radians
        ts_tilt[itime] = np.sqrt(w[0] ** 2 + w[1] ** 2)
            # 7/21/06.II.6(19), amount of tilt in radians

        #---------------------------------------------------------------
        #
        # Here I calculate horizontal quantities only
        # ts_dh is horizontal dilatation (+ --> expansion).
        # Total dilatation, ts_dd, will be calculated outside the time
        # step loop.
        #
        ts_dh[itime] = e[0, 0] + e[1, 1]
        #
        # find maximum shear strain in horizontal plane, and find its azimuth
        eh = np.r_[np.c_[e[0, 0], e[0, 1]], np.c_[e[1, 0], e[1, 1]]]
            # 7/21/06.II.2(4)
        gammah = eh - np.trace(eh) * np.eye(2) / 2.
            # 9/14/92.II.4, 7/21/06.II.2(5)

        # eigvecs are principal axes, eigvals are principal strains
        [eigvals, _eigvecs] = np.linalg.eig(gammah)
        # max shear strain, from Fung (1965, p71, eqn (8)
        ts_sh[itime] = .5 * (max(eigvals) - min(eigvals))

        # calculate max of total shear strain, not just horizontal strain
        # eigvecs are principal axes, eigvals are principal strains
        [eigvalt, _eigvect] = np.linalg.eig(e)
        # max shear strain, from Fung (1965, p71, eqn (8)
        ts_s[itime] = .5 * (max(eigvalt) - min(eigvalt))
        #

    #=========================================================================
    #
    # (total) dilatation is a scalar times horizontal dilatation owing to there
    # free surface boundary condition
    ts_d = ts_dh * (1 - eta)

    # load output structure
    out = dict()

    out['A'] = A
    out['g'] = g
    out['Ce'] = Ce

    out['ts_d'] = ts_d
    out['sigmad'] = sigmad

    out['ts_dh'] = ts_dh
    out['sigmadh'] = sigmadh

    out['ts_s'] = ts_s
    out['Cgamma'] = Cgamma

    out['ts_sh'] = ts_sh
    out['Cgammah'] = Cgammah

    out['ts_wmag'] = ts_wmag
    out['Cw'] = Cw

    out['ts_w1'] = ts_w1
    out['sigmaw1'] = sigmaw1
    out['ts_w2'] = ts_w2
    out['sigmaw2'] = sigmaw2
    out['ts_w3'] = ts_w3
    out['sigmaw3'] = sigmaw3

    out['ts_tilt'] = ts_tilt
    out['sigmat'] = sigmat

    out['ts_data'] = ts_data
    out['ts_pred'] = ts_pred
    out['ts_misfit'] = ts_misfit
    out['ts_M'] = ts_M
    out['ts_e'] = ts_e

    out['ts_ptilde'] = ts_ptilde
    out['Cp'] = Cp

    out['ts_M'] = ts_M

    return out


def sonic_pp(stream, win_len, win_frac, sll_x, slm_x, sll_y, slm_y, sl_s,
          semb_thres, vel_thres, frqlow, frqhigh, stime, etime, prewhiten,
          verbose=False, coordsys='lonlat', timestamp='mlabday', njobs=2,
          ppservers=(), secret='verysecret'):
    """
    Parrallelized Version of sonic. EXPERIMENTAL!

    .. rubric:: Usage

    - multiprocessing on local machine only: just replace sonic by sonic_pp
      and set njobs to number of cores in local machine

    - serveral machines:
        1. goto clients and start workers (important to make sure it uses
           the right python version with obspy installed)::

               $ python /usr/bin/ppserver -s <secret> -w <ncpus>

        2. replace sonic by sonic_pp and set njobs, clientlist and secret:

           >>> sonic_pp(*args, **kwargs, njobs=njobs,  # doctest: +SKIP
           ...          pservers=('client1', 'client2',), secret=<secret>)
    """
    import pp

    job_server = pp.Server(ppservers=ppservers, secret=secret)
    if verbose:
        print("Starting pp with", job_server.get_ncpus(), "workers")
    jobs = list()
    job_len = (etime - stime) / njobs

    for ts in np.arange(njobs):
        job_stime = stime + ts * job_len
        job_etime = stime + (ts + 1) * job_len + win_len
        if job_etime > etime:
            job_etime = etime

        jobs.append(job_server.submit(sonic, (stream, win_len, win_frac, sll_x,
            slm_x, sll_y, slm_y, sl_s, semb_thres, vel_thres, frqlow, frqhigh,
            job_stime, job_etime, prewhiten, verbose, coordsys, timestamp),
            (sonic, bbfk, get_geometry, get_timeshift, get_spoint,
             ndarray2ptr3D, cosine_taper), ('math', 'warnings', 'ctypes',
            'numpy', 'obspy.signal.util', 'obspy.core', 'scipy.integrate'),
            globals=globals()))

    i = 0
    for job in jobs:
        if i == 0:
            ret = job()
        else:
            ret = np.r_[ret, job()]
        i += 1

    if verbose:
        job_server.print_stats()

    return ret


def sonic(stream, win_len, win_frac, sll_x, slm_x, sll_y, slm_y, sl_s,
          semb_thres, vel_thres, frqlow, frqhigh, stime, etime, prewhiten,
          verbose=False, coordsys='lonlat', timestamp='mlabday'):
    """
    Method for Seismic-Array-Beamforming/FK-Analysis

    :param stream: Stream object, the trace.stats dict like class must
        contain a obspy.core.util.AttribDict with 'latitude', 'longitude' (in
        degrees) and 'elevation' (in km), or 'x', 'y', 'elevation' (in km)
        items/attributes. See param coordsys
    :type win_len: Float
    :param win_len: Sliding window length in seconds
    :type win_frac: Float
    :param win_frac: Fraction of sliding window to use for step
    :type sll_x: Float
    :param sll_x: slowness x min (lower)
    :type slm_x: Float
    :param slm_x: slowness x max
    :type sll_y: Float
    :param sll_y: slowness y min (lower)
    :type slm_y: Float
    :param slm_y: slowness y max
    :type sl_s: Float
    :param sl_s: slowness step
    :type semb_thres: Float
    :param semb_thres: Threshold for semblance
    :type vel_thres: Float
    :param vel_thres: Threshold for velocity
    :type frqlow: Float
    :param frqlow: lower frequency for fk
    :type frqhigh: Float
    :param frqhigh: higher frequency for fk
    :type stime: UTCDateTime
    :param stime: Starttime of interest
    :type etime: UTCDateTime
    :param etime: Endtime of interest
    :type prewhiten: int
    :param prewhiten: Do prewhitening, values: 1 or 0
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :type timestamp: string
    :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec' returns
        the timestamp in secons since 1970-01-01T00:00:00, 'mlabday'
        returns the timestamp in days (decimals represent hours, minutes
        and seconds) since '0001-01-01T00:00:00' as needed for matplotlib
        date plotting (see e.g. matplotlibs num2date)
    :return: numpy.ndarray of timestamp, relative power, absolute power,
        backazimut, slowness
    """
    res = []
    eotr = True
    #XXX move all the the ctypes related stuff to bbfk (Moritz's job)

    # check that sampling rates do not vary
    df = stream[0].stats.sampling_rate
    if len(stream) != len(stream.select(sampling_rate=df)):
        msg = 'in sonic sampling rates of traces in stream are not equal'
        raise ValueError(msg)

    grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
    grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

    geometry = get_geometry(stream, coordsys=coordsys, verbose=verbose)

    if verbose:
        print("geometry:")
        print(geometry)
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    time_shift_table_numpy = get_timeshift(geometry, sll_x, sll_y,
                                                    sl_s, grdpts_x, grdpts_y)
    time_shift_table = ndarray2ptr3D(time_shift_table_numpy)
    # fillup the double trace pointer
    nstat = len(stream)
    trace = (C.c_void_p * nstat)()
    ntrace = np.empty(nstat, dtype="int32", order="C")
    for i, tr in enumerate(stream):
        # assure data are of correct type
        tr.data = np.require(tr.data, 'float64', ['C_CONTIGUOUS'])
        trace[i] = tr.data.ctypes.data_as(C.c_void_p)
        ntrace[i] = len(tr.data)

    # offset of arrays
    spoint, _epoint = get_spoint(stream, stime, etime)
    #
    # loop with a sliding window over the data trace array and apply bbfk
    #
    df = stream[0].stats.sampling_rate
    nsamp = int(win_len * df)
    nstep = int(nsamp * win_frac)

    # generate plan for rfftr
    nfft = nextpow2(nsamp)
    newstart = stime
    offset = 0
    while eotr:
        try:
            buf = bbfk(spoint, offset, trace, ntrace, time_shift_table, frqlow,
                       frqhigh, df, nsamp, nstat, prewhiten, grdpts_x,
                       grdpts_y, nfft)
            abspow, power, ix, iy = buf
        except IndexError:
            break

        # here we compute baz, slow
        slow_x = sll_x + ix * sl_s
        slow_y = sll_y + iy * sl_s

        slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
        if slow < 1e-8:
            slow = 1e-8
        azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
        baz = azimut - np.sign(azimut) * 180
        if power > semb_thres and 1. / slow > vel_thres:
            res.append(np.array([newstart.timestamp, power, abspow, baz,
                                 slow]))
            if verbose:
                print(newstart, (newstart + (nsamp / df)), res[-1][1:])
        if (newstart + (nsamp + nstep) / df) > etime:
            eotr = False
        offset += nstep

        newstart += nstep / df
    res = np.array(res)
    if timestamp == 'julsec':
        pass
    elif timestamp == 'mlabday':
        # 719162 == hours between 1970 and 0001
        res[:, 0] = res[:, 0] / (24. * 3600) + 719162
    else:
        msg = "Option timestamp must be one of 'julsec', or 'mlabday'"
        raise ValueError(msg)
    return np.array(res)


def bbfk(spoint, offset, trace, ntrace, stat_tshift_table, flow, fhigh,
         digfreq, nsamp, nstat, prewhiten, grdpts_x, grdpts_y, nfft):
    """
    Note: Interface not fixed jet

    :type spoint: int
    :param spoint: Start sample point, probably in julian seconds
    :type offset: int
    :param offset: The Offset which is counted upwards nwin for shifting array
    :type trace: ??
    :param trace: The trace matrix, containing the time serious for various
        stations
    :type ntrace: float
    :param ntrace: ntrace vector
    :type stat_tshift_table: ??
    :param stat_tshift_table: The time shift table for each station for the
        slowness grid
    :type flow: float
    :param flow: Lower frequency for fk
    :type fhigh: float
    :param fhigh: Higher frequency for fk
    :type digfreq: float
    :param digfreq: The common sampling rate in group
    :type nsamp: int
    :int nsamp: Number of samples
    :tpye nstat: int
    :param nstat: Number of stations
    :type prewhiten: int
    :param prewhiten: Integer regulating prewhitening
    :type grdpts_x: int
    :param grdpts_x: Number of grid points in x direction to loop over
    :type grdpts_y: int
    :param grdpts_y: Number of grid points in y direction to loop over
    :type nfft: int
    :param nfft: Number of points to use for fft

    :return: Tuple with fields:
        | **float abs:** The absolut power, output variable printed to file
        | **float rel:** The relative power, output variable printed to file
        | **int ix:** ix output for backazimuth calculation
        | **int iy:** iy output for backazimuth calculation
    """
    # allocate output variables
    abspow = C.c_float()
    power = C.c_float()
    ix = C.c_int()
    iy = C.c_int()

    errcode = clibsignal.bbfk(spoint, offset,
                              C.cast(trace, C.POINTER(C.c_void_p)), ntrace,
                              C.byref(stat_tshift_table), C.byref(abspow),
                              C.byref(power), C.byref(ix), C.byref(iy), flow,
                              fhigh, digfreq, nsamp, nstat, prewhiten,
                              grdpts_x, grdpts_y, nfft)

    if errcode == 0:
        pass
    elif errcode == 1:
        raise IndexError('bbfk: Index out of bounds, window exceeds data')
    else:
        raise Exception('bbfk: C-Extension returned error %d' % errcode)
    return abspow.value, power.value, ix.value, iy.value


def get_geometry(stream, coordsys='lonlat', return_center=False,
                 verbose=False):
    """
    Method to calculate the array geometry and the center coordinates in km

    :param stream: Stream object, the trace.stats dict like class must
        contain a obspy.core.util.attribdict with 'latitude', 'longitude' (in
        degrees) and 'elevation' (in km), or 'x', 'y', 'elevation' (in km)
        items/attributes. See param coordsys
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :param return_center: Retruns the center coordinates as extra tuple
    :return: Returns the geometry of the stations as 2d numpy.ndarray
            The first dimension are the station indexes with the same order
            as the traces in the stream object. The second index are the
            values of [lat, lon, elev] in km
            last index contains center [lat, lon, elev] in degrees and km if
            return_center is true
    """
    nstat = len(stream)
    center_lat = 0.
    center_lon = 0.
    center_h = 0.
    geometry = np.empty((nstat, 3))

    if isinstance(stream, Stream):
        for i, tr in enumerate(stream):
            if coordsys == 'lonlat':
                geometry[i, 0] = tr.stats.coordinates.longitude
                geometry[i, 1] = tr.stats.coordinates.latitude
                geometry[i, 2] = tr.stats.coordinates.elevation
            elif coordsys == 'xy':
                geometry[i, 0] = tr.stats.coordinates.x
                geometry[i, 1] = tr.stats.coordinates.y
                geometry[i, 2] = tr.stats.coordinates.elevation
    elif isinstance(stream, np.ndarray):
        geometry = stream.copy()
    else:
        raise TypeError('only Stream or numpy.ndarray allowed')

    if verbose:
        print("coordys = " + coordsys)

    if coordsys == 'lonlat':
        center_lon = geometry[:, 0].mean()
        center_lat = geometry[:, 1].mean()
        center_h = geometry[:, 2].mean()
        for i in np.arange(nstat):
            x, y = utlGeoKm(center_lon, center_lat, geometry[i, 0],
                            geometry[i, 1])
            geometry[i, 0] = x
            geometry[i, 1] = y
            geometry[i, 2] -= center_h
    elif coordsys == 'xy':
        geometry[:, 0] -= geometry[:, 0].mean()
        geometry[:, 1] -= geometry[:, 1].mean()
        geometry[:, 2] -= geometry[:, 2].mean()
    else:
        raise ValueError("Coordsys must be one of 'lonlat', 'xy'")

    if return_center:
        return np.c_[geometry.T,
                     np.array((center_lon, center_lat, center_h))].T
    else:
        return geometry


def get_timeshift(geometry, sll_x, sll_y, sl_s, grdpts_x, grdpts_y):
    """
    Returns timeshift table for given array geometry

    :param geometry: Nested list containing the arrays geometry, as returned by
            get_group_geometry
    :param sll_x: slowness x min (lower)
    :param sll_y: slowness y min (lower)
    :param sl_s: slowness step
    :param grdpts_x: number of grid points in x direction
    :param grdpts_x: number of grid points in y direction
    """
    nstat = len(geometry)  # last index are center coordinates

    time_shift_tbl = np.empty((nstat, grdpts_x, grdpts_y), dtype="float32")
    for k in xrange(grdpts_x):
        sx = sll_x + k * sl_s
        for l in xrange(grdpts_y):
            sy = sll_y + l * sl_s
            time_shift_tbl[:, k, l] = sx * geometry[:, 0] + sy * geometry[:, 1]

    return time_shift_tbl


def get_spoint(stream, stime, etime):
    """
    :param stime: UTCDateTime to start
    :param etime: UTCDateTime to end
    """
    slatest = stream[0].stats.starttime
    eearliest = stream[0].stats.endtime
    for tr in stream:
        if tr.stats.starttime >= slatest:
            slatest = tr.stats.starttime
        if tr.stats.endtime <= eearliest:
            eearliest = tr.stats.endtime

    nostat = len(stream)
    spoint = np.empty(nostat, dtype="int32", order="C")
    epoint = np.empty(nostat, dtype="int32", order="C")
    # now we have to adjust to the beginning of real start time
    if slatest > stime:
        msg = "Specified start-time is smaller than starttime in stream"
        raise ValueError(msg)
    if eearliest < etime:
        msg = "Specified end-time bigger is than endtime in stream"
        raise ValueError(msg)
    for i in xrange(nostat):
        offset = int(((stime - slatest) / stream[i].stats.delta + 1.))
        negoffset = int(((eearliest - etime) / stream[i].stats.delta + 1.))
        diffstart = slatest - stream[i].stats.starttime
        frac, ddummy = math.modf(diffstart)
        spoint[i] = int(ddummy)
        if frac > stream[i].stats.delta * 0.25:
            msg = "Difference in start times exceeds 25% of samp rate"
            warnings.warn(msg)
        spoint[i] += offset
        diffend = stream[i].stats.endtime - eearliest
        frac, ddummy = math.modf(diffend)
        epoint[i] = int(ddummy)
        epoint[i] += negoffset

    return spoint, epoint


def ndarray2ptr3D(ndarray):
    """
    Construct pointer for ctypes from numpy.ndarray
    """
    ptr = C.c_void_p
    dim1, dim2, _dim3 = ndarray.shape
    voids = []
    for i in xrange(dim1):
        row = ndarray[i]
        p = (ptr * dim2)(*[col.ctypes.data_as(ptr) for col in row])
        voids.append(C.cast(p, C.c_void_p))
    return (ptr * dim1)(*voids)


def cosine_taper(ndat, fraction=0.1):
    """
    Returns cosine taper of size ndat and taper_fraction.

    C Extension for generating cosine taper

    :param ndat: Number of data points
    :param fraction: Taper fraction. Default is 10% which tapers 5% from
        the beginning and 5% form the end

    .. rubric:: Example

    >>> tap = cosine_taper(100, fraction=1.0)
    >>> buf = 0.5*(1+np.cos(np.linspace(np.pi, 3*np.pi, 100)))
    >>> abs(tap - buf).max() < 1e-2
    True
    """
    data = np.empty(ndat, dtype='float64')
    # the c extension tapers fraction from the beginning and the end,
    # therefore we half it
    frac = C.c_double(fraction / 2.0)

    errcode = clibsignal.cosine_taper(data, ndat, frac)
    if errcode != 0:
        raise Exception('bbfk: C-Extension returned error %d' % errcode)
    return data


def array_transff_wavenumber(coords, klim, kstep, coordsys='lonlat'):
    """
    Returns array transfer function as a function of wavenumber difference

    :type coords: numpy.ndarray
    :param coords: coordinates of stations in longitude and latitude in degrees
        elevation in km, or x, y, z in km
    :type coordsys: string
    :param coordsys: valid values: 'lonlat' and 'xy', choose which coordinates
        to use
    :param klim: either a float to use symmetric limits for wavenumber
        differences or the tupel (kxmin, kxmax, kymin, kymax)
    """
    coords = get_geometry(coords, coordsys)
    if isinstance(klim, float):
        kxmin = -klim
        kxmax = klim
        kymin = -klim
        kymax = klim
    elif isinstance(klim, tuple):
        if len(klim) == 4:
            kxmin = klim[0]
            kxmax = klim[1]
            kymin = klim[2]
            kymax = klim[3]
    else:
        raise TypeError('klim must either be a float or a tuple of length 4')

    nkx = np.ceil((kxmax + kstep / 10. - kxmin) / kstep)
    nky = np.ceil((kymax + kstep / 10. - kymin) / kstep)

    transff = np.empty((nkx, nky))

    for i, kx in enumerate(np.arange(kxmin, kxmax + kstep / 10., kstep)):
        for j, ky in enumerate(np.arange(kymin, kymax + kstep / 10., kstep)):
            _sum = 0j
            for k in xrange(len(coords)):
                _sum += np.exp(complex(0.,
                        coords[k, 0] * kx + coords[k, 1] * ky))
            transff[i, j] = abs(_sum) ** 2

    transff /= transff.max()
    return transff


def array_transff_freqslowness(coords, slim, sstep, fmin, fmax, fstep,
                               coordsys='lonlat'):
    """
    Returns array transfer function as a function of slowness difference and
    frequency.

    :type coords: numpy.ndarray
    :param coords: coordinates of stations in longitude and latitude in degrees
        elevation in km, or x, y, z in km
    :type coordsys: string
    :param coordsys: valid values: 'lonlat' and 'xy', choose which coordinates
        to use
    :param slim: either a float to use symmetric limits for slowness
        differences or the tupel (sxmin, sxmax, symin, symax)
    :type fmin: double
    :param fmin: minimum frequency in signal
    :type fmax: double
    :param fmin: maximum frequency in signal
    :type fstep: double
    :param fmin: frequency sample distance
    """
    coords = get_geometry(coords, coordsys)
    if isinstance(slim, float):
        sxmin = -slim
        sxmax = slim
        symin = -slim
        symax = slim
    elif isinstance(slim, tuple):
        if len(slim) == 4:
            sxmin = slim[0]
            sxmax = slim[1]
            symin = slim[2]
            symax = slim[3]
    else:
        raise TypeError('slim must either be a float or a tuple of length 4')

    nsx = np.ceil((sxmax + sstep / 10. - sxmin) / sstep)
    nsy = np.ceil((symax + sstep / 10. - symin) / sstep)
    nf = np.ceil((fmax + fstep / 10. - fmin) / fstep)

    transff = np.empty((nsx, nsy))
    buff = np.zeros(nf)

    for i, sx in enumerate(np.arange(sxmin, sxmax + sstep / 10., sstep)):
        for j, sy in enumerate(np.arange(symin, symax + sstep / 10., sstep)):
            for k, f in enumerate(np.arange(fmin, fmax + fstep / 10., fstep)):
                _sum = 0j
                for l in np.arange(len(coords)):
                    _sum += np.exp(complex(0., (coords[l, 0] * sx
                        + coords[l, 1] * sy) * 2 * np.pi * f))
                buff[k] = abs(_sum) ** 2
            transff[i, j] = cumtrapz(buff, dx=fstep)[-1]

    transff /= transff.max()
    return transff



#def generalized_beamformer(spoint, offset, stream, ntrace, steer, flow, fhigh,
#         digfreq, nsamp, nstat, prewhiten, grdpts_x, grdpts_y, nfft,method):
#    """
#
#    """
#    # start the code -------------------------------------------------
#    # This assumes that all stations and components have the same number of
#    # time samples, nt
#    df = digfreq/float(nfft)
#    nf = int((fhigh-flow)/df)+1
#    if nf > (nfft/2+1): nf = nfft/2+1
#    nlow = int(flow/df)
#
#    tap = cosTaper(nsamp,p=0.1)
#
#    trace = np.zeros((nstat,nfft),dtype=float)
#    for i,tr in enumerate(stream):
#       trace[i][:nsamp] = tr.data[spoint[i]+offset:spoint[i]+offset+nsamp]
#       trace[i][:nsamp] = detrend(trace[i][:nsamp],type='constant')
#       trace[i][:nsamp] *= tap
#
#
#    # in general, beamforming is done by simply computing the co-variances 
#    # of the signal at different receivers and than stear the matrix R with 
#    # "weights" which are the trial-DOAs e.g., Kirlin & Done, 1999
#    R = np.zeros((nstat, nstat,nf),dtype=complex)
#    dpow = 0.
#
#    # fill up R
#    for i in xrange(nstat):
#       for j in xrange(i,nstat):
#            xx = np.fft.rfft(trace[i],nfft) * np.fft.rfft(trace[j],nfft).conjugate()
#            if method == 'capon':
#                 R[i,j,0:nf] = xx[nlow:nlow+nf]/np.abs(np.sum(xx[nlow:nlow+nf]))
#                 if i != j:
#                     R[j,i,0:nf] = xx[nlow:nlow+nf].conjugate()/np.abs(np.sum(xx[nlow:nlow+nf]))
#            else :
#                 R[i,j,0:nf] = xx[nlow:nlow+nf]
#                 if i != j:
#                     R[j,i,0:nf] = xx[nlow:nlow+nf].conjugate()
#                 else:
#                     dpow += np.abs(np.sum(R[i,j,:]))
#
#    dpow *= nstat
#    
#    p = np.zeros((grdpts_x,grdpts_y,nf),dtype=float)
#    abspow = np.zeros((grdpts_x,grdpts_y),dtype=float)
#    relpow = np.zeros((grdpts_x,grdpts_y),dtype=float)
#    white = np.zeros((nf),dtype=float)
#
#    if method == "bf":
#    # P(f) = e.H R(f) e
#        for x in xrange(grdpts_x):
#            for y in xrange(grdpts_y):
#              for n in xrange(nf):
#                 e = np.zeros(nstat,dtype=complex)
#                 C = R[:,:,n]
#                 e = steer[:,x,y,n]
#                 eH = e.T
#                 Ce = np.dot(C,e) 
#                 p[x,y,n] = np.abs(np.dot(eH.conjugate(),Ce))
#              if prewhiten == 0:
#                 abspow[x,y] = np.sum(p[x,y,:])
#                 relpow[x,y] = abspow[x,y]/dpow
#              if prewhiten == 1:
#                 for n in xrange(nf):
#                   if p[x][y][n] > white[n]:
#                      white[n] = p[x,y,n]
#        if prewhiten == 1:
#            for x in xrange(grdpts_x):
#               for y in xrange(grdpts_y):
#                   abspow[x,y] = np.sum(p[x,y,:])
#                   relpow[x,y] = np.sum(p[x,y,:]/(white[:]*nf*nstat))
#        
#    elif method == "capon":
#    # P(f) = 1/(e.H R(f)^-1 e)
#        for x in xrange(grdpts_x):
#            for y in xrange(grdpts_y):
#              for n in xrange(nf):
#                  e = np.zeros(nstat,dtype=complex)
#                  C = R[:,:,n]
#                  IC = np.linalg.pinv(C)
#                  e = steer[:,x,y,n]
#                  eH = e.T
#                  ICe = np.dot(IC,e)
#                  p[x,y,n] = np.abs(1./np.dot(eH.conjugate(),ICe))
#              abspow[x,y] = np.sum(p[x,y,:])
#              if prewhiten == 0:
#                 relpow[x,y] = abspow[x,y]
#              if prewhiten == 1:
#                 for n in xrange(nf):
#                   if p[x][y][n] > white[n]:
#                      white[n] = p[x,y,n]
#        if prewhiten == 1:
#            for x in xrange(grdpts_x):
#               for y in xrange(grdpts_y):
#                  relpow[x,y] = np.sum(p[x,y,:]/(white[:]*nf*nstat))
#
#    # find the maximum in the map and return its value and the indices
#    ix,iy = pl.unravel_index(relpow.argmax(), relpow.shape)
#
#    print "%lf %lf %d %d %d %d\n"%(abspow.max(), relpow.max(), ix, iy,grdpts_x,grdpts_y)
#
#    return abspow.max(), relpow.max(), ix, iy


def array_processing(stream, win_len, win_frac, sll_x, slm_x, sll_y, slm_y, sl_s,
          semb_thres, vel_thres, frqlow, frqhigh, stime, etime, prewhiten,
          verbose=False, coordsys='lonlat', timestamp='mlabday',method='bbfk'):
    """
    Method for Seismic-Array-Beamforming/FK-Analysis/Capon

    :param stream: Stream object, the trace.stats dict like class must
        contain a obspy.core.util.AttribDict with 'latitude', 'longitude' (in
        degrees) and 'elevation' (in km), or 'x', 'y', 'elevation' (in km)
        items/attributes. See param coordsys
    :type win_len: Float
    :param win_len: Sliding window length in seconds
    :type win_frac: Float
    :param win_frac: Fraction of sliding window to use for step
    :type sll_x: Float
    :param sll_x: slowness x min (lower)
    :type slm_x: Float
    :param slm_x: slowness x max
    :type sll_y: Float
    :param sll_y: slowness y min (lower)
    :type slm_y: Float
    :param slm_y: slowness y max
    :type sl_s: Float
    :param sl_s: slowness step
    :type semb_thres: Float
    :param semb_thres: Threshold for semblance
    :type vel_thres: Float
    :param vel_thres: Threshold for velocity
    :type frqlow: Float
    :param frqlow: lower frequency for fk/capon
    :type frqhigh: Float
    :param frqhigh: higher frequency for fk/capon
    :type stime: UTCDateTime
    :param stime: Starttime of interest
    :type etime: UTCDateTime
    :param etime: Endtime of interest
    :type prewhiten: int
    :param prewhiten: Do prewhitening, values: 1 or 0
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :type timestamp: string
    :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec' returns
        the timestamp in secons since 1970-01-01T00:00:00, 'mlabday'
        returns the timestamp in days (decimals represent hours, minutes
        and seconds) since '0001-01-01T00:00:00' as needed for matplotlib
        date plotting (see e.g. matplotlibs num2date)
    :return: numpy.ndarray of timestamp, relative power, absolute power,
        backazimut, slowness
    """
    res = []
    eotr = True
    #XXX move all the the ctypes related stuff to bbfk (Moritz's job)

    # check that sampling rates do not vary
    df = stream[0].stats.sampling_rate
    if len(stream) != len(stream.select(sampling_rate=df)):
        msg = 'in sonic sampling rates of traces in stream are not equal'
        raise ValueError(msg)

    grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
    grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

    geometry = get_geometry(stream, coordsys=coordsys, verbose=verbose)

    if verbose:
        print("geometry:")
        print(geometry)
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    time_shift_table_numpy = get_timeshift(geometry, sll_x, sll_y,
                                                    sl_s, grdpts_x, grdpts_y)
    time_shift_table = ndarray2ptr3D(time_shift_table_numpy)
    # fillup the double trace pointer
    nstat = len(stream)
    trace = (C.c_void_p * nstat)()
    ntrace = np.empty(nstat, dtype="int32", order="C")
    for i, tr in enumerate(stream):
        # assure data are of correct type
        tr.data = np.require(tr.data, 'float64', ['C_CONTIGUOUS'])
        trace[i] = tr.data.ctypes.data_as(C.c_void_p)
        ntrace[i] = len(tr.data)

    # offset of arrays
    spoint, _epoint = get_spoint(stream, stime, etime)
    #
    # loop with a sliding window over the data trace array and apply bbfk
    #
    df = stream[0].stats.sampling_rate
    nsamp = int(win_len * df)
    nstep = int(nsamp * win_frac)

    # generate plan for rfftr
    nfft = nextpow2(nsamp)
    deltaf = df/float(nfft)
    nf = int((frqhigh-frqlow)/deltaf)+1
    if nf > (nfft/2+1): nf = nfft/2+1
    nlow = int(frqlow/deltaf)
    # to spead up the routine a bit we estimate all steering vectors in advance
    steer = np.zeros((nstat,grdpts_x,grdpts_y,nf),dtype=complex)
    steerH = np.zeros((nstat,grdpts_x,grdpts_y,nf),dtype=complex)
    for i in xrange(nstat):
           for x in xrange(grdpts_x):
                 for y in xrange(grdpts_y):
                       for n in xrange(nf):
                           wtau = 2.*np.pi*float(nlow+n)*deltaf*time_shift_table_numpy[i,x,y]
                           steer[i,x,y,n] = complex(np.cos(wtau), -1.*np.sin(wtau))
                           steerH[i,x,y,n] = complex(np.cos(wtau), np.sin(wtau))
    newstart = stime
    offset = 0
    while eotr:
        if method == 'bbfk':
            try:
                  buf = bbfk(spoint, offset, trace, ntrace, time_shift_table, frqlow,
                       frqhigh, df, nsamp, nstat, prewhiten, grdpts_x,
                       grdpts_y, nfft)
                  abspow, power, ix, iy = buf
            except IndexError:
                  break
        elif method == 'bf' or method == 'capon':
            tap = cosTaper(nsamp,p=0.1)
            trace = np.zeros((nstat,nsamp),dtype=float)
            try:
                  for i,tr in enumerate(stream):
                     trace[i][0:nsamp] = tr.data[spoint[i]+offset:spoint[i]+offset+nsamp]
                     trace[i][0:nsamp] = detrend(trace[i][0:nsamp],type='constant')
                     trace[i][0:nsamp] = trace[i][0:nsamp]*tap[0:nsamp]

                  buf = genbeam.generalized_beamformer(trace, steer, steerH, frqlow, frqhigh, df, nsamp, nstat, prewhiten, grdpts_x, grdpts_y, nfft, nf, method)
                  abspow, power, ix, iy = buf
            except IndexError:
                  break


        # here we compute baz, slow
        slow_x = sll_x + ix * sl_s
        slow_y = sll_y + iy * sl_s

        slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
        if slow < 1e-8:
            slow = 1e-8
        azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
        baz = azimut - np.sign(azimut) * 180
        if power > semb_thres and 1. / slow > vel_thres:
            res.append(np.array([newstart.timestamp, power, abspow, baz,
                                 slow]))
            if verbose:
                print(newstart, (newstart + (nsamp / df)), res[-1][1:])
        if (newstart + (nsamp + nstep) / df) > etime:
            eotr = False
        offset += nstep

        newstart += nstep / df
    res = np.array(res)
    if timestamp == 'julsec':
        pass
    elif timestamp == 'mlabday':
        # 719162 == hours between 1970 and 0001
        res[:, 0] = res[:, 0] / (24. * 3600) + 719162
    else:
        msg = "Option timestamp must be one of 'julsec', or 'mlabday'"
        raise ValueError(msg)
    return np.array(res)

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)

