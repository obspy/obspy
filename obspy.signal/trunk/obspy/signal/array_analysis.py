#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: array.py
#  Purpose: Functions for Array Analysis
#   Author: Martin van Driel
#    Email: driel@geophysik.uni-muenchen.de
#
# Copyright (C) 2010 Martin van Driel
#---------------------------------------------------------------------
"""
Functions for Array Analysis

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import warnings
import numpy as np

def array_rotation_strain(subarray, ts1, ts2, ts3, vp, vs, array_coords,
                          sigmau):
    """
    This routine calculates the best-fitting rigid body rotation and 
    uniform strain as functions of time, and their formal errors, given 
    three-component ground motion time series recorded on a seismic array. 
    The theory implemented herein is presented in the papers:: 
    
      Spudich et al. (J. Geophys. Res., 1995), (abbreviated S95 herein) 
      Spudich and Fletcher (Bull. Seismol. Soc. Am., 2008)  (SF08) 
      Spudich and Fletcher (Bull. Seismol. Soc. Am., 2009). (SF09)

    
    This is a translation of the Matlab Code presented in (SF09) with 
    small changes in details only. Output has been checked to be the same 
    as the original Matlab Code.

    .. note:: 
        ts_ below means "time series"
    
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
        - If sigmau is a scalar, it will be used for all components of all
          stations. 
        - If sigmau is a 1D array of length Na, sigmau[i] will be the noise
          assigned to all components of the station corresponding to
          array_coords[i,:]
        - If sigmau is a 2D array of dimension  Na x 3, then sigmau[i,j] is
          used as the noise of station i, component j.  
        In all cases, this routine assumes that the noise covariance between
        different stations and/or components is zero.  
    :type subarray: numpy.ndarray 
    :param subarray: numpy array of subarray stations to use. I.e. if subarray
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
        | **g:** (array, dimension 6 x 3N) - generlized inverse matrix
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

    Warnings
    --------
    This routine does not check to verify that your array is small
    enough to conform to the assumption that the array aperture is less
    than 1/4 of the shortest seismic wavelength in the data.  See SF08
    for a discussion of this assumption.
    
    this code assumes that ts1[j,:], ts2[j,:], and ts3[j,:] are all sampled
    SIMULTANEOUSLY. 

    Notes
    -----
    ::

        Note On Specifying Input Array And Selecting Subarrays
        
        This routine allows the user to input the coordinates and ground
        motion time series of all stations in a seismic array having Na
        stations and the user may select for analysis a subarray of Nplus1
        <= Na stations.


        Discussion Of Physical Units Of Input And Ouput
        
        If the input seismograms are in units of displacement, the output
        strains and rotations will be in units of strain (unitless) and
        angle (radians).  If the input seismograms are in units of
        velocity, the output will be strain rate (units = 1/s) and rotation
        rate (rad/s).  Higher temporal derivative inputs yield higher
        temporal derivative outputs. 

        Input units of the array station coordinates must match the spatial
        units of the seismograms.  For example, if the input seismograms
        are in units of m/s^2, array coordinates must be entered in m. 
        

        Note On Coordinate System
        
        This routine assumes x1-x2-x3 is a RIGHT handed orthogonal
        coordinate system. x3 must point either UP or DOWN.  

    """

    
    # start the code -------------------------------------------------

    # This assumes that all stations and components have the same number of 
    # time samples, nt
    [nt,Na] = np.shape(ts1)

    # check to ensure all components have same duration
    if ts1.shape != ts2.shape:
        raise ValueError('ts1 and ts2 have different sizes')
    if ts1.shape != ts3.shape:
        raise ValueError('ts1 and ts3 have different sizes')

    # check to verify that the number of stations in ts1 agrees with the number
    # of stations in array_coords
    [nrac, ncac] = array_coords.shape
    if nrac != Na:
        msg = 'ts1 has %s columns(stations) but array_coords has ' % Na + \
              '%s rows(stations)' % nrac
        raise ValueError(msg)

    # check stations in subarray exist
    if min(subarray) < 0:
        raise ValueError('In subarray you have specified a station number < 0')
    if max(subarray) > Na:
        raise ValueError('In subarray you have specified a station number > Na')

    # extract the stations of the subarray to be used
    subarraycoords = array_coords[subarray ,:]

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
    eta = 1 - 2*vs**2/vp**2

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
        ss = subarraycoords[i+1,:] - subarraycoords[0,:]
        A[3*i:3*i + 3,:] = np.c_[np.r_[ss, z3t], np.r_[z3t, ss], \
            np.array([-eta*ss[2], \
            0., -ss[0], 0., -eta*ss[2], -ss[1]])].transpose()

    #------------------------------------------------------
    # define data covariance matrix Cd. 
    # step 1 - define data differencing matrix D
    # dimension of D is (3*N) * (3*Nplus1)
    I3 = np.eye(3)
    II = np.eye(3*N)
    Cd = np.array([])
    D = -I3

    for i in xrange(N-1):
        D = np.c_[D, -I3]
    D = np.r_[D, II].T

    # step 2 - define displacement u covariance matrix Cu
    # This assembles a covariance matrix Cu that reflects actual data errors.
    # populate Cu depending on the size of sigmau
    if np.size(sigmau) == 1:
        # sigmau is a scalar.  Make all diag elements of Cu the same
        Cu = sigmau**2 * np.eye(3*Nplus1)
        #print 'sigmau is scalar'
    elif np.shape(sigmau) == (np.size(sigmau),):
        # sigmau is a row or column vector
        # check dimension is okay
        if np.size(sigmau) != Na:
            raise ValueError('sigmau must have %s elements' % Na)
        junk = (np.c_[sigmau, sigmau, sigmau])**2; # matrix of variances
        Cu = np.diag( np.reshape( junk[subarray,:], (3*Nplus1)) )
        #print 'sigmau is vector'
    elif sigmau.shape == (Na, 3):
        Cu = np.diag( np.reshape( ((sigmau[subarray,:])**2).transpose(), \
            (3*Nplus1)) )
        #print 'sigmau is matrix'
    else:
        raise ValueError('sigmau has the wrong dimensions')

    # Cd is the covariance matrix of the displ differences
    # dim(Cd) is (3*N) * (3*N)
    Cd = np.dot(np.dot(D, Cu), D.T)


    #---------------------------------------------------------
    # form generalized inverse matrix g.  dim(g) is 6 x (3*N)
    Cdi = np.linalg.inv(Cd)
    AtCdiA = np.dot(np.dot(A.T,  Cdi),  A)
    g = np.dot(np.dot(np.linalg.inv(AtCdiA), A.T), Cdi)

    condition_number = np.linalg.cond(AtCdiA)

    if condition_number > 100:
        msg = 'Condition number is %s' % condition_number
        warnings.warn(msg)
    #else:
    #    print 'In array_rotation_strain, matrix condition number = ' + \
    #        str(condition_number)
                
    # set up storage for vectors that will contain time series
    ts_wmag = np.NaN * np.empty(nt)
    ts_w1 = np.NaN * np.empty(nt)
    ts_w2 = np.NaN * np.empty(nt)
    ts_w3 = np.NaN * np.empty(nt)
    ts_tilt = np.NaN * np.empty(nt)
    ts_dh = np.NaN * np.empty(nt)
    ts_dd = np.NaN * np.empty(nt)
    ts_sh = np.NaN * np.empty(nt)
    ts_s = np.NaN * np.empty(nt)
    ts_M = np.NaN * np.empty(nt)
    ts_pred = np.NaN * np.empty((nt,3*N))
    ts_misfit = np.NaN * np.empty((nt,3*N))
    ts_M = np.NaN * np.empty(nt)
    ts_data = np.NaN * np.empty((nt,3*N))
    ts_ptilde = np.NaN * np.empty((nt,6))

    # other matrices
    udif = np.NaN * np.empty((3,N))


    #---------------------------------------------------------------
    # here we define 4x6 Be and 3x6 Bw matrices.  these map the solution
    # ptilde to strain or to rotation.  These matrices will be used 
    # in the calculation of the covariances of strain and rotation.
    # Columns of both matrices correspond to the model solution vector
    # containing elements [u1,1 u1,2 u1,3 u2,1 u2,2 u2,3 ]'
    #
    # the rows of Be correpond to e11 e21 e22 and e33
    Be = np.zeros((4,6))
    Be[0,0] = 2.
    Be[1,1] = 1.
    Be[1,3] = 1.
    Be[2,4] = 2.
    Be[3,0] = -2*eta
    Be[3,4] = -2*eta
    Be = Be * .5
    #
    # the rows of Bw correspond to w21 w31 and w32
    Bw = np.zeros((3,6))
    Bw[0,1] = 1.
    Bw[0,3] = -1.
    Bw[1,2] = 2.
    Bw[2,5] = 2.
    Bw = Bw * .5
    #
    # this is the 4x6 matrix mapping solution to total shear strain gamma
    # where gamma = strain - tr(strain)/3 * eye(3)
    # the four elements of shear are 11, 12, 22, and 33.  It is symmetric.
    aa = (2 + eta) / 3
    b = (1 - eta) / 3
    c = (1 + 2 * eta) / 3
    Bgamma = np.zeros((4,6))
    Bgamma[0,0] = aa
    Bgamma[0,4] = -b
    Bgamma[2,2] = .5
    Bgamma[1,3] = .5
    Bgamma[2,0] = -b
    Bgamma[2,4] = aa
    Bgamma[3,0] = -c
    Bgamma[3,4] = -c
    #
    # this is the 3x6 matrix mapping solution to horizontal shear strain 
    # gamma 
    # the four elements of horiz shear are 11, 12, and 22.  It is symmetric.
    Bgammah = np.zeros((3,6))
    Bgammah[0,0] = .5
    Bgammah[0,4] = -.5
    Bgammah[1,1] = .5
    Bgammah[1,3] = .5
    Bgammah[2,0] = -.5
    Bgammah[2,4] = .5

    # solution covariance matrix.  dim(Cp) = 6 * 6
    # corresponding to solution elements [u1,1 u1,2 u1,3 u2,1 u2,2 u2,3 ]
    Cp = np.dot(np.dot(g, Cd), g.T)

    # Covariance of strain tensor elements 
    # Ce should be 4x4, corresp to e11, e21, e22, e33
    Ce = np.dot(np.dot(Be, Cp), Be.T)
    # Cw should be 3x3 corresp to w21, w31, w32
    Cw = np.dot(np.dot(Bw, Cp), Bw.T)

    # Cgamma is 4x4 corresp to 11, 12, 22, and 33.  
    Cgamma = np.dot(np.dot(Bgamma, Cp), Bgamma.T)
    #
    #  Cgammah is 3x3 corresp to 11, 12, and 22
    Cgammah = np.dot(np.dot(Bgammah, Cp), Bgammah.T)
    #
    #
    # covariance of the horizontal dilatation and the total dilatation
    # both are 1x1, i.e. scalars
    Cdh = Cp[0,0] + 2 * Cp[0,4] + Cp[4,4]
    sigmadh = np.sqrt(Cdh)

    # covariance of the (total) dilatation, ts_dd
    sigmadsq = (1 - eta)**2 * Cdh
    sigmad = np.sqrt(sigmadsq)
    #  
    # Cw3, covariance of w3 rotation, i.e. torsion, is 1x1, i.e. scalar
    Cw3 = (Cp[1,1] - 2*Cp[1,3] + Cp[3,3]) / 4
    sigmaw3 = np.sqrt(Cw3)

    # For tilt cannot use same approach because tilt is not a linear function
    # of the solution.  Here is an approximation : 
    # For tilt use conservative estimate from 
    # Papoulis (1965, p. 195, example 7.8)
    sigmaw1 = np.sqrt(Cp[5,5])
    sigmaw2 = np.sqrt(Cp[2,2])
    sigmat = max(sigmaw1,sigmaw2) * np.sqrt(2 - np.pi/2 )

    #
    # BEGIN LOOP OVER DATA POINTS IN TIME SERIES==============================
    #
    for itime in xrange(nt):
        #
        # data vector is differences of stn i displ from stn 1 displ
        # sum the lengths of the displ difference vectors
        sumlen = 0
        for i in xrange(N):
            udif[0,i] = ts1[itime, subarray[i+1]] - ts1[itime, subarray[0]]
            udif[1,i] = ts2[itime, subarray[i+1]] - ts2[itime, subarray[0]]
            udif[2,i] = ts3[itime, subarray[i+1]] - ts3[itime, subarray[0]]
            sumlen = sumlen + np.sqrt(np.sum(udif[:,i].T**2)) 

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
        uij_vector = np.r_[ptilde, u31, u32, u33 ]
        #
        # calculate predicted data
        pred = np.dot(A, ptilde) # 9/8/92.I.3(9) and 8/26/92.I.3.T bottom
        #
        # calculate  residuals (misfits concatenated for all stations)
        misfit = pred - data
        
        # Calculate ts_M, misfit ratio.
        # calculate summed length of misfits (residual displacments)
        misfit_sq = misfit**2
        misfit_sq = np.reshape(misfit_sq,(N,3)).T
        misfit_sumsq = np.NaN * np.empty(N)
        for i in xrange(N):
            misfit_sumsq[i] = misfit_sq[:,i].sum()
        misfit_len = np.sum(np.sqrt(misfit_sumsq))
        ts_M[itime] = misfit_len / sumlen
        #
        ts_data[itime,0:3*N] = data.T
        ts_pred[itime,0:3*N] = pred.T
        ts_misfit[itime,0:3*N] = misfit.T
        ts_ptilde[itime,:] = ptilde.T
        #
        #---------------------------------------------------------------
        #populate the displacement gradient matrix U
        U = np.zeros(9)
        U[:] = uij_vector
        U = U.reshape((3,3))
        #
        # calculate strain tensors
        # Fung eqn 5.1 p 97 gives dui = (eij-wij)*dxj
        e = .5 * (U + U.T)
     
        # Three components of the rotation vector omega (=w here)
        w = np.NaN * np.empty(3)
        w[0] = -ptilde[5]
        w[1] = ptilde[2]
        w[2] = .5*(ptilde[3]-ptilde[1])

        # amount of total rotation is length of rotation vector
        ts_wmag[itime] = np.sqrt(np.sum(w**2))
        #
        # Calculate tilt and torsion
        ts_w1[itime] = w[0]
        ts_w2[itime] = w[1]
        ts_w3[itime] = w[2] # torsion in radians
        ts_tilt[itime] = np.sqrt( w[0]**2 + w[1]**2 ) 
            # 7/21/06.II.6(19), amount of tilt in radians
      
        #---------------------------------------------------------------
        #
        # Here I calculate horizontal quantities only
        # ts_dh is horizontal dilatation (+ --> expansion).  
        # Total dilatation, ts_dd, will be calculated outside the time 
        # step loop.
        # 
        ts_dh[itime] = e[0,0] + e[1,1]
        #
        # find maximum shear strain in horizontal plane, and find its azimuth
        eh = np.r_[np.c_[e[0,0], e[0,1]], np.c_[e[1,0], e[1,1]]]
            # 7/21/06.II.2(4)
        gammah = eh - np.trace(eh) * np.eye(2) / 2.
            # 9/14/92.II.4, 7/21/06.II.2(5)

        # eigvecs are principal axes, eigvals are principal strains
        [eigvals,eigvecs] = np.linalg.eig(gammah)
        # max shear strain, from Fung (1965, p71, eqn (8)
        ts_sh[itime] = .5 * (max(eigvals) - min(eigvals))

        # calculate max of total shear strain, not just horizontal strain
        # eigvecs are principal axes, eigvals are principal strains
        [eigvalt,eigvect] = np.linalg.eig(e)
        # max shear strain, from Fung (1965, p71, eqn (8)
        ts_s[itime] = .5 * (max(eigvalt) - min(eigvalt) )
        #
        
    #=========================================================================
    #
    # (total) dilatation is a scalar times horizontal dilatation owing to ther
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

    out['ts_ptilde'] = ts_ptilde
    out['Cp'] = Cp

    out['ts_M'] = ts_M

    return out
