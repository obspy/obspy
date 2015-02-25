# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: polarization.py
#   Author: Conny Hammer,Joachim Wassermann
#    Email: conny.hammer@geo.uni-potsdam.de,j.wassermann@lmu.de
#
# Copyright (C) 2008-2012 Conny Hammer, 2013 Joachim Wassermann
"""
Polarization Analysis

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import math
import warnings

import numpy as np
import scipy.odr
from scipy import signal
from scipy.optimize import fminbound

from obspy.signal.invsim import cosTaper


def eigval(datax, datay, dataz, fk, normf=1):
    """
    Polarization attributes of a signal.

    Computes the rectilinearity, the planarity and the eigenvalues of the given
    data which can be windowed or not.
    The time derivatives are calculated by central differences and the
    parameter ``fk`` describes the coefficients of the used polynomial. The
    values of ``fk`` depend on the order of the derivative you want to
    calculate. If you do not want to use derivatives you can simply
    use [1, 1, 1, 1, 1] for ``fk``.

    The algorithm is mainly based on the paper by [Jurkevics1988]_. The rest is
    just the numerical differentiation by central differences (carried out by
    the routine :func:`scipy.signal.lfilter(data, 1, fk)`).

    :type datax: :class:`~numpy.ndarray`
    :param datax: Data of x component. Note this is most usefull with
        windowed data, represented by a 2 dimensional array. First
        dimension window number, second dimension the actualy data.
    :type datay: :class:`~numpy.ndarray`
    :param datay: Data of y component. See also note in datax.
    :type dataz: :class:`~numpy.ndarray`
    :param dataz: Data of z component. See also note in datax.
    :type fk: list
    :param fk: Coefficients of polynomial used for calculating the time
        derivatives.
    :param normf: Factor for normalization.
    :return: **leigenv1, leigenv2, leigenv3, rect, plan, dleigenv, drect,
        dplan** - Smallest eigenvalue, Intermediate eigenvalue, Largest
        eigenvalue, Rectilinearity, Planarity, Time derivative of eigenvalues,
        time derivative of rectilinearity, Time derivative of planarity.
    """
    # function is made for windowed (two dimensional input).
    # However be nice and allow one dimensional input, see #919
    datax = np.atleast_2d(datax)
    datay = np.atleast_2d(datay)
    dataz = np.atleast_2d(dataz)
    covmat = np.zeros([3, 3])
    leigenv1 = np.zeros(datax.shape[0], dtype=np.float64)
    leigenv2 = np.zeros(datax.shape[0], dtype=np.float64)
    leigenv3 = np.zeros(datax.shape[0], dtype=np.float64)
    dleigenv = np.zeros([datax.shape[0], 3], dtype=np.float64)
    rect = np.zeros(datax.shape[0], dtype=np.float64)
    plan = np.zeros(datax.shape[0], dtype=np.float64)
    i = 0
    for i in range(datax.shape[0]):
        covmat[0][0] = np.cov(datax[i, :], rowvar=False)
        covmat[0][1] = covmat[1][0] = np.cov(datax[i, :], datay[i, :],
                                             rowvar=False)[0, 1]
        covmat[0][2] = covmat[2][0] = np.cov(datax[i, :], dataz[i, :],
                                             rowvar=False)[0, 1]
        covmat[1][1] = np.cov(datay[i, :], rowvar=False)
        covmat[1][2] = covmat[2][1] = np.cov(dataz[i, :], datay[i, :],
                                             rowvar=False)[0, 1]
        covmat[2][2] = np.cov(dataz[i, :], rowvar=False)
        _eigvec, eigenval, _v = (np.linalg.svd(covmat))
        eigenv = np.sort(eigenval)
        leigenv1[i] = eigenv[0]
        leigenv2[i] = eigenv[1]
        leigenv3[i] = eigenv[2]
        rect[i] = 1 - ((eigenv[1] + eigenv[0]) / (2 * eigenv[2]))
        plan[i] = 1 - ((2 * eigenv[0]) / (eigenv[1] + eigenv[2]))
    leigenv1 = leigenv1 / normf
    leigenv2 = leigenv2 / normf
    leigenv3 = leigenv3 / normf

    leigenv1_add = np.append(np.append([leigenv1[0]] * (np.size(fk) // 2),
                                       leigenv1),
                             [leigenv1[np.size(leigenv1) - 1]] *
                             (np.size(fk) // 2))
    dleigenv1 = signal.lfilter(fk, 1, leigenv1_add)
    dleigenv[:, 0] = dleigenv1[len(fk) - 1:]

    leigenv2_add = np.append(np.append([leigenv2[0]] * (np.size(fk) // 2),
                                       leigenv2),
                             [leigenv2[np.size(leigenv2) - 1]] *
                             (np.size(fk) // 2))
    dleigenv2 = signal.lfilter(fk, 1, leigenv2_add)
    dleigenv[:, 1] = dleigenv2[len(fk) - 1:]

    leigenv3_add = np.append(np.append([leigenv3[0]] * (np.size(fk) // 2),
                                       leigenv3),
                             [leigenv3[np.size(leigenv3) - 1]] *
                             (np.size(fk) // 2))
    dleigenv3 = signal.lfilter(fk, 1, leigenv3_add)
    dleigenv[:, 2] = dleigenv3[len(fk) - 1:]

    rect_add = np.append(np.append([rect[0]] * (np.size(fk) // 2), rect),
                         [rect[np.size(rect) - 1]] * (np.size(fk) // 2))
    drect = signal.lfilter(fk, 1, rect_add)
    drect = drect[len(fk) - 1:]

    plan_add = np.append(np.append([plan[0]] * (np.size(fk) // 2), plan),
                         [plan[np.size(plan) - 1]] * (np.size(fk) // 2))
    dplan = signal.lfilter(fk, 1, plan_add)
    dplan = dplan[len(fk) - 1:]

    return leigenv1, leigenv2, leigenv3, rect, plan, dleigenv, drect, dplan


def flinn(stream, noise_thres=0):
    """
    Computes the azimuth, incidence, rectilinearity and planarity after the
    eigenstructur decomposition method of [Flinn1965b]_.

    :param stream: ZNE sorted trace data
    :param noise_tresh: variance of noise sphere; data points are excluded when
        falling within the sphere with radius
     sqrt(noise_thres), default is set to 0
    :type noise_thres: float
    :return azimuth, incidence, reclin, plan:  azimuth, incidence,
        rectilinearity and planarity
    :type azimuth, incidence, reclin, plan: flaot, float, float, float
    """

    Z = []
    N = []
    E = []
    comp, npts = np.shape(stream)

    for i in range(0, npts):
        if (stream[0][i] ** 2 + stream[1][i] ** 2 + stream[2][i] ** 2) \
           > noise_thres:
            Z.append(stream[0][i])
            N.append(stream[1][i])
            E.append(stream[2][i])

    covmat = np.zeros([3, 3])
    X = np.vstack((E, N, Z))
    covmat = np.cov(X)
    eigvec, eigenval, v = (np.linalg.svd(covmat))
    # rectilinearity defined after Montalbetti & Kanasewich, 1970
    rect = 1. - np.sqrt(eigenval[1] / eigenval[0])
    # planarity defined after Jurkevics, 1988
    plan = 1. - (2.*eigenval[2] / (eigenval[1] + eigenval[0]))
    azimuth = 180 * math.atan2(eigvec[0][0], eigvec[1][0]) / math.pi
    eve = np.sqrt(eigvec[0][0] ** 2 + eigvec[1][0] ** 2)
    incidence = 180 * math.atan2(eve, eigvec[2][0]) / math.pi
    if azimuth < 0.:
        azimuth = 360. + azimuth
    if incidence < 0.0:
        incidence += 180.
    if incidence > 90.:
        incidence = 180. - incidence
        if azimuth > 180.:
            azimuth -= 180.
        else:
            azimuth += 180.
    if azimuth > 180.:
        azimuth -= 180.

    return azimuth, incidence, rect, plan


def instantFreq(data, sampling_rate):
    """
    simple program to estimate the instaneuous frequency based on the
    derivative of data and the analytical (hilbert) data

    :param data: ndarray(dtype(float))
    :type data: ndarray(dtype(float))
    :param sampling_rate: in Hz
    :type sampling_rate: float
    """
    x = data.copy()
    X = signal.hilbert(x)
    DX = np.gradient(X) * sampling_rate

    instf = (X.real * DX.imag - X.imag * DX.real) / \
            (2 * math.pi * (abs(X) ** 2))

    return instf


def vidaleAdapt(stream, noise_thres, fs, flow, fhigh, spoint, stime, etime):
    """
    Adaptive window polarization analysis after [Vidale1986]_ with the
    modification of adapted analysis window estimated by estimating the
    instantenous frequency. It returns the azimuth, incidence, rectilinearity
    planarity and ellipticity.

    :param stream: ZNE sorted trace data
    :param noise_thres: variance of noise sphere; data points are excluded when
        falling within the sphere with radius sqrt(noise_thres), Default = 0
    :type noise_thres: float
    :param fs: sampling rate
    :type fs: float
    :param flow: lower frequency for analysis
    :type flow: float
    :param fhigh: upper frequency limit for analysis
    :tpye fhigh: float
    :param spoint: array with trace individual start times in sample
    :type spoint: npdarray(type(int))
    :param stime: starttime (UTCDateTime) for analysis
    :type stime: UTCDateTime
    :param etime: endtime (UTCDateTime) for analysis
    :type etime: UTCDateTime
    :return azimuth, incidence, rectlit, plan, ellip: azimuth, incidence,
        rectilinearity planarity and ellipticity
    :type azimuth, incidence, rectlit, plan, ellip: float, float, float, float,
        float
    """
    W = 3.
    stream.sort(reverse=True)
    Z = stream[0].data.copy()
    N = stream[1].data.copy()
    E = stream[2].data.copy()

    Zi = instantFreq(Z, fs)
    Za = signal.hilbert(Z)
    Ni = instantFreq(N, fs)
    Na = signal.hilbert(N)
    Ei = instantFreq(E, fs)
    Ea = signal.hilbert(E)
    res = []

    # tap = cosTaper(nsamp, p=0.22)  # 0.22 matches 0.2 of historical C bbfk.c
    offset = int(3 * fs / flow)
    while True:
        adapt = int(3. * W * fs / (Zi[offset] + Ni[offset] + Ei[offset]))
        # in order to account for errors in the inst freq estimation
        if adapt > int(3 * fs / flow):
            adapt = int(3 * fs / flow)
        if adapt < int(3 * fs / fhigh):
            adapt = int(3 * fs / fhigh)
        adapt /= 2
        adapt = (2 * adapt) + 1
        newstart = stime + offset / fs
        if (newstart + (adapt / 2) / fs) > etime:
            break

        Zx = Za[int(spoint[2] + offset - adapt / 2):
                int(spoint[2] + offset + adapt / 2)]
        Nx = Na[int(spoint[1] + offset - adapt / 2):
                int(spoint[1] + offset + adapt / 2)]
        Ex = Ea[int(spoint[0] + offset - adapt / 2):
                int(spoint[0] + offset + adapt / 2)]
        Zx -= Zx.mean()
        Nx -= Nx.mean()
        Ex -= Ex.mean()

        covmat = np.zeros([3, 3], dtype=complex)

        covmat[0][0] = np.dot(Ex, Ex.conjugate())
        covmat[0][1] = np.dot(Ex, Nx.conjugate())
        covmat[1][0] = covmat[0][1].conjugate()
        covmat[0][2] = np.dot(Ex, Zx.conjugate())
        covmat[2][0] = covmat[0][2].conjugate()
        covmat[1][1] = np.dot(Nx, Nx.conjugate())
        covmat[1][2] = np.dot(Zx, Nx.conjugate())
        covmat[2][1] = covmat[1][2].conjugate()
        covmat[2][2] = np.dot(Zx, Zx.conjugate())

        eigvec, eigenval, v = (np.linalg.svd(covmat))

        def fun(x):
            return 1. - math.sqrt(
                ((eigvec[0][0] * (math.cos(x) + math.sin(x) * 1j)).real) ** 2 +
                ((eigvec[1][0] * (math.cos(x) + math.sin(x) * 1j)).real) ** 2 +
                ((eigvec[2][0] * (math.cos(x) + math.sin(x) * 1j)).real) ** 2)

        final = fminbound(fun, 0., math.pi, full_output=True)
        X = 1. - final[1]
        ellip = math.sqrt(1 - X ** 2) / X
        # rectilinearity defined after Montalbetti & Kanasewich, 1970
        rect = 1. - np.sqrt(eigenval[1] / eigenval[0])
        # planarity defined after Jurkevics, 1988
        plan = 1. - (2.*eigenval[2] / (eigenval[1] + eigenval[0]))

        azimuth = 180 * math.atan2(eigvec[0][0].real, eigvec[1][0].real) / \
            math.pi
        eve = np.sqrt(eigvec[0][0].real ** 2 + eigvec[1][0].real ** 2)

        incidence = 180 * math.atan2(eve, eigvec[2][0].real) / math.pi
        if azimuth < 0.:
            azimuth = 360. + azimuth
        if incidence < 0.0:
            incidence += 180.
        if incidence > 90.:
            incidence = 180. - incidence
            if azimuth > 180.:
                azimuth -= 180.
            else:
                azimuth += 180.
        if azimuth > 180.:
            azimuth -= 180.

        res.append(np.array([newstart.timestamp, azimuth, incidence, rect,
                             plan, ellip]))
        offset += 1

    return res


def particleMotionOdr(stream, noise_thres=0):
    """
    Computes the orientation of the particle motion vector based on
    orthogonal regression algorithm.

    :param stream: ZNE sorted trace data
    :param noise_tres: variance of noise sphere; data points are excluded when
        falling within the sphere with radius sqrt(noise_thres)
    :type noise_thres: float
    :return azimuth, incidence, az_error, in_error: Returns azimuth, incidence,
        error of azimuth, error of incidence
    """
    Z = []
    N = []
    E = []
    comp, npts = np.shape(stream)

    for i in range(0, npts):
        if (stream[0][i] ** 2 + stream[1][i] ** 2 + stream[2][i] ** 2) \
                > noise_thres:
            Z.append(stream[0][i])
            N.append(stream[1][i])
            E.append(stream[2][i])

    def fit_func(beta, x):
        # Eventually this is correct: return beta[0] * x + beta[1]
        return beta[0] * x

    data = scipy.odr.Data(E, N)
    model = scipy.odr.Model(fit_func)
    odr = scipy.odr.ODR(data, model, beta0=[1.])
    out = odr.run()
    az_slope = out.beta[0]
    az_error = out.sd_beta[0]
    # az_relerror = out.rel_error

    N = np.asarray(N)
    E = np.asarray(E)
    Z = np.asarray(Z)
    R = np.sqrt(N ** 2 + E ** 2)

    data = scipy.odr.Data(R, np.sqrt(Z ** 2))
    model = scipy.odr.Model(fit_func)
    odr = scipy.odr.ODR(data, model, beta0=[1.0])
    out = odr.run()
    in_slope = out.beta[0]
    in_error = out.sd_beta[0]

    azim = math.atan2(1.0, az_slope)
    inc = math.atan2(1.0, in_slope)
    az_error = 1.0 / ((1.0 ** 2 + az_slope ** 2) * azim) * az_error
    in_error = 1.0 / ((1.0 ** 2 + in_slope ** 2) * inc) * in_error
    azim *= 180.0 / math.pi
    inc *= 180.0 / math.pi

    return azim, inc, az_error, in_error


def getSpoint(stream, stime, etime):
    """
    Function for computing trace dependend start time in samples

    :param stime: UTCDateTime to start
    :type : UTCDatTime
    :param etime: UTCDateTime to end
    :type : UTCDatTime
    :return: spoint, epoint
    :type npdarray(int)
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
    for i in range(nostat):
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


def polarizationAnalysis(stream, win_len, win_frac, frqlow, frqhigh, stime,
                         etime, verbose=False, timestamp='mlabday',
                         method="pm", var_noise=0.0):
    """
    Method for Flinn/Jurkevics/ParticleMotion/Vidale calling

    :param stream: Stream object, the trace.stats dict like class must contain
        a three component trace. In case of Jurkevics or Vidale also an array
        like processing should be possible
    :type win_len: Float
    :param win_len: Sliding window length in seconds
    :type win_frac: Float
    :param win_frac: Fraction of sliding window to use for step
    :type var_noise: Float
    :param var_noise: resembles a sphere of noise in PM where the 3C is
        excluded
    :type frqlow: Float
    :param frqlow: lower frequency for PM
    :type frqhigh: Float
    :param frqhigh: higher frequency for PM
    :type stime: UTCDateTime
    :param stime: Starttime of interest
    :type etime: UTCDateTime
    :param etime: Endtime of interest
    :type timestamp: string
    :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec' returns
        the timestamp in secons since 1970-01-01T00:00:00, 'mlabday'
        returns the timestamp in days (decimals represent hours, minutes
        and seconds) since '0001-01-01T00:00:00' as needed for matplotlib
        date plotting (see e.g. matplotlibs num2date)
    :type method: str
    :param method: the method to use. one of "pm", "flinn" or "vidale".
    :return numpy.ndarray of timestamp, azimuth, incidence, reclin (az_error),
        plan (in_error), (ellip):
    """
    if method.lower() not in ["pm", "flinn", "vidale"]:
        msg = "Invalid method ('%s')" % method
        raise ValueError(msg)

    res = []

    # check that sampling rates do not vary
    fs = stream[0].stats.sampling_rate
    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = 'in array sampling rates of traces in stream are not equal'
        raise ValueError(msg)

    if verbose:
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    # offset of arrays
    spoint, _epoint = getSpoint(stream, stime, etime)
    # loop with a sliding window over the dat trace array and apply bbfk
    fs = stream[0].stats.sampling_rate
    if method.lower() == "vidale":
        res = vidaleAdapt(stream, var_noise, fs, frqlow, frqhigh, spoint,
                          stime, etime)
    else:
        nsamp = int(win_len * fs)
        nstep = int(nsamp * win_frac)
        newstart = stime
        # 0.22 matches 0.2 of historical C bbfk.c
        tap = cosTaper(nsamp, p=0.22)
        offset = 0
        # tr.sort(reverse=True)
        while (newstart + (nsamp + nstep) / fs) < etime:
            try:
                data = []
                Z = []
                N = []
                E = []
                for i, tr in enumerate(stream):
                    dat = tr.data[spoint[i] + offset:
                                  spoint[i] + offset + nsamp]
                    dat = (dat - dat.mean()) * tap
                    if 'Z' in tr.stats.channel:
                        Z = dat.copy()
                    if 'N' in tr.stats.channel:
                        N = dat.copy()
                    if 'E' in tr.stats.channel:
                        E = dat.copy()

                data.append(Z)
                data.append(N)
                data.append(E)
            except IndexError:
                break

            if method.lower() == "pm":
                azimuth, incidence, error_az, error_inc = \
                    particleMotionOdr(data, var_noise)
                if abs(error_az) < 0.1 and abs(error_inc) < 0.1:
                    res.append(np.array([newstart.timestamp + nsamp / fs,
                                         azimuth, incidence, error_az,
                                         error_inc]))
            if method.lower() == "flinn":
                azimuth, incidence, reclin, plan = flinn(data, var_noise)
                res.append(np.array([newstart.timestamp + nsamp / fs, azimuth,
                                     incidence, reclin, plan]))

            if verbose:
                print(newstart, (newstart + (nsamp / fs)), res[-1][1:])
            offset += nstep

            newstart += nstep / fs
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
