# -*- coding: utf-8 -*-
"""
Functions for polarization analysis.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import math
import warnings

import numpy as np
import scipy.odr
from scipy import signal
from scipy.optimize import fminbound

from obspy.signal.invsim import cosine_taper


def eigval(datax, datay, dataz, fk, normf=1.0):
    """
    Polarization attributes of a signal.

    Computes the rectilinearity, the planarity and the eigenvalues of the given
    data which can be windowed or not.
    The time derivatives are calculated by central differences and the
    parameter ``fk`` describes the coefficients of the used polynomial. The
    values of ``fk`` depend on the order of the derivative you want to
    calculate. If you do not want to use derivatives you can simply use
    [1, 1, 1, 1, 1] for ``fk``.

    The algorithm is mainly based on the paper by [Jurkevics1988]_. The rest is
    just the numerical differentiation by central differences (carried out by
    the routine :func:`scipy.signal.lfilter` (data, 1, fk)).

    :param datax: Data of x component. Note this is most useful with
        windowed data, represented by a 2 dimensional array. First
        dimension is the window number, second dimension is the data.
    :type datax: :class:`~numpy.ndarray`
    :param datay: Data of y component. See description of ``datax``.
    :type datay: :class:`~numpy.ndarray`
    :param dataz: Data of z component. See description of ``datax``.
    :type dataz: :class:`~numpy.ndarray`
    :param fk: Coefficients of polynomial used to calculate the time
        derivatives.
    :type fk: list
    :param normf: Factor for normalization.
    :type normf: float
    :return: **leigenv1, leigenv2, leigenv3, rect, plan, dleigenv, drect,
        dplan** - Smallest eigenvalue, Intermediate eigenvalue, Largest
        eigenvalue, Rectilinearity, Planarity, Time derivative of eigenvalues,
        time derivative of rectilinearity, Time derivative of planarity.
    """
    # The function is made for windowed data (two dimensional input).
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
    eigenstructure decomposition method of [Flinn1965b]_.

    :param stream: ZNE sorted trace data
    :type stream: list
    :param noise_tresh: Variance of the noise sphere; data points are excluded
        when falling within the sphere of radius sqrt(noise_thres),
        default is set to 0.
    :type noise_thres: float
    :returns:  azimuth, incidence, rectilinearity, and planarity
    """
    mask = (stream[0][:] ** 2 + stream[1][:] ** 2 + stream[2][:] ** 2
            ) > noise_thres
    x = np.zeros((3, mask.sum()), dtype=np.float64)
    # East
    x[0, :] = stream[2][mask]
    # North
    x[1, :] = stream[1][mask]
    # Z
    x[2, :] = stream[0][mask]

    covmat = np.cov(x)
    eigvec, eigenval, v = np.linalg.svd(covmat)
    # Rectilinearity defined after Montalbetti & Kanasewich, 1970
    rect = 1.0 - np.sqrt(eigenval[1] / eigenval[0])
    # Planarity defined after [Jurkevics1988]_
    plan = 1.0 - (2.0 * eigenval[2] / (eigenval[1] + eigenval[0]))
    azimuth = math.degrees(math.atan2(eigvec[0][0], eigvec[1][0]))
    eve = np.sqrt(eigvec[0][0] ** 2 + eigvec[1][0] ** 2)
    incidence = math.degrees(math.atan2(eve, eigvec[2][0]))
    if azimuth < 0.0:
        azimuth = 360.0 + azimuth
    if incidence < 0.0:
        incidence += 180.0
    if incidence > 90.0:
        incidence = 180.0 - incidence
        if azimuth > 180.0:
            azimuth -= 180.0
        else:
            azimuth += 180.0
    if azimuth > 180.0:
        azimuth -= 180.0

    return azimuth, incidence, rect, plan


def instantaneous_frequency(data, sampling_rate):
    """
    Simple function to estimate the instantaneous frequency based on the
    derivative of the data and the analytical (hilbert) data.

    :param data: The data array.
    :type data: :class:`numpy.ndarray`
    :param sampling_rate: The sampling rate in Hz.
    :type sampling_rate: float
    """
    x = signal.hilbert(data)
    dx = np.gradient(x) * sampling_rate

    instf = (x.real * dx.imag - x.imag * dx.real) / \
            (2 * math.pi * (abs(x) ** 2))

    return instf


def vidale_adapt(stream, noise_thres, fs, flow, fhigh, spoint, stime, etime,
                 adaptive=True):
    """
    Adaptive window polarization analysis after [Vidale1986]_ with the
    modification of adapted analysis window estimated by estimating the
    instantaneous frequency. It returns the azimuth, incidence, rectilinearity
    planarity and ellipticity.

    :param stream: ZNE containing trace data
    :type stream: :class:`~obspy.core.stream.Stream`
    :param noise_thres: Variance of the noise sphere; data points are excluded
        when falling within the sphere of radius sqrt(noise_thres),
        Default = 0
    :type noise_thres: float
    :param fs: sampling rate
    :type fs: float
    :param flow: lower frequency limit for analysis
    :type flow: float
    :param fhigh: upper frequency limit for analysis
    :type fhigh: float
    :param spoint: array with traces' individual start times in samples
    :type spoint: :class:`numpy.ndarray`
    :param stime: start time of the analysis
    :type stime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param etime: end time for the analysis
    :type etime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param adaptive: switch for adaptive window estimation (defaults to
        ``True``). If set to ``False``, the window will be estimated as
        ``3 * max(1/(fhigh-flow), 1/flow)``.
    :type adaptive: bool
    :returns: list of tuples containing azimuth, incidence, rectilinearity,
        planarity, and ellipticity
    """
    w = 3.0
    # sort for ZNE
    stream.sort(reverse=True)
    z = stream[0].data
    n = stream[1].data
    e = stream[2].data

    zi = instantaneous_frequency(z, fs)
    za = signal.hilbert(z)
    ni = instantaneous_frequency(n, fs)
    na = signal.hilbert(n)
    ei = instantaneous_frequency(e, fs)
    ea = signal.hilbert(e)
    res = []

    offset = int(3 * fs / flow)
    covmat = np.zeros([3, 3], dtype=np.complex128)
    while True:
        if adaptive:
            adapt = int(3.0 * w * fs / (zi[offset] + ni[offset] + ei[offset]))
            # in order to account for errors in the inst freq estimation
            if adapt > int(3.0 * fs / flow):
                adapt = int(3.0 * fs / flow)
            elif adapt < int(3.0 * fs / fhigh):
                adapt = int(3.0 * fs / fhigh)
            # XXX: was adapt /= 2
            adapt //= 2
            adapt = (2 * adapt) + 1
        else:
            adapt = max(int(3. * fs / (fhigh - flow)), int(3. * fs / flow))
        newstart = stime + offset / fs
        if (newstart + (adapt / 2) / fs) > etime:
            break

        zx = za[int(spoint[2] + offset - adapt / 2):
                int(spoint[2] + offset + adapt / 2)]
        nx = na[int(spoint[1] + offset - adapt / 2):
                int(spoint[1] + offset + adapt / 2)]
        ex = ea[int(spoint[0] + offset - adapt / 2):
                int(spoint[0] + offset + adapt / 2)]
        zx -= zx.mean()
        nx -= nx.mean()
        ex -= ex.mean()

        mask = (stream[0][:] ** 2 + stream[1][:] ** 2 + stream[2][:] ** 2) > \
            noise_thres
        xx = np.zeros((3, mask.sum()), dtype=np.complex128)
        # East
        xx[0, :] = ea
        # North
        xx[1, :] = na
        # Z
        xx[2, :] = za

        covmat = np.cov(xx)
        eigvec, eigenval, v = np.linalg.svd(covmat)

        # very similar to function flinn, possible could be unified
        def fun(x):
            return 1. - math.sqrt(
                ((eigvec[0][0] * (math.cos(x) + math.sin(x) * 1j)).real) ** 2 +
                ((eigvec[1][0] * (math.cos(x) + math.sin(x) * 1j)).real) ** 2 +
                ((eigvec[2][0] * (math.cos(x) + math.sin(x) * 1j)).real) ** 2)

        final = fminbound(fun, 0.0, math.pi, full_output=True)
        x = 1. - final[1]
        ellip = math.sqrt(1.0 - x ** 2) / x
        # rectilinearity defined after Montalbetti & Kanasewich, 1970
        rect = 1. - np.sqrt(eigenval[1] / eigenval[0])
        # planarity defined after [Jurkevics1988]_
        plan = 1. - (2.0 * eigenval[2] / (eigenval[1] + eigenval[0]))

        azimuth = 180 * math.atan2(eigvec[0][0].real, eigvec[1][0].real) / \
            math.pi
        eve = np.sqrt(eigvec[0][0].real ** 2 + eigvec[1][0].real ** 2)

        incidence = 180 * math.atan2(eve, eigvec[2][0].real) / math.pi
        if azimuth < 0.0:
            azimuth = 360.0 + azimuth
        if incidence < 0.0:
            incidence += 180.0
        if incidence > 90.0:
            incidence = 180.0 - incidence
            if azimuth > 180.0:
                azimuth -= 180.0
            else:
                azimuth += 180.0
        if azimuth > 180.0:
            azimuth -= 180.0
        res.append((newstart.timestamp, azimuth, incidence, rect, plan, ellip))
        offset += 1

    return res


def particle_motion_odr(stream, noise_thres=0):
    """
    Computes the orientation of the particle motion vector based on an
    orthogonal regression algorithm.

    :param stream: ZNE sorted trace data
    :type stream: :class:`~obspy.core.stream.Stream`
    :param noise_tres: variance of the noise sphere; data points are excluded
        when falling within the sphere of radius sqrt(noise_thres)
    :type noise_thres: float
    :returns: azimuth, incidence, error of azimuth, error of incidence
    """
    z = []
    n = []
    e = []
    comp, npts = np.shape(stream)

    for i in range(0, npts):
        if (stream[0][i] ** 2 + stream[1][i] ** 2 + stream[2][i] ** 2) \
                > noise_thres:
            z.append(stream[0][i])
            n.append(stream[1][i])
            e.append(stream[2][i])

    def fit_func(beta, x):
        # XXX: Eventually this is correct: return beta[0] * x + beta[1]
        return beta[0] * x

    data = scipy.odr.Data(e, n)
    model = scipy.odr.Model(fit_func)
    odr = scipy.odr.ODR(data, model, beta0=[1.])
    out = odr.run()
    az_slope = out.beta[0]
    az_error = out.sd_beta[0]

    n = np.asarray(n)
    e = np.asarray(e)
    z = np.asarray(z)
    r = np.sqrt(n ** 2 + e ** 2)

    data = scipy.odr.Data(r, abs(z))
    model = scipy.odr.Model(fit_func)
    odr = scipy.odr.ODR(data, model, beta0=[1.0])
    out = odr.run()
    in_slope = out.beta[0]
    in_error = out.sd_beta[0]

    azimuth = math.atan2(1.0, az_slope)
    incidence = math.atan2(1.0, in_slope)

    az_error = 1.0 / ((1.0 ** 2 + az_slope ** 2) * azimuth) * az_error
    # az_error = math.degrees(az_error)
    in_error = 1.0 / ((1.0 ** 2 + in_slope ** 2) * incidence) * in_error
    # in_error = math.degrees(in_error)

    azimuth = math.degrees(azimuth)
    incidence = math.degrees(incidence)

    if azimuth < 0.0:
        azimuth = 360.0 + azimuth
    if incidence < 0.0:
        incidence += 180.0
    if incidence > 90.0:
        incidence = 180.0 - incidence
        if azimuth > 180.0:
            azimuth -= 180.0
        else:
            azimuth += 180.0
    if azimuth > 180.0:
        azimuth -= 180.0

    return azimuth, incidence, az_error, in_error


def _get_s_point(stream, stime, etime):
    """
    Function for computing the trace dependent start time in samples

    :param stime: time to start
    :type stime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param etime: time to end
    :type etime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :returns: spoint, epoint
    """
    slatest = stream[0].stats.starttime
    eearliest = stream[0].stats.endtime
    for tr in stream:
        if tr.stats.starttime >= slatest:
            slatest = tr.stats.starttime
        if tr.stats.endtime <= eearliest:
            eearliest = tr.stats.endtime

    nostat = len(stream)
    spoint = np.empty(nostat, dtype=np.int32)
    epoint = np.empty(nostat, dtype=np.int32)
    # now we have to adjust to the beginning of real start time
    if slatest > stime:
        msg = "Specified start time is before latest start time in stream"
        raise ValueError(msg)
    if eearliest < etime:
        msg = "Specified end time is after earliest end time in stream"
        raise ValueError(msg)
    for i in range(nostat):
        offset = int(((stime - slatest) / stream[i].stats.delta + 1.))
        negoffset = int(((eearliest - etime) / stream[i].stats.delta + 1.))
        diffstart = slatest - stream[i].stats.starttime
        frac, _ = math.modf(diffstart)
        spoint[i] = int(diffstart)
        if frac > stream[i].stats.delta * 0.25:
            msg = "Difference in start times exceeds 25% of sampling rate"
            warnings.warn(msg)
        spoint[i] += offset
        diffend = stream[i].stats.endtime - eearliest
        frac, _ = math.modf(diffend)
        epoint[i] = int(diffend)
        epoint[i] += negoffset

    return spoint, epoint


def polarization_analysis(stream, win_len, win_frac, frqlow, frqhigh, stime,
                          etime, verbose=False, method="pm", var_noise=0.0,
                          adaptive=True):
    """
    Method carrying out polarization analysis with the [Flinn1965b]_,
    [Jurkevics1988]_, ParticleMotion, or [Vidale1986]_ algorithm.

    :param stream: 3 component input data.
    :type stream: :class:`~obspy.core.stream.Stream`
    :param win_len: Sliding window length in seconds.
    :type win_len: float
    :param win_frac: Fraction of sliding window to use for step.
    :type win_frac: float
    :param var_noise: resembles a sphere of noise in PM where the 3C is
        excluded
    :type var_noise: float
    :param frqlow: lower frequency. Only used for ``method='vidale'``.
    :type frqlow: float
    :param frqhigh: higher frequency. Only used for ``method='vidale'``.
    :type frqhigh: float
    :param stime: Start time of interest
    :type stime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param etime: End time of interest
    :type etime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param method: the method to use. one of ``"pm"``, ``"flinn"`` or
        ``"vidale"``.
    :type method: str
    :param adaptive: switch for adaptive window estimation (defaults to
        ``True``). If set to ``False``, the window will be estimated as
        ``3 * max(1/(fhigh-flow), 1/flow)``.
    :type adaptive: bool
    :rtype: dict
    :returns: Dictionary with keys ``"timestamp"`` (POSIX timestamp, can be
        used to initialize :class:`~obspy.core.utcdatetime.UTCDateTime`
        objects), ``"azimuth"``, ``"incidence"`` (incidence angle) and
        additional keys depending on used method: ``"azimuth_error"`` and
        ``"incidence_error"`` (for method ``"pm"``), ``"rectilinearity"`` and
        ``"planarity"`` (for methods ``"flinn"`` and ``"vidale"``) and
        ``"ellipticity"`` (for method ``"flinn"``). Under each key a
        :class:`~numpy.ndarray` is stored, giving the respective values
        corresponding to the ``"timestamp"`` :class:`~numpy.ndarray`.
    """
    if method.lower() not in ["pm", "flinn", "vidale"]:
        msg = "Invalid method ('%s')" % method
        raise ValueError(msg)

    res = []

    if stream.get_gaps():
        msg = 'Input stream must not include gaps:\n' + str(stream)
        raise ValueError(msg)

    if len(stream) != 3:
        msg = 'Input stream expected to be three components:\n' + str(stream)
        raise ValueError(msg)

    # check that sampling rates do not vary
    fs = stream[0].stats.sampling_rate
    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = "sampling rates of traces in stream are not equal"
        raise ValueError(msg)

    if verbose:
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    spoint, _epoint = _get_s_point(stream, stime, etime)
    if method.lower() == "vidale":
        res = vidale_adapt(stream, var_noise, fs, frqlow, frqhigh, spoint,
                           stime, etime)
    else:
        nsamp = int(win_len * fs)
        nstep = int(nsamp * win_frac)
        newstart = stime
        tap = cosine_taper(nsamp, p=0.22)
        offset = 0
        while (newstart + (nsamp + nstep) / fs) < etime:
            timestamp = newstart.timestamp + (float(nsamp) / 2 / fs)
            try:
                for i, tr in enumerate(stream):
                    dat = tr.data[spoint[i] + offset:
                                  spoint[i] + offset + nsamp]
                    dat = (dat - dat.mean()) * tap
                    if tr.stats.channel[-1].upper() == "Z":
                        z = dat.copy()
                    elif tr.stats.channel[-1].upper() == "N":
                        n = dat.copy()
                    elif tr.stats.channel[-1].upper() == "E":
                        e = dat.copy()
                    else:
                        msg = "Unexpected channel code '%s'" % tr.stats.channel
                        raise ValueError(msg)

                data = [z, n, e]
            except IndexError:
                break

            # we plot against the centre of the sliding window
            if method.lower() == "pm":
                azimuth, incidence, error_az, error_inc = \
                    particle_motion_odr(data, var_noise)
                res.append(np.array([
                    timestamp, azimuth, incidence, error_az, error_inc]))
            if method.lower() == "flinn":
                azimuth, incidence, reclin, plan = flinn(data, var_noise)
                res.append(np.array([
                    timestamp, azimuth, incidence, reclin, plan]))

            if verbose:
                print(newstart, newstart + float(nsamp) / fs, res[-1][1:])
            offset += nstep

            newstart += float(nstep) / fs

    res = np.array(res)

    result_dict = {"timestamp": res[:, 0],
                   "azimuth": res[:, 1],
                   "incidence": res[:, 2]}
    if method.lower() == "pm":
        result_dict["azimuth_error"] = res[:, 3]
        result_dict["incidence_error"] = res[:, 4]
    elif method.lower() == "vidale":
        result_dict["rectilinearity"] = res[:, 3]
        result_dict["planarity"] = res[:, 4]
        result_dict["ellipticity"] = res[:, 5]
    elif method.lower() == "flinn":
        result_dict["rectilinearity"] = res[:, 3]
        result_dict["planarity"] = res[:, 4]
    return result_dict


if __name__ == "__main__":
    import doctest
    doctest.testmod(exclude_empty=True)
