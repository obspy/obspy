#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: 3C_analysis.py
#  Purpose: Functions for Polariyation Analysis
#   Author: Joachim Wassermann
#    Email: j.wassermann@lmu.de
#
# Copyright (C) 2013 Joachim Wassermann
#---------------------------------------------------------------------
"""
Functions for Polarization Analysis

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import math
import warnings
import numpy as np
from obspy.signal.util import utlGeoKm, nextpow2
from obspy.signal.headers import clibsignal
from obspy.core import Stream
from obspy.core.util.decorator import deprecated
from scipy.integrate import cumtrapz
from obspy.signal.invsim import cosTaper


def particle_motion_ord(stream,npts,noise_thres):
    """
    :param stream: ZNE sorted trace data
    :param npts:  number of points for analysis
    :param noise_tresh: variance of noise sphere; data points are excluded when falling within the sphere with radius 
    sqrt(noise_thres)
    """


    Z = N = E = []
    for i in xrange(0,npts):
        if (stream[0][i]**2 + stream[1][i]**2  + stream[2][i]**2) > noise_thres:
            Z.append()=stream[0][i]
            N.append()=stream[1][i]
            E.append()=stream[2][i]

     new_points = length(Z)

    try:
        import scipy.odr
        #def fit_func(beta, x):
        #    return beta[0] * x + beta[1]

        fit_func = lambda beta, x: beta[0] * x 

        data = scipy.odr.Data(E, N)
        model = scipy.odr.Model(fit_func)
        odr = scipy.odr.ODR(data, model, beta0=[1.])
        out = odr.run()
        print out
        az_slope = out.beta[0]
        az_error = out.sd_beta[0]
        az_relerror = out.rel_error

        R = np.sqrt(N**2 + E**2)
        data = scipy.odr.Data(R, Z)
        model = scipy.odr.Model(fit_func)
        odr = scipy.odr.ODR(data, model, beta0=[1.])
        out = odr.run()
        print out
        in_slope = out.beta[0]
        in_error = out.sd_beta[0]
        in_relerror = out.rel_error

        azim = math.atan2(1,az_slope)*180./math.pi
        inc = math.atan2(1,in_slope)*180./math.pi

        retrun bazim,inc,az_relerror,in_relerror


    except ImportError:
                print "Error importing scipy.odr..."

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


def polarization_analysis(stream, win_len, win_frac, 
    var_noise, frqlow, frqhigh, stime, etime, verbose=False, timestamp='mlabday', method=0, store=nop):
    """
    Method for Flinn/Jurkevics-ParticleMotion-Vidale

    :param stream: Stream object, the trace.stats dict like class must
        contain a three component trace. In case of Jurkevics or Vidale also an array like
        processing should be possible
    :type win_len: Float
    :param win_len: Sliding window length in seconds
    :type win_frac: Float
    :param win_frac: Fraction of sliding window to use for step
    :type var_noise: Float
    :param var_noise: resembles a sphere of noise in PM where the 3C is excluded
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
    :type method: int
    :param method: the method to use 0 == PM, 1 == flinn, 2 == vidale
    :type store: int
    :param store: a function which is called on each iteration with the
        relative power map and the time offset as argument. Usefull for storing
        or plotting the map for each iteration. For this purpose the dump and
        nop function of this module can be used.
    :return: numpy.ndarray of timestamp, relative relpow, absolute relpow,
        backazimut, slowness
    """
    PM, JURKEVICS, VIDALE = 0, 1, 2
    res = []
    eotr = True

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
    spoint, _epoint = get_spoint(stream, stime, etime)
    #
    # loop with a sliding window over the dat trace array and apply bbfk
    #
    nstat = len(stream)
    fs = stream[0].stats.sampling_rate
    nsamp = int(win_len * fs)
    nstep = int(nsamp * win_frac)

    # generate plan for rfftr
    newstart = stime
    tap = cosTaper(nsamp, p=0.22)  # 0.22 matches 0.2 of historical C bbfk.c
    offset = 0
    while eotr:
        data = []
        data += []
        data += []
        try:
            for i, tr in enumerate(stream):
                dat = tr.data[spoint[i] + offset:
                    spoint[i] + offset + nsamp]
                dat = (dat - dat.mean()) * tap
                if 'Z' in tr.stats.channel:
                    data[0] = dat.copy()
                if 'N' in tr.stats.channel:
                    data[1] = dat.copy()
                if 'E' in tr.stats.channel:
                    data[2] = dat.copy()

        except IndexError:
            break
        if method == PM:
            azimuth,incidence,error_az,error_inc = particle_motion_ord(data,var_noise)
            if error_az < math.sqrt(var_noise) and error_inc < math.sqrt(var_noise):
                res.append(np.array([newstart.timestamp, azimuth, incidense, error_az,
                                 error_inc]))
        if method == JURKEVICS:
            ev1,ev2,ev3,evec,reclin = particle_motion(dat_z,dat_n,dat_e,var_noise)
            azimut = 180 * math.atan2(evec[0],evec[1]) / math.pi
            eve = np.sqrt(evec[0]**2 + evec[1]**2)
            incidense = 180 * math.atan2(eve,evec[3]) / math.pi
        if method == VIDLAE:
            azimut = 180 * math.atan2(slow_x, slow_y) / math.pi

        if verbose:
            print(newstart, (newstart + (nsamp / fs)), res[-1][1:])
        if (newstart + (nsamp + nstep) / fs) > etime:
            eotr = False
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
