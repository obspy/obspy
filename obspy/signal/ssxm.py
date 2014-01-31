#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: ssxm.py
#  Purpose: RSAM, RSEM, SSAM and SSEM calculations
#   Author: Thomas Lecocq & Corentin Caudron
#    Email: thomas.lecocq@seismology.be
#
# Copyright (C) 2012-2014 T. Lecocq, C. Caudron
#-------------------------------------------------------------------
"""
Real-time Amplitude/Energy Measurement and
Spectral Seismic Amplitude/Energy Measurement

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import pandas as pd
import numpy as np
from obspy.signal import bandpass
from scipy.stats import scoreatpercentile


def ssxm(data, fs, id, starttime, rule='30S', bands=None, corners=4,
         zerophase=False, percentiles=None):
    """
    SSxM of a signal.

    Computes the SSxM of the given data which can be windowed or not.
    The sonogram is determined by the power in half octave bands of the given
    data.

    If data are windowed the analytic signal and the envelope of each window
    is returned.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to make envelope of.
    :param fs: Sampling frequency in Hz.
    :param id: ID of the trace.
    :param starttime: The starttime of the trace :class:`~datetime.datetime`.
    :param rule: Windowing rule, following pandas' conventions.
    :param bands: :class:`~list` of tuples containing upper and
        lower frequencies for the bandpass. If None, only RSAM and RSEM
        are returned.
    :param corners: Filter corners. Note: This is twice the value of PITSA's
        filter sections.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :param percentiles: :class:`~list` containing percentiles to compute in
        addition to.
    :return: SSxM data in a :class:`~pandas.DataFrame`.
    """
    # Convert input data to a pandas.Series object
    npts = len(data)
    delta = 1./fs
    t = pd.date_range(starttime, periods=npts,
                      freq="%ims" % (delta * 1000))
    t = pd.Index(t, name='timestamp')
    s = pd.Series(data=data, index=t, name=id, dtype=data.dtype)
    del npts, delta

    bands.insert(0, [0, 0])
    first = True

    for band in bands:
        if band != [0, 0]:
            tmp = s.copy()
            tmp.data = bandpass(tmp, band[0], band[1], fs, corners=corners,
                                zerophase=zerophase)
        else:
            tmp = s

        df = pd.DataFrame(tmp.resample(rule, how=ssam), columns=['mean'])
        df['std'] = tmp.resample(rule, how=ssem)
        df['low'] = band[0]
        df['high'] = band[1]
        if percentiles:
            for perc in percentiles:
                P = Percentile(percentile=perc)
                df["p%i" % (perc)] = tmp.resample(rule,
                                                  how=P.scoreatpercentile)
        if first:
            data = df
            first = False
        else:
            data = pd.concat((data, df))
        del tmp
    del s
    data.set_index(['low', 'high'], inplace=True, append=True)
    return data


def ssam(d):
    """
    Computes the SSAM of an given array.
    """
    return np.mean(np.abs(d))


def ssem(d):
    """
    Computes the SSEM of an given array.
    """
    return np.std(d)


class Percentile():
    def __init__(self, percentile):
        self.percentile = percentile

    def scoreatpercentile(self, a):
        return scoreatpercentile(np.abs(a), self.percentile)
