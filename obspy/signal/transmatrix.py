# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: transmatrix.py
#  Purpose: Function for transformation matrix between 2 seismometers
#   Author: Maxime Bes de Berc
#    Email: mbesdeberc@unistra.fr
#
# Copyright (C) 2017 Maxime Bes de Berc
# --------------------------------------------------------------------
"""
Function for transformation matrix between 2 seismometers
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np

from obspy.core import Stream


def transMatrix(res_stream, mon_stream, fmin=0.125, fmax=0.5):
    """
    Function calculating a transformation matrix between a stream of 3 \
    orthogonal traces (ouput of unknown seismometer, defined as res_stream) \
    and another stream of 3 orthogonal reference traces (output of known \
    seismometer, defined as mon_stream) recording the same signal. For each \
    trace of unknown stream, it uses the linear regression method to express \
    it as a combination of the reference stream.
    Therefore, the result is 3 coeff per unknown channel, ie a 3x3 matrix:
        |X|   |a00 a01 a02|   |Xref|
        |Y| = |a10 a11 a12| x |Yref|
        |Z|   |a20 a21 a22|   |Zref|
    From this matrix, one can calculate the orientation error between \
    seismometers:
        alpha = arctan(a10/a00)
        alpha = arctan(-a01, a11)
    the gains:
        Gx = a00/(cos(alpha)*Gxref)
        Gy = a11/(cos(alpha)*Gyref)
        Gz = a22/Gzref
    More, apply this method between a stream and itself gives a direct result \
    of diaphony/orthogonility errors. In fact, the matrix becomes ideally:
        |X|   |1 0 0|   |X|
        |Y| = |0 1 0| . |Y|
        |Z|   |0 0 1|   |Z|
    Values close to 0 give addition of the orthogonality/diaphony errors of \
    the two streams.
    Finally, that supposes strong coherence between signals. It is therefore \
    necessary to filter the signals over an appropriate range (ie. \
    micro-seismic peak).

    :type res_stream: Stream object from module obspy.core.stream
    :param res_stream: Stream containing 3 traces with the signal from the \
    unknown seismometer.
    :type mon_stream: Stream object from module obspy.core.stream
    :param mon_stream: Stream containing 3 traces with the signal from the \
    known seismometer.
    :type fmin: float
    :param fmin: Minimal frequency of the bandpass filter applied.
    :type fmax: float
    :param fmax: Maximal frequency of the bandpass filter applied.
    """

    for st in (mon_stream, res_stream):

        # Verify if object is a stream
        if not isinstance(st, Stream):
            msg = "Given object is not a stream!"
            raise ValueError(msg)
        if len(st) != 3:
            msg = "Stream must strictly have 3 traces!"
            raise ValueError(msg)

        # Verify each trace of stream: same sampling rate?
        if st[0].stats.sampling_rate != st[1].stats.sampling_rate \
           or st[0].stats.sampling_rate != st[2].stats.sampling_rate \
           or st[1].stats.sampling_rate != st[2].stats.sampling_rate:
            msg = "Sampling rates are not identical between traces!"
            raise ValueError(msg)

        # Verify each trace of stream: same length?
        if st[0].stats.npts != st[1].stats.npts \
           or st[0].stats.npts != st[2].stats.npts \
           or st[1].stats.npts != st[2].stats.npts:
            msg = "Traces does not have the same length!"
            raise ValueError(msg)

        # Verify each trace of stream: same start time?
        if st[0].stats.starttime-st[1].stats.starttime >= \
           st[0].stats.sampling_rate/2 \
           or st[1].stats.starttime-st[2].stats.starttime >= \
           st[1].stats.sampling_rate/2\
           or st[0].stats.starttime-st[2].stats.starttime >= \
           st[0].stats.sampling_rate/2:
            msg = "Traces does not have the same start time!"
            raise ValueError(msg)

        # Sort stream in ENZ or 12Z
        st.sort()
        # Detrend, taper and filter stream
        st.detrend('demean')
        st.detrend('linear')
        st.taper(0.2)
        st.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True,
                  corners=8)

    # Verify streams between them: same sampling rate?
    if mon_stream[0].stats.sampling_rate != res_stream[0].stats.sampling_rate:
        msg = "Sampling rates are not identical between streams"
        raise ValueError(msg)

    # Verify streams between them: same length?
    if mon_stream[0].stats.npts != res_stream[0].stats.npts:
        msg = "Streams does not have the same length"
        raise ValueError(msg)

    # Verify streams between them: same start time?
    if mon_stream[0].stats.starttime-res_stream[0].stats.starttime >= \
       mon_stream[0].stats.sampling_rate/2:
        msg = "Stream does not have the same start time"
        raise ValueError(msg)

    # Create matrix of data with shape 3 x npts
    coeff_matrix = np.matrix([mon_stream[0].data, mon_stream[1].data,
                             mon_stream[2].data]).transpose()

    # Create empty matrix
    matrix = np.array([])
    # Feed it with 9 coefficients calculated with linear regression
    for i in range(3):
        matrix = np.append(matrix, np.linalg.lstsq(coeff_matrix,
                                                   res_stream[i].data)[0])
    # Reshape correctly the final matrix
    matrix = np.reshape(matrix, (3, 3))

    return matrix
