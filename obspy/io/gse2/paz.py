#!/usr/bin/env python
# ------------------------------------------------------------------
# Filename: paz.py
#  Purpose: Python routines for reading GSE poles and zero files
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2012 Moritz Beyreuther
# --------------------------------------------------------------------
"""
Python routines for reading GSE pole and zero (PAZ) files.

The read in PAZ information can be used with
:mod:`~obspy.signal` for instrument correction.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np

from obspy.core import AttribDict


def read_paz(paz_file):
    '''
    Read GSE PAZ / Calibration file format and returns poles, zeros and the
    seismometer_gain.

    Do not use this function in connection with the ObsPy instrument
    simulation, the A0_normalization_factor might be set wrongly. Use
    :func:`~obspy.io.gse2.paz.attach_paz` instead.

    >>> import io
    >>> f = io.StringIO(
    ... """CAL1 RJOB   LE-3D    Z  M24    PAZ 010824 0001
    ... 2
    ... -4.39823 4.48709
    ... -4.39823 -4.48709
    ... 3
    ... 0.0 0.0
    ... 0.0 0.0
    ... 0.0 0.0
    ... 0.4""")
    >>> p, z, k = read_paz(f)
    >>> print('%.4f %.4f %.4f' % (p[0].real, z[0].real, k))
    -4.3982 0.0000 0.4000
    '''
    poles = []
    zeros = []

    if isinstance(paz_file, str):
        with open(paz_file, 'rt') as fh:
            paz = fh.readlines()
    else:
        paz = paz_file.readlines()
    if paz[0][0:4] != 'CAL1':
        raise NameError("Unknown GSE PAZ format %s" % paz[0][0:4])
    if paz[0][31:34] != 'PAZ':
        raise NameError("%s type is not known" % paz[0][31:34])

    ind = 1
    npoles = int(paz[ind])
    for i in range(npoles):
        try:
            poles.append(complex(*[float(n)
                                   for n in paz[i + 1 + ind].split()]))
        except ValueError:
            poles.append(complex(float(paz[i + 1 + ind][:8]),
                                 float(paz[i + 1 + ind][8:])))

    ind += i + 2
    nzeros = int(paz[ind])
    for i in range(nzeros):
        try:
            zeros.append(complex(*[float(n)
                                   for n in paz[i + 1 + ind].split()]))
        except ValueError:
            zeros.append(complex(float(paz[i + 1 + ind][:8]),
                                 float(paz[i + 1 + ind][8:])))

    ind += i + 2
    # in the observatory this is the seismometer gain [muVolt/nm/s]
    # the A0_normalization_factor is hardcoded to 1.0
    seismometer_gain = float(paz[ind])
    return poles, zeros, seismometer_gain


def attach_paz(tr, paz_file):
    '''
    Attach tr.stats.paz AttribDict to trace from GSE2 paz_file

    This is experimental code, nevertheless it might be useful. It
    makes several assumption on the gse2 paz format which are valid for the
    geophysical observatory in Fuerstenfeldbruck but might be wrong in
    other cases.

    Attaches to a trace a paz AttribDict containing poles zeros and gain.
    The A0_normalization_factor is set to 1.0.

    :param tr: An ObsPy trace object containing the calib and gse2 calper
            attributes
    :param paz_file: path to pazfile or file pointer

    >>> from obspy.core import Trace
    >>> import io
    >>> tr = Trace(header={'calib': .094856, 'gse2': {'calper': 1}})
    >>> f = io.StringIO(
    ... """CAL1 RJOB   LE-3D    Z  M24    PAZ 010824 0001
    ... 2
    ... -4.39823 4.48709
    ... -4.39823 -4.48709
    ... 3
    ... 0.0 0.0
    ... 0.0 0.0
    ... 0.0 0.0
    ... 0.4""")
    >>> attach_paz(tr, f)
    >>> print(round(tr.stats.paz.sensitivity / 10E3) * 10E3)
    671140000.0
    '''
    poles, zeros, seismometer_gain = read_paz(paz_file)

    # remove zero at 0,0j to undo integration in GSE PAZ
    for i, zero in enumerate(list(zeros)):
        if zero == complex(0, 0j):
            zeros.pop(i)
            break
    else:
        raise Exception("Could not remove (0,0j) zero to undo GSE integration")

    # ftp://www.orfeus-eu.org/pub/software/conversion/GSE_UTI/gse2001.pdf
    # page 3
    calibration = tr.stats.calib * 2 * np.pi / tr.stats.gse2.calper

    # fill up ObsPy Poles and Zeros AttribDict
    tr.stats.paz = AttribDict()
    # convert seismometer gain from [muVolt/nm/s] to [Volt/m/s]
    tr.stats.paz.seismometer_gain = seismometer_gain * 1e3
    # convert digitizer gain [count/muVolt] to [count/Volt]
    tr.stats.paz.digitizer_gain = 1e6 / calibration
    tr.stats.paz.poles = poles
    tr.stats.paz.zeros = zeros
    tr.stats.paz.sensitivity = tr.stats.paz.digitizer_gain * \
        tr.stats.paz.seismometer_gain
    # A0_normalization_factor convention for gse2 paz in Observatory in FFB
    tr.stats.paz.gain = 1.0


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
