#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Routines related to time domain deconvolution

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from copy import deepcopy

import scipy
import numpy as np

from obspy.core.util import AttribDict


def deconvolve_volt_to_velocity(
        raw_data_in_volts, digital_paz, filter_low=None, filter_high=None,
        bitweight=None, dec=None, demean=True):
    """
    Deconvolve digital Poles and Zeros from raw data in Volts.
    """
    # demean
    if demean:
        raw_data_in_volts -= np.mean(raw_data_in_volts)

    dpz = digital_paz
    # invert dpz
    izpg = AttribDict()
    izpg.pole = dpz.zpg.zero
    izpg.zero = dpz.zpg.pole
    izpg.gain = 1.0 / dpz.zpg.gain
    izpg = deepcopy(izpg)

    if filter_low is not None:
        z, p, k = scipy.signal.butter(
            2, filter_low * 2 * dpz.delta, btype="highpass", analog=False,
            output="zpk")
        izpg.pole.extend(p)
        izpg.zero.extend(z)
        izpg.gain *= k
    # convert to ARMA
    b, a = scipy.signal.zpk2tf(izpg.zero, izpg.pole, izpg.gain)
    result = scipy.signal.lfilter(b, a, raw_data_in_volts)
    # alterenative: SOS
    # sos = scipy.signal.zpk2sos(izpg.zero, izpg.pole, izpg.gain)
    # result = scipy.signal.sosfilt(sos, raw_data_in_volts)
    return result


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
