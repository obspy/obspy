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

import numpy as np


def deconvolve_volt_to_velocity(
        raw_data_in_volts, digital_paz, filter_low=None, filter_high=None,
        bitweight=None, dec=None):
    """
    Deconvolve digital Poles and Zeros from raw data in Volts.
    """
    result = np.empty_like(raw_data_in_volts, dtype=np.float64)
    return result


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
