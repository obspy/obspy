#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python module containing detrend methods.

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


def simple(data):
    """
    Detrend signal simply by subtracting a line through the first and last
    point of the trace

    :param data: Data to detrend, type numpy.ndarray.
    :return: Detrended data.
    """
    ndat = len(data)
    x1, x2 = data[0], data[-1]
    return data - (x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1))


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
