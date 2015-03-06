#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Holds various helper classes to keep the file number manageable.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

from collections import namedtuple

import numpy as np


class SlownessModelError(Exception):
    pass


class TauModelError(Exception):
    pass


SlownessLayer = np.dtype([
    (native_str('topP'), np.float_),
    (native_str('topDepth'), np.float_),
    (native_str('botP'), np.float_),
    (native_str('botDepth'), np.float_),
])


"""
Holds the ray parameter, time and distance increments, and optionally a
depth, for a ray passing through some layer.
"""
TimeDist = np.dtype([
    (native_str('p'), np.float_),
    (native_str('time'), np.float_),
    (native_str('dist'), np.float_),
    (native_str('depth'), np.float_),
])


"""
Tracks critical points (discontinuities or reversals in slowness gradient)
within slowness and velocity models.
"""
CriticalDepth = np.dtype([
    (native_str('depth'), np.float_),
    (native_str('velLayerNum'), np.int_),
    (native_str('pLayerNum'), np.int_),
    (native_str('sLayerNum'), np.int_),
])


class DepthRange:
    """
    Convenience class for storing a depth range. It has a top and a bottom and
    can have an associated ray parameter.
    """
    def __init__(self, topDepth=None, botDepth=None, ray_param=-1):
        self.topDepth = topDepth
        self.botDepth = botDepth
        self.ray_param = ray_param


SplitLayerInfo = namedtuple(
    'SplitLayerInfo',
    ['sMod', 'neededSplit', 'movedSample', 'ray_param']
)
