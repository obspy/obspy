#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Holds various helper classes to keep the file number manageable.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np


class SlownessModelError(Exception):
    pass


class TauModelError(Exception):
    pass


class TimeDist:
    """
    Holds the ray parameter, time and distance increments, and optionally a
    depth, for a ray passing through some layer.
    Note it is 'cloneable' in Java, that just means you're allowed to make a
    deep copy.
    """
    def __init__(self, p=0, time=0, distRadian=0, depth=0):
        # FIXME: Remove try/except once code correctly uses NumPy.

        # Careful: p must remain first element because of how class is called
        # e.g. in SlownessModel.approxDistance!
        self.p = p
        if isinstance(depth, np.ndarray):
            try:
                self.depth = depth[0]
            except IndexError:
                self.depth = depth[()]
        else:
            self.depth = depth
        if isinstance(time, np.ndarray):
            try:
                self.time = time[0]
            except IndexError:
                self.time = time[()]
        else:
                self.time = time
        self.distRadian = distRadian

    def add(self, td):
        self.time += td.time
        self.distRadian += td.distRadian


class CriticalDepth:
    """
    Utility class to keep track of critical points (discontinuities or
    reversals in slowness gradient) within slowness and velocity models.
    """
    def __init__(self, depth, velLayerNum, pLayerNum, sLayerNum):
        self.depth = depth
        self.velLayerNum = velLayerNum
        self.pLayerNum = pLayerNum
        self.sLayerNum = sLayerNum


class DepthRange:
    """
    Convenience class for storing a depth range. It has a top and a bottom and
    can have an associated ray parameter.
    """
    def __init__(self, topDepth=None, botDepth=None, ray_param=-1):
        self.topDepth = topDepth
        self.botDepth = botDepth
        self.ray_param = ray_param


class SplitLayerInfo:
    def __init__(self, sMod, neededSplit, movedSample, ray_param):
        self.sMod = sMod
        self.neededSplit = neededSplit
        self.movedSample = movedSample
        self.ray_param = ray_param
