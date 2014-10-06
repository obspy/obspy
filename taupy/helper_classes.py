#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Holds various helper classes to keep the file number manageable.


class SlownessModelError(Exception):
    pass


class TauModelError(Exception):
    pass


class TimeDist:
    """Holds the ray parameter, time and distance increments, and optionally a
    depth, for a ray passing through some layer.
    Note it is 'cloneable' in Java, that just means you're allowed to make a deep copy."""
    def __init__(self, p=0, time=0, distRadian=0, depth=0):
        # Careful: p must remain first element because of how class is called
        # e.g. in SlownessModel.approxDistance!
        self.p = p
        self.depth = depth
        self.time = time
        self.distRadian = distRadian

    def add(self, td):
        self.time += td.time
        self.distRadian += td.distRadian


class CriticalDepth:
    """Utility class to keep track of critical points (discontinuities or reversals
    in slowness gradient) within slowness and velocity models."""
    def __init__(self, depth, velLayerNum, pLayerNum, sLayerNum):
        self.depth = depth
        self.velLayerNum = velLayerNum
        self.pLayerNum = pLayerNum
        self.sLayerNum = sLayerNum


class DepthRange:
    """Convenience class for storing a depth range. It has a top and a bottom and
    can have an associated ray parameter."""
    def __init__(self, topDepth=None, botDepth=None, rayParam=-1):
        self.topDepth = topDepth
        self.botDepth = botDepth
        self.rayParam = rayParam


class SplitLayerInfo:
    def __init__(self, sMod, neededSplit, movedSample, rayParam):
        self.sMod = sMod
        self.neededSplit = neededSplit
        self.movedSample = movedSample
        self.rayParam = rayParam