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

    def _to_array(self):
        """
        Store all attributes for serialization in a structured array.
        """
        arr = np.empty(3, dtype=np.float_)
        arr[0] = self.topDepth
        arr[1] = self.botDepth
        arr[2] = self.ray_param
        return arr

    @staticmethod
    def _from_array(arr):
        """
        Create instance object from a structured array used in serialization.
        """
        depth_range = DepthRange()
        depth_range.topDepth = arr[0]
        depth_range.botDepth = arr[1]
        depth_range.ray_param = arr[2]
        return depth_range


SplitLayerInfo = namedtuple(
    'SplitLayerInfo',
    ['sMod', 'neededSplit', 'movedSample', 'ray_param']
)


class Arrival(object):
    """
    Convenience class for storing parameters associated with a phase arrival.

    :ivar phase: Phase that generated this arrival
    :vartype phase: :class:`~obspy.taup.seismic_phase.SeismicPhase`
    :ivar distance: Actual distance in degrees
    :vartype distance: float
    :ivar time: Travel time in seconds
    :vartype time: float
    :ivar purist_dist: Purist angular distance (great circle) in radians
    :vartype purist_dist: float
    :ivar ray_param: Ray parameter in seconds per radians
    :vartype ray_param: float
    :ivar name: Phase name
    :vartype name: str
    :ivar purist_name: Phase name changed for true depths
    :vartype purist_name: str
    :ivar source_depth: Source depth in kilometers
    :vartype source_depth: float
    :ivar incident_angle: Angle (in degrees) at which the ray arrives at the
        receiver
    :vartype incident_angle: float
    :ivar takeoff_angle: Angle (in degrees) at which the ray leaves the source
    :vartype takeoff_angle: float
    :ivar pierce: Points pierced by ray
    :vartype pierce: :class:`~numpy.ndarray` (dtype = :const:`~TimeDist`)
    :ivar path: Path taken by ray
    :vartype path: :class:`~numpy.ndarray` (dtype = :const:`~TimeDist`)
    """
    def __init__(self, phase, distance, time, purist_dist, ray_param,
                 ray_param_index, name, purist_name, source_depth,
                 takeoff_angle, incident_angle):
        self.phase = phase
        self.distance = distance
        self.time = time
        self.purist_dist = purist_dist
        self.ray_param = ray_param
        self.ray_param_index = ray_param_index
        self.name = name
        self.purist_name = purist_name
        self.source_depth = source_depth
        self.incident_angle = incident_angle
        self.takeoff_angle = takeoff_angle
        self.pierce = None
        self.path = None

    def __str__(self):
        return "%s phase arrival at %.3f seconds" % (self.phase.name,
                                                     self.time)

    @property
    def ray_param_sec_degree(self):
        """
        Return the ray parameter in seconds per degree.
        """
        return self.ray_param * np.pi / 180.0

    @property
    def purist_distance(self):
        """
        Return the purist distance in degrees.
        """
        return self.purist_dist * 180.0 / np.pi
