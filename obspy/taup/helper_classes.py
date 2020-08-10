# -*- coding: utf-8 -*-
"""
Holds various helper classes to keep the file number manageable.
"""
from collections import namedtuple

import numpy as np


class SlownessModelError(Exception):
    pass


class TauModelError(Exception):
    pass


SlownessLayer = np.dtype([
    ('top_p', np.float_),
    ('top_depth', np.float_),
    ('bot_p', np.float_),
    ('bot_depth', np.float_),
])


"""
Holds the ray parameter, time and distance increments, and optionally a
depth, for a ray passing through some layer.
"""
TimeDist = np.dtype([
    ('p', np.float_),
    ('time', np.float_),
    ('dist', np.float_),
    ('depth', np.float_),
])


"""
Holds the ray parameter, time and distance increments, and optionally a
depth, latitude and longitude for a ray passing through some layer.
"""
TimeDistGeo = np.dtype([
    ('p', np.float_),
    ('time', np.float_),
    ('dist', np.float_),
    ('depth', np.float_),
    ('lat', np.float_),
    ('lon', np.float_)
])


"""
Tracks critical points (discontinuities or reversals in slowness gradient)
within slowness and velocity models.
"""
CriticalDepth = np.dtype([
    ('depth', np.float_),
    ('vel_layer_num', np.int_),
    ('p_layer_num', np.int_),
    ('s_layer_num', np.int_),
])


class DepthRange:
    """
    Convenience class for storing a depth range. It has a top and a bottom and
    can have an associated ray parameter.
    """
    def __init__(self, top_depth=None, bot_depth=None, ray_param=-1):
        self.top_depth = top_depth
        self.bot_depth = bot_depth
        self.ray_param = ray_param

    def _to_array(self):
        """
        Store all attributes for serialization in a structured array.
        """
        arr = np.empty(3, dtype=np.float_)
        arr[0] = self.top_depth
        arr[1] = self.bot_depth
        arr[2] = self.ray_param
        return arr

    @staticmethod
    def _from_array(arr):
        """
        Create instance object from a structured array used in serialization.
        """
        depth_range = DepthRange()
        depth_range.top_depth = arr[0]
        depth_range.bot_depth = arr[1]
        depth_range.ray_param = arr[2]
        return depth_range


SplitLayerInfo = namedtuple(
    'SplitLayerInfo',
    ['s_mod', 'needed_split', 'moved_sample', 'ray_param']
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
                 receiver_depth, takeoff_angle=None, incident_angle=None):
        if np.isnan(time):
            raise ValueError('Time cannot be NaN')
        if ray_param_index < 0:
            raise ValueError(
                'ray_param_index cannot be negative: %d' % (ray_param_index, ))

        self.phase = phase
        self.distance = distance
        self.time = time
        self.purist_dist = purist_dist
        self.ray_param = ray_param
        self.ray_param_index = ray_param_index
        self.name = name
        self.purist_name = purist_name
        self.source_depth = source_depth
        self.receiver_depth = receiver_depth
        if takeoff_angle is None:
            self.takeoff_angle = phase.calc_takeoff_angle(ray_param)
        else:
            self.takeoff_angle = takeoff_angle
        if incident_angle is None:
            self.incident_angle = phase.calc_incident_angle(ray_param)
        else:
            self.incident_angle = incident_angle
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
