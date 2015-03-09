#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from math import pi


class Arrival(object):
    """
    Convenience class for storing parameters associated with a phase arrival.

    :ivar phase: :class:`~obspy.taup.seismic_phase.SeismicPhase` that
        generated this arrival
    :ivar distance: Actual distance in degrees
    :ivar time: Travel time in seconds
    :ivar purist_dist: Purist angular distance (great circle) in radians
    :ivar ray_param: Ray parameter in seconds per radians
    :ivar name: Phase name
    :ivar purist_name: Phase name changed for true depths
    :ivar source_depth: Source depth in kilometers
    :ivar incident_angle:
    :ivar takeoff_angle:
    :ivar pierce: pierce points
    :ivar path: path points
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
        return self.ray_param * pi / 180.0

    @property
    def purist_distance(self):
        """
        Return the purist distance in degrees.
        """
        return self.purist_dist * 180.0 / pi
