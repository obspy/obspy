#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from math import pi


class Arrival(object):
    """
    Convenience class for storing the parameters associated with a phase
    arrival.
    """
    def __init__(self, phase, time, dist, ray_param, ray_param_index,
                 name, purist_name, source_depth, takeoff_angle,
                 incident_angle):
        # phase that generated this arrival
        self.phase = phase
        # travel time in seconds
        self.time = time
        # angular distance (great circle) in radians
        self.dist = dist
        # ray parameter in seconds per radians
        self.ray_param = ray_param
        self.ray_param_index = ray_param_index
        # phase name
        self.name = name
        # phase name changed for true depths
        self.purist_name = purist_name
        # source depth in kilometers
        self.source_depth = source_depth
        self.incident_angle = incident_angle
        self.takeoff_angle = takeoff_angle
        # pierce and path points
        self.pierce = None
        self.path = None

    def __str__(self):
        return "%s phase arrival at %.3f seconds" % (self.phase.name,
                                                     self.time)

    @property
    def ray_param_sec_degree(self):
        """
        Returns the ray parameter in seconds per degree.
        """
        return self.ray_param * pi / 180.0

    @property
    def purist_distance(self):
        return self.dist * 180.0 / pi

    def get_pierce(self):
        """
        Returns pierce points as TimeDist objects.
        """
        if not self.pierce:
            self.pierce == self.phase.calc_pierce(self).get_pierce()
        return self.pierce

    def get_path(self):
        """
        Returns pierce points as TimeDist objects.
        """
        if not self.path:
            self.path == self.phase.calc_path(self).get_path()
        return self.path

    def get_modulo_dist_deg(self):
        """
        Returns distance in degrees from 0 - 180. Note this may not be the
        actual distance travelled.
        """
        moduloDist = ((180.0 / pi) * self.dist) % 360.0
        if moduloDist > 180:
            moduloDist = 360 - moduloDist
        return moduloDist
