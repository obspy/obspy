#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download helpers.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from collections import namedtuple
from obspy.fdsn.header import URL_MAPPINGS


domain = namedtuple("domain", [
    "min_latitude",
    "max_latitude",
    "min_longitude",
    "max_longitude",
    # For ciruclar requests.
    "latitude",
    "longitude",
    "mi_nradius",
    "max_radius"])


class Domain(object):
    def get_query_parameters(self):
        raise NotImplementedError

    def is_in_domain(self, latitude, longitude):
        return None


class RectangularDomain(Domain):
    def __init__(self, min_latitude, max_latitude, min_longitude,
                 max_longitude):
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude

    def get_query_parameters(self):
        return domain(
            self.min_latitude,
            self.max_latitude,
            self.min_longitude,
            self.max_longitude,
            None,
            None,
            None,
            None)


class CircularDomain(Domain):
    def __init__(self, latitude, longitude, min_radius, max_radius):
        self.latitude = latitude
        self.longitude = longitude
        self.min_radius = min_radius
        self.max_radius = max_radius

    def get_query_parameters(self):
        return domain(
            None, None, None, None,
            self.latitude,
            self.longitude,
            self.min_radius,
            self.max_radius)


class GlobalDomain(Domain):
    def get_query_parameters(self):
        return domain(None, None, None, None, None, None, None, None)
