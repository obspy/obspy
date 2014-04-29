#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Domain definitions for the the download helpers.

Subclass if you need more complex custom domains.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod


class Domain(with_metaclass(ABCMeta)):
    """
    Abstract base class defining a domain. Subclass to define a custom domain.

    Each subclass has to implement the
    :meth:`~obspy.fdsn.download_helpers.Domain.get_query_parameters` method and
    optionally the
    :meth:`~obspy.fdsn.download_helpers.Domain.in_in_domain` method.

    The :meth:`~obspy.fdsn.download_helpers.Domain.get_query_parameters` method
    must return the query parameters to download as much data as required. The
    :meth:`~obspy.fdsn.download_helpers.Domain.in_in_domain` can later be
    used to refine the domain after the data has been downloaded.
    """
    @abstractmethod
    def get_query_parameters(self):
        """
        Return the domain specific query parameters for the
        :meth:`~obspy.fdsn.client.Client.get_stations' method. Possibilities
        are ``minlatitude``, ``maxlatitude``, ``minlongitude``, and
        ``maxlongitude`` for rectangular queries or ``latitude``,
        ``longitude``, ``minradius``, and ``maxradius`` for circular queries.
        """
        pass

    def is_in_domain(self, latitude, longitude):
        """
        Returns True/False depending on the point being in the domain. If
        not implemented no further restrictions will be applied after the
        data has been downloaded.
        """
        raise NotImplementedError


class RectangularDomain(Domain):
    """
    A rectangular domain defined by latitude and longitude bounds.
    """
    def __init__(self, minlatitude, maxlatitude, minlongitude,
                 maxlongitude):
        self.minlatitude = minlatitude
        self.maxlatitude = maxlatitude
        self.minlongitude = minlongitude
        self.maxlongitude = maxlongitude

    def get_query_parameters(self):
        return {
            "minlatitude": self.minlatitude,
            "maxlatitude": self.maxlatitude,
            "minlongitude": self.minlongitude,
            "maxlongitude": self.maxlongitude}


class CircularDomain(Domain):
    """
    A circular domain defined by a center point and minimum and maximum
    radius from that point.
    """
    def __init__(self, latitude, longitude, minradius, maxradius):
        self.latitude = latitude
        self.longitude = longitude
        self.minradius = minradius
        self.maxradius = maxradius

    def get_query_parameters(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "minradius": self.minradius,
            "maxradius": self.maxradius}


class GlobalDomain(Domain):
    """
    Domain spanning the whole globe.
    """
    def get_query_parameters(self):
        return {}
