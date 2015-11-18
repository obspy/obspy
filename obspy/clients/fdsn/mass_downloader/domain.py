#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Domain definitions for the download helpers.

Subclass the :class:`~obspy.clients.fdsn.mass_downloader.domain.Domain` class
to define custom and potentially more complex domains. See its documentation
for an example.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014-2015
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod


class Domain(with_metaclass(ABCMeta)):
    """
    Abstract base class defining a domain - subclass it to define a new domain.

    Each subclass must implement the :meth:`~.get_query_parameters`
    method and optionally the :meth:`~.is_in_domain` method which enables
    the construction of arbitrarily complex domains. The
    :meth:`~.get_query_parameters` method must return the query parameters to
    download as much data as required. The :meth:`~.is_in_domain` can later be
    used to refine the domain after the data has been downloaded.


    It can be thought of as a boolean operation - first the rough domain is
    specified including all the possible points, then some points are
    removed again. This is illustrated with an example domain representing
    Germany utilizing the external packages
    `shapely <http://toblerity.org/shapely/>`_ and
    `fiona <http://toblerity.org/fiona/>`_. Shapefiles can be found online
    on many websites, for example on
    `http://www.gadm.org/ <http://www.gadm.org/>`_. The example works by
    first extracting the bounds of the country to formulate the FDSN query
    and then removing points outside of the exact shape.


    .. code-block:: python

        import fiona
        import shapely.geometry

        from obspy.clients.fdsn.mass_downloader import Domain


        class Germany(Domain):
            def __init__(self):
                Domain.__init__(self)

                fiona_collection = fiona.open("./DEU_adm/DEU_adm0.shp")
                geometry = fiona_collection.next()["geometry"]

                self.shape = shapely.geometry.asShape(geometry)
                self.b = fiona_collection.bounds

            def get_query_parameters(self):
                return {"minlatitude": self.b[1],
                        "minlongitude": self.b[0],
                        "maxlatitude": self.b[3],
                        "maxlongitude": self.b[2]}

            def is_in_domain(self, latitude, longitude):
                if self.shape.contains(shapely.geometry.Point(longitude,
                                                              latitude)):
                    return True
                return False


    This is further illustrated by the following image. The green rectangle
    denotes the original FDSN query which returns the blue points. In the
    second step the red points are discarded leaving only points (stations)
    within Germany.

    .. figure:: /_images/expensive_plots/mass_downloader_domain.png
        :align: center

    """
    @abstractmethod
    def get_query_parameters(self):
        """
        Return the domain specific query parameters for the
        :meth:`~obspy.clients.fdsn.client.Client.get_stations` method as a
        dictionary. Possibilities keys for rectangular queries are

        * ``minlatitude``
        * ``maxlatitude``
        * ``minlongitude``
        * ``maxlongitude``

        For circular queries:

        * ``latitude``
        * ``longitude``
        * ``minradius``
        * ``maxradius``
        """
        pass

    def is_in_domain(self, latitude, longitude):
        """
        Returns True/False depending on the point being in the domain.

        If not implemented no further restrictions will be applied after the
        data has been downloaded.
        """
        raise NotImplementedError


class RectangularDomain(Domain):
    """
    A rectangular domain defined by latitude and longitude bounds.

    >>> domain = RectangularDomain(minlatitude=30, maxlatitude=50,
    ...                            minlongitude=5, maxlongitude=35)

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
    radius from that point in degrees.

    >>> domain = CircularDomain(latitude=37.52, longitude=143.04,
    ...                         minradius=70.0, maxradius=90.0)

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

    >>> domain = GlobalDomain()
    """
    def get_query_parameters(self):
        return {}


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
