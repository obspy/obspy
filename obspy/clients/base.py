# -*- coding: utf-8 -*-
"""
Base classes for uniform Client interfaces.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)


Enforce common interfaces for ObsPy client classes using Abstract Base Classes.
Common interfaces provide the ability to write applications that employ any
Client, regardless of its origin, and to expect to get a Stream from a
get_waveforms method, a Catalog from a get_events method, and an Inventory from
a get_stations method. This encourages Client writers to connect their data
sources to Stream, Inventory, and Catalog types, and encourage users to rely on
them in their applications.  Three base classes are provided: one for clients
that return waveforms, one for those that return events, and one for those that
return stations.  Each inherits from a common base class, which contains
methods common to all.

Individual client classes inherit from one or more of WaveformClient,
EventClient, and StationClient, and re-program the get_waveforms, get_events,
and/or get_stations methods, like in the example below.


.. rubric:: Example

class MyNewClient(WaveformClient, StationClient):
    def __init__(self, url=None):
        self._version = '1.0'
        if url:
            self.conn = open(url)

    def get_server_version(self):
        self.conn.get_version()

    def get_waveforms(self, network, station, location, channel, starttime,
                      endtime):
        return self.conn.fetch_mseed(network, station, location, channel,
                                     starttime, endtime)

    def get_stations(self, network, station, location, channel, starttime,
                     endtime):
        return self.conn.fetch_inventory(network, station, location, channel,
                                         starttime, endtime)

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod


class ClientException(Exception):
    """Base exception for Client classes."""
    pass


class BaseClient(with_metaclass(ABCMeta, object)):
    """
    Base class for common methods.

    """
    @abstractmethod
    def get_service_version(self):
        """Return a semantic version number as a string."""
        pass


class WaveformClient(with_metaclass(ABCMeta, BaseClient)):
    """
    Base class for Clients supporting Stream objects.

    """
    @abstractmethod
    def get_waveforms(self, *args, **kwargs):
        """
        Returns a Stream.

        Keyword arguments are passed to the underlying concrete class.

        :type network: str
        :param network: Select one or more network codes. Can be SEED network
            codes or data center defined codes. Multiple codes are
            comma-separated. Wildcards are allowed.
        :type station: str
        :param station: Select one or more SEED station codes. Multiple codes
            are comma-separated. Wildcards are allowed.
        :type location: str
        :param location: Select one or more SEED location identifiers. Multiple
            identifiers are comma-separated. Wildcards are allowed.
        :type channel: str
        :param channel: Select one or more SEED channel codes. Multiple codes
            are comma-separated.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Limit results to time series samples on or after the
            specified start time
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Limit results to time series samples on or before the
            specified end time

        """
        pass


class EventClient(with_metaclass(ABCMeta, BaseClient)):
    """
    Base class for Clients supporting Catalog objects.

    """
    @abstractmethod
    def get_events(self, *args, **kwargs):
        """
        Returns a Catalog.

        Keyword arguments are passed to the underlying concrete class.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param starttime: Limit to events on or after the specified start time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: Limit to events on or before the specified end time.
        :type minlatitude: float, optional
        :param minlatitude: Limit to events with a latitude larger than the
            specified minimum.
        :type maxlatitude: float, optional
        :param maxlatitude: Limit to events with a latitude smaller than the
            specified maximum.
        :type minlongitude: float, optional
        :param minlongitude: Limit to events with a longitude larger than the
            specified minimum.
        :type maxlongitude: float, optional
        :param maxlongitude: Limit to events with a longitude smaller than the
            specified maximum.
        :type latitude: float, optional
        :param latitude: Specify the latitude to be used for a radius search.
        :type longitude: float, optional
        :param longitude: Specify the longitude to the used for a radius
            search.
        :type minradius: float, optional
        :param minradius: Limit to events within the specified minimum number
            of degrees from the geographic point defined by the latitude and
            longitude parameters.
        :type maxradius: float, optional
        :param maxradius: Limit to events within the specified maximum number
            of degrees from the geographic point defined by the latitude and
            longitude parameters.
        :type mindepth: float, optional
        :param mindepth: Limit to events with depth more than the specified
            minimum.
        :type maxdepth: float, optional
        :param maxdepth: Limit to events with depth less than the specified
            maximum.
        :type minmagnitude: float, optional
        :param minmagnitude: Limit to events with a magnitude larger than the
            specified minimum.
        :type maxmagnitude: float, optional
        :param maxmagnitude: Limit to events with a magnitude smaller than the
            specified maximum.
        :type magnitudetype: str, optional
        :param magnitudetype: Specify a magnitude type to use for testing the
            minimum and maximum limits.

        """
        pass


class StationClient(with_metaclass(ABCMeta, BaseClient)):
    """
    Base class for Clients supporting Inventory objects.

    """
    @abstractmethod
    def get_stations(self, *args, **kwargs):
        """
        Returns an Inventory.

        Keyword arguments are passed to the underlying concrete class.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Limit to metadata epochs starting on or after the
            specified start time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Limit to metadata epochs ending on or before the
            specified end time.
        :type startbefore: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param startbefore: Limit to metadata epochs starting before specified
            time.
        :type startafter: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param startafter: Limit to metadata epochs starting after specified
            time.
        :type endbefore: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endbefore: Limit to metadata epochs ending before specified
            time.
        :type endafter: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endafter: Limit to metadata epochs ending after specified time.
        :type network: str
        :param network: Select one or more network codes. Can be SEED network
            codes or data center defined codes. Multiple codes are
            comma-separated.
        :type station: str
        :param station: Select one or more SEED station codes. Multiple codes
            are comma-separated.
        :type location: str
        :param location: Select one or more SEED location identifiers. Multiple
            identifiers are comma-separated. As a special case ``“--“`` (two
            dashes) will be translated to a string of two space characters to
            match blank location IDs.
        :type channel: str
        :param channel: Select one or more SEED channel codes. Multiple codes
            are comma-separated.
        :type minlatitude: float
        :param minlatitude: Limit to stations with a latitude larger than the
            specified minimum.
        :type maxlatitude: float
        :param maxlatitude: Limit to stations with a latitude smaller than the
            specified maximum.
        :type minlongitude: float
        :param minlongitude: Limit to stations with a longitude larger than the
            specified minimum.
        :type maxlongitude: float
        :param maxlongitude: Limit to stations with a longitude smaller than
            the specified maximum.
        :type latitude: float
        :param latitude: Specify the latitude to be used for a radius search.
        :type longitude: float
        :param longitude: Specify the longitude to the used for a radius
            search.
        :type minradius: float
        :param minradius: Limit results to stations within the specified
            minimum number of degrees from the geographic point defined by the
            latitude and longitude parameters.
        :type maxradius: float
        :param maxradius: Limit results to stations within the specified
            maximum number of degrees from the geographic point defined by the
            latitude and longitude parameters.

        """
        pass
