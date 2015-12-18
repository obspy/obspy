# -*- coding: utf-8 -*-
"""
An ObsPy base client for uniform interfaces.

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

class SeedlinkClient(WaveformClient, EventClient):
    def __init__(self, url=None):
        self._version = '1.0'
        if url:
            self.conn = open(url)

    def get_server_version(self):
        self.conn.get_version()

    def get_waveforms(self, station, channel):
        return self.conn.fetch_mseed(station, channel)

    def get_events(self, starttime, endtime):
        return self.conn.fetch_catalog()

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future import standard_library
from future.utils import native_str

from abc import ABCMeta, abstractmethod, abstractproperty

class BaseClient(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_service_version(self):
        """Returns a semantic versioning string."""
        pass


class WaveformClient(BaseClient):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_waveforms():
        """Returns a Stream."""
        pass


class EventClient(BaseClient):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_events():
        """Returns a Catalog."""
        pass


class StationClient(BaseClient):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_stations():
        """Returns an Inventory."""
        pass
