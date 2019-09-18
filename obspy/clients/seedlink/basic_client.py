# -*- coding: utf-8 -*-
"""
SeedLink request client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import fnmatch
import warnings

from lxml import etree

from obspy import Stream
from .slclient import SLClient, SLPacket
from .client.seedlinkconnection import SeedLinkConnection


class Client(object):
    """
    SeedLink request client.

    This client is intended for requests of specific, finite time windows.
    To work with continuous realtime data streams please see
    :class:`~obspy.clients.seedlink.slclient.SLClient` and
    :class:`~obspy.clients.seedlink.easyseedlink.EasySeedLinkClient`.

    :type server: str
    :param server: Server name or IP address to connect to (e.g.
        "localhost", "rtserver.ipgp.fr")
    :type port: int
    :param port: Port at which the seedlink server is operating (default is
        `18000`).
    :type timeout: float
    :param timeout: Network timeout for low-level network connection in
        seconds.
    :type debug: bool
    :param debug: Switches on debugging output.
    """
    def __init__(self, server, port=18000, timeout=20, debug=False):
        """
        Initializes the SeedLink request client.
        """
        self.timeout = timeout
        self.debug = debug
        self.loglevel = debug and "DEBUG" or "CRITICAL"
        self._server_url = "%s:%i" % (server, port)
        self._station_cache = None
        self._station_cache_level = None

    def _init_client(self):
        """
        Make fresh connection to seedlink server

        Should be done before any request to server, since SLClient keeps
        things like multiselect etc for subsequent requests
        """
        self._slclient = SLClient(loglevel=self.loglevel, timeout=self.timeout)

    def _connect(self):
        """
        Open new connection to seedlink server.
        """
        self._slclient.slconn = SeedLinkConnection(timeout=self.timeout)
        self._slclient.slconn.set_sl_address(self._server_url)
        self._slclient.slconn.netto = self.timeout

    def get_waveforms(self, network, station, location, channel, starttime,
                      endtime):
        """
        Request waveform data from the seedlink server.

        >>> from obspy import UTCDateTime
        >>> client = Client('rtserver.ipgp.fr')
        >>> t = UTCDateTime() - 1500
        >>> st = client.get_waveforms("G", "FDF", "00", "BHZ", t, t + 5)
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        G.FDF.00.BHZ | 20... | 20.0 Hz, ... samples

        Most servers support '?' single-character wildcard in location and
        channel code fields:

        >>> st = client.get_waveforms("G", "FDF", "??", "B??", t, t + 5)
        >>> st = st.sort(reverse=True)
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        G.FDF.00.BHZ | 20... | 20.0 Hz, ... samples
        G.FDF.00.BHN | 20... | 20.0 Hz, ... samples
        G.FDF.00.BHE | 20... | 20.0 Hz, ... samples

        .. note::

            Support of wildcards strongly depends on the queried seedlink
            server. In general, '?' as single character wildcard seems to work
            well in location code and channel code fields for most servers.
            Usage of '*' for multiple characters in location and channel code
            field is not supported. No wildcards are supported in
            network and station code fields by ObsPy.

        :type network: str
        :param network: Network code. No wildcards supported by ObsPy.
        :type station: str
        :param station: Station code. No wildcards supported by ObsPy.
        :type location: str
        :param location: Location code. See note on wildcards above.
        :type channel: str
        :param channel: Channel code. See note on wildcards above.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start time of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End time of requested time window.
        """
        if len(location) > 2:
            msg = ("Location code ('%s') only supports a maximum of 2 "
                   "characters.") % location
            raise ValueError(msg)
        elif len(location) == 1:
            msg = "Single character location codes are untested."
            warnings.warn(msg)
        if location:
            loccha = "%2s%3s" % (location, channel)
        else:
            loccha = channel
        seedlink_id = "%s_%s:%s" % (network, station, loccha)
        return self._multiselect_request(seedlink_id, starttime, endtime)

    def _multiselect_request(self, multiselect, starttime, endtime):
        """
        Make a multiselect request to underlying seedlink client

        Multiselect string is one or more comma separated
        network/station/location/channel combinations as defined by seedlink
        standard, e.g.
        "NETWORK_STATION:LOCATIONCHANNEL,NETWORK_STATION:LOCATIONCHANNEL"
        where location+channel may contain '?' characters but should be exactly
        5 characters long.

        :rtype: :class:`~obspy.core.stream.Stream`
        """
        self._init_client()
        self._slclient.multiselect = multiselect
        self._slclient.begin_time = starttime
        self._slclient.end_time = endtime
        self._connect()
        self._slclient.initialize()
        self.stream = Stream()
        self._slclient.run(packet_handler=self._packet_handler)
        stream = self.stream
        stream.trim(starttime, endtime)
        self.stream = None
        return stream

    def get_stations(self, network=None, station=None, cache=True):
        """
        Request available stations from the seedlink server.

        Supports ``fnmatch`` wildcards, e.g. ``*`` and ``?`` in ``network`` and
        ``station``.

        >>> client = Client('rtserver.ipgp.fr')
        >>> netsta = client.get_stations(station="FDF")
        >>> print(netsta)
        [('G', 'FDF')]

        Available station information is cached after the first request to the
        server, so use ``cache=False`` on subsequent requests if there is a
        need to force fetching new information from the server (should only
        concern programs running in background for a very long time).

        :type network: str
        :param network: Network code. No wildcards supported by ObsPy.
        :type station: str
        :param station: Station code. No wildcards supported by ObsPy.
        :type cache: bool
        :param cache: Subsequent function calls are cached, use ``cache=False``
            to force fetching station metadata again from the server.
        :rtype: list
        :returns: list of 2-tuples with network/station code combinations for
            which data is served by the server.
        """
        if cache and self._station_cache is not None:
            stations = [(net, sta) for net, sta in self._station_cache
                        if fnmatch.fnmatch(net, network or '*') and
                        fnmatch.fnmatch(sta, station or '*')]
            return sorted(stations)

        self._init_client()
        self._slclient.infolevel = "STATIONS"
        self._slclient.verbose = 1
        self._connect()
        self._slclient.initialize()
        # self._slclient.run()
        self._slclient.run(packet_handler=self._packet_handler)
        info = self._slclient.slconn.info_string
        xml = etree.fromstring(info)
        station_cache = set()
        for tag in xml.xpath('./station'):
            net = tag.attrib['network']
            sta = tag.attrib['name']
            station_cache.add((net, sta))
        self._station_cache = station_cache
        return self.get_stations(network=network, station=station, cache=True)

    def _packet_handler(self, count, slpack):
        """
        Custom packet handler that accumulates all waveform packets in a
        stream.
        """
        # check if not a complete packet
        if slpack is None or (slpack == SLPacket.SLNOPACKET) or \
                (slpack == SLPacket.SLERROR):
            return False

        # get basic packet info
        type_ = slpack.get_type()
        if self.debug:
            print(type_)

        # process INFO packets here
        if type_ == SLPacket.TYPE_SLINF:
            if self.debug:
                print(SLPacket.TYPE_SLINF)
            return False
        elif type_ == SLPacket.TYPE_SLINFT:
            if self.debug:
                print("Complete INFO:" + self.slconn.get_info_string())
            return False

        # process packet data
        trace = slpack.get_trace()
        if trace is None:
            if self.debug:
                print("Blockette contains no trace")
            return False

        # new samples add to the main stream which is then trimmed
        self.stream += trace
        self.stream.merge(-1)
        return False


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
