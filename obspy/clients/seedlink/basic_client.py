# -*- coding: utf-8 -*-
"""
SeedLink request client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
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
        self._server_url = "%s:%i" % (server, port)
        self._station_cache = None
        self._station_cache_level = None

    def _init_client(self):
        """
        Make fresh connection to seedlink server

        Should be done before any request to server, since SLClient keeps
        things like multiselect etc for subsequent requests
        """
        self._slclient = SLClient(timeout=self.timeout)

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
        >>> st = client.get_waveforms("G", "FDFM", "00", "BHZ", t, t + 5)
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        G.FDFM.00.BHZ | 20... | 20.0 Hz, ... samples

        Most servers support '?' single-character wildcard in location and
        channel code fields:

        >>> st = client.get_waveforms("G", "FDFM", "??", "B??", t, t + 5)
        >>> st = st.sort(reverse=True)
        >>> print(st)  # doctest: +ELLIPSIS
        6 Trace(s) in Stream:
        G.FDFM.10.BHZ | 20... | 20.0 Hz, ... samples
        G.FDFM.10.BHN | 20... | 20.0 Hz, ... samples
        G.FDFM.10.BHE | 20... | 20.0 Hz, ... samples
        G.FDFM.00.BHZ | 20... | 20.0 Hz, ... samples
        G.FDFM.00.BHN | 20... | 20.0 Hz, ... samples
        G.FDFM.00.BHE | 20... | 20.0 Hz, ... samples

        Depending on server capabilities, '*' multi-character wildcards might
        work in any parameter:

        >>> st = client.get_waveforms("*", "FDFM", "*", "B*", t, t + 5)
        >>> st = st.sort(reverse=True)
        >>> print(st)  # doctest: +ELLIPSIS
        6 Trace(s) in Stream:
        G.FDFM.10.BHZ | 20... | 20.0 Hz, ... samples
        G.FDFM.10.BHN | 20... | 20.0 Hz, ... samples
        G.FDFM.10.BHE | 20... | 20.0 Hz, ... samples
        G.FDFM.00.BHZ | 20... | 20.0 Hz, ... samples
        G.FDFM.00.BHN | 20... | 20.0 Hz, ... samples
        G.FDFM.00.BHE | 20... | 20.0 Hz, ... samples

        .. note::

            Support of wildcards strongly depends on the queried seedlink
            server. In general, '?' as single character wildcard seems to work
            well in location code and channel code fields for most servers.
            Usage of '*' relies on the server supporting info requests on
            station or even channel level, see :meth:`Client.get_info()`.

        :type network: str
        :param network: Network code. See note on wildcards above.
        :type station: str
        :param station: Station code. See note on wildcards above.
        :type location: str
        :param location: Location code. See note on wildcards above.
        :type channel: str
        :param channel: Channel code. See note on wildcards above.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start time of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End time of requested time window.
        """
        # need to do an info request?
        if any('*' in x for x in (network, station, location, channel)) \
                or ('?' in x for x in (network, station)):
            # need to do an info request on channel level?
            if any('*' in x for x in (location, channel)):
                info = self.get_info(network=network, station=station,
                                     location=location, channel=channel,
                                     level='channel', cache=True)
                multiselect = ["%s_%s:%s%s" % (net, sta, loc, cha)
                               for net, sta, loc, cha in info]
            # otherwise keep location/channel wildcards and do request on
            # station level only
            else:
                info = self.get_info(network=network, station=station,
                                     level='station', cache=True)
                multiselect = ["%s_%s:%s%s" % (net, sta, location, channel)
                               for net, sta in info]
            multiselect = ','.join(multiselect)
            return self._multiselect_request(multiselect, starttime, endtime)

        # if no info request is needed, we just work with the given input
        # (might have some '?' wildcards in loc/cha)
        if len(location) > 2:
            msg = ("Location code ('%s') only supports a maximum of 2 "
                   "characters.") % location
            raise ValueError(msg)
        elif len(location) == 1:
            msg = ("Single character location codes that are not an '*' are "
                   "untested.")
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
        stream.sort()
        return stream

    def get_info(self, network=None, station=None, location=None, channel=None,
                 level='station', cache=True):
        """
        Request available stations information from the seedlink server.

        Supports ``fnmatch`` wildcards, e.g. ``*`` and ``?``, in ``network``,
        ``station``, ``location`` and ``channel``.

        >>> client = Client('rtserver.ipgp.fr')
        >>> info = client.get_info(station="FDFM")
        >>> print(info)
        [('G', 'FDFM')]
        >>> info = client.get_info(station="FD?M", channel='*Z',
        ...                        level='channel')
        >>> print(info)  # doctest: +NORMALIZE_WHITESPACE
        [('G', 'FDFM', '00', 'BHZ'), ('G', 'FDFM', '00', 'HHZ'),
         ('G', 'FDFM', '00', 'HNZ'), ('G', 'FDFM', '00', 'LHZ'),
         ('G', 'FDFM', '10', 'BHZ'), ('G', 'FDFM', '10', 'HHZ'),
         ('G', 'FDFM', '10', 'LHZ')]

        Available station information is cached after the first request to the
        server, so use ``cache=False`` on subsequent requests if there is a
        need to force fetching new information from the server (should only
        concern programs running in background for a very long time).

        :type network: str
        :param network: Network code. Supports ``fnmatch`` wildcards, e.g.
            ``*`` and ``?``.
        :type station: str
        :param station: Station code. Supports ``fnmatch`` wildcards, e.g.
            ``*`` and ``?``.
        :type location: str
        :param location: Location code. Supports ``fnmatch`` wildcards, e.g.
            ``*`` and ``?``.
        :type channel: str
        :param channel: Channel code. Supports ``fnmatch`` wildcards, e.g.
            ``*`` and ``?``.
        :type cache: bool
        :param cache: Subsequent function calls are cached, use ``cache=False``
            to force fetching station metadata again from the server.
        :rtype: list
        :returns: list of 2-tuples (or 4-tuples with ``level='channel'``) with
            network/station (network/station/location/channel, respectively)
            code combinations for which data is served by the server.
        """
        if level not in ('station', 'channel'):
            msg = "Invalid option for 'level': '%s'" % str(level)
            raise ValueError(msg)
        if level == 'station' and \
                any(x is not None for x in (location, channel)):
            msg = ("location and channel options are ignored in get_info() if "
                   "level='station'.")
            warnings.warn(msg)
        # deteremine if we have a usable cache and check if it is at least the
        # requested level of detail
        if cache and self._station_cache is not None \
                and level in ('station', self._station_cache_level):
            if level == 'station':
                if self._station_cache_level == 'station':
                    info = [(net, sta) for net, sta in self._station_cache
                            if fnmatch.fnmatch(net, network or '*') and
                            fnmatch.fnmatch(sta, station or '*')]
                    return sorted(info)
                else:
                    info = [(net, sta) for net, sta, loc, cha
                            in self._station_cache
                            if fnmatch.fnmatch(net, network or '*') and
                            fnmatch.fnmatch(sta, station or '*')]
                    return sorted(set(info))
            info = [(net, sta, loc, cha) for net, sta, loc, cha in
                    self._station_cache if
                    fnmatch.fnmatch(net, network or '*') and
                    fnmatch.fnmatch(sta, station or '*') and
                    fnmatch.fnmatch(loc, location or '*') and
                    fnmatch.fnmatch(cha, channel or '*')]
            return sorted(info)

        self._init_client()
        if level == 'station':
            self._slclient.infolevel = "STATIONS"
        elif level == 'channel':
            self._slclient.infolevel = "STREAMS"
        self._slclient.verbose = 1
        self._connect()
        self._slclient.initialize()
        # self._slclient.run()
        self._slclient.run(packet_handler=self._packet_handler)
        info = self._slclient.slconn.info_string
        try:
            xml = etree.fromstring(info)
        except ValueError as e:
            msg = 'Unicode strings with encoding declaration are not supported'
            if msg not in str(e):
                raise
            parser = etree.XMLParser(encoding='utf-8')
            xml = etree.fromstring(info.encode('utf-8'), parser=parser)
        station_cache = set()
        for tag in xml.xpath('./station'):
            net = tag.attrib['network']
            sta = tag.attrib['name']
            item = (net, sta)
            if level == 'channel':
                subtags = tag.xpath('./stream')
                for subtag in subtags:
                    loc = subtag.attrib['location']
                    cha = subtag.attrib['seedname']
                    station_cache.add(item + (loc, cha))
                # If no data is in ring buffer (e.g. station outage?) then it
                # seems the seedlink server replies with no subtags for the
                # channels
                if not subtags:
                    station_cache.add(item + (None, None))
            else:
                station_cache.add(item)
        # change results to an Inventory object
        self._station_cache = station_cache
        self._station_cache_level = level
        return self.get_info(
            network=network, station=station, location=location,
            channel=channel, cache=True, level=level)

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
                print("Complete INFO:",
                      self._slclient.slconn.get_info_string())
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
