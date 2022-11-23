#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Non-geographical restrictions and constraints for the mass downloader.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014-2015
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import collections.abc

import obspy


class Restrictions(object):
    """
    Class storing non-domain restrictions for a query. This is best explained
    with two examples. See the list below for a more detailed explanation
    of the parameters. The first set of restrictions is useful for event
    based earthquake set queries.

    >>> import obspy
    >>> restrictions = Restrictions(
    ...     # Get data from 5 minutes before the event to one hour after the
    ...     # event.
    ...     starttime=obspy.UTCDateTime(2012, 1, 1),
    ...     endtime=obspy.UTCDateTime(2012, 1, 2),
    ...     # You might not want to deal with gaps in the data.
    ...     reject_channels_with_gaps=True,
    ...     # And you might only want waveforms that have data for at least
    ...     # 95 % of the requested time span.
    ...     minimum_length=0.95,
    ...     # No two stations should be closer than 10 km to each other.
    ...     minimum_interstation_distance_in_m=10E3,
    ...     # Only HH or BH channels. If a station has HH channels,
    ...     # those will be downloaded, otherwise the BH. Nothing will be
    ...     # downloaded if it has neither.
    ...     channel_priorities=["HH[ZNE]", "BH[ZNE]"],
    ...     # Location codes are arbitrary and there is no rule as to which
    ...     # location is best.
    ...     location_priorities=["", "00", "10"])


    And the restrictions for downloading a noise data set might look similar to
    the following:

    >>> import obspy
    >>> restrictions = Restrictions(
    ...     # Get data for a whole year.
    ...     starttime=obspy.UTCDateTime(2012, 1, 1),
    ...     endtime=obspy.UTCDateTime(2013, 1, 1),
    ...     # Chunk it to have one file per day.
    ...     chunklength_in_sec=86400,
    ...     # Considering the enormous amount of data associated with
    ...     # continuous requests, you might want to limit the data based on
    ...     # SEED identifiers. If the location code is specified, the
    ...     # location priority list is not used; the same is true for the
    ...     # channel argument and priority list.
    ...     network="BW", station="A*", location="", channel="BH*",
    ...     # The typical use case for such a data set are noise correlations
    ...     # where gaps are dealt with at a later stage.
    ...     reject_channels_with_gaps=False,
    ...     # Same is true with the minimum length. Any data during a day
    ...     # might be useful.
    ...     minimum_length=0.0,
    ...     # Sanitize makes sure that each MiniSEED file also has an
    ...     # associated StationXML file, otherwise the MiniSEED files will
    ...     # be deleted afterwards. This is not desirable for large noise
    ...     # data sets.
    ...     sanitize=False,
    ...     # Guard against the same station having different names.
    ...     minimum_interstation_distance_in_m=100.0)

    The ``network``, ``station``, ``location``, and ``channel`` codes are
    directly passed to the `station` service of each fdsn-ws implementation
    and can thus take comma separated string lists as arguments, i.e.

    .. code-block:: python

        restrictions = Restrictions(
            ...
            network="BW,G?", station="A*,B*",
            ...
            )

    Not all fdsn-ws implementations support the direct exclusion of network
    or station codes. The ``exclude_networks`` and ``exclude_stations``
    arguments should thus be used for that purpose to ensure compatibility
    across all data providers, e.g.

    .. code-block:: python

        restrictions = Restrictions(
            ...
            network="B*,G*", station="A*, B*",
            exclude_networks=["BW", "GR"],
            exclude_stations=["AL??", "*O"],
            ...
            )

    It is also possible to restrict the downloaded stations to stations part of
    an existing inventory object which can originate from a StationXML file or
    from other sources. It will only keep stations that are part of the
    inventory object. Channels are still selected dynamically based on the
    other restrictions. Keep in mind that all other restrictions still apply -
    passing an inventory will just further restrict the possibly downloaded
    data.

    .. code-block:: python

        restrictions = Restrictions(
            ...
            limit_stations_to_inventory=inv,
            ...
            )

    :param starttime: The start time of the data to be downloaded.
    :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param endtime: The end time of the data.
    :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param station_starttime: The start time of the station files. If not
        given, the ``starttime`` argument will be used. This is useful when
        trying to incorporate multiple waveform datasets with a central
        station file archive as StationXML files can be downloaded once and
        for the whole time span.
    :type station_starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param station_endtime: The end time of the station files. Analogous to
        the ``station_starttime`` argument.
    :type station_endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param chunklength_in_sec: The length of one chunk in seconds. If set,
        the time between ``starttime`` and ``endtime`` will be divided into
        segments of ``chunklength_in_sec`` seconds. Useful for continuous data
        requests. Set to ``None`` if one piece of data is desired between
        ``starttime`` and ``endtime`` (the default).
    :type chunklength_in_sec: float
    :param network: The network code. Can contain wildcards.
    :type network: str
    :param station: The station code. Can contain wildcards.
    :type station: str
    :param location: The location code. Can contain wildcards.
    :type location: str
    :param channel: The channel code. Can contain wildcards.
    :type channel: str
    :param exclude_networks: A list of potentially wildcarded networks that
        should not be downloaded.
    :type exclude_networks: list[str]
    :param exclude_stations: A list of potentially wildcarded stations that
        should not be downloaded.
    :type exclude_stations: list[str]
    :param limit_stations_to_inventory: If given, only stations part of the
        this inventory object will be downloaded. All other restrictions
        still apply - this just serves to further limit the set of stations
        to download.
    :type limit_stations_to_inventory:
        :class:`~obspy.core.inventory.inventory.Inventory`
    :param reject_channels_with_gaps: If True (default), MiniSEED files with
        gaps and/or overlaps will be rejected.
    :type reject_channels_with_gaps: bool
    :param minimum_length: The minimum length of the data as a fraction of
        the requested time frame. After a channel has been downloaded it
        will be checked that its total length is at least that fraction of
        the requested time span. Will be rejected otherwise. Must be between
        ``0.0`` and ``1.0``, defaults to ``0.9``.
    :type minimum_length: float
    :param sanitize: Sanitize makes sure that each MiniSEED file also has an
         associated StationXML file, otherwise the MiniSEED files will be
         deleted afterwards. This is potentially not desirable for large noise
         data sets.
    :type sanitize: bool
    :param minimum_interstation_distance_in_m: The minimum inter-station
        distance. Data from any new station closer to any existing station
        will not be downloaded. Also used for duplicate station detection as
        sometimes stations have different names for different webservice
        providers. Defaults to `1000 m`.
    :type minimum_interstation_distance_in_m: float
    :param channel_priorities: Priority list for the channels. Will not be
        used if the ``channel`` argument is used.
    :type channel_priorities: list[str]
    :param location_priorities: Priority list for the locations. Will not be
        used if the ``location`` argument is used.
    :type location_priorities: list[str]
    """
    def __init__(self, starttime, endtime,
                 station_starttime=None, station_endtime=None,
                 chunklength_in_sec=None,
                 network=None, station=None, location=None, channel=None,
                 exclude_networks=tuple(), exclude_stations=tuple(),
                 limit_stations_to_inventory=None,
                 reject_channels_with_gaps=True, minimum_length=0.9,
                 sanitize=True, minimum_interstation_distance_in_m=1000,
                 channel_priorities=("HH[ZNE12]", "BH[ZNE12]",
                                     "MH[ZNE12]", "EH[ZNE12]",
                                     "LH[ZNE12]", "HL[ZNE12]",
                                     "BL[ZNE12]", "ML[ZNE12]",
                                     "EL[ZNE12]", "LL[ZNE12]",
                                     "SH[ZNE12]"),
                 location_priorities=("", "00", "10", "01", "20", "02", "30",
                                      "03", "40", "04", "50", "05", "60",
                                      "06", "70", "07", "80", "08", "90",
                                      "09")):
        # Awkward logic to keep track whether or not the location priorities
        # are equal to the default values. This "solution" keeps the function
        # signature intact and it also located close to where the location
        # priorities are set.
        if location_priorities == (
                "", "00", "10", "01", "20", "02", "30", "03", "40", "04", "50",
                "05", "60", "06", "70", "07", "80", "08", "90", "09"):
            self._loc_prios_are_default_values = True
        else:
            self._loc_prios_are_default_values = False

        self.starttime = obspy.UTCDateTime(starttime)
        self.endtime = obspy.UTCDateTime(endtime)
        self.station_starttime = station_starttime and \
            obspy.UTCDateTime(station_starttime)
        self.station_endtime = station_endtime and \
            obspy.UTCDateTime(station_endtime)
        if self.station_starttime and self.station_starttime > self.starttime:
            raise ValueError("The station start time must be smaller than the "
                             "main start time.")
        if self.station_endtime and self.station_endtime < self.endtime:
            raise ValueError("The station end time must be larger than the "
                             "main end time.")
        self.chunklength = chunklength_in_sec and float(chunklength_in_sec)
        self.network = network
        self.station = station
        self.location = location
        self.channel = channel
        self.exclude_networks = exclude_networks
        self.exclude_stations = exclude_stations
        self.reject_channels_with_gaps = reject_channels_with_gaps
        self.minimum_length = minimum_length
        self.sanitize = bool(sanitize)

        # These must be iterables, but not strings.
        if not isinstance(channel_priorities, collections.abc.Iterable) \
                or isinstance(channel_priorities, str):
            msg = "'channel_priorities' must be a list or other iterable " \
                  "container."
            raise TypeError(msg)

        if not isinstance(location_priorities, collections.abc.Iterable) \
                or isinstance(location_priorities, str):
            msg = "'location_priorities' must be a list or other iterable " \
                  "container."
            raise TypeError(msg)

        self.channel_priorities = channel_priorities
        self.location_priorities = location_priorities

        self.minimum_interstation_distance_in_m = \
            float(minimum_interstation_distance_in_m)

        # Further restrict the possibly downloaded networks and station to
        # the one in the given inventory.
        if limit_stations_to_inventory is not None:
            self.limit_stations_to_inventory = set()
            for net in limit_stations_to_inventory:
                for sta in net:
                    self.limit_stations_to_inventory.add((net.code, sta.code))
        else:
            self.limit_stations_to_inventory = None

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other

    def __iter__(self):
        """
        Iterator yielding time intervals based on the chunklength and
        temporal settings.
        """
        if not self.chunklength:
            return iter([(self.starttime, self.endtime)])

        def it():
            """
            Tiny iterator.
            """
            starttime = self.starttime
            endtime = self.endtime
            chunklength = self.chunklength

            while starttime < endtime:
                yield (starttime, min(starttime + chunklength, endtime))
                starttime += chunklength
            return

        return it()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
