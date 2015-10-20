#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Non-geographical restrictions and constraints for the mass downloader.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014-2015
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

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
    ...     channel_priorities=("HH[ZNE]", "BH[ZNE]"),
    ...     # Location codes are arbitrary and there is no rule as to which
    ...     # location is best.
    ...     location_priorities=("", "00", "10"))


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
    :type channel_priorities: list of str
    :param location_priorities: Priority list for the locations. Will not be
        used if the ``location`` argument is used.
    :type location_priorities: list of str
    """
    def __init__(self, starttime, endtime,
                 station_starttime=None, station_endtime=None,
                 chunklength_in_sec=None,
                 network=None, station=None, location=None, channel=None,
                 reject_channels_with_gaps=True, minimum_length=0.9,
                 sanitize=True, minimum_interstation_distance_in_m=1000,
                 channel_priorities=("HH[ZNE]", "BH[ZNE]",
                                     "MH[ZNE]", "EH[ZNE]",
                                     "LH[ZNE]"),
                 location_priorities=("", "00", "10")):
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
        self.reject_channels_with_gaps = reject_channels_with_gaps
        self.minimum_length = minimum_length
        self.sanitize = bool(sanitize)
        self.channel_priorities = channel_priorities
        self.location_priorities = location_priorities
        self.minimum_interstation_distance_in_m = \
            float(minimum_interstation_distance_in_m)

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
            raise StopIteration

        return it()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
