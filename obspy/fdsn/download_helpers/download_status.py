#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class representing the current status of a download in progress.

Intended to simplify and stabilize the logic of the download helpers and make
it understandable in the first place.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future import standard_library
with standard_library.hooks():
    import itertools

import collections
import copy
import time

from .utils import ERRORS, filter_channel_priority

# Some application.wadls are wrong...
OVERWRITE_CAPABILITIES = {
    "resif": None
}


TimeInterval = collections.namedtuple(
    "TimeInterval",  ["start", "end", "filename", "status"])


class Station(object):
    __slots__ = ["network", "station", "latitude", "longitude"]

    def __init__(self, network, station, latitude, longitude, channels):
        self.network = network
        self.station = station
        self.latitude = latitude
        self.longitude = longitude
        self.channels = channels
        self.stationxml_filename = None
        self.stationxml_status = None

    def __str__(self):
        channels = "\n".join(str(i) for i in self.channels)
        channels = "\n\t".join(channels.splitlines())
        return (
            "Station '{network}.{station}' [Lat: {lat:.2f}, Lng: {lng:.2f}]\n"
            "\t-> Filename: {filename}, Status: {status}"
            "\n\t{channels}"
        ).format(
            network=self.network,
            station=self.station,
            lat=self.latitude,
            lng=self.longitude,
            filename=self.stationxml_filename,
            status=self.stationxml_status,
            channels=channels)


class Channel(object):
    """
    Object representing a Channel. Each time interval should end up in one
    MiniSEED file.
    """
    __slots__ = ["location", "channel", "intervals"]

    def __init__(self, location, channel, intervals):
        self.location = location
        self.channel = channel
        self.intervals = intervals

    def __str__(self):
        return "Channel '{location}.{channel}:'\n\t{intervals}".format(
            location=self.location, channel=self.channel,
            intervals="\n\t".join([str(i) for i in self.intervals]))


class ClientDownloadHelper(object):
    """
    :type client: :class:`obspy.fdsn.client.Client`
    :param client: An initialized FDSN client.
    :type client_name: str
    :param client_name: The name of the client. Only used for logging.
    :type restrictions: :class:`obspy.fdsn.download_helpers.Restrictions`
    :param restrictions: The non-domain related restrictions for the query.
    :type domain: :class:`obspy.fdsn.download_helpers.Domain` subclass
    :param domain: The domain definition.
    :rtype: dict
    """
    def __init__(self, client, client_name, restrictions, domain, logger):
        self.client = client
        self.client_name = client_name
        self.restrictions = restrictions
        self.domain = domain
        self.logger = logger
        self.stations = {}
        self.is_availability_reliable = None

    def __str__(self):
        if self.is_availability_reliable is None:
            reliability = "Unknown reliability of availability information"
        elif self.is_availability_reliable is True:
            reliability = "Reliable availability information"
        elif self.is_availability_reliable is False:
            reliability = "Non-reliable availability information"
        else:
            raise NotImplementedError
        return (
            "ClientDownloadHelper object for client '{client}' ({url})\n"
            "-> {reliability}\n"
            "-> Manages {station_count} stations.\n{stations}").format(
                client=self.client_name,
                url=self.client.base_url,
                reliability=reliability,
                station_count=len(self),
                stations="\n".join([str(_i) for _i in self.stations.values()]))

    def __len__(self):
        return len(self.stations)

    def get_availability(self):
        """
        Queries the current client for information of what stations are
        available given the spatial and temporal restrictions.
        """
        # Check if stations needs to be filtered after downloading or if the
        # restrictions one can impose with the FDSN webservices queries are
        # enough. This depends on the domain definition.
        try:
            self.domain.is_in_domain(0, 0)
            needs_filtering = True
        except NotImplementedError:
            needs_filtering = False

        arguments = {
            "network": self.restrictions.network,
            "station": self.restrictions.station,
            "location": self.restrictions.location,
            "channel": self.restrictions.channel,
            "starttime": self.restrictions.starttime,
            "endtime": self.restrictions.endtime,
            # Request at the channel level.
            "level": "channel"
        }
        # Add the domain specific query parameters.
        arguments.update(self.domain.get_query_parameters())

        # Check the capabilities of the service and see what is the most
        # appropriate way of acquiring availability information. Some services
        # right now require manual overwriting of what they claim to be
        # capable of.
        if self.client_name.lower() in OVERWRITE_CAPABILITIES:
            cap = OVERWRITE_CAPABILITIES[self.client_name.lower()]
            if cap is None:
                self.is_availability_reliable = False
            elif cap == "matchtimeseries":
                self.is_availability_reliable = True
                arguments["matchtimeseries"] = True
            elif cap == "includeavailability":
                self.is_availability_reliable = True
                arguments["includeavailability"] = True
            else:
                raise NotImplementedError
        elif "matchtimeseries" in self.client.services["station"]:
            arguments["matchtimeseries"] = True
            self.is_availability_reliable = True
        elif "includeavailability" in self.client.services["station"]:
            self.is_availability_reliable = True
            arguments["includeavailability"] = True
        else:
            self.is_availability_reliable = False

        if self.is_availability_reliable:
            self.logger.info("Client '%s' - Requesting reliable "
                             "availability." % self.client_name)
        else:
            self.logger.info(
                "Client '%s' - Requesting unreliable availability." %
                self.client_name)

        try:
            start = time.time()
            inv = self.client.get_stations(**arguments)
            end = time.time()
        except ERRORS as e:
            if "no data available" in str(e).lower():
                self.logger.info(
                    "Client '%s' - No data available for request." %
                    self.client_name)
                return
            self.logger.error(
                "Client '{0}' - Failed getting availability: %s".format(
                self.client_name), str(e))
            return
        self.logger.info("Client '%s' - Successfully requested availability "
                         "(%.2f seconds)" % (self.client_name, end - start))

        # Get the time intervals from the restrictions.
        intervals = [TimeInterval(start=_i[0], end=_i[1], filename=None,
                                  status=None) for _i in self.restrictions]
        for network in inv:
            for station in network:
                # Skip the station if it is not in the desired domain.
                if needs_filtering is True and \
                        not self.domain.is_in_domain(station.latitude,
                                                     station.longitude):
                    continue

                channels = []
                for channel in station.channels:
                    # Remove channels that somehow slipped past the temporal
                    # constraints due to weird behaviour from the data center.
                    if (channel.start_date > self.restrictions.starttime) or \
                            (channel.end_date < self.restrictions.endtime):
                        continue
                    # Use availability information if possible. In the other
                    # cases it should already work.
                    if "includeavailability" in arguments and \
                            arguments["includeavailability"]:
                        da = channel.data_availability
                        if da is None:
                            self.logger.warning(
                                "Client '%s' supports the "
                                "'includeavailability' parameter but returns "
                                "channels without availability information. "
                                "The final availability might not be "
                                "complete" % self.client_name)
                            continue
                        if (da.start > self.restrictions.starttime) or \
                                (da.end < self.restrictions.endtime):
                            continue
                    channels.append(Channel(
                        location=channel.location_code, channel=channel.code,
                        intervals=copy.deepcopy(intervals)))

                # Group by locations and apply the channel priority filter to
                # each.
                filtered_channels = []
                get_loc = lambda x: x.location
                for location, _channels in itertools.groupby(
                        sorted(channels, key=get_loc), get_loc):
                    filtered_channels.extend(filter_channel_priority(
                        list(_channels), key="channel",
                        priorities=self.restrictions.channel_priorities))
                channels = filtered_channels

                # Filter to remove unwanted locations according to the priority
                # list.
                channels = filter_channel_priority(
                    channels, key="location",
                    priorities=self.restrictions.location_priorities)

                if not channels:
                    continue

                self.stations[(network.code, station.code)] = Station(
                    network=network.code,
                    station=station.code,
                    latitude=station.latitude,
                    longitude=station.longitude,
                    channels=channels)
        self.logger.info("Client '%s' - Found %i station (%i channels)." % (
            self.client_name, len(self.stations),
            sum([len(_i.channels) for _i in self.stations.values()])))
