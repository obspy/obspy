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
import fnmatch
import logging
from multiprocessing.pool import ThreadPool
import collections
import itertools
import obspy
from obspy.core.util.obspy_types import OrderedDict
from obspy.fdsn.header import URL_MAPPINGS, FDSNException
from obspy.fdsn import Client
import warnings

# Setup the logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Prevent propagating to higher loggers.
logger.propagate = 0

# Console log handler.
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Add formatter
FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)


# Namedtuple containing some query restrictions.
Restrictions = namedtuple("Restrictions", [
    "network",
    "station",
    "location",
    "channel",
    "starttime",
    "endtime"
])


def _filter_channel_priority(channels, priorities=("HH[Z,N,E]", "BH[Z,N,E]",
                                                   "MH[Z,N,E]", "EH[Z,N,E]",
                                                   "LH[Z,N,E]")):
    """
    This function takes a dictionary containing channels keys and returns a new
    one filtered with the given priorities list.

    All channels matching the first pattern in the list will be retrieved. If
    one or more channels are found it stops. Otherwise it will attempt to
    retrieve channels matching the next pattern. And so on.

    :type channels: list
    :param channels: A list containing channel names.
    :type priorities: list of unicode
    :param priorities: The desired channels with descending priority. Channels
        will be matched by fnmatch.fnmatch() so wildcards and sequences are
        supported. The advisable form to request the three standard components
        of a channel is "HH[Z,N,E]" to avoid getting e.g. rotated components.
    :returns: A new list containing only the filtered channels.
    """
    filtered_channels = []
    for pattern in priorities:
        if filtered_channels:
            break
        for channel in channels:
            if fnmatch.fnmatch(channel, pattern):
                filtered_channels.append(channel)
                continue
    return filtered_channels


def get_availability(client, client_name, restrictions, domain):
    """
    Returns availability information from an initialized FDSN client.

    :type client: :class:`obspy.fdsn.client.Client`
    :param client: An initialized FDSN client.
    :type client_name: str
    :param client_name: The name of the client. Only used for logging.
    :type restrictions: :class:`obspy.fdsn.download_helpers.Restrictions`
    :param restrictions: The non-domain related restrictions for the query.
    :type domain: :class:`obspy.fdsn.download_helpers.Domain` subclass
    :param domain: The domain related restrictions.
    :rtype: dict

    Return a dictionary akin to the following containing information about
    all available channels according to the webservice.

    .. code-block:: python

         {"NET.STA1": {
             "latitude": 1.0,
             "longitude": 2.0,
             "elevation_in_m": 10.0,
             "channels": [".BHE", ".BHN", ".BHZ", "00.LHE", "00.LHE", ...]},
          "NET.STA2": {...},
          ...
         }
    """
    # Check if stations needs to be filtered after downloading or if the
    # restrictions one can impose with the FDSN webservices are enough.
    needs_filtering = True
    if domain.is_in_domain(0, 0) is None:
        needs_filtering = False

    arguments = {
        "network": restrictions.network,
        "station": restrictions.station,
        "location": restrictions.location,
        # Channels work by setting priority lists.
        "channel": None,
        "starttime": restrictions.starttime,
        "endtime": restrictions.endtime,
        # Request at the channel level.
        "level": "channel"
    }
    # Add the domain specific query parameters.
    arguments.update(domain.get_query_parameters())

    logger.info("Requesting availability from client '%s'" % client_name)
    try:
        inv = client.get_stations(**arguments)
    except FDSNException as e:
        logger.exception(
            "Failed getting availability for client '{0}': %s".format(
                client_name), e)
        return
    logger.info("Successfully requested availability from client '%s'" %
                client_name)

    availability = {}

    for network in inv:
        for station in network:
            # Skip the station if it is not in the desired domain.
            if needs_filtering is True:
                if domain.is_in_domain(station.latitude,
                                       station.longitude) is False:
                    continue

            # Filter each location's channels.
            locations = collections.defaultdict(list)
            for channel in station:
                locations[channel.location_code].append(channel.code)
            for key, value in locations.items():
                locations[key] = _filter_channel_priority(value)
            channels = itertools.chain(locations.values())

            availability["{0}.{1}".format(network.code, station.code)] = {
                "latitude": station.latitude,
                "longitude": station.longitude,
                "elevation_in_m": station.elevation,
                "channels": channels
            }

    logger.info("Found %i matching channels from client '%s'." % client_name)
    return availability


class DownloadHelper(object):
    def __init__(self, providers=None):
        """

        :param providers: List of FDSN client names or service URLS. Will use
            all FDSN implementations known to ObsPy if set to None. The order
            in the list also determines the priority, if data is available at
            more then one provider it will always be downloaded from the
            provider that comes first in the list.
        """
        if providers is None:
            providers = URL_MAPPINGS.keys()
        # Immutable tuple.
        self.providers = tuple(providers)

        # Initialize all clients.
        self._initialized_clients = OrderedDict()
        self.__initialize_clients()

    def __initialize_clients(self):
        """
        Initialize all clients.
        """
        logger.info("Initializing FDSN clients for %s."
                    % ", ".join(self.providers))

        def _get_client(client_name):
            try:
                this_client = Client(client_name)
            except FDSNException:
                logger.warn("Failed to initialize client '%s'."
                            % client_name)
                return client_name, None
            services = sorted([_i for _i in this_client.services.keys()
                               if not _i.startswith("available")])
            if "dataselect" not in services or "station" not in services:
                logger.info("Cannot use client '%s' as it does not have "
                            "'dataselect' and/or 'station' services."
                            % (client_name))
                return client_name, None
            return client_name, this_client

        # Catch warnings in the main thread. The catch_warnings() context
        # manager does not reliably work when used in multiple threads.
        p = ThreadPool(len(self.providers))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clients = p.map(_get_client, self.providers)
        p.close()
        for warning in w:
            logger.debug("Warning during initializing one of the clients: " +
                         str(warning.message))

        clients = {key: value for key, value in clients if value is not None}
        # Write to initialized clients dictionary preserving order.
        for client in self.providers:
            if client not in clients:
                continue
            self._initialized_clients[client] = clients[client]

        logger.info("Successfully initialized %i clients: %s."
                    % (len(self._initialized_clients),
                       ", ".join(self._initialized_clients.keys())))
