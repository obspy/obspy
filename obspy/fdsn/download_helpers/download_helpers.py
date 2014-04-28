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
from obspy.fdsn.header import URL_MAPPINGS, FDSNException
from obspy.fdsn import Client
import warnings

# Setup the logger.
FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger("obspy.fdsn.download_helpers")

# Namedtuple containing some query restrictions.
Restrictions = namedtuple("Restrictions", [
    "network",
    "station",
    "location",
    "channel",
    "starttime",
    "endtime"
])


def _filter_channel_priority(channels, priorities=["HH[Z,N,E]", "BH[Z,N,E]",
                                                   "MH[Z,N,E]", "EH[Z,N,E]",
                                                   "LH[Z,N,E]"]):
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
    def __init__(self, clients=None):
        if clients is None:
            clients = URL_MAPPINGS.keys()
        self.clients = clients[:]

        self._initialized_clients = {}

        self.__initialize_clients()

    def __initialize_clients(self):
        """
        Initialize all clients.
        """
        def _get_client(client):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    this_client = Client(client)
                except FDSNException:
                    logger.warn("Failed to initialize client '%s'." % client)
                    return (client, None)
                for warning in w:
                    logger.warn(("Client '%s': " % client) +
                                str(warning.message))
            logger.info("Successfully intialized client '%s'. Available "
                        "services: %s" % (
                        client, ", ".join(sorted(
                            this_client.services.keys()))))
            return client, this_client

        p = ThreadPool(len(self.clients))
        clients = p.map(_get_client, self.clients)
        p.close()

        for c in clients:
            self._initialized_clients[c[0]] = c[1]


if __name__ == "__main__":
    # Setup the domain.
    domain = RectangularDomain(
        min_latitude=40,
        max_longitude=50,
        min_latitude=10,
        max_latitude=20)

    # Some more restrictions.
    restrictions = Restrictions(
        network="*",
        station="*",
        location="*",
        # Channels are a priority list.
        channel=["BH[E,N,Z]", "EH[E,N,Z"],
        starttime=obspy.UTCDateTime(2012, 1, 1),
        endtime=obspy.UTCDateTime(2012, 1, 1, 1),
        interval=None
    )

    # Initialize the download helper.
    helper = DownloadHelper()

    # Start the actual download.
    helper.download(domain, restrictions)
