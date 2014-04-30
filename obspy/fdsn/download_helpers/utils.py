import collections
import copy
import fnmatch
import itertools
from urllib2 import HTTPError
import numpy as np
from scipy.spatial import cKDTree

from obspy.fdsn.client import FDSNException

# mean earth radius in meter as defined by the International Union of
# Geodesy and Geophysics.
EARTH_RADIUS = 6371009


# Used to keep track of what to download.
Station = collections.namedtuple("Station",
                                 ["network", "station", "latitude",
                                  "longitude", "elevation_in_m", "channels",
                                  "client"])
Channel = collections.namedtuple("Channel", ["location", "channel"])


def get_availability(client, client_name, restrictions, domain, logger):
    """
    Returns availability information from an initialized FDSN client.

    :type client: :class:`obspy.fdsn.client.Client`
    :param client: An initialized FDSN client.
    :type client_name: str
    :param client_name: The name of the client. Only used for logging.
    :type restrictions: :class:`obspy.fdsn.download_helpers.Restrictions`
    :param restrictions: The non-domain related restrictions for the query.
    :type domain: :class:`obspy.fdsn.download_helpers.Domain` subclass
    :param domain: The domain definition.
    :rtype: dict

    Return a dictionary akin to the following containing information about
    all available channels according to the webservice.

    .. code-block:: python

         {("NET", "STA1"): Station(network="NET", station="STA1",
            latitude=1.0, longitude=2.0, elevation_in_m=3.0,
            channels=(Channel(location="", channel="EHE"),
                      Channel(...),  ...),
            client="IRIS"),
          ("NET", "STA2"): Station(network="NET", station="STA2",
            latitude=1.0, longitude=2.0, elevation_in_m=3.0,
            channels=(Channel(location="", channel="EHE"),
                      Channel(...),  ...),
            client="IRIS"),
          ...
         }
    """
    # Check if stations needs to be filtered after downloading or if the
    # restrictions one can impose with FDSN webservice query are enough.
    try:
        domain.is_in_domain(0, 0)
        needs_filtering = True
    except NotImplementedError:
        needs_filtering = False

    arguments = {
        "network": restrictions.network,
        "station": restrictions.station,
        "location": restrictions.location,
        "channel": restrictions.channel,
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
    except (FDSNException, HTTPError) as e:
        logger.error(
            "Failed getting availability for client '{0}': %s".format(
                client_name), str(e))
        return client_name, None
    logger.info("Successfully requested availability from client '%s'" %
                client_name)

    availability = {}

    for network in inv:
        for station in network:
            # Skip the station if it is not in the desired domain.
            if needs_filtering is True and \
                    not domain.is_in_domain(station.latitude,
                                            station.longitude):
                continue

            # Group by locations and apply the channel priority filter to
            # each.
            filtered_channels = []
            for location, channels in itertools.groupby(
                    station.channels, lambda x: x.location_code):
                channels = list(set([_i.code for _i in channels]))
                filtered_channels.extend([
                    Channel(location, _i) for _i in
                    filter_channel_priority(
                        channels, priorities=restrictions.channel_priorities)])

            if not filtered_channels:
                continue

            availability[(network.code, station.code)] = Station(
                network=network.code,
                station=station.code,
                latitude=station.latitude,
                longitude=station.longitude,
                elevation_in_m=station.elevation,
                channels=filtered_channels,
                client=client_name
            )

    logger.info("Found %i matching channels from client '%s'." %
                (sum([len(_i.channels) for _i in availability.values()]),
                 client_name))
    return client_name, availability


class SphericalNearestNeighbour(object):
    """
    Spherical nearest neighbour queries using scipy's fast
    kd-tree implementation.
    """
    def __init__(self, data):
        cart_data = self.spherical2cartesian(data)
        self.data = data
        self.kd_tree = cKDTree(data=cart_data, leafsize=10)

    def query(self, points, k=10):
        points = self.spherical2cartesian(points)
        d, i = self.kd_tree.query(points, k=k)
        return d, i

    def query_pairs(self, maximum_distance):
        return self.kd_tree.query_pairs(maximum_distance)

    @staticmethod
    def spherical2cartesian(data):
        """
        Converts a list of :class:`~obspy.fdsn.download_helpers.Station`
        objects to an array of shape(len(list), 3) containing x/y/z in meters.
        """
        # Create three arrays containing lat/lng/elevation.
        shape = len(data)
        lat = np.array([_i.latitude for _i in data], dtype=np.float64)
        lon = np.array([_i.longitude for _i in data], dtype=np.float64)
        r = np.array([EARTH_RADIUS + _i.elevation_in_m for _i in data],
                     dtype=np.float64)
        # Convert data from lat/lng to x/y/z.
        colat = 90.0 - lat
        cart_data = np.empty((shape, 3), dtype=np.float64)

        cart_data[:, 0] = r * np.sin(np.deg2rad(colat)) * \
            np.cos(np.deg2rad(lon))
        cart_data[:, 1] = r * np.sin(np.deg2rad(colat)) * \
            np.sin(np.deg2rad(lon))
        cart_data[:, 2] = r * np.cos(np.deg2rad(colat))

        return cart_data


def filter_channel_priority(channels, priorities=("HH[Z,N,E]", "BH[Z,N,E]",
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
    :type priorities: list of unicode or None
    :param priorities: The desired channels with descending priority. Channels
        will be matched by fnmatch.fnmatch() so wildcards and sequences are
        supported. The advisable form to request the three standard components
        of a channel is "HH[Z,N,E]" to avoid getting e.g. rotated components.
    :returns: A new list containing only the filtered channels.
    """
    if priorities is None:
        return channels
    filtered_channels = []
    for pattern in priorities:
        if filtered_channels:
            break
        for channel in channels:
            if fnmatch.fnmatch(channel, pattern):
                filtered_channels.append(channel)
                continue
    return filtered_channels



def filter_stations(stations, minimum_distance_in_m):
    """
    Removes stations until all stations have a certain minimum distance to
    each other.
    """
    stations = copy.copy(stations)
    nd_tree = SphericalNearestNeighbour(stations)
    nns = nd_tree.query_pairs(minimum_distance_in_m)

    indexes_to_remove = []

    # Keep removing the station with the most pairs until no pairs are left.
    while nns:
        most_common = collections.Counter(
            itertools.chain.from_iterable(nns)).most_common()[0][0]
        indexes_to_remove.append(most_common)
        nns = list(itertools.ifilterfalse(lambda x: most_common in x, nns))

    # Remove these indices.
    return [_i[1] for _i in itertools.ifilterfalse(
            lambda x: x[0] in indexes_to_remove,
            enumerate(stations))]


def merge_stations(stations, other_stations, minimum_distance_in_m=0.0):
    """
    Merges two lists containing station objects. The first list is assumed
    to already be filtered with
    :func:`~obspy.fdsn.download_helpers.utils.filter_stations` and therefore
    contain no two stations within ``minimum_distance_in_m`` from each other.
    The stations in the ``other_station`` list will be successively added to
    ``stations`` while ensuring the minimum inter-station distance will
    remain be honored.
    """
    if not minimum_distance_in_m:
        raise NotImplementedError

    stations = copy.copy(stations)
    for station in other_stations:
        kd_tree = SphericalNearestNeighbour(stations)
        neighbours = kd_tree.query([station])[0][0]
        if np.isinf(neighbours[0]):
            continue
        min_distance = neighbours[0]
        if min_distance < minimum_distance_in_m:
            continue
        stations.append(station)

    return stations
