import collections
import copy
import itertools
import numpy as np
from scipy.spatial import cKDTree

# mean earth radius in meter as defined by the International Union of
# Geodesy and Geophysics.
EARTH_RADIUS = 6371009


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
