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
import logging
from multiprocessing.pool import ThreadPool
from obspy.fdsn.header import URL_MAPPINGS, FDSNException
from obspy.fdsn import Client
import warnings

FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger("obspy.fdsn.download_helpers")


domain = namedtuple("domain", [
    "min_latitude",
    "max_latitude",
    "min_longitude",
    "max_longitude",
    # For ciruclar requests.
    "latitude",
    "longitude",
    "mi_nradius",
    "max_radius"])


class Domain(object):
    def get_query_parameters(self):
        raise NotImplementedError

    def is_in_domain(self, latitude, longitude):
        return None


class RectangularDomain(Domain):
    def __init__(self, min_latitude, max_latitude, min_longitude,
                 max_longitude):
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude

    def get_query_parameters(self):
        return domain(
            self.min_latitude,
            self.max_latitude,
            self.min_longitude,
            self.max_longitude,
            None,
            None,
            None,
            None)


class CircularDomain(Domain):
    def __init__(self, latitude, longitude, min_radius, max_radius):
        self.latitude = latitude
        self.longitude = longitude
        self.min_radius = min_radius
        self.max_radius = max_radius

    def get_query_parameters(self):
        return domain(
            None, None, None, None,
            self.latitude,
            self.longitude,
            self.min_radius,
            self.max_radius)


class GlobalDomain(Domain):
    def get_query_parameters(self):
        return domain(None, None, None, None, None, None, None, None)


class DownloadHelper(object):
    def __init__(self, clients=None):
        if clients is None:
            clients = URL_MAPPINGS.keys()
        self.clients = []
        self.clients.extend(clients)

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
    DownloadHelper()
