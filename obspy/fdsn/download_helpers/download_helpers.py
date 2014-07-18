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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import logging
from multiprocessing.pool import ThreadPool
import itertools
import shutil
import tempfile
import obspy
from obspy.core.util.obspy_types import OrderedDict
from obspy.fdsn.header import URL_MAPPINGS, FDSNException
from obspy.fdsn import Client
import warnings

from .utils import filter_stations, merge_stations, get_availability


# Setup the logger.
logger = logging.getLogger("obspy-download-helper")
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


class FDSNDownloadHelperException(FDSNException):
    pass


class Restrictions(object):
    """
    Class storing non-domain bounds restrictions of a query.
    """
    __slots__ = ("starttime", "endtime", "network", "station", "location",
                 "channel", "channel_priorities",
                 "minimum_interstation_distance_in_m")

    def __init__(self, starttime, endtime, network=None, station=None,
                 location=None, channel=None,
                 minimum_interstation_distance_in_m=1000,
                 channel_priorities=("HH[Z,N,E]", "BH[Z,N,E]",
                                     "MH[Z,N,E]", "EH[Z,N,E]",
                                     "LH[Z,N,E]")):
        self.starttime = obspy.UTCDateTime(starttime)
        self.endtime = obspy.UTCDateTime(endtime)
        self.network = network
        self.station = station
        self.location = location
        self.channel = channel
        self.channel_priorities = channel_priorities
        self.minimum_interstation_distance_in_m = \
            float(minimum_interstation_distance_in_m)


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

        # Each call to self.download() will create a new temporary directory
        # and save it in this variable.
        self.temp_dir = None

        # Initialize all clients.
        self._initialized_clients = OrderedDict()
        self.__initialize_clients()

    def download(self, domain, restrictions, settings):
        # Create a temporary directory which will be deleted after the
        # downloading has been finished.
        self.temp_dir = tempfile.mkdtemp(prefix="obspy_fdsn_download_helper")
        logger.info("Using temporary directory: %s" % self.temp_dir)
        try:
            self._download(domain, restrictions, settings)
        finally:
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            logger.info("Deleted temporary directory.")

    def _download(self, domain, restrictions, settings):
        """
        """
        def star_get_availability(args):
            try:
                avail = get_availability(*args, logger=logger)
            except FDSNException as e:
                logger.error(str(e))
                return None
            return avail

        p = ThreadPool(len(self._initialized_clients))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("default")
            availabilities = p.map(
                star_get_availability,
                [(client, client_name, restrictions, domain) for
                 client_name, client in self._initialized_clients.items()])
        p.close()

        # Sort by priority and only include those that actually have
        # available information.
        availabilities = {key: value for key, value in availabilities if
                          value is not None}
        availability = OrderedDict()
        for client in self.providers:
            if client not in availabilities:
                continue
            availability[client] = availabilities[client]

        if not availability:
            msg = "No suitable channel found across FDSN providers."
            logger.error(msg)
            raise FDSNDownloadHelperException(msg)

        # Create the master availability.
        master_availability = availabilities.values()[0]
        master_availability = filter_stations(
            master_availability,
            restrictions.minimum_interstation_distance_in_m)

        for stations in availabilities.values()[1:]:
            master_availability = merge_stations(
                stations, restrictions.minimum_interstation_distance_in_m)

        # Filter out already existing files.
        for station in stations:
            pass

        # Group available stations per client.
        availability = {
            (client_name, self._initialized_clients[client_name]): stations
            for client_name, stations in
            itertools.groupby(master_availability, lambda x: x.client)}

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
                            % client_name)
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
