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
from future import standard_library
with standard_library.hooks():
    from urllib.error import HTTPError, URLError

import copy
import itertools
import logging
from multiprocessing.pool import ThreadPool
import os
import shutil
import tempfile
import warnings

import obspy
from obspy.core.util.obspy_types import OrderedDict
from obspy.fdsn.header import URL_MAPPINGS, FDSNException
from obspy.fdsn import Client

from . import utils


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
            # In that case make sure IRIS is first, and ORFEUS second! The
            # remaining items will be sorted alphabetically.
            _p = []
            if "IRIS" in providers:
                _p.append("IRIS")
                providers.remove("IRIS")
            if "ORFEUS" in providers:
                _p.append("ORFEUS")
                providers.remove("ORFEUS")
            _p.extend(sorted(providers))
            providers = _p

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
            self.get_availability(domain, restrictions)
        finally:
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            logger.info("Deleted temporary directory.")

    def get_availability(self, domain, restrictions):
        """
        """
        def star_get_availability(args):
            try:
                avail = utils.get_availability_from_client(*args,
                                                           logger=logger)
            except (FDSNException, HTTPError, URLError) as e:
                logger.error(str(e))
                return None
            return avail

        p = ThreadPool(len(self._initialized_clients))
        with warnings.catch_warnings(record=True):
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
        master_availability = utils.filter_stations(
            master_availability.values(),
            restrictions.minimum_interstation_distance_in_m)

        for stations in availabilities.values()[1:]:
            master_availability = utils.merge_stations(
                master_availability,
                stations.values(),
                restrictions.minimum_interstation_distance_in_m)

        logger.info("%i stations remain after merging availability from all "
                    "clients." % len(master_availability))

        # Group available stations per client. Evaluate the iterator right
        # away as it does not stick around.
        availability = {
            (client_name, self._initialized_clients[client_name]):
            list(stations) for client_name, stations in
            itertools.groupby(master_availability, lambda x: x.client)}

        # Final logging messages.
        for client, stations in availability.items():
            logger.info("%i station (%i channels) found for client %s." %
                        (len(stations), sum([len(_i.channels) for _i in
                                             stations]), client[0]))

        return availability

    def download_mseed(self, availability, restrictions, mseed_path,
                       temp_folder, chunk_size=25, threads_per_client=5):
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        avail = {}
        for c, stations in availability.items():
            stations = copy.deepcopy(stations)
            for station in stations:
                channels = []
                for channel in station.channels:
                    filename = utils.get_mseed_filename(mseed_path,
                                                        station.network,
                                                        station.station,
                                                        channel.location,
                                                        channel.channel)
                    channels.append((channel, filename))
                station.channels[:] = channels
            # Split into chunks.
            avail[c] = [stations[i:i + chunk_size]
                        for i in range(0, len(stations), chunk_size)]

        def star_download_mseed(args):
            try:
                ret_val = utils.download_and_split_mseed_bulk(
                    *args, temp_folder=temp_folder, logger=logger)
            except (FDSNException, HTTPError, URLError) as e:
                logger.error(("Client '%s': " % args[1]) + str(e))
                return None
            return None

        thread_pools = []
        thread_results = []
        for c, chunks in avail.items():
            client_name, client = c

            p = ThreadPool(min(threads_per_client, len(chunks)))
            thread_pools.append(p)

            thread_results.append(p.map(
                star_download_mseed, [
                    (client, client_name, restrictions.starttime,
                     restrictions.endtime, chunk) for chunk in chunks]))

        for p in thread_pools:
            p.close()

    def download_stationxml(self, availability, restrictions,
                            stationxml_path, threads_per_client=5):
        avail = {}
        for c, stations in availability.items():
            s = []
            for station in stations:
                filename = utils.get_stationxml_filename(stationxml_path,
                                                         station.network,
                                                         station.station)
                s.append((station, filename))
            avail[c] = s

        def star_download_station(args):
            try:
                utils.download_stationxml(*args, logger=logger)
            except (FDSNException, HTTPError, URLError) as e:
                logger.error(str(e))
                return None
            return None

        thread_pools = []
        thread_results = []
        for c, s in avail.items():
            client_name, client = c

            p = ThreadPool(min(threads_per_client, len(s)))
            thread_pools.append(p)

            thread_results.append(p.map(
                star_download_station, [
                    (client, client_name, restrictions.starttime,
                     restrictions.endtime, station, filename) for
                    station, filename in s]))

        for p in thread_pools:
            p.close()

    def __initialize_clients(self):
        """
        Initialize all clients.
        """
        logger.info("Initializing FDSN clients for %s."
                    % ", ".join(self.providers))

        def _get_client(client_name):
            try:
                this_client = Client(client_name)
            except (FDSNException, HTTPError, URLError):
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
