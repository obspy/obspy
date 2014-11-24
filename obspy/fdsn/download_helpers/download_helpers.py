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

import logging
from multiprocessing.pool import ThreadPool
import warnings

import obspy
from obspy.core.util.obspy_types import OrderedDict
from obspy.fdsn.header import URL_MAPPINGS, FDSNException
from obspy.fdsn import Client

from . import utils
from .download_status import ClientDownloadHelper


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


class DownloadHelper(object):
    """
    Class facilitating data acquistion across all FDSN web service
    implementation.

    :param providers: List of FDSN client names or service URLS. Will use
        all FDSN implementations known to ObsPy if set to None. The order
        in the list also determines their priority, if data is available at
        more then one provider it will always be downloaded from the
        provider that comes first in the list.
    """
    def __init__(self, providers=None):
        if providers is None:
            providers = dict(URL_MAPPINGS.items())
            # In that case make sure IRIS is first, and ORFEUS second! The
            # remaining items will be sorted alphabetically to make it
            # deterministic at least to a certain extent.
            _p = []
            if "IRIS" in providers:
                _p.append("IRIS")
                del providers["IRIS"]
            if "ORFEUS" in providers:
                _p.append("ORFEUS")
                del providers["ORFEUS"]
            _p.extend(sorted(providers))
            providers = _p

        self.providers = tuple(providers)

        # Initialize all clients.
        self._initialized_clients = OrderedDict()
        self.__initialize_clients()

    def download(self, domain, restrictions, mseed_storage,
                 stationxml_storage, download_chunk_size_in_mb=50,
                 threads_per_client=5):
        """
        Download data.

        :param domain:
        :param restrictions:
        :param mseed_storage:
        :param stationxml_storage:
        :param download_chunk_size_in_mb:
        :param threads_per_client:
        """
        # Collect all the downloaded stations.
        existing_stations = set()

        # Set of network and station tuples, e.g. {(“NET1”, “STA1”),
        # (“NET2”, “STA2”), …}. Will be used to not attempt to download
        # stations that have been rejected during a previous loop iteration.
        # Station can be rejected if they are too close to an already existing
        # station.
        discarded_station_ids = set()

        report = []

        # Do it sequentially for each client. Doing it in parallel is not
        # really feasible as long as the availability queries are not reliable.
        for client_name, client in self._initialized_clients.items():
            logger.info("Stations already acquired during this run: %i" %
                        len(existing_stations))

            # The client download helper object is responsible for the
            # downloads of a single FDSN endpoint.
            helper = ClientDownloadHelper(
                client=client, client_name=client_name,
                restrictions=restrictions, domain=domain,
                mseed_storage=mseed_storage,
                stationxml_storage=stationxml_storage, logger=logger)

            # Request the availability.
            helper.get_availability()
            # Continue if there is not data.
            if not helper:
                report.append({"client": client_name, "data": []})
                continue

            # First filter stage. Remove stations based on the station id,
            # e.g. NETWORK.STATION. Remove all that already exist and all
            # that are in the discarded station ids set.
            helper.discard_stations(existing_stations.union(
                discarded_station_ids))

            logger.info("Client '%s' - After discarding duplicates based on "
                        "the station id, %i stations remain." % (
                            client_name, len(helper)))
            # If nothing is there, no need to keep going.
            if not helper:
                report.append({"client": client_name, "data": []})
                continue

            # Filter based on the distance to the next closest station. If
            # info["reliable"] is True, it is assumed that we can actually
            # get all the data in the availability, otherwise everything
            # will be attempted to be downloaded.
            # f = utils.filter_based_on_interstation_distance(
            #     existing_stations=existing_stations,
            #     new_stations=availability,
            #     reliable_new_stations=info["reliable"],
            #     minimum_distance_in_m=
            #     restrictions.minimum_interstation_distance_in_m)
            # # Add the rejected stations to the set of discarded station ids
            # # so they will not be attempted to be downloaded again.
            # for station in f["rejected_stations"]:
            #     discarded_station_ids.add((station.network, station.station))
            # availability = f["accepted_stations"]
            #
            # logger.info("Client '%s' - %i station(s) satisfying the "
            #             "minimum inter-station distance found." % (
            #             client_name, len(availability)))
            # if not availability:
            #     report.append({"client": client_name, "data": []})
            #     continue

            # Download MiniSEED data.
            helper.prepare_mseed_download()
            helper.download_mseed(chunk_size_in_mb=download_chunk_size_in_mb,
                                  threads_per_client=threads_per_client)

            # Download StationXML data.
            helper.prepare_stationxml_download()
            helper.download_stationxml()

            # Sanitize the downloaded things. Assures that all waveform data
            # also has corresponding
            helper.sanitize_downloads()

        return report

    def __initialize_clients(self):
        """
        Initialize all clients.
        """
        logger.info("Initializing FDSN client(s) for %s."
                    % ", ".join(self.providers))

        def _get_client(client_name):
            try:
                this_client = Client(client_name)
            except utils.ERRORS as e:
                if "timeout" in str(e).lower():
                    extra = " (timeout)"
                else:
                    extra = ""
                logger.warn("Failed to initialize client '%s'.%s"
                            % (client_name, extra))
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

        logger.info("Successfully initialized %i client(s): %s."
                    % (len(self._initialized_clients),
                       ", ".join(self._initialized_clients.keys())))