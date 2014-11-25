#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDSN web services download helpers.

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
import warnings

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
    """
    Base exception raised by the download helpers.
    """
    pass


class DownloadHelper(object):
    """
    Class facilitating data acquisition across all FDSN web service
    implementation.

    :param providers: List of FDSN client names or service URLS. Will use
        all FDSN implementations known to ObsPy if set to None. The order
        in the list also determines their priority, if data is available at
        more then one provider it will always be downloaded from the
        provider that comes first in the list.
    :type providers: list of str
    """
    def __init__(self, providers=None):
        if providers is None:
            providers = dict(URL_MAPPINGS.items())
            # In that case make sure IRIS is first, and Orfeus last! The
            # remaining items will be sorted alphabetically to make it
            # deterministic at least to a certain extent.
            # Orfeus is last as it currently returns StationXML for all
            # European stations without actually containing the data.
            _p = []
            if "IRIS" in providers:
                _p.append("IRIS")
                del providers["IRIS"]

            orfeus = False
            if "ORFEUS" in providers:
                orfeus = providers["ORFEUS"]
                del providers["ORFEUS"]

            _p.extend(sorted(providers))

            if orfeus:
                _p.append(orfeus)

            providers = _p

        self.providers = tuple(providers)

        # Initialize all clients.
        self._initialized_clients = OrderedDict()
        self.__initialize_clients()

    def download(self, domain, restrictions, mseed_storage,
                 stationxml_storage, download_chunk_size_in_mb=20,
                 threads_per_client=5):
        """
        Launch the actual data download.

        :param domain: The download domain.
        :type domain: :class:`~obspy.fdsn.download_helpers.domain.Domain`
            subclass
        :param restrictions: Non-spatial downloading restrictions.
        :type restrictions:
            :class:`~obspy.fdsn.download_helpers.restrictions.Restrictions`
        :param mseed_storage: Where to store the waveform files. See
            the :module:~obspy.fdsn.download_helpers` for more details.
        :type mseed_storage: str or fct
        :param stationxml_storage: Where to store the StationXML files. See
            the :module:~obspy.fdsn.download_helpers` for more details.
        :type stationxml_storage: str of fct
        :param download_chunk_size_in_mb: MiniSEED data will be downloaded
            in bulk chunks. This settings limits the chunk size. A higher
            numbers means that less total download requests will be send,
            but each individual download request will be larger.
        :type download_chunk_size_in_mb: float, optional
        :param threads_per_client: The number of download threads launched
            per client.
        :type threads_per_client: int, optional
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
        # really feasible as long as the availability queries are not
        # reliable for all endpoints.
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

            # Sanitize the downloaded things if desired. Assures that all
            # waveform data also has corresponding station information.
            if restrictions.sanitize:
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