#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helpers to download potentially big data sets across a number of FDSN web
services in an automated fashion.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014-2015
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future import standard_library

with standard_library.hooks():
    from collections import OrderedDict

import collections
import logging
from multiprocessing.pool import ThreadPool
import os
import warnings

from obspy.clients.fdsn.header import URL_MAPPINGS, FDSNException
from obspy.clients.fdsn.client import Client

from . import utils
from .download_helpers import ClientDownloadHelper, STATUS


# Setup the logger.
logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
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


class FDSNMassDownloaderException(FDSNException):
    """
    Base exception raised by the download helpers.
    """
    pass


class MassDownloader(object):
    """
    Class facilitating data acquisition across all FDSN web service
    implementations.

    :param providers: List of FDSN client names or service URLS. Will use
        all FDSN implementations known to ObsPy if set to None. The order
        in the list also determines their priority, if data is available at
        more then one provider it will always be downloaded from the
        provider that comes first in the list.
    :param debug: Debug flag passed to the underlying FDSN web service clients.
    :type providers: list of str
    """
    def __init__(self, providers=None, debug=False):
        self.debug = debug
        # If not given, use all providers ObsPy knows. They will be sorted
        # alphabetically except that ORFEUS is second to last and IRIS last.
        # The reason for this order is that smaller data centers can be
        # expected to have more up-to-date meta information about their own
        # data and thus should be prioritized when downloading data.
        if providers is None:
            providers = dict(URL_MAPPINGS.items())
            _p = []

            if "IRIS" in providers:
                has_iris = True
                del providers["IRIS"]
            else:
                has_iris = False

            if "ODC" in providers:
                providers["ORFEUS"] = providers["ODC"]
                del providers["ODC"]

            if "ORFEUS" in providers:
                has_orfeus = True
                del providers["ORFEUS"]
            else:
                has_orfeus = False

            _p = sorted(providers)
            if has_orfeus:
                _p.append("ORFEUS")
            if has_iris:
                _p.append("IRIS")

            providers = _p

        self.providers = tuple(providers)

        # Initialize all clients.
        self._initialized_clients = OrderedDict()
        self._initialize_clients()

    def download(self, domain, restrictions, mseed_storage,
                 stationxml_storage, download_chunk_size_in_mb=20,
                 threads_per_client=3, print_report=True):
        """
        Launch the actual data download.

        :param domain: The download domain.
        :type domain: :class:`~.domain.Domain`
        :param restrictions: Non-spatial downloading restrictions.
        :type restrictions: :class:`~.restrictions.Restrictions`
        :param mseed_storage: Where to store the waveform files. See
            the :mod:`~obspy.clients.fdsn.mass_downloader` documentation for
            more details.
        :type mseed_storage: str or function
        :param stationxml_storage: Where to store the StationXML files. See
            the :mod:`~obspy.clients.fdsn.mass_downloader` documentation for
            more details.
        :type stationxml_storage: str or function
        :param download_chunk_size_in_mb: MiniSEED data will be downloaded
            in bulk chunks. This settings limits the chunk size. A higher
            number means that less total download requests will be sent,
            but each individual download request will be larger.
        :type download_chunk_size_in_mb: float
        :param threads_per_client: The number of download threads launched
            per client.
        :type threads_per_client: int
        """
        # The downloads from each client will be handled separately.
        # Nonetheless collect all in this dictionary.
        client_download_helpers = {}

        # Do it sequentially for each client. Doing it in parallel is not
        # really feasible as long as the availability queries are not
        # reliable for all endpoints.
        for client_name, client in self._initialized_clients.items():
            # Log some information about preexisting data.
            station_count = 0
            for _c in client_download_helpers.values():
                station_count += len([
                    _i for _i in _c.stations.values() if
                    (_i.stationxml_status == STATUS.EXISTS) or
                    (_i.has_existing_or_downloaded_time_intervals)])
            logger.info("Total acquired or preexisting stations: %i" %
                        station_count)

            # The client download helper object is responsible for the
            # downloads of a single FDSN endpoint.
            helper = ClientDownloadHelper(
                client=client, client_name=client_name,
                restrictions=restrictions, domain=domain,
                mseed_storage=mseed_storage,
                stationxml_storage=stationxml_storage, logger=logger)
            existing_client_dl_helpers = list(
                client_download_helpers.values())
            client_download_helpers[client_name] = helper

            # Request the availability.
            helper.get_availability()

            # Continue if there is no data.
            if not helper:
                logger.info("Client '%s' - No data available." % client_name)
                continue

            # First filter stage. Remove stations based on the station id,
            # e.g. NETWORK.STATION. Remove all that already exist.
            helper.discard_stations(
                existing_client_dl_helpers=existing_client_dl_helpers)

            # Continue if there is no data.
            if not helper:
                logger.info("Client '%s' - No new data available after "
                            "discarding already downloaded data." %
                            client_name)
                continue

            # If the availability information is reliable, the filtering
            # will happen before the downloading.
            if helper.is_availability_reliable:
                helper.filter_stations_based_on_minimum_distance(
                    existing_client_dl_helpers=existing_client_dl_helpers)
                # Continue if there is no data left after the filtering.
                if not helper:
                    logger.info("Client '%s' - No new data available after "
                                "discarding based on the minimal "
                                "inter-station distance." % client_name)
                    continue

            logger.info("Client '%s' - Will attempt to download data from %i "
                        "stations." % (client_name, len(helper)))

            # Download MiniSEED data.
            helper.prepare_mseed_download()
            helper.download_mseed(chunk_size_in_mb=download_chunk_size_in_mb,
                                  threads_per_client=threads_per_client)

            # Download StationXML data.
            helper.prepare_stationxml_download()
            helper.download_stationxml()

            # Sanitize the downloaded things if desired. Assures that all
            # waveform data also has the corresponding station information.
            if restrictions.sanitize:
                helper.sanitize_downloads()

            # Filter afterwards if availability information is not reliable.
            # This unfortunately results in already downloaded data being
            # discarded but it is the only currently feasible way.
            if not helper.is_availability_reliable:
                helper.filter_stations_based_on_minimum_distance(
                    existing_client_dl_helpers=existing_client_dl_helpers)

        if print_report:
            # Collect already existing things.
            existing_miniseed_files = []
            existing_stationxml_files = []
            new_miniseed_files = collections.defaultdict(list)
            new_stationxml_files = collections.defaultdict(list)

            for cdh in client_download_helpers.values():
                for station in cdh.stations.values():
                    if station.stationxml_status == STATUS.EXISTS:
                        existing_stationxml_files.append(
                            station.stationxml_filename)
                    elif station.stationxml_status == STATUS.DOWNLOADED:
                        new_stationxml_files[cdh.client_name].append(
                            station.stationxml_filename)
                    for channel in station.channels:
                        for ti in channel.intervals:
                            if ti.status == STATUS.EXISTS:
                                existing_miniseed_files.append(ti.filename)
                            elif ti.status == STATUS.DOWNLOADED:
                                new_miniseed_files[cdh.client_name].append(
                                    ti.filename)

            def count_filesize(list_of_files):
                return sum([os.path.getsize(_i) for _i in list_of_files if
                            os.path.exists(_i)])

            logger.info(30 * "=" + " Final report")
            logger.info("%i MiniSEED files [%.1f MB] already existed." % (
                len(existing_miniseed_files),
                count_filesize(existing_miniseed_files) / 1024.0 ** 2))
            logger.info("%i StationXML files [%.1f MB] already existed." % (
                len(existing_stationxml_files),
                count_filesize(existing_stationxml_files) / 1024.0 ** 2))

            total_downloaded_filesize = 0
            for cdh in client_download_helpers.values():
                mseed_files = new_miniseed_files[cdh.client_name]
                stationxml_files = new_stationxml_files[cdh.client_name]
                mseed_filesize = count_filesize(mseed_files)
                stationxml_filesize = count_filesize(stationxml_files)
                total_downloaded_filesize += mseed_filesize
                total_downloaded_filesize += stationxml_filesize
                logger.info("Client '%s' - Acquired %i MiniSEED files "
                            "[%.1f MB]." % (cdh.client_name, len(mseed_files),
                                            mseed_filesize / 1024.0 ** 2))
                logger.info("Client '%s' - Acquired %i StationXML files "
                            "[%.1f MB]." % (
                                cdh.client_name, len(stationxml_files),
                                stationxml_filesize / 1024.0 ** 2))
            logger.info("Downloaded %.1f MB in total." % (
                total_downloaded_filesize / 1024.0 ** 2))

        return client_download_helpers

    def _initialize_clients(self):
        """
        Initialize all clients.
        """
        logger.info("Initializing FDSN client(s) for %s."
                    % ", ".join(self.providers))

        def _get_client(client_name):
            try:
                this_client = Client(client_name, debug=self.debug)
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
            logger.debug("Warning during initialization of one of the "
                         "clients: " + str(warning.message))

        clients = {key: value for key, value in clients if value is not None}
        # Write to initialized clients dictionary preserving order.
        for client in self.providers:
            if client not in clients:
                continue
            self._initialized_clients[client] = clients[client]

        logger.info("Successfully initialized %i client(s): %s."
                    % (len(self._initialized_clients),
                       ", ".join(self._initialized_clients.keys())))


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
