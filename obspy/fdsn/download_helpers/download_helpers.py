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
import collections
import itertools
import logging
from multiprocessing.pool import ThreadPool
import os
from socket import timeout as SocketTimeout
import shutil
import tempfile
import time
import warnings

import obspy
from obspy.core.util.obspy_types import OrderedDict
from obspy.fdsn.header import URL_MAPPINGS, FDSNException
from obspy.fdsn import Client

from . import utils

ERRORS = (FDSNException, HTTPError, URLError, SocketTimeout)


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
    def __init__(self, starttime, endtime, network=None, station=None,
                 location=None, channel=None,
                 minimum_interstation_distance_in_m=1000,
                 channel_priorities=("HH[Z,N,E]", "BH[Z,N,E]",
                                     "MH[Z,N,E]", "EH[Z,N,E]",
                                     "LH[Z,N,E]"),
                 location_priorities=("", "00", "10")):
        self.starttime = obspy.UTCDateTime(starttime)
        self.endtime = obspy.UTCDateTime(endtime)
        self.network = network
        self.station = station
        self.location = location
        self.channel = channel
        self.reject_channels_with_gaps = True
        self.minimum_length = 0.9
        self.channel_priorities = channel_priorities
        self.location_priorities = location_priorities
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

    def download(self, domain, restrictions, chunk_size=25,
                 threads_per_client=5, mseed_path=None,
                 stationxml_path=None):
        # Collect all the downloaded stations.
        existing_stations = set()

        # Do it sequentially for each client.
        for client_name, client in self._initialized_clients.items():
            logger.info("Stations already acquired during this run: %i" %
                        len(existing_stations))

            info = utils.get_availability_from_client(
                client, client_name, restrictions, domain, logger)

            availability = info["availability"]
            if not availability:
                continue

            # Two cases. Reliable availability means that the client
            # supports the `matchtimeseries` or `includeavailability`
            # parameter and thus the availability information is treated as
            # reliable. Otherwise not. If it is reliable, the stations will
            # be filtered before they are downloaded, otherwise it will
            # happen in reverse to assure as much data as possible is
            # downloaded.
            if info["reliable"]:
                # Filter the stations to be downloaded based on geographical
                # and geometric factors.
                all_stations = utils.merge_stations(
                    existing_stations, info["availability"].values(),
                    restrictions.minimum_interstation_distance_in_m)
            else:
                all_stations = info["availabilty"].values()

            new_stations = all_stations - existing_stations
            logger.info("Client '%s' - %i station(s) satisfying the "
                        "minimum inter-station distance found." % (
                        client_name, len(new_stations)))
            if not new_stations:
                continue

            # "Filter" the station list to get a list of channels that
            # are actually required and a list of existing filenames.
            # The existing filenames will be parsed to download
            # StationXML files for them in case they do not yet exist.
            mseed_stations, existing_miniseed_filenames = \
                self._attach_miniseed_filenames(
                    stations=copy.deepcopy(new_stations),
                    restrictions=restrictions,
                    mseed_path=mseed_path)

            if not mseed_stations and not existing_miniseed_filenames:
                logger.info("Nothing to be downloaded for client %s." %
                            client_name)
                continue

            logger.info("Client '%s' - MiniSEED data from %i channels "
                        "already exists." % (
                        client_name, len(existing_miniseed_filenames)))
            logger.info("Client '%s' - MiniSEED data from %i channels "
                        "will be downloaded." % (
                        client_name, sum(len(_i.channels) for _i in
                                         mseed_stations)))

            # Download the missing channels and get a list of filenames
            # that just got downloaded.
            if mseed_stations:
                a = time.time()
                downloaded_miniseed_filenames = self.download_mseed(
                    client, client_name, mseed_stations, restrictions,
                    chunk_size=chunk_size,
                    threads_per_client=threads_per_client)
                b = time.time()
                f = sum(os.path.getsize(_i) for _i in
                        downloaded_miniseed_filenames)
                f_kb = f / 1024.0
                f_mb = f_kb / 1024.0
                logger.info("Client '%s' - Downloaded %i MiniSEED files "
                            "(%.2f MB) in %.1f seconds (%.1f KB/sec)" % (
                            client_name,
                            len(downloaded_miniseed_filenames), f_mb,
                            b - a, f_kb / (b - a)))
            else:
                downloaded_miniseed_filenames = []

            # Parse the just downloaded and existing MiniSEED files and
            # make a list of stations that require StationXML files.
            miniseed_channels = self._parse_miniseed_filenames(
                downloaded_miniseed_filenames +
                existing_miniseed_filenames, restrictions)
            # Stations will be list of Stations with Channels that all
            # need a StationXML file. This is necessary as MiniSEED data
            # might not be available for all stations.
            stations = utils.filter_stations_with_channel_list(
                new_stations, miniseed_channels)

            # Attach filenames to the StationXML files and get a list of
            # already existing StationXML files.
            stations_to_download = []
            existing_stationxml_channels = []
            for stat in copy.deepcopy(stations):
                filename = utils.get_stationxml_filename(
                    stationxml_path, stat.network, stat.station)
                if not filename:
                    continue
                # If the StationXML file already exists, make sure it
                # contains all the necessary information. Otherwise
                # delete it and it will be downloaded again in the
                # following.
                if os.path.exists(filename):
                    contents = utils.get_stationxml_contents(filename)
                    all_channels_good = True
                    for chan in stat.channels:
                        if utils.is_in_list_of_channel_availability(
                                stat.network, stat.station,
                                chan.location, chan.channel,
                                restrictions.starttime,
                                restrictions.endtime, contents):
                            continue
                        all_channels_good = False
                        break
                    if all_channels_good is False:
                        logger.warning(
                            "StationXML file '%s' already exists but it "
                            "does not contain matching data for all "
                            "MiniSEED data available for this stations. "
                            "It will be deleted and redownloaded." %
                            filename)
                        utils.safe_delete(filename)
                    else:
                        existing_stationxml_channels.extend(contents)
                        continue
                dirname = os.path.dirname(filename)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                stat.stationxml_filename = filename
                stations_to_download.append(stat)

            logger.info("Client '%s' - StationXML data for %i stations ("
                        "%i channels) already exists." % (client_name,
                        len(set((_i.network, _i.station) for _i in
                                existing_stationxml_channels)),
                        len(existing_stationxml_channels)))
            logger.info("Client '%s' - StationXML data for %i stations ("
                        "%i channels) will be downloaded." % (
                        client_name, len(stations_to_download), sum(len(
                        _i.channels) for _i in stations_to_download)))

            # Now download the station information for the downloaded
            # MiniSEED files.
            if stations_to_download:
                i = self.download_stationxml(
                    client, client_name, stations_to_download,
                    restrictions, threads_per_client)
                b = time.time()
                f = sum(os.path.getsize(_i) for _i in i)
                f_kb = f / 1024.0
                f_mb = f_kb / 1024.0
                logger.info("Client '%s' - Downloaded %i StationXML files "
                            "(%.2f MB) in %.1f seconds (%.1f KB/sec)" % (
                                client_name,
                                len(i), f_mb,
                                b - a, f_kb / (b - a)))

                downloaded_stationxml_channels = \
                    list(itertools.chain.from_iterable([
                        utils.get_stationxml_contents(_i) for _i in i]))
            else:
                downloaded_stationxml_channels = []

            stationxml_channels = collections.defaultdict(list)
            for i in existing_stationxml_channels + \
                    downloaded_stationxml_channels:
                stationxml_channels["%s.%s" % (i.network,
                                               i.station)].append(i)

            for mseed in miniseed_channels:
                station = "%s.%s" % (mseed.network, mseed.station)
                if station not in stationxml_channels:
                    msg = "No station file for '%s'. Will be deleted."
                    logger.warning(msg % mseed.filename)
                    utils.safe_delete(mseed.filename)
                    continue
                station = stationxml_channels[station]
                # Try to find the correct timespan!
                found_ts = False
                for channel in station:
                    if (channel.location == mseed.location) and \
                            (channel.channel == mseed.channel) and \
                            (channel.starttime <= mseed.starttime) and \
                            (channel.endtime >= mseed.endtime):
                        found_ts = True
                        break
                if found_ts is False:
                    msg = "No corresponding station file could be " \
                          "retrieved for '%s'. Will be deleted."
                    logger.warning(msg % mseed.filename)
                    utils.safe_delete(mseed.filename)
            if not info["reliable"]:
                from IPython.core.debugger import Tracer; Tracer(colors="Linux")()

    def download_mseed(self, client, client_name, stations, restrictions,
                       chunk_size=25, threads_per_client=5):
        # Split into chunks.
        station_chunks = [stations[i:i + chunk_size]
                          for i in range(0, len(stations), chunk_size)]

        def star_download_mseed(args):
            try:
                ret_val = utils.download_and_split_mseed_bulk(
                    *args, logger=logger)
            except ERRORS as e:
                msg = ("Client '%s': " % args[1]) + str(e)
                if "no data available" in msg.lower():
                    logger.info(msg)
                else:
                    logger.error(msg)
                return []
            return ret_val

        pool = ThreadPool(min(threads_per_client, len(station_chunks)))

        result = pool.map(
            star_download_mseed,
            [(client, client_name, restrictions.starttime,
              restrictions.endtime, chunk) for chunk in station_chunks])

        pool.close()

        filenames = itertools.chain.from_iterable(result)
        return list(filenames)

    def _parse_miniseed_filenames(self, filenames, restrictions):
        time_range = restrictions.minimum_length * (restrictions.endtime -
                                                    restrictions.starttime)
        channel_availability = []
        for filename in filenames:
            st = obspy.read(filename, format="MSEED", headonly=True)
            if restrictions.reject_channels_with_gaps and len(st) > 1:
                logger.warning("Channel %i has gap or overlap. Will be "
                               "removed." % st[0].id)
                try:
                    os.remove(filename)
                except OSError:
                    pass
                continue
            elif len(st) == 0:
                logger.error("MiniSEED file with no data detected. Should "
                             "not happen!")
                continue
            tr = st[0]
            duration = tr.stats.endtime - tr.stats.starttime
            if restrictions.minimum_length and duration < time_range:
                logger.warning("Channel %s does not satisfy the minimum "
                               "length requirement. %.2f seconds instead of "
                               "the required %.2f seconds." % (
                               tr.id, duration, time_range))
                try:
                    os.remove(filename)
                except OSError:
                    pass
                continue
            channel_availability.append(utils.ChannelAvailability(
                tr.stats.network, tr.stats.station, tr.stats.location,
                tr.stats.channel, tr.stats.starttime, tr.stats.endtime,
                filename))
        return channel_availability

    def _attach_miniseed_filenames(self, stations, restrictions, mseed_path):
        stations_to_download = []
        existing_miniseed_filenames = []

        for station in stations:
            channels = []
            for channel in station.channels:
                filename = utils.get_mseed_filename(
                    mseed_path, station.network, station.station,
                    channel.location, channel.channel)
                if not filename:
                    continue
                if os.path.exists(filename):
                    existing_miniseed_filenames.append(filename)
                    continue
                dirname = os.path.dirname(filename)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                channel.mseed_filename = filename
                channels.append(channel)
            if not channels:
                continue
            station.channels = channels
            stations_to_download.append(station)

        return stations_to_download, existing_miniseed_filenames

    def download_stationxml(self, client, client_name, stations,
                            restrictions, threads_per_client=10):

        def star_download_station(args):
            try:
                ret_val = utils.download_stationxml(*args, logger=logger)
            except ERRORS as e:
                logger.error(str(e))
                return None
            return ret_val

        pool = ThreadPool(min(threads_per_client, len(stations)))

        results = pool.map(star_download_station, [
                           (client, client_name, restrictions.starttime,
                            restrictions.endtime, s) for s in stations])
        pool.close()

        if isinstance(results[0], list):
            results = itertools.chain.from_iterable(results)
        return [_i for _i in results if _i is not None]

    def __initialize_clients(self):
        """
        Initialize all clients.
        """
        logger.info("Initializing FDSN client(s) for %s."
                    % ", ".join(self.providers))

        def _get_client(client_name):
            try:
                this_client = Client(client_name)
            except ERRORS:
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

        logger.info("Successfully initialized %i client(s): %s."
                    % (len(self._initialized_clients),
                       ", ".join(self._initialized_clients.keys())))
