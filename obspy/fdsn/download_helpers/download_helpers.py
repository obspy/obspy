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
            try:
                info = utils.get_availability_from_client(
                    client, client_name, restrictions, domain, logger)
            except ERRORS as e:
                msg = "Availability for client '%s': %s" % (
                    client_name,  str(e))
                if "no data available" in msg.lower():
                    logger.info(msg)
                else:
                    logger.error(msg)
                continue

            availability = info["availability"]
            if not availability:
                logger.info("Availability for client '%s': No suitable "
                            "data found." % client_name)
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
                new_stations = all_stations - existing_stations
                if not new_stations:
                    continue

                # "Filter" the station list to get a list of channels that
                # are actually required and a list of existing filenames.
                # The existing filenames will be parsed to download
                # StationXML files for them in case they do not yet exist.
                mseed_stations, existing_miniseed_filenames = \
                    self._attach_miniseed_filenames(
                        stations=new_stations, restrictions=restrictions,
                        mseed_path=mseed_path)

                if not mseed_stations and not existing_miniseed_filenames:
                    logger.info("Nothing to be downloaded for client %s." %
                                client_name)
                    continue

                # Download the missing channels and get a list of filenames
                # that just got downloaded.
                if mseed_stations:
                    downloaded_miniseed_filenames = self.download_mseed(
                        client, client_name, mseed_stations, restrictions,
                        chunk_size=chunk_size,
                        threads_per_client=threads_per_client)
                else:
                    downloaded_miniseed_filenames = []

                # Parse the just downloaded and existing MiniSEED files and
                # make a list of stations that require StationXML files.
                miniseed_channels = self._parse_miniseed_filenames(
                    downloaded_miniseed_filenames +
                    existing_miniseed_filenames, restrictions)
                # Stations will be list of Stations with Channels that all
                # need a StationXML file.
                stations = utils.filter_stations_with_channel_list(
                    new_stations, miniseed_channels)

                # Attach filenames to the StationXML files and get a list of
                # already existing StationXML files.
                stations_to_download = []
                existing_stationxml_channels = []
                for stat in stations:
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
                            os.remove(filename)
                        else:
                            existing_stationxml_channels.append(contents)
                            continue
                    dirname = os.path.dirname(filename)
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    stat.stationxml_filename = filename
                    stations_to_download.append(stat)

                # Now download the station information for the downloaded
                # stations.
                del stations[0]
                downloaded_stationxml = self.download_stationxml(
                    client, client_name, stations,
                    restrictions, threads_per_client)

                dict_downloaded_mseed = collections.defaultdict(list)
                dict_downloaded_xmls = collections.defaultdict(list)

                mseed_tspan = collections.namedtuple(
                    "mseed_tspan", ["starttime", "endtime", "filename"])
                xml_tspan = collections.namedtuple(
                    "xml_tspan", ["starttime", "endtime"])
                for chans in miniseed_channels:
                    dict_downloaded_mseed['%s.%s.%s.%s' % (
                        chans.network, chans.station, chans.location,
                        chans.channel)].append(mseed_tspan(
                            chans.starttime, chans.endtime, chans.filename))
                for chans in downloaded_stationxml:
                    dict_downloaded_xmls['%s.%s.%s.%s' % (
                        chans.network, chans.station, chans.location,
                        chans.channel)].append(xml_tspan(
                            chans.starttime, chans.endtime))

                for mseed_chan in dict_downloaded_mseed.keys():
                    if not mseed_chan in dict_downloaded_xmls.keys():
                        logger.warning(
                            "Stationxml for %s has not been downloaded, "
                            "the mseed file is removed!" % mseed_chan)
                        os.remove(
                            dict_downloaded_mseed[mseed_chan][0].filename)
                        continue

                    xml_time_spans = dict_downloaded_xmls[mseed_chan]
                    for t in xml_time_spans:
                        for mseed_range in dict_downloaded_mseed[mseed_chan]:
                            if t.starttime <= mseed_range.starttime and \
                                    t.endtime >= mseed_range.endtime:
                                break
                        else:
                            pass

                    else:
                        logger.warning(
                            "Stationxml for %s does not cover the whole time "
                            "span of the mseed file, the mseed file is "
                            "removed!" % mseed_chan)
                        os.remove(
                            dict_downloaded_mseed[mseed_chan][0].filename)
                        continue

                # Remove waveforms that did not succeed in having available
                # stationxml files.
                #self.delete_extraneous_waveform_files()
            else:
                # If it is not reliable, e.g. the client does not have the
                # "includeavailability" or "matchtimeseries" flags,
                # then first download everything and filter later!
                raise NotImplementedError

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

        all_channels = []
        for filename in results:
            if not filename:
                continue
            all_channels.extend(utils.get_stationxml_contents(filename))
        return all_channels

    def __initialize_clients(self):
        """
        Initialize all clients.
        """
        logger.info("Initializing FDSN clients for %s."
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

        logger.info("Successfully initialized %i clients: %s."
                    % (len(self._initialized_clients),
                       ", ".join(self._initialized_clients.keys())))
