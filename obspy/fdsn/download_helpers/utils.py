#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions required for the download helpers.

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
    import itertools
    from urllib.error import HTTPError, URLError

import collections
import copy
import fnmatch
import os
from lxml import etree
import numpy as np
from scipy.spatial import cKDTree
from socket import timeout as SocketTimeout
import tempfile
import time
from uuid import uuid4
import obspy
import warnings

from obspy.core.util.base import NamedTemporaryFile
from obspy.fdsn.client import FDSNException
from obspy.mseed.util import getRecordInformation


# Different types of errors that can happen when downloading data via the
# FDSN clients.
ERRORS = (FDSNException, HTTPError, URLError, SocketTimeout)


# mean earth radius in meter as defined by the International Union of
# Geodesy and Geophysics.
EARTH_RADIUS = 6371009


ChannelAvailability = collections.namedtuple(
    "ChannelAvailability",
    ["network", "station", "location", "channel", "starttime", "endtime",
     "filename"])


class Station(object):
    __slots__ = ["network", "station", "latitude", "longitude",
                 "elevation_in_m", "channels", "stationxml_filename"]

    def __init__(self, network, station, latitude, longitude,
                 elevation_in_m, channels=None, stationxml_filename=None):
        self.network = network
        self.station = station
        self.latitude = latitude
        self.longitude = longitude
        self.elevation_in_m = elevation_in_m
        self.channels = channels if channels else []
        self.stationxml_filename = stationxml_filename

    def __repr__(self):
        return "Station(%s, %s, %s, %s, %s, %s, %s)" % (
            self.network.__repr__(),
            self.station.__repr__(),
            self.latitude.__repr__(),
            self.longitude.__repr__(),
            self.elevation_in_m.__repr__(),
            self.channels.__repr__(),
            self.stationxml_filename.__repr__())

    def __eq__(self, other):
        try:
            for key in self.__slots__:
                if getattr(self, key) != getattr(other, key):
                    return False
        except AttributeError as e:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(str(getattr(self, i)) for i in self.__slots__))


class Channel(object):
    __slots__ = ["location", "channel", "mseed_filename"]

    def __init__(self, location, channel, mseed_filename=None):
        self.location = location
        self.channel = channel
        self.mseed_filename = mseed_filename

    def __repr__(self):
        return "Channel(%s, %s, %s)" % (
            self.location.__repr__(),
            self.channel.__repr__(),
            self.mseed_filename.__repr__())

    def __eq__(self, other):
        try:
            for key in self.__slots__:
                if getattr(self, key) != getattr(other, key):
                    return False
        except AttributeError:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(getattr(self, i) for i in self.__slots__))


def format_report(report):
    """
    Pretty print the report returned from the download() function of the
    download helper.
    """
    print("\nAttempted to acquire data from %i clients." % len(report))
    for info in report:
        stationxmls = []
        mseeds = []
        for station in info["data"]:
            stationxmls.append(station.stationxml_filename)
            mseeds.extend([i.mseed_filename for i in station.channels])
        filesize = sum(os.path.getsize(i) for i in (mseeds + stationxmls))
        filesize /= (1024.0 * 1024.0)
        print("\tClient %10s - %4i StationXML files | %5i MiniSEED files "
              "| Total Size: %.2f MB" %
              ('%s' % info["client"], len(stationxmls), len(mseeds),
               filesize))


def is_in_list_of_channel_availability(network, station, location, channel,
                                       starttime, endtime, availabilities):
    """
    Helper function checking if a given channel is in a list of
    ChannelAvailability tuples.

    :param network: The network code.
    :param station: The station code.
    :param location: The location code.
    :param channel: The channel code.
    :param starttime: The starttime of the data.
    :param endtime: The endtime of the data
    :param availabilities: List of ChannelAvailability objects.
    """
    for avail in availabilities:
        if (avail.network == network) and \
                (avail.station == station) and \
                (avail.location == location) and \
                (avail.channel == channel) and \
                (avail.starttime <= starttime) and \
                (avail.endtime >= endtime):
            return True
    return False


def attach_stationxml_filenames(stations, restrictions, stationxml_path,
                                logger):
    # Attach filenames to the StationXML files and get a list of
    # already existing StationXML files.
    stations_to_download = []
    existing_stationxml_channels = []

    for stat in copy.deepcopy(stations):
        filename = get_stationxml_filename(
            stationxml_path, stat.network, stat.station, stat.channels)
        # If it returns a dictionary like objects
        if isinstance(filename, collections.Container):
            pass
        # If the StationXML file already exists, make sure it
        # contains all the necessary information. Otherwise
        # delete it and it will be downloaded again in the
        # following.
        if os.path.exists(filename):
            contents = get_stationxml_contents(filename)
            all_channels_good = True
            for chan in stat.channels:
                if is_in_list_of_channel_availability(
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
                safe_delete(filename)
            else:
                existing_stationxml_channels.extend(contents)
                continue
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        stat.stationxml_filename = filename
        stations_to_download.append(stat)

    return {
        "stations_to_download": stations_to_download,
        "existing_stationxml_contents": existing_stationxml_channels
    }


def attach_miniseed_filenames(stations, mseed_path):
    """
    Attach filenames to the channels in the stations list splitting the
    dataset into already existing channels and new channels.
    """
    stations_to_download = []
    existing_miniseed_filenames = []
    ignored_channel_count = 0

    stations = copy.deepcopy(stations)

    for station in stations:
        channels = []
        for channel in station.channels:
            filename = get_mseed_filename(
                mseed_path, station.network, station.station,
                channel.location, channel.channel)
            # If True, the channel will essentially be ignored.
            if filename is True:
                ignored_channel_count += 1
                continue
            # If the path exists, it will not be downloaded again.
            elif os.path.exists(filename):
                existing_miniseed_filenames.append(filename)
                continue
            # Make sure the directories exist.
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            channel.mseed_filename = filename
            channels.append(channel)
        if not channels:
            continue
        station.channels = channels
        stations_to_download.append(station)

    return {
        "stations_to_download": stations_to_download,
        "existing_miniseed_filenames": existing_miniseed_filenames,
        "ignored_channel_count": ignored_channel_count
    }


def filter_duplicate_and_discarded_stations(
        existing_stations, discarded_station_ids, new_stations):
    """
    :param existing_stations: A set of :class:`~.Station` object. detailing
        already existing stations.
    :param discarded_station_ids: A set of tuples denoting discarded
        station ids, e.g. station ids that have been discarded due to some
        reason.
    :param new_stations: A set or list of new :class:`~.Station` objects
        that will be filtered.
    :return: A set of filtered :class:`~.Station` objects.
    """
    existing_stations = [(_i.network, _i.station) for _i in existing_stations]
    invalid_station_ids = set(existing_stations).union(discarded_station_ids)

    return list(itertools.filterfalse(
        lambda x: (x.network, x.station) in invalid_station_ids,
        new_stations))


def filter_stations_with_channel_list(stations, channels):
    station_channels = {}
    get_station = lambda x: "%s.%s" % (x.network, x.station)
    for s, c in itertools.groupby(sorted(channels, key=get_station),
                                  get_station):
        station_channels[s] = [(_i.location, _i.channel) for _i in c]

    final_stations = []
    for station in stations:
        station_id = "%s.%s" % (station.network, station.station)
        if station_id not in station_channels:
            continue
        station_chan = station_channels[station_id]
        good_channels = []
        for channel in station.channels:
            if (channel.location, channel.channel) not in station_chan:
                continue
            good_channels.append(channel)
        if good_channels:
            station = copy.deepcopy(station)
            station.channels = good_channels
            final_stations.append(station)
    return final_stations


def download_stationxml(client, client_name, starttime, endtime, station,
                        logger):
    bulk = [(station.network, station.station, _i.location, _i.channel,
             starttime, endtime) for _i in station.channels]
    try:
        client.get_stations_bulk(bulk, level="response",
                                 filename=station.stationxml_filename)
    except Exception as e:
        logger.info("Failed to downloaded StationXML from %s for station "
                    "%s.%s." %
                    (client_name, station.network, station.station))
        return None
    logger.info("Client '%s' - Successfully downloaded '%s'." %
                (client_name, station.stationxml_filename))
    return station.stationxml_filename


def download_and_split_mseed_bulk(client, client_name, chunks, logger):
    """
    Downloads the channels of a list of stations in bulk, saves it in the
    temp folder and splits it at the record level to obtain the final
    MiniSEED files.

    :param client: An active client instance.
    :param client_name: The name of the client instance used for logging
        purposes.
    :param chunks: A list of tuples, each denoting a single MiniSEED chunk.
        Each chunk is a tuple of network, station, location, channel,
        starttime, endtime, and desired filename.
    :param logger: An active logger instance.
    """
    # Create a dictionary of channel ids, each containing the
    filenames = collections.defaultdict(list)
    for chunk in chunks:
        filenames[tuple(chunk[:4])].append({
            "starttime": chunk[4],
            "endtime": chunk[5],
            "filename": chunk[6],
            "current_latest_endtime": None,
            "sequence_number": None})

    sequence_number = [0]

    def get_filename(starttime, endtime, c):
        # Make two passes. First find all candidates.
        candidates = [
            _i for _i in c if
            (_i["starttime"] <= starttime <= _i["endtime"]) or
            (_i["starttime"] <= endtime <= _i["endtime"])]
        if not candidates:
            return None

        # If more then one candidate, apply some heuristics to find the
        # correct time interval. The main complication arises when the same
        # record is downloaded twice as it overlaps into two different
        # requested time intervals.
        if len(candidates) == 2:
            candidates.sort(key=lambda x: x["starttime"])
            first, second = candidates
            # Make sure the assumptions about the type of overlap are correct.
            if starttime > first["endtime"] or endtime < second["starttime"]:
                raise NotImplementedError
            # It must either be the last record of the first, or the first
            # record of the second candidate.
            if first["sequence_number"] is None and \
                    second["sequence_number"] is None:
                candidates = [second]
            # Unlikely to happen. Only if nothing but the very last record
            # of the first interval was available and the second interval
            # was first in the file.
            elif first["sequence_number"] is None:
                candidates = [first]
            # This is fairly likely and requires and additional check with
            # the latest time in the first interval.
            elif second["sequence_number"] is None:
                if starttime <= first["current_latest_endtime"]:
                    pass
                else:
                    pass
            # Neither are None. Just use the one with the higher sequence
            # number.
            else:
                if first["sequence_number"] > second["sequence_number"]:
                    candidates = [first]
                else:
                    candidates = [second]
        elif len(candidates) >= 2:
            raise NotImplementedError

        # Finally found the correct chunk
        ret_val = candidates[0]
        # Increment sequence number and make sure the current chunk is aware
        # of it.
        sequence_number[0] += 1
        ret_val["sequence_number"] = sequence_number[0]
        # Also write the time of the last chunk.
        ce = ret_val["current_latest_endtime"]
        if not ce or endtime > ce:
            ret_val["current_latest_endtime"] = endtime
        return ret_val["filename"]

    # Only the filename is not needed for the actual data request.
    bulk = [_i[:-1] for _i in chunks]

    # Save first to a temporary file, then cut the file into seperate files.
    temp_filename = NamedTemporaryFile().name

    try:
        client.get_waveforms_bulk(bulk, filename=temp_filename)

        open_files = {}
        # If that succeeds, split the old file into multiple new ones.
        file_size = os.path.getsize(temp_filename)

        with open(temp_filename, "rb") as fh:
            try:
                while True:
                    if fh.tell() >= (file_size - 256):
                        break
                    info = getRecordInformation(fh)
                    channel_id = (info["network"], info["station"],
                                  info["location"], info["channel"])

                    # Sometimes the services return something nobody wants...
                    if channel_id not in filenames:
                        fh.read(info["record_length"])
                        continue
                    # Get the best matching filename.
                    filename = get_filename(
                        starttime=info["starttime"], endtime=info["endtime"],
                        c=filenames[channel_id])
                    # Again sometimes there are time ranges nobody asked for...
                    if filename is None:
                        fh.read(info["record_length"])
                        continue
                    if filename not in open_files:
                        open_files[filename] = open(filename, "wb")
                    open_files[filename].write(fh.read(info["record_length"]))
            finally:
                for f in open_files.values():
                    try:
                        f.close()
                    except:
                        pass
    finally:
        try:
            os.remove(temp_filename)
        except:
            pass
    logger.info("Client '%s' - Successfully downloaded %i channels (of %i)" % (
        client_name, len(open_files), len(bulk)))
    return open_files.keys()


class SphericalNearestNeighbour(object):
    """
    Spherical nearest neighbour queries using scipy's fast kd-tree
    implementation.
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


def filter_channel_priority(channels, key, priorities=None):
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
            if fnmatch.fnmatch(getattr(channel, key), pattern):
                filtered_channels.append(channel)
                continue
    return filtered_channels


def safe_delete(filename):
    """
    "Safely" delete a file. It really just checks if it exists and is a file.

    :param filename: The filename to delete.
    :return:
    """
    if not os.path.exists(filename):
        return
    elif not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)
    os.remove(filename)


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
        nns = list(itertools.filterfalse(lambda x: most_common in x, nns))

    # Remove these indices.
    return set([_i[1] for _i in itertools.filterfalse(
                lambda x: x[0] in indexes_to_remove,
                enumerate(stations))])


def filter_based_on_interstation_distance(
        existing_stations, new_stations, reliable_new_stations,
        minimum_distance_in_m=0):
    """
    Filter two lists of stations, successively adding each station in one
    list to the stations in the list of existing stations satisfying the
    required minimum inter-station distances. If minimum distance in meter
    is 0, it will just merge both lists and return.

    The duplicate ids between new and existing stations can be assumed to
    already have been removed.

    Returns a dictionary containing two sets, one with the accepted stations
    and one with the rejected stations.

    :param reliable_new_stations: Determines if the contribution of the new
        stations will also be taken into account when calculating the
        minimum inter-station distance. If True, the check for each new
        station will be performed by successively adding stations. Otherwise
        it will be performed only against the existing stations.
    """
    # Shallow copies.
    existing_stations = set(copy.copy(existing_stations))
    new_stations = set(copy.copy(new_stations))

    # If no requirement given, just merge
    if not minimum_distance_in_m:
        return {
            "accepted_stations": existing_stations.intersection(new_stations),
            "rejected_stations": set()
        }

    # If no existing stations yet, just make sure the minimum inner station
    # distances are satisfied.
    if not existing_stations:
        if reliable_new_stations:
            accepted_stations = filter_stations(new_stations,
                                                minimum_distance_in_m)
            return {
                "accepted_stations": accepted_stations,
                "rejected_stations": new_stations.difference(accepted_stations)
            }
        else:
            return {
                "accepted_stations": new_stations,
                "rejected_stations": set()
            }

    accepted_stations = set()
    rejected_stations = set()

    test_set = copy.copy(existing_stations)

    for station in new_stations:
        kd_tree = SphericalNearestNeighbour(test_set)
        neighbours = kd_tree.query([station])[0][0]
        if np.isinf(neighbours[0]):
            from IPython.core.debugger import Tracer; Tracer(colors="Linux")()
            continue
        min_distance = neighbours[0]
        if min_distance < minimum_distance_in_m:
            rejected_stations.add(station)
            continue
        accepted_stations.add(station)
        if reliable_new_stations:
            test_set.add(station)

    return {
        "accepted_stations": accepted_stations,
        "rejected_stations": rejected_stations,
    }


def download_waveforms_and_stations(client, client_name, station_list,
                                    starttime, endtime, temporary_directory):
    # Create the bulk download list. This is the same for waveform and
    # station bulk downloading.
    (((s.network, s.station, c.location, c.channel, starttime, endtime)
      for c in s.channels) for s in station_list)


def does_file_contain_all_channels(filename, station, logger=None):
    """
    Test whether the StationXML file located at filename contains
    information about all channels in station.

    :type filename: str
    :param filename: Filename of the StationXML file to check.
    :type station: :class:`~obspy.fdsn.download_helpers.utils.Station`
    :param station: Station object containing channel information.
    :type logger: :class:`logging.Logger`
    :param logger: Logger to log exceptions to.
    """
    try:
        available_channels = get_stationxml_contents(filename)
    except etree.XMLSyntaxError:
        msg = "'%s' is not a valid XML file. Will be overwritten." % filename
        if logger is not None:
            logger.warning(msg)
        else:
            warnings.warn(msg)


def get_stationxml_contents(filename):
    """
    Really fast way to get all channels with a response in a StationXML
    file. Sometimes it is too expensive to parse the full file with ObsPy.
    This is usually orders of magnitudes faster.

    :param filename: The path to the file.
    :returns: list of ChannelAvailability objects.
    """
    # Small state machine.
    network, station, location, channel, starttime, endtime = [None] * 6

    ns = "http://www.fdsn.org/xml/station/1"
    network_tag = "{%s}Network" % ns
    station_tag = "{%s}Station" % ns
    channel_tag = "{%s}Channel" % ns
    response_tag = "{%s}Response" % ns

    context = etree.iterparse(filename, events=("start", ),
                              tag=(network_tag, station_tag, channel_tag,
                                   response_tag))

    channels = []
    for event, elem in context:
        if elem.tag == channel_tag:
            channel = elem.get('code')
            location = elem.get('locationCode').strip()
            starttime = obspy.UTCDateTime(elem.get('startDate'))
            end_date = elem.get('endDate')
            if end_date is not None:
                endtime = obspy.UTCDateTime(elem.get('endDate'))
            else:
                endtime = obspy.UTCDateTime(2599, 1, 1)
        elif elem.tag == response_tag:
            channels.append(ChannelAvailability(
                network, station, location, channel, starttime, endtime,
                filename))
        elif elem.tag == station_tag:
            station = elem.get('code')
            location, channel, starttime, endtime = \
                None, None, None, None
        elif elem.tag == network_tag:
            network = elem.get('code')
            station, location, channel, starttime, endtime = \
                None, None, None, None, None
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return channels


def get_stationxml_filename(str_or_fct, network, station, channels):
    """
    Helper function getting the filename of a stationxml file.

    The rule are simple, if it is a function, network and station are passed
    as arguments and the resulting string is returned. Furthermore a list of
    channels is passed.

    If it is a string, and it contains ``"{network}"``, and ``"{station}"``
    formatting specifiers, ``str.format()`` is called.

    Otherwise it is considered to be a folder name and the resulting
    filename will be ``"FOLDER_NAME/NET.STA.xml"``
    """
    # Call if possible.
    if callable(str_or_fct):
        path = str_or_fct(network, station, channels)
    # Check if its a format template.
    elif ("{network}" in str_or_fct) and ("{station}" in str_or_fct):
        path = str_or_fct.format(network=network, station=station)
    # Otherwise assume its a path.
    else:
        path = os.path.join(str_or_fct, "{network}.{station}.xml".format(
            network=network, station=station))
    if isinstance(path, (str, bytes)):
        return path
    elif isinstance(path, collections.Container):
        if "available_channels" not in path or \
                "missing_channels" not in path or \
                "filename" not in path:
            raise ValueError(
                "The dictionary returned by the stationxml filename function "
                "must contain the following keys: 'available_channels', "
                "'missing_channels', and 'filename'.")
        if not isinstance(path["available_channels"], collections.Iterable) or\
                not isinstance(path["missing_channels"],
                               collections.Iterable) or \
                not isinstance(path["filename"], (str, bytes)):
            raise ValueError("Return types must be two lists of channels and "
                             "a string for the filename.")
        return path

    else:
        raise TypeError("'%s' is not a filepath." % str(path))


def get_mseed_filename(str_or_fct, network, station, location, channel,
                       starttime, endtime):
    """
    Helper function getting the filename of a MiniSEED file.

    The rule are simple, if it is a function, network, station, location,
    channel, starttime, and endtime are passed as arguments and the resulting
    string is returned.

    If it is a string, and it contains ``"{network}"``,  ``"{station}"``,
    ``"{location}"``, ``"{channel}"``, ``"{starttime}"``, and ``"{endtime}"``
    formatting specifiers, ``str.format()`` is called.

    Otherwise it is considered to be a folder name and the resulting
    filename will be
    ``"FOLDER_NAME/NET.STA.LOC.CHAN__STARTTIME__ENDTIME.mseed"``

    In the last two cases, the times will be formatted with
    ``"%Y-%m-%dT%H-%M-%SZ"``.
    """
    strftime = "%Y-%m-%dT%H-%M-%SZ"
    if callable(str_or_fct):
        path = str_or_fct(network, station, location, channel, starttime,
                          endtime)
    elif ("{network}" in str_or_fct) and ("{station}" in str_or_fct) and \
            ("{location}" in str_or_fct) and ("{channel}" in str_or_fct) and \
            ("{starttime}" in str_or_fct) and ("{endtime}" in str_or_fct):
        path = str_or_fct.format(
            network=network, station=station, location=location,
            channel=channel, starttime=starttime.strftime(strftime),
            endtime=endtime.strftime(strftime))
    else:
        path = os.path.join(
            str_or_fct,
            "{network}.{station}.{location}.{channel}__{s}__{e}.mseed".format(
                network=network, station=station, location=location,
                channel=channel, s=starttime.strftime(strftime),
                e=endtime.strftime(strftime)))

    if path is True:
        return path
    elif not isinstance(path, (str, bytes)):
        raise TypeError("'%s' is not a filepath." % str(path))
    return path
