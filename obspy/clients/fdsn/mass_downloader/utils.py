# -*- coding: utf-8 -*-
"""
Utility functions required for the download helpers.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014-2105
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import collections
import fnmatch
import itertools
import os
from http.client import HTTPException
from socket import timeout as socket_timeout
from urllib.error import HTTPError, URLError

import numpy as np
from lxml import etree
from scipy.spatial import cKDTree

import obspy
from obspy.core.util.base import NamedTemporaryFile
from obspy.clients.fdsn.client import FDSNException
from obspy.io.mseed.util import get_record_information


# Different types of errors that can happen when downloading data via the
# FDSN clients.
ERRORS = (ConnectionError, FDSNException, HTTPError, HTTPException, URLError,
          socket_timeout, )

# mean earth radius in meter as defined by the International Union of
# Geodesy and Geophysics. Used for the spherical kd-tree.
EARTH_RADIUS = 6371009


ChannelAvailability = collections.namedtuple(
    "ChannelAvailability",
    ["network", "station", "location", "channel", "starttime", "endtime",
     "filename"])


def download_stationxml(client, client_name, bulk, filename, logger):
    """
    Download all channels for a station in the already prepared bulk list.

    :param client: An active client instance.
    :param client_name: The name of the client mainly used for logging
        purposes.
    :param bulk: An already prepared bulk download list for all channels and
        time intervals for the given station. All items in there are assumed
        to come from the same station.
    :param filename: The filename to download to.
    :param logger: The logger instance to use for logging.

    :returns: A tuple with the network and station id and the filename upon
        success
    """
    network = bulk[0][0]
    station = bulk[0][1]
    try:
        client.get_stations_bulk(bulk=bulk, level="response",
                                 filename=filename)
    except Exception:
        logger.info("Failed to download StationXML from '%s' for station "
                    "'%s.%s'." % (client_name, network, station))
        return None
    logger.info("Client '%s' - Successfully downloaded '%s'." %
                (client_name, filename))
    return ((network, station), filename)


def download_and_split_mseed_bulk(client, client_name, chunks, logger):
    """
    Downloads the channels of a list of stations in bulk, saves it to a
    temporary folder and splits it at the record level to obtain the final
    MiniSEED files.

    The big advantage of this approach is that it does not mess with the
    MiniSEED files at all. Each record, including all blockettes, will end
    up in the final files as they are served from the data centers.

    :param client: An active client instance.
    :param client_name: The name of the client instance used for logging
        purposes.
    :param chunks: A list of tuples, each denoting a single MiniSEED chunk.
        Each chunk is a tuple of network, station, location, channel,
        starttime, endtime, and desired filename.
    :param logger: An active logger instance.
    """
    # Create a dictionary of channel ids, each containing a list of
    # intervals, each of which will end up in a separate file.
    filenames = collections.defaultdict(list)
    for chunk in chunks:
        candidate = {
            "starttime": chunk[4],
            "endtime": chunk[5],
            "filename": chunk[6],
            "current_latest_endtime": None,
            "sequence_number": None}
        # Should not be necessary if chunks have been deduplicated before but
        # better safe than sorry.
        if candidate in filenames[tuple(chunk[:4])]:
            continue
        filenames[tuple(chunk[:4])].append(candidate)

    sequence_number = [0]

    def get_filename(starttime, endtime, c):
        """
        Helper function finding the corresponding filename in all filenames.

        :param starttime: The start time of the record.
        :param endtime: The end time of the record.
        :param c: A list of candidates.
        """
        # Make two passes. First find all candidates. This assumes that a
        # record cannot be larger than a single desired time interval. This
        # is probably always given except if somebody wants to download
        # files split into 1 second intervals...
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

            # This is fairly likely and requires an additional check with
            # the latest time in the first interval.
            elif second["sequence_number"] is None:
                if starttime <= first["current_latest_endtime"]:
                    candidates = [second]
                else:
                    candidates = [first]

            # Neither are None. Just use the one with the higher sequence
            # number. This probably does not happen. If it happens something
            # else is a bit strange.
            else:
                if first["sequence_number"] > second["sequence_number"]:
                    candidates = [first]
                else:
                    candidates = [second]
        elif len(candidates) >= 2:
            raise NotImplementedError(
                "Please contact the developers. candidates: %s" %
                str(candidates))

        # Finally found the correct chunk
        ret_val = candidates[0]

        # Increment sequence number and make sure the current chunk is aware
        # of it.
        sequence_number[0] += 1
        ret_val["sequence_number"] = sequence_number[0]

        # Also write the time of the last chunk to it if necessary.
        ce = ret_val["current_latest_endtime"]
        if not ce or endtime > ce:
            ret_val["current_latest_endtime"] = endtime

        return ret_val["filename"]

    # Only the filename is not needed for the actual data request.
    bulk = [list(_i[:-1]) for _i in chunks]
    original_bulk_length = len(bulk)

    # Merge adjacent bulk-request for continuous downloads. This is a bit
    # redundant after splitting it up before, but eases the logic in the
    # other parts and puts less strain on the data centers' FDSN
    # implementation. It furthermore avoid the repeated download of records
    # that are part of two neighbouring time intervals.
    bulk_channels = collections.defaultdict(list)
    for b in bulk:
        bulk_channels[(b[0], b[1], b[2], b[3])].append(b)

    # Merge them.
    for key, value in bulk_channels.items():
        # Sort based on starttime.
        value = sorted(value, key=lambda x: x[4])
        # Merge adjacent.
        cur_bulk = value[0:1]
        for b in value[1:]:
            # Random threshold of 2 seconds. Reasonable for most real world
            # cases.
            if b[4] <= cur_bulk[-1][5] + 2:
                cur_bulk[-1][5] = b[5]
                continue
            cur_bulk.append(b)
        bulk_channels[key] = cur_bulk
    bulk = list(itertools.chain.from_iterable(bulk_channels.values()))

    # Save first to a temporary file, then cut the file into separate files.
    with NamedTemporaryFile() as tf:
        temp_filename = tf.name
        open_files = {}

        client.get_waveforms_bulk(bulk, filename=temp_filename)
        # If that succeeds, split the old file into multiple new ones.
        file_size = os.path.getsize(temp_filename)

        with open(temp_filename, "rb") as fh:
            try:
                while True:
                    if fh.tell() >= (file_size - 256):
                        break
                    info = get_record_information(fh)
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
                    except Exception:
                        pass
    logger.info("Client '%s' - Successfully downloaded %i channels (of %i)" % (
        client_name, len(open_files), original_bulk_length))
    return sorted(open_files.keys())


class SphericalNearestNeighbour(object):
    """
    Spherical nearest neighbour queries using scipy's fast kd-tree
    implementation.
    """
    def __init__(self, data):
        cart_data = self.spherical2cartesian(data)
        self.data = data
        self.kd_tree = cKDTree(data=cart_data, leafsize=10)

    def query(self, points):
        points = self.spherical2cartesian(points)
        d, i = self.kd_tree.query(points)

        # Filter NaNs. Happens when not enough points are available.
        m = np.isfinite(d)
        return d[m], i[m]

    def query_pairs(self, maximum_distance):
        return self.kd_tree.query_pairs(maximum_distance)

    @staticmethod
    def spherical2cartesian(data):
        """
        Converts a list of :class:`~obspy.clients.fdsn.download_status.Station`
        objects to an array of shape(len(list), 3) containing x/y/z in meters.
        """
        # Create three arrays containing lat/lng/radius.
        shape = len(data)
        lat = np.array([_i.latitude for _i in data], dtype=np.float64)
        lon = np.array([_i.longitude for _i in data], dtype=np.float64)
        r = np.array([EARTH_RADIUS for _ in data], dtype=np.float64)
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
    of a channel is "HH[ZNE]" to avoid getting e.g. rotated components.
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
    "Safely" delete a file. It really just checks if it exists and if it is a
    file.

    :param filename: The filename to delete.
    """
    if not os.path.exists(filename):
        return
    elif not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)
    try:
        os.remove(filename)
    except Exception as e:
        raise ValueError("Could not delete '%s' because: %s" % (filename,
                                                                str(e)))


def get_stationxml_contents(filename):
    """
    Really fast way to get all channels with a response in a StationXML
    file. Sometimes it is too expensive to parse the full file with ObsPy.
    This is usually orders of magnitudes faster.

    Will only returns channels that contain response information.

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

    try:
        context = etree.iterparse(filename, events=("start", ),
                                  tag=(network_tag, station_tag, channel_tag,
                                       response_tag))
    except TypeError:  # pragma: no cover
        # Some old lxml version have a way less powerful iterparse()
        # function. Fall back to parsing with ObsPy.
        return _get_stationxml_contents_slow(filename)

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


def _get_stationxml_contents_slow(filename):
    """
    Slow variant for get_stationxml_contents() for old lxml versions.

    :param filename: The path to the file.
    :returns: list of ChannelAvailability objects.
    """
    channels = []
    inv = obspy.read_inventory(filename)
    for net in inv:
        for sta in net:
            for cha in sta:
                if not cha.response:
                    continue
                channels.append(ChannelAvailability(
                    net.code, sta.code, cha.location_code, cha.code,
                    cha.start_date, cha.end_date
                    if cha.end_date else obspy.UTCDateTime(2599, 1, 1),
                    filename))
    return channels


def get_stationxml_filename(str_or_fct, network, station, channels,
                            starttime, endtime):
    """
    Helper function getting the filename of a StationXML file.

    :param str_or_fct: The string or function to be evaluated.
    :type str_or_fct: callable or str
    :param network: The network code.
    :type network: str
    :param station: The station code.
    :type station: str
    :param channels: The channels. Each channel is a tuple of two strings:
        location code and channel code.
    :type channels: list[tuple]
    :param starttime: The start time.
    :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param endtime: The end time.
    :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`

    The rules are simple, if it is a function, then the network, station,
    channels, start time, and end time parameters are passed to it.

    If it is a string, and it contains ``"{network}"``, and ``"{station}"``
    formatting specifiers, ``str.format()`` is called.

    Otherwise it is considered to be a folder name and the resulting
    filename will be ``"FOLDER_NAME/NET.STA.xml"``

    This function will either return a string or a dictionary with three
    keys: ``"available_channels"``, ``"missing_channels"``, and ``"filename"``.
    """
    # Call if possible.
    if callable(str_or_fct):
        path = str_or_fct(network, station, channels, starttime, endtime)
    # Check if it's a format template.
    elif ("{network}" in str_or_fct) and ("{station}" in str_or_fct):
        path = str_or_fct.format(network=network, station=station)
    # Otherwise assume it's a path.
    else:
        path = os.path.join(str_or_fct, "{network}.{station}.xml".format(
            network=network, station=station))

    # If it is just a filename, return that.
    if isinstance(path, (str, bytes)):
        return path

    elif isinstance(path, collections.abc.Container):
        if "available_channels" not in path or \
                "missing_channels" not in path or \
                "filename" not in path:
            raise ValueError(
                "The dictionary returned by the stationxml filename function "
                "must contain the following keys: 'available_channels', "
                "'missing_channels', and 'filename'.")
        if not isinstance(path["available_channels"],
                          collections.abc.Iterable) or\
                not isinstance(path["missing_channels"],
                               collections.abc.Iterable) or \
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

    The rules are simple, if it is a function, then the network, station,
    location, channel, start time, and end time parameters are passed to it
    and the resulting string is returned. If the return values is ``True``,
    the particular time interval will be ignored.

    If it is a string, and it contains ``"{network}"``,  ``"{station}"``,
    ``"{location}"``, ``"{channel}"``, ``"{starttime}"``, and ``"{endtime}"``
    formatting specifiers, ``str.format()`` is called.

    Otherwise it is considered to be a folder name and the resulting
    filename will be
    ``"FOLDER_NAME/NET.STA.LOC.CHAN__STARTTIME__ENDTIME.mseed"``

    In the last two cases, the times will be formatted with
    ``"%Y%m%dT%H%M%SZ"``.
    """
    strftime = "%Y%m%dT%H%M%SZ"
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
        return True
    elif not isinstance(path, (str, bytes)):
        raise TypeError("'%s' is not a filepath." % str(path))
    return path
