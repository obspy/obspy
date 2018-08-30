# -*- coding: utf-8 -*-
"""
Time series extraction client for a database created by the
IRIS mseedindex program.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import os
from os.path import relpath
from glob import glob
import uuid
import subprocess
import copy_reg
from multiprocessing import Pool
import types
import logging
from collections import namedtuple
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.clients.filesystem.miniseed import MiniseedDataExtractor, \
    NoDataError
    

# Setup the logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Prevent propagating to higher loggers.
logger.propagate = 0
# Console log handler. By default any logs of level info and above are
# written to the console
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Add formatter
FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)


def _pickle_method(m):
    """
    Allows serializing of class and instance methods.
    """
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
copy_reg.pickle(types.MethodType, _pickle_method)


class Client(object):
    """
    Time series extraction client for a database created by the
    IRIS mseedindex program.
    """

    def __init__(self, sqlitedb, datapath_replace=None, debug=False):
        """
        Initializes the client.

        :type sqlitedb: str or
            ~obspy.clients.filesystem.tsindex.TSIndexDatabaseHandler
        :param sqlitedb: Path to sqlite tsindex database or a
            TSIndexDatabaseHandler object
        :type datapath_replace: tuple
        :param datapath_replace: A 2-value tuple, where any occurrence
            of the first value will be replaced with the second value in
            filename paths from the index.
        :type debug: bool
        :param debug: Debug flag.
        :type logger: logging.Logger
        :param logger: The logger instance to use for logging.
        """
        self.debug = debug
        if self.debug == True:
            # write debug level logs to the console
            ch.setLevel(logging.DEBUG)

        if not os.path.isfile(sqlitedb):
            raise OSError("No sqlite3 database file exists at `{}`."
                          .format(sqlitedb))

        if isinstance(sqlitedb, (str, native_str)):
            self.request_handler = TSIndexDatabaseHandler(sqlitedb,
                                                          debug=self.debug)
        elif isinstance(sqlitedb, TSIndexDatabaseHandler):
            self.request_handler = sqlitedb
        else:
            raise ValueError("sqlitedb must be a string or "
                             "TSIndexDatabaseHandler object.")

        # Create and configure the data extraction
        self.data_extractor = MiniseedDataExtractor(
                                                dp_replace=datapath_replace,
                                                debug=self.debug)

    def get_waveforms(self, network, station, location,
                      channel, starttime, endtime, merge=-1):
        """
        Query tsindex database and read miniSEED data from local
        indexed directory tree.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
            Wildcards '*' and '?' are supported.
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
            Wildcards '*' and '?' are supported.
        :type location: str
        :param location: Location code of requested data (e.g. "").
            Wildcards '*' and '?' are supported.
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
            Wildcards '*' and '?' are supported.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of requested time window.
        :type merge: int or None
        :param merge: Specifies, which merge operation should be performed
            on the stream before returning the data. Default (``-1``) means
            only a conservative cleanup merge is performed to merge seamless
            traces (e.g. when reading across day boundaries). See
            :meth:`Stream.merge(...) <obspy.core.stream.Stream.merge>` for
            details. If set to ``None`` (or ``False``) no merge operation at
            all will be performed.
        """
        query_rows = [(network, station, location,
                       channel, starttime, endtime)]
        return self._get_waveforms(query_rows, merge)

    def get_waveforms_bulk(self, query_rows, merge=-1):
        """
        Query tsindex database and read miniSEED data from local
        indexed directory tree using a bulk request.

        :type query_rows: str
        :param network: A list of tuples [(net, sta, loc, cha, starttime,
            endtime),...] containing information on what timeseries should be
            returned from the indexed archive.
            Wildcards '*' and '?' are supported.
        :param merge: Specifies, which merge operation should be performed
            on the stream before returning the data. Default (``-1``) means
            only a conservative cleanup merge is performed to merge seamless
            traces (e.g. when reading across day boundaries). See
            :meth:`Stream.merge(...) <obspy.core.stream.Stream.merge>` for
            details. If set to ``None`` (or ``False``) no merge operation at
            all will be performed.
        """
        return self._get_waveforms(query_rows, merge)

    def get_nslc(self, network, station, location,
                 channel, starttime, endtime):
        """
        Return a list of tuples [(net, sta, loc, cha),...] containing
        information on what streams are included in the tsindex database.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
            Wildcards '*' and '?' are supported.
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
            Wildcards '*' and '?' are supported.
        :type location: str
        :param location: Location code of requested data (e.g. "").
            Wildcards '*' and '?' are supported.
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
            Wildcards '*' and '?' are supported.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of requested time window.
        """
        summary_rows = self._get_summary_rows(network, station, location,
                                             channel, starttime, endtime)

        nslc_list = []
        for row in summary_rows:
            nslc = (row.network, row.station, row.location, row.channel)
            nslc_list.append(nslc)
        return nslc_list

    def get_availability_extent(self, network, station, location,
                                channel, starttime, endtime):
        """
        Return a list of tuples [(network, station, location, channel,
        earliest, latest)] containing data extent info for time series
        included in the tsindex database.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
            Wildcards '*' and '?' are supported.
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
            Wildcards '*' and '?' are supported.
        :type location: str
        :param location: Location code of requested data (e.g. "").
            Wildcards '*' and '?' are supported.
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
            Wildcards '*' and '?' are supported.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of requested time window.
        """
        summary_rows = self._get_summary_rows(network, station,
                                             location, channel,
                                             starttime, endtime)

        availability_extents = []
        for row in summary_rows:
            extent = (row.network, row.station, row.location, row.channel,
                      UTCDateTime(row.earliest), UTCDateTime(row.latest))
            availability_extents.append(extent)
        return availability_extents

    def get_availability(self, network, station, location,
                         channel, starttime, endtime,
                         include_sample_rate=False,
                         merge_overlap=False):
        """
        Return a list of tuples [(network, station, location, channel,
        starttime, endtime),...] containing data availability info for
        time series included in the tsindex database.

        If include_sample_rate=True, then a tuple containing the sample
        rate [(net, sta, loc, cha, start, end, sample_rate),...] is returned.

        If merge_overlap=True, then all time spans that overlap are merged.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
            Wildcards '*' and '?' are supported.
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
            Wildcards '*' and '?' are supported.
        :type location: str
        :param location: Location code of requested data (e.g. "").
            Wildcards '*' and '?' are supported.
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
            Wildcards '*' and '?' are supported.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of requested time window.
        :type include_sample_rate: bool
        :param include_sample_rate: If include_sample_rate=True, then
            a tuple containing the sample rate [(net, sta, loc, cha,
            start, end, sample_rate),...] is returned.
        :type mege_overlap: bool
        :param merge_overlap: If merge_overlap=True, then all time
            spans that overlap are merged.
        """

        tsindex_rows = self._get_tsindex_rows(network, station,
                                             location, channel,
                                             starttime, endtime)

        grouped_channels = {}
        for row in tsindex_rows:
            if include_sample_rate == True:
                # split on different sample rates when merging
                hash = "{}_{}_{}_{}_{}".format(row.network,
                                               row.station,
                                               row.location,
                                               row.channel,
                                               row.samplerate)
            else:
                # ignore sample rate when merging
                hash = "{}_{}_{}_{}".format(row.network,
                                               row.station,
                                               row.location,
                                               row.channel)
            timespans = self._create_timespans_list(row.timespans)
            
            if grouped_channels.get(hash) is not None:
                group = grouped_channels[hash]
                if row.samplerate not in \
                        grouped_channels[hash]["samplerates"]:
                    grouped_channels[hash]["samplerates"].append(
                                                            row.samplerate)
                grouped_channels[hash]["timespans"].extend(timespans)
            else:
                grouped_channels[hash] = {}
                grouped_channels[hash]["samplerates"] = [row.samplerate]
                grouped_channels[hash]["timespans"] = timespans

        # sort timespans
        for _, channel_group in grouped_channels.items():
            channel_group["timespans"].sort()

        # join timespans
        joined_avail_tuples = []
        for sncl, channel_group in grouped_channels.items():
            net, sta, loc, cha = sncl.split("_")[:4]
            samplerates = channel_group["samplerates"]
            timespans = channel_group["timespans"]
            avail_data = self._get_availability_from_timespans(
                                                      net,
                                                      sta,
                                                      loc,
                                                      cha,
                                                      samplerates,
                                                      include_sample_rate,
                                                      merge_overlap,
                                                      timespans
                                                      )
            # extend complete list of available data
            joined_avail_tuples.extend(avail_data)
        return joined_avail_tuples

    def get_availability_percentage(self, network, station, location,
                                    channel, starttime, endtime):
        """
        Get percentage of available data.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
        :type location: str
        :param location: Location code of requested data (e.g. "").
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of requested time window.
        :rtype: 2-tuple (float, int)
        :returns: 2-tuple of percentage of available data (``0.0`` to ``1.0``)
            and number of gaps/overlaps.
        """
        avail = self.get_availability(network, station,
                                      location, channel,
                                      starttime, endtime,
                                      merge_overlap=True)

        if not avail:
            return (0, 1)

        total_duration = endtime - starttime
        # sum up gaps in the middle
        gap_sum = 0
        gap_count = 0
        for idx, cur_ts in enumerate(avail[1:]):
            prev_ts = avail[idx]
            gap_count = gap_count + 1
            gap_sum += cur_ts[4] - prev_ts[5]

        # check if we have a gap at start or end
        earliest = min([ts[4] for ts in avail])
        latest = max([ts[5] for ts in avail])
        if earliest > starttime:
            gap_sum += earliest - starttime
            gap_count += 1
        if latest < endtime:
            gap_sum += endtime - latest
            gap_count += 1
        return (1 - (gap_sum / total_duration), gap_count)

    def has_data(self, network, station, location,
                 channel, starttime, endtime):
        """
        Return whether there is data for a specified network, station,
        location, channel, starttime, and endtime combination.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
            Wildcards '*' and '?' are supported.
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
            Wildcards '*' and '?' are supported.
        :type location: str
        :param location: Location code of requested data (e.g. "").
            Wildcards '*' and '?' are supported.
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
            Wildcards '*' and '?' are supported.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        """
        avail_percentage = self.get_availability_percentage(network,
                                                            station,
                                                            location,
                                                            channel,
                                                            starttime,
                                                            endtime)[0]
        if avail_percentage > 0:
            return True
        else:
            return False

    def _get_summary_rows(self, network, station, location, channel,
                         starttime, endtime):
        """
        Return a list of tuples [(net, sta, loc, cha, earliest, latest),...]
        containing information found in the tsindex_summary table.
        
        Information about the tsindex_summary schema may be found at:
        https://github.com/iris-edu/mseedindex/wiki/Database-Schema#suggested-time-series-summary-table # NOQA

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
            Wildcards '*' and '?' are supported.
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
            Wildcards '*' and '?' are supported.
        :type location: str
        :param location: Location code of requested data (e.g. "").
            Wildcards '*' and '?' are supported.
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
            Wildcards '*' and '?' are supported.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        """
        query_rows = [(network, station, location,
                       channel, starttime, endtime)]
        return self.request_handler._fetch_summary_rows(query_rows)

    def _get_waveforms(self, query_rows, merge=-1):
        """
        Query tsindex database and read miniSEED data from local
        indexed directory tree using a bulk request and return a
        ~obspy.core.stream.Stream object containing the requested
        timeseries data.

        :type query_rows: str
        :param network: A list of tuples [(net, sta, loc, cha, starttime,
            endtime),...] containing information on what timeseries should be
            returned from the indexed archive.
            Wildcards '*' and '?' are supported.
        :param merge: Specifies, which merge operation should be performed
            on the stream before returning the data. Default (``-1``) means
            only a conservative cleanup merge is performed to merge seamless
            traces (e.g. when reading across day boundaries). See
            :meth:`Stream.merge(...) <obspy.core.stream.Stream.merge>` for
            details. If set to ``None`` (or ``False``) no merge operation at
            all will be performed.
        """
        # Get the corresponding index DB entries
        index_rows = self.request_handler._fetch_index_rows(query_rows)

        total_bytes = 0
        src_bytes = {}

        logger.debug("Starting data return")
        st = Stream(traces=[])
        try:
            # Extract the data, writing each returned segment to the response
            for data_segment in self.data_extractor.extract_data(index_rows):
                bytes_written = data_segment.get_num_bytes()
                src_name = data_segment.get_src_name()
                if bytes_written > 0:
                    st_segment = data_segment.read_stream()
                    st += st_segment
                    total_bytes += bytes_written
                    src_bytes.setdefault(src_name, 0)
                    src_bytes[src_name] += bytes_written
        except NoDataError:
            logger.debug("No data matched selection")

        logger.debug("Wrote {} bytes".format(total_bytes))

        if merge is None or merge is False:
            pass
        else:
            st.merge(merge)
        return st

    def _get_tsindex_rows(self, network, station, location, channel,
                         starttime, endtime):
        """
        Return a list of tuples [(net, sta, loc, cha, quality... etc.),...]
        containing information found in the tsindex table.

        Information about the tsindex schema may be found at:
        https://github.com/iris-edu/mseedindex/wiki/Database-Schema#sqlite-schema-version-11 # NOQA

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
            Wildcards '*' and '?' are supported.
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
            Wildcards '*' and '?' are supported.
        :type location: str
        :param location: Location code of requested data (e.g. "").
            Wildcards '*' and '?' are supported.
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
            Wildcards '*' and '?' are supported.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        """
        query_rows = [(network, station, location,
                       channel, starttime, endtime)]
        return self.request_handler._fetch_index_rows(query_rows)
    
    def _get_availability_from_timespans(self, network, station,
                                         location, channel,
                                         samplerates,
                                         include_sample_rate,
                                         merge_overlap,
                                         timespans,
                                         _sncl_joined_avail_tuples=None):
        """
        Recurse over a list of timespans, joining adjacent timespans,
        and merging if merge_overlaps is True.

        Returns a list of tuples (network, station, location, channel,
        earliest, latest) representing available data.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
            Wildcards '*' and '?' are supported.
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
            Wildcards '*' and '?' are supported.
        :type location: str
        :param location: Location code of requested data (e.g. "").
            Wildcards '*' and '?' are supported.
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
            Wildcards '*' and '?' are supported.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of requested time window.
        :type timespans: list
        :param timespans: List of timespan tuples
        """
        if _sncl_joined_avail_tuples is None:
            _sncl_joined_avail_tuples = []

        sr = min(samplerates)
        if len(timespans) > 1:
            prev_ts = timespans.pop(0)
            cur_ts = timespans.pop(0)
            if merge_overlap is True and \
                    self._do_timespans_overlap(prev_ts, cur_ts) is True:
                # merge if overlapping timespans and merge_overlap
                # option is set to true
                earliest_tuple = min([prev_ts, cur_ts],
                                     key=lambda t: t.earliest)
                latest_tuple = max([prev_ts, cur_ts],
                                   key=lambda t: t.latest)
                merged_ts = self._create_timespan(prev_ts.earliest,
                                                  cur_ts.latest)
                timespans.insert(0, merged_ts)
                return self._get_availability_from_timespans(
                                                network, station,
                                                location, channel,
                                                samplerates,
                                                include_sample_rate,
                                                merge_overlap,
                                                timespans,
                                                _sncl_joined_avail_tuples)
            elif self._are_timespans_adjacent(prev_ts, cur_ts, sr, 0.5):
                # merge if timespans are next to each other within
                # a 0.5 sample tolerance
                merged_ts = self._create_timespan(prev_ts.earliest,
                                                  cur_ts.latest)
                timespans.insert(0, merged_ts)
                return self._get_availability_from_timespans(
                                                network, station,
                                                location, channel,
                                                samplerates,
                                                include_sample_rate,
                                                merge_overlap,
                                                timespans,
                                                _sncl_joined_avail_tuples)
            else:
                # timespan shouldn't be merged so add to list
                avail_tuple = self._create_avail_tuple(
                                          network, station,
                                          location, channel,
                                          prev_ts.earliest,
                                          prev_ts.latest,
                                          sr=sr if include_sample_rate \
                                                else None)
                _sncl_joined_avail_tuples.append(avail_tuple)
                timespans.insert(0, cur_ts)
                return self._get_availability_from_timespans(
                                                network, station,
                                                location, channel,
                                                samplerates,
                                                include_sample_rate,
                                                merge_overlap,
                                                timespans,
                                                _sncl_joined_avail_tuples)
        else:
            # no other timespans to merge with
            cur_ts = timespans.pop(0)
            avail_tuple = self._create_avail_tuple(
                                          network, station,
                                          location, channel,
                                          cur_ts.earliest,
                                          cur_ts.latest,
                                          sr=sr if include_sample_rate \
                                                else None)
            _sncl_joined_avail_tuples.append(avail_tuple)
        return _sncl_joined_avail_tuples

    def _are_timespans_adjacent(self, ts1, ts2, sample_rate, tolerance=0.5):
        """
        Checks whether or not two time span named tuples
        (e.g. NameTuple(earliest, latest)) are adjacent within
        a given tolerance

        :type ts1: namedtuple
        :param ts1: Earliest timespan.
        :type ts2: namedtuple
        :param ts2: Latest timespan.
        :type sample_rate: int
        :param sample_rate: Sensor sample rate
        :type tolerance: float
        :param tolerance: Tolerance to determine whether a adjacent
            timespan should be merged.
        """
        # @40Hz sample period = 0.025
        sample_period = 1. / float(sample_rate)
        expected_next = ts1.latest + sample_period
        # @40Hz tolerance = 0.0125
        tolerance_amount = (tolerance*sample_period)
        actual_next = ts2.earliest
        if expected_next + tolerance_amount > actual_next and \
           expected_next - tolerance_amount < actual_next:
            return True
        else:
            return False

    def _do_timespans_overlap(self, ts1, ts2):
        """
        Checks whether or not two time span named tuples
        (e.g. NameTuple(earliest, latest)) intersect with
        one another.

        :type ts1: namedtuple
        :param ts1: Earliest timespan.
        :type ts2: namedtuple
        :param ts2: Latest timespan.
        """
        if ts1.earliest <= ts2.latest and \
           ts1.latest >= ts2.earliest:
            return True
        else:
            return False

    def _create_avail_tuple(self, network, station, location, channel,
                            earliest, latest, sr=None):
        """
        Returns a tuple representing available waveform data.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
            Wildcards '*' and '?' are supported.
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
            Wildcards '*' and '?' are supported.
        :type location: str
        :param location: Location code of requested data (e.g. "").
            Wildcards '*' and '?' are supported.
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
            Wildcards '*' and '?' are supported.
        :type earliest: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param earliest: Earliest date of timespan.
        :type latest: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param latest: Latest date of timespan.
        """
        if sr is not None:
            avail_record = (network, station, location, channel,
                            UTCDateTime(float(earliest)),
                            UTCDateTime(float(latest)), sr)
        else:
            avail_record = (network, station, location, channel,
                            UTCDateTime(float(earliest)),
                            UTCDateTime(float(latest)))
        return avail_record

    def _create_timespan(self, earliest, latest):
        """
        Create a TimeSpan named tuple object given a earliest and latest date.

        :param earliest: Earliest date of timespan.
        :param latest: Latest date of timespan.
        """
        TimeSpan = namedtuple('TimeSpan',
                              ['earliest', 'latest'])
        return TimeSpan(earliest, latest)

    def _create_timespans_list(self, raw_timespans):
        """
        Given a timespans string from the database, return
        a list of named tuples.

        :type raw_timespans: str
        :param raw_timespans: timespans field from tsindex database table.
        """
        timespans = []
        unparsed_timespans = \
            raw_timespans.replace("[", "").replace("]", "").split(",")
        for t in unparsed_timespans:
            earliest, latest = t.split(":")
            ts = self._create_timespan(float(earliest), float(latest))
            timespans.append(ts)
        return timespans


class Indexer(object):
    """
    Build an index for miniSEED data using IRIS's mseedindex program.
    Recursively search for files matching `filename_pattern` starting
    from `root_path` and run `index_cmd` for each target file found that
    is not already in the index. After all new files are indexed a summary
    table is generated with the extents of each time series.
    """

    def __init__(self, root_path, sqlitedb="timeseries.sqlite",
                 leap_seconds_file=None, index_cmd='mseedindex',
                 bulk_params=None, filename_pattern='*',
                 parallel=5, debug=False):
        """
        Initializes the Indexer.

        :type root_path: str
        :param root_path: Root path to the directory structure to index.
        :type sqlitedb: str or
            ~obspy.clients.filesystem.tsindex.TSIndexDatabaseHandler
        :param sqlitedb: Path to sqlite tsindex database or a
            TSIndexDatabaseHandler object. A database will be created
            if one does not already exists at the specified path.
        :type leap_seconds_file: str
        :param leap_seconds_file: Path to leap seconds file. See the
            `mseedindex wiki <https://github.com/iris-edu/mseedindex/blob/"
            "master/doc/mseedindex.md#leap-second-list-file>` "
            "for more information.
        :type index_cmd: str
        :param index_cmd: Command to be run for each target file found that
            is not already in the index
        :type bulk_params: dict
        :param bulk_params: Dictionary of options to pass to index_cmd.
        :type filename_pattern: str        
        :param filename_pattern: 
        :type parallel: int
        :param parallel: Max number of index_cmd instances to run in parallel.
            By default a max of 5 parallel process are run.
        :type debug: bool
        :param debug: Debug flag. Sets logging level to debug.
        """
        self.debug = debug
        if self.debug == True:
            # write debug level logs to the console
            ch.setLevel(logging.DEBUG)

        self.root_path = root_path
        self.index_cmd = index_cmd
        if bulk_params is None:
            bulk_params = {}
        self.bulk_params = bulk_params
        self.filename_pattern = filename_pattern
        self.parallel = parallel
        self.sqlitedb = sqlitedb
        self.leap_seconds_file = leap_seconds_file

        if isinstance(self.sqlitedb, (str, native_str)):
            self.request_handler = TSIndexDatabaseHandler(self.sqlitedb,
                                                          debug=self.debug)
        elif isinstance(self.sqlitedb, TSIndexDatabaseHandler):
            self.request_handler = self.sqlitedb
        else:
            raise ValueError("sqlitedb must be a string or "
                             "TSIndexDatabaseHandler object.")

    def run(self, build_summary=True, relative_paths=False, reindex=False):
        """
        Execute the file discovery and indexing.

        :type build_summary: bool
        :param build_summary: By default, a summary table is (re)generated
            containing the extents for each time series in the index. This can
            be turned off by setting `build_summary` to False.
        :type relative_paths: bool
        :param relative_paths: By default, the absolute path to each file is
            stored in the index. If `relative_paths` is True, the file paths
            will be relative to the `root_path`.
        type reindex: bool
        :param reindex: By default, files are not indexed that are already in
            the index and have not been modified.  The `reindex` option can be
            set to True to force a re-indexing of all files regardless.
        """
        self.is_mseedindex_installed()
        self.request_handler._init_database_for_indexing()
        file_paths = self.build_file_list(relative_paths)
        
        if self.leap_seconds_file is not None and \
                not os.path.isfile(self.leap_seconds_file):
            raise OSError("No leap seconds file exists at `{}`."
                          .format(self.leap_seconds_file))

        # always keep the original file paths as specified. absolute and
        # relative paths are determined in the build_file_list method
        self.bulk_params["-kp"] = None
        if self.bulk_params.get("-table") is None:
            # set db table to write to
            self.bulk_params["-table"] = self.request_handler.tsindex_table
        if self.bulk_params.get("-sqlite") is None:
            # set path to sqlite database
            self.bulk_params['-sqlite'] = self.sqlitedb

        # run mseedindex on each file in parallel
        pool = Pool(processes=self.parallel)     
        for file_name in file_paths:
            logger.debug("Indexing file '{}'.".format(file_name))
            proc = pool.apply_async(Indexer._run_index_command,
                                    args=(self.index_cmd,
                                          self.root_path,
                                          file_name,
                                          self.bulk_params))
            # If the remote call raised an exception
            # then that exception will be reraised by get()
            proc.get()
        pool.close()
        pool.join()

        if build_summary is True:
            self.request_handler.build_tsindex_summary()

    def build_file_list(self, relative_paths=False, reindex=False):
        """
        Create a list of absolute paths to all files under root_path that match
        the filename_pattern.
        
        :type relative_paths: bool
        :param relative_paths: By default, the absolute path to each file is
            stored in the index. If `relative_paths` is True, the file paths
            will be relative to the `root_path`.
        """
        file_list = [y for x in os.walk(self.root_path)
                    for y in glob(os.path.join(x[0], self.filename_pattern))]

        result = []
        if relative_paths is True:
            for abs_path in file_list:
                result.append(relpath(abs_path, self.root_path))
        else:
            result = file_list
        if not result:
            raise OSError("No files matching filename pattern '{}' "
                          "were found under root path '{}'."
                          .format(self.filename_pattern, self.root_path))
        return result

    def is_mseedindex_installed(self):
        """
        Checks if mseedindex is installed.
        """
        try:
            subprocess.call(["mseedindex", "-V"])
        except OSError:
            raise OSError(
                    "Required program mseedindex is not installed. Install "
                    "mseedindex at https://github.com/iris-edu/mseedindex/.")

    @classmethod
    def _run_index_command(cls, index_cmd, root_path, file_name, bulk_params):
        """
        Execute a command to perform indexing.
        
        :type index_cmd: str
        :param index_cmd: Name of indexing command to execute. Defaults to
            `mseedindex`. 
        :type file_name: str
        :param file_name: Name of file to index.
        :type bulk_params: dict
        :param bulk_params: Dictionary of options to pass to index_cmd.
        """
        try:
            cmd = [index_cmd]
            for option, value in bulk_params.iteritems():
                params = [option, value]
                cmd.extend(params)
            cmd.append(file_name)
            # boolean options have a value of None
            cmd = [c for c in cmd if c is not None]
            proc = subprocess.Popen(cmd, cwd=root_path)
            proc.wait()
        except Exception as err:
            msg = ("Error running command `{}` - {}"
                   .format(index_cmd, err))
            raise OSError(msg)


class TSIndexDatabaseHandler(object):

    def __init__(self, sqlitedb, tsindex_table="tsindex",
                 tsindex_summary_table="tsindex_summary",
                 debug=False):
        """
        Main query interface to timeseries index database.

        :type sqlitedb: str or
            ~obspy.clients.filesystem.tsindex.TSIndexDatabaseHandler
        :param sqlitedb: Path to sqlite tsindex database or a
            TSIndexDatabaseHandler object
        :type tsindex_table: str
        :param tsindex_table: Name of timeseries index table
        :type tsindex_summary_table: str
        :param tsindex_summary_table: Name of timeseries index summary table
        :type debug: bool
        :param debug: Debug flag.
        """
        self.debug = debug
        if self.debug == True:
            # write debug level logs to the console
            ch.setLevel(logging.DEBUG)

        self.sqlitedb = sqlitedb
        self.tsindex_table = tsindex_table
        self.tsindex_summary_table = tsindex_summary_table

        self.db_path = "sqlite:///{}".format(sqlitedb)
        self.engine = create_engine(self.db_path, poolclass=QueuePool)
        
    def build_tsindex_summary(self, connection=None, temporary=False):
        if connection is None:
            connection = self.engine.connect()
        # test if tsindex table exists
        if not self.engine.dialect.has_table(self.engine, 'tsindex'):
            raise ValueError("No tsindex table '{}' exists in database '{}'."
                             .format(self.tsindex_table, self.sqlitedb))
        if temporary is False:
            connection.execute("DROP TABLE IF EXISTS {};"
                               .format(self.tsindex_summary_table))
        connection.execute(
            "CREATE {0} TABLE {1} AS "
            "SELECT network, station, location, channel, "
            "  min(starttime) AS earliest, max(endtime) AS latest, "
            "  datetime('now') as updt "
            "FROM {2} "
            "GROUP BY 1,2,3,4;"
            .format("TEMPORARY" if temporary is True else "",
                    self.tsindex_summary_table,
                    self.tsindex_table)
        )
        if connection is None:
            connection.close()

    def _fetch_index_rows(self, query_rows=[], bulk_params={}):
        '''
        Fetch index rows matching specified request
        :type query_rows: list
        :param query_rows: List of tuples containing (net,sta,loc,chan,start,
            end). By default everything is selected.
        :type bulk_params: dict
        :param bulk_params: Dict of bulk parameters (e.g. quality)
            Request elements may contain '?' and '*' wildcards.  The start and
            end elements can be a single '*' if not a date-time string.
            Return rows as list of named tuples containing:
            (network, station, location, channel, quality, starttime, endtime,
            samplerate, filename, byteoffset, bytes, hash, timeindex,
            timespans, timerates, format, filemodtime, updated, scanned,
            requeststart, requestend)
        '''
        query_rows = self._clean_query_rows(query_rows)

        my_uuid = uuid.uuid4().hex
        request_table = "request_%s" % my_uuid
        try:
            connection = self.engine.connect()
        except Exception as err:
            raise ValueError(str(err))

        logger.debug("Opening SQLite database for "
                     "index rows: %s" % self.sqlitedb)

        # Store temporary table(s) in memory
        try:
            connection.execute("PRAGMA temp_store=MEMORY")
        except Exception as err:
            raise ValueError(str(err))

        # Create temporary table and load request
        try:
            connection.execute("CREATE TEMPORARY TABLE {0} "
                               "(network TEXT, station TEXT, location TEXT, "
                               "channel TEXT, starttime TEXT, endtime TEXT) "
                               .format(request_table))

            for req in query_rows:
                connection.execute("INSERT INTO {0} (network,station,location,"
                                   "channel,starttime,endtime) "
                                   "VALUES (?,?,?,?,?,?) "
                                   .format(request_table), (req[0], req[1],
                                                            req[2], req[3],
                                                            req[4], req[5]))
        except Exception as err:
            raise ValueError(str(err))

        result = connection.execute("SELECT count(*) FROM sqlite_master "
                                    "WHERE type='table' and name='{0}'"
                                    .format(self.tsindex_summary_table))

        summary_present = result.fetchone()[0]
        wildcards = False
        for req in query_rows:
            for field in req:
                if '*' in str(field) or '?' in str(field):
                    wildcards = True
                    break

        if wildcards:
            # Resolve wildcards using summary if present to:
            # a) resolve wildcards, allows use of '=' operator and table index
            # b) reduce index table search to channels that are known included
            if summary_present:
                self._resolve_request(connection, request_table)
                wildcards = False
            # Replace wildcarded starttime and endtime with extreme date-times
            else:
                connection.execute("UPDATE {0} "
                                   "SET starttime='0000-00-00T00:00:00' "
                                   "WHERE starttime='*'".format(request_table))
                connection.execute("UPDATE {0} "
                                   "SET endtime='5000-00-00T00:00:00' "
                                   "WHERE endtime='*'".format(request_table))

        # Fetch final results by joining resolved and index table
        try:
            sql = ("SELECT DISTINCT ts.network,ts.station,ts.location,"
                   "ts.channel,ts.quality,ts.starttime,ts.endtime,"
                   "ts.samplerate,ts.filename,ts.byteoffset,ts.bytes,ts.hash, "
                   "ts.timeindex,ts.timespans,ts.timerates,ts.format,"
                   "ts.filemodtime,ts.updated,ts.scanned,r.starttime,"
                   "r.endtime "
                   "FROM {0} ts, {1} r "
                   "WHERE "
                   "  ts.network {2} r.network "
                   "  AND ts.station {2} r.station "
                   "  AND ts.location {2} r.location "
                   "  AND ts.channel {2} r.channel "
                   "  AND ts.starttime <= r.endtime "
                   "  AND ts.endtime >= r.starttime "
                   "ORDER BY ts.network, ts.station, ts.location, "
                    "  ts.channel, ts.starttime, ts.endtime"
                   .format(self.tsindex_table,
                           request_table,
                           "GLOB" if wildcards else "="))

            # Add quality identifer criteria
            if 'quality' in bulk_params and \
                    bulk_params['quality'] in ('D', 'R', 'Q'):
                sql = sql + " AND quality = '{0}' ".format(
                                                        bulk_params['quality'])
            result = connection.execute(sql)
        except Exception as err:
            raise ValueError(str(err))

        # Map raw tuples to named tuples for clear referencing
        NamedRow = namedtuple('NamedRow',
                              ['network', 'station', 'location', 'channel',
                               'quality', 'starttime', 'endtime', 'samplerate',
                               'filename', 'byteoffset', 'bytes', 'hash',
                               'timeindex', 'timespans', 'timerates', 'format',
                               'filemodtime', 'updated', 'scanned',
                               'requeststart', 'requestend'])

        index_rows = []
        for row in result:
            index_rows.append(NamedRow(*row))

        logger.debug("Fetched %d index rows" % len(index_rows))

        connection.execute("DROP TABLE {0}".format(request_table))
        connection.close()

        return index_rows

    def _fetch_summary_rows(self, query_rows):
        '''
        Fetch summary rows matching specified request. A temporary tsindex
        summary table is created if one does not exists

        Returns rows as list of named tuples containing:
        (network,station,location,channel,earliest,latest,updated)

        :type query_rows: list
        :param: List of tuples containing (net,sta,loc,chan,start,end)
            Request elements may contain '?' and '*' wildcards. The start and
            end elements can be a single '*' if not a date-time string.
        '''
        query_rows = self._clean_query_rows(query_rows)

        try:
            connection = self.engine.connect()
        except Exception as err:
            raise Exception(err)
        
        logger.debug("Opening sqlite3 database for "
                     "summary rows: %s" % self.sqlitedb)

        # Store temporary table(s) in memory
        try:
            connection.execute("PRAGMA temp_store=MEMORY")
        except Exception as err:
            raise ValueError(str(err))
        
        result = connection.execute("SELECT count(*) FROM sqlite_master "
                                    "WHERE type='table' and name='{0}'"
                                    .format(self.tsindex_summary_table))

        summary_present = result.fetchone()[0]
        if not summary_present:
            logger.warning("No tsindex_summary table found! A temporary "
                           "tsindex_summary table will be created.")
            self.build_tsindex_summary(connection=connection,
                                       temporary=True)
            logger.info(
                       "For improved performance create a permanent "
                       "tsindex_summary table by running the "
                       "`~obspy.clients.filesystem.tsindex."
                       "TSIndexDatabaseHandler.build_tsindex_summary()` "
                       "instance method.")

        summary_rows = []
        my_uuid = uuid.uuid4().hex
        request_table = "request_%s" % my_uuid

        # Create temporary table and load request
        try:
            connection.execute("CREATE TEMPORARY TABLE {0}"
                               " (network TEXT, station TEXT,"
                               " location TEXT, channel TEXT,"
                               " starttime TEXT, endtime TEXT)"
                               .format(request_table))
            for req in query_rows:
                connection.execute("INSERT INTO {0} (network,station,"
                                   "  location,channel, starttime, endtime) "
                                   "VALUES (?,?,?,?,?,?) "
                                   .format(request_table),
                                   req)
        except Exception as err:
            raise ValueError(str(err))

        result_perm = connection.execute("SELECT count(*) "
                                         "FROM sqlite_master "
                                         "WHERE type='table' "
                                         "and name='{0}'"
                                         .format(self.tsindex_summary_table))
        result_temp = connection.execute("SELECT count(*) "
                                         "FROM sqlite_temp_master "
                                         "WHERE type='table' "
                                         "and name='{0}'"
                                         .format(self.tsindex_summary_table))
        
        summary_present = result_perm.fetchone()[0] or \
                                result_temp.fetchone()[0]

        if summary_present:
            # Select summary rows by joining with summary table
            try:
                sql = ("SELECT DISTINCT s.network,s.station,s.location,"
                       "s.channel,s.earliest,s.latest,s.updt "
                       "FROM {0} s, {1} r "
                       "WHERE "
                       "  (r.starttime='*' OR r.starttime <= s.latest) "
                       "  AND (r.endtime='*' OR r.endtime >= s.earliest) "
                       "  AND (r.network='*' OR s.network GLOB r.network) "
                       "  AND (r.station='*' OR s.station GLOB r.station) "
                       "  AND (r.location='*' OR s.location GLOB r.location) "
                       "  AND (r.channel='*' OR s.channel GLOB r.channel) "
                       "ORDER BY s.network, s.station, s.location, "
                       "s.channel, s.earliest, s.latest".
                       format(self.tsindex_summary_table, request_table))
                result = connection.execute(sql)
            except Exception as err:
                raise ValueError(str(err))

            # Map raw tuples to named tuples for clear referencing
            NamedRow = namedtuple('NamedRow',
                                  ['network', 'station', 'location', 'channel',
                                   'earliest', 'latest', 'updated'])

            summary_rows = []
            for row in result:
                summary_rows.append(NamedRow(*row))

            logger.debug("Fetched %d summary rows" % len(summary_rows))

            connection.execute("DROP TABLE {0}".format(request_table))
        
        connection.close()

        return summary_rows

    def _resolve_request(self, connection, request_table):
        '''
        Resolve request table by expanding wildcards using summary.

        :type connection: sqlalchemy.engine.base.Connection
        :param: database connection to resolve request with
        :type request_table: str
        :param request_table: table to resolve
            Resolve any '?' and '*' wildcards in the specified request table.
            The original table is renamed, rebuilt with a join to summary
            and then original table is then removed.
        '''

        request_table_orig = request_table + "_orig"

        # Rename request table
        try:
            connection.execute("ALTER TABLE {0} "
                               "RENAME TO {1}".format(request_table,
                                                      request_table_orig))
        except Exception as err:
            raise ValueError(str(err))

        # Create resolved request table by joining with summary
        try:
            sql = ("CREATE TEMPORARY TABLE {0} "
                   "(network TEXT, station TEXT, location TEXT, channel TEXT, "
                   "starttime TEXT, endtime TEXT) ".format(request_table))
            connection.execute(sql)

            sql = ("INSERT INTO {0} (network, station, location, "
                   "channel, starttime, endtime) "
                   "SELECT s.network, s.station, s.location, s.channel,"
                   "CASE WHEN r.starttime='*' "
                   "THEN s.earliest ELSE r.starttime END,"
                   "CASE WHEN r.endtime='*' "
                   "THEN s.latest ELSE r.endtime END "
                   "FROM {1} s, {2} r "
                   "WHERE "
                   "  (r.starttime='*' OR r.starttime <= s.latest) "
                   "  AND (r.endtime='*' OR r.endtime >= s.earliest) "
                   "  AND (r.network='*' OR s.network GLOB r.network) "
                   "  AND (r.station='*' OR s.station GLOB r.station) "
                   "  AND (r.location='*' OR s.location GLOB r.location) "
                   "  AND (r.channel='*' OR s.channel GLOB r.channel) ".
                   format(request_table,
                          self.tsindex_summary_table,
                          request_table_orig))
            connection.execute(sql)

        except Exception as err:
            raise ValueError(str(err))

        resolvedrows = connection.execute("SELECT COUNT(*) FROM {0}"
                                          .format(request_table)).fetchone()[0]

        logger.debug("Resolved request with "
                     "summary into %d rows" % resolvedrows)

        connection.execute("DROP TABLE {0}".format(request_table_orig))

        return resolvedrows

    def _create_query_row(self, network, station, location,
                         channel, starttime, endtime):
        """
        Returns a tuple (network, station, location, channel, starttime,
        endtime) with elements that have been formatted to match database
        entries. This allows for accurate comparisons when querying the
        database.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
            Wildcards '*' and '?' are supported.
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
            Wildcards '*' and '?' are supported.
        :type location: str
        :param location: Location code of requested data (e.g. "").
            Wildcards '*' and '?' are supported.
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
            Wildcards '*' and '?' are supported.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        """
        # Replace "--" location ID request alias with true empty value
        if location == "--":
            location = ""
        if isinstance(starttime, UTCDateTime):
            starttime = starttime.isoformat()
        if isinstance(endtime, UTCDateTime):
            endtime = endtime.isoformat()
        return (network, station, location, channel, starttime, endtime)

    def _clean_query_rows(self, query_rows):
        """
        Reformats query rows to match what is stored in the database.
        
        :type query_rows: list
        :param query_rows: List of tuples containing (network, station,
            location, channel, starttime, endtime).
        """
        if query_rows == []:
            # if an empty list is supplied then select everything
            select_all_query = self._create_query_row('*', '*', '*',
                                                      '*', '*', '*')
            query_rows = [select_all_query]
        else:
            # perform some formatting on the query rows to ensure that they
            # query the database properly.
            for i, qr in enumerate(query_rows):
                query_rows[i] = self._create_query_row(*qr)
        
        flat_query_rows = []
        # flatten query rows
        for req in query_rows:
            networks = req[0].replace(" ", "").split(",")
            stations = req[1].replace(" ", "").split(",")
            locations = req[2].replace(" ", "").split(",")
            channels = req[3].replace(" ", "").split(",")
            starttime = req[4]
            endtime = req[5]
            for net in networks:
                for sta in stations:
                    for loc in locations:
                        for cha in channels:
                            qr = self._create_query_row(net, sta, loc, cha,
                                                        starttime, endtime)
                            flat_query_rows.append(qr)
        return flat_query_rows

    def _init_database_for_indexing(self):
        """
        Setup a sqlite3 database for indexing.
        """
        try:
            logger.debug('Setting up sqlite3 database at %s' % self.sqlitedb)
            # setup the sqlite database
            connection = self.engine.connect()
            # https://www.sqlite.org/foreignkeys.html
            connection.execute('PRAGMA foreign_keys = ON')
            # as used by mseedindex
            connection.execute('PRAGMA case_sensitive_like = ON')
            # enable Write-Ahead Log for better concurrency support
            connection.execute('PRAGMA journal_mode=WAL')
            connection.close()
        except Exception as e:
            raise OSError("Failed to setup sqlite3 database for indexing.")
