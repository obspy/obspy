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

from sqlite3 import OperationalError
import uuid
from collections import namedtuple
from sqlalchemy import create_engine
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.clients.filesystem.miniseed import MiniseedDataExtractor, \
    NoDataError
from __builtin__ import True


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
        """
        if isinstance(sqlitedb, (str, native_str)):
            self.request_handler = TSIndexDatabaseHandler(sqlitedb,
                                                          debug=debug)
        elif isinstance(sqlitedb, TSIndexDatabaseHandler):
            self.request_handler = sqlitedb
        else:
            raise ValueError("sqlitedb must be a string or "
                             "TSIndexDatabaseHandler object.")
            
        # Create and configure the data extraction
        self.data_extractor = MiniseedDataExtractor(dp_replace=
                                                        datapath_replace,
                                                    debug=debug)
        self.debug = debug

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
        query_row = TSIndexDatabaseHandler.create_query_row(network, station,
                                                           location, channel,
                                                           starttime, endtime)
        query_rows = [query_row]
        return self._get_waveforms(query_rows, merge)
    
    def get_waveforms_bulk(self, query_rows, merge=-1):
        """
        Query tsindex database and read miniSEED data from local
        indexed directory tree using a bulk request
        
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
        cleaned_query_rows = []
        # perform any necessary cleanup on query rows
        for query_row in query_rows:
            cleaned_query_row = TSIndexDatabaseHandler.create_query_row(
                                                            *query_row)
            cleaned_query_rows.append(cleaned_query_row)
        return self._get_waveforms(cleaned_query_rows, merge)
    
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
        index_rows = self.request_handler.fetch_index_rows(query_rows)

        total_bytes = 0
        src_bytes = {}

        if self.debug is True:
            print("Starting data return")
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
            if self.debug is True:
                print("No data matched selection")

        if self.debug is True:
            print("Wrote {} bytes".format(total_bytes))

        if merge is None or merge is False:
            pass
        else:
            st.merge(merge)
        return st
    
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
        summary_rows = self.get_summary_rows(network, station, location,
                                              channel, starttime, endtime)

        nslc_list = []
        for row in summary_rows:
            nslc = (row.network, row.station, row.location, row.channel)
            nslc_list.append(nslc)
        return nslc_list
            
    
    def get_availability_extent(self, network, station, location,
                                channel, starttime, endtime):
        """
        Return a list of tuples [(net, sta, loc, cha, earliest, latest)]
        containing data extent info for time series included in the
        tsindex database.

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
        summary_rows = self.get_summary_rows(network, station,
                                              location, channel,
                                              UTCDateTime(starttime), 
                                              UTCDateTime(endtime))
        
        availability_extents = []
        for row in summary_rows:
            extent = (row.network, row.station, row.location, row.channel,
                      UTCDateTime(row.earliest), UTCDateTime(row.latest))
            availability_extents.append(extent)
        return availability_extents
        
    def _are_timespans_adjacent(self, ts1, ts2, sample_rate, tolerance=0.5):
        """
        Checks whether or not two time span named tuples
        (e.g. NameTuple(earliest, latest)) are adjacent within
        a given tolerance
        """
        sample_period = 1. / float(sample_rate) # @40Hz sample period = 0.025
        expected_next = ts1.latest + sample_period
        tolerance_amount = (tolerance*sample_period) # @40Hz tolerance = 0.0125
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
        one another
        """
        if ts1.earliest <= ts2.latest and \
           ts1.latest >= ts2.earliest:
            return True
        else:
            return False

    def _create_avail_tuple(self, net, sta, loc, cha,
                            earliest, latest, sr=None):
        """
        Returns a tuple representing available waveform data
        """
        if sr is not None:
            avail_record = (net, sta, loc, cha,
                            UTCDateTime(float(earliest)),
                            UTCDateTime(float(latest)), sr)
        else:
            avail_record = (net, sta, loc, cha,
                            UTCDateTime(float(earliest)),
                            UTCDateTime(float(latest)))
        return avail_record
    
    def _create_timespan(self, earliest, latest):
        TimeSpan = namedtuple('TimeSpan',
                              ['earliest', 'latest'])
        return TimeSpan(earliest, latest)
    
    def _create_timespans_list(self, raw_timespans):
        """
        Given a timespans string from the database, return 
        a python list of named tuples
        """
        timespans = []
        unparsed_timespans = raw_timespans.replace("[","") \
                                .replace("]","").split(",")
        for t in unparsed_timespans:
            earliest, latest = t.split(":")
            ts = self._create_timespan(float(earliest), float(latest))
            timespans.append(ts)
        return timespans

    def get_availability(self, network, station, location,
                         channel, starttime, endtime,
                         include_sample_rate=False,
                         merge_overlap=False):
        """
        Return a list of tuples [(net, sta, loc, cha, start, end),...]
        containing data availability info for time series included in
        the tsindex database.
        
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
        
        def _join(timespans, sncl_joined_avail_tuples=[]):
            """
            Recurse over a list of timespans, joining adjacent timespans,
            and merging if merge_overlaps is True.
            """
            if len(timespans) > 1:
                prev_ts = timespans.pop(0)
                cur_ts = timespans.pop(0)
                if merge_overlap == True and \
                    self._do_timespans_overlap(prev_ts, cur_ts) == True:
                    # merge if overlapping timespans and merge_overlap
                    # option is set to true
                    earliest_tuple = min([prev_ts, cur_ts],
                                       key = lambda t: t.earliest)
                    latest_tuple = max([prev_ts, cur_ts],
                                       key = lambda t: t.latest)
                    avail_tuple = self._create_avail_tuple(
                                                  net, sta, loc, cha,
                                                  earliest_tuple.earliest,
                                                  latest_tuple.latest)
                    ts = self._create_timespan(avail_tuple[4], avail_tuple[5])
                    timespans.insert(0, ts)
                    return _join(timespans, sncl_joined_avail_tuples)
                elif self._are_timespans_adjacent(prev_ts, cur_ts, sr, 0.5):
                    # merge if timespans are next to each other within
                    # a 0.5 sample tolerance
                    avail_tuple = self._create_avail_tuple(
                                                        net, sta, loc, cha,
                                                        prev_ts.earliest,
                                                        cur_ts.latest)
                    ts = self._create_timespan(avail_tuple[4], avail_tuple[5])
                    timespans.insert(0, ts)
                    return _join(timespans, sncl_joined_avail_tuples)
                else:
                    # timespan shouldn't be merged so add to list
                    avail_tuple = self._create_avail_tuple(
                                                  net, sta, loc, cha,
                                                  prev_ts.earliest,
                                                  prev_ts.latest)
                    sncl_joined_avail_tuples.append(avail_tuple)
                    timespans.insert(0, cur_ts)
                    return _join(timespans, sncl_joined_avail_tuples)
            else:
                # no other timespans to merge with
                cur_ts = timespans[0]
                avail_tuple = self._create_avail_tuple(
                                              net, sta, loc, cha,
                                              cur_ts.earliest,
                                              cur_ts.latest)
                sncl_joined_avail_tuples.append(avail_tuple)
            return sncl_joined_avail_tuples
        
        tsindex_rows = self.get_tsindex_rows(network, station,
                                             location, channel,
                                             starttime, endtime)
        grouped_timespans = {} # list of timespans grouped by NSLC and SR
        for row in tsindex_rows:
            hash = "{}_{}_{}_{}_{}".format(row.network,
                                           row.station,
                                           row.location,
                                           row.channel,
                                           row.samplerate)
            timespans = self._create_timespans_list(row.timespans)
            if grouped_timespans.get(hash) is not None:
                grouped_timespans[hash].extend(timespans)
            else:
                grouped_timespans[hash] = timespans

        # sort timespans
        for _, timespans in grouped_timespans.items():
            timespans.sort()

        # join timespans
        joined_avail_tuples = []
        for sncl, timespans in grouped_timespans.items():
            net, sta, loc, cha, sr = sncl.split("_")
            sncl_joined_avail_tuples = _join(timespans)
            # extend complete list
            joined_avail_tuples.extend(sncl_joined_avail_tuples)    
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

        if starttime >= endtime:
            msg = ("'endtime' must be after 'starttime'.")
            raise ValueError(msg)
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
    
    def get_summary_rows(self, network, station, location, channel,
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
        query_row = TSIndexDatabaseHandler.create_query_row(network, station,
                                                           location, channel,
                                                           starttime, endtime)
        query_rows = [query_row]
        return self.request_handler.fetch_summary_rows(query_rows)
    
    def get_tsindex_rows(self, network, station, location, channel,
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
        query_row = TSIndexDatabaseHandler.create_query_row(network, station,
                                                           location, channel,
                                                           starttime, endtime)
        query_rows = [query_row]
        return self.request_handler.fetch_index_rows(query_rows)


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
        self.sqlitedb = sqlitedb
        self.tsindex_table = tsindex_table
        self.tsindex_summary_table = tsindex_summary_table
        
        self.db_path = "sqlite:///{}".format(sqlitedb)
        self.engine = create_engine(self.db_path, encoding=native_str('utf-8'),
                                    convert_unicode=True)
        self.debug = debug

    @classmethod
    def create_query_row(cls, network, station, location,
                         channel, starttime, endtime):
        # Replace "--" location ID request alias with true empty value
        if location == "--":
            location = ""
        if isinstance(starttime, UTCDateTime):
            starttime = starttime.isoformat()
        if isinstance(endtime, UTCDateTime):
            endtime = endtime.isoformat()
        return (network, station, location, channel, starttime, endtime)
    
    def fetch_index_rows(self, query_rows, bulk_params={}):
        '''
        Fetch index rows matching specified request
        `query_rows`: List of tuples containing (net,sta,loc,chan,start,end)
        `bulk_params`: Dict of bulk parameters (e.g. quality, minsegmentlength)
        Request elements may contain '?' and '*' wildcards.  The start and
        end elements can be a single '*' if not a date-time string.
        Return rows as list of named tuples containing:
        (network,station,location,channel,quality,starttime,endtime,samplerate,
         filename,byteoffset,bytes,hash,timeindex,timespans,timerates,
         format,filemodtime,updated,scanned,requeststart,requestend)
        '''
        my_uuid = uuid.uuid4().hex
        request_table = "request_%s" % my_uuid
        try:
            connection = self.engine.connect()
        except Exception as err:
            raise ValueError(str(err))

        if self.debug is True:
            print("Opening SQLite database for "
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
                               "VALUES (?,?,?,?,?,?) ".format(request_table),
                                                            (req[0],
                                                             req[1],
                                                             req[2],
                                                             req[3],
                                                             req[4],
                                                             req[5]))
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
                self.resolve_request(connection, request_table)
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
                   "ts.filemodtime,ts.updated,ts.scanned,""r.starttime,"
                   "r.endtime "
                   "FROM {0} ts, {1} r "
                   "WHERE "
                   "  ts.network {2} r.network "
                   "  AND ts.station {2} r.station "
                   "  AND ts.location {2} r.location "
                   "  AND ts.channel {2} r.channel "
                   "  AND ts.starttime <= r.endtime "
                   "  AND ts.starttime >= datetime(r.starttime,'-10 days') "
                   "  AND ts.endtime >= r.starttime "
                   .format(self.tsindex_table,
                           request_table, "GLOB" if wildcards else "="))

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
        while True:
            row = result.fetchone()
            if row is None:
                break
            index_rows.append(NamedRow(*row))

        # Sort results in application (ORDER BY in SQL 
        # triggers bad index usage)
        index_rows.sort()

        if self.debug is True:
            print("Fetched %d index rows" % len(index_rows))

        connection.execute("DROP TABLE {0}".format(request_table))

        return index_rows

    def fetch_summary_rows(self, query_rows):
        '''
        Fetch summary rows matching specified request
        `query_rows`: List of tuples containing (net,sta,loc,chan,start,end)
        Request elements may contain '?' and '*' wildcards.  The start and
        end elements can be a single '*' if not a date-time string.
        Return rows as list of named tuples containing:
        (network,station,location,channel,earliest,latest,updated)
        '''
        summary_rows = []
        my_uuid = uuid.uuid4().hex
        request_table = "request_%s" % my_uuid
        
        try:
            connection = self.engine.connect()
        except Exception as err:
            raise Exception(err)

        if self.debug is True:
            print("Opening SQLite database for "
                  "summary rows: %s" % self.sqlitedb)

        # Store temporary table(s) in memory
        try:
            connection.execute("PRAGMA temp_store=MEMORY")
        except Exception as err:
            raise ValueError(str(err))

        # Create temporary table and load request
        try:
            connection.execute("CREATE TEMPORARY TABLE {0}"
                        " (network TEXT, station TEXT,"
                        " location TEXT, channel TEXT)".
                        format(request_table))
            for req in query_rows:
                connection.execute("INSERT INTO {0} (network,station,"
                            "  location,channel) "
                            "VALUES (?,?,?,?) ".format(request_table), req[:4])


        except Exception as err:
            raise ValueError(str(err))

        result = connection.execute("SELECT count(*) "
                                    "FROM sqlite_master WHERE type='table' "
                                    "and name='{0}'"
                                    .format(self.tsindex_summary_table))
        summary_present = result.fetchone()[0]

        if summary_present:
            # Select summary rows by joining with summary table
            try:
                sql = ("SELECT DISTINCT s.network,s.station,s.location,"
                       "s.channel,s.earliest,s.latest,s.updt "
                       "FROM {0} s, {1} r "
                       "WHERE "
                       "  (r.network='*' OR s.network GLOB r.network) "
                       "  AND (r.station='*' OR s.station GLOB r.station) "
                       "  AND (r.location='*' OR s.location GLOB r.location) "
                       "  AND (r.channel='*' OR s.channel GLOB r.channel) ".
                       format(self.tsindex_summary_table, request_table))
                result = connection.execute(sql)

            except Exception as err:
                raise ValueError(str(err))

            # Map raw tuples to named tuples for clear referencing
            NamedRow = namedtuple('NamedRow',
                                  ['network', 'station', 'location', 'channel',
                                   'earliest', 'latest', 'updated'])

            summary_rows = []
            while True:
                row = result.fetchone()
                if row is None:
                    break
                summary_rows.append(NamedRow(*row))

            # Sort results in application (ORDER BY in SQL triggers
            # bad index usage)
            summary_rows.sort()

            if self.debug is True:
                print("Fetched %d summary rows" % len(summary_rows))

            connection.execute("DROP TABLE {0}".format(request_table))

        return summary_rows
    

    def resolve_request(self, connection, request_table):
        '''Resolve request table using summary
        `connection`: Database connection
        `request_table`: request table to resolve
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

        if self.debug is True:
            print("Resolved request with "
                  "summary into %d rows" % resolvedrows)

        connection.execute("DROP TABLE {0}".format(request_table_orig))

        return resolvedrows
