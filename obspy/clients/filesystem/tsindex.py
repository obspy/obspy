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

import sqlite3
import uuid
from logging import getLogger
from collections import namedtuple
import dateutil.parser
from sqlalchemy import create_engine
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.clients.filesystem.db import Base, TSIndex, TSIndexSummary
from obspy.clients.filesystem.miniseed import MiniseedDataExtractor, \
    NoDataError

logger = getLogger(__name__)


class Client(object):
    """
    Time series extraction client for a database created by the
    IRIS mseedindex program.
    """

    def __init__(self, sqlitedb, datapath_replace=None):
        """
        Initializes the client.
        
        If `datapath_replace` is specified it must be a 2-value tuple,
        where any occurrence of the first value will be replaced with
        the second value in filename paths from the index.
        
        :type sqlitedb: str or
            ~obspy.clients.filesystem.tsindex.TSIndexRequestHandler
        :param sqlitedb: Path to sqlite tsindex database or a
            TSIndexRequestHandler object
        """
        if isinstance(sqlitedb, (str, native_str)):
            self.request_handler = TSIndexRequestHandler(sqlitedb)
        elif isinstance(sqlitedb, TSIndexRequestHandler):
            self.request_handler = sqlitedb
        else:
            raise ValueError("sqlitedb must be a string or "
                             "TSIndexRequestHandler object.")
            
        # Create and configure the data extraction
        self.data_extractor = MiniseedDataExtractor(datapath_replace)

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
        query_row = TSIndexRequestHandler.create_query_row(network, station,
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
        return self._get_waveforms(query_rows, merge)
    
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
            logger.warning("No data matched selection")
        
        logger.debug("Wrote {} bytes".format(total_bytes))
        if merge is None or merge is False:
            pass
        else:
            st.merge(merge)
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
        summary_rows = self._get_summary_rows(network, station,
                                              location, channel,
                                              UTCDateTime(starttime), 
                                              UTCDateTime(endtime))
        
        availability_extents = []
        for row in summary_rows:
            extent = (row.network, row.station, row.location, row.channel,
                      UTCDateTime(row.earliest), UTCDateTime(row.latest))
            availability_extents.append(extent)
        return availability_extents
    
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
        query_row = TSIndexRequestHandler.create_query_row(network, station,
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
        query_row = TSIndexRequestHandler.create_query_row(network, station,
                                                           location, channel,
                                                           starttime, endtime)
        query_rows = [query_row]
        return self.request_handler.fetch_index_rows(query_rows)
    
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
        
        def group_by_one_to_one_match(obj_dicts, compare_function):
            """
            Group every pair of items for which compare_function return True.
        
            Example: we have a list of 2-character strings representing segments
            of the alphabet (eg. 'af' represents 'a' through 'f'), and we want to 
            group all the ones that overlap alphabetically (so 'af' and 'eg'
            overlap, but 'ab' and 'cd' do not).
        
            >>> cmp_fn = lambda a, b: a[0] < b[1] and b[0] < a[1]
            >>> items = ['ab', 'ab', 'bc', 'bc', 'be', 'ef', 'fg']
            >>> group_by_one_to_one_match(items, cmp_fn)
             {'ab,ab': ['ab', 'ab'],
             'bc,bc,be': ['bc', 'bc', 'be'],
             'ef': ['ef'],
             'fg': ['fg']}
            """
            groups = [[] for x in obj_dicts]
            for index, obj_dict in enumerate(obj_dicts, start=0):
                index2 = index + 1
                groups[index].append(index)
                for obj_dict2 in obj_dicts[index2:]:
                    if compare_function(obj_dict, obj_dict2):
                        groups[index].append(index2)
                        groups[index2].append(index)
                    index2 = index2 + 1
        
            unique_groups = {}
            for index, group in enumerate(groups, start=0):
                key = ",".join([str(x) for x in group])
                if not unique_groups.get(key):
                    unique_groups[key] = [obj_dicts[index]]
                else:
                    unique_groups[key].append(obj_dicts[index])
        
            return unique_groups
        
        def should_join(row1, row2):
            """
            Check if two rows representing available timespans should
            be joined together to form one continuous timespan.
            """
            if row1.network == row2.network and \
              row1.station == row2.station and \
              row1.location == row2.location and \
              row1.channel == row2.channel and \
              row1.samplerate == row2.samplerate:
                # Find the next sample and compoare it against the 
                # calculate expected next sample
                last = UTCDateTime(row1.timespans[-1].split(":")[1])
                sample_period = 1 / sample_rate
                expected_next = last + sample_period + tolerance
                actual_next = row2.timespans[0].split(":")[0]
                if actual_next < expected_next or \
                    actual_next > expected_next:
                    return True
                else:
                    return False
            else:
                return False
            
        def should_merge(avail1, avail2):
            """
            Check if two rows representing available timespans should
            be joined together to form one continuous timespan.
            """
            if avail1[:3] == avail2[:3]: # check if NSLC match
                # Find the next sample and compoare it against the 
                # calculate expected next sample
                avail1_start = avail1[4]
                avail1_end = avail1[5]
                avail2_start = avail2[4]
                avail2_end = avail2[5]
                if avail1_start <= avail2_end or \
                    avail1_end >= avail2_start:
                    return True
                else:
                    return False
            else:
                return False
            
        def create_avail_record(net, sta, loc, cha,
                                earliest, latest, sr=None):
            if sr is not None:
                avail_record = (net, sta, loc, cha,
                                UTCDateTime(float(earliest)),
                                UTCDateTime(float(latest)), sr)
            else:
                avail_record = (net, sta, loc, cha,
                                UTCDateTime(float(earliest)),
                                UTCDateTime(float(latest)))
            return avail_record
        
        tsindex_rows = self.get_tsindex_rows(network, station,
                                             location, channel,
                                             starttime, endtime)
        # Create a dictionary of all groupings of interesecting timespans 
        # based on :meth:`Client.get_availability.should_join(...)
        # <obspy.clients.filesystem.tsindex.Client.get_availability.should_join>`. # NOQA
        grouped_rows = group_by_one_to_one_match(tsindex_rows, should_join)
        
        # create a list of tuples representing merged timespans 
        avail_timespans = []
        for _, row_list in grouped_rows.items():
            for r in row_list:
                net = r.network
                sta = r.station
                loc = r.location
                cha = r.channel
                sr = r.samplerate if include_sample_rate else None
                # create a python list from the 'timepsans' column returned by 
                # :meth:`Cllient.get_tsindex_rows(...)
                # <obspy.clients.filesystem.tsindex.Client.get_tsindex_rows>`.
                timespan_list = r.timespans.replace("[","") \
                                    .replace("]","").split(",")
                earliest_times = []
                latest_times = []
                for t in timespan_list:
                    earliest, latest = t.split(":")
                    avail_record = create_avail_record(net,
                                                       sta,
                                                       loc,
                                                       cha,
                                                       earliest,
                                                       latest,
                                                       sr)
                    avail_timespans.append(avail_record)
        avail_timespans.sort(key=lambda x: (x[4], x[5]))
        
        if merge_overlap:
            merged_avail_timespans = []
            grouped_rows = group_by_one_to_one_match(avail_timespans,
                                                     should_merge)
            for _, row_list in grouped_rows.items():
                row = row_list[0]
                net = row[0]
                sta = row[1]
                loc = row[2]
                cha = row[3]
                sr = row[4] if include_sample_rate else None
                earliest = min(row_list, key = lambda t: t[4])[4]
                latest = max(row_list, key = lambda t: t[5])[5]
                print(earliest)
                print(latest)
                avail_record = create_avail_record(net,
                                                   sta,
                                                   loc,
                                                   cha,
                                                   earliest,
                                                   latest,
                                                   sr)
                merged_avail_timespans.append(row)
            avail_timespans = merged_avail_timespans
        return avail_timespans

    def get_availability_percentage(self, network, station, location,
                                    channel, starttime, endtime):
        """
        Get percentage of available data for a specified network, station,
        location, channel, starttime, endtime combination.
        """
        avail_extents = self.get_availability_extent(network, station,
                                                     location, channel,
                                                     starttime, endtime)
        availability = self.get_availability(network, station,
                                             location, channel,
                                             starttime, endtime,
                                             merge_overlap=True)
        total_avail_extents = 0
        for avail in avail_extents:
            earliest, latest = avail[4], avail[5]
            total_avail_extents += latest - earliest
        total_avail = 0
        for avail in availability:
            earliest, latest = avail[4], avail[5]
            total_avail += latest - earliest
        print(total_avail)
        print(total_avail_extents)
        percent_avail = total_avail / total_avail_extents
        num_gaps = len(availability)
        return (percent_avail, num_gaps)
    
    def has_data(self, network, station, location,
                 channel, starttime, endtime):
        """
        Check if specified stream has any data using the tsindex database.
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


class TSIndexRequestHandler(object):
    
    def __init__(self, sqlitedb, tsindex_table="tsindex",
                 tsindex_summary_table="tsindex_summary",
                 maxsectiondays=10):
        self.sqlitedb = sqlitedb
        self.tsindex_table = tsindex_table
        self.tsindex_summary_table = tsindex_summary_table
        self.maxsectiondays = maxsectiondays
        
        db_path = "sqlite:///{}".format(sqlitedb)
        self.engine = create_engine(db_path, encoding=native_str('utf-8'),
                                    convert_unicode=True)
        Base.metadata.create_all(self.engine,  # @UndefinedVariable
                                 checkfirst=True)

    @classmethod
    def create_query_row(cls, network, station, location,
                         channel, starttime, endtime):
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
        
        connection = self.engine.connect()

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
                # Replace "--" location ID request alias with true empty value
                if req[2] == "--":
                    req[2] = ""

                start = dateutil.parser.parse(req[4].isoformat())
                end = dateutil.parser.parse(req[5].isoformat())
                connection.execute("INSERT INTO {0} (network,station,location,"
                               "channel,starttime,endtime) "
                               "VALUES (?,?,?,?,?,?) ".format(request_table),
                                                            (req[0],
                                                             req[1],
                                                             req[2],
                                                             req[3],
                                                             start,
                                                             end))
        except Exception as err:
            import traceback
            traceback.print_exc()
            raise ValueError(str(err))

        result = connection.execute("SELECT count(*) FROM sqlite_master "
                                    "WHERE type='table' and name='{0}'"
                                    .format(self.tsindex_summary_table))

        summary_present = result.fetchone()[0]

        wildcards = False
        for req in query_rows:
            for field in req:
                if '*' in field or '?' in field:
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
                   "  AND ts.starttime >= datetime(r.starttime,'-{3} days') "
                   "  AND ts.endtime >= r.starttime "
                   .format(self.tsindex_table,
                           request_table, "GLOB" if wildcards else "=",
                           self.maxsectiondays))

            # Add quality identifer criteria
            if 'quality' in bulk_params and \
                bulk_params['quality'] in ('D', 'R', 'Q'):
                sql = sql + " AND quality = '{0}' ".format(
                                                        bulk_params['quality'])

            result = connection.execute(sql)

        except Exception as err:
            import traceback
            traceback.print_exc()
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

        logger.debug("Fetched %d index rows" % len(index_rows))

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
        
        connection = self.engine.connect()

        logger.debug("Opening SQLite database for "
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
                # Replace "--" location ID request alias with true empty value
                if req[2] == "--":
                    req[2] = ""
                connection.execute("INSERT INTO {0} (network,station,"
                            "  location,channel) "
                            "VALUES (?,?,?,?) ".format(request_table), req[:4])


        except Exception as err:
            import traceback
            traceback.print_exc()
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

            logger.debug("Fetched %d summary rows" % len(summary_rows))

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

        logger.debug("Resolved request with "
                     "summary into %d rows" % resolvedrows)

        connection.execute("DROP TABLE {0}".format(request_table_orig))

        return resolvedrows


if __name__ == "__main__":
    network = "AK"
    station = "BAGL"
    location = "*"
    channel = "LCC"
    starttime = "2010-08-10T22:08:33.270000"
    endtime = "2019-08-10T22:24:48.950000"
    
    
    request = (network, station, location, channel, starttime, endtime)
    
    client = Client("/Users/nick/Development/workspace/tsindex-dataset/timeseries.sqlite")
    #st = client.get_waveforms(network, station,
    #                          location, channel,
    #                          UTCDateTime(starttime),
    #                          UTCDateTime(endtime))

    #print(st)
    #import obspy
    #st2 = obspy.read("/Users/nick/Downloads/fdsnws-dataselect_2018-08-20t22_12_01z.mseed")
    #print(st2)
    
    #nslc = client.get_nslc(network, station,
    #                       location, channel,
    #                       UTCDateTime(starttime),
    #                       UTCDateTime(endtime))

    #print(nslc)
    avail_extents = client.get_availability_extent(network, station,
                                                   location, channel,
                                                   UTCDateTime(starttime),
                                                   UTCDateTime(endtime))
    for a in avail_extents:
        print('  '.join(str(i) for i in a))

    print("\n--------------------------\n")
    avail = client.get_availability(network, station,
                                    location, channel,
                                    UTCDateTime(starttime),
                                    UTCDateTime(endtime))
    for a in avail:
        print('  '.join(str(i) for i in a))
        
    avail_percentage = client.get_availability_percentage(network, station,
                                                         location, channel,
                                                         UTCDateTime(starttime),
                                                         UTCDateTime(endtime))
    print("avail = %s %s" % avail_percentage)
