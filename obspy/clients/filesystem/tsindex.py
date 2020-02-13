# -*- coding: utf-8 -*-
r"""
obspy.clients.filesystem.tsindex - IRIS TSIndex Client and Indexer
==================================================================

The obspy.clients.filesystem.tsindex module includes a timeseries extraction
:class:`Client` class for a database created by the IRIS
`mseedindex <https://github.com/iris-edu/mseedindex>`_ program, as well as, a
:class:`Indexer` class for creating a SQLite3 database that follows the IRIS
`tsindex database schema
<https://github.com/iris-edu/mseedindex/wiki/Database-Schema/>`_\.

:copyright:
    Nick Falco, Chad Trabant, IRISDMC, 2018
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)


.. contents:: Contents
    :local:
    :depth: 2

Client Usage
------------

The first step is always to initialize a client object.

.. highlight:: python

>>> from obspy.clients.filesystem.tsindex import Client
>>> from obspy.clients.filesystem.tests.test_tsindex \
...     import get_test_data_filepath
>>> import os
>>> # for this example get the file path to test data
>>> filepath = get_test_data_filepath()
>>> db_path = os.path.join(filepath, 'timeseries.sqlite')
>>> # create a new Client instance
>>> client = Client(db_path, datapath_replace=("^", filepath))

The example below uses the test SQLite3 tsindex database included with ObsPy to
illustrate how to do the following:

* Determine what data is available in the tsindex database using
  :meth:`~Client.get_availability_extent()`
  and :meth:`~Client.get_availability()`, as
  well as, the percentage of data available using
  :meth:`~Client.get_availability_percentage()`.
* Request available timeseries data using
  :meth:`~Client.get_waveforms()` and
  :meth:`~Client.get_waveforms_bulk()`.

Determining Data Availability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`~Client.get_availability_extent()`:
  Returns a list of (network, station, location, channel, earliest, latest)
  tuples that represent the full extent of available data. This example
  retrieves from the very small obspy test tsindex database a list of all
  available ("BHZ") channel extents from the Global Seismograph Network
  ("IU") for all times.

>>> extents = client.get_availability_extent(network="IU", channel="BHZ")
>>> for extent in extents:
...     print("{0:<3} {1:<6} {2:<3} {3:<4} {4} {5}".format(*extent))
IU  ANMO   10  BHZ  2018-01-01T00:00:00.019500Z 2018-01-01T00:00:59.994536Z
IU  COLA   10  BHZ  2018-01-01T00:00:00.019500Z 2018-01-01T00:00:59.994538Z

* :meth:`~Client.get_availability()`: Works in the same way as
  :meth:`~Client.get_availability_extent()` but returns a list of (network,
  station, location, channel, starttime, endtime) tuples representing
  contiguous time spans for selected channels and time ranges.

* :meth:`~Client.get_availability_percentage()`:
  Returns the tuple(float, int) of percentage of available data
  (``0.0`` to ``1.0``) and number of gaps/overlaps. Availability percentage is
  calculated relative to the provided ``starttime`` and ``endtime``.

>>> from obspy import UTCDateTime
>>> avail_percentage = client.get_availability_percentage(
...     "IU", "ANMO", "10", "BHZ",
...     UTCDateTime(2018, 1, 1, 0, 0, 0, 19500),
...     UTCDateTime(2018, 1, 1, 0, 1, 57, 994536))
>>> print(avail_percentage)
(0.5083705674817509, 1)

Requesting Timeseries Data
^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`~Client.get_waveforms()`:
  This example illustrates how to request 1 second of available ("IU")
  timeseries data in the test tsindex database. Results are returned as a
  :class:`~obspy.core.stream.Stream` object. See the
  :meth:`~Client.get_waveforms_bulk()`
  method for information on how to make multiple requests at once.

>>> t = UTCDateTime("2018-01-01T00:00:00.019500")
>>> st = client.get_waveforms("IU", "*", "*", "BHZ", t, t + 1)
>>> st.plot()  # doctest: +SKIP

.. plot::

    from obspy import UTCDateTime
    from obspy.clients.filesystem.tsindex import Client
    from obspy.clients.filesystem.tests.test_tsindex \
        import get_test_data_filepath
    # for this example get the file path to test data
    filepath = get_test_data_filepath()
    db_path = os.path.join(filepath, 'timeseries.sqlite')
    # create a new Client instance
    client = Client(db_path, datapath_replace=("^", filepath))
    t = UTCDateTime("2018-01-01T00:00:00.019500")
    st = client.get_waveforms("IU", "*", "*", "BHZ", t, t + 1)
    st.plot()

Indexer Usage
-------------

The :class:`~Indexer` provides a high level
API for indexing a directory tree of miniSEED files using the IRIS
`mseedindex <https://github.com/iris-edu/mseedindex/>`_ software.

Initialize an indexer object by supplying the root path to data to be indexed.

>>> from obspy.clients.filesystem.tsindex import Indexer
>>> from obspy.clients.filesystem.tests.test_tsindex \
...     import get_test_data_filepath
>>> # for this example get the file path to test data
>>> filepath = get_test_data_filepath()
>>> # create a new Indexer instance
>>> indexer = Indexer(filepath, filename_pattern='*.mseed')

Index a directory tree of miniSEED files by calling
:meth:`~Indexer.run`. By default this will
create a database called ``timeseries.sqlite`` in the current working
directory. The name of the index database can be changed by supplying the
``database`` parameter when instantiating the
:class:`~Indexer` object.

.. code-block:: python

  indexer.run()

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import copyreg
import datetime
import logging
import os
import requests
import sqlalchemy as sa
import subprocess
import types
import warnings

from collections import namedtuple
from glob import glob
from multiprocessing import Pool
from os.path import relpath
from sqlalchemy.exc import ResourceClosedError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from obspy import UTCDateTime
from obspy.clients.filesystem.miniseed import _MiniseedDataExtractor, \
    NoDataError
from obspy.clients.filesystem.db import _get_tsindex_table, \
    _get_tsindex_summary_table
from obspy.core.stream import Stream


logger = logging.getLogger('obspy.clients.filesystem.tsindex')


try:
    import sqlalchemy
    # TSIndex needs sqlalchemy 1.0.0
    if not hasattr(sqlalchemy.engine.reflection.Inspector,
                   'get_temp_table_names'):
        raise ImportError
except ImportError:
    msg = ('TSIndex module expects sqlachemy version >1.0.0. Some '
           'functionality might not work.')
    warnings.warn(msg)
    _sqlalchemy_version_insufficient = True
else:
    _sqlalchemy_version_insufficient = False


def _pickle_method(m):
    """
    Allows serializing of class and instance methods.
    """
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


copyreg.pickle(types.MethodType, _pickle_method)


class Client(object):
    """
    Time series extraction client for IRIS tsindex database schema.
    """

    def __init__(self, database, datapath_replace=None, loglevel="WARNING"):
        """
        Initializes the client.

        :type database: str or
            :class:`~TSIndexDatabaseHandler`
        :param database: Path to sqlite tsindex database or a
            TSIndexDatabaseHandler object
        :type datapath_replace: tuple(str, str)
        :param datapath_replace: A ``tuple(str, str)``, where any
            occurrence of the first value will be replaced with the second
            value in filename paths from the index.
        :type loglevel: str
        :param loglevel: logging verbosity
        """
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        logging.basicConfig(level=numeric_level)
        logger.setLevel(numeric_level)

        # setup handler for database
        if isinstance(database, (str, native_str)):
            self.request_handler = TSIndexDatabaseHandler(
                os.path.normpath(database),
                loglevel=loglevel)
        elif isinstance(database, TSIndexDatabaseHandler):
            self.request_handler = database
        else:
            raise ValueError("database must be a string or "
                             "TSIndexDatabaseHandler object.")

        # Create and configure the data extraction
        self.data_extractor = _MiniseedDataExtractor(
            dp_replace=datapath_replace,
            loglevel=loglevel)

    def get_waveforms(self, network, station, location,
                      channel, starttime, endtime, merge=-1):
        """
        Query tsindex database and read miniSEED data from local indexed
        directory tree.

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
            :meth:`Stream.merge() <obspy.core.stream.Stream.merge>` for
            details. If set to ``None`` (or ``False``) no merge operation at
            all will be performed.
        :rtype: :class:`~obspy.core.stream.Stream`
        :returns: A ObsPy :class:`~obspy.core.stream.Stream` object containing
            requested timeseries data.
        """
        query_rows = [(network, station, location,
                       channel, starttime, endtime)]
        return self._get_waveforms(query_rows, merge)

    def get_waveforms_bulk(self, query_rows, merge=-1):
        """
        Query tsindex database and read miniSEED data from local indexed
        directory tree using a bulk request.

        :type query_rows: list(tuple(str, str, str, str,
            :class:`~obspy.core.utcdatetime.UTCDateTime`,
            :class:`~obspy.core.utcdatetime.UTCDateTime`)
        :param query_rows: A list of tuples [(net, sta, loc, cha, starttime,
            endtime),...] containing information on what timeseries should be
            returned from the indexed archive.
            Wildcards '*' and '?' are supported.
        :type merge: int or None
        :param merge: Specifies, which merge operation should be performed
            on the stream before returning the data. Default (``-1``) means
            only a conservative cleanup merge is performed to merge seamless
            traces (e.g. when reading across day boundaries). See
            :meth:`Stream.merge() <obspy.core.stream.Stream.merge>` for
            details. If set to ``None`` (or ``False``) no merge operation at
            all will be performed.
        :rtype: :class:`~obspy.core.stream.Stream`
        :returns: A ObsPy :class:`~obspy.core.stream.Stream` object containing
            requested timeseries data.
        """
        return self._get_waveforms(query_rows, merge)

    def get_nslc(self, network=None, station=None, location=None,
                 channel=None, starttime=None, endtime=None):
        """
        Get a list of tuples [(net, sta, loc, cha),...] containing information
        on what streams are included in the tsindex database.

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
        :rtype: list(tuple(str, str, str, str))
        :returns: A list of tuples [(network, station, location, channel)...]
            containing information on what streams are included in the tsindex
            database.
        """
        summary_rows = self._get_summary_rows(network, station, location,
                                              channel, starttime, endtime)

        nslc_list = []
        for row in summary_rows:
            nslc = (row.network, row.station, row.location, row.channel)
            nslc_list.append(nslc)
        nslc_list.sort()
        return nslc_list

    def get_availability_extent(self, network=None, station=None,
                                location=None, channel=None, starttime=None,
                                endtime=None):
        """
        Get a list of tuples [(network, station, location, channel,
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
        :rtype: list(tuple(str, str, str, str,
            :class:`~obspy.core.utcdatetime.UTCDateTime`,
            :class:`~obspy.core.utcdatetime.UTCDateTime`))
        :returns: A list of tuples [(network, station, location, channel,
            earliest, latest)...] containing data extent info for time series
            included in the tsindex database.
        """
        summary_rows = self._get_summary_rows(network, station,
                                              location, channel,
                                              starttime, endtime)

        availability_extents = []
        for row in summary_rows:
            extent = (row.network, row.station, row.location, row.channel,
                      UTCDateTime(row.earliest), UTCDateTime(row.latest))
            availability_extents.append(extent)
        availability_extents.sort()
        return availability_extents

    def get_availability(self, network=None, station=None, location=None,
                         channel=None, starttime=None, endtime=None,
                         include_sample_rate=False,
                         merge_overlap=False):
        """
        Get a list of tuples [(network, station, location, channel,
        starttime, endtime),...] containing data availability info for
        time series included in the tsindex database.

        If ``include_sample_rate=True``, then a tuple containing the sample
        rate [(net, sta, loc, cha, start, end, sample_rate),...] is returned.

        If ``merge_overlap=True``, then all time spans that overlap are merged.

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
        :param include_sample_rate: If ``include_sample_rate=True``, then
            a tuple containing the sample rate [(net, sta, loc, cha,
            start, end, sample_rate),...] is returned.
        :type merge_overlap: bool
        :param merge_overlap: If ``merge_overlap=True``, then all time
            spans that overlap are merged.
        :rtype: list(tuple(str, str, str, str,
            :class:`~obspy.core.utcdatetime.UTCDateTime`,
            :class:`~obspy.core.utcdatetime.UTCDateTime`))
        :returns: A list of tuples [(network, station, location, channel,
            earliest, latest)...] representing contiguous time spans for
            selected channels and time ranges.
        """

        tsindex_rows = self._get_tsindex_rows(network, station,
                                              location, channel,
                                              starttime, endtime)

        grouped_channels = {}
        for row in tsindex_rows:
            if include_sample_rate is True:
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
        joined_avail_tuples.sort()
        return joined_avail_tuples

    def get_availability_percentage(self, network, station,
                                    location, channel,
                                    starttime, endtime):
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
        :rtype: tuple(float, int)
        :returns: Tuple of percentage of available data (``0.0`` to ``1.0``)
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

    def has_data(self, network=None, station=None, location=None,
                 channel=None, starttime=None, endtime=None):
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
        :param starttime: Start of requested time window. Defaults to
            minimum possible start date.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of requested time window. Defaults to maximum
            possible end date.
        :rtype: bool
        :returns: Returns ``True`` if there is data in the index for a given
            network, station, location, channel, starttime, endtime.
        """
        if starttime is None:
            starttime = UTCDateTime(year=datetime.MINYEAR,
                                    month=1,
                                    day=1)
        if endtime is None:
            endtime = UTCDateTime(year=datetime.MAXYEAR,
                                  month=12,
                                  day=31)
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

        Information about the tsindex_summary schema may be found in the
        `mseedindex wiki <https://github.com/iris-edu/mseedindex/wiki/\
        Database-Schema#suggested-time-series-summary-table>`_.

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
        :class:`~obspy.core.stream.Stream` object containing the requested
        timeseries data.

        :type query_rows: list(tuple(str, str, str, str,
            :class:`~obspy.core.utcdatetime.UTCDateTime`,
            :class:`~obspy.core.utcdatetime.UTCDateTime`))
        :param query_rows: A list of tuples [(net, sta, loc, cha, starttime,
            endtime),...] containing information on what timeseries should be
            returned from the indexed archive.
            Wildcards '*' and '?' are supported.
        :type merge: int or None
        :param merge: Specifies, which merge operation should be performed
            on the stream before returning the data. Default (``-1``) means
            only a conservative cleanup merge is performed to merge seamless
            traces (e.g. when reading across day boundaries). See
            :meth:`Stream.merge() <obspy.core.stream.Stream.merge>` for
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

        if merge:
            st.merge(merge)
        return st

    def _get_tsindex_rows(self, network, station, location, channel, starttime,
                          endtime):
        """
        Return a list of tuples [(net, sta, loc, cha, quality... etc.),...]
        containing information found in the tsindex table.

        Information about the tsindex schema may be found in the
        `mseedindex wiki <https://github.com/iris-edu/mseedindex/wiki/\
        Database-Schema#sqlite-schema-version-11>`_.

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
        and merging if merge_overlaps is ``True``.

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
        :param timespans: List of timespan tuples.
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
                                          sr=sr if include_sample_rate
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
                                          sr=sr if include_sample_rate
                                          else None)
            _sncl_joined_avail_tuples.append(avail_tuple)
        return _sncl_joined_avail_tuples

    def _are_timespans_adjacent(self, ts1, ts2, sample_rate, tolerance=0.5):
        """
        Checks whether or not two time span named tuples
        (e.g. NameTuple(earliest, latest)) are adjacent within
        a given tolerance.

        :type ts1: NamedTuple
        :param ts1: Earliest timespan.
        :type ts2: NamedTuple
        :param ts2: Latest timespan.
        :type sample_rate: int
        :param sample_rate: Sensor sample rate.
        :type tolerance: float
        :param tolerance: Tolerance in seconds to determine whether a adjacent
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
        :type sr: float
        :param sr: Sensor sample rate (optional).
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
        Create a ``TimeSpan`` named tuple object given a earliest and latest
        date.

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
    Recursively search for files matching ``filename_pattern`` starting
    from ``root_path`` and run ``index_cmd`` for each target file found that
    is not already in the index. After all new files are indexed a summary
    table is generated with the extents of each timeseries.
    """

    def __init__(self, root_path, database="timeseries.sqlite",
                 leap_seconds_file="SEARCH", index_cmd='mseedindex',
                 bulk_params=None, filename_pattern='*', parallel=5,
                 loglevel="WARNING"):
        """
        Initializes the Indexer.

        :type root_path: str
        :param root_path: Root path to the directory structure to index.
        :type database: str or
            :class:`~TSIndexDatabaseHandler`
        :param database: Path to sqlite tsindex database or a
            TSIndexDatabaseHandler object. A database will be created
            if one does not already exists at the specified path.
        :type leap_seconds_file: str
        :param leap_seconds_file: Path to leap seconds file. If set to
            "SEARCH" (default), then the program looks for a leap seconds file
            in the same directory as the sqlite3 database. If set to `None`
            then no leap seconds file will be used.

            In :meth:`~Indexer.run` the leap
            seconds listed in this file will be used to adjust the time
            coverage for records that contain a leap second. Also, leap second
            indicators in the miniSEED headers will be ignored. See the
            `mseedindex wiki <https://github.com/iris-edu/mseedindex/blob/"
            "master/doc/mseedindex.md#leap-second-list-file>`_ for more"
            "for more information regarding this file.
        :type index_cmd: str
        :param index_cmd: Command to be run for each target file found that
            is not already in the index
        :type bulk_params: dict
        :param bulk_params: Dictionary of options to pass to ``index_cmd``.
        :type filename_pattern: str
        :param filename_pattern: Glob pattern to determine what files to index.
        :type parallel: int
        :param parallel: Max number of ``index_cmd`` instances to run in
            parallel. By default a max of 5 parallel process are run.
        :type loglevel: str
        :param loglevel: logging verbosity
        """
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        logging.basicConfig(level=numeric_level)
        logger.setLevel(numeric_level)

        self.index_cmd = index_cmd
        if bulk_params is None:
            bulk_params = {}
        self.bulk_params = bulk_params
        self.filename_pattern = filename_pattern
        self.parallel = parallel

        # setup handler for database
        if isinstance(database, (str, native_str)):
            self.request_handler = TSIndexDatabaseHandler(database,
                                                          loglevel=loglevel)
        elif isinstance(database, TSIndexDatabaseHandler):
            self.request_handler = database
        else:
            raise ValueError("Database must be a string or "
                             "TSIndexDatabaseHandler object.")

        self.leap_seconds_file = self._get_leap_seconds_file(leap_seconds_file)

        self.root_path = os.path.abspath(root_path)
        if not os.path.isdir(self.root_path):
            raise OSError("Root path `{}` does not exists."
                          .format(self.root_path))

    def run(self, build_summary=True, relative_paths=False, reindex=False):
        """
        Execute the file discovery and indexing.

        :type build_summary: bool
        :param build_summary: By default, a summary table is (re)generated
            containing the extents for each time series in the index. This can
            be turned off by setting `build_summary` to False.
        :type relative_paths: bool
        :param relative_paths: By default, the absolute path to each file is
            stored in the index. If ``relative_paths`` is True, the file paths
            will be relative to the ``root_path``.
        :type reindex: bool
        :param reindex: By default, files are not indexed that are already in
            the index and have not been modified.  The ``reindex`` option can
            be set to ``True`` to force a re-indexing of all files regardless.
        """
        if self._is_index_cmd_installed() is False:
            raise OSError(
                    "Required program '{}' is not installed. Hint: Install "
                    "mseedindex at https://github.com/iris-edu/mseedindex/."
                    .format(self.index_cmd))
        self.request_handler._set_sqlite_pragma()
        file_paths = self.build_file_list(relative_paths, reindex)

        # always keep the original file paths as specified. absolute and
        # relative paths are determined in the build_file_list method
        self.bulk_params["-kp"] = None
        if self.bulk_params.get("-table") is None:
            # set db table to write to
            self.bulk_params["-table"] = self.request_handler.tsindex_table
        if self.bulk_params.get("-sqlite") is None:
            # set path to sqlite database
            self.bulk_params['-sqlite'] = self.request_handler.database

        pool = Pool(processes=self.parallel)
        # run mseedindex on each file in parallel
        try:
            proccesses = []
            for file_name in file_paths:
                logger.debug("Indexing file '{}'.".format(file_name))
                proc = pool.apply_async(Indexer._run_index_command,
                                        args=(self.index_cmd,
                                              self.root_path,
                                              file_name,
                                              self.bulk_params))
                proccesses.append(proc)
            pool.close()
            # Without timeout, cannot respond to KeyboardInterrupt.
            # Also need get to raise the exceptions workers may throw.
            for proc in proccesses:
                cmd, rc, out, err = proc.get(timeout=999999)
                if rc:
                    logger.warning("FAIL [{0}] '{1}' out: '{2}' err: '{3}'"
                                   .format(rc, cmd, out, err))
            pool.join()
        except KeyboardInterrupt:
            logger.warning('Parent received keyboard interrupt.')
            if build_summary is True:
                logger.warning("Skipped building timeseries summary "
                               "table since indexing was ended "
                               "prematurely.")
            pool.terminate()
        else:
            if build_summary is True:
                self.request_handler.build_tsindex_summary()

    def build_file_list(self, relative_paths=False, reindex=False):
        """
        Create a list of absolute paths to all files under ``root_path`` that
        match the ``filename_pattern``.

        :type relative_paths: bool
        :param relative_paths: By default, the absolute path to each file is
            stored in the index. If ``relative_paths`` is ``True``, then the
            file paths will be relative to the ``root_path``.
        :type reindex: bool
        :param reindex: If ``reindex`` is ``True``, then already indexed
            files will be reindexed.
        :rtype: list(str)
        :returns: A list of files under the ``root_path`` matching
            ``filename_pattern``.
        """
        logger.debug("Building a list of files to index.")
        file_list = self._get_rootpath_files(relative_paths=False)
        # find relative file paths in case they are stored in the database as
        # relative paths.
        file_list_relative = self._get_rootpath_files(relative_paths=True)
        result = []
        if reindex is False and self.request_handler.has_tsindex():
            # remove any files already in the tsindex table
            unindexed_abs = []
            unindexed_rel = []
            tsindex = self.request_handler._fetch_index_rows()
            tsindex_filenames = [os.path.normpath(row.filename)
                                 for row in tsindex]
            for abs_fn, rel_fn in zip(file_list, file_list_relative):
                if abs_fn not in tsindex_filenames and \
                   rel_fn not in tsindex_filenames:
                    unindexed_abs.append(abs_fn)
                    unindexed_rel.append(rel_fn)
            if relative_paths is True:
                result = unindexed_rel
            else:
                result = unindexed_abs
        elif relative_paths is True:
            result = file_list_relative
        else:
            result = file_list
        if not result:
            raise OSError("No {}files matching filename pattern '{}' "
                          "were found under root path '{}'."
                          .format("unindexed " if reindex is False else "",
                                  self.filename_pattern,
                                  self.root_path))
        return result

    def download_leap_seconds_file(self, file_path=None):
        """
        Attempt to download leap-seconds.list from Internet Engineering Task
        Force (IETF) and save to a file.

        :type file_path: str
        :param file_path: Optional path to file path where leap seconds
            file should be downloaded. By default the file is downloaded to
            the same directory as the
            :class:`~Indexer` instances
            sqlite3 timeseries index database path.

        :rtype: str
        :returns: Path to downloaded leap seconds file.
        """
        try:
            logger.info("Downloading leap seconds file from the IETF.")
            r = self._download(
                        "http://www.ietf.org/timezones/data/leap-seconds.list")
            if file_path is None:
                file_path = os.path.join(
                            os.path.dirname(self.request_handler.database),
                            "leap-seconds.list")
                logger.debug("No leap seconds file path specified. Attempting "
                             "to create a leap seconds file at {}."
                             .format(file_path))
        except Exception as e:  # pragma: no cover
            raise OSError(
                ("Failed to download leap seconds file due to: {}. "
                 "No leap seconds file will be used.").format(str(e)))
        try:
            logger.debug("Writing IETF leap seconds info to a file at {}."
                         .format(file_path))
            with open(file_path, "w") as fh:
                fh.write(r.text)
        except Exception as e:  # pragma: no cover
            raise OSError("Failed to create leap seconds file at {} due to {}."
                          .format(file_path, str(e)))
        return file_path

    def _get_rootpath_files(self, relative_paths=False):
        """
        Return a list of absolute paths to files under the rootpath that
        match the Indexers filename pattern
        """
        file_list = [os.path.normpath(y) for x in os.walk(self.root_path)
                     for y in glob(os.path.join(x[0], self.filename_pattern))
                     if os.path.isfile(y)]
        if relative_paths:
            file_list_relative = []
            for abs_path in file_list:
                file_list_relative.append(os.path.normpath(relpath(
                    abs_path, self.root_path)))
            return file_list_relative
        else:
            return file_list

    def _download(self, url):
        return requests.get(url)

    def _get_leap_seconds_file(self, leap_seconds_file):
        """
        Return path to leap second file and set appropriate environment
        variable for mseedindex.

        :type leap_seconds_file: str or None
        :param leap_seconds_file: Leap second file options defined in the
            :class:`~Indexer` constructor.
        """
        if leap_seconds_file is not None:
            if leap_seconds_file == "SEARCH":
                dbpath = os.path.dirname(self.request_handler.database)
                file_path = os.path.join(dbpath, "leap-seconds.list")
                # leap seconds file will be downloaded when calling mseedindex
                if os.path.isfile(file_path):
                    leap_seconds_file = os.path.abspath(file_path)
                else:
                    leap_seconds_file = "NONE"
                    logger.warning("Leap seconds file `{}` not found. "
                                   "No leap seconds file will be used for "
                                   "indexing.".format(file_path))
            elif os.path.isfile(leap_seconds_file):
                # use leap seconds file provided by user
                leap_seconds_file = os.path.abspath(leap_seconds_file)
            else:
                raise OSError("No leap seconds file exists at `{}`. "
                              .format(leap_seconds_file))
            os.environ["LIBMSEED_LEAPSECOND_FILE"] = os.path.abspath(
                                                        leap_seconds_file)
        else:
            # warn user and don't use a leap seconds file
            logger.warning("No leap second file specified. This is highly "
                           "recommended.")
            os.environ["LIBMSEED_LEAPSECOND_FILE"] = "NONE"
        return leap_seconds_file

    def _is_index_cmd_installed(self):
        """
        Checks if the index command (e.g. mseedindex) is installed.

        :rtype: bool
        :returns: Returns ``True`` if the ``index_cmd`` is installed.
        """
        try:
            subprocess.call([self.index_cmd, "-V"])
        except OSError:
            return False
        else:
            return True

    @classmethod
    def _run_index_command(cls, index_cmd, root_path, file_name, bulk_params):
        """
        Execute a command to perform indexing.

        :type index_cmd: str
        :param index_cmd: Name of indexing command to execute. Defaults to
            ``mseedindex``.
        :type file_name: str
        :param file_name: Name of file to index.
        :type bulk_params: dict
        :param bulk_params: Dictionary of options to pass to ``index_cmd``.
        """
        try:
            cmd = [index_cmd]
            for option, value in bulk_params.items():
                params = [option, value]
                cmd.extend(params)
            cmd.append(file_name)
            # boolean options have a value of None
            cmd = [c for c in cmd if c is not None]
            proc = subprocess.Popen(cmd,
                                    cwd=root_path,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            out, err = proc.communicate()
            return (index_cmd, proc.returncode,
                    out.strip(), err.strip())
        except Exception as err:
            msg = ("Error running command `{}` - {}"
                   .format(index_cmd, err))
            raise OSError(msg)


class TSIndexDatabaseHandler(object):
    """
    Supports direct tsindex database data access and manipulation.
    """

    def __init__(self, database=None, tsindex_table="tsindex",
                 tsindex_summary_table="tsindex_summary",
                 session=None, loglevel="WARNING"):
        """
        Main query interface to timeseries index database.

        :type database: str
        :param database: Path to sqlite tsindex database
        :type tsindex_table: str
        :param tsindex_table: Name of timeseries index table
        :type tsindex_summary_table: str
        :param tsindex_summary_table: Name of timeseries index summary table
        :type session: :class:`sqlalchemy.orm.session.Session`
        :param session: An existing database session object.
        :type loglevel: str
        :param loglevel: logging verbosity
        """
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        logging.basicConfig(level=numeric_level)
        logger.setLevel(numeric_level)

        self.tsindex_table = tsindex_table
        self.tsindex_summary_table = tsindex_summary_table
        self.TSIndexTable = _get_tsindex_table(self.tsindex_table)
        self.TSIndexSummaryTable = \
            _get_tsindex_summary_table(self.tsindex_summary_table)

        if database and session:
            raise ValueError("Both a database path and an existing database "
                             "session object were supplied. Supply one or "
                             "the other but not both.")

        if session:
            self.session = session
            self.engine = session().get_bind()
        elif database:
            if isinstance(database, (str, native_str)):
                self.database = os.path.abspath(database)
                db_dirpath = os.path.dirname(self.database)
                if not os.path.exists(db_dirpath):
                    raise OSError("Database path '{}' does not exist."
                                  .format(db_dirpath))
                elif not os.path.isfile(self.database):
                    logger.warning("No sqlite3 database file exists at `{}`."
                                   .format(self.database))
            else:
                raise ValueError("database must be a string.")
            db_path = "sqlite:///{}".format(self.database)
            self.engine = sa.create_engine(db_path, poolclass=QueuePool)
            self.session = sessionmaker(bind=self.engine)
        else:
            raise ValueError("Either a database path or an existing "
                             "database session object must be supplied.")

    def get_tsindex_summary_cte(self):
        """
        :rtype: `sqlalchemy.sql.expression.CTE`
        :returns: Returns a common table expression (CTE) containing the
            tsindex summary information. If a tsindex summary table has been
            created in the database it will be used as the source for the CTE,
            otherwise it will be created by querying the tsindex table.
        """
        session = self.session()
        tsindex_summary_cte_name = "tsindex_summary"
        if self.has_tsindex_summary():
            # get tsindex summary cte by querying tsindex_summary table
            tsindex_summary_cte = \
                (session
                 .query(self.TSIndexSummaryTable)
                 .group_by(self.TSIndexSummaryTable.network,
                           self.TSIndexSummaryTable.station,
                           self.TSIndexSummaryTable.location,
                           self.TSIndexSummaryTable.channel)
                 .cte(name=tsindex_summary_cte_name)
                 )
        else:
            logger.warning("No {0} table found! A {0} "
                           "CTE will be created by querying the {1} "
                           "table, which could be slow!"
                           .format(self.tsindex_summary_table,
                                   self.tsindex_table))
            logger.info("For improved performance create a permanent "
                        "{0} table by running the "
                        "TSIndexDatabaseHandler.build_tsindex_summary() "
                        "instance method."
                        .format(self.tsindex_summary_table))
            # create the tsindex summary cte by querying the tsindex table.
            tsindex_summary_cte = \
                (session
                 .query(self.TSIndexTable.network,
                        self.TSIndexTable.station,
                        self.TSIndexTable.location,
                        self.TSIndexTable.channel,
                        sa.func.min(self.TSIndexTable.starttime)
                        .label("earliest"),
                        sa.func.max(self.TSIndexTable.endtime)
                        .label("latest"),
                        sa.literal(
                            UTCDateTime.now().isoformat()).label("updt")
                        )
                 .group_by(self.TSIndexTable.network,
                           self.TSIndexTable.station,
                           self.TSIndexTable.location,
                           self.TSIndexTable.channel)
                 .cte(name=tsindex_summary_cte_name)
                 )
        return tsindex_summary_cte

    def build_tsindex_summary(self):
        """
        Builds a tsindex_summary table using the table name supplied to the
        Indexer instance (defaults to 'tsindex_summary').
        """
        # test if tsindex table exists
        if not self.has_tsindex():
            raise ValueError("No tsindex table '{}' exists in database '{}'."
                             .format(self.tsindex_table, self.database))
        if self.has_tsindex_summary():
            self.TSIndexSummaryTable.__table__.drop(self.engine)

        session = self.session()
        self.TSIndexSummaryTable.__table__.create(self.engine)
        rows = (session
                .query(self.TSIndexTable.network,
                       self.TSIndexTable.station,
                       self.TSIndexTable.location,
                       self.TSIndexTable.channel,
                       sa.func.min(self.TSIndexTable.starttime)
                       .label("earliest"),
                       sa.func.max(self.TSIndexTable.endtime)
                       .label("latest"),
                       sa.literal(UTCDateTime().now().isoformat()))
                .group_by(self.TSIndexTable.network,
                          self.TSIndexTable.station,
                          self.TSIndexTable.location,
                          self.TSIndexTable.channel))
        session.execute(self.TSIndexSummaryTable.__table__.insert(),
                        [{'network': r[0],
                          'station': r[1],
                          'location': r[2],
                          'channel': r[3],
                          'earliest': r[4],
                          'latest': r[5],
                          'updt': r[6]
                          }
                        for r in rows])
        session.commit()

    def has_tsindex_summary(self):
        """
        Returns ``True`` if there is a tsindex_summary table in the database.

        :rtype: bool
        :returns: Returns ``True`` if there a tsindex_summary table is present
            in the database.
        """
        table_names = sa.inspect(self.engine).get_table_names()
        temp_table_names = sa.inspect(self.engine).get_temp_table_names()
        if self.tsindex_summary_table in table_names or \
                self.tsindex_summary_table in temp_table_names:
            return True
        else:
            return False

    def has_tsindex(self):
        """
        Returns ``True`` if there is a tsindex table in the database.

        :rtype: bool
        :returns: Returns ``True`` if there a tsindex table is present
            in the database.
        """
        table_names = sa.inspect(self.engine).get_table_names()
        temp_table_names = sa.inspect(self.engine).get_temp_table_names()
        if self.tsindex_table in table_names or \
                self.tsindex_table in temp_table_names:
            return True
        else:
            return False

    def _fetch_index_rows(self, query_rows=None, bulk_params=None):
        '''
        Fetch index rows matching specified request. This method is marked as
        private because the index schema is subject to change.

        :type query_rows: list(tuple(str, str, str, str,
            :class:`~obspy.core.utcdatetime.UTCDateTime`,
            :class:`~obspy.core.utcdatetime.UTCDateTime`))
        :param query_rows: List of tuples containing (net,sta,loc,chan,start,
            end). By default everything is selected.
        :type bulk_params: dict
        :param bulk_params: Dict of bulk parameters (e.g. quality)
            Request elements may contain '?' and '*' wildcards.  The start and
            end elements can be a single '*' if not a date-time string.
        :rtype: list(tuple)
        :returns: Return rows as list of named tuples containing:
            (network, station, location, channel, quality, starttime, endtime,
            samplerate, filename, byteoffset, bytes, hash, timeindex,
            timespans, timerates, format, filemodtime, updated, scanned,
            requeststart, requestend).
        '''

        session = self.session()

        if query_rows is None:
            query_rows = []
        if bulk_params is None:
            bulk_params = {}

        query_rows = self._clean_query_rows(query_rows)
        request_cte_name = "raw_request_cte"

        result = []
        # Create temporary table and load request
        try:
            stmts = [
                sa.select([
                    sa.literal(a).label("network"),
                    sa.literal(b).label("station"),
                    sa.literal(c).label("location"),
                    sa.literal(d).label("channel"),
                    sa.literal(e).label("starttime")
                    if e != '*' else
                    sa.literal('0000-00-00T00:00:00').label("starttime"),
                    sa.literal(f).label("endtime")
                    if f != '*' else
                    sa.literal('5000-00-00T00:00:00').label("endtime")
                ])
                for idx, (a, b, c, d, e, f) in enumerate(query_rows)
            ]
            requests = sa.union_all(*stmts)
            requests_cte = requests.cte(name=request_cte_name)
            wildcards = False
            for req in query_rows:
                for field in req:
                    if '*' in str(field) or '?' in str(field):
                        wildcards = True
                        break
            summary_present = self.has_tsindex_summary()
            if wildcards and summary_present:
                # Resolve wildcards using summary if present to:
                # a) resolve wildcards, allows use of '=' operator
                #    and table index
                # b) reduce index table search to channels that are
                #    known included
                flattened_request_cte_name = 'flattened_request_cte'
                # expand
                flattened_request_cte = (
                    session
                    .query(self.TSIndexSummaryTable.network,
                           self.TSIndexSummaryTable.station,
                           self.TSIndexSummaryTable.location,
                           self.TSIndexSummaryTable.channel,
                           self.TSIndexSummaryTable.network,
                           sa.case([
                                    (requests_cte.c.starttime == '*',
                                     self.TSIndexSummaryTable.earliest),
                                    (requests_cte.c.starttime != '*',
                                     requests_cte.c.starttime)
                                   ])
                           .label('starttime'),
                           sa.case([
                                    (requests_cte.c.endtime == '*',
                                     self.TSIndexSummaryTable.latest),
                                    (requests_cte.c.endtime != '*',
                                     requests_cte.c.endtime)
                                   ])
                           .label('endtime'))
                    .filter(self.TSIndexSummaryTable.network.op('GLOB')
                            (requests_cte.c.network))
                    .filter(self.TSIndexSummaryTable.station.op('GLOB')
                            (requests_cte.c.station))
                    .filter(self.TSIndexSummaryTable.location.op('GLOB')
                            (requests_cte.c.location))
                    .filter(self.TSIndexSummaryTable.channel.op('GLOB')
                            (requests_cte.c.channel))
                    .filter(self.TSIndexSummaryTable.earliest <=
                            requests_cte.c.endtime)
                    .filter(self.TSIndexSummaryTable.latest >=
                            requests_cte.c.starttime)
                    .order_by(self.TSIndexSummaryTable.network,
                              self.TSIndexSummaryTable.station,
                              self.TSIndexSummaryTable.location,
                              self.TSIndexSummaryTable.channel,
                              self.TSIndexSummaryTable.earliest,
                              self.TSIndexSummaryTable.latest)
                    .cte(name=flattened_request_cte_name))
                result = (
                    session
                    .query(self.TSIndexTable,
                           requests_cte.c.starttime,
                           requests_cte.c.endtime)
                    .filter(self.TSIndexTable.network ==
                            flattened_request_cte.c.network)
                    .filter(self.TSIndexTable.station ==
                            flattened_request_cte.c.station)
                    .filter(self.TSIndexTable.location ==
                            flattened_request_cte.c.location)
                    .filter(self.TSIndexTable.channel ==
                            flattened_request_cte.c.channel)
                    .filter(self.TSIndexTable.starttime <=
                            flattened_request_cte.c.endtime)
                    .filter(self.TSIndexTable.endtime >=
                            flattened_request_cte.c.starttime)
                    .order_by(self.TSIndexTable.network,
                              self.TSIndexTable.station,
                              self.TSIndexTable.location,
                              self.TSIndexTable.channel,
                              self.TSIndexTable.starttime,
                              self.TSIndexTable.endtime))
                wildcards = False
            else:
                result = (
                    session
                    .query(self.TSIndexTable,
                           requests_cte.c.starttime,
                           requests_cte.c.endtime
                           )
                    .filter(self.TSIndexTable.network.op('GLOB')
                            (requests_cte.c.network))
                    .filter(self.TSIndexTable.station.op('GLOB')
                            (requests_cte.c.station))
                    .filter(self.TSIndexTable.location.op('GLOB')
                            (requests_cte.c.location))
                    .filter(self.TSIndexTable.channel.op('GLOB')
                            (requests_cte.c.channel))
                    .filter(self.TSIndexTable.starttime <=
                            requests_cte.c.endtime)
                    .filter(self.TSIndexTable.endtime >=
                            requests_cte.c.starttime)
                    .order_by(self.TSIndexTable.network,
                              self.TSIndexTable.station,
                              self.TSIndexTable.location,
                              self.TSIndexTable.channel,
                              self.TSIndexTable.starttime,
                              self.TSIndexTable.endtime))
        except Exception as err:
            raise ValueError(str(err))

        index_rows = []
        try:
            for rt in result:
                # convert to a named tuple
                NamedRow = namedtuple('NamedRow',
                                      ['network', 'station', 'location',
                                       'channel', 'quality', 'version',
                                       'starttime', 'endtime', 'samplerate',
                                       'filename', 'byteoffset', 'bytes',
                                       'hash', 'timeindex', 'timespans',
                                       'timerates', 'format', 'filemodtime',
                                       'updated', 'scanned', 'requeststart',
                                       'requestend'])
                row, requeststart, requestend = rt
                nrow = NamedRow(
                        row.network, row.station, row.location,
                        row.channel, row.quality, row.version,
                        row.starttime, row.endtime, row.samplerate,
                        row.filename, row.byteoffset, row.bytes,
                        row.hash, row.timeindex, row.timespans,
                        row.timerates, row.format, row.filemodtime,
                        row.updated, row.scanned,
                        requeststart, requestend
                       )
                index_rows.append(nrow)
        except ResourceClosedError:
            pass  # query returned no results
        logger.debug("Fetched %d index rows" % len(index_rows))
        return index_rows

    def _fetch_summary_rows(self, query_rows):
        '''
        Fetch summary rows matching specified request. A temporary tsindex
        summary table is created if one does not exists. This method is marked
        as private because the index schema is subject to change.

        Returns rows as list of named tuples containing:
        (network,station,location,channel,earliest,latest,updated)

        :type query_rows: list(tuple(str, str, str, str,
            :class:`~obspy.core.utcdatetime.UTCDateTime`,
            :class:`~obspy.core.utcdatetime.UTCDateTime`))
        :param query_rows: List of tuples containing
            (net,sta,loc,chan,start,end) Request elements may contain '?'
            and '*' wildcards. The start and end elements can be a single
            '*' if not a date-time string.
        :rtype: list(tuple)
        :returns: Return rows as list of named tuples containing:
            (network, station, location, channel, earliest, latest, updated).
        '''
        session = self.session()
        query_rows = self._clean_query_rows(query_rows)
        tsindex_summary_cte = self.get_tsindex_summary_cte()
        # Create temporary table and load request
        try:
            request_cte_name = "request_cte"
            stmts = [
                sa.select([
                    sa.literal(a).label("network"),
                    sa.literal(b).label("station"),
                    sa.literal(c).label("location"),
                    sa.literal(d).label("channel"),
                    sa.literal(e).label("starttime")
                    if e != '*' else
                    sa.literal('0000-00-00T00:00:00').label("starttime"),
                    sa.literal(f).label("endtime")
                    if f != '*' else
                    sa.literal('5000-00-00T00:00:00').label("endtime")
                ])
                for idx, (a, b, c, d, e, f) in enumerate(query_rows)
            ]
            requests = sa.union_all(*stmts)
            requests_cte = requests.cte(name=request_cte_name)
        except Exception as err:
            raise ValueError(str(err))

        # Select summary rows by joining with summary table
        try:
            # expand
            result = (
                session
                .query(tsindex_summary_cte.c.network,
                       tsindex_summary_cte.c.station,
                       tsindex_summary_cte.c.location,
                       tsindex_summary_cte.c.channel,
                       tsindex_summary_cte.c.earliest,
                       tsindex_summary_cte.c.latest,
                       tsindex_summary_cte.c.updt)
                .filter(tsindex_summary_cte.c.network.op('GLOB')
                        (requests_cte.c.network))
                .filter(tsindex_summary_cte.c.station.op('GLOB')
                        (requests_cte.c.station))
                .filter(tsindex_summary_cte.c.location.op('GLOB')
                        (requests_cte.c.location))
                .filter(tsindex_summary_cte.c.channel.op('GLOB')
                        (requests_cte.c.channel))
                .filter(tsindex_summary_cte.c.earliest <=
                        requests_cte.c.endtime)
                .filter(tsindex_summary_cte.c.latest >=
                        requests_cte.c.starttime)
                .order_by(tsindex_summary_cte.c.network,
                          tsindex_summary_cte.c.station,
                          tsindex_summary_cte.c.location,
                          tsindex_summary_cte.c.channel,
                          tsindex_summary_cte.c.earliest,
                          tsindex_summary_cte.c.latest))
        except Exception as err:
            raise ValueError(str(err))

        # Map raw tuples to named tuples for clear referencing
        NamedRow = namedtuple('NamedRow',
                              ['network', 'station', 'location', 'channel',
                               'earliest', 'latest', 'updated'])
        summary_rows = []
        try:
            for row in result:
                summary_rows.append(NamedRow(*row))
        except ResourceClosedError:
            pass  # query returned no results
        logger.debug("Fetched %d summary rows" % len(summary_rows))
        return summary_rows

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

    def _format_date(self, dt):
        try:
            if dt is None:
                dt = "*"
            else:
                dt = UTCDateTime(dt)
        except TypeError:
            raise TypeError("'{}' could not be converted to "
                            "type 'UTCDateTime'.".format(dt))
        return dt

    def _clean_query_rows(self, query_rows):
        """
        Reformats query rows to match what is stored in the database.

        :type query_rows: list(tuple(str, str, str, str,
            :class:`~obspy.core.utcdatetime.UTCDateTime`,
            :class:`~obspy.core.utcdatetime.UTCDateTime`))
        :param query_rows: List of tuples containing (network, station,
            location, channel, starttime, endtime).
        """
        flat_query_rows = []
        if query_rows == []:
            # if an empty list is supplied then select everything
            select_all_query = self._create_query_row('*', '*', '*',
                                                      '*', '*', '*')
            flat_query_rows = [select_all_query]
        else:
            # perform some formatting on the query rows to ensure that they
            # query the database properly.
            for i, qr in enumerate(query_rows):
                query_rows[i] = self._create_query_row(*qr)

            # flatten query rows
            for req in query_rows:
                networks = (req[0].replace(" ", "").split(",")
                            if req[0] else "*")
                stations = (req[1].replace(" ", "").split(",")
                            if req[1] else "*")
                locations = (req[2].replace(" ", "").split(",")
                             if req[2] else "*")
                channels = (req[3].replace(" ", "").split(",")
                            if req[3] else "*")
                starttime = self._format_date(req[4])
                endtime = self._format_date(req[5])

                for net in networks:
                    for sta in stations:
                        for loc in locations:
                            for cha in channels:
                                qr = self._create_query_row(net, sta, loc, cha,
                                                            starttime, endtime)
                                flat_query_rows.append(qr)
        return flat_query_rows

    def _set_sqlite_pragma(self):
        """
        Setup a sqlite3 database for indexing.
        """
        try:
            logger.debug('Setting up sqlite3 database at %s' % self.database)
            # setup the sqlite database
            session = self.session()
            # https://www.sqlite.org/foreignkeys.html
            session.execute('PRAGMA foreign_keys = ON')
            # as used by mseedindex
            session.execute('PRAGMA case_sensitive_like = ON')
            # enable Write-Ahead Log for better concurrency support
            session.execute('PRAGMA journal_mode=WAL')
            # Store temporary table(s) in memory
            session.execute("PRAGMA temp_store=MEMORY")
        except Exception:
            raise OSError("Failed to setup sqlite3 database for indexing.")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
