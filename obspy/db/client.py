# -*- coding: utf-8 -*-
"""
Client for a database created by obspy.db.

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

from sqlalchemy import and_, create_engine, func, or_
from sqlalchemy.orm import sessionmaker

from obspy.core.preview import merge_previews
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime
from obspy.db.db import Base, WaveformChannel, WaveformFile, WaveformPath


class Client(object):
    """
    Client for a database created by obspy.db.
    """
    def __init__(self, url=None, session=None, debug=False):
        """
        Initializes the client.

        :type url: str, optional
        :param url: A string that indicates database dialect and connection
            arguments. See
            http://docs.sqlalchemy.org/en/latest/core/engines.html for more
            information about database dialects and urls.
        :type session: :class:`sqlalchemy.orm.session.Session`, optional
        :param session: An existing database session object.
        :type debug: bool, optional
        :param debug: Enables verbose output.
        """
        if url:
            self.engine = create_engine(url, encoding=native_str('utf-8'),
                                        convert_unicode=True)
            Base.metadata.create_all(self.engine,  # @UndefinedVariable
                                     checkfirst=True)
            # enable verbosity after table creations
            self.engine.echo = debug
            self.session = sessionmaker(bind=self.engine)
        else:
            self.session = session

    def get_network_ids(self):
        """
        Fetches all possible network id's.
        """
        session = self.session()
        query = session.query(WaveformChannel.network)
        query = query.group_by(WaveformChannel.network)
        results = query.all()
        session.close()
        return [r[0] for r in results if len(r) == 1]

    def get_station_ids(self, network=None):
        """
        Fetches all possible station id's.

        :type network: str, optional
        :param network: Filter result by given network id if given. Defaults
            to ``None``.
        """
        session = self.session()
        query = session.query(WaveformChannel.station)
        if network:
            query = query.filter(WaveformChannel.network == network)
        query = query.group_by(WaveformChannel.station)
        results = query.all()
        session.close()
        return [r[0] for r in results if len(r) == 1]

    def get_location_ids(self, network=None, station=None):
        """
        Fetches all possible location id's.

        :type network: str, optional
        :param network: Filter result by given network id if given. Defaults
            to ``None``.
        :type station: str, optional
        :param station: Filter result by given station id if given. Defaults
            to ``None``.
        """
        session = self.session()
        query = session.query(WaveformChannel.location)
        if network:
            query = query.filter(WaveformChannel.network == network)
        if station:
            query = query.filter(WaveformChannel.station == station)
        query = query.group_by(WaveformChannel.location)
        results = query.all()
        session.close()
        return [r[0] for r in results if len(r) == 1]

    def get_channel_ids(self, network=None, station=None, location=None):
        """
        Fetches all possible channel id's.

        :type network: str, optional
        :param network: Filter result by given network id if given. Defaults
            to ``None``.
        :type station: str, optional
        :param station: Filter result by given station id if given. Defaults
            to ``None``.
        :type location: str, optional
        :param location: Filter result by given location id if given. Defaults
            to ``None``.
        """
        session = self.session()
        query = session.query(WaveformChannel.channel)
        if network:
            query = query.filter(WaveformChannel.network == network)
        if station:
            query = query.filter(WaveformChannel.station == station)
        if location:
            query = query.filter(WaveformChannel.location == location)
        query = query.group_by(WaveformChannel.channel)
        results = query.all()
        session.close()
        return [r[0] for r in results if len(r) == 1]

    def get_endtimes(self, network=None, station=None, location=None,
                     channel=None):
        """
        Generates a list of last end times for each channel.
        """
        # build up query
        session = self.session()
        query = session.query(
            WaveformChannel.network, WaveformChannel.station,
            WaveformChannel.location, WaveformChannel.channel,
            func.max(WaveformChannel.endtime).label('latency')
        )
        query = query.group_by(
            WaveformChannel.network, WaveformChannel.station,
            WaveformChannel.location, WaveformChannel.channel
        )
        # process arguments
        kwargs = {'network': network, 'station': station,
                  'location': location, 'channel': channel}
        for key, value in kwargs.items():
            if value is None:
                continue
            col = getattr(WaveformChannel, key)
            if '*' in value or '?' in value:
                value = value.replace('?', '_')
                value = value.replace('*', '%')
                query = query.filter(col.like(value))
            else:
                query = query.filter(col == value)
        results = query.all()
        session.close()
        adict = {}
        for result in results:
            key = '%s.%s.%s.%s' % (result[0], result[1], result[2], result[3])
            adict[key] = UTCDateTime(result[4])
        return adict

    def get_waveform_path(self, network=None, station=None, location=None,
                          channel=None, starttime=None, endtime=None):
        """
        Generates a list of available waveform files.
        """
        # build up query
        session = self.session()
        query = session.query(WaveformPath.path,
                              WaveformFile.file,
                              WaveformChannel.network,
                              WaveformChannel.station,
                              WaveformChannel.location,
                              WaveformChannel.channel)
        query = query.filter(WaveformPath.id == WaveformFile.path_id)
        query = query.filter(WaveformFile.id == WaveformChannel.file_id)
        # process arguments
        kwargs = {'network': network, 'station': station,
                  'location': location, 'channel': channel}
        for key, value in kwargs.items():
            if value is None:
                continue
            col = getattr(WaveformChannel, key)
            if '*' in value or '?' in value:
                value = value.replace('?', '_')
                value = value.replace('*', '%')
                query = query.filter(col.like(value))
            else:
                query = query.filter(col == value)
        # start and end time
        try:
            starttime = UTCDateTime(starttime)
        except Exception:
            starttime = UTCDateTime() - 60 * 20
        finally:
            query = query.filter(WaveformChannel.endtime > starttime.datetime)
        try:
            endtime = UTCDateTime(endtime)
        except Exception:
            # 10 minutes
            endtime = UTCDateTime()
        finally:
            query = query.filter(WaveformChannel.starttime < endtime.datetime)
        results = query.all()
        session.close()
        # execute query
        file_dict = {}
        for result in results:
            fname = os.path.join(result[0], result[1])
            key = '%s.%s.%s.%s' % (result[2], result[3], result[4], result[5])
            file_dict.setdefault(key, []).append(fname)
        return file_dict

    def get_preview(self, trace_ids=[], starttime=None, endtime=None,
                    network=None, station=None, location=None, channel=None,
                    pad=False):
        """
        Returns the preview trace.
        """
        # build up query
        session = self.session()
        query = session.query(WaveformChannel)
        # start and end time
        try:
            starttime = UTCDateTime(starttime)
        except Exception:
            starttime = UTCDateTime() - 60 * 20
        finally:
            query = query.filter(WaveformChannel.endtime > starttime.datetime)
        try:
            endtime = UTCDateTime(endtime)
        except Exception:
            # 10 minutes
            endtime = UTCDateTime()
        finally:
            query = query.filter(WaveformChannel.starttime < endtime.datetime)
        # process arguments
        if trace_ids:
            # filter over trace id list
            trace_filter = or_()
            for trace_id in trace_ids:
                temp = trace_id.split('.')
                if len(temp) != 4:
                    continue
                trace_filter.append(and_(
                    WaveformChannel.network == temp[0],
                    WaveformChannel.station == temp[1],
                    WaveformChannel.location == temp[2],
                    WaveformChannel.channel == temp[3]))
            if trace_filter.clauses:
                query = query.filter(trace_filter)
        else:
            # filter over network/station/location/channel id
            kwargs = {'network': network, 'station': station,
                      'location': location, 'channel': channel}
            for key, value in kwargs.items():
                if value is None:
                    continue
                col = getattr(WaveformChannel, key)
                if '*' in value or '?' in value:
                    value = value.replace('?', '_')
                    value = value.replace('*', '%')
                    query = query.filter(col.like(value))
                else:
                    query = query.filter(col == value)
        # execute query
        results = query.all()
        session.close()
        # create Stream
        st = Stream()
        for result in results:
            preview = result.get_preview()
            st.append(preview)
        # merge and trim
        st = merge_previews(st)
        st.trim(starttime, endtime, pad=pad)
        return st
