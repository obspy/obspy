# -*- coding: utf-8 -*-

from lxml.etree import parse
from obspy.core import UTCDateTime
import os
import sqlite3


class EventDB(object):
    """
    Class that handles the event database.

    It contains two tables. One for the events and one for the picks. Each
    event has a unique id that will be used as a key.
    """
    def __init__(self, env):
        self.env = env
        # Database.
        self.db_file = self.env.sqlite_db
        # Create Database and tables if necessary.
        if not os.path.exists(self.db_file):
            self.db = sqlite3.connect(self.db_file)
            # Cursor to be able to execute commands.
            self.c = self.db.cursor()
            self.createTables()
        else:
            self.db = sqlite3.connect(self.db_file)
            # Cursor to be able to execute commands.
            self.c = self.db.cursor()
        # Available tables.
        self.tables = ['events', 'picks']

    def getPickAndEventCount(self, starttime=None, endtime=None):
        """
        Reads the database and returns a tuple with the first number being the
        number of events in the timeframe and the second being the number of
        picks.

        :type starttime: obspy.core.UTCDateTime
        :param starttime: Start of the timeframe. If none is given, it will be
                          read from self.env.
        :type endtime: obspy.core.UTCDateTime
        :param endtime: End of the timeframe. If none is given, it will be
                        read from self.env.
        """
        if not starttime:
            starttime = self.env.starttime
        if not endtime:
            endtime = self.env.endtime
        start = '"%s"' % str(starttime)
        end = '"%s"' % str(endtime)
        command = '''SELECT event_id FROM picks WHERE time > %s
        AND time < %s''' % (start, end)
        self.c.execute(command)
        self.db.commit()
        picks = self.c.fetchall()
        return (len(set(picks)), len(picks))

    def getPicksForChannel(self, channel_id):
        """
        Returns a list with a dictionary for each pick for the station.

        :type channel_id: String
        :param channel_id: Channel id in the form
                           network.station.location.channel
        """
        network, station, location, channel = channel_id.split('.')
        # First get all picks and then the corresponding events. One call for
        # all is too slow for some reason.
        #msg =  '''SELECT e.event_id, e.origin_time, e.origin_latitude,
        #          e.origin_longitude, e.origin_depth, p.time, p.phaseHint,
        #          p.polarity, e.magnitude, e.magnitude_type, p.network,
        #          p.station, p.location, p.channel FROM events e, picks p WHERE 
        #          (p.network = "%s") AND (p.station = "%s") AND
        #          (p.location = "%s") AND (p.channel = "%s") AND e.event_id =
        #          p.event_id''' % \
        #          (network, station, location, channel)
        #import time
        #a = time.time()
        #self.c.execute(msg)
        #self.db.commit()
        #print 'Time taken: ', time.time()-a
        msg = '''select event_id, time, phaseHint, polarity from picks WHERE
                 network = "%s" AND station = "%s" AND
                 location = "%s" AND channel = "%s"''' % \
                 (network, station, location, channel)
        self.c.execute(msg)
        self.db.commit()
        results = self.c.fetchall()
        result_dicts = []
        for result in results:
            # XXX: Ugly workaround for non existing pick times..need better
            # error handling.
            try:
                event = self.getEvent(result[0])
            except:
                pass
            r_dict = {'event_id': result[0],
                      'origin_time': event['time'],
                      'origin_latitude': event['latitude'],
                      'origin_longitude': event['longitude'],
                      'origin_depth': event['depth'],
                      'time': UTCDateTime(result[1]),
                      'phaseHint': result[2], 'polarity': result[3],
                      'magnitude': event['magnitude'],
                      'magnitude_type': event['magnitude_type'],
                      'event_type' : event['event_type']}
            result_dicts.append(r_dict)
        return result_dicts

    def getEventsInTimeSpan(self, starttime=None, endtime=None):
        """
        Returns the events.
        """
        if not starttime:
            starttime = self.env.starttime
        if not endtime:
            endtime = self.env.endtime
        starttime = str(starttime)
        endtime = str(endtime)
        msg = '''SELECT event_id, event_type, magnitude, origin_time,
                 origin_latitude, origin_longitude FROM events
                 WHERE origin_time > "%s" and origin_time < "%s"''' % \
                         (starttime, endtime)
        self.c.execute(msg)
        self.db.commit()
        events = self.c.fetchall()
        evs = []
        for event in events:
            evs.append({'event_id': event[0],
             'event_type': event[1],
             'magnitude': event[2],
             'origin_time': UTCDateTime(event[3]),
             'origin_latitude': event[4],
             'origin_longitude': event[5] })
        return evs

    def getEvent(self, event_id):
        """
        Returns a dictionary containing information about event event_id.

        :type event_id: String
        :param event_id: Event id of the event
        """
        msg = '''SELECT origin_time, origin_latitude,
                  origin_longitude, origin_depth, magnitude, magnitude_type,
                  event_type
                  FROM events WHERE event_id = "%s"''' % event_id
        self.c.execute(msg)
        self.db.commit()
        event = self.c.fetchall()
        if len(event) > 1:
            raise
        event = event[0]
        try:
            magnitude = float(event[4])
        except:
            magnitude = 0.0
        return {'time': UTCDateTime(event[0]), 'latitude': float(event[1]),
                'longitude': float(event[2]), 'depth': float(event[3]),
                'magnitude': magnitude, 'magnitude_type': event[5],
                'event_type': event[6]}

    def getFilesAndModification(self):
        """
        Returns a list with all event files and the date of the last
        modification of the file.
        """
        command = '''SELECT event_file, event_file_last_modified FROM events'''
        self.c.execute(command)
        self.db.commit()
        files = self.c.fetchall()
        return [(os.path.basename(_i[0]), _i[1]) for _i in files]

    def getChannelsWithPicks(self, starttime=None, endtime=None):
        """
        Reads the database and returns a list with all channel ids that have
        picks in the chosen timeframe.

        :type starttime: obspy.core.UTCDateTime
        :param starttime: Start of the timeframe. If none is given, it will be
                          read from self.env.
        :type endtime: obspy.core.UTCDateTime
        :param endtime: End of the timeframe. If none is given, it will be
                        read from self.env.
        """
        if not starttime:
            starttime = self.env.starttime
        if not endtime:
            endtime = self.env.endtime
        start = '"%s"' % str(starttime)
        end = '"%s"' % str(endtime)
        command = '''SELECT network, station, location, channel FROM picks
                     WHERE time > %s AND time < %s''' % (start, end)
        self.c.execute(command)
        self.db.commit()
        picks = self.c.fetchall()
        channels = list(set([str('.'.join(_i)) for _i in picks]))
        channels.sort()
        return channels

    def addEventFile(self, file, last_modification_datetime, filename):
        """
        Adds an event xml file to the database. If the event id already exists
        in the database, the existing event and all associated picks will be
        deleted in the database.

        :type file: String
        :param file: Filename of the event XML file to be added.
        :type last_modification_datetime: obspy.core.UTCDateTime
        :type last_modification_datetime: Last modification as returned by the
                                          SeisHub server.
        """
        # Parse the StringIO.
        tree = parse(file)
        # Get root element.
        root = tree.getroot()
        # Discard wrong xml file.
        if root.tag != 'event':
            print '%s is not a valid event file.' % event
            return
        # Get the event id.
        event_id = root.xpath('event_id/value')[0].text
        # Create dictionary that will later be used to create the database
        # entry.
        event_dict = {'event_file': filename,
                      'event_file_last_modified': str(last_modification_datetime),
                      'event_id': event_id}
        # Remove any old references to this event.
        self.removeEventAndPicks(event_id)
        # Parse XML and write event dictionary. Use try/except blocks for all
        # optional tags.
        # XXX: Better way than try/except?
        event_dict['event_type'] = root.xpath('event_type/value')[0].text
        try:
            event_dict['user'] = root.xpath('event_type/user')[0].text
        except: pass
        try:
            event_dict['puplic'] = root.xpath('event_type/puplic')[0].text
        except: pass
        try:
            event_dict['origin_time'] = \
                    str(UTCDateTime(root.xpath('origin/time/value')[0].text))
        except:
            pass
        try:
            event_dict['origin_time_uncertainty'] = \
                    UTCDateTime(root.xpath('origin/time/uncertainty')[0].text)
        except: pass
        try:
            event_dict['origin_latitude'] = \
                    float(root.xpath('origin/latitude/value')[0].text)
        except: pass
        try:
            event_dict['origin_latitude_uncertainty'] = \
                    float(root.xpath('origin/latitude/uncertainty')[0].text)
        except: pass
        try:
            event_dict['origin_longitude'] = \
                    float(root.xpath('origin/longitude/value')[0].text)
        except: pass
        try:
            event_dict['origin_longitude_uncertainty'] = \
                    float(root.xpath('origin/longitude/uncertainty')[0].text)
        except: pass
        try:
            event_dict['origin_depth'] = \
                    float(root.xpath('origin/depth/value')[0].text)
        except: pass
        try:
            event_dict['origin_depth_uncertainty'] = \
                    float(root.xpath('origin/depth/uncertainty')[0].text)
        except: pass
        try:
            event_dict['origin_depth_type'] = \
                    root.xpath('origin/depth_type')[0].text
        except: pass
        try:
            event_dict['origin_earth_mod'] = \
                    root.xpath('origin/earth_mod')[0].text
        except: pass
        try:
            event_dict['magnitude'] = \
                    float(root.xpath('magnitude/mag/value')[0].text)
        except: pass
        try:
            event_dict['magnitude_uncertainty'] = \
                    float(root.xpath('magnitude/mag/uncertainty')[0].text)
        except: pass
        try:
            event_dict['magnitude_program'] = \
                    root.xpath('magnitude/program')[0].text
        except: pass
        try:
            event_dict['magnitude_type'] = \
                    root.xpath('magnitude/type')[0].text
        except: pass
        # Write event.
        self.writeToDB('events', event_dict)
        # Loop over all picks.
        picks = root.xpath('pick')
        for pick in picks:
            pick_dict = {'event_id': event_id}
            chan_id = pick.xpath('waveform')[0]
            # XXX: Other way??
            def get_id(id_type):
                return chan_id.values()[chan_id.keys().index(id_type)]
            try:
                pick_dict['network'] = get_id('networkCode')
                pick_dict['station'] = get_id('stationCode')
                pick_dict['location'] = get_id('locationCode')
                pick_dict['channel'] = get_id('channelCode')
                pick_dict['time'] = pick.xpath('time/value')[0].text
            except:
                print 'Problem with %s' % event_id
                continue
            # Account for weird ids.
            if not pick_dict['network']:
                pick_dict['network'] = 'BW'
            if pick_dict['location'] == '01' and pick_dict['channel'] == 'HHZ'\
               and 'earthworm' in filename:
                pick_dict['location'] = ''
                pick_dict['channel'] = 'EHZ'
            try:
                pick_dict['time_uncertainty'] = pick.xpath('time/uncertainty')[0].text
            except: pass
            pick_dict['phaseHint'] = pick.xpath('phaseHint')[0].text
            try:
                pick_dict['onset'] = pick.xpath('onset')[0].text
            except: pass
            try:
                pick_dict['polarity'] = pick.xpath('polarity')[0].text
            except: pass
            try:
                pick_dict['weight'] = float(pick.xpath('weight')[0].text)
            except: pass
            # Write to database.
            self.writeToDB('picks', pick_dict)
        self.db.commit()

    def writeToDB(self, table_name, table_dict):
        """
        Convenience method that writes every key in table_dict to the table
        table_name. This is not safe at all but security is not an issue for this
        application.

        :type table_name: String
        :param table_name: Table that the dictionary will be inserted in.
        :type table_dict: Dictionary
        :param table_dict: Dictionary which just contains keys with a string
                           value that will all be inserted in table_name.
        """
        # Small security checks.
        if table_name not in self.tables:
            raise
        keys = '(%s)' % ','.join(table_dict.keys())
        values = '(%s)' % ','.join([("'%s'" % _i if type(_i) == str or \
                                    _i is None else \
                                     str(_i)) for _i in table_dict.values()])
        sql_com = '''INSERT INTO %s %s VALUES %s''' % (table_name, keys, values)
        self.c.execute(sql_com)

    def removeEventAndPicks(self, event_id):
        """
        Removes event with event_id and all associated picks from the database.

        :type event_id: String
        :param event_id: Event id as written in the XML file for the event to be
                         removed.
        """
        # XXX: Combine into one statement.
        self.c.execute('''DELETE FROM events
            WHERE event_id=:event_id''', {'event_id': event_id})
        self.c.execute('''DELETE FROM picks
            WHERE event_id=:event_id''', {'event_id': event_id})

    def createTables(self):
        """
        Creates the tables if they do not already exists.
        """
        # Create event table.
        self.c.execute('''CREATE TABLE events(
            key INTEGER PRIMARY KEY AUTOINCREMENT,
            event_file TEXT,
            event_file_last_modified TEXT,
            event_id TEXT,
            event_type TEXT,
            user TEXT,
            public TEXT,
            origin_time TEXT,
            origin_time_uncertainty REAL,
            origin_latitude REAL,
            origin_latitude_uncertainty REAL,
            origin_longitude REAL,
            origin_longitude_uncertainty REAL,
            origin_depth REAL,
            origin_depth_uncertainty REAL,
            origin_depth_type TEXT,
            origin_earth_mod TEXT,
            magnitude REAL,
            magnitude_program TEXT,
            magnitude_uncertainty REAL,
            magnitude_type TEXT)''')
        # Create pick table.
        self.c.execute('''CREATE TABLE picks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT,
            network TEXT,
            station TEXT,
            location TEXT,
            channel TEXT,
            time TEXT,
            time_uncertainty REAL,
            phaseHINT TEXT,
            onset TEXT,
            polarity TEXT,
            weight REAL)''')
        # Create indexes.
        self.c.execute('''CREATE INDEX idx_trace_id 
            ON picks (network, station, location, channel)''')
        self.c.execute('''CREATE INDEX idx_event_id 
            ON events (event_id)''')
        self.db.commit()
