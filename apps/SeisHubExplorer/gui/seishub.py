#!/usr/bin/env python
# -*- coding: utf-8 -*-

from obspy.core import read, Stream, UTCDateTime
from obspy.seishub import Client
import base64
import os
import lxml
from lxml.etree import parse
from glob import glob
import pickle
import time
import urllib2
from StringIO import StringIO

class Seishub(object):
    """
    Class that handles all waveform reading and transforming operations.
    """
    def __init__(self, env, *args, **kwargs):
        """
        Prepare the object.
        """
        self.env = env
        # Set some class variables.
        self.server = env.seishub_server
        self.cache_dir = env.cache_dir
        self.network_index = self.env.network_index
        self.picks = {}
        self.pick_count = 0
        self.pick_programs = set()
        self.seishubEventList = []
        self.seishubEventCount = 0

    def startup(self):
        """
        Gets called once the main loop has been started.
        """
        # Connect to server.
        self.connectToServer()
        # Load the Index of the SeisHub Server.
        self.getIndex()
        # Write to environment. Delete the unnecessary keys.
        del self.networks['Date']
        del self.networks['Server']
        # Get the events for the global time span if a connection exists.
        if self.online:
            self.updateEventListFromSeishub(self.env.event_starttime,
                                            self.env.event_endtime)
            # Download and cache events.
            self.downloadAndParseEvents()
        # Otherwise read cached events and get those for the given time_span.
        else:
            msg = 'No connection to server. Will only use cached events..'
            self.env.setSplash(msg)

    def getPreview(self, network, station, location, channel, starttime,
                   endtime):
        # Return empty stream object if no online connection is available.
        if not self.online:
            if self.env.debug:
                print 'No connection to SeisHub server.'
            return None
        if not starttime:
            starttime = self.env.starttime
        if not endtime:
            endtime = self.env.endtime
        if self.env.debug:
            print '==========================================================='
            print 'Requesting %s.%s.%s.%s' % \
                    (network, station, location, channel)
            print ' * from %s-%s' % (starttime, endtime)
            print ' * SeisHub Server = %s' % self.server
        try:
            stream = self.client.waveform.getPreview(network, station,
                                         location, channel, starttime, endtime)
        # Catch the server errors and return None to indicate that no Stream
        # could be retrieved.
        except urllib2.HTTPError, e:
            msg = '%s while trying to retrieve %s.%s.%s.%s' % (str(e),
                           network, station, location, channel)
            print msg
            # Show the message in the Status Bar for 10 Seconds maximum.
            self.env.st.showMessage(msg, 10000)
            # Create an emtpy Stream object.
            return None

        if self.env.debug:
            print ''
            print ' * Received from Seishub:'
            print stream
            print '==========================================================='
        return stream

    def connectToServer(self):
        """
        Connects to the SeisHub server.
        """
        self.client = Client(base_url=self.server, user=self.env.seishub_user,
                            password=self.env.seishub_password,
                            timeout=self.env.seishub_timeout)
        self.ping()

    def ping(self):
        """
        Ping the server.
        """
        try:
            status = self.client.ping()
            if status:
                self.online = True
                return True
        except:
            pass
        self.online = False
        return False

    def getIndex(self):
        """
        Gets the index if not available or reads the pickled file.
        """
        # Create Cache path if it does not exist.
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        if not os.path.exists(self.network_index):
            self.reload_infos()
        # Read the dict.
        file = open(self.network_index, 'rb')
        self.networks = pickle.load(file)
        file.close()

    def reload_infos(self):
        """
        Reads all available networks, stations, ... from the seishub server.
        """
        if not self.online:
            msg = 'No connection to server. Cannot load the server index.'
            self.env.setSplash(msg)
            return
        msg = 'Loading network index...'
        self.env.setSplash(msg)
        self.networks = {}
        networks = self.client.waveform.getNetworkIds()
        network_count = len(networks)
        # Get stations.
        for _j, key in enumerate(networks):
            if not key:
                continue
            self.networks[key] = {}
            stations = self.client.waveform.getStationIds(network_id=key)
            station_count = len(stations)
            for _i, station in enumerate(stations):
                msg = 'Loading Network[%i/%i]/Station[%i/%i]:  %s.%s' % (_j + 1,
                             network_count, _i+1, station_count, key, station)
                self.env.setSplash(msg)
                if not station:
                    continue
                self.networks[key][station] = {}
                # Append informations about the station.
                info = self.client.station.getList(key, station)
                if len(info):
                    info = info[0]
                    # Need to make normal python objects out of it.
                    # XXX: Maybe rather change obspy.seishub??
                    new_info = {}
                    for _i in info.keys():
                        value = info[_i]
                        if type(value) == lxml.objectify.StringElement:
                            new_info[_i] = str(value)
                        elif type(value) == lxml.objectify.FloatElement:
                            new_info[_i] = float(value)
                        elif type(value) == lxml.objectify.IntElement:
                            new_info[_i] = int(value)
                    self.networks[key][station]['info'] = new_info
                else:
                    self.networks[key][station]['info'] = {}
                # Get locations.
                locations = self.client.waveform.getLocationIds(network_id=key,
                                                        station_id=station)
                for location in locations:
                    channels = self.client.waveform.getChannelIds(\
                        network_id=key , station_id=station,
                        location_id=location)
                    self.networks[key][station][location] = [channels]
        # Add current date to Dictionary.
        self.networks['Date'] = UTCDateTime()
        # Also add the server to it.
        self.networks['Server'] = self.client.base_url
        # Open file.
        file = open(self.network_index, 'wb')
        pickle.dump(self.networks, file, protocol=2)
        file.close()

    def updateEventListFromSeishub(self, starttime, endtime):
        """
        Searches for events in the database and stores a list of resource
        names. All events with at least one pick set in between start- and
        endtime are returned.

        Adapted from obspyck
        
        :param starttime: Start datetime as UTCDateTime
        :param endtime: End datetime as UTCDateTime
        """
        self.env.setSplash('Getting event list from SeisHub...')
        # two search criteria are applied:
        # - first pick of event must be before stream endtime
        # - last pick of event must be after stream starttime
        # thus we get any event with at least one pick in between start/endtime
        url = self.server + "/seismology/event/getList?" + \
            "min_last_pick=%s&max_first_pick=%s" % \
            (str(starttime), str(endtime))
        req = urllib2.Request(url)
        auth = base64.encodestring('%s:%s' % (self.env.seishub_user,
                                              self.env.seishub_password))[:-1]
        req.add_header("Authorization", "Basic %s" % auth)
        f = urllib2.urlopen(req)
        xml = parse(f)
        f.close()
        # populate list with resource names of all available events
        eventList = xml.xpath(u".//resource_name")
        eventList_last_modified = xml.xpath(u".//document_last_modified")
        self.seishubEventList = [(_i.text, UTCDateTime(_j.text)) for _i, _j \
                                 in zip(eventList, eventList_last_modified)]
        self.seishubEventCount = len(self.seishubEventList)

    def downloadAndParseEvents(self):
        """
        Compares the fetched event list with the database and fetches all new
        or updated events and writes them to the database.
        """
        # Nothing to do if no events.
        if not self.seishubEventCount:
            return
        # Loop over all events.
        event_count = len(self.seishubEventList)
        # Get events and Modification Date from database.
        self.db_files = self.env.db.getFilesAndModification()
        for _i, event in enumerate(self.seishubEventList):
            # Update splash screen.
            self.env.setSplash('Loading event %i of %i' % ((_i + 1), event_count))
            if (event[0], str(event[1])) in self.db_files:
                continue
            url = self.server + "/xml/seismology/event/%s" % event[0]
            if self.env.debug:
                print 'Requesting %s' % url
            req = urllib2.Request(url)
            auth = base64.encodestring('%s:%s' % (self.env.seishub_user,
                                              self.env.seishub_password))[:-1]
            req.add_header("Authorization", "Basic %s" % auth)
            # Big try except for lost connection or other issues.
            try:
                e = urllib2.urlopen(req)
            except:
                print 'Error requesting %s' % url
                continue
            # Write StringIO.
            file = StringIO(e.read())
            self.env.db.addEventFile(file, event[1], event[0])
            e.close()
