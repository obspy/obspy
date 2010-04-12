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
        self.pickle_file = os.path.join(self.cache_dir, 'pickle_dict')
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
            self.updateEventListFromSeishub(self.env.starttime, self.env.endtime)
            # Download and cache events.
            self.downloadAndCacheEvents()
            self.parseEvents()
        # Otherwise read cached events and get those for the given time_span.
        else:
            msg = 'No connection to server. Will only use cached events..'
            self.env.setSplash(msg)

    def getPreview(self, network, station, location, channel, starttime,
                   endtime):
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
        stream = self.client.waveform.getPreview(network, station, location, channel,
                                      starttime, endtime)
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
        self.client = Client(base_url=self.server)
        if self.ping():
            self.online = True
        else:
            self.online = False

    def ping(self):
        """
        Ping the server.
        """
        try:
            self.client.ping()
            return True
        except:
            return False

    def getIndex(self):
        """
        Gets the index if not available or reads the pickled file.
        """
        # Create Cache path if it does not exist.
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        if not os.path.exists(self.pickle_file):
            self.reload_infos()
        # Read the dict.
        file = open(self.pickle_file, 'rb')
        self.networks = pickle.load(file)
        file.close()

    def reload_infos(self):
        """
        Reads all available networks, stations, ... from the seishub server.
        """
        print 'Reloading infos...'
        self.networks = {}
        networks = self.client.waveform.getNetworkIds()
        print networks
        # Get stations.
        for key in networks:
            print 'Network:', key
            if not key:
                continue
            self.networks[key] = {}
            stations = self.client.waveform.getStationIds(network_id=key)
            for station in stations:
                print 'Station:', station
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
        print 'Done...start pickling...'
        # Add current date to Dictionary.
        self.networks['Date'] = UTCDateTime()
        # Also add the server to it.
        self.networks['Server'] = self.client.base_url
        # Open file.
        file = open(self.pickle_file, 'wb')
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
        auth = base64.encodestring('%s:%s' % ("admin", "admin"))[:-1]
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

    def removePicksOutofRange(self):
        """
        Reads all cached events and returns those with at least one pick in the
        selected time frame.
        """
        self.seishubEventCount = 0
        picks = self.picks
        self.picks = {}
        self.pick_count = 0
        events = set()
        for channel in picks.keys():
            chan = picks[channel]
            chan_list = []
            for pick in chan:
                if pick['pick']['time'] >= self.env.starttime and \
                   pick['pick']['time'] <= self.env.endtime:
                    chan_list.append(pick)
                    events.add(pick['meta']['id'])
                    self.pick_count += 1
            # Add to list if if picks are in the selected time frame.
            if chan_list:
                self.picks[channel] = chan_list
        self.seishubEventCount = len(events)

    def downloadAndCacheEvents(self):
        """
        Uses the event list and looks all events. If they are already cached do
        not download them again.
        """
        # Nothing to do if no events.
        if not self.seishubEventCount:
            return
        event_cache = os.path.join(self.cache_dir, 'event')
        # Create path if necessary.
        if not os.path.exists(event_cache):
            os.mkdir(event_cache)
        # Get a list of xml files in there.
        path = os.path.join(event_cache, '*')
        files = glob(path)
        files = [os.path.split(file)[1] for file in files]
        # Loop over all events.
        amount = len(self.seishubEventList)
        # Check if event already in database.
        self.db_files = self.env.db.getFilesAndModification()
        for _i, event in enumerate(self.seishubEventList):
            self.env.setSplash('Loading event %i of %i' % ((_i + 1), amount))
            if (event[0], str(event[1])) in self.db_files:
                continue
            # If it exists, skip it.
            if event[0] in files:
                continue
            url = self.server + "/xml/seismology/event/%s" % event[0]
            if self.env.debug:
                print 'Requesting %s' % url
            req = urllib2.Request(url)
            auth = base64.encodestring('%s:%s' % ("admin", "admin"))[:-1]
            req.add_header("Authorization", "Basic %s" % auth)
            # Big try except for lost connection or other issues.
            try:
                e = urllib2.urlopen(req)
            except:
                print 'Error requesting %s' % url
                continue
            # Write the cached file.
            f = open(os.path.join(event_cache, event[0]), 'w')
            f.write(e.read())
            e.close()
            f.close()
        # Once again to account for events that did not get downloaded.
        self.files = glob(path)
        if self.env.debug:
            print 'Done downloading events.'

    def parseEvents(self):
        """
        Parses all events in self.files and writes them to the database.
        """
        if self.env.debug:
            a = time.time()
            print 'Start parsing events.'
        events_dict = {}
        files = [os.path.split(file)[1] for file in self.files]
        amount = len(self.seishubEventList)
        for _k, event in enumerate(self.seishubEventList):
            self.env.setSplash('Parsing event %i of %i' % ((_k + 1), amount))
            if (event[0], str(event[1])) in self.db_files:
                continue
            # Check if available.
            if not event[0] in files:
                continue
            file = os.path.join(self.cache_dir, 'event', event[0])
            self.env.db.addEventFile(file, event[1])
        if self.env.debug:
            t = time.time() - a
            print 'Parsed %i events with %i picks in %f seconds.' % \
                        (self.seishubEventCount, self.pick_count, t)
