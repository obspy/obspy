#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gui_element import GUIElement
from obspy.core import read, Stream, UTCDateTime
from obspy.seishub import Client
import os
import pickle

class Seishub(GUIElement):
    """
    Class that handles all waveform reading and transforming operations.
    """
    def __init__(self, *args, **kwargs):
        """
        Prepare the object.
        """
        super(Seishub, self).__init__(self, **kwargs)
        # Set some class variables.
        self.server = self.env.seishub_server
        self.cache_dir = self.env.cache_dir
        self.pickle_file = os.path.join(self.cache_dir, 'pickle_dict')
        # Connect to server.
        self.connectToServer()
        # Load the Index of the SeisHub Server.
        self.getIndex()
        # Write to environment. Delete the unnecessary keys.
        del self.networks['Date']
        del self.networks['Server']
        self.env.networks = self.networks

    def getPreview(self, network, station, location, channel, starttime,
                   endtime):
        if not starttime:
            starttime = self.win.starttime
        if not endtime:
            endtime = self.win.endtime
        if self.env.debug:
            print '==========================================================='
            print 'Requesting %s.%s.%s.%s' % \
                    (network, station, location, channel)
            print ' * from %s-%s' % (starttime, endtime)
            print ' * SeisHub Server = %s' % self.env.seishub_server
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
        self.client = Client(base_url = self.server)
        self.ping()

    def ping(self):
        """
        Ping the server.
        """
        try:
            self.client.ping()
        except:
            msg = 'No connection to SeisHub server %s' % self.server
            self.win.default_error = msg

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
        file = open(self.pickle_file, 'r')
        self.networks = pickle.load(file)
        file.close()

    def reload_infos(self):
        """
        Reads all available networks, stations, ... from the seishub server.
        """
        self.networks = {}
        networks = self.client.waveform.getNetworkIds()
        # Get stations.
        for key in networks:
            if not key:
                continue
            self.networks[key] = {}
            stations = self.client.waveform.getStationIds(network_id=key)
            for station in stations:
                if not station:
                    continue
                self.networks[key][station] = {}
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
        file = open(self.pickle_file, 'wb')
        pickle.dump(self.networks, file, protocol = 2)
        file.close()
