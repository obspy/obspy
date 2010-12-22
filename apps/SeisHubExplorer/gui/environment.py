# -*- coding: utf-8 -*-

from PyQt4 import QtCore
from obspy.core import UTCDateTime
import os
from seishub import Seishub
from waveform_handler import WaveformHandler
from channel_lists import ChannelListParser
from event_db import EventDB
from config import DBConfigParser


class Environment(object):
    """
    Simple class that stores all global variables and other stuff important for
    many parts of the code.

    Sets up the environment.

    Setis all variables necessary for many parts of the application.
    Reads the config file.
    Inits the SQLite database.
    Parses the channel groups.
    Inits the Seishub Connection.
    Starts the waveform handler.
    """
    def __init__(self, *args, **kwargs):
        self._getApplicationHomeDir()

        # Specify where the config file is supposed to be.
        self.config_file = os.path.join(self.home_dir, 'config.cfg')

        # Get the root directory of the application.
        self.root_dir = os.path.split(os.path.abspath(\
                             os.path.dirname(__file__)))[0]

        # Resources directory.
        self.res_dir = os.path.join(self.root_dir, 'resources')

        # Read/write the configuration file.
        DBConfigParser(env=self)

        # Create the cache directory if it does not exists.
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Temporary resources directory.
        self.temp_res_dir = os.path.join(self.cache_dir, 'temporary_resources')
        if not os.path.exists(self.temp_res_dir):
            os.makedirs(self.temp_res_dir)

        # Set the maximum zoom level in seconds. Add +1 just to be sure it
        # works.
        # XXX: Not yet dynamically adjusted.
        self.maximum_zoom_level = self.preview_delta * (self.detail + 1)

        # SQLite event database file.
        self.sqlite_db = kwargs.get('sqlite_db', os.path.join(self.cache_dir,
                                                'events.db'))
        self.css = os.path.join(self.res_dir, 'seishub.css')
        # The xml file with the stored lists.
        self.channel_lists_xml = os.path.join(self.cache_dir,
                                             'channel_lists.xml')
        self.channel_lists = {}
        # Parse the Waveform List.
        self.channel_lists_parser = ChannelListParser(env=self)
        # Network index pickled dict.
        self.network_index = os.path.join(self.cache_dir, 'networks_index.pickle')

        # Calculate the time range. This is needed a lot and therefore kept as
        # a variable.
        self.time_range = self.endtime - self.starttime

        # Convert buffer to seconds.
        self.buffer *= 86400

        # Start the SeisHub class.
        self.seishub = Seishub(env=self)
        # Init the Database first. Does not need a network connection. Should
        # always work.
        self.db = EventDB(env=self)
        # Start the waveform handler.
        self.handler = WaveformHandler(env=self)

    def _getApplicationHomeDir(self):
        """
        Creates a .seishub_explorer in the users home dir for all cached files
        and sets self.home_dir to it.
        If the folder already exists it will not create one.
        """
        self.home_dir = os.path.abspath(os.path.expanduser('~'))
        self.home_dir = os.path.join(self.home_dir, '.seishub_explorer')
        if not os.path.exists(self.home_dir):
            os.makedirs(self.home_dir)

    def setSplash(self, text):
        """
        Updates the splash screen.
        """
        self.splash.showMessage(text, QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom,
                       QtCore.Qt.black)
        self.qApp.processEvents()
