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
    """
    def __init__(self, *args, **kwargs):
        self.config_file = 'dbviewer.cfg'
        # Read the configuration file.
        DBConfigParser(env=self)
        
        # Create the cache directory if it does not exists.
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Set the maximum zoom level in seconds. Add +1 just to be sure it
        # works.
        # XXX: Not yet dynamically adjusted.
        self.maximum_zoom_level = self.preview_delta * (self.detail+1)
        
        # Resources directory.
        self.res_dir = kwargs.get('res_dir', 'resources')

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

    def setSplash(self, text):
        """
        Updates the splash screen.
        """
        self.splash.showMessage(text, QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom,
                       QtCore.Qt.black)
        self.qApp.processEvents()
