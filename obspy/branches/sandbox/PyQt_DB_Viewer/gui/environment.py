from PyQt4 import QtCore
from obspy.core import UTCDateTime
import os
from seishub import Seishub
from waveform_handler import WaveformHandler
from event_db import EventDB

class Environment(object):
    """
    Simple class that stores all global variables and other stuff important for
    many parts of the code.
    """
    def __init__(self, *args, **kwargs):
        # Cache directory.
        self.cache_dir = kwargs.get('cache_dir', 'cache')
        # Resources directory.
        self.res_dir = kwargs.get('res_dir', 'resources')
        # SQLite event database file.
        self.sqlite_db = kwargs.get('sqlite_db', os.path.join(self.res_dir,
                                                'events.db'))
        # Handle the times for the plots.
        self.starttime = UTCDateTime(2009, 1, 1)
        self.endtime = UTCDateTime(2010, 1, 1) - 1
        self.time_range = self.endtime - self.starttime
        # Debug.
        self.debug = kwargs.get('debug', False)
        # Details of the plots.
        self.detail = kwargs.get('detail', 100)
        # How much data is loaded before and after the requested timespan in
        # minutes.
        self.buffer = kwargs.get('buffer', 120)
        self.buffer *= 60
        # Seishub Server.
        self.seishub_server = kwargs.get('seishub_server',
                                         'http://teide:8080')
        # Scale of the plots.
        self.log_scale = False
        # Start the SeisHub class.
        self.seishub = Seishub(env = self)
        # Init the Database first. Does not need a network connection. Should
        # always work.
        self.db = EventDB(env = self)
        # Start the waveform handler.
        self.handler = WaveformHandler(env = self)

    def setSplash(self, text):
        """
        Updates the splash screen.
        """
        self.splash.showMessage(text, QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom,
                       QtCore.Qt.black) 
        self.qApp.processEvents()
