# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: obspyck_pyqt.py
#  Purpose: PyQt GUI for obspyck. This file parses all options and
#           launches the actual interface.
#   Author: Tobias Megies, Lion Krischer
#    Email: megies@geophysik.uni-muenchen.de
#  License: GPLv2
#
# Copyright (C) 2009, 2010 Tobias Megies, Lion Krischer
#---------------------------------------------------------------------

from PyQt4 import QtCore, QtGui

import fnmatch
from obspy.core import read, UTCDateTime
from obspy.seishub import Client
from optparse import OptionParser
import sys
from urllib2 import URLError

from interface import Ui_MainWindow
from obspyck import ObsPyck


class Obspyck(QtGui.QMainWindow):
    """
    Main Window with the design loaded from the Qt Designer. Also inits a
    Picker object to handle the actual picking.

    client, stream and options need to be determined by parsing the command
    line.
    """
    def __init__(self, client, stream, options):
        """
        Standard init.
        """
        QtGui.QMainWindow.__init__(self)
        # Init the interface. All GUI elements will be accessible via
        # self.interface.name_of_element.
        self.interface = Ui_MainWindow()
        self.interface.setupUi(self)
        # Small adjustments to the margins around the buttons. I couldn't
        # figure out how to do this in the Qt Designer.
        self.interface.leftVerticalLayout.setContentsMargins(\
                            QtCore.QMargins(1,1,1,1))
        self.interface.leftVerticalLayout.setSpacing(1)
        # Add write methods to the stdout and stderr displays to enable the
        # redirections later on.
        self._enableTextBrowserWrite()
        # Init the actual picker object which will handle the rest.
        self.picker = ObsPyck(self.interface, client, stream, options)

    def _enableTextBrowserWrite(self):
        """
        Add write methods to both text browsers to be able to redirect the
        stdout and sdterr to them.
        """
        self.interface.stderrPlainTextEdit.write =\
            self.interface.stderrPlainTextEdit.appendPlainText
        self.interface.stdoutPlainTextEdit.write =\
            self.interface.stdoutPlainTextEdit.appendPlainText

def main():
    """
    Gets executed when the program starts.
    """
        streams = []
        sta_fetched = set()
        for id in options.ids.split(","):
            net, sta_wildcard, loc, cha = id.split(".")
            # Catch non reacheable server.
            try:
                stationIds = client.waveform.getStationIds(network_id=net)
            except URLError:
                print 'Error connecting to %s.\n' % baseurl +\
                      'Please check your connection'
                return
            for sta in stationIds:
                if not fnmatch.fnmatch(sta, sta_wildcard):
                    continue
                # make sure we dont fetch a single station of
                # one network twice (could happen with wildcards)
                net_sta = "%s:%s" % (net, sta)
                if net_sta in sta_fetched:
                    print net_sta, "was already retrieved. Skipping!"
                    continue
                try:
                    st = client.waveform.getWaveform(net, sta, loc, cha, t,
                                                     t + options.duration,
                                                     apply_filter=True)
                    print net_sta, "fetched successfully."
                    sta_fetched.add(net_sta)
                except:
                    print net_sta, "could not be retrieved. Skipping!"
                    continue
                st.sort()
                st.reverse()
                streams.append(st)
        # sort streams by station name
        streams = sorted(streams, key=lambda st: st[0].stats['station'])

    print 'Finished preprocessing...'

    # Create the GUI application
    qApp = QtGui.QApplication(sys.argv)
    main_window = Obspyck(client, streams, options)
    # Make the application accessible from the main window.
    main_window.qApp = qApp
    # Start maximized.
    main_window.show()
    # start the Qt main loop execution, exiting from this script with
    # the same return code of Qt the application
    sys.exit(qApp.exec_())

if __name__ == "__main__":
    main()
