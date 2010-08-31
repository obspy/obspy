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
from obspyck import ObsPyckGUI


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
        self.picker = Picker(self.interface, client, stream, options)

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
    # Option parser stuff.
    usage = "USAGE: %prog -t <datetime> -d <duration> -i <channelids>"
    parser = OptionParser(usage)
    parser.add_option("-t", "--time", dest="time",
                      help="Starttime of seismogram to retrieve. It takes a "
                           "string which UTCDateTime can convert. "
                           "E.g. '2010-01-10T05:00:00'",
                      default='2009-07-21T04:33:00')
    parser.add_option("-d", "--duration", type="float", dest="duration",
                      help="Duration of seismogram in seconds",
                      default=120)
    parser.add_option("-i", "--ids", dest="ids",
                      help="Ids to retrieve, star for channel and "
                           "wildcards for stations are allowed, e.g. "
                           "'BW.RJOB..EH*,BW.RM?*..EH*'",
                      default='BW.RJOB..EH*,BW.RMOA..EH*')
    # XXX: Change to teide:8080!
    parser.add_option("-s", "--servername", dest="servername",
                      help="Servername of the seishub server",
                      default='localhost')
    parser.add_option("-p", "--port", type="int", dest="port",
                      help="Port of the seishub server",
                      default=7777)
    parser.add_option("--user", dest="user", default='obspyck',
                      help="Username for seishub server")
    parser.add_option("--password", dest="password", default='obspyck',
                      help="Password for seishub server")
    parser.add_option("--timeout", dest="timeout", type="int", default=10,
                      help="Timeout for seishub server")
    parser.add_option("-l", "--local", action="store_true", dest="local",
                      default=False,
                      help="use local files for design purposes " + \
                           "(overwrites event xml with fixed id)")
    parser.add_option("-k", "--keys", action="store_true", dest="keybindings",
                      default=False, help="Show keybindings and quit")
    parser.add_option("--lowpass", type="float", dest="lowpass",
                      help="Frequency for Lowpass-Slider", default=20.)
    parser.add_option("--highpass", type="float", dest="highpass",
                      help="Frequency for Highpass-Slider", default=1.)
    parser.add_option("--nozeromean", action="store_true", dest="nozeromean",
                      help="Deactivate offset removal of traces",
                      default=False)
    parser.add_option("--pluginpath", dest="pluginpath",
                      default="/baysoft/obspyck/",
                      help="Path to local directory containing the folders" + \
                           " with the files for the external programs")
    parser.add_option("--starttime-offset", type="float",
                      dest="starttime_offset", default=0.0,
                      help="Offset to add to specified starttime in " + \
                      "seconds. Thus a time from an automatic picker " + \
                      "can be used with a specified offset for the " + \
                      "starttime. E.g. to request a waveform starting 30 " + \
                      "seconds earlier than the specified time use -30.")
    parser.add_option("-m", "--merge", type="string", dest="merge",
                      help="After fetching the streams from seishub run a " + \
                      "merge operation on every stream. If not done, " + \
                      "streams with gaps and therefore more traces per " + \
                      "channel get discarded.\nTwo methods are supported " + \
                      "(see http://svn.geophysik.uni-muenchen.de/obspy/" + \
                      "docs/packages/auto/obspy.core.trace.Trace.__add__" + \
                      ".html for details)\n" + \
                      "  \"safe\": overlaps are discarded completely\n" + \
                      "  \"overwrite\": the second trace is used for " + \
                      "overlapping parts of the trace",
                      default="")
    (options, args) = parser.parse_args()
    for req in ['-d','-t','-i']:
        if not getattr(parser.values,parser.get_option(req).dest):
            parser.print_help()
            return
    # XXX: Remove!
    options.local = True
    
    # If keybindings option is set, don't prepare streams.
    # We then only print the keybindings and exit.
    if options.keybindings:
        streams = None
        client = None
    # If local option is set we read the locally stored traces.
    # Just for testing purposes, sent event xmls always overwrite the same xml.
    elif options.local:
        streams=[]
        streams.append(read('20091227_105240_Z.RJOB'))
        streams[0].append(read('20091227_105240_N.RJOB')[0])
        streams[0].append(read('20091227_105240_E.RJOB')[0])
        streams.append(read('20091227_105240_Z.RMOA'))
        streams[1].append(read('20091227_105240_N.RMOA')[0])
        streams[1].append(read('20091227_105240_E.RMOA')[0])
        streams.append(read('20091227_105240_Z.RNON'))
        streams[2].append(read('20091227_105240_N.RNON')[0])
        streams[2].append(read('20091227_105240_E.RNON')[0])
        streams.append(read('20091227_105240_Z.RTBE'))
        streams[3].append(read('20091227_105240_N.RTBE')[0])
        streams[3].append(read('20091227_105240_E.RTBE')[0])
        streams.append(read('20091227_105240_Z.RWMO'))
        streams[4].append(read('20091227_105240_N.RWMO')[0])
        streams[4].append(read('20091227_105240_E.RWMO')[0])
        streams.append(read('20091227_105240_Z.RWMO'))
        streams[5].append(read('20091227_105240_N.RWMO')[0])
        streams[5].append(read('20091227_105240_E.RWMO')[0])
        baseurl = "http://" + options.servername + ":%i" % options.port
        client = Client(base_url=baseurl, user=options.user,
                        password=options.password, timeout=options.timeout)
        print baseurl
        print 'got here'
    # Otherwise connect to Seishub.
    else:
        try:
            t = UTCDateTime(options.time)
            t = t + options.starttime_offset
            baseurl = "http://" + options.servername + ":%i" % options.port
            client = Client(base_url=baseurl, user=options.user,
                            password=options.password, timeout=options.timeout)
        except:
            print "Error while connecting to server!"
            raise
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
