#!/usr/bin/env python

# Standart Python imports.
from lxml.etree import SubElement as Sub
from lxml.etree import fromstring, Element, parse, tostring
from optparse import OptionParser
import numpy as np
import fnmatch
import shutil
import sys
import os
import platform
import subprocess
import httplib
import base64
import datetime
import time
import urllib2
import tempfile

# ObsPy imports.
from obspy.core import read, UTCDateTime
from obspy.seishub import Client
from obspy.signal.filter import bandpass, bandstop, lowpass, highpass
from obspy.signal.util import utlLonLat, utlGeoKm
from obspy.signal.invsim import estimateMagnitude
from obspy.imaging.spectrogram import spectrogram
from obspy.imaging.beachball import Beachball

# Matplotlib imports.
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure

from plotting import MultiCursor, Plotting
from GUI_connectors import GUI_connectors
#from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
#from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as Toolbar

class Picker(object):
    """
    Class that handles the actual picking.
    """
    def __init__(self, interface, client=None, streams=None, options=None):
        """
        Standard init.

        interface is the link to all gui elements.
        """
        self.client = client
        self.streams = streams
        self.options = options
        # The gui interface.
        self.interface = interface

        # Matplotlib figure.
        self.fig = self.interface.canvas.fig
        # we bind the figure to the FigureCanvas, so that it will be
        # drawn using the specific backend graphic functions
        self.canv = self.fig.canvas

        # The plotting class.
        self.plotting = Plotting(self.fig, self)

        # Connects all buttons so they do stuff
        self.connectors = GUI_connectors(self)

        #Define some flags, dictionaries and plotting options
        self.flagWheelZoom = True #Switch use of mousewheel for zooming
        #this next flag indicates if we zoom on time or amplitude axis
        self.flagWheelZoomAmplitude = False
        self.dictPhaseColors = {'P':'red', 'S':'blue', 'Psynth':'black',
                                'Ssynth':'black', 'Mag':'green'}
        self.dictPhaseLinestyles = {'P':'-', 'S':'-', 'Psynth':'--',
                                    'Ssynth':'--'}
        #Estimating the maximum/minimum in a sample-window around click
        self.magPickWindow = 10
        self.magMinMarker = 'x'
        self.magMaxMarker = 'x'
        self.magMarkerEdgeWidth = 1.8
        self.magMarkerSize = 20
        self.axvlinewidths = 1.2
        #dictionary for key-bindings
        self.dictKeybindings = {'setPick': 'alt', 'setPickError': ' ',
                'delPick': 'escape', 'setMagMin': 'alt', 'setMagMax': ' ',
                'switchPhase': 'control', 'delMagMinMax': 'escape',
                'switchWheelZoom': 'z', 'switchPan': 'p', 'prevStream': 'y',
                'nextStream': 'x', 'setWeight0': '0', 'setWeight1': '1',
                'setWeight2': '2', 'setWeight3': '3',
                #'setWeight4': '4', 'setWeight5': '5',
                'setPolUp': 'u', 'setPolPoorUp': '+', 'setPolDown': 'd',
                'setPolPoorDown': '-', 'setOnsetImpulsive': 'i',
                'setOnsetEmergent': 'e', 'switchWheelZoomAxis': 'shift'}
        # check for conflicting keybindings. 
        # we check twice, because keys for setting picks and magnitudes
        # are allowed to interfere...
        tmp_keys = self.dictKeybindings.copy()
        tmp_keys.pop('setMagMin')
        tmp_keys.pop('setMagMax')
        tmp_keys.pop('delMagMinMax')
        if len(set(tmp_keys.keys())) != len(set(tmp_keys.values())):
            msg = "Interfering keybindings. Please check variable " + \
                  "self.dictKeybindings"
            raise(msg)
        if len(set(tmp_keys.keys())) != len(set(tmp_keys.values())):
            msg = "Interfering keybindings. Please check variable " + \
                  "self.dictKeybindings"
            raise(msg)

        self.tmp_dir = tempfile.mkdtemp() + '/'

        # we have to control which binaries to use depending on architecture...
        architecture = platform.architecture()[0]
        if architecture == '32bit':
            self.threeDlocBinaryName = '3dloc_pitsa_32bit'
            self.hyp2000BinaryName = 'hyp2000_32bit'
            self.focmecScriptName = 'rfocmec_32bit'
            self.nllocBinaryName = 'NLLoc_32bit'
        elif architecture == '64bit':
            self.threeDlocBinaryName = '3dloc_pitsa_64bit'
            self.hyp2000BinaryName = 'hyp2000_64bit'
            self.focmecScriptName = 'rfocmec_64bit'
            self.nllocBinaryName = 'NLLoc_64bit'
        else:
            msg = "Warning: Could not determine architecture (32/64bit). " + \
                  "Using 32bit 3dloc binary."
            warnings.warn(msg)
            self.threeDlocBinaryName = '3dloc_pitsa_32bit'
            self.hyp2000BinaryName = 'hyp2000_32bit'
            self.focmecScriptName = 'rfocmec_32bit'
            self.nllocBinaryName = 'NLLoc_32bit'
        
        #######################################################################
        # 3dloc ###############################################################
        self.threeDlocPath = self.options.pluginpath + '/3dloc/'
        self.threeDlocPath_D3_VELOCITY = self.threeDlocPath + 'D3_VELOCITY'
        self.threeDlocPath_D3_VELOCITY_2 = self.threeDlocPath + 'D3_VELOCITY_2'
        self.threeDlocOutfile = self.tmp_dir + '3dloc-out'
        self.threeDlocInfile = self.tmp_dir + '3dloc-in'
        # copy 3dloc files to temp directory
        subprocess.call('cp -P %s/* %s &> /dev/null' % \
                (self.threeDlocPath, self.tmp_dir), shell=True)
        self.threeDlocPreCall = 'rm %s %s &> /dev/null' \
                % (self.threeDlocOutfile, self.threeDlocInfile)
        self.threeDlocCall = 'export D3_VELOCITY=%s/;' % \
                self.threeDlocPath_D3_VELOCITY + \
                'export D3_VELOCITY_2=%s/;' % \
                self.threeDlocPath_D3_VELOCITY_2 + \
                'cd %s; ./%s' % (self.tmp_dir, self.threeDlocBinaryName)
        #######################################################################
        # Hyp2000 #############################################################
        self.hyp2000Path = self.options.pluginpath + '/hyp_2000/'
        self.hyp2000Controlfilename = 'bay2000.inp'
        self.hyp2000Phasefile = self.tmp_dir + 'hyp2000.pha'
        self.hyp2000Stationsfile = self.tmp_dir + 'stations.dat'
        self.hyp2000Summary = self.tmp_dir + 'hypo.prt'
        # copy hypo2000 files to temp directory
        subprocess.call('cp -P %s/* %s &> /dev/null' % \
                (self.hyp2000Path, self.tmp_dir), shell=True)
        self.hyp2000PreCall = 'rm %s %s %s &> /dev/null' \
                % (self.hyp2000Phasefile, self.hyp2000Stationsfile,
                   self.hyp2000Summary)
        self.hyp2000Call = 'export HYP2000_DATA=%s;' % (self.tmp_dir) + \
                'cd $HYP2000_DATA; ./%s < %s &> /dev/null' % \
                (self.hyp2000BinaryName, self.hyp2000Controlfilename)
        #######################################################################
        # NLLoc ###############################################################
        self.nllocPath = self.options.pluginpath + '/nlloc/'
        self.nllocPhasefile = self.tmp_dir + 'nlloc.obs'
        self.nllocSummary = self.tmp_dir + 'nlloc.hyp'
        self.nllocScatterBin = self.tmp_dir + 'nlloc.scat'
        # copy nlloc files to temp directory
        subprocess.call('cp -P %s/* %s &> /dev/null' % \
                (self.nllocPath, self.tmp_dir), shell=True)
        self.nllocPreCall = 'rm %s/nlloc* &> /dev/null' % (self.tmp_dir)
        self.nllocCall = 'cd %s; ./%s %%s' % (self.tmp_dir,
                                              self.nllocBinaryName) + \
                '; mv nlloc.*.*.*.loc.hyp %s' % self.nllocSummary + \
                '; mv nlloc.*.*.*.loc.scat %s' % self.nllocScatterBin
        #######################################################################
        # focmec ##############################################################
        self.focmecPath = self.options.pluginpath + '/focmec/'
        self.focmecPhasefile = self.tmp_dir + 'focmec.dat'
        self.focmecStdout = self.tmp_dir + 'focmec.stdout'
        self.focmecSummary = self.tmp_dir + 'focmec.out'
        # copy focmec files to temp directory
        subprocess.call('cp -P %s/* %s &> /dev/null' % \
                (self.focmecPath, self.tmp_dir), shell=True)
        self.focmecCall = 'cd %s; ./%s' % (self.tmp_dir, self.focmecScriptName)
        self.dictOrigin = {}
        self.dictMagnitude = {}
        self.dictFocalMechanism = {} # currently selected focal mechanism
        self.focMechList = [] # list for all focal mechanisms from focmec
        # indicates which of the available focal mechanisms is selected
        self.focMechCurrent = None 
        # indicates how many focal mechanisms are available from focmec
        self.focMechCount = None
        self.dictEvent = {}
        self.dictEvent['xmlEventID'] = None
        self.spectrogramColormap = matplotlib.cm.jet
        # indicates which of the available events from seishub was loaded
        self.seishubEventCurrent = None 
        # indicates how many events are available from seishub
        self.seishubEventCount = None
        # save username of current user
        try:
            self.username = os.getlogin()
        except:
            self.username = os.environ['USER']
        # setup server information
        self.server = {}
        self.server['Name'] = self.options.servername # "teide"
        self.server['Port'] = self.options.port # 8080
        self.server['Server'] = self.server['Name'] + \
                                ":%i" % self.server['Port']
        self.server['BaseUrl'] = "http://" + self.server['Server']
        self.server['User'] = self.options.user # "obspyck"
        self.server['Password'] = self.options.password # "obspyck"
        
        # If keybindings option is set only show keybindings and exit
        if self.options.keybindings:
            for key, value in self.dictKeybindings.iteritems():
                print "%s: \"%s\"" % (key, value)
            return

        # Return, if no streams are given
        if not streams:
            return

        # Merge on every stream if this option is passed on command line:
        if self.options.merge:
            if self.options.merge.lower() == "safe":
                for st in self.streams:
                    st.merge(method=0)
            elif self.options.merge.lower() == "overwrite":
                for st in self.streams:
                    st.merge(method=1)
            else:
                err = "Unrecognized option for merging traces. Try " + \
                      "\"safe\" or \"overwrite\"."
                raise Exception(err)

        # Sort streams again, if there was a merge this could be necessary 
        for st in self.streams:
            st.sort()
            st.reverse()

        # Define some forbidden scenarios.
        # We assume there are:
        # - either one Z or three ZNE traces
        # - no two streams for any station (of same network)
        sta_list = set()
        # we need to go through streams/dicts backwards in order not to get
        # problems because of the pop() statement
        warn_msg = ""
        merge_msg = ""
        # XXX we need the list() because otherwise the iterator gets garbled if
        # XXX removing streams inside the for loop!!
        for st in list(self.streams):
            net_sta = "%s:%s" % (st[0].stats.network.strip(),
                                 st[0].stats.station.strip())
            # Here we make sure that a station/network combination is not
            # present with two streams. XXX For dynamically acquired data from
            # seishub this is already done before initialising the GUI and
            # thus redundant. Here it is only necessary if working with
            # traces from local file system (option -l)
            if net_sta in sta_list:
                msg = "Warning: Station/Network combination \"%s\" already " \
                      % net_sta + "in stream list. Discarding stream."
                print msg
                warn_msg += msg + "\n"
                self.streams.remove(st)
                continue
            if len(st) not in [1, 3]:
                msg = 'Warning: All streams must have either one Z trace ' + \
                      'or a set of three ZNE traces.'
                print msg
                warn_msg += msg + "\n"
                # remove all unknown channels ending with something other than
                # Z/N/E and try again...
                removed_channels = ""
                for tr in st:
                    if tr.stats.channel[-1] not in ["Z", "N", "E"]:
                        removed_channels += " " + tr.stats.channel
                        st.remove(tr)
                if len(st.traces) in [1, 3]:
                    msg = 'Warning: deleted some unknown channels in ' + \
                          'stream %s:%s' % (net_sta, removed_channels)
                    print msg
                    warn_msg += msg + "\n"
                    continue
                else:
                    msg = 'Stream %s discarded.\n' % net_sta + \
                          'Reason: Number of traces != (1 or 3)'
                    print msg
                    warn_msg += msg + "\n"
                    #for j, tr in enumerate(st.traces):
                    #    msg = 'Trace no. %i in Stream: %s\n%s' % \
                    #            (j + 1, tr.stats.channel, tr.stats)
                    msg = str(st)
                    print msg
                    warn_msg += msg + "\n"
                    self.streams.remove(st)
                    merge_msg = '\nIMPORTANT:\nYou can try the command line ' + \
                            'option merge (-m safe or -m overwrite) to ' + \
                            'avoid losing streams due gaps/overlaps.'
                    continue
            if len(st) == 1 and st[0].stats.channel[-1] != 'Z':
                msg = 'Warning: All streams must have either one Z trace ' + \
                      'or a set of three ZNE traces.'
                msg += 'Stream %s discarded. Reason: ' % net_sta + \
                       'Exactly one trace present but this is no Z trace'
                print msg
                warn_msg += msg + "\n"
                #for j, tr in enumerate(st.traces):
                #    msg = 'Trace no. %i in Stream: %s\n%s' % \
                #            (j + 1, tr.stats.channel, tr.stats)
                msg = str(st)
                print msg
                warn_msg += msg + "\n"
                self.streams.remove(st)
                continue
            if len(st) == 3 and (st[0].stats.channel[-1] != 'Z' or
                                 st[1].stats.channel[-1] != 'N' or
                                 st[2].stats.channel[-1] != 'E' or
                                 st[0].stats.station.strip() !=
                                 st[1].stats.station.strip() or
                                 st[0].stats.station.strip() !=
                                 st[2].stats.station.strip()):
                msg = 'Warning: All streams must have either one Z trace ' + \
                      'or a set of three ZNE traces.'
                msg += 'Stream %s discarded. Reason: ' % net_sta + \
                       'Exactly three traces present but they are not ZNE'
                print msg
                warn_msg += msg + "\n"
                #for j, tr in enumerate(st.traces):
                #    msg = 'Trace no. %i in Stream: %s\n%s' % \
                #            (j + 1, tr.stats.channel, tr.stats)
                msg = str(st)
                print msg
                warn_msg += msg + "\n"
                self.streams.remove(st)
                continue
            sta_list.add(net_sta)
        
        # if it was assigned at some point show the merge info message now
        if merge_msg:
            print merge_msg

        # exit if no streams are left after removing everthing with missing
        # information:
        if self.streams == []:
            print "Error: No streams."
            return

        #set up a list of dictionaries to store all picking data
        # set all station magnitude use-flags False
        self.dicts = []
        for ignored in self.streams:
            self.dicts.append({})
        #XXX not used: self.dictsMap = {} #XXX not used yet!
        self.eventMapColors = []
        # we need to go through streams/dicts backwards in order not to get
        # problems because of the pop() statement
        for i in range(len(self.streams))[::-1]:
            dict = self.dicts[i]
            st = self.streams[i]
            dict['MagUse'] = True
            sta = st[0].stats.station.strip()
            dict['Station'] = sta
            #XXX not used: self.dictsMap[sta] = dict
            self.eventMapColors.append((0.,  1.,  0.,  1.))
            net = st[0].stats.network.strip()
            print "=" * 70
            print sta
            if net == '':
                net = 'BW'
                print "Warning: Got no network information, setting to " + \
                      "default: BW"
            print "-" * 70
            date = st[0].stats.starttime.date
            print 'fetching station metadata from seishub...'
            try:
                lon, lat, ele = getCoord(self.client, net, sta)
                print lon, lat, ele
                dict['StaLon'] = lon
                dict['StaLat'] = lat
                dict['StaEle'] = ele / 1000. # all depths in km!
                dict['pazZ'] = self.client.station.getPAZ(net, sta, date,
                        channel_id=st[0].stats.channel)
                print dict['pazZ']
                if len(st) == 3:
                    dict['pazN'] = self.client.station.getPAZ(net, sta, date,
                            channel_id=st[1].stats.channel)
                    dict['pazE'] = self.client.station.getPAZ(net, sta, date,
                            channel_id=st[2].stats.channel)
                    print dict['pazN']
                    print dict['pazE']
            except:
                # XXX: Remove continue.
                continue
                print 'Error: could not fetch station metadata. Discarding stream.'
                self.streams.pop(i)
                self.dicts.pop(i)
                continue
            print 'done.'
        print "=" * 70
        
        # demean traces if not explicitly deactivated on command line
        if not self.options.nozeromean:
            for st in self.streams:
                for tr in st:
                    tr.data -= tr.data.mean()

        #Define a pointer to navigate through the streams
        self.stNum = len(self.streams)
        self.stPt = 0
    
        d = {}

        self.plotting.drawAxes()

        # Activate all mouse/key/Cursor-events
        # XXX: Reenable all connects!
        #self.canv.mpl_connect('key_press_event', self.keypress)
        #self.canv.mpl_connect('key_release_event', self.keyrelease)
        #self.canv.mpl_connect('scroll_event', self.scroll)
        #self.canv.mpl_connect('button_release_event', self.buttonrelease)
        #self.canv.mpl_connect('button_press_event', self.buttonpress)
        self.multicursor = MultiCursor(self.canv, self.plotting.axs, useblit=True,
                                       color='k', linewidth=1, ls='dotted')
        
        # fill the combobox list with the streams' station name.
        # XXX: Reenable!
        #self.interface.comboboxStreamName.clear()
        #for st in self.streams:
        #    self.interface.comboboxStreamName.addItem(st[0].stats['station'])

        # set the filter default values according to command line options
        # or command line default values
        self.interface.spinbuttonHighpass.setValue(self.options.highpass)
        self.interface.spinbuttonLowpass.setValue(self.options.lowpass)

        # XXX: Reenable!
        #self.updateStreamLabels()
        #self.multicursorReinit()



        #self.canv.show()

        # redirect stdout and stderr
        # we need to remember the original handles because we need to switch
        # back to them when going to debug mode
        self.stdout_backup = sys.stdout
        self.stderr_backup = sys.stderr
        sys.stdout = self.interface.stdoutPlainTextEdit
        sys.stderr = self.interface.stderrPlainTextEdit
        self.interface.stderrPlainTextEdit.write(warn_msg)

    def debug(self):
        sys.stdout = self.stdout_backup
        sys.stderr = self.stderr_backup
        try:
            import ipdb
            ipdb.set_trace()
        except ImportError:
            import pdb
            pdb.set_trace()
        self.stdout_backup = sys.stdout
        self.stderr_backup = sys.stderr
        sys.stdout = self.textviewStdOutImproved
        sys.stderr = self.textviewStdErrImproved

    def setFocusToMatplotlib(self):
        """
        Sets the focus to the matplotlib widget.
        """
        self.canv.grab_focus()

    def cleanQuit(self):
        """
        Deletes all temporary files and closes the application.
        """
        try:
            shutil.rmtree(self.tmp_dir)
        except:
            pass
        # Close all QT windows.
        self.gui.qApp.closeAllWindows()
    
    #lookup multicursor source: http://matplotlib.sourcearchive.com/documentation/0.98.1/widgets_8py-source.html
    def multicursorReinit(self):
        self.canv.mpl_disconnect(self.multicursor.id1)
        self.canv.mpl_disconnect(self.multicursor.id2)
        self.multicursor.__init__(self.canv, self.axs, useblit=True,
                                  color='black', linewidth=1, ls='dotted')
        self.updateMulticursorColor()
        self.canv.widgetlock.release(self.toolbar)

    def updateMulticursorColor(self):
        phase_name = self.comboboxPhaseType.get_active_text()
        color = self.dictPhaseColors[phase_name]
        for l in self.multicursor.lines:
            l.set_color(color)

    def updateButtonPhaseTypeColor(self):
        phase_name = self.comboboxPhaseType.get_active_text()
        style = self.buttonPhaseType.get_style().copy()
        color = gtk.gdk.color_parse(self.dictPhaseColors[phase_name])
        style.bg[gtk.STATE_INSENSITIVE] = color
        self.buttonPhaseType.set_style(style)

    def updateStreamNumberLabel(self):
        self.interface.labelStreamNumber.setText("<tt>%02i/%02i</tt>" % \
                (self.stPt + 1, self.stNum))
    
    def updateStreamNameCombobox(self):
        self.interface.comboboxStreamName.setCurrentIndex(self.stPt)

    def updateStreamLabels(self):
        self.updateStreamNumberLabel()
        self.updateStreamNameCombobox()

    def load3dlocSyntheticPhases(self):
        try:
            fhandle = open(self.threeDlocOutfile, 'r')
            phaseList = fhandle.readlines()
            fhandle.close()
        except:
            return
        self.delPsynth()
        self.delSsynth()
        self.delPsynthLine()
        self.delPsynthLabel()
        self.delSsynthLine()
        self.delSsynthLabel()
        for phase in phaseList[1:]:
            # example for a synthetic pick line from 3dloc:
            # RJOB P 2009 12 27 10 52 59.425 -0.004950 298.199524 136.000275
            # station phase YYYY MM DD hh mm ss.sss (picked time!) residual
            # (add this to get synthetic time) azimuth? incidenceangle?
            # XXX maybe we should avoid reading this absolute time and rather
            # use our dict['P'] or dict['S'] time and simple subtract the
            # residual to simplify things!?
            phase = phase.split()
            phStat = phase[0]
            phType = phase[1]
            phUTCTime = UTCDateTime(int(phase[2]), int(phase[3]),
                                    int(phase[4]), int(phase[5]),
                                    int(phase[6]), float(phase[7]))
            phResid = float(phase[8])
            # residual is defined as P-Psynth by NLLOC and 3dloc!
            phUTCTime = phUTCTime - phResid
            for i, dict in enumerate(self.dicts):
                st = self.streams[i]
                # check for matching station names
                if not phStat == st[0].stats.station.strip():
                    continue
                else:
                    # check if synthetic pick is within time range of stream
                    if (phUTCTime > st[0].stats.endtime or \
                        phUTCTime < st[0].stats.starttime):
                        err = "Warning: Synthetic pick outside timespan."
                        self.textviewStdErrImproved.write(err)
                        continue
                    else:
                        # phSeconds is the time in seconds after the stream-
                        # starttime at which the time of the synthetic phase
                        # is located
                        phSeconds = phUTCTime - st[0].stats.starttime
                        if phType == 'P':
                            dict['Psynth'] = phSeconds
                            dict['Pres'] = phResid
                        elif phType == 'S':
                            dict['Ssynth'] = phSeconds
                            dict['Sres'] = phResid
        self.drawPsynthLine()
        self.drawPsynthLabel()
        self.drawSsynthLine()
        self.drawSsynthLabel()
        self.redraw()

    def do3dLoc(self):
        self.setXMLEventID()
        #subprocess.call(self.threeDlocPreCall, shell=True)
        sub = subprocess.Popen(self.threeDlocPreCall, shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        msg = "".join(sub.stdout.readlines())
        err = "".join(sub.stderr.readlines())
        self.textviewStdOutImproved.write(msg)
        self.textviewStdErrImproved.write(err)
        f = open(self.threeDlocInfile, 'w')
        network = "BW"
        fmt = "%04s  %s        %s %5.3f -999.0 0.000 -999. 0.000 T__DR_ %9.6f %9.6f %8.6f\n"
        self.coords = []
        for i, dict in enumerate(self.dicts):
            st = self.streams[i]
            lon = dict['StaLon']
            lat = dict['StaLat']
            ele = dict['StaEle']
            self.coords.append([lon, lat])
            # if the error picks are not set, we use a default of three samples
            default_error = 3 / st[0].stats.sampling_rate
            if 'P' in dict:
                t = st[0].stats.starttime
                t += dict['P']
                date = t.strftime("%Y %m %d %H %M %S")
                date += ".%03d" % (t.microsecond / 1e3 + 0.5)
                if 'PErr1' in dict:
                    error_1 = dict['PErr1']
                else:
                    err = "Warning: Left error pick for P missing. " + \
                          "Using a default of 3 samples left of P."
                    self.textviewStdErrImproved.write(err)
                    error_1 = dict['P'] - default_error
                if 'PErr2' in dict:
                    error_2 = dict['PErr2']
                else:
                    err = "Warning: Right error pick for P missing. " + \
                          "Using a default of 3 samples right of P."
                    self.textviewStdErrImproved.write(err)
                    error_2 = dict['P'] + default_error
                delta = error_2 - error_1
                f.write(fmt % (dict['Station'], 'P', date, delta, lon, lat,
                               ele))
            if 'S' in dict:
                t = st[0].stats.starttime
                t += dict['S']
                date = t.strftime("%Y %m %d %H %M %S")
                date += ".%03d" % (t.microsecond / 1e3 + 0.5)
                if 'SErr1' in dict:
                    error_1 = dict['SErr1']
                else:
                    err = "Warning: Left error pick for S missing. " + \
                          "Using a default of 3 samples left of S."
                    self.textviewStdErrImproved.write(err)
                    error_1 = dict['S'] - default_error
                if 'SErr2' in dict:
                    error_2 = dict['SErr2']
                else:
                    err = "Warning: Right error pick for S missing. " + \
                          "Using a default of 3 samples right of S."
                    self.textviewStdErrImproved.write(err)
                    error_2 = dict['S'] + default_error
                delta = error_2 - error_1
                f.write(fmt % (dict['Station'], 'S', date, delta, lon, lat,
                               ele))
        f.close()
        msg = 'Phases for 3Dloc:'
        self.textviewStdOutImproved.write(msg)
        self.catFile(self.threeDlocInfile)
        #subprocess.call(self.threeDlocCall, shell=True)
        sub = subprocess.Popen(self.threeDlocCall, shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        msg = "".join(sub.stdout.readlines())
        err = "".join(sub.stderr.readlines())
        self.textviewStdOutImproved.write(msg)
        self.textviewStdErrImproved.write(err)
        msg = '--> 3dloc finished'
        self.textviewStdOutImproved.write(msg)
        self.catFile(self.threeDlocOutfile)

    def doFocmec(self):
        f = open(self.focmecPhasefile, 'w')
        f.write("\n") #first line is ignored!
        #Fortran style! 1: Station 2: Azimuth 3: Incident 4: Polarity
        #fmt = "ONTN  349.00   96.00C"
        fmt = "%4s  %6.2f  %6.2f%1s\n"
        count = 0
        for dict in self.dicts:
            if 'PAzim' not in dict or 'PInci' not in dict or 'PPol' not in dict:
                continue
            sta = dict['Station'][:4] #focmec has only 4 chars
            azim = dict['PAzim']
            inci = dict['PInci']
            if dict['PPol'] == 'up':
                pol = 'U'
            elif dict['PPol'] == 'poorup':
                pol = '+'
            elif dict['PPol'] == 'down':
                pol = 'D'
            elif dict['PPol'] == 'poordown':
                pol = '-'
            else:
                continue
            count += 1
            f.write(fmt % (sta, azim, inci, pol))
        f.close()
        msg = 'Phases for focmec: %i' % count
        self.textviewStdOutImproved.write(msg)
        self.catFile(self.focmecPhasefile)
        sub = subprocess.Popen(self.focmecCall, shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        msg = "".join(sub.stdout.readlines())
        err = "".join(sub.stderr.readlines())
        self.textviewStdOutImproved.write(msg)
        self.textviewStdErrImproved.write(err)
        if sub.returncode == 1:
            err = "Error: focmec did not find a suitable solution!"
            self.textviewStdErrImproved.write(err)
            return
        msg = '--> focmec finished'
        self.textviewStdOutImproved.write(msg)
        lines = open(self.focmecSummary, "r").readlines()
        msg = '%i suitable solutions found:' % len(lines)
        self.textviewStdOutImproved.write(msg)
        self.focMechList = []
        for line in lines:
            line = line.split()
            tempdict = {}
            tempdict['Program'] = "focmec"
            tempdict['Dip'] = float(line[0])
            tempdict['Strike'] = float(line[1])
            tempdict['Rake'] = float(line[2])
            tempdict['Errors'] = int(float(line[3])) # not used in xml
            tempdict['Station Polarity Count'] = count
            tempdict['Possible Solution Count'] = len(lines)
            msg = "Dip: %6.2f  Strike: %6.2f  Rake: %6.2f  Errors: %i/%i" % \
                    (tempdict['Dip'], tempdict['Strike'], tempdict['Rake'],
                     tempdict['Errors'], tempdict['Station Polarity Count'])
            self.textviewStdOutImproved.write(msg)
            self.focMechList.append(tempdict)
        self.focMechCount = len(self.focMechList)
        self.focMechCurrent = 0
        msg = "selecting Focal Mechanism No.  1 of %2i:" % self.focMechCount
        self.textviewStdOutImproved.write(msg)
        self.dictFocalMechanism = self.focMechList[0]
        dF = self.dictFocalMechanism
        msg = "Dip: %6.2f  Strike: %6.2f  Rake: %6.2f  Errors: %i/%i" % \
                (dF['Dip'], dF['Strike'], dF['Rake'], dF['Errors'],
                 dF['Station Polarity Count'])
        self.textviewStdOutImproved.write(msg)

    def nextFocMec(self):
        if self.focMechCount is None:
            return
        self.focMechCurrent = (self.focMechCurrent + 1) % self.focMechCount
        self.dictFocalMechanism = self.focMechList[self.focMechCurrent]
        msg = "selecting Focal Mechanism No. %2i of %2i:" % \
                (self.focMechCurrent + 1, self.focMechCount)
        self.textviewStdOutImproved.write(msg)
        msg = "Dip: %6.2f  Strike: %6.2f  Rake: %6.2f  Errors: %i%i" % \
                (self.dictFocalMechanism['Dip'],
                 self.dictFocalMechanism['Strike'],
                 self.dictFocalMechanism['Rake'],
                 self.dictFocalMechanism['Errors'],
                 self.dictFocalMechanism['Station Polarity Count'])
        self.textviewStdOutImproved.write(msg)
    
    def drawFocMec(self):
        if self.dictFocalMechanism == {}:
            err = "Error: No focal mechanism data!"
            self.textviewStdErrImproved.write(err)
            return
        # make up the figure:
        fig = self.fig
        self.axsFocMec = []
        axs = self.axsFocMec
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        
        # plot the selected solution
        dF = self.dictFocalMechanism
        axs.append(Beachball([dF['Strike'], dF['Dip'], dF['Rake']], fig=fig))
        # plot the alternative solutions
        if self.focMechList != []:
            for dict in self.focMechList:
                axs.append(Beachball([dict['Strike'], dict['Dip'],
                          dict['Rake']],
                          nofill=True, fig=fig, edgecolor='k',
                          linewidth=1., alpha=0.3))
        text = "Focal Mechanism (%i of %i)" % \
               (self.focMechCurrent + 1, self.focMechCount)
        text += "\nDip: %6.2f  Strike: %6.2f  Rake: %6.2f" % \
                (dF['Dip'], dF['Strike'], dF['Rake'])
        if 'Errors' in dF:
            text += "\nErrors: %i/%i" % (dF['Errors'],
                                         dF['Station Polarity Count'])
        else:
            text += "\nUsed Polarities: %i" % dF['Station Polarity Count']
        #fig.canvas.set_window_title("Focal Mechanism (%i of %i)" % \
        #        (self.focMechCurrent + 1, self.focMechCount))
        fig.subplots_adjust(top=0.88) # make room for suptitle
        # values 0.02 and 0.96 fit best over the outer edges of beachball
        #ax = fig.add_axes([0.00, 0.02, 1.00, 0.96], polar=True)
        self.axFocMecStations = fig.add_axes([0.00,0.02,1.00,0.84], polar=True)
        ax = self.axFocMecStations
        ax.set_title(text)
        ax.set_axis_off()
        for dict in self.dicts:
            if 'PAzim' in dict and 'PInci' in dict and 'PPol' in dict:
                if dict['PPol'] == "up":
                    color = "black"
                elif dict['PPol'] == "poorup":
                    color = "darkgrey"
                elif dict['PPol'] == "poordown":
                    color = "lightgrey"
                elif dict['PPol'] == "down":
                    color = "white"
                else:
                    continue
                # southern hemisphere projection
                if dict['PInci'] > 90:
                    inci = 180. - dict['PInci']
                    azim = -180. + dict['PAzim']
                else:
                    inci = dict['PInci']
                    azim = dict['PAzim']
                #we have to hack the azimuth because of the polar plot
                #axes orientation
                plotazim = (np.pi / 2.) - ((azim / 180.) * np.pi)
                ax.scatter([plotazim], [inci], facecolor=color)
                ax.text(plotazim, inci, " " + dict['Station'], va="top")
        #this fits the 90 degree incident value to the beachball edge best
        ax.set_ylim([0., 91])
        self.canv.draw()

    def delFocMec(self):
        if hasattr(self, "axFocMecStations"):
            self.fig.delaxes(self.axFocMecStations)
            del self.axFocMecStations
        if hasattr(self, "axsFocMec"):
            for ax in self.axsFocMec:
                if ax in self.fig.axes: 
                    self.fig.delaxes(ax)
                del ax

    def doHyp2000(self):
        """
        Writes input files for hyp2000 and starts the hyp2000 program via a
        system call.
        """
        self.setXMLEventID()
        sub = subprocess.Popen(self.hyp2000PreCall, shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        msg = "".join(sub.stdout.readlines())
        err = "".join(sub.stderr.readlines())
        self.textviewStdOutImproved.write(msg)
        self.textviewStdErrImproved.write(err)

        f = open(self.hyp2000Phasefile, 'w')
        phases_hypo71 = self.dicts2hypo71Phases()
        f.write(phases_hypo71)
        f.close()

        f2 = open(self.hyp2000Stationsfile, 'w')
        stations_hypo71 = self.dicts2hypo71Stations()
        f2.write(stations_hypo71)
        f2.close()

        msg = 'Phases for Hypo2000:'
        self.textviewStdOutImproved.write(msg)
        self.catFile(self.hyp2000Phasefile)
        msg = 'Stations for Hypo2000:'
        self.textviewStdOutImproved.write(msg)
        self.catFile(self.hyp2000Stationsfile)

        sub = subprocess.Popen(self.hyp2000Call, shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        msg = "".join(sub.stdout.readlines())
        err = "".join(sub.stderr.readlines())
        self.textviewStdOutImproved.write(msg)
        self.textviewStdErrImproved.write(err)
        msg = '--> hyp2000 finished'
        self.textviewStdOutImproved.write(msg)
        self.catFile(self.hyp2000Summary)

    def doNLLoc(self):
        """
        Writes input files for NLLoc and starts the NonLinLoc program via a
        system call.
        """
        # determine which model should be used in location
        controlfilename = "locate_%s.nlloc" % \
                          self.comboboxNLLocModel.get_active_text()
        nllocCall = self.nllocCall % controlfilename

        self.setXMLEventID()
        sub = subprocess.Popen(self.nllocPreCall, shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        msg = "".join(sub.stdout.readlines())
        err = "".join(sub.stderr.readlines())
        self.textviewStdOutImproved.write(msg)
        self.textviewStdErrImproved.write(err)

        f = open(self.nllocPhasefile, 'w')
        phases_hypo71 = self.dicts2hypo71Phases()
        f.write(phases_hypo71)
        f.close()

        msg = 'Phases for NLLoc:'
        self.textviewStdOutImproved.write(msg)
        self.catFile(self.nllocPhasefile)

        sub = subprocess.Popen(nllocCall, shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        msg = "".join(sub.stdout.readlines())
        err = "".join(sub.stderr.readlines())
        self.textviewStdOutImproved.write(msg)
        self.textviewStdErrImproved.write(err)
        msg = '--> NLLoc finished'
        self.textviewStdOutImproved.write(msg)
        self.catFile(self.nllocSummary)

    def catFile(self, file):
        lines = open(file).readlines()
        msg = ""
        for line in lines:
            msg += line
        self.textviewStdOutImproved.write(msg)

    def loadNLLocOutput(self):
        lines = open(self.nllocSummary).readlines()
        if not lines:
            err = "Error: NLLoc output file (%s) does not exist!" % \
                    self.nllocSummary
            self.textviewStdErrImproved.write(err)
            return
        # goto maximum likelihood origin location info line
        try:
            line = lines.pop(0)
            while not line.startswith("HYPOCENTER"):
                line = lines.pop(0)
        except:
            err = "Error: No correct location info found in NLLoc " + \
                  "outputfile (%s)!" % self.nllocSummary
            self.textviewStdErrImproved.write(err)
            return
        
        line = line.split()
        x = float(line[2])
        y = float(line[4])
        depth = - float(line[6]) # depth: negative down!
        
        lon, lat = gk2lonlat(x, y)
        
        # goto origin time info line
        try:
            line = lines.pop(0)
            while not line.startswith("GEOGRAPHIC  OT"):
                line = lines.pop(0)
        except:
            err = "Error: No correct location info found in NLLoc " + \
                  "outputfile (%s)!" % self.nllocSummary
            self.textviewStdErrImproved.write(err)
            return
        
        line = line.split()
        year = int(line[2])
        month = int(line[3])
        day = int(line[4])
        hour = int(line[5])
        minute = int(line[6])
        seconds = float(line[7])
        time = UTCDateTime(year, month, day, hour, minute, seconds)

        # goto location quality info line
        try:
            line = lines.pop(0)
            while not line.startswith("QUALITY"):
                line = lines.pop(0)
        except:
            err = "Error: No correct location info found in NLLoc " + \
                  "outputfile (%s)!" % self.nllocSummary
            self.textviewStdErrImproved.write(err)
            return
        
        line = line.split()
        rms = float(line[8])
        gap = int(line[12])

        # goto location quality info line
        try:
            line = lines.pop(0)
            while not line.startswith("STATISTICS"):
                line = lines.pop(0)
        except:
            err = "Error: No correct location info found in NLLoc " + \
                  "outputfile (%s)!" % self.nllocSummary
            self.textviewStdErrImproved.write(err)
            return
        
        line = line.split()
        # read in the error ellipsoid representation of the location error.
        # this is given as azimuth/dip/length of axis 1 and 2 and as length
        # of axis 3.
        azim1 = float(line[20])
        dip1 = float(line[22])
        len1 = float(line[24])
        azim2 = float(line[26])
        dip2 = float(line[28])
        len2 = float(line[30])
        len3 = float(line[32])

        errX, errY, errZ = errorEllipsoid2CartesianErrors(azim1, dip1, len1,
                                                          azim2, dip2, len2,
                                                          len3)
        
        # XXX
        # NLLOC uses error ellipsoid for 68% confidence interval relating to
        # one standard deviation in the normal distribution.
        # We multiply all errors by 2 to approximately get the 95% confidence
        # level (two standard deviations)...
        errX *= 2
        errY *= 2
        errZ *= 2

        # determine which model was used:
        controlfile = self.tmp_dir + "/last.in"
        lines2 = open(controlfile).readlines()
        line2 = lines2.pop()
        while not line2.startswith("LOCFILES"):
            line2 = lines2.pop()
        line2 = line2.split()
        model = line2[3]
        model = model.split("/")[-1]

        # assign origin info
        dO = self.dictOrigin
        dO['Longitude'] = lon
        dO['Latitude'] = lat
        dO['Depth'] = depth
        dO['Longitude Error'] = errX
        dO['Latitude Error'] = errY
        dO['Depth Error'] = errZ
        dO['Standarderror'] = rms #XXX stimmt diese Zuordnung!!!?!
        dO['Azimuthal Gap'] = gap
        dO['Depth Type'] = "from location program"
        dO['Earth Model'] = model
        dO['Time'] = time
        
        # goto synthetic phases info lines
        try:
            line = lines.pop(0)
            while not line.startswith("PHASE ID"):
                line = lines.pop(0)
        except:
            err = "Error: No correct synthetic phase info found in NLLoc " + \
                  "outputfile (%s)!" % self.nllocSummary
            self.textviewStdErrImproved.write(err)
            return

        # remove all non phase-info-lines from bottom of list
        try:
            badline = lines.pop()
            while not badline.startswith("END_PHASE"):
                badline = lines.pop()
        except:
            err = "Error: Could not remove unwanted lines at bottom of " + \
                  "NLLoc outputfile (%s)!" % self.nllocSummary
            self.textviewStdErrImproved.write(err)
            return
        
        dO['used P Count'] = 0
        dO['used S Count'] = 0

        # go through all phase info lines
        for line in lines:
            line = line.split()
            # check which type of phase
            if line[4] == "P":
                type = "P"
            elif line[4] == "S":
                type = "S"
            else:
                continue
            # get values from line
            station = line[0]
            azimuth = float(line[23])
            incident = float(line[24])
            # if we do the location on traveltime-grids without angle-grids we
            # do not get ray azimuth/incidence. but we can at least use the
            # station to hypocenter azimuth which is very close (~2 deg) to the
            # ray azimuth
            if azimuth == 0.0 and incident == 0.0:
                azimuth = float(line[22])
                incident = np.nan
            if line[3] == "I":
                onset = "impulsive"
            elif line[3] == "E":
                onset = "emergent"
            else:
                onset = None
            if line[5] == "U":
                polarity = "up"
            elif line[5] == "D":
                polarity = "down"
            else:
                polarity = None
            res = float(line[16])
            weight = float(line[17])

            # search for streamnumber corresponding to pick
            streamnum = None
            for i, dict in enumerate(self.dicts):
                if station.strip() != dict['Station']:
                    continue
                else:
                    streamnum = i
                    break
            if streamnum is None:
                err = "Warning: Did not find matching stream for pick " + \
                      "data with station id: \"%s\"" % station.strip()
                self.textviewStdErrImproved.write(err)
                continue
            
            # assign synthetic phase info
            dict = self.dicts[streamnum]
            if type == "P":
                dO['used P Count'] += 1
                #dict['Psynth'] = res + dict['P']
                # residual is defined as P-Psynth by NLLOC and 3dloc!
                dict['Psynth'] = dict['P'] - res
                dict['Pres'] = res
                dict['PAzim'] = azimuth
                dict['PInci'] = incident
                if onset:
                    dict['POnset'] = onset
                if polarity:
                    dict['PPol'] = polarity
                # we use weights 0,1,2,3 but NLLoc outputs floats...
                dict['PsynthWeight'] = weight
            elif type == "S":
                dO['used S Count'] += 1
                # residual is defined as S-Ssynth by NLLOC and 3dloc!
                dict['Ssynth'] = dict['S'] - res
                dict['Sres'] = res
                dict['SAzim'] = azimuth
                dict['SInci'] = incident
                if onset:
                    dict['SOnset'] = onset
                if polarity:
                    dict['SPol'] = polarity
                # we use weights 0,1,2,3 but NLLoc outputs floats...
                dict['SsynthWeight'] = weight
        dO['used Station Count'] = len(self.dicts)
        for dict in self.dicts:
            if not ('Psynth' in dict or 'Ssynth' in dict):
                dO['used Station Count'] -= 1

    def loadHyp2000Data(self):
        #self.load3dlocSyntheticPhases()
        lines = open(self.hyp2000Summary).readlines()
        if lines == []:
            err = "Error: Hypo2000 output file (%s) does not exist!" % \
                    self.hyp2000Summary
            self.textviewStdErrImproved.write(err)
            return
        # goto origin info line
        while True:
            try:
                line = lines.pop(0)
            except:
                break
            if line.startswith(" YEAR MO DA  --ORIGIN--"):
                break
        try:
            line = lines.pop(0)
        except:
            err = "Error: No location info found in Hypo2000 outputfile " + \
                  "(%s)!" % self.hyp2000Summary
            self.textviewStdErrImproved.write(err)
            return

        year = int(line[1:5])
        month = int(line[6:8])
        day = int(line[9:11])
        hour = int(line[13:15])
        minute = int(line[15:17])
        seconds = float(line[18:23])
        time = UTCDateTime(year, month, day, hour, minute, seconds)
        lat_deg = int(line[25:27])
        lat_min = float(line[28:33])
        lat = lat_deg + (lat_min / 60.)
        if line[27] == "S":
            lat = -lat
        lon_deg = int(line[35:38])
        lon_min = float(line[39:44])
        lon = lon_deg + (lon_min / 60.)
        if line[38] == " ":
            lon = -lon
        depth = -float(line[46:51]) # depth: negative down!
        rms = float(line[52:57])
        errXY = float(line[58:63])
        errZ = float(line[64:69])

        # goto next origin info line
        while True:
            try:
                line = lines.pop(0)
            except:
                break
            if line.startswith(" NSTA NPHS  DMIN MODEL"):
                break
        line = lines.pop(0)

        #model = line[17:22].strip()
        gap = int(line[23:26])

        line = lines.pop(0)
        model = line[49:].strip()

        # assign origin info
        dO = self.dictOrigin
        dO['Longitude'] = lon
        dO['Latitude'] = lat
        dO['Depth'] = depth
        dO['Longitude Error'] = errXY
        dO['Latitude Error'] = errXY
        dO['Depth Error'] = errZ
        dO['Standarderror'] = rms #XXX stimmt diese Zuordnung!!!?!
        dO['Azimuthal Gap'] = gap
        dO['Depth Type'] = "from location program"
        dO['Earth Model'] = model
        dO['Time'] = time
        
        # goto station and phases info lines
        while True:
            try:
                line = lines.pop(0)
            except:
                break
            if line.startswith(" STA NET COM L CR DIST AZM"):
                break
        
        dO['used P Count'] = 0
        dO['used S Count'] = 0
        #XXX caution: we sometimes access the prior element!
        for i in range(len(lines)):
            # check which type of phase
            if lines[i][32] == "P":
                type = "P"
            elif lines[i][32] == "S":
                type = "S"
            else:
                continue
            # get values from line
            station = lines[i][0:6].strip()
            if station == "":
                station = lines[i-1][0:6].strip()
                azimuth = int(lines[i-1][23:26])
                #XXX check, if incident is correct!!
                incident = int(lines[i-1][27:30])
            else:
                azimuth = int(lines[i][23:26])
                #XXX check, if incident is correct!!
                incident = int(lines[i][27:30])
            if lines[i][31] == "I":
                onset = "impulsive"
            elif lines[i][31] == "E":
                onset = "emergent"
            else:
                onset = None
            if lines[i][33] == "U":
                polarity = "up"
            elif lines[i][33] == "D":
                polarity = "down"
            else:
                polarity = None
            res = float(lines[i][61:66])
            weight = float(lines[i][68:72])

            # search for streamnumber corresponding to pick
            streamnum = None
            for i, dict in enumerate(self.dicts):
                if station.strip() != dict['Station']:
                    continue
                else:
                    streamnum = i
                    break
            if streamnum is None:
                err = "Warning: Did not find matching stream for pick " + \
                      "data with station id: \"%s\"" % station.strip()
                self.textviewStdErrImproved.write(err)
                continue
            
            # assign synthetic phase info
            dict = self.dicts[streamnum]
            if type == "P":
                dO['used P Count'] += 1
                # residual is defined as P-Psynth by NLLOC and 3dloc!
                # XXX does this also hold for hyp2000???
                dict['Psynth'] = dict['P'] - res
                dict['Pres'] = res
                dict['PAzim'] = azimuth
                dict['PInci'] = incident
                if onset:
                    dict['POnset'] = onset
                if polarity:
                    dict['PPol'] = polarity
                # we use weights 0,1,2,3 but hypo2000 outputs floats...
                dict['PsynthWeight'] = weight
            elif type == "S":
                dO['used S Count'] += 1
                # residual is defined as S-Ssynth by NLLOC and 3dloc!
                # XXX does this also hold for hyp2000???
                dict['Ssynth'] = dict['S'] - res
                dict['Sres'] = res
                dict['SAzim'] = azimuth
                dict['SInci'] = incident
                if onset:
                    dict['SOnset'] = onset
                if polarity:
                    dict['SPol'] = polarity
                # we use weights 0,1,2,3 but hypo2000 outputs floats...
                dict['SsynthWeight'] = weight
        dO['used Station Count'] = len(self.dicts)
        for dict in self.dicts:
            if not ('Psynth' in dict or 'Ssynth' in dict):
                dO['used Station Count'] -= 1

    def load3dlocData(self):
        #self.load3dlocSyntheticPhases()
        event = open(self.threeDlocOutfile).readline().split()
        dO = self.dictOrigin
        dO['Longitude'] = float(event[8])
        dO['Latitude'] = float(event[9])
        dO['Depth'] = float(event[10])
        dO['Longitude Error'] = float(event[11])
        dO['Latitude Error'] = float(event[12])
        dO['Depth Error'] = float(event[13])
        dO['Standarderror'] = float(event[14])
        dO['Azimuthal Gap'] = float(event[15])
        dO['Depth Type'] = "from location program"
        dO['Earth Model'] = "STAUFEN"
        dO['Time'] = UTCDateTime(int(event[2]), int(event[3]), int(event[4]),
                                 int(event[5]), int(event[6]), float(event[7]))
        dO['used P Count'] = 0
        dO['used S Count'] = 0
        lines = open(self.threeDlocInfile).readlines()
        for line in lines:
            pick = line.split()
            for st in self.streams:
                if pick[0].strip() == st[0].stats.station.strip():
                    if pick[1] == 'P':
                        dO['used P Count'] += 1
                    elif pick[1] == 'S':
                        dO['used S Count'] += 1
                    break
        lines = open(self.threeDlocOutfile).readlines()
        for line in lines[1:]:
            pick = line.split()
            for i, st in enumerate(self.streams):
                if pick[0].strip() == st[0].stats.station.strip():
                    dict = self.dicts[i]
                    if pick[1] == 'P':
                        dict['PAzim'] = float(pick[9])
                        dict['PInci'] = float(pick[10])
                    elif pick[1] == 'S':
                        dict['SAzim'] = float(pick[9])
                        dict['SInci'] = float(pick[10])
                    break
        dO['used Station Count'] = len(self.dicts)
        for dict in self.dicts:
            if not ('Psynth' in dict or 'Ssynth' in dict):
                dO['used Station Count'] -= 1
    
    def updateNetworkMag(self):
        msg = "updating network magnitude..."
        self.textviewStdOutImproved.write(msg)
        dM = self.dictMagnitude
        dM['Station Count'] = 0
        dM['Magnitude'] = 0
        staMags = []
        for dict in self.dicts:
            if dict['MagUse'] and 'Mag' in dict:
                msg = "%s: %.1f" % (dict['Station'], dict['Mag'])
                self.textviewStdOutImproved.write(msg)
                dM['Station Count'] += 1
                dM['Magnitude'] += dict['Mag']
                staMags.append(dict['Mag'])
        if dM['Station Count'] == 0:
            dM['Magnitude'] = np.nan
            dM['Uncertainty'] = np.nan
        else:
            dM['Magnitude'] /= dM['Station Count']
            dM['Uncertainty'] = np.var(staMags)
        msg = "new network magnitude: %.2f (Variance: %.2f)" % \
                (dM['Magnitude'], dM['Uncertainty'])
        self.textviewStdOutImproved.write(msg)
        self.netMagLabel = '\n\n\n\n\n %.2f (Var: %.2f)' % (dM['Magnitude'],
                                                           dM['Uncertainty'])
        try:
            self.netMagText.set_text(self.netMagLabel)
        except:
            pass
    
    def calculateEpiHypoDists(self):
        if not 'Longitude' in self.dictOrigin or \
           not 'Latitude' in self.dictOrigin:
            err = "Error: No coordinates for origin!"
            self.textviewStdErrImproved.write(err)
        epidists = []
        for dict in self.dicts:
            x, y = utlGeoKm(self.dictOrigin['Longitude'],
                            self.dictOrigin['Latitude'],
                            dict['StaLon'], dict['StaLat'])
            z = abs(dict['StaEle'] - self.dictOrigin['Depth'])
            dict['distX'] = x
            dict['distY'] = y
            dict['distZ'] = z
            dict['distEpi'] = np.sqrt(x**2 + y**2)
            # Median and Max/Min of epicentral distances should only be used
            # for stations with a pick that goes into the location.
            # The epicentral distance of all other stations may be needed for
            # magnitude estimation nonetheless.
            if 'Psynth' in dict or 'Ssynth' in dict:
                epidists.append(dict['distEpi'])
            dict['distHypo'] = np.sqrt(x**2 + y**2 + z**2)
        self.dictOrigin['Maximum Distance'] = max(epidists)
        self.dictOrigin['Minimum Distance'] = min(epidists)
        self.dictOrigin['Median Distance'] = np.median(epidists)

    def calculateStationMagnitudes(self):
        for i, dict in enumerate(self.dicts):
            st = self.streams[i]
            if 'MagMin1' in dict and 'MagMin2' in dict and \
               'MagMax1' in dict and 'MagMax2' in dict:
                
                amp = dict['MagMax1'] - dict['MagMin1']
                timedelta = abs(dict['MagMax1T'] - dict['MagMin1T'])
                mag = estimateMagnitude(dict['pazN'], amp, timedelta,
                                        dict['distHypo'])
                amp = dict['MagMax2'] - dict['MagMin2']
                timedelta = abs(dict['MagMax2T'] - dict['MagMin2T'])
                mag += estimateMagnitude(dict['pazE'], amp, timedelta,
                                         dict['distHypo'])
                mag /= 2.
                dict['Mag'] = mag
                dict['MagChannel'] = '%s,%s' % (st[1].stats.channel,
                                                st[2].stats.channel)
                msg = 'calculated new magnitude for %s: %0.2f (channels: %s)' \
                      % (dict['Station'], dict['Mag'], dict['MagChannel'])
                self.textviewStdOutImproved.write(msg)
            
            elif 'MagMin1' in dict and 'MagMax1' in dict:
                amp = dict['MagMax1'] - dict['MagMin1']
                timedelta = abs(dict['MagMax1T'] - dict['MagMin1T'])
                mag = estimateMagnitude(dict['pazN'], amp, timedelta,
                                        dict['distHypo'])
                dict['Mag'] = mag
                dict['MagChannel'] = '%s' % st[1].stats.channel
                msg = 'calculated new magnitude for %s: %0.2f (channels: %s)' \
                      % (dict['Station'], dict['Mag'], dict['MagChannel'])
                self.textviewStdOutImproved.write(msg)
            
            elif 'MagMin2' in dict and 'MagMax2' in dict:
                amp = dict['MagMax2'] - dict['MagMin2']
                timedelta = abs(dict['MagMax2T'] - dict['MagMin2T'])
                mag = estimateMagnitude(dict['pazE'], amp, timedelta,
                                        dict['distHypo'])
                dict['Mag'] = mag
                dict['MagChannel'] = '%s' % st[2].stats.channel
                msg = 'calculated new magnitude for %s: %0.2f (channels: %s)' \
                      % (dict['Station'], dict['Mag'], dict['MagChannel'])
                self.textviewStdOutImproved.write(msg)
    
    #see http://www.scipy.org/Cookbook/LinearRegression for alternative routine
    #XXX replace with drawWadati()
    def drawWadati(self):
        """
        Shows a Wadati diagram plotting P time in (truncated) Julian seconds
        against S-P time for every station and doing a linear regression
        using rpy. An estimate of Vp/Vs is given by the slope + 1.
        """
        try:
            import rpy
        except:
            err = "Error: Package rpy could not be imported!\n" + \
                  "(We should switch to scipy polyfit, anyway!)"
            self.textviewStdErrImproved.write(err)
            return
        pTimes = []
        spTimes = []
        stations = []
        for i, dict in enumerate(self.dicts):
            if 'P' in dict and 'S' in dict:
                st = self.streams[i]
                p = st[0].stats.starttime
                p += dict['P']
                p = "%.3f" % p.getTimeStamp()
                p = float(p[-7:])
                pTimes.append(p)
                sp = dict['S'] - dict['P']
                spTimes.append(sp)
                stations.append(dict['Station'])
            else:
                continue
        if len(pTimes) < 2:
            err = "Error: Less than 2 P-S Pairs!"
            self.textviewStdErrImproved.write(err)
            return
        my_lsfit = rpy.r.lsfit(pTimes, spTimes)
        gradient = my_lsfit['coefficients']['X']
        intercept = my_lsfit['coefficients']['Intercept']
        vpvs = gradient + 1.
        ressqrsum = 0.
        for res in my_lsfit['residuals']:
            ressqrsum += (res ** 2)
        y0 = 0.
        x0 = - (intercept / gradient)
        x1 = max(pTimes)
        y1 = (gradient * float(x1)) + intercept

        fig = self.fig
        self.axWadati = fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=0.07, top=0.95, left=0.07, right=0.98)
        ax = self.axWadati
        ax = fig.add_subplot(111)

        ax.scatter(pTimes, spTimes)
        for i, station in enumerate(stations):
            ax.text(pTimes[i], spTimes[i], station, va = "top")
        ax.plot([x0, x1], [y0, y1])
        ax.axhline(0, color="blue", ls=":")
        # origin time estimated by wadati plot
        ax.axvline(x0, color="blue", ls=":",
                   label="origin time from wadati diagram")
        # origin time from event location
        if 'Time' in self.dictOrigin:
            otime = "%.3f" % self.dictOrigin['Time'].getTimeStamp()
            otime = float(otime[-7:])
            ax.axvline(otime, color="red", ls=":",
                       label="origin time from event location")
        ax.text(0.1, 0.7, "Vp/Vs: %.2f\nSum of squared residuals: %.3f" % \
                (vpvs, ressqrsum), transform=ax.transAxes)
        ax.text(0.1, 0.1, "Origin time from event location", color="red",
                transform=ax.transAxes)
        #ax.axis("auto")
        ax.set_xlim(min(x0 - 1, otime - 1), max(pTimes) + 1)
        ax.set_ylim(-1, max(spTimes) + 1)
        ax.set_xlabel("absolute P times (julian seconds, truncated)")
        ax.set_ylabel("P-S times (seconds)")
        ax.set_title("Wadati Diagram")
        self.canv.draw()

    def delWadati(self):
        if hasattr(self, "axWadati"):
            self.fig.delaxes(self.axWadati)
            del self.axWadati

    def drawStreamOverview(self):
        stNum = len(self.streams)
        self.axs = []
        self.plts = []
        self.trans = []
        self.t = []
        #we start all our x-axes at 0 with the starttime of the first (Z) trace
        starttime_global = self.streams[0][0].stats.starttime
        for i, st in enumerate(self.streams):
            tr = st[0]
            npts = tr.stats.npts
            smprt = tr.stats.sampling_rate
            #make sure that the relative times of the x-axes get mapped to our
            #global stream (absolute) starttime (starttime of first (Z) trace)
            starttime_local = tr.stats.starttime - starttime_global
            dt = 1. / smprt
            sampletimes = np.arange(starttime_local,
                    starttime_local + (dt * npts), dt)
            # sometimes our arange is one item too long (why??), so we just cut
            # off the last item if this is the case
            if len(sampletimes) == npts + 1:
                sampletimes = sampletimes[:-1]
            self.t.append(sampletimes)
            if i == 0:
                self.axs.append(self.fig.add_subplot(stNum,1,i+1))
            else:
                self.axs.append(self.fig.add_subplot(stNum, 1, i+1, 
                        sharex=self.axs[0], sharey=self.axs[0]))
                self.axs[i].xaxis.set_ticks_position("top")
            self.trans.append(matplotlib.transforms.blended_transform_factory(
                    self.axs[i].transData, self.axs[i].transAxes))
            self.axs[i].xaxis.set_major_formatter(FuncFormatter(
                                                  formatXTicklabels))
            if self.togglebuttonFilter.get_active():
                zerophase = self.checkbuttonZeroPhase.get_active()
                freq_highpass = self.spinbuttonHighpass.get_value()
                freq_lowpass = self.spinbuttonLowpass.get_value()
                filter_name = self.comboboxFilterType.get_active_text()
                if filter_name == "Bandpass":
                    filt_data = bandpass(tr.data, freq_highpass, freq_lowpass, df=tr.stats.sampling_rate, zerophase=zerophase)
                elif filter_name == "Bandstop":
                    filt_data = bandstop(tr.data, freq_highpass, freq_lowpass, df=tr.stats.sampling_rate, zerophase=zerophase)
                elif filter_name == "Lowpass":
                    filt_data = lowpass(tr.data, freq_lowpass, df=tr.stats.sampling_rate, zerophase=zerophase)
                elif filter_name == "Highpass":
                    filt_data = highpass(tr.data, freq_highpass, df=tr.stats.sampling_rate, zerophase=zerophase)
                self.plts.append(self.axs[i].plot(self.t[i], filt_data, color='k',zorder=1000)[0])
            else:
                self.plts.append(self.axs[i].plot(self.t[i], tr.data, color='k',zorder=1000)[0])
            self.axs[i].text(0.01, 0.95, st[0].stats.station, va="top",
                             ha="left", fontsize=18, color="b", zorder=10000,
                             transform=self.axs[i].transAxes)
        self.axs[-1].xaxis.set_ticks_position("both")
        self.supTit = self.fig.suptitle("%s.%03d -- %s.%03d" % (tr.stats.starttime.strftime("%Y-%m-%d  %H:%M:%S"),
                                                         tr.stats.starttime.microsecond / 1e3 + 0.5,
                                                         tr.stats.endtime.strftime("%H:%M:%S"),
                                                         tr.stats.endtime.microsecond / 1e3 + 0.5), ha="left", va="bottom", x=0.01, y=0.01)
        self.xMin, self.xMax=self.axs[0].get_xlim()
        self.yMin, self.yMax=self.axs[0].get_ylim()
        self.fig.subplots_adjust(bottom=0.001, hspace=0.000, right=0.999, top=0.999, left=0.001)
        self.toolbar.update()
        self.toolbar.pan(False)
        self.toolbar.zoom(True)

    def drawEventMap(self):
        dM = self.dictMagnitude
        dO = self.dictOrigin
        if dO == {}:
            err = "Error: No hypocenter data!"
            self.textviewStdErrImproved.write(err)
            return
        #toolbar.pan()
        #XXX self.figEventMap.canvas.widgetlock.release(toolbar)
        self.axEventMap = self.fig.add_subplot(111)
        self.axEventMap.set_aspect('equal', adjustable="datalim")
        self.fig.subplots_adjust(bottom=0.07, top=0.95, left=0.07, right=0.98)
        self.axEventMap.scatter([dO['Longitude']], [dO['Latitude']], 30,
                                color='red', marker='o')
        errLon, errLat = utlLonLat(dO['Longitude'], dO['Latitude'],
                                   dO['Longitude Error'], dO['Latitude Error'])
        errLon -= dO['Longitude']
        errLat -= dO['Latitude']
        ypos = 0.97
        xpos = 0.03
        self.axEventMap.text(xpos, ypos,
                             '%7.3f +/- %0.2fkm\n' % \
                             (dO['Longitude'], dO['Longitude Error']) + \
                             '%7.3f +/- %0.2fkm\n' % \
                             (dO['Latitude'], dO['Latitude Error']) + \
                             '  %.1fkm +/- %.1fkm' % \
                             (dO['Depth'], dO['Depth Error']),
                             va='top', ha='left', family='monospace',
                             transform=self.axEventMap.transAxes)
        if 'Standarderror' in dO:
            self.axEventMap.text(xpos, ypos, "\n\n\n\n Residual: %.3f s" % \
                    dO['Standarderror'], va='top', ha='left',
                    color=self.dictPhaseColors['P'],
                    transform=self.axEventMap.transAxes,
                    family='monospace')
        if 'Magnitude' in dM and 'Uncertainty' in dM:
            self.netMagLabel = '\n\n\n\n\n %.2f (Var: %.2f)' % \
                    (dM['Magnitude'], dM['Uncertainty'])
            self.netMagText = self.axEventMap.text(xpos, ypos,
                    self.netMagLabel, va='top', ha='left',
                    transform=self.axEventMap.transAxes,
                    color=self.dictPhaseColors['Mag'], family='monospace')
        errorell = Ellipse(xy = [dO['Longitude'], dO['Latitude']],
                width=errLon, height=errLat, angle=0, fill=False)
        self.axEventMap.add_artist(errorell)
        self.scatterMagIndices = []
        self.scatterMagLon = []
        self.scatterMagLat = []
        for i, dict in enumerate(self.dicts):
            # determine which stations are used in location
            if 'Pres' in dict or 'Sres' in dict:
                stationColor = 'black'
            else:
                stationColor = 'gray'
            # plot stations at respective coordinates with names
            self.axEventMap.scatter([dict['StaLon']], [dict['StaLat']], s=300,
                                    marker='v', color='',
                                    edgecolor=stationColor)
            self.axEventMap.text(dict['StaLon'], dict['StaLat'],
                                 '  ' + dict['Station'],
                                 color=stationColor, va='top',
                                 family='monospace')
            if 'Pres' in dict:
                presinfo = '\n\n %+0.3fs' % dict['Pres']
                if 'PPol' in dict:
                    presinfo += '  %s' % dict['PPol']
                self.axEventMap.text(dict['StaLon'], dict['StaLat'], presinfo,
                                     va='top', family='monospace',
                                     color=self.dictPhaseColors['P'])
            if 'Sres' in dict:
                sresinfo = '\n\n\n %+0.3fs' % dict['Sres']
                if 'SPol' in dict:
                    sresinfo += '  %s' % dict['SPol']
                self.axEventMap.text(dict['StaLon'], dict['StaLat'], sresinfo,
                                     va='top', family='monospace',
                                     color=self.dictPhaseColors['S'])
            if 'Mag' in dict:
                self.scatterMagIndices.append(i)
                self.scatterMagLon.append(dict['StaLon'])
                self.scatterMagLat.append(dict['StaLat'])
                self.axEventMap.text(dict['StaLon'], dict['StaLat'],
                                     '  ' + dict['Station'], va='top',
                                     family='monospace')
                self.axEventMap.text(dict['StaLon'], dict['StaLat'],
                                     '\n\n\n\n  %0.2f (%s)' % \
                                     (dict['Mag'], dict['MagChannel']),
                                     va='top', family='monospace',
                                     color=self.dictPhaseColors['Mag'])
            if len(self.scatterMagLon) > 0 :
                self.scatterMag = self.axEventMap.scatter(self.scatterMagLon,
                        self.scatterMagLat, s=150, marker='v', color='',
                        edgecolor='black', picker=10)
                
        self.axEventMap.set_xlabel('Longitude')
        self.axEventMap.set_ylabel('Latitude')
        time = dO['Time']
        timestr = time.strftime("%Y-%m-%d  %H:%M:%S")
        timestr += ".%02d" % (time.microsecond / 1e4 + 0.5)
        self.axEventMap.set_title(timestr)
        #####XXX disabled because it plots the wrong info if the event was
        ##### fetched from seishub
        #####lines = open(self.threeDlocOutfile).readlines()
        #####infoEvent = lines[0].rstrip()
        #####infoPicks = ''
        #####for line in lines[1:]:
        #####    infoPicks += line
        #####self.axEventMap.text(0.02, 0.95, infoEvent, transform = self.axEventMap.transAxes,
        #####                  fontsize = 12, verticalalignment = 'top',
        #####                  family = 'monospace')
        #####self.axEventMap.text(0.02, 0.90, infoPicks, transform = self.axEventMap.transAxes,
        #####                  fontsize = 10, verticalalignment = 'top',
        #####                  family = 'monospace')
        # save id to disconnect when switching back to stream dislay
        self.eventMapPickEvent = self.canv.mpl_connect('pick_event',
                                                       self.selectMagnitudes)
        try:
            self.scatterMag.set_facecolors(self.eventMapColors)
        except:
            pass

        # make hexbin scatter plot, if located with NLLoc
        # XXX no vital commands should come after this block, as we do not
        # handle exceptions!
        if 'Program' in dO and dO['Program'] == "NLLoc" and \
           os.path.isfile(self.nllocScatterBin):
            cmap = matplotlib.cm.gist_heat_r
            data = readNLLocScatter(self.nllocScatterBin,
                                    self.textviewStdErrImproved)
            data = data.swapaxes(0, 1)
            self.axEventMap.hexbin(data[0], data[1], cmap=cmap, zorder=-1000)

            self.axEventMapInletXY = self.fig.add_axes([0.8, 0.8, 0.16, 0.16])
            self.axEventMapInletXZ = self.fig.add_axes([0.8, 0.73, 0.16, 0.06],
                    sharex=self.axEventMapInletXY)
            self.axEventMapInletZY = self.fig.add_axes([0.73, 0.8, 0.06, 0.16],
                    sharey=self.axEventMapInletXY)
            
            # z axis in km
            self.axEventMapInletXY.hexbin(data[0], data[1], cmap=cmap)
            self.axEventMapInletXZ.hexbin(data[0], data[2]/1000., cmap=cmap)
            self.axEventMapInletZY.hexbin(data[2]/1000., data[1], cmap=cmap)

            self.axEventMapInletXZ.invert_yaxis()
            self.axEventMapInletZY.invert_xaxis()
            self.axEventMapInletXY.axis("equal")
            
            formatter = FormatStrFormatter("%.3f")
            self.axEventMapInletXY.xaxis.set_major_formatter(formatter)
            self.axEventMapInletXY.yaxis.set_major_formatter(formatter)
            
            # only draw very few ticklabels in our tiny subaxes
            for ax in [self.axEventMapInletXZ.xaxis,
                       self.axEventMapInletXZ.yaxis,
                       self.axEventMapInletZY.xaxis,
                       self.axEventMapInletZY.yaxis]:
                ax.set_major_locator(MaxNLocator(nbins=3))
            
            # hide ticklabels on XY plot
            for ax in [self.axEventMapInletXY.xaxis,
                       self.axEventMapInletXY.yaxis]:
                plt.setp(ax.get_ticklabels(), visible=False)

    def delEventMap(self):
        try:
            self.canv.mpl_disconnect(self.eventMapPickEvent)
        except AttributeError:
            pass
        if hasattr(self, "axEventMapInletXY"):
            self.fig.delaxes(self.axEventMapInletXY)
            del self.axEventMapInletXY
        if hasattr(self, "axEventMapInletXZ"):
            self.fig.delaxes(self.axEventMapInletXZ)
            del self.axEventMapInletXZ
        if hasattr(self, "axEventMapInletZY"):
            self.fig.delaxes(self.axEventMapInletZY)
            del self.axEventMapInletZY
        if hasattr(self, "axEventMap"):
            self.fig.delaxes(self.axEventMap)
            del self.axEventMap

    def selectMagnitudes(self, event):
        if not self.togglebuttonShowMap.get_active():
            return
        if event.artist != self.scatterMag:
            return
        i = self.scatterMagIndices[event.ind[0]]
        j = event.ind[0]
        dict = self.dicts[i]
        dict['MagUse'] = not dict['MagUse']
        #print event.ind[0]
        #print i
        #print event.artist
        #for di in self.dicts:
        #    print di['MagUse']
        #print i
        #print self.dicts[i]['MagUse']
        if dict['MagUse']:
            self.eventMapColors[j] = (0.,  1.,  0.,  1.)
        else:
            self.eventMapColors[j] = (0.,  0.,  0.,  0.)
        #print self.eventMapColors
        self.scatterMag.set_facecolors(self.eventMapColors)
        #print self.scatterMag.get_facecolors()
        #event.artist.set_facecolors(self.eventMapColors)
        self.updateNetworkMag()
        self.canv.draw()
    
    def dicts2hypo71Stations(self):
        """
        Returns the station location information in self.dicts in hypo71
        stations file format as a string. This string can then be written to
        a file.
        """
        fmt = "%6s%02i%05.2fN%03i%05.2fE%4i\n"
        hypo71_string = ""

        for i, dict in enumerate(self.dicts):
            sta = dict['Station']
            lon = dict['StaLon']
            lon_deg = int(lon)
            lon_min = (lon - lon_deg) * 60.
            lat = dict['StaLat']
            lat_deg = int(lat)
            lat_min = (lat - lat_deg) * 60.
            # hypo 71 format uses elevation in meters not kilometers
            ele = dict['StaEle'] * 1000
            hypo71_string += fmt % (sta, lat_deg, lat_min, lon_deg, lon_min,
                                    ele)

        return hypo71_string
    
    def dicts2hypo71Phases(self):
        """
        Returns the pick information in self.dicts in hypo71 phase file format
        as a string. This string can then be written to a file.

        Information on the file formats can be found at:
        http://geopubs.wr.usgs.gov/open-file/of02-171/of02-171.pdf p.30

        Quote:
        The traditional USGS phase data input format (not Y2000 compatible)
        Some fields were added after the original HYPO71 phase format
        definition.
        
        Col. Len. Format Data
         1    4  A4       4-letter station site code. Also see col 78.
         5    2  A2       P remark such as "IP". If blank, any P time is
                          ignored.
         7    1  A1       P first motion such as U, D, +, -, C, D.
         8    1  I1       Assigned P weight code.
         9    1  A1       Optional 1-letter station component.
        10   10  5I2      Year, month, day, hour and minute.
        20    5  F5.2     Second of P arrival.
        25    1  1X       Presently unused.
        26    6  6X       Reserved remark field. This field is not copied to
                          output files.
        32    5  F5.2     Second of S arrival. The S time will be used if this
                          field is nonblank.
        37    2  A2, 1X   S remark such as "ES".
        40    1  I1       Assigned weight code for S.
        41    1  A1, 3X   Data source code. This is copied to the archive
                          output.
        45    3  F3.0     Peak-to-peak amplitude in mm on Develocorder viewer
                          screen or paper record.
        48    3  F3.2     Optional period in seconds of amplitude read on the
                          seismogram. If blank, use the standard period from
                          station file.
        51    1  I1       Amplitude magnitude weight code. Same codes as P & S.
        52    3  3X       Amplitude magnitude remark (presently unused).
        55    4  I4       Optional event sequence or ID number. This number may
                          be replaced by an ID number on the terminator line.
        59    4  F4.1     Optional calibration factor to use for amplitude
                          magnitudes. If blank, the standard cal factor from
                          the station file is used.
        63    3  A3       Optional event remark. Certain event remarks are
                          translated into 1-letter codes to save in output.
        66    5  F5.2     Clock correction to be added to both P and S times.
        71    1  A1       Station seismogram remark. Unused except as a label
                          on output.
        72    4  F4.0     Coda duration in seconds.
        76    1  I1       Duration magnitude weight code. Same codes as P & S.
        77    1  1X       Reserved.
        78    1  A1       Optional 5th letter of station site code.
        79    3  A3       Station component code.
        82    2  A2       Station network code.
        84-85 2  A2     2-letter station location code (component extension).
        """

        fmtP = "%4s%1sP%1s%1i %15s"
        fmtS = "%12s%1sS%1s%1i\n"
        hypo71_string = ""

        for i, dict in enumerate(self.dicts):
            sta = dict['Station']
            if not 'P' in dict and not 'S' in dict:
                continue
            if 'P' in dict:
                t = self.streams[i][0].stats.starttime
                t += dict['P']
                date = t.strftime("%y%m%d%H%M%S")
                date += ".%02d" % (t.microsecond / 1e4 + 0.5)
                if 'POnset' in dict:
                    if dict['POnset'] == 'impulsive':
                        onset = 'I'
                    elif dict['POnset'] == 'emergent':
                        onset = 'E'
                    else: #XXX check for other names correctly!!!
                        onset = '?'
                else:
                    onset = '?'
                if 'PPol' in dict:
                    if dict['PPol'] == "up" or dict['PPol'] == "poorup":
                        polarity = "U"
                    elif dict['PPol'] == "down" or dict['PPol'] == "poordown":
                        polarity = "D"
                    else: #XXX check for other names correctly!!!
                        polarity = "?"
                else:
                    polarity = "?"
                if 'PWeight' in dict:
                    weight = int(dict['PWeight'])
                else:
                    weight = 0
                hypo71_string += fmtP % (sta, onset, polarity, weight, date)
            if 'S' in dict:
                if not 'P' in dict:
                    err = "Warning: Trying to print a Hypo2000 phase file " + \
                          "with an S phase without P phase.\n" + \
                          "This case might not be covered correctly and " + \
                          "could screw our file up!"
                    self.textviewStdErrImproved.write(err)
                t2 = self.streams[i][0].stats.starttime
                t2 += dict['S']
                # if the S time's absolute minute is higher than that of the
                # P pick, we have to add 60 to the S second count for the
                # hypo 2000 output file
                # +60 %60 is necessary if t.min = 57, t2.min = 2 e.g.
                mindiff = (t2.minute - t.minute + 60) % 60
                abs_sec = t2.second + (mindiff * 60)
                if abs_sec > 99:
                    err = "Warning: S phase seconds are greater than 99 " + \
                          "which is not covered by the hypo phase file " + \
                          "format! Omitting S phase of station %s!" % sta
                    self.textviewStdErrImproved.write(err)
                    hypo71_string += "\n"
                    continue
                date2 = str(abs_sec)
                date2 += ".%02d" % (t2.microsecond / 1e4 + 0.5)
                if 'SOnset' in dict:
                    if dict['SOnset'] == 'impulsive':
                        onset2 = 'I'
                    elif dict['SOnset'] == 'emergent':
                        onset2 = 'E'
                    else: #XXX check for other names correctly!!!
                        onset2 = '?'
                else:
                    onset2 = '?'
                if 'SPol' in dict:
                    if dict['SPol'] == "up" or dict['SPol'] == "poorup":
                        polarity2 = "U"
                    elif dict['SPol'] == "down" or dict['SPol'] == "poordown":
                        polarity2 = "D"
                    else: #XXX check for other names correctly!!!
                        polarity2 = "?"
                else:
                    polarity2 = "?"
                if 'SWeight' in dict:
                    weight2 = int(dict['SWeight'])
                else:
                    weight2 = 0
                hypo71_string += fmtS % (date2, onset2, polarity2, weight2)
            else:
                hypo71_string += "\n"

        return hypo71_string

    def dicts2XML(self):
        """
        Returns information of all dictionaries as xml file (type string)
        """
        xml =  Element("event")
        Sub(Sub(xml, "event_id"), "value").text = self.dictEvent['xmlEventID']
        event_type = Sub(xml, "event_type")
        Sub(event_type, "value").text = "manual"

        # if the sysop checkbox is checked, we set the account in the xml
        # to sysop (and also use sysop as the seishub user)
        if self.checkbuttonSysop.get_active():
            Sub(event_type, "account").text = "sysop"
        else:
            Sub(event_type, "account").text = self.server['User']
        
        Sub(event_type, "user").text = self.username

        Sub(event_type, "public").text = "%s" % \
                self.checkbuttonPublishEvent.get_active()
        
        # XXX standard values for unset keys!!!???!!!???
        epidists = []
        # go through all stream-dictionaries and look for picks
        for i, dict in enumerate(self.dicts):
            st = self.streams[i]

            # write P Pick info
            if 'P' in dict:
                pick = Sub(xml, "pick")
                wave = Sub(pick, "waveform")
                wave.set("networkCode", st[0].stats.network) 
                wave.set("stationCode", st[0].stats.station) 
                wave.set("channelCode", st[0].stats.channel) 
                wave.set("locationCode", st[0].stats.location) 
                date = Sub(pick, "time")
                # prepare time of pick
                picktime = st[0].stats.starttime
                picktime += dict['P']
                Sub(date, "value").text = picktime.isoformat() # + '.%06i' % picktime.microsecond)
                if 'PErr1' in dict and 'PErr2' in dict:
                    temp = dict['PErr2'] - dict['PErr1']
                    Sub(date, "uncertainty").text = str(temp)
                else:
                    Sub(date, "uncertainty")
                Sub(pick, "phaseHint").text = "P"
                phase_compu = ""
                if 'POnset' in dict:
                    Sub(pick, "onset").text = dict['POnset']
                    if dict['POnset'] == "impulsive":
                        phase_compu += "I"
                    elif dict['POnset'] == "emergent":
                        phase_compu += "E"
                else:
                    Sub(pick, "onset")
                    phase_compu += "?"
                phase_compu += "P"
                if 'PPol' in dict:
                    Sub(pick, "polarity").text = dict['PPol']
                    if dict['PPol'] == 'up':
                        phase_compu += "U"
                    elif dict['PPol'] == 'poorup':
                        phase_compu += "+"
                    elif dict['PPol'] == 'down':
                        phase_compu += "D"
                    elif dict['PPol'] == 'poordown':
                        phase_compu += "-"
                else:
                    Sub(pick, "polarity")
                    phase_compu += "?"
                if 'PWeight' in dict:
                    Sub(pick, "weight").text = '%i' % dict['PWeight']
                    phase_compu += "%1i" % dict['PWeight']
                else:
                    Sub(pick, "weight")
                    phase_compu += "?"
                Sub(Sub(pick, "min_amp"), "value") #XXX what is min_amp???
                
                if 'Psynth' in dict:
                    Sub(pick, "phase_compu").text = phase_compu
                    Sub(Sub(pick, "phase_res"), "value").text = str(dict['Pres'])
                    if 'PsynthWeight' in dict:
                        Sub(Sub(pick, "phase_weight"), "value").text = \
                                str(dict['PsynthWeight'])
                    else:
                        Sub(Sub(pick, "phase_weight"), "value")
                    Sub(Sub(pick, "phase_delay"), "value")
                    Sub(Sub(pick, "azimuth"), "value").text = str(dict['PAzim'])
                    Sub(Sub(pick, "incident"), "value").text = str(dict['PInci'])
                    Sub(Sub(pick, "epi_dist"), "value").text = \
                            str(dict['distEpi'])
                    Sub(Sub(pick, "hyp_dist"), "value").text = \
                            str(dict['distHypo'])
        
            # write S Pick info
            if 'S' in dict:
                axind = dict['Saxind']
                pick = Sub(xml, "pick")
                wave = Sub(pick, "waveform")
                wave.set("networkCode", st[axind].stats.network) 
                wave.set("stationCode", st[axind].stats.station) 
                wave.set("channelCode", st[axind].stats.channel) 
                wave.set("locationCode", st[axind].stats.location) 
                date = Sub(pick, "time")
                # prepare time of pick
                picktime = st[0].stats.starttime
                picktime += dict['S']
                Sub(date, "value").text = picktime.isoformat() # + '.%06i' % picktime.microsecond)
                if 'SErr1' in dict and 'SErr2' in dict:
                    temp = dict['SErr2'] - dict['SErr1']
                    Sub(date, "uncertainty").text = str(temp)
                else:
                    Sub(date, "uncertainty")
                Sub(pick, "phaseHint").text = "S"
                phase_compu = ""
                if 'SOnset' in dict:
                    Sub(pick, "onset").text = dict['SOnset']
                    if dict['SOnset'] == "impulsive":
                        phase_compu += "I"
                    elif dict['SOnset'] == "emergent":
                        phase_compu += "E"
                else:
                    Sub(pick, "onset")
                    phase_compu += "?"
                phase_compu += "S"
                if 'SPol' in dict:
                    Sub(pick, "polarity").text = dict['SPol']
                    if dict['SPol'] == 'up':
                        phase_compu += "U"
                    elif dict['SPol'] == 'poorup':
                        phase_compu += "+"
                    elif dict['SPol'] == 'down':
                        phase_compu += "D"
                    elif dict['SPol'] == 'poordown':
                        phase_compu += "-"
                else:
                    Sub(pick, "polarity")
                    phase_compu += "?"
                if 'SWeight' in dict:
                    Sub(pick, "weight").text = '%i' % dict['SWeight']
                    phase_compu += "%1i" % dict['SWeight']
                else:
                    Sub(pick, "weight")
                    phase_compu += "?"
                Sub(Sub(pick, "min_amp"), "value") #XXX what is min_amp???
                
                if 'Ssynth' in dict:
                    Sub(pick, "phase_compu").text = phase_compu
                    Sub(Sub(pick, "phase_res"), "value").text = str(dict['Sres'])
                    if 'SsynthWeight' in dict:
                        Sub(Sub(pick, "phase_weight"), "value").text = \
                                str(dict['SsynthWeight'])
                    else:
                        Sub(Sub(pick, "phase_weight"), "value")
                    Sub(Sub(pick, "phase_delay"), "value")
                    Sub(Sub(pick, "azimuth"), "value").text = str(dict['SAzim'])
                    Sub(Sub(pick, "incident"), "value").text = str(dict['SInci'])
                    Sub(Sub(pick, "epi_dist"), "value").text = \
                            str(dict['distEpi'])
                    Sub(Sub(pick, "hyp_dist"), "value").text = \
                            str(dict['distHypo'])

        #origin output
        dO = self.dictOrigin
        #we always have one key 'Program', if len > 1 we have real information
        #its possible that we have set the 'Program' key but afterwards
        #the actual program run does not fill our dictionary...
        if len(dO) > 1:
            origin = Sub(xml, "origin")
            Sub(origin, "program").text = dO['Program']
            date = Sub(origin, "time")
            Sub(date, "value").text = dO['Time'].isoformat() # + '.%03i' % self.dictOrigin['Time'].microsecond
            Sub(date, "uncertainty")
            lat = Sub(origin, "latitude")
            Sub(lat, "value").text = str(dO['Latitude'])
            Sub(lat, "uncertainty").text = str(dO['Latitude Error']) #XXX Lat Error in km!!
            lon = Sub(origin, "longitude")
            Sub(lon, "value").text = str(dO['Longitude'])
            Sub(lon, "uncertainty").text = str(dO['Longitude Error']) #XXX Lon Error in km!!
            depth = Sub(origin, "depth")
            Sub(depth, "value").text = str(dO['Depth'])
            Sub(depth, "uncertainty").text = str(dO['Depth Error'])
            if 'Depth Type' in dO:
                Sub(origin, "depth_type").text = str(dO['Depth Type'])
            else:
                Sub(origin, "depth_type")
            if 'Earth Model' in dO:
                Sub(origin, "earth_mod").text = dO['Earth Model']
            else:
                Sub(origin, "earth_mod")
            if dO['Program'] == "hyp2000":
                uncertainty = Sub(origin, "originUncertainty")
                Sub(uncertainty, "preferredDescription").text = "uncertainty ellipse"
                Sub(uncertainty, "horizontalUncertainty")
                Sub(uncertainty, "minHorizontalUncertainty")
                Sub(uncertainty, "maxHorizontalUncertainty")
                Sub(uncertainty, "azimuthMaxHorizontalUncertainty")
            else:
                Sub(origin, "originUncertainty")
            quality = Sub(origin, "originQuality")
            Sub(quality, "P_usedPhaseCount").text = '%i' % dO['used P Count']
            Sub(quality, "S_usedPhaseCount").text = '%i' % dO['used S Count']
            Sub(quality, "usedPhaseCount").text = '%i' % (dO['used P Count'] + dO['used S Count'])
            Sub(quality, "usedStationCount").text = '%i' % dO['used Station Count']
            Sub(quality, "associatedPhaseCount").text = '%i' % (dO['used P Count'] + dO['used S Count'])
            Sub(quality, "associatedStationCount").text = '%i' % len(self.dicts)
            Sub(quality, "depthPhaseCount").text = "0"
            Sub(quality, "standardError").text = str(dO['Standarderror'])
            Sub(quality, "azimuthalGap").text = str(dO['Azimuthal Gap'])
            Sub(quality, "groundTruthLevel")
            Sub(quality, "minimumDistance").text = str(dO['Minimum Distance'])
            Sub(quality, "maximumDistance").text = str(dO['Maximum Distance'])
            Sub(quality, "medianDistance").text = str(dO['Median Distance'])
        
        #magnitude output
        dM = self.dictMagnitude
        #we always have one key 'Program', if len > 1 we have real information
        #its possible that we have set the 'Program' key but afterwards
        #the actual program run does not fill our dictionary...
        if len(dM) > 1:
            magnitude = Sub(xml, "magnitude")
            Sub(magnitude, "program").text = dM['Program']
            mag = Sub(magnitude, "mag")
            if np.isnan(dM['Magnitude']):
                Sub(mag, "value")
                Sub(mag, "uncertainty")
            else:
                Sub(mag, "value").text = str(dM['Magnitude'])
                Sub(mag, "uncertainty").text = str(dM['Uncertainty'])
            Sub(magnitude, "type").text = "Ml"
            Sub(magnitude, "stationCount").text = '%i' % dM['Station Count']
            for i, dict in enumerate(self.dicts):
                st = self.streams[i]
                if 'Mag' in dict:
                    stationMagnitude = Sub(xml, "stationMagnitude")
                    mag = Sub(stationMagnitude, 'mag')
                    Sub(mag, 'value').text = str(dict['Mag'])
                    Sub(mag, 'uncertainty').text
                    Sub(stationMagnitude, 'station').text = str(dict['Station'])
                    if dict['MagUse']:
                        Sub(stationMagnitude, 'weight').text = str(1. / dM['Station Count'])
                    else:
                        Sub(stationMagnitude, 'weight').text = "0"
                    Sub(stationMagnitude, 'channels').text = str(dict['MagChannel'])
        
        #focal mechanism output
        dF = self.dictFocalMechanism
        #we always have one key 'Program', if len > 1 we have real information
        #its possible that we have set the 'Program' key but afterwards
        #the actual program run does not fill our dictionary...
        if len(dF) > 1:
            focmec = Sub(xml, "focalMechanism")
            Sub(focmec, "program").text = dF['Program']
            nodplanes = Sub(focmec, "nodalPlanes")
            nodplanes.set("preferredPlane", "1")
            nodplane1 = Sub(nodplanes, "nodalPlane1")
            strike = Sub(nodplane1, "strike")
            Sub(strike, "value").text = str(dF['Strike'])
            Sub(strike, "uncertainty")
            dip = Sub(nodplane1, "dip")
            Sub(dip, "value").text = str(dF['Dip'])
            Sub(dip, "uncertainty")
            rake = Sub(nodplane1, "rake")
            Sub(rake, "value").text = str(dF['Rake'])
            Sub(rake, "uncertainty")
            Sub(focmec, "stationPolarityCount").text = "%i" % \
                    dF['Station Polarity Count']
            Sub(focmec, "stationPolarityErrorCount").text = "%i" % dF['Errors']
            Sub(focmec, "possibleSolutionCount").text = "%i" % \
                    dF['Possible Solution Count']

        return tostring(xml, pretty_print=True, xml_declaration=True)
    
    def setXMLEventID(self):
        #XXX is problematic if two people make a location at the same second!
        # then one event is overwritten with the other during submission.
        self.dictEvent['xmlEventID'] = \
                datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    def uploadSeishub(self):
        """
        Upload xml file to seishub
        """

        # check, if the event should be uploaded as sysop. in this case we use
        # the sysop-useraccount in seishub for the upload (and also set
        # user_account in the xml to "sysop").
        # the correctness of the sysop password is tested when checking the
        # sysop box and entering the password immediately.
        if self.checkbuttonSysop.get_active():
            userid = "sysop"
            passwd = self.entrySysopPassword.get_text()
        else:
            userid = self.server['User']
            passwd = self.server['Password']

        auth = 'Basic ' + (base64.encodestring(userid + ':' + passwd)).strip()

        path = '/xml/seismology/event'
        
        # overwrite the same xml file always when using option local
        # which is intended for testing purposes only
        if self.options.local:
            self.dictEvent['xmlEventID'] = '19700101000000'
        # if we did no location at all, and only picks hould be saved the
        # EventID ist still not set, so we have to do this now.
        if self.dictEvent['xmlEventID'] is None:
            self.setXMLEventID()
        name = "obspyck_%s" % (self.dictEvent['xmlEventID']) #XXX id of the file
        # create XML and also save in temporary directory for inspection purposes
        msg = "creating xml..."
        self.textviewStdOutImproved.write(msg)
        data = self.dicts2XML()
        tmpfile = self.tmp_dir + name + ".xml"
        msg = "writing xml as %s (for debugging purposes only!)" % tmpfile
        self.textviewStdOutImproved.write(msg)
        open(tmpfile, "w").write(data)

        #construct and send the header
        webservice = httplib.HTTP(self.server['Server'])
        webservice.putrequest("PUT", path + '/' + name)
        webservice.putheader('Authorization', auth )
        webservice.putheader("Host", "localhost")
        webservice.putheader("User-Agent", "obspyck")
        webservice.putheader("Content-type", "text/xml; charset=\"UTF-8\"")
        webservice.putheader("Content-length", "%d" % len(data))
        webservice.endheaders()
        webservice.send(data)

        # get the response
        statuscode, statusmessage, header = webservice.getreply()
        msg = "Account: %s" % userid
        msg += "\nUser: %s" % self.username
        msg += "\nName: %s" % name
        msg += "\nServer: %s%s" % (self.server['Server'], path)
        msg += "\nResponse: %s %s" % (statuscode, statusmessage)
        #msg += "\nHeader:"
        #msg += "\n%s" % str(header).strip()
        self.textviewStdOutImproved.write(msg)

    def deleteEventInSeishub(self, resource_name):
        """
        Delete xml file from seishub.
        (Move to seishubs trash folder if this option is activated)
        """

        # check, if the event should be deleted as sysop. in this case we
        # use the sysop-useraccount in seishub for the DELETE request.
        # sysop may delete resources from any user.
        # at the moment deleted resources go to seishubs trash folder (and can
        # easily be resubmitted using the http interface).
        # the correctness of the sysop password is tested when checking the
        # sysop box and entering the password immediately.
        if self.checkbuttonSysop.get_active():
            userid = "sysop"
            passwd = self.entrySysopPassword.get_text()
        else:
            userid = self.server['User']
            passwd = self.server['Password']

        auth = 'Basic ' + (base64.encodestring(userid + ':' + passwd)).strip()

        path = '/xml/seismology/event'
        
        #construct and send the header
        webservice = httplib.HTTP(self.server['Server'])
        webservice.putrequest("DELETE", path + '/' + resource_name)
        webservice.putheader('Authorization', auth )
        webservice.putheader("Host", "localhost")
        webservice.putheader("User-Agent", "obspyck")
        webservice.endheaders()

        # get the response
        statuscode, statusmessage, header = webservice.getreply()
        msg = "Deleting Event!"
        msg += "\nAccount: %s" % userid
        msg += "\nUser: %s" % self.username
        msg += "\nName: %s" % resource_name
        msg += "\nServer: %s%s" % (self.server['Server'], path)
        msg += "\nResponse: %s %s" % (statuscode, statusmessage)
        #msg += "\nHeader:"
        #msg += "\n%s" % str(header).strip()
        self.textviewStdOutImproved.write(msg)
    
    def clearDictionaries(self):
        msg = "Clearing previous data."
        self.textviewStdOutImproved.write(msg)
        dont_delete = ['Station', 'StaLat', 'StaLon', 'StaEle',
                       'pazZ', 'pazN', 'pazE']
        for dict in self.dicts:
            for key in dict.keys():
                if not key in dont_delete:
                    del dict[key]
            dict['MagUse'] = True
        self.dictOrigin = {}
        self.dictMagnitude = {}
        self.dictFocalMechanism = {}
        self.focMechList = []
        self.focMechCurrent = None
        self.focMechCount = None
        self.dictEvent = {}
        self.dictEvent['xmlEventID'] = None

    def clearOriginMagnitudeDictionaries(self):
        msg = "Clearing previous origin and magnitude data."
        self.textviewStdOutImproved.write(msg)
        dont_delete = ['Station', 'StaLat', 'StaLon', 'StaEle', 'pazZ', 'pazN',
                       'pazE', 'P', 'PErr1', 'PErr2', 'POnset', 'PPol',
                       'PWeight', 'S', 'SErr1', 'SErr2', 'SOnset', 'SPol',
                       'SWeight', 'Saxind',
                       #dont delete the manually picked maxima/minima
                       'MagMin1', 'MagMin1T', 'MagMax1', 'MagMax1T',
                       'MagMin2', 'MagMin2T', 'MagMax2', 'MagMax2T',]
        # we need to delete all station magnitude information from all dicts
        for dict in self.dicts:
            for key in dict.keys():
                if not key in dont_delete:
                    del dict[key]
            dict['MagUse'] = True
        self.dictOrigin = {}
        self.dictMagnitude = {}
        self.dictEvent = {}
        self.dictEvent['xmlEventID'] = None

    def clearFocmecDictionary(self):
        msg = "Clearing previous focal mechanism data."
        self.textviewStdOutImproved.write(msg)
        self.dictFocalMechanism = {}
        self.focMechList = []
        self.focMechCurrent = None
        self.focMechCount = None

    def delAllItems(self):
        self.delPLine()
        self.delPErr1Line()
        self.delPErr2Line()
        self.delPLabel()
        self.delPsynthLine()
        self.delPsynthLabel()
        self.delSLine()
        self.delSErr1Line()
        self.delSErr2Line()
        self.delSLabel()
        self.delSsynthLine()
        self.delSsynthLabel()
        self.delMagMaxCross1()
        self.delMagMinCross1()
        self.delMagMaxCross2()
        self.delMagMinCross2()

    def drawAllItems(self):
        self.drawPLine()
        self.drawPErr1Line()
        self.drawPErr2Line()
        self.drawPLabel()
        self.drawPsynthLine()
        self.drawPsynthLabel()
        self.drawSLine()
        self.drawSErr1Line()
        self.drawSErr2Line()
        self.drawSLabel()
        self.drawSsynthLine()
        self.drawSsynthLabel()
        self.drawMagMaxCross1()
        self.drawMagMinCross1()
        self.drawMagMaxCross2()
        self.drawMagMinCross2()
    
    def getEventFromSeishub(self, resource_name):
        #document = xml.xpath(".//document_id")
        #document_id = document[self.seishubEventCurrent].text
        # Hack to show xml resource as document id
        resource_url = self.server['BaseUrl'] + "/xml/seismology/event/" + \
                       resource_name
        resource_req = urllib2.Request(resource_url)
        userid = self.server['User']
        passwd = self.server['Password']
        auth = base64.encodestring('%s:%s' % (userid, passwd))[:-1]
        resource_req.add_header("Authorization", "Basic %s" % auth)
        fp = urllib2.urlopen(resource_req)
        resource_xml = parse(fp)
        fp.close()
        if resource_xml.xpath(u".//event_type/account"):
            account = resource_xml.xpath(u".//event_type/account")[0].text
        else:
            account = None
        if resource_xml.xpath(u".//event_type/user"):
            user = resource_xml.xpath(u".//event_type/user")[0].text
        else:
            user = None

        #analyze picks:
        for pick in resource_xml.xpath(u".//pick"):
            # attributes
            id = pick.find("waveform").attrib
            network = id["networkCode"]
            station = id["stationCode"]
            location = id["locationCode"]
            channel = id['channelCode']
            streamnum = None
            # search for streamnumber corresponding to pick
            for i, dict in enumerate(self.dicts):
                if station.strip() != dict['Station']:
                    continue
                else:
                    streamnum = i
                    break
            if streamnum is None:
                err = "Warning: Did not find matching stream for pick " + \
                      "data with station id: \"%s\"" % station.strip()
                self.textviewStdErrImproved.write(err)
                continue
            # values
            time = pick.xpath(".//time/value")[0].text
            uncertainty = pick.xpath(".//time/uncertainty")[0].text
            try:
                onset = pick.xpath(".//onset")[0].text
            except:
                onset = None
            try:
                polarity = pick.xpath(".//polarity")[0].text
            except:
                polarity = None
            try:
                weight = pick.xpath(".//weight")[0].text
            except:
                weight = None
            try:
                phase_res = pick.xpath(".//phase_res/value")[0].text
            except:
                phase_res = None
            try:
                phase_weight = pick.xpath(".//phase_res/weight")[0].text
            except:
                phase_weight = None
            try:
                azimuth = pick.xpath(".//azimuth/value")[0].text
            except:
                azimuth = None
            try:
                incident = pick.xpath(".//incident/value")[0].text
            except:
                incident = None
            try:
                epi_dist = pick.xpath(".//epi_dist/value")[0].text
            except:
                epi_dist = None
            try:
                hyp_dist = pick.xpath(".//hyp_dist/value")[0].text
            except:
                hyp_dist = None
            # convert UTC time to seconds after stream starttime
            time = UTCDateTime(time)
            time -= self.streams[streamnum][0].stats.starttime
            # map uncertainty in seconds to error picks in seconds
            if uncertainty:
                uncertainty = float(uncertainty)
                uncertainty /= 2.
            # assign to dictionary
            dict = self.dicts[streamnum]
            if pick.xpath(".//phaseHint")[0].text == "P":
                dict['P'] = time
                if uncertainty:
                    dict['PErr1'] = time - uncertainty
                    dict['PErr2'] = time + uncertainty
                if onset:
                    dict['POnset'] = onset
                if polarity:
                    dict['PPol'] = polarity
                if weight:
                    dict['PWeight'] = int(weight)
                if phase_res:
                    # residual is defined as P-Psynth by NLLOC and 3dloc!
                    # XXX does this also hold for hyp2000???
                    dict['Psynth'] = time - float(phase_res)
                    dict['Pres'] = float(phase_res)
                # hypo2000 uses this weight internally during the inversion
                # this is not the same as the weight assigned during picking
                if phase_weight:
                    dict['PsynthWeight'] = phase_weight
                if azimuth:
                    dict['PAzim'] = float(azimuth)
                if incident:
                    dict['PInci'] = float(incident)
            if pick.xpath(".//phaseHint")[0].text == "S":
                dict['S'] = time
                # XXX maybe dangerous to check last character:
                if channel.endswith('N'):
                    dict['Saxind'] = 1
                if channel.endswith('E'):
                    dict['Saxind'] = 2
                if uncertainty:
                    dict['SErr1'] = time - uncertainty
                    dict['SErr2'] = time + uncertainty
                if onset:
                    dict['SOnset'] = onset
                if polarity:
                    dict['SPol'] = polarity
                if weight:
                    dict['SWeight'] = int(weight)
                if phase_res:
                    # residual is defined as S-Ssynth by NLLOC and 3dloc!
                    # XXX does this also hold for hyp2000???
                    dict['Ssynth'] = time - float(phase_res)
                    dict['Sres'] = float(phase_res)
                # hypo2000 uses this weight internally during the inversion
                # this is not the same as the weight assigned during picking
                if phase_weight:
                    dict['SsynthWeight'] = phase_weight
                if azimuth:
                    dict['SAzim'] = float(azimuth)
                if incident:
                    dict['SInci'] = float(incident)
            if epi_dist:
                dict['distEpi'] = float(epi_dist)
            if hyp_dist:
                dict['distHypo'] = float(hyp_dist)

        #analyze origin:
        try:
            origin = resource_xml.xpath(u".//origin")[0]
            try:
                program = origin.xpath(".//program")[0].text
                self.dictOrigin['Program'] = program
            except:
                pass
            try:
                time = origin.xpath(".//time/value")[0].text
                self.dictOrigin['Time'] = UTCDateTime(time)
            except:
                pass
            try:
                lat = origin.xpath(".//latitude/value")[0].text
                self.dictOrigin['Latitude'] = float(lat)
            except:
                pass
            try:
                lon = origin.xpath(".//longitude/value")[0].text
                self.dictOrigin['Longitude'] = float(lon)
            except:
                pass
            try:
                errX = origin.xpath(".//longitude/uncertainty")[0].text
                self.dictOrigin['Longitude Error'] = float(errX)
            except:
                pass
            try:
                errY = origin.xpath(".//latitude/uncertainty")[0].text
                self.dictOrigin['Latitude Error'] = float(errY)
            except:
                pass
            try:
                z = origin.xpath(".//depth/value")[0].text
                self.dictOrigin['Depth'] = float(z)
            except:
                pass
            try:
                errZ = origin.xpath(".//depth/uncertainty")[0].text
                self.dictOrigin['Depth Error'] = float(errZ)
            except:
                pass
            try:
                self.dictOrigin['Depth Type'] = \
                        origin.xpath(".//depth_type")[0].text
            except:
                pass
            try:
                self.dictOrigin['Earth Model'] = \
                        origin.xpath(".//earth_mod")[0].text
            except:
                pass
            try:
                self.dictOrigin['used P Count'] = \
                        int(origin.xpath(".//originQuality/P_usedPhaseCount")[0].text)
            except:
                pass
            try:
                self.dictOrigin['used S Count'] = \
                        int(origin.xpath(".//originQuality/S_usedPhaseCount")[0].text)
            except:
                pass
            try:
                self.dictOrigin['used Station Count'] = \
                        int(origin.xpath(".//originQuality/usedStationCount")[0].text)
            except:
                pass
            try:
                self.dictOrigin['Standarderror'] = \
                        float(origin.xpath(".//originQuality/standardError")[0].text)
            except:
                pass
            try:
                self.dictOrigin['Azimuthal Gap'] = \
                        float(origin.xpath(".//originQuality/azimuthalGap")[0].text)
            except:
                pass
            try:
                self.dictOrigin['Minimum Distance'] = \
                        float(origin.xpath(".//originQuality/minimumDistance")[0].text)
            except:
                pass
            try:
                self.dictOrigin['Maximum Distance'] = \
                        float(origin.xpath(".//originQuality/maximumDistance")[0].text)
            except:
                pass
            try:
                self.dictOrigin['Median Distance'] = \
                        float(origin.xpath(".//originQuality/medianDistance")[0].text)
            except:
                pass
        except:
            pass

        #analyze magnitude:
        try:
            magnitude = resource_xml.xpath(u".//magnitude")[0]
            try:
                program = magnitude.xpath(".//program")[0].text
                self.dictMagnitude['Program'] = program
            except:
                pass
            try:
                mag = magnitude.xpath(".//mag/value")[0].text
                self.dictMagnitude['Magnitude'] = float(mag)
                self.netMagLabel = '\n\n\n\n\n %.2f (Var: %.2f)' % (self.dictMagnitude['Magnitude'], self.dictMagnitude['Uncertainty'])
            except:
                pass
            try:
                magVar = magnitude.xpath(".//mag/uncertainty")[0].text
                self.dictMagnitude['Uncertainty'] = float(magVar)
            except:
                pass
            try:
                stacount = magnitude.xpath(".//stationCount")[0].text
                self.dictMagnitude['Station Count'] = int(stacount)
            except:
                pass
        except:
            pass

        #analyze stationmagnitudes:
        for stamag in resource_xml.xpath(u".//stationMagnitude"):
            station = stamag.xpath(".//station")[0].text
            streamnum = None
            # search for streamnumber corresponding to pick
            for i, dict in enumerate(self.dicts):
                if station.strip() != dict['Station']:
                    continue
                else:
                    streamnum = i
                    break
            if streamnum is None:
                err = "Warning: Did not find matching stream for station " + \
                      "magnitude data with id: \"%s\"" % station.strip()
                self.textviewStdErrImproved.write(err)
                continue
            # values
            mag = float(stamag.xpath(".//mag/value")[0].text)
            mag_channel = stamag.xpath(".//channels")[0].text
            mag_weight = float(stamag.xpath(".//weight")[0].text)
            if mag_weight == 0:
                mag_use = False
            else:
                mag_use = True
            # assign to dictionary
            dict = self.dicts[streamnum]
            dict['Mag'] = mag
            dict['MagUse'] = mag_use
            dict['MagChannel'] = mag_channel
        
        #analyze focal mechanism:
        try:
            focmec = resource_xml.xpath(u".//focalMechanism")[0]
            try:
                program = focmec.xpath(".//program")[0].text
                self.dictFocalMechanism['Program'] = program
            except:
                pass
            try:
                program = focmec.xpath(".//program")[0].text
                self.dictFocalMechanism['Program'] = program
            except:
                pass
            try:
                strike = focmec.xpath(".//nodalPlanes/nodalPlane1/strike/value")[0].text
                self.dictFocalMechanism['Strike'] = float(strike)
                self.focMechCount = 1
                self.focMechCurrent = 0
            except:
                pass
            try:
                dip = focmec.xpath(".//nodalPlanes/nodalPlane1/dip/value")[0].text
                self.dictFocalMechanism['Dip'] = float(dip)
            except:
                pass
            try:
                rake = focmec.xpath(".//nodalPlanes/nodalPlane1/rake/value")[0].text
                self.dictFocalMechanism['Rake'] = float(rake)
            except:
                pass
            try:
                staPolCount = focmec.xpath(".//stationPolarityCount")[0].text
                self.dictFocalMechanism['Station Polarity Count'] = int(staPolCount)
            except:
                pass
            try:
                staPolErrCount = focmec.xpath(".//stationPolarityErrorCount")[0].text
                self.dictFocalMechanism['Errors'] = int(staPolErrCount)
            except:
                pass
        except:
            pass
        msg = "Fetched event %i of %i: %s (account: %s, user: %s)"% \
              (self.seishubEventCurrent + 1, self.seishubEventCount,
               resource_name, account, user)
        self.textviewStdOutImproved.write(msg)

    def getEventListFromSeishub(self, starttime, endtime):
        """
        Searches for events in the database and returns a lxml ElementTree
        object. All events with at least one pick set in between start- and
        endtime are returned.

        :param starttime: Start datetime as UTCDateTime
        :param endtime: End datetime as UTCDateTime
        """
        
        # two search criteria are applied:
        # - first pick of event must be before stream endtime
        # - last pick of event must be after stream starttime
        # thus we get any event with at least one pick in between start/endtime
        url = self.server['BaseUrl'] + \
              "/seismology/event/getList?" + \
              "min_last_pick=%s&max_first_pick=%s" % \
              (str(starttime), str(endtime))
        req = urllib2.Request(url)
        userid = self.server['User']
        passwd = self.server['Password']
        auth = base64.encodestring('%s:%s' % (userid, passwd))[:-1]
        req.add_header("Authorization", "Basic %s" % auth)

        f = urllib2.urlopen(req)
        xml = parse(f)
        f.close()

        return xml

    def updateEventListFromSeishub(self, starttime, endtime):
        """
        Searches for events in the database and stores a list of resource
        names. All events with at least one pick set in between start- and
        endtime are returned.

        :param starttime: Start datetime as UTCDateTime
        :param endtime: End datetime as UTCDateTime
        """
        # get event list from seishub
        xml = self.getEventListFromSeishub(starttime, endtime)
        # populate list with resource information of all available events
        self.seishubEventList = xml.xpath(u"Item")

        self.seishubEventCount = len(self.seishubEventList)
        # we set the current event-pointer to the last list element, because we
        # iterate the counter immediately when fetching the first event...
        self.seishubEventCurrent = self.seishubEventCount - 1
        msg = "%i events are available from Seishub" % self.seishubEventCount
        for event in self.seishubEventList:
            resource_name = event.xpath(u"resource_name")[0].text
            account = event.xpath(u"account")
            user = event.xpath(u"user")
            if account:
                account = account[0].text
            else:
                account = None
            if user:
                user = user[0].text
            else:
                user = None
            msg += "\n  - %s (account: %s, user: %s)" % (resource_name,
                                                         account, user)
        self.textviewStdOutImproved.write(msg)

    def checkForSysopEventDuplicates(self, starttime, endtime):
        """
        checks if there is more than one sysop event with picks in between
        starttime and endtime. if that is the case, a warning is issued.
        the user should then resolve this conflict by deleting events until
        only one instance remains.
        at the moment this check is conducted for the current timewindow when
        submitting a sysop event.
        """
        # get event list from seishub
        xml = self.getEventListFromSeishub(starttime, endtime)

        list_events = xml.xpath("Item")

        list_sysop_events = []
        
        for element in list_events:
            account = element.xpath(u"account")
            if not account:
                continue
            if account[0].text == "sysop":
                resource_name = element.xpath(u"resource_name")[0].text
                list_sysop_events.append(resource_name)

        # if there is a possible duplicate, pop up a warning window and print a
        # warning in the GUI error textview:
        if len(list_sysop_events) > 1:
            err = "ObsPyck found more than one sysop event with picks in " + \
                  "the current time window! Please check if these are " + \
                  "duplicate events and delete old resources."
            errlist = "\n".join(list_sysop_events)
            self.textviewStdErrImproved.write(err)
            self.textviewStdErrImproved.write(errlist)

            dialog = gtk.MessageDialog(self.win, gtk.DIALOG_MODAL,
                                       gtk.MESSAGE_WARNING, gtk.BUTTONS_CLOSE)
            dialog.set_markup(err + "\n\n<b><tt>%s</tt></b>" % errlist)
            dialog.set_title("Possible Duplicate Event!")
            response = dialog.run()
            dialog.destroy()


