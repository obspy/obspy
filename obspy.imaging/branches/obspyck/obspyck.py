#!/usr/bin/env python

#check for textboxes and other stuff:
#http://code.enthought.com/projects/traits/docs/html/tutorials/traits_ui_scientific_app.html

#matplotlib.use('gtkagg')

from lxml.etree import SubElement as Sub, parse, tostring
from lxml.etree import fromstring, Element
from optparse import OptionParser
import numpy as np
import fnmatch
import shutil
import sys
import os
import subprocess
import httplib
import base64
import datetime
import time
import urllib2
import warnings
import tempfile

#sys.path.append('/baysoft/obspy/obspy/branches/symlink')
from obspy.core import read, UTCDateTime
from obspy.seishub import Client
from obspy.signal.filter import bandpass, bandpassZPHSH, bandstop, bandstopZPHSH
from obspy.signal.filter import lowpass, lowpassZPHSH, highpass, highpassZPHSH
from obspy.signal.util import utlLonLat, utlGeoKm
from obspy.signal.invsim import estimateMagnitude
from obspy.imaging.spectrogram import spectrogram
from obspy.imaging.beachball import Beachball

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor as mplMultiCursor
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter

#imports for the buttons
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.mathtext as mathtext
import matplotlib.artist as artist
import matplotlib.image as image

#gtk
import gtk
import gtk.glade
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as Toolbar

#XXX VERY dirty hack to unset for ALL widgets the property "CAN_FOCUS"
# we have to do this, so the focus always remains with our matplotlib
# inset and all events are directed to the matplotlib canvas...
# there is a bug in glade that does not set this flag to false even
# if it is selected in the glade GUI for the widget.
# see: https://bugzilla.gnome.org/show_bug.cgi?id=322340
def nofocus_recursive(widget):
    # we have to exclude SpinButtons otherwise we cannot put anything into
    # the spin button
    if not isinstance(widget, gtk.SpinButton):
        widget.unset_flags(("GTK_CAN_FOCUS", "GTK_RECEIVES_DEFAULT"))
    try:
        children = widget.get_children()
    except AttributeError:
        return
    for w in children:
        nofocus_recursive(w)

def appendTextview(GTKtextview, string):
    """
    Appends text 'string' to the given GTKtextview instance.
    At a certain length (currently 200 lines) we cut off some lines at the top
    of the corresponding text buffer.
    """
    buffer = GTKtextview.get_buffer()
    if buffer.get_line_count() > 200:
        start = buffer.get_start_iter()
        newstart = buffer.get_iter_at_line(150)
        buffer.delete(start, newstart)
    end = buffer.get_end_iter()
    buffer.insert(end, "\n" + string)
    end = buffer.get_end_iter()
    endmark = buffer.create_mark("end", end, False)
    GTKtextview.scroll_mark_onscreen(endmark)

#Monkey patch (need to remember the ids of the mpl_connect-statements to remove them later)
#See source: http://matplotlib.sourcearchive.com/documentation/0.98.1/widgets_8py-source.html
class MultiCursor(mplMultiCursor):
    def __init__(self, canvas, axes, useblit=True, **lineprops):
        self.canvas = canvas
        self.axes = axes
        xmin, xmax = axes[-1].get_xlim()
        xmid = 0.5*(xmin+xmax)
        self.lines = [ax.axvline(xmid, visible=False, **lineprops) for ax in axes]
        self.visible = True
        self.useblit = useblit
        self.background = None
        self.needclear = False
        self.id1=self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.id2=self.canvas.mpl_connect('draw_event', self.clear)
    
def getCoord(client, network, station):
    """
    Returns longitude, latitude and elevation of given station from given
    client instance
    """
    coord = []

    resource = "dataless.seed.%s_%s.xml" % (network, station)
    xml = fromstring(client.station.getResource(resource, format='metadata'))

    for attrib in [u'Longitude (\xb0)', u'Latitude (\xb0)',  u'Elevation (m)']:
        node =  xml.xpath(u".//item[@title='%s']" % attrib)[0]
        value = float(node.getchildren()[0].attrib['text'])
        coord.append(value)

    return coord

def formatXTicklabels(x,pos):
    """
    Make a nice formatting for y axis ticklabels: minutes:seconds.microsec
    """
    # x is of type numpy.float64, the string representation of that float
    # strips of all tailing zeros
    # pos returns the position of x on the axis while zooming, None otherwise
    min = int(x / 60.)
    if min > 0:
        sec = x % 60
        return "%i:%06.3f" % (min, sec)
    else:
        return "%.3f" % x

class PickingGUI:
    
    ###########################################################################
    # Start of list of event handles that get connected to GUI Elements       #
    # Note: All fundtions starting with "on_" get connected to GUI Elements   #
    ###########################################################################
    def on_windowObspyck_destroy(self, event):
        self.cleanQuit()

    def on_buttonClearAll_clicked(self, event):
        self.delAllItems()
        self.clearDictionaries()
        self.drawAllItems()
        self.redraw()

    def on_buttonClearOrigMag_clicked(self, event):
        self.delAllItems()
        self.clearOriginMagnitudeDictionaries()
        self.drawAllItems()
        self.redraw()

    def on_buttonClearFocMec_clicked(self, event):
        self.clearFocmecDictionary()

    def on_buttonDoHyp2000_clicked(self, event):
        self.delAllItems()
        self.clearOriginMagnitudeDictionaries()
        self.dictOrigin['Program'] = "hyp2000"
        self.doHyp2000()
        self.loadHyp2000Data()
        self.calculateEpiHypoDists()
        self.dictMagnitude['Program'] = "obspy"
        self.calculateStationMagnitudes()
        self.updateNetworkMag()
        self.showEventMap()
        self.drawAllItems()
        self.redraw()

    def on_buttonDo3dloc_clicked(self, event):
        self.delAllItems()
        self.clearOriginMagnitudeDictionaries()
        self.dictOrigin['Program'] = "3dloc"
        self.do3dLoc()
        self.load3dlocSyntheticPhases()
        self.load3dlocData()
        self.calculateEpiHypoDists()
        self.dictMagnitude['Program'] = "obspy"
        self.calculateStationMagnitudes()
        self.updateNetworkMag()
        self.showEventMap()
        self.drawAllItems()
        self.redraw()

    def on_buttonCalcMag_clicked(self, event):
        self.calculateEpiHypoDists()
        self.dictMagnitude['Program'] = "obspy"
        self.calculateStationMagnitudes()
        self.updateNetworkMag()

    def on_buttonDoFocmec_clicked(self, event):
        self.clearFocmecDictionary()
        self.dictFocalMechanism['Program'] = "focmec"
        self.doFocmec()

    def on_buttonShowMap_clicked(self, event):
        self.showEventMap()

    def on_buttonShowFocMec_clicked(self, event):
        self.showFocMec()

    def on_buttonNextFocMec_clicked(self, event):
        self.nextFocMec()
        self.showFocMec()

    def on_buttonShowWadati_clicked(self, event):
        self.showWadati()

    def on_buttonGetNextEvent_clicked(self, event):
        self.delAllItems()
        self.clearDictionaries()
        self.getNextEventFromSeishub(self.streams[0][0].stats.starttime, 
                                     self.streams[0][0].stats.endtime)
        self.drawAllItems()
        self.redraw()

    def on_buttonSendEvent_clicked(self, event):
        self.uploadSeishub()

    def on_checkbuttonPublicEvent_toggled(self, event):
        newstate = self.checkbuttonPublicEvent.get_active()
        msg = "Setting \"public\" flag of event to: %s" % newstate
        appendTextview(self.textviewStdOut, msg)

    def on_buttonSetFocusOnPlot_clicked(self, event):
        self.setFocusToMatplotlib()

    def on_buttonDebug_clicked(self, event):
        self.debug()

    def on_buttonQuit_clicked(self, event):
        self.cleanQuit()

    def on_buttonPreviousStream_clicked(self, event):
        self.stPt = (self.stPt - 1) % self.stNum
        xmin, xmax = self.axs[0].get_xlim()
        self.delAxes()
        self.drawAxes()
        self.drawSavedPicks()
        self.multicursorReinit()
        self.axs[0].set_xlim(xmin, xmax)
        self.updatePlot()
        msg = "Going to previous stream"
        self.updateStreamLabels()
        appendTextview(self.textviewStdOut, msg)

    def on_buttonNextStream_clicked(self, event):
        self.stPt = (self.stPt + 1) % self.stNum
        xmin, xmax = self.axs[0].get_xlim()
        self.delAxes()
        self.drawAxes()
        self.drawSavedPicks()
        self.multicursorReinit()
        self.axs[0].set_xlim(xmin, xmax)
        self.updatePlot()
        msg = "Going to next stream"
        self.updateStreamLabels()
        appendTextview(self.textviewStdOut, msg)

    def on_comboboxPhaseType_changed(self, event):
        self.updateMulticursorColor()
        #self.updateComboboxPhaseTypeColor()
        self.updateButtonPhaseTypeColor()
        self.redraw()

    def on_togglebuttonFilter_toggled(self, event):
        self.updatePlot()

    def on_comboboxFilterType_changed(self, event):
        if self.togglebuttonFilter.get_active():
            self.updatePlot()

    def on_checkbuttonZeroPhase_toggled(self, event):
        # if the filter flag is not set, we don't have to update the plot
        if self.togglebuttonFilter.get_active():
            self.updatePlot()

    def on_spinbuttonHighpass_value_changed(self, event):
        if not self.togglebuttonFilter.get_active() or \
           self.comboboxFilterType.get_active_text() == "Lowpass":
            return
        # if the filter flag is not set, we don't have to update the plot
        # XXX if we have a lowpass, we dont need to update!! Not yet implemented!! XXX
        if self.spinbuttonLowpass.get_value() < self.spinbuttonHighpass.get_value():
            err = "Warning: Lowpass frequency below Highpass frequency!"
            appendTextview(self.textviewStdErr, err)
        # XXX maybe the following check could be done nicer
        # XXX check this criterion!
        minimum  = float(self.streams[self.stPt][0].stats.sampling_rate) / \
                self.streams[self.stPt][0].stats.npts
        if self.spinbuttonHighpass.get_value() < minimum:
            err = "Warning: Lowpass frequency is not supported by length of trace!"
            appendTextview(self.textviewStdErr, err)
        self.updatePlot()
        # XXX we could use this for the combobox too!
        # reset focus to matplotlib figure
        self.canv.grab_focus()

    def on_spinbuttonLowpass_value_changed(self, event):
        if not self.togglebuttonFilter.get_active() or \
           self.comboboxFilterType.get_active_text() == "Highpass":
            return
        # if the filter flag is not set, we don't have to update the plot
        # XXX if we have a highpass, we dont need to update!! Not yet implemented!! XXX
        if self.spinbuttonLowpass.get_value() < self.spinbuttonHighpass.get_value():
            err = "Warning: Lowpass frequency below Highpass frequency!"
            appendTextview(self.textviewStdErr, err)
        # XXX maybe the following check could be done nicer
        # XXX check this criterion!
        maximum  = self.streams[self.stPt][0].stats.sampling_rate / 2.0
        if self.spinbuttonLowpass.get_value() > maximum:
            err = "Warning: Highpass frequency is lower than Nyquist!"
            appendTextview(self.textviewStdErr, err)
        self.updatePlot()
        # XXX we could use this for the combobox too!
        # reset focus to matplotlib figure
        self.canv.grab_focus()

    def on_togglebuttonSpectrogram_toggled(self, event):
        err = "Error: Not implemented yet!"
        appendTextview(self.textviewStdErr, err)
    ###########################################################################
    # End of list of event handles that get connected to GUI Elements         #
    ###########################################################################

        #>         self.buffer.insert (self.buffer.get_end_iter(), input_data)
        #>         if self.buffer.get_line_count() > 400:
        #>             self.buffer.delete(self.buffer.get_start_iter(),
        #>                                 self.buffer.get_iter_at_line (200))
        #>         mark = self.buffer.create_mark("end",
        #>                                 self.buffer.get_end_iter(), False)
        #>         self.textview.scroll_to_mark(mark, 0.05, True, 0.0, 1.0)

    def debug(self):
        import ipdb
        ipdb.set_trace()

    def setFocusToMatplotlib(self):
        self.canv.grab_focus()

    def cleanQuit(self):
        try:
            shutil.rmtree(self.tmp_dir)
        except:
            pass
        gtk.main_quit()

    def __init__(self, client = None, streams = None, options = None):
        self.client = client
        self.streams = streams
        self.options = options
        #Define some flags, dictionaries and plotting options
        self.flagWheelZoom = True #Switch use of mousewheel for zooming
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
                'setOnsetEmergent': 'e'}
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
        self.threeDlocPath = self.options.pluginpath + '/3dloc/'
        self.threeDlocOutfile = self.tmp_dir + '3dloc-out'
        self.threeDlocInfile = self.tmp_dir + '3dloc-in'
        # copy 3dloc files to temp directory (only na.in)
        subprocess.call('cp %s/* %s &> /dev/null' % \
                (self.threeDlocPath, self.tmp_dir), shell=True)
        self.threeDlocPreCall = 'rm %s %s &> /dev/null' \
                % (self.threeDlocOutfile, self.threeDlocInfile)
        self.threeDlocCall = 'export D3_VELOCITY=/scratch/rh_vel/vp_5836/;' + \
                'export D3_VELOCITY_2=/scratch/rh_vel/vs_32220/;' + \
                'cd %s; 3dloc_pitsa' % self.tmp_dir
        self.hyp2000Path = self.options.pluginpath + '/hyp_2000/'
        self.hyp2000Controlfile = self.hyp2000Path + 'bay2000.inp'
        self.hyp2000Phasefile = self.tmp_dir + 'hyp2000.pha'
        self.hyp2000Stationsfile = self.tmp_dir + 'stations.dat'
        self.hyp2000Summary = self.tmp_dir + 'hypo.prt'
        # copy hypo2000 files to temp directory
        subprocess.call('cp %s/* %s &> /dev/null' % \
                (self.hyp2000Path, self.tmp_dir), shell=True)
        self.hyp2000PreCall = 'rm %s %s %s &> /dev/null' \
                % (self.hyp2000Phasefile, self.hyp2000Stationsfile,
                   self.hyp2000Summary)
        self.hyp2000Call = 'export HYP2000_DATA=%s;' % (self.tmp_dir) + \
                           'cd $HYP2000_DATA;' + \
                           './hyp2000 < bay2000.inp &> /dev/null'
        self.focmecPath = self.options.pluginpath + '/focmec/'
        self.focmecPhasefile = self.tmp_dir + 'focmec.dat'
        self.focmecStdout = self.tmp_dir + 'focmec.stdout'
        self.focmecSummary = self.tmp_dir + 'focmec.out'
        # copy focmec files to temp directory
        subprocess.call('cp %s/* %s &> /dev/null' % \
                (self.focmecPath, self.tmp_dir), shell=True)
        self.focmecCall = 'cd %s;' % (self.tmp_dir) + \
                          './rfocmec'
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
        self.flagSpectrogram = False
        # indicates which of the available events from seishub was loaded
        self.seishubEventCurrent = None 
        # indicates how many events are available from seishub
        self.seishubEventCount = None
        # save username of current user
        self.username = os.environ['USER']
        # setup server information
        self.server = {}
        self.server['Name'] = self.options.servername # "teide"
        self.server['Port'] = self.options.port # 8080
        self.server['Server'] = self.server['Name'] + \
                                ":%i" % self.server['Port']
        self.server['BaseUrl'] = "http://" + self.server['Server']
        
        # If keybindings option is set only show keybindings and exit
        if self.options.keybindings:
            for k, v in self.dictKeybindings.iteritems():
                print "%s: \"%s\"" % (k, v)
            return

        # Return, if no streams are given
        if not streams:
            return

        # Define some forbidden scenarios.
        # We assume there are:
        # - either one Z or three ZNE traces
        # - no two streams for any station (of same network)
        sta_list = set()
        for i in range(len(self.streams))[::-1]:
            st = self.streams[i]
            net_sta = "%s:%s" % (st[0].stats.network.strip(),
                                 st[0].stats.station.strip())
            # Here we make sure that a station/network combination is not
            # present with two streams. XXX For dynamically acquired data from
            # seishub this is already done before initialising the GUI and
            # thus redundant. Here it is only necessary if working with
            # traces from local file system (option -l)
            if net_sta in sta_list:
                print "Warning: Station/Network combination \"%s\" already " \
                      % net_sta + "in stream list. Discarding stream."
                self.streams.pop(i)
                continue
            if not (len(st.traces) == 1 or len(st.traces) == 3):
                print 'Warning: All streams must have either one Z trace or a set of three ZNE traces. Stream discarded.'
                self.streams.pop(i)
                continue
            if len(st.traces) == 1 and st[0].stats.channel[-1] != 'Z':
                print 'Warning: All streams must have either one Z trace or a set of three ZNE traces. Stream discarded'
                self.streams.pop(i)
                continue
            if len(st.traces) == 3 and (st[0].stats.channel[-1] != 'Z' or
                                        st[1].stats.channel[-1] != 'N' or
                                        st[2].stats.channel[-1] != 'E' or
                                        st[0].stats.station.strip() !=
                                        st[1].stats.station.strip() or
                                        st[0].stats.station.strip() !=
                                        st[2].stats.station.strip()):
                print 'Warning: All streams must have either one Z trace or a set of three ZNE traces. Stream discarded.'
                self.streams.pop(i)
                continue
            sta_list.add(net_sta)

        #set up a list of dictionaries to store all picking data
        # set all station magnitude use-flags False
        self.dicts = []
        for i in range(len(self.streams)):
            self.dicts.append({})
        self.dictsMap = {} #XXX not used yet!
        self.eventMapColors = []
        for i in range(len(self.streams))[::-1]:
            self.dicts[i]['MagUse'] = True
            sta = self.streams[i][0].stats.station.strip()
            self.dicts[i]['Station'] = sta
            self.dictsMap[sta] = self.dicts[i]
            self.eventMapColors.append((0.,  1.,  0.,  1.))
            #XXX uncomment following lines for use with dynamically acquired data from seishub!
            net = self.streams[i][0].stats.network.strip()
            print "=" * 70
            print sta
            if net == '':
                net = 'BW'
                print "Warning: Got no network information, setting to default: BW"
            print "-" * 70
            date = self.streams[i][0].stats.starttime.date
            print 'fetching station metadata from seishub...'
            try:
                lon, lat, ele = getCoord(self.client,  net, sta) 
                self.dicts[i]['StaLon'] = lon
                self.dicts[i]['StaLat'] = lat
                self.dicts[i]['StaEle'] = ele / 1000. # all depths in km!
                print self.dicts[i]['StaLon'], self.dicts[i]['StaLat'], \
                      self.dicts[i]['StaEle']
                self.dicts[i]['pazZ'] = self.client.station.getPAZ(net, sta, date,
                        channel_id = self.streams[i][0].stats.channel)
                print self.dicts[i]['pazZ']
                if len(self.streams[i]) == 3:
                    self.dicts[i]['pazN'] = self.client.station.getPAZ(net, sta,
                            date, channel_id = self.streams[i][1].stats.channel)
                    self.dicts[i]['pazE'] = self.client.station.getPAZ(net, sta,
                            date, channel_id = self.streams[i][2].stats.channel)
                    print self.dicts[i]['pazN']
                    print self.dicts[i]['pazE']
            except:
                print 'Error: could not fetch station metadata. Discarding stream.'
                self.streams.pop(i)
                self.dicts.pop(i)
                continue
            print 'done.'
        print "=" * 70
        
        # exit if no streams are left after removing everthing with missing
        # information:
        if self.streams == []:
            print "Error: No streams."
            return

        # demean traces if not explicitly deactivated on command line
        if not self.options.nozeromean:
            for st in self.streams:
                for tr in st:
                    tr.data -= tr.data.mean()

        #Define a pointer to navigate through the streams
        self.stNum=len(self.streams)
        self.stPt=0
    
        #XXX gtk gui
        self.gla = gtk.glade.XML('obspyck.glade', 'windowObspyck')
        # commodity dictionary to connect event handles
        # example:
        # d = {'on_buttonQuit_clicked': gtk.main_quit}
        # self.gla.signal_autoconnect(d)
        d = {}
        # include every funtion starting with "on_" in the dictionary we use
        # to autoconnect to all the buttons etc. in the GTK GUI
        for func in dir(self):
            if func.startswith("on_"):
                exec("d['%s'] = self.%s" % (func, func))
        self.gla.signal_autoconnect(d)
        # get the main window widget and set its title
        self.win = self.gla.get_widget('windowObspyck')
        self.win.set_title("ObsPyck")
        # matplotlib code to generate an empty Axes
        # we define no dimensions for Figure because it will be
        # expanded to the whole empty space on main window widget
        self.fig = Figure()
        #self.fig.set_facecolor("0.9")
        # we bind the figure to the FigureCanvas, so that it will be
        # drawn using the specific backend graphic functions
        self.canv = FigureCanvas(self.fig)
        try:
            #might not be working with ion3 and other windowmanagers...
            #self.fig.set_size_inches(20, 10, forward = True)
            self.win.maximize()
        except:
            pass
        # embed the canvas into the empty area left in glade window
        place1 = self.gla.get_widget("hboxObspyck")
        place1.pack_start(self.canv, True, True)
        place2 = self.gla.get_widget("vboxObspyck")
        self.toolbar = Toolbar(self.canv, self.win)
        place2.pack_start(self.toolbar, False, False)
        self.toolbar.zoom()
        self.canv.widgetlock.release(self.toolbar)

        # define handles for all buttons/GUI-elements we interact with
        self.checkbuttonPublicEvent = \
                self.gla.get_widget("checkbuttonPublicEvent")
        self.buttonPreviousStream = self.gla.get_widget("buttonPreviousStream")
        self.labelStreamNumber = self.gla.get_widget("labelStreamNumber")
        self.labelStreamName = self.gla.get_widget("labelStreamName")
        self.buttonNextStream = self.gla.get_widget("buttonNextStream")
        self.buttonPhaseType = self.gla.get_widget("buttonPhaseType")
        self.comboboxPhaseType = self.gla.get_widget("comboboxPhaseType")
        self.togglebuttonFilter = self.gla.get_widget("togglebuttonFilter")
        self.comboboxFilterType = self.gla.get_widget("comboboxFilterType")
        self.checkbuttonZeroPhase = self.gla.get_widget("checkbuttonZeroPhase")
        self.spinbuttonHighpass = self.gla.get_widget("spinbuttonHighpass")
        self.spinbuttonLowpass = self.gla.get_widget("spinbuttonLowpass")
        self.togglebuttonSpectrogram = \
                self.gla.get_widget("togglebuttonSpectrogram")
        self.textviewStdOut = self.gla.get_widget("textviewStdOut")
        self.textviewStdErr = self.gla.get_widget("textviewStdErr")

        # Set up initial plot
        #self.fig = plt.figure()
        #self.fig.canvas.set_window_title("ObsPyck")
        #try:
        #    #not working with ion3 and other windowmanagers...
        #    self.fig.set_size_inches(20, 10, forward = True)
        #except:
        #    pass
        self.drawAxes()
        #redraw()
        #self.fig.canvas.draw()
        # Activate all mouse/key/Cursor-events
        self.canv.mpl_connect('key_press_event', self.keypress)
        self.canv.mpl_connect('scroll_event', self.scroll)
        self.canv.mpl_connect('button_release_event', self.buttonrelease)
        self.canv.mpl_connect('button_press_event', self.buttonpress)
        self.multicursor = MultiCursor(self.canv, self.axs, useblit=True,
                                       color='k', linewidth=1, ls='dotted')
        
        # there's a bug in glade so we have to set the default value for the
        # two comboboxes here by hand, otherwise the boxes are empty at startup
        # we also have to make a connection between the combobox labels and our
        # internal event handling (to determine what to do on the various key
        # press events...)
        # activate first item in the combobox designed with glade:
        self.comboboxPhaseType.set_active(0)
        self.comboboxFilterType.set_active(0)
        
        # correct some focus issues and start the GTK+ main loop
        # grab focus, otherwise the mpl key_press events get lost...
        # XXX how can we get the focus to the mpl box permanently???
        # >> although manually removing the "GTK_CAN_FOCUS" flag of all widgets
        # >> the combobox-type buttons grab the focus. if really necessary the
        # >> focus can be rest by clicking the "set focus on plot" button...
        # XXX a possible workaround would be to manually grab focus whenever
        # one of the combobox buttons or spinbuttons is clicked/updated (DONE!)
        nofocus_recursive(self.win)
        self.comboboxPhaseType.set_focus_on_click(False)
        self.comboboxFilterType.set_focus_on_click(False)
        self.canv.set_property("can_default", True)
        self.canv.set_property("can_focus", True)
        self.canv.grab_default()
        self.canv.grab_focus()
        # set the filter default values according to command line options
        # or command line default values
        self.spinbuttonHighpass.set_value(self.options.highpass)
        self.spinbuttonLowpass.set_value(self.options.lowpass)
        self.updateStreamLabels()
        self.canv.show()
        gtk.main()

    
    ## Trim all to same length, us Z as reference
    #start, end = stZ[0].stats.starttime, stZ[0].stats.endtime
    #stN.trim(start, end)
    #stE.trim(start, end)
    
    
    def drawAxes(self):
        npts = self.streams[self.stPt][0].stats.npts
        smprt = self.streams[self.stPt][0].stats.sampling_rate
        dt = 1. / smprt
        self.t = np.arange(0., dt * npts, dt)
        self.axs = []
        self.plts = []
        self.trans = []
        trNum = len(self.streams[self.stPt].traces)
        for i in range(trNum):
            if i == 0:
                self.axs.append(self.fig.add_subplot(trNum,1,i+1))
                self.trans.append(matplotlib.transforms.blended_transform_factory(self.axs[i].transData,
                                                                             self.axs[i].transAxes))
            else:
                self.axs.append(self.fig.add_subplot(trNum, 1, i+1, 
                        sharex=self.axs[0], sharey=self.axs[0]))
                self.trans.append(matplotlib.transforms.blended_transform_factory(self.axs[i].transData,
                                                                             self.axs[i].transAxes))
                self.axs[i].xaxis.set_ticks_position("top")
            self.axs[-1].xaxis.set_ticks_position("both")
            #self.axs[i].set_ylabel(self.streams[self.stPt][i].stats.station+" "+self.streams[self.stPt][i].stats.channel)
            self.axs[i].xaxis.set_major_formatter(FuncFormatter(formatXTicklabels))
            if not self.flagSpectrogram:
                self.plts.append(self.axs[i].plot(self.t, self.streams[self.stPt][i].data, color='k',zorder=1000)[0])
            else:
                spectrogram(self.streams[self.stPt][i].data,
                            self.streams[self.stPt][i].stats.sampling_rate,
                            axis = self.axs[i],
                            nwin = self.streams[self.stPt][i].stats.npts * 4 / self.streams[self.stPt][i].stats.sampling_rate)
        self.supTit = self.fig.suptitle("%s.%03d -- %s.%03d" % (self.streams[self.stPt][0].stats.starttime.strftime("%Y-%m-%d  %H:%M:%S"),
                                                         self.streams[self.stPt][0].stats.starttime.microsecond / 1e3 + 0.5,
                                                         self.streams[self.stPt][0].stats.endtime.strftime("%H:%M:%S"),
                                                         self.streams[self.stPt][0].stats.endtime.microsecond / 1e3 + 0.5))
        self.xMin, self.xMax=self.axs[0].get_xlim()
        self.yMin, self.yMax=self.axs[0].get_ylim()
        #self.fig.subplots_adjust(bottom=0.04, hspace=0.01, right=0.999, top=0.94, left=0.06)
        self.fig.subplots_adjust(bottom=0.001, hspace=0.000, right=0.999, top=0.999, left=0.001)
    
    def drawSavedPicks(self):
        self.drawPLine()
        self.drawPLabel()
        self.drawPErr1Line()
        self.drawPErr2Line()
        self.drawPsynthLine()
        self.drawPsynthLabel()
        self.drawSLine()
        self.drawSLabel()
        self.drawSErr1Line()
        self.drawSErr2Line()
        self.drawSsynthLine()
        self.drawSsynthLabel()
        self.drawMagMinCross1()
        self.drawMagMaxCross1()
        self.drawMagMinCross2()
        self.drawMagMaxCross2()
    
    def drawPLine(self):
        if not self.dicts[self.stPt].has_key('P'):
            return
        self.PLines=[]
        for i in range(len(self.axs)):
            self.PLines.append(self.axs[i].axvline(self.dicts[self.stPt]['P'],color=self.dictPhaseColors['P'],linewidth=self.axvlinewidths,label='P',linestyle=self.dictPhaseLinestyles['P']))
    
    def delPLine(self):
        try:
            for i in range(len(self.axs)):
                self.axs[i].lines.remove(self.PLines[i])
        except:
            pass
        try:
            del self.PLines
        except:
            pass
    
    def drawPsynthLine(self):
        if not self.dicts[self.stPt].has_key('Psynth'):
            return
        self.PsynthLines=[]
        for i in range(len(self.axs)):
            self.PsynthLines.append(self.axs[i].axvline(self.dicts[self.stPt]['Psynth'],color=self.dictPhaseColors['Psynth'],linewidth=self.axvlinewidths,label='Psynth',linestyle=self.dictPhaseLinestyles['Psynth']))
    
    def delPsynthLine(self):
        try:
            for i in range(len(self.axs)):
                self.axs[i].lines.remove(self.PsynthLines[i])
        except:
            pass
        try:
            del self.PsynthLines
        except:
            pass
    
    def drawPLabel(self):
        if not self.dicts[self.stPt].has_key('P'):
            return
        PLabelString = 'P:'
        if not self.dicts[self.stPt].has_key('POnset'):
            PLabelString += '_'
        else:
            if self.dicts[self.stPt]['POnset'] == 'impulsive':
                PLabelString += 'I'
            elif self.dicts[self.stPt]['POnset'] == 'emergent':
                PLabelString += 'E'
            else:
                PLabelString += '?'
        if not self.dicts[self.stPt].has_key('PPol'):
            PLabelString += '_'
        else:
            if self.dicts[self.stPt]['PPol'] == 'up':
                PLabelString += 'U'
            elif self.dicts[self.stPt]['PPol'] == 'poorup':
                PLabelString += '+'
            elif self.dicts[self.stPt]['PPol'] == 'down':
                PLabelString += 'D'
            elif self.dicts[self.stPt]['PPol'] == 'poordown':
                PLabelString += '-'
            else:
                PLabelString += '?'
        if not self.dicts[self.stPt].has_key('PWeight'):
            PLabelString += '_'
        else:
            PLabelString += str(self.dicts[self.stPt]['PWeight'])
        self.PLabel = self.axs[0].text(self.dicts[self.stPt]['P'], 1 - 0.04 * len(self.axs),
                                       '  ' + PLabelString, transform = self.trans[0],
                                       color = self.dictPhaseColors['P'],
                                       family = 'monospace')
    
    def delPLabel(self):
        try:
            self.axs[0].texts.remove(self.PLabel)
        except:
            pass
        try:
            del self.PLabel
        except:
            pass
    
    def drawPsynthLabel(self):
        if not self.dicts[self.stPt].has_key('Psynth'):
            return
        PsynthLabelString = 'Psynth: %+.3fs' % self.dicts[self.stPt]['Pres']
        self.PsynthLabel = self.axs[0].text(self.dicts[self.stPt]['Psynth'], 1 - 0.08 * len(self.axs), '  ' + PsynthLabelString,
                             transform = self.trans[0], color=self.dictPhaseColors['Psynth'])
    
    def delPsynthLabel(self):
        try:
            self.axs[0].texts.remove(self.PsynthLabel)
        except:
            pass
        try:
            del self.PsynthLabel
        except:
            pass
    
    def drawPErr1Line(self):
        if not self.dicts[self.stPt].has_key('P') or not self.dicts[self.stPt].has_key('PErr1'):
            return
        self.PErr1Lines=[]
        for i in range(len(self.axs)):
            self.PErr1Lines.append(self.axs[i].axvline(self.dicts[self.stPt]['PErr1'],ymin=0.25,ymax=0.75,color=self.dictPhaseColors['P'],linewidth=self.axvlinewidths,label='PErr1'))
    
    def delPErr1Line(self):
        try:
            for i in range(len(self.axs)):
                self.axs[i].lines.remove(self.PErr1Lines[i])
        except:
            pass
        try:
            del self.PErr1Lines
        except:
            pass
    
    def drawPErr2Line(self):
        if not self.dicts[self.stPt].has_key('P') or not self.dicts[self.stPt].has_key('PErr2'):
            return
        self.PErr2Lines=[]
        for i in range(len(self.axs)):
            self.PErr2Lines.append(self.axs[i].axvline(self.dicts[self.stPt]['PErr2'],ymin=0.25,ymax=0.75,color=self.dictPhaseColors['P'],linewidth=self.axvlinewidths,label='PErr2'))
    
    def delPErr2Line(self):
        try:
            for i in range(len(self.axs)):
                self.axs[i].lines.remove(self.PErr2Lines[i])
        except:
            pass
        try:
            del self.PErr2Lines
        except:
            pass

    def drawSLine(self):
        if not self.dicts[self.stPt].has_key('S'):
            return
        self.SLines=[]
        for i in range(len(self.axs)):
            self.SLines.append(self.axs[i].axvline(self.dicts[self.stPt]['S'],color=self.dictPhaseColors['S'],linewidth=self.axvlinewidths,label='S',linestyle=self.dictPhaseLinestyles['S']))
    
    def delSLine(self):
        try:
            for i in range(len(self.axs)):
                self.axs[i].lines.remove(self.SLines[i])
        except:
            pass
        try:
            del self.SLines
        except:
            pass
    
    def drawSsynthLine(self):
        if not self.dicts[self.stPt].has_key('Ssynth'):
            return
        self.SsynthLines=[]
        for i in range(len(self.axs)):
            self.SsynthLines.append(self.axs[i].axvline(self.dicts[self.stPt]['Ssynth'],color=self.dictPhaseColors['Ssynth'],linewidth=self.axvlinewidths,label='Ssynth',linestyle=self.dictPhaseLinestyles['Ssynth']))
    
    def delSsynthLine(self):
        try:
            for i in range(len(self.axs)):
                self.axs[i].lines.remove(self.SsynthLines[i])
        except:
            pass
        try:
            del self.SsynthLines
        except:
            pass
    
    def drawSLabel(self):
        if not self.dicts[self.stPt].has_key('S'):
            return
        SLabelString = 'S:'
        if not self.dicts[self.stPt].has_key('SOnset'):
            SLabelString += '_'
        else:
            if self.dicts[self.stPt]['SOnset'] == 'impulsive':
                SLabelString += 'I'
            elif self.dicts[self.stPt]['SOnset'] == 'emergent':
                SLabelString += 'E'
            else:
                SLabelString += '?'
        if not self.dicts[self.stPt].has_key('SPol'):
            SLabelString += '_'
        else:
            if self.dicts[self.stPt]['SPol'] == 'up':
                SLabelString += 'U'
            elif self.dicts[self.stPt]['SPol'] == 'poorup':
                SLabelString += '+'
            elif self.dicts[self.stPt]['SPol'] == 'down':
                SLabelString += 'D'
            elif self.dicts[self.stPt]['SPol'] == 'poordown':
                SLabelString += '-'
            else:
                SLabelString += '?'
        if not self.dicts[self.stPt].has_key('SWeight'):
            SLabelString += '_'
        else:
            SLabelString += str(self.dicts[self.stPt]['SWeight'])
        self.SLabel = self.axs[0].text(self.dicts[self.stPt]['S'], 1 - 0.04 * len(self.axs),
                                       '  ' + SLabelString, transform = self.trans[0],
                                       color = self.dictPhaseColors['S'],
                                       family = 'monospace')
    
    def delSLabel(self):
        try:
            self.axs[0].texts.remove(self.SLabel)
        except:
            pass
        try:
            del self.SLabel
        except:
            pass
    
    def drawSsynthLabel(self):
        if not self.dicts[self.stPt].has_key('Ssynth'):
            return
        SsynthLabelString = 'Ssynth: %+.3fs' % self.dicts[self.stPt]['Sres']
        self.SsynthLabel = self.axs[0].text(self.dicts[self.stPt]['Ssynth'], 1 - 0.08 * len(self.axs), '\n  ' + SsynthLabelString,
                             transform = self.trans[0], color=self.dictPhaseColors['Ssynth'])
    
    def delSsynthLabel(self):
        try:
            self.axs[0].texts.remove(self.SsynthLabel)
        except:
            pass
        try:
            del self.SsynthLabel
        except:
            pass
    
    def drawSErr1Line(self):
        if not self.dicts[self.stPt].has_key('S') or not self.dicts[self.stPt].has_key('SErr1'):
            return
        self.SErr1Lines=[]
        for i in range(len(self.axs)):
            self.SErr1Lines.append(self.axs[i].axvline(self.dicts[self.stPt]['SErr1'],ymin=0.25,ymax=0.75,color=self.dictPhaseColors['S'],linewidth=self.axvlinewidths,label='SErr1'))
    
    def delSErr1Line(self):
        try:
            for i in range(len(self.axs)):
                self.axs[i].lines.remove(self.SErr1Lines[i])
        except:
            pass
        try:
            del self.SErr1Lines
        except:
            pass
    
    def drawSErr2Line(self):
        if not self.dicts[self.stPt].has_key('S') or not self.dicts[self.stPt].has_key('SErr2'):
            return
        self.SErr2Lines=[]
        for i in range(len(self.axs)):
            self.SErr2Lines.append(self.axs[i].axvline(self.dicts[self.stPt]['SErr2'],ymin=0.25,ymax=0.75,color=self.dictPhaseColors['S'],linewidth=self.axvlinewidths,label='SErr2'))
    
    def delSErr2Line(self):
        try:
            for i in range(len(self.axs)):
                self.axs[i].lines.remove(self.SErr2Lines[i])
        except:
            pass
        try:
            del self.SErr2Lines
        except:
            pass
    
    def drawMagMinCross1(self):
        if not self.dicts[self.stPt].has_key('MagMin1') or len(self.axs) < 2:
            return
        #we have to force the graph to the old axes limits because of the completely new line object creation
        xlims = list(self.axs[0].get_xlim())
        ylims = list(self.axs[0].get_ylim())
        self.MagMinCross1 = self.axs[1].plot([self.dicts[self.stPt]['MagMin1T']] ,
                                   [self.dicts[self.stPt]['MagMin1']] ,
                                   markersize = self.magMarkerSize ,
                                   markeredgewidth = self.magMarkerEdgeWidth ,
                                   color = self.dictPhaseColors['Mag'],
                                   marker = self.magMinMarker, zorder=2000)[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delMagMinCross1(self):
        try:
            self.axs[1].lines.remove(self.MagMinCross1)
        except:
            pass
    
    def drawMagMaxCross1(self):
        if not self.dicts[self.stPt].has_key('MagMax1') or len(self.axs) < 2:
            return
        #we have to force the graph to the old axes limits because of the completely new line object creation
        xlims = list(self.axs[0].get_xlim())
        ylims = list(self.axs[0].get_ylim())
        self.MagMaxCross1 = self.axs[1].plot([self.dicts[self.stPt]['MagMax1T']],
                                   [self.dicts[self.stPt]['MagMax1']],
                                   markersize = self.magMarkerSize,
                                   markeredgewidth = self.magMarkerEdgeWidth,
                                   color = self.dictPhaseColors['Mag'],
                                   marker = self.magMaxMarker, zorder=2000)[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delMagMaxCross1(self):
        try:
            self.axs[1].lines.remove(self.MagMaxCross1)
        except:
            pass
    
    def drawMagMinCross2(self):
        if not self.dicts[self.stPt].has_key('MagMin2') or len(self.axs) < 3:
            return
        #we have to force the graph to the old axes limits because of the completely new line object creation
        xlims = list(self.axs[0].get_xlim())
        ylims = list(self.axs[0].get_ylim())
        self.MagMinCross2 = self.axs[2].plot([self.dicts[self.stPt]['MagMin2T']] ,
                                   [self.dicts[self.stPt]['MagMin2']] ,
                                   markersize = self.magMarkerSize ,
                                   markeredgewidth = self.magMarkerEdgeWidth ,
                                   color = self.dictPhaseColors['Mag'],
                                   marker = self.magMinMarker, zorder=2000)[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delMagMinCross2(self):
        try:
            self.axs[2].lines.remove(self.MagMinCross2)
        except:
            pass
    
    def drawMagMaxCross2(self):
        if not self.dicts[self.stPt].has_key('MagMax2') or len(self.axs) < 3:
            return
        #we have to force the graph to the old axes limits because of the completely new line object creation
        xlims = list(self.axs[0].get_xlim())
        ylims = list(self.axs[0].get_ylim())
        self.MagMaxCross2 = self.axs[2].plot([self.dicts[self.stPt]['MagMax2T']],
                                   [self.dicts[self.stPt]['MagMax2']],
                                   markersize = self.magMarkerSize,
                                   markeredgewidth = self.magMarkerEdgeWidth,
                                   color = self.dictPhaseColors['Mag'],
                                   marker = self.magMaxMarker, zorder=2000)[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delMagMaxCross2(self):
        try:
            self.axs[2].lines.remove(self.MagMaxCross2)
        except:
            pass
    
    def delP(self):
        try:
            del self.dicts[self.stPt]['P']
            msg = "P Pick deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delPsynth(self):
        try:
            del self.dicts[self.stPt]['Psynth']
            msg = "synthetic P Pick deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delPWeight(self):
        try:
            del self.dicts[self.stPt]['PWeight']
            msg = "P Pick weight deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delPPol(self):
        try:
            del self.dicts[self.stPt]['PPol']
            msg = "P Pick polarity deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delPOnset(self):
        try:
            del self.dicts[self.stPt]['POnset']
            msg = "P Pick onset deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delPErr1(self):
        try:
            del self.dicts[self.stPt]['PErr1']
            msg = "PErr1 Pick deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delPErr2(self):
        try:
            del self.dicts[self.stPt]['PErr2']
            msg = "PErr2 Pick deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delS(self):
        try:
            del self.dicts[self.stPt]['S']
            del self.dicts[self.stPt]['Saxind']
            msg = "S Pick deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delSsynth(self):
        try:
            del self.dicts[self.stPt]['Ssynth']
            msg = "synthetic S Pick deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delSWeight(self):
        try:
            del self.dicts[self.stPt]['SWeight']
            msg = "S Pick weight deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delSPol(self):
        try:
            del self.dicts[self.stPt]['SPol']
            msg = "S Pick polarity deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delSOnset(self):
        try:
            del self.dicts[self.stPt]['SOnset']
            msg = "S Pick onset deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delSErr1(self):
        try:
            del self.dicts[self.stPt]['SErr1']
            msg = "SErr1 Pick deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delSErr2(self):
        try:
            del self.dicts[self.stPt]['SErr2']
            msg = "SErr2 Pick deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delMagMin1(self):
        try:
            del self.dicts[self.stPt]['MagMin1']
            del self.dicts[self.stPt]['MagMin1T']
            msg = "Magnitude Minimum Estimation Pick deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delMagMax1(self):
        try:
            del self.dicts[self.stPt]['MagMax1']
            del self.dicts[self.stPt]['MagMax1T']
            msg = "Magnitude Maximum Estimation Pick deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delMagMin2(self):
        try:
            del self.dicts[self.stPt]['MagMin2']
            del self.dicts[self.stPt]['MagMin2T']
            msg = "Magnitude Minimum Estimation Pick deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
            
    def delMagMax2(self):
        try:
            del self.dicts[self.stPt]['MagMax2']
            del self.dicts[self.stPt]['MagMax2T']
            msg = "Magnitude Maximum Estimation Pick deleted"
            appendTextview(self.textviewStdOut, msg)
        except:
            pass
    
    def delAxes(self):
        for a in self.axs:
            try:
                self.fig.delaxes(a)
                del a
            except:
                pass
        try:
            self.fig.texts.remove(self.supTit)
        except:
            pass
    
    def redraw(self):
        for line in self.multicursor.lines:
            line.set_visible(False)
        self.canv.draw()
    
    def updatePlot(self):
        filt=[]
        #filter data
        if self.togglebuttonFilter.get_active():
            zerophase = self.checkbuttonZeroPhase.get_active()
            freq_highpass = self.spinbuttonHighpass.get_value()
            freq_lowpass = self.spinbuttonLowpass.get_value()
            filter_name = self.comboboxFilterType.get_active_text()
            for tr in self.streams[self.stPt].traces:
                if filter_name == "Bandpass":
                    filt.append(bandpass(tr.data, freq_highpass, freq_lowpass,
                            df=tr.stats.sampling_rate, zerophase=zerophase))
                    msg = "%s (zerophase=%s): %.2f-%.2f Hz" % \
                            (filter_name, zerophase, freq_highpass,
                             freq_lowpass)
                elif filter_name == "Bandstop":
                    filt.append(bandstop(tr.data, freq_highpass, freq_lowpass,
                            df=tr.stats.sampling_rate, zerophase=zerophase))
                    msg = "%s (zerophase=%s): %.2f-%.2f Hz" % \
                            (filter_name, zerophase, freq_highpass,
                             freq_lowpass)
                elif filter_name == "Lowpass":
                    filt.append(lowpass(tr.data, freq_lowpass,
                            df=tr.stats.sampling_rate, zerophase=zerophase))
                    msg = "%s (zerophase=%s): %.2f Hz" % (filter_name,
                                                          zerophase,
                                                          freq_lowpass)
                elif filter_name == "Highpass":
                    filt.append(highpass(tr.data, freq_highpass,
                            df=tr.stats.sampling_rate, zerophase=zerophase))
                    msg = "%s (zerophase=%s): %.2f Hz" % (filter_name,
                                                          zerophase,
                                                          freq_highpass)
                else:
                    err = "Unrecognized Filter Option. Showing unfiltered " + \
                          "data."
                    appendTextview(self.textviewStdErr, err)
                    filt.append(tr.data)
            appendTextview(self.textviewStdOut, msg)
            #make new plots
            for i, plot in enumerate(self.plts):
                plot.set_ydata(filt[i])
        else:
            #make new plots
            for i, plot in enumerate(self.plts):
                plot.set_ydata(self.streams[self.stPt][i].data)
            msg = "Unfiltered Traces"
            appendTextview(self.textviewStdOut, msg)
        # Update all subplots
        self.redraw()
    
    # Define the event that handles the setting of P- and S-wave picks
    def keypress(self, event):
        phase_type = self.comboboxPhaseType.get_active_text()
        keys = self.dictKeybindings
        dict = self.dicts[self.stPt]
        
        #######################################################################
        # Start of key events related to picking                              #
        #######################################################################
        # For some key events (picking events) we need information on the x/y
        # position of the cursor:
        if event.key in (keys['setPick'], keys['setPickError'],
                         keys['setMagMin'], keys['setMagMax']):
            # some keypress events only make sense inside our matplotlib axes
            if not event.inaxes in self.axs:
                return
            #We want to round from the picking location to
            #the time value of the nearest time sample:
            samp_rate = self.streams[self.stPt][0].stats.sampling_rate
            pickSample = event.xdata * samp_rate
            pickSample = round(pickSample)
            pickSample = pickSample / samp_rate
            # we need the position of the cursor location
            # in the seismogram array:
            xpos = pickSample * samp_rate

        if event.key == keys['setPick']:
            # some keypress events only make sense inside our matplotlib axes
            if not event.inaxes in self.axs:
                return
            if phase_type == 'P':
                self.delPLine()
                self.delPLabel()
                self.delPsynthLine()
                dict['P'] = pickSample
                self.drawPLine()
                self.drawPLabel()
                self.drawPsynthLine()
                self.drawPsynthLabel()
                #check if the new P pick lies outside of the Error Picks
                if 'PErr1' in dict and dict['P'] < dict['PErr1']:
                    self.delPErr1Line()
                    self.delPErr1()
                if 'PErr2' in dict and dict['P'] > dict['PErr2']:
                    self.delPErr2Line()
                    self.delPErr2()
                self.redraw()
                msg = "P Pick set at %.3f" % dict['P']
                appendTextview(self.textviewStdOut, msg)
                return
            elif phase_type == 'S':
                self.delSLine()
                self.delSLabel()
                self.delSsynthLine()
                dict['S'] = pickSample
                dict['Saxind'] = self.axs.index(event.inaxes)
                self.drawSLine()
                self.drawSLabel()
                self.drawSsynthLine()
                self.drawSsynthLabel()
                #check if the new S pick lies outside of the Error Picks
                if 'SErr1' in dict and dict['S'] < dict['SErr1']:
                    self.delSErr1Line()
                    self.delSErr1()
                if 'SErr2' in dict and dict['S'] > dict['SErr2']:
                    self.delSErr2Line()
                    self.delSErr2()
                self.redraw()
                msg = "S Pick set at %.3f" % dict['S']
                appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['setWeight0']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PWeight']=0
                self.drawPLabel()
                self.redraw()
                msg = "P Pick weight set to %i"%dict['PWeight']
                appendTextview(self.textviewStdOut, msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SWeight']=0
                self.drawSLabel()
                self.redraw()
                msg = "S Pick weight set to %i"%dict['SWeight']
                appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['setWeight1']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PWeight']=1
                msg = "P Pick weight set to %i"%dict['PWeight']
                appendTextview(self.textviewStdOut, msg)
                self.drawPLabel()
                self.redraw()
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SWeight']=1
                self.drawSLabel()
                self.redraw()
                msg = "S Pick weight set to %i"%dict['SWeight']
                appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['setWeight2']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PWeight']=2
                msg = "P Pick weight set to %i"%dict['PWeight']
                appendTextview(self.textviewStdOut, msg)
                self.drawPLabel()
                self.redraw()
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SWeight']=2
                self.drawSLabel()
                self.redraw()
                msg = "S Pick weight set to %i"%dict['SWeight']
                appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['setWeight3']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PWeight']=3
                msg = "P Pick weight set to %i"%dict['PWeight']
                appendTextview(self.textviewStdOut, msg)
                self.drawPLabel()
                self.redraw()
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SWeight']=3
                self.drawSLabel()
                self.redraw()
                msg = "S Pick weight set to %i"%dict['SWeight']
                appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['setPolUp']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PPol']='up'
                self.drawPLabel()
                self.redraw()
                msg = "P Pick polarity set to %s"%dict['PPol']
                appendTextview(self.textviewStdOut, msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SPol']='up'
                self.drawSLabel()
                self.redraw()
                msg = "S Pick polarity set to %s"%dict['SPol']
                appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['setPolPoorUp']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PPol']='poorup'
                self.drawPLabel()
                self.redraw()
                msg = "P Pick polarity set to %s"%dict['PPol']
                appendTextview(self.textviewStdOut, msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SPol']='poorup'
                self.drawSLabel()
                self.redraw()
                msg = "S Pick polarity set to %s"%dict['SPol']
                appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['setPolDown']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PPol']='down'
                self.drawPLabel()
                self.redraw()
                msg = "P Pick polarity set to %s"%dict['PPol']
                appendTextview(self.textviewStdOut, msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SPol']='down'
                self.drawSLabel()
                self.redraw()
                msg = "S Pick polarity set to %s"%dict['SPol']
                appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['setPolPoorDown']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PPol']='poordown'
                self.drawPLabel()
                self.redraw()
                msg = "P Pick polarity set to %s"%dict['PPol']
                appendTextview(self.textviewStdOut, msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SPol']='poordown'
                self.drawSLabel()
                self.redraw()
                msg = "S Pick polarity set to %s"%dict['SPol']
                appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['setOnsetImpulsive']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['POnset'] = 'impulsive'
                self.drawPLabel()
                self.redraw()
                msg = "P pick onset set to %s" % dict['POnset']
                appendTextview(self.textviewStdOut, msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SOnset'] = 'impulsive'
                self.drawSLabel()
                self.redraw()
                msg = "S pick onset set to %s" % dict['SOnset']
                appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['setOnsetEmergent']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['POnset'] = 'emergent'
                self.drawPLabel()
                self.redraw()
                msg = "P pick onset set to %s" % dict['POnset']
                appendTextview(self.textviewStdOut, msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SOnset'] = 'emergent'
                self.drawSLabel()
                self.redraw()
                msg = "S pick onset set to %s" % dict['SOnset']
                appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['delPick']:
            if phase_type == 'P':
                self.delPLine()
                self.delP()
                self.delPWeight()
                self.delPPol()
                self.delPOnset()
                self.delPLabel()
                self.delPErr1Line()
                self.delPErr1()
                self.delPErr2Line()
                self.delPErr2()
                self.redraw()
                return
            elif phase_type == 'S':
                self.delSLine()
                self.delS()
                self.delSWeight()
                self.delSPol()
                self.delSOnset()
                self.delSLabel()
                self.delSErr1Line()
                self.delSErr1()
                self.delSErr2Line()
                self.delSErr2()
                self.redraw()
                return

        if event.key == keys['setPickError']:
            # some keypress events only make sense inside our matplotlib axes
            if not event.inaxes in self.axs:
                return
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                # Set left Error Pick
                if pickSample < dict['P']:
                    self.delPErr1Line()
                    dict['PErr1'] = pickSample
                    self.drawPErr1Line()
                    self.redraw()
                    msg = "P Error Pick 1 set at %.3f" % dict['PErr1']
                    appendTextview(self.textviewStdOut, msg)
                # Set right Error Pick
                elif pickSample > dict['P']:
                    self.delPErr2Line()
                    dict['PErr2'] = pickSample
                    self.drawPErr2Line()
                    self.redraw()
                    msg = "P Error Pick 2 set at %.3f" % dict['PErr2']
                    appendTextview(self.textviewStdOut, msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                # Set left Error Pick
                if pickSample < dict['S']:
                    self.delSErr1Line()
                    dict['SErr1'] = pickSample
                    self.drawSErr1Line()
                    self.redraw()
                    msg = "S Error Pick 1 set at %.3f" % dict['SErr1']
                    appendTextview(self.textviewStdOut, msg)
                # Set right Error Pick
                elif pickSample > dict['S']:
                    self.delSErr2Line()
                    dict['SErr2'] = pickSample
                    self.drawSErr2Line()
                    self.redraw()
                    msg = "S Error Pick 2 set at %.3f" % dict['SErr2']
                    appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['setMagMin']:
            # some keypress events only make sense inside our matplotlib axes
            if not event.inaxes in self.axs:
                return
            if phase_type == 'Mag':
                if len(self.axs) != 3:
                    err = "Magnitude picking only supported with 3 axes."
                    appendTextview(self.textviewStdErr, err)
                    return
                if event.inaxes is self.axs[1]:
                    self.delMagMinCross1()
                    ydata = event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
                    cutoffSamples = xpos - self.magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
                    dict['MagMin1'] = np.min(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    # save time of magnitude minimum in seconds
                    tmp_magtime = cutoffSamples + np.argmin(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    tmp_magtime = tmp_magtime / samp_rate
                    dict['MagMin1T'] = tmp_magtime
                    #delete old MagMax Pick, if new MagMin Pick is higher
                    if 'MagMax1' in dict and dict['MagMin1'] > dict['MagMax1']:
                        self.delMagMaxCross1()
                        self.delMagMax1()
                    self.drawMagMinCross1()
                    self.redraw()
                    msg = "Minimum for magnitude estimation set: %s at %.3f" \
                            % (dict['MagMin1'], dict['MagMin1T'])
                    appendTextview(self.textviewStdOut, msg)
                elif event.inaxes is self.axs[2]:
                    self.delMagMinCross2()
                    ydata = event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
                    cutoffSamples = xpos - self.magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
                    dict['MagMin2'] = np.min(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    # save time of magnitude minimum in seconds
                    tmp_magtime = cutoffSamples + np.argmin(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    tmp_magtime = tmp_magtime / samp_rate
                    dict['MagMin2T'] = tmp_magtime
                    #delete old MagMax Pick, if new MagMin Pick is higher
                    if 'MagMax2' in dict and dict['MagMin2'] > dict['MagMax2']:
                        self.delMagMaxCross2()
                        self.delMagMax2()
                    self.drawMagMinCross2()
                    self.redraw()
                    msg = "Minimum for magnitude estimation set: %s at %.3f" \
                            % (dict['MagMin2'], dict['MagMin2T'])
                    appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['setMagMax']:
            # some keypress events only make sense inside our matplotlib axes
            if not event.inaxes in self.axs:
                return
            if phase_type == 'Mag':
                if len(self.axs) != 3:
                    err = "Magnitude picking only supported with 3 axes."
                    appendTextview(self.textviewStdErr, err)
                    return
                if event.inaxes is self.axs[1]:
                    self.delMagMaxCross1()
                    ydata = event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
                    cutoffSamples = xpos - self.magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
                    dict['MagMax1'] = np.max(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    # save time of magnitude maximum in seconds
                    tmp_magtime = cutoffSamples + np.argmax(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    tmp_magtime = tmp_magtime / samp_rate
                    dict['MagMax1T'] = tmp_magtime
                    #delete old MagMax Pick, if new MagMax Pick is higher
                    if 'MagMin1' in dict and dict['MagMin1'] > dict['MagMax1']:
                        self.delMagMinCross1()
                        self.delMagMin1()
                    self.drawMagMaxCross1()
                    self.redraw()
                    msg = "Maximum for magnitude estimation set: %s at %.3f" \
                            % (dict['MagMax1'], dict['MagMax1T'])
                    appendTextview(self.textviewStdOut, msg)
                elif event.inaxes is self.axs[2]:
                    self.delMagMaxCross2()
                    ydata = event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
                    cutoffSamples = xpos - self.magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
                    dict['MagMax2'] = np.max(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    # save time of magnitude maximum in seconds
                    tmp_magtime = cutoffSamples + np.argmax(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    tmp_magtime = tmp_magtime / samp_rate
                    dict['MagMax2T'] = tmp_magtime
                    #delete old MagMax Pick, if new MagMax Pick is higher
                    if 'MagMin2' in dict and dict['MagMin2'] > dict['MagMax2']:
                        self.delMagMinCross2()
                        self.delMagMin2()
                    self.drawMagMaxCross2()
                    self.redraw()
                    msg = "Maximum for magnitude estimation set: %s at %.3f" \
                            % (dict['MagMax2'], dict['MagMax2T'])
                    appendTextview(self.textviewStdOut, msg)
                return

        if event.key == keys['delMagMinMax']:
            if phase_type == 'Mag':
                if event.inaxes is self.axs[1]:
                    self.delMagMaxCross1()
                    self.delMagMinCross1()
                    self.delMagMin1()
                    self.delMagMax1()
                elif event.inaxes is self.axs[2]:
                    self.delMagMaxCross2()
                    self.delMagMinCross2()
                    self.delMagMin2()
                    self.delMagMax2()
                else:
                    return
                self.redraw()
                return
        #######################################################################
        # End of key events related to picking                                #
        #######################################################################
        
        if event.key == keys['switchWheelZoom']:
            self.flagWheelZoom = not self.flagWheelZoom
            if self.flagWheelZoom:
                msg = "Mouse wheel zooming activated"
                appendTextview(self.textviewStdOut, msg)
            else:
                msg = "Mouse wheel zooming deactivated"
                appendTextview(self.textviewStdOut, msg)
            return

        if event.key == keys['switchPan']:
            self.toolbar.pan()
            self.canv.widgetlock.release(self.toolbar)
            self.redraw()
            msg = "Switching pan mode"
            appendTextview(self.textviewStdOut, msg)
            return
        
        # iterate the phase type combobox
        if event.key == keys['switchPhase']:
            combobox = self.comboboxPhaseType
            phase_count = len(combobox.get_model())
            phase_next = (combobox.get_active() + 1) % phase_count
            combobox.set_active(phase_next)
            msg = "Switching Phase button"
            appendTextview(self.textviewStdOut, msg)
            return
            
        if event.key == keys['prevStream']:
            #would be nice if the button would show the click, but it is
            #too brief to be seen
            #self.buttonPreviousStream.set_state(True)
            self.buttonPreviousStream.clicked()
            #self.buttonPreviousStream.set_state(False)
            return

        if event.key == keys['nextStream']:
            #would be nice if the button would show the click, but it is
            #too brief to be seen
            #self.buttonNextStream.set_state(True)
            self.buttonNextStream.clicked()
            #self.buttonNextStream.set_state(False)
            return
            
    # Define zooming for the mouse scroll wheel
    def scroll(self, event):
        if not self.flagWheelZoom:
            return
        # Calculate and set new axes boundaries from old ones
        (left, right) = self.axs[0].get_xbound()
        # Zoom in on scroll-up
        if event.button == 'up':
            left += (event.xdata - left) / 2
            right -= (right - event.xdata) / 2
        # Zoom out on scroll-down
        elif event.button == 'down':
            left -= (event.xdata - left) / 2
            right += (right - event.xdata) / 2
        self.axs[0].set_xbound(lower=left, upper=right)
        self.redraw()
    
    # Define zoom reset for the mouse button 2 (always scroll wheel!?)
    def buttonpress(self, event):
        # set widgetlock when pressing mouse buttons and dont show cursor
        # cursor should not be plotted when making a zoom selection etc.
        if event.button == 1 or event.button == 3:
            self.multicursor.visible = False
            self.canv.widgetlock(self.toolbar)
        # show traces from start to end
        # (Use Z trace limits as boundaries)
        elif event.button == 2:
            self.axs[0].set_xbound(lower=self.xMin, upper=self.xMax)
            self.axs[0].set_ybound(lower=self.yMin, upper=self.yMax)
            # Update all subplots
            self.redraw()
            msg = "Resetting axes"
            appendTextview(self.textviewStdOut, msg)
    
    def buttonrelease(self, event):
        # release widgetlock when releasing mouse buttons
        if event.button == 1 or event.button == 3:
            self.multicursor.visible = True
            self.canv.widgetlock.release(self.toolbar)
    
    #lookup multicursor source: http://matplotlib.sourcearchive.com/documentation/0.98.1/widgets_8py-source.html
    def multicursorReinit(self):
        self.canv.mpl_disconnect(self.multicursor.id1)
        self.canv.mpl_disconnect(self.multicursor.id2)
        self.multicursor.__init__(self.canv, self.axs, useblit=True,
                                  color='black', linewidth=1, ls='dotted')
        self.updateMulticursorColor()

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

    #def updateComboboxPhaseTypeColor(self):
    #    phase_name = self.comboboxPhaseType.get_active_text()
    #    props = self.comboboxPhaseType.get_cells()[0].props
    #    color = gtk.gdk.color_parse(self.dictPhaseColors[phase_name])
    #    props.cell_background_gdk = color

    def updateStreamLabels(self):
        self.labelStreamNumber.set_markup("<tt>%02i/%02i</tt>" % \
                (self.stPt + 1, self.stNum))
        self.labelStreamName.set_markup("<tt><b>%s</b></tt>" % \
                self.dicts[self.stPt]['Station'])
    
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
            phase = phase.split()
            phStat = phase[0]
            phType = phase[1]
            phUTCTime = UTCDateTime(int(phase[2]), int(phase[3]),
                                    int(phase[4]), int(phase[5]),
                                    int(phase[6]), float(phase[7]))
            phResid = float(phase[8])
            phUTCTime += phResid
            for i in range(len(self.streams)):
                # check for matching station names
                if not phStat == self.streams[i][0].stats.station.strip():
                    continue
                else:
                    # check if synthetic pick is within time range of stream
                    if (phUTCTime > self.streams[i][0].stats.endtime or
                        phUTCTime < self.streams[i][0].stats.starttime):
                        msg = "Synthetic pick outside timespan."
                        warnings.warn(msg)
                        continue
                    else:
                        # phSeconds is the time in seconds after the stream-
                        # starttime at which the time of the synthetic phase
                        # is located
                        phSeconds = phUTCTime - self.streams[i][0].stats.starttime
                        if phType == 'P':
                            self.dicts[i]['Psynth'] = phSeconds
                            self.dicts[i]['Pres'] = phResid
                        elif phType == 'S':
                            self.dicts[i]['Ssynth'] = phSeconds
                            self.dicts[i]['Sres'] = phResid
        self.drawPsynthLine()
        self.drawPsynthLabel()
        self.drawSsynthLine()
        self.drawSsynthLabel()
        self.redraw()

    def do3dLoc(self):
        self.setXMLEventID()
        subprocess.call(self.threeDlocPreCall, shell = True)
        f = open(self.threeDlocInfile, 'w')
        network = "BW"
        fmt = "%04s  %s        %s %5.3f -999.0 0.000 -999. 0.000 T__DR_ %9.6f %9.6f %8.6f\n"
        self.coords = []
        for i in range(len(self.streams)):
            lon = self.dicts[i]['StaLon']
            lat = self.dicts[i]['StaLat']
            ele = self.dicts[i]['StaEle']
            self.coords.append([lon, lat])
            if self.dicts[i].has_key('P'):
                t = self.streams[i][0].stats.starttime
                t += self.dicts[i]['P']
                date = t.strftime("%Y %m %d %H %M %S")
                date += ".%03d" % (t.microsecond / 1e3 + 0.5)
                delta = self.dicts[i]['PErr2'] - self.dicts[i]['PErr1']
                f.write(fmt % (self.dicts[i]['Station'], 'P', date, delta,
                               lon, lat, ele / 1e3))
            if self.dicts[i].has_key('S'):
                t = self.streams[i][0].stats.starttime
                t += self.dicts[i]['S']
                date = t.strftime("%Y %m %d %H %M %S")
                date += ".%03d" % (t.microsecond / 1e3 + 0.5)
                delta = self.dicts[i]['SErr2'] - self.dicts[i]['SErr1']
                f.write(fmt % (self.dicts[i]['Station'], 'S', date, delta,
                               lon, lat, ele / 1e3))
        f.close()
        msg = 'Phases for 3Dloc:'
        appendTextview(self.textviewStdOut, msg)
        self.catFile(self.threeDlocInfile)
        subprocess.call(self.threeDlocCall, shell = True)
        msg = '--> 3dloc finished'
        appendTextview(self.textviewStdOut, msg)
        self.catFile(self.threeDlocOutfile)

    def doFocmec(self):
        f = open(self.focmecPhasefile, 'w')
        f.write("\n") #first line is ignored!
        #Fortran style! 1: Station 2: Azimuth 3: Incident 4: Polarity
        #fmt = "ONTN  349.00   96.00C"
        fmt = "%4s  %6.2f  %6.2f%1s\n"
        count = 0
        for d in self.dicts:
            if 'PAzim' not in d or 'PInci' not in d or 'PPol' not in d:
                continue
            sta = d['Station'][:4] #focmec has only 4 chars
            azim = d['PAzim']
            inci = d['PInci']
            if d['PPol'] == 'up':
                pol = 'U'
            elif d['PPol'] == 'poorup':
                pol = '+'
            elif d['PPol'] == 'down':
                pol = 'D'
            elif d['PPol'] == 'poordown':
                pol = '-'
            else:
                continue
            count += 1
            f.write(fmt % (sta, azim, inci, pol))
        f.close()
        msg = 'Phases for focmec: %i' % count
        appendTextview(self.textviewStdOut, msg)
        self.catFile(self.focmecPhasefile)
        exitcode = subprocess.call(self.focmecCall, shell=True)
        if exitcode == 1:
            err = "Error: focmec did not find a suitable solution!"
            appendTextview(self.textviewStdErr, err)
            return
        msg = '--> focmec finished'
        appendTextview(self.textviewStdOut, msg)
        lines = open(self.focmecSummary, "r").readlines()
        msg = '%i suitable solutions found:' % len(lines)
        appendTextview(self.textviewStdOut, msg)
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
            msg = "Dip: %6.2f  Strike: %6.2f  Rake: %6.2f  Errors: %i/%i" % \
                    (tempdict['Dip'], tempdict['Strike'], tempdict['Rake'],
                     tempdict['Errors'], tempdict['Station Polarity Count'])
            appendTextview(self.textviewStdOut, msg)
            self.focMechList.append(tempdict)
        self.focMechCount = len(self.focMechList)
        self.focMechCurrent = 0
        msg = "selecting Focal Mechanism No.  1 of %2i:" % self.focMechCount
        appendTextview(self.textviewStdOut, msg)
        self.dictFocalMechanism = self.focMechList[0]
        msg = "Dip: %6.2f  Strike: %6.2f  Rake: %6.2f  Errors: %i/%i" % \
                (self.dictFocalMechanism['Dip'],
                 self.dictFocalMechanism['Strike'],
                 self.dictFocalMechanism['Rake'],
                 self.dictFocalMechanism['Errors'],
                 self.dictFocalMechanism['Station Polarity Count'])
        appendTextview(self.textviewStdOut, msg)

    def nextFocMec(self):
        if not self.focMechCount:
            return
        self.focMechCurrent = (self.focMechCurrent + 1) % self.focMechCount
        self.dictFocalMechanism = self.focMechList[self.focMechCurrent]
        msg = "selecting Focal Mechanism No. %2i of %2i:" % \
                (self.focMechCurrent + 1, self.focMechCount)
        appendTextview(self.textviewStdOut, msg)
        msg = "Dip: %6.2f  Strike: %6.2f  Rake: %6.2f  Errors: %i%i" % \
                (self.dictFocalMechanism['Dip'],
                 self.dictFocalMechanism['Strike'],
                 self.dictFocalMechanism['Rake'],
                 self.dictFocalMechanism['Errors'],
                 self.dictFocalMechanism['Station Polarity Count'])
        appendTextview(self.textviewStdOut, msg)

    def showFocMec(self):
        if self.dictFocalMechanism == {}:
            err = "Error: No focal mechanism data!"
            appendTextview(self.textviewStdErr, err)
            return
        # make up the figure:
        fig = plt.figure(1002, figsize=(2, 2))
        fig.clear()
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        
        # plot the selected solution
        Beachball([self.dictFocalMechanism['Strike'],
                   self.dictFocalMechanism['Dip'],
                   self.dictFocalMechanism['Rake']], fig=fig)
        # plot the alternative solutions
        if self.focMechList != []:
            for d in self.focMechList:
                Beachball([d['Strike'], d['Dip'], d['Rake']],
                          nofill=True, fig=fig, edgecolor='k',
                          linewidth=1., alpha=0.3)
        try:
            fig.suptitle("Dip: %6.2f  Strike: %6.2f  Rake: %6.2f\n" % \
                    (self.dictFocalMechanism['Dip'],
                     self.dictFocalMechanism['Strike'],
                     self.dictFocalMechanism['Rake']) + \
                     "Errors: %i/%i" % \
                     (self.dictFocalMechanism['Errors'],
                      self.dictFocalMechanism['Station Polarity Count']),
                     fontsize=10)
        except:
            fig.suptitle("Dip: %6.2f  Strike: %6.2f  Rake: %6.2f\n" % \
                    (self.dictFocalMechanism['Dip'],
                     self.dictFocalMechanism['Strike'],
                     self.dictFocalMechanism['Rake']) + \
                     "Used Polarities: %i" % \
                     self.dictFocalMechanism['Station Polarity Count'],
                     fontsize=10)
        fig.canvas.set_window_title("Focal Mechanism (%i of %i)" % \
                (self.focMechCurrent + 1,
                 self.focMechCount))
        fig.subplots_adjust(top=0.88) # make room for suptitle
        # values 0.02 and 0.96 fit best over the outer edges of beachball
        #ax = fig.add_axes([0.00, 0.02, 1.00, 0.96], polar=True)
        ax = fig.add_axes([0.00, 0.02, 1.00, 0.84], polar=True)
        ax.set_axis_off()
        for d in self.dicts:
            if 'PAzim' in d and 'PInci' in d and 'PPol' in d:
                if d['PPol'] == "up":
                    color = "black"
                elif d['PPol'] == "poorup":
                    color = "darkgrey"
                elif d['PPol'] == "poordown":
                    color = "lightgrey"
                elif d['PPol'] == "down":
                    color = "white"
                else:
                    continue
                # southern hemisphere projection
                if d['PInci'] > 90:
                    inci = 180. - d['PInci']
                    #azim = -180. + d['PAzim']
                else:
                    inci = d['PInci']
                azim = d['PAzim']
                #we have to hack the azimuth because of the polar plot
                #axes orientation
                plotazim = (np.pi / 2.) - ((azim / 180.) * np.pi)
                ax.scatter([plotazim], [inci], facecolor=color)
                ax.text(plotazim, inci, " " + d['Station'],
                        fontsize=10, va="top")
        #this fits the 90 degree incident value to the beachball edge best
        ax.set_ylim([0., 91])
        fig.canvas.draw()

    def doHyp2000(self):
        """
        Writes input files for hyp2000 and starts the hyp2000 program via a
        system call.
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
        self.setXMLEventID()
        subprocess.call(self.hyp2000PreCall, shell = True)
        f = open(self.hyp2000Phasefile, 'w')
        f2 = open(self.hyp2000Stationsfile, 'w')
        network = "BW"
        #fmt = "RWMOIP?0 091229124412.22       13.99IS?0"
        fmtP = "%4s%1sP%1s%1i %15s"
        fmtS = "%12s%1sS%1s%1i\n"
        #fmt2 = "  BGLD4739.14N01300.75E 930"
        fmt2 = "%6s%02i%05.2fN%03i%05.2fE%4i\n"
        #self.coords = []
        for i in range(len(self.streams)):
            sta = self.dicts[i]['Station']
            lon = self.dicts[i]['StaLon']
            lon_deg = int(lon)
            lon_min = (lon - lon_deg) * 60.
            lat = self.dicts[i]['StaLat']
            lat_deg = int(lat)
            lat_min = (lat - lat_deg) * 60.
            ele = self.dicts[i]['StaEle'] * 1000
            f2.write(fmt2 % (sta, lat_deg, lat_min, lon_deg, lon_min, ele))
            #self.coords.append([lon, lat])
            if not self.dicts[i].has_key('P') and not self.dicts[i].has_key('S'):
                continue
            if self.dicts[i].has_key('P'):
                t = self.streams[i][0].stats.starttime
                t += self.dicts[i]['P']
                date = t.strftime("%y%m%d%H%M%S")
                date += ".%02d" % (t.microsecond / 1e4 + 0.5)
                if self.dicts[i].has_key('POnset'):
                    if self.dicts[i]['POnset'] == 'impulsive':
                        onset = 'I'
                    elif self.dicts[i]['POnset'] == 'emergent':
                        onset = 'E'
                    else: #XXX check for other names correctly!!!
                        onset = '?'
                else:
                    onset = '?'
                if self.dicts[i].has_key('PPol'):
                    if self.dicts[i]['PPol'] == "up" or \
                       self.dicts[i]['PPol'] == "poorup":
                        polarity = "U"
                    elif self.dicts[i]['PPol'] == "down" or \
                         self.dicts[i]['PPol'] == "poordown":
                        polarity = "D"
                    else: #XXX check for other names correctly!!!
                        polarity = "D"
                else:
                    polarity = "?"
                if self.dicts[i].has_key('PWeight'):
                    weight = int(self.dicts[i]['PWeight'])
                else:
                    weight = 0
                f.write(fmtP % (sta, onset, polarity, weight, date))
            if self.dicts[i].has_key('S'):
                if not self.dicts[i].has_key('P'):
                    err = "Warning: Trying to print a Hypo2000 phase file " + \
                          "with an S phase without P phase.\n" + \
                          "This case might not be covered correctly and " + \
                          "could screw our file up!"
                    appendTextview(self.textviewStdErr, err)
                t2 = self.streams[i][0].stats.starttime
                t2 += self.dicts[i]['S']
                # if the S time's absolute minute is higher than that of the
                # P pick, we have to add 60 to the S second count for the
                # hypo 2000 output file
                # +60 %60 is necessary if t.min = 57, t2.min = 2 e.g.
                mindiff = (t2.minute - t.minute + 60) % 60
                abs_sec = t2.second + (mindiff * 60)
                if abs_sec > 99:
                    msg = "S phase seconds are greater than 99 which is " + \
                          "not covered by the hypo phase file format! " + \
                          "Omitting S phase of station %s!!!" % sta
                    warnings.warn(msg)
                    f.write("\n")
                    continue
                date2 = str(abs_sec)
                date2 += ".%02d" % (t2.microsecond / 1e4 + 0.5)
                if self.dicts[i].has_key('SOnset'):
                    if self.dicts[i]['SOnset'] == 'impulsive':
                        onset2 = 'I'
                    elif self.dicts[i]['SOnset'] == 'emergent':
                        onset2 = 'E'
                    else: #XXX check for other names correctly!!!
                        onset2 = '?'
                else:
                    onset2 = '?'
                if self.dicts[i].has_key('SPol'):
                    if self.dicts[i]['SPol'] == "up" or \
                       self.dicts[i]['SPol'] == "poorup":
                        polarity2 = "U"
                    elif self.dicts[i]['SPol'] == "down" or \
                         self.dicts[i]['SPol'] == "poordown":
                        polarity2 = "D"
                    else: #XXX check for other names correctly!!!
                        polarity2 = "D"
                else:
                    polarity2 = "?"
                if self.dicts[i].has_key('SWeight'):
                    weight2 = int(self.dicts[i]['SWeight'])
                else:
                    weight2 = 0
                f.write(fmtS % (date2, onset2, polarity2, weight2))
            else:
                f.write("\n")
        f.close()
        f2.close()
        msg = 'Phases for Hypo2000:'
        appendTextview(self.textviewStdOut, msg)
        self.catFile(self.hyp2000Phasefile)
        msg = 'Stations for Hypo2000:'
        appendTextview(self.textviewStdOut, msg)
        self.catFile(self.hyp2000Stationsfile)
        subprocess.call(self.hyp2000Call, shell = True)
        msg = '--> hyp2000 finished'
        appendTextview(self.textviewStdOut, msg)
        self.catFile(self.hyp2000Summary)

    def catFile(self, file):
        lines = open(file).readlines()
        msg = ""
        for line in lines:
            msg += line
        appendTextview(self.textviewStdOut, msg)

    def loadHyp2000Data(self):
        #self.load3dlocSyntheticPhases()
        lines = open(self.hyp2000Summary).readlines()
        if lines == []:
            err = "Error: Hypo2000 output file (%s) does not exist!" % \
                    self.hyp2000Summary
            appendTextview(self.textviewStdErr, err)
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
            appendTextview(self.textviewStdErr, err)
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
        self.dictOrigin['Longitude'] = lon
        self.dictOrigin['Latitude'] = lat
        self.dictOrigin['Depth'] = depth
        self.dictOrigin['Longitude Error'] = errXY
        self.dictOrigin['Latitude Error'] = errXY
        self.dictOrigin['Depth Error'] = errZ
        self.dictOrigin['Standarderror'] = rms #XXX stimmt diese Zuordnung!!!?!
        self.dictOrigin['Azimuthal Gap'] = gap
        self.dictOrigin['Depth Type'] = "from location program"
        self.dictOrigin['Earth Model'] = model
        self.dictOrigin['Time'] = time
        
        # goto station and phases info lines
        while True:
            try:
                line = lines.pop(0)
            except:
                break
            if line.startswith(" STA NET COM L CR DIST AZM"):
                break
        
        self.dictOrigin['used P Count'] = 0
        self.dictOrigin['used S Count'] = 0
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
            for i in range(len(self.streams)):
                if station.strip() != self.dicts[i]['Station']:
                    continue
                else:
                    streamnum = i
                    break
            if streamnum == None:
                message = "Did not find matching stream for pick data " + \
                          "with station id: \"%s\"" % station.strip()
                warnings.warn(message)
                continue
            
            # assign synthetic phase info
            if type == "P":
                self.dictOrigin['used P Count'] += 1
                self.dicts[streamnum]['Psynth'] = res + \
                                                  self.dicts[streamnum]['P']
                self.dicts[streamnum]['Pres'] = res
                self.dicts[streamnum]['PAzim'] = azimuth
                self.dicts[streamnum]['PInci'] = incident
                if onset:
                    self.dicts[streamnum]['POnset'] = onset
                if polarity:
                    self.dicts[streamnum]['PPol'] = polarity
                #XXX how to set the weight???
                # we use weights 0,1,2,3 but hypo2000 outputs floats...
                self.dicts[streamnum]['PsynthWeight'] = weight
                #self.dicts[streamnum]['PResInfo'] = '\n\n %+0.3fs' % res
                #if self.dicts[streamnum].has_key('PPol'):
                #    self.dicts[streamnum]['PResInfo'] += '  %s' % \
                #            self.dicts[streamnum]['PPol']
            elif type == "S":
                self.dictOrigin['used S Count'] += 1
                self.dicts[streamnum]['Ssynth'] = res + \
                                                  self.dicts[streamnum]['S']
                self.dicts[streamnum]['Sres'] = res
                self.dicts[streamnum]['SAzim'] = azimuth
                self.dicts[streamnum]['SInci'] = incident
                if onset:
                    self.dicts[streamnum]['SOnset'] = onset
                if polarity:
                    self.dicts[streamnum]['SPol'] = polarity
                #XXX how to set the weight???
                # we use weights 0,1,2,3 but hypo2000 outputs floats...
                self.dicts[streamnum]['SsynthWeight'] = weight
                #self.dicts[streamnum]['SResInfo'] = '\n\n\n %+0.3fs' % res
                #if self.dicts[streamnum].has_key('SPol'):
                #    self.dicts[streamnum]['SResInfo'] += '  %s' % \
                #            self.dicts[streamnum]['SPol']
        self.dictOrigin['used Station Count'] = len(self.dicts)
        for st in self.dicts:
            if not (st.has_key('Psynth') or st.has_key('Ssynth')):
                self.dictOrigin['used Station Count'] -= 1

    def load3dlocData(self):
        #self.load3dlocSyntheticPhases()
        event = open(self.threeDlocOutfile).readline().split()
        self.dictOrigin['Longitude'] = float(event[8])
        self.dictOrigin['Latitude'] = float(event[9])
        self.dictOrigin['Depth'] = float(event[10])
        self.dictOrigin['Longitude Error'] = float(event[11])
        self.dictOrigin['Latitude Error'] = float(event[12])
        self.dictOrigin['Depth Error'] = float(event[13])
        self.dictOrigin['Standarderror'] = float(event[14])
        self.dictOrigin['Azimuthal Gap'] = float(event[15])
        self.dictOrigin['Depth Type'] = "from location program"
        self.dictOrigin['Earth Model'] = "STAUFEN"
        self.dictOrigin['Time'] = UTCDateTime(int(event[2]), int(event[3]),
                                              int(event[4]), int(event[5]),
                                              int(event[6]), float(event[7]))
        self.dictOrigin['used P Count'] = 0
        self.dictOrigin['used S Count'] = 0
        lines = open(self.threeDlocInfile).readlines()
        for line in lines:
            pick = line.split()
            for i in range(len(self.streams)):
                if pick[0].strip() == self.streams[i][0].stats.station.strip():
                    if pick[1] == 'P':
                        self.dictOrigin['used P Count'] += 1
                    elif pick[1] == 'S':
                        self.dictOrigin['used S Count'] += 1
                    break
        lines = open(self.threeDlocOutfile).readlines()
        for line in lines[1:]:
            pick = line.split()
            for i in range(len(self.streams)):
                if pick[0].strip() == self.streams[i][0].stats.station.strip():
                    if pick[1] == 'P':
                        self.dicts[i]['PAzim'] = float(pick[9])
                        self.dicts[i]['PInci'] = float(pick[10])
                        #self.dicts[i]['PResInfo'] = '\n\n %+0.3fs' % float(pick[8])
                        #if self.dicts[i].has_key('PPol'):
                        #    self.dicts[i]['PResInfo'] += '  %s' % self.dicts[i]['PPol']
                            
                    elif pick[1] == 'S':
                        self.dicts[i]['SAzim'] = float(pick[9])
                        self.dicts[i]['SInci'] = float(pick[10])
                        #self.dicts[i]['SResInfo'] = '\n\n\n %+0.3fs' % float(pick[8])
                        #if self.dicts[i].has_key('SPol'):
                        #    self.dicts[i]['SResInfo'] += '  %s' % self.dicts[i]['SPol']
                    break
        self.dictOrigin['used Station Count'] = len(self.dicts)
        for st in self.dicts:
            if not (st.has_key('Psynth') or st.has_key('Ssynth')):
                self.dictOrigin['used Station Count'] -= 1
    
    def updateNetworkMag(self):
        msg = "updating network magnitude..."
        appendTextview(self.textviewStdOut, msg)
        self.dictMagnitude['Station Count'] = 0
        self.dictMagnitude['Magnitude'] = 0
        staMags = []
        for i in range(len(self.streams)):
            if self.dicts[i]['MagUse'] and self.dicts[i].has_key('Mag'):
                msg = "%s: %.1f" % (self.dicts[i]['Station'],
                                    self.dicts[i]['Mag'])
                appendTextview(self.textviewStdOut, msg)
                self.dictMagnitude['Station Count'] += 1
                self.dictMagnitude['Magnitude'] += self.dicts[i]['Mag']
                staMags.append(self.dicts[i]['Mag'])
        if self.dictMagnitude['Station Count'] == 0:
            self.dictMagnitude['Magnitude'] = np.nan
            self.dictMagnitude['Uncertainty'] = np.nan
        else:
            self.dictMagnitude['Magnitude'] /= self.dictMagnitude['Station Count']
            self.dictMagnitude['Uncertainty'] = np.var(staMags)
        msg = "new network magnitude: %.2f (Variance: %.2f)" % \
                (self.dictMagnitude['Magnitude'],
                 self.dictMagnitude['Uncertainty'])
        appendTextview(self.textviewStdOut, msg)
        self.netMagLabel = '\n\n\n\n  %.2f (Var: %.2f)' % (self.dictMagnitude['Magnitude'], self.dictMagnitude['Uncertainty'])
        try:
            self.netMagText.set_text(self.netMagLabel)
        except:
            pass
    
    def calculateEpiHypoDists(self):
        if not 'Longitude' in self.dictOrigin or \
           not 'Latitude' in self.dictOrigin:
            err = "Error: No coordinates for origin!"
            appendTextview(self.textviewStdErr, err)
        epidists = []
        for i in range(len(self.streams)):
            x, y = utlGeoKm(self.dictOrigin['Longitude'], self.dictOrigin['Latitude'],
                            self.dicts[i]['StaLon'], self.dicts[i]['StaLat'])
            z = abs(self.dicts[i]['StaEle'] - self.dictOrigin['Depth'])
            self.dicts[i]['distX'] = x
            self.dicts[i]['distY'] = y
            self.dicts[i]['distZ'] = z
            self.dicts[i]['distEpi'] = np.sqrt(x**2 + y**2)
            # Median and Max/Min of epicentral distances should only be used
            # for stations with a pick that goes into the location.
            # The epicentral distance of all other stations may be needed for
            # magnitude estimation nonetheless.
            if self.dicts[i].has_key('Psynth') or self.dicts[i].has_key('Ssynth'):
                epidists.append(self.dicts[i]['distEpi'])
            self.dicts[i]['distHypo'] = np.sqrt(x**2 + y**2 + z**2)
        self.dictOrigin['Maximum Distance'] = max(epidists)
        self.dictOrigin['Minimum Distance'] = min(epidists)
        self.dictOrigin['Median Distance'] = np.median(epidists)

    def calculateStationMagnitudes(self):
        for i in range(len(self.streams)):
            if (self.dicts[i].has_key('MagMin1') and
                self.dicts[i].has_key('MagMin2') and
                self.dicts[i].has_key('MagMax1') and
                self.dicts[i].has_key('MagMax2')):
                
                amp = self.dicts[i]['MagMax1'] - self.dicts[i]['MagMin1']
                timedelta = abs(self.dicts[i]['MagMax1T'] - self.dicts[i]['MagMin1T'])
                mag = estimateMagnitude(self.dicts[i]['pazN'], amp, timedelta,
                                        self.dicts[i]['distHypo'])
                amp = self.dicts[i]['MagMax2'] - self.dicts[i]['MagMin2']
                timedelta = abs(self.dicts[i]['MagMax2T'] - self.dicts[i]['MagMin2T'])
                mag += estimateMagnitude(self.dicts[i]['pazE'], amp, timedelta,
                                         self.dicts[i]['distHypo'])
                mag /= 2.
                self.dicts[i]['Mag'] = mag
                self.dicts[i]['MagChannel'] = '%s,%s' % (self.streams[i][1].stats.channel, self.streams[i][2].stats.channel)
                msg = 'calculated new magnitude for %s: %0.2f (channels: %s)' \
                      % (self.dicts[i]['Station'], self.dicts[i]['Mag'],
                         self.dicts[i]['MagChannel'])
                appendTextview(self.textviewStdOut, msg)
            
            elif (self.dicts[i].has_key('MagMin1') and
                  self.dicts[i].has_key('MagMax1')):
                amp = self.dicts[i]['MagMax1'] - self.dicts[i]['MagMin1']
                timedelta = abs(self.dicts[i]['MagMax1T'] - self.dicts[i]['MagMin1T'])
                mag = estimateMagnitude(self.dicts[i]['pazN'], amp, timedelta,
                                        self.dicts[i]['distHypo'])
                self.dicts[i]['Mag'] = mag
                self.dicts[i]['MagChannel'] = '%s' % self.streams[i][1].stats.channel
                msg = 'calculated new magnitude for %s: %0.2f (channels: %s)' \
                      % (self.dicts[i]['Station'], self.dicts[i]['Mag'],
                         self.dicts[i]['MagChannel'])
                appendTextview(self.textviewStdOut, msg)
            
            elif (self.dicts[i].has_key('MagMin2') and
                  self.dicts[i].has_key('MagMax2')):
                amp = self.dicts[i]['MagMax2'] - self.dicts[i]['MagMin2']
                timedelta = abs(self.dicts[i]['MagMax2T'] - self.dicts[i]['MagMin2T'])
                mag = estimateMagnitude(self.dicts[i]['pazE'], amp, timedelta,
                                        self.dicts[i]['distHypo'])
                self.dicts[i]['Mag'] = mag
                self.dicts[i]['MagChannel'] = '%s' % self.streams[i][2].stats.channel
                msg = 'calculated new magnitude for %s: %0.2f (channels: %s)' \
                      % (self.dicts[i]['Station'], self.dicts[i]['Mag'],
                         self.dicts[i]['MagChannel'])
                appendTextview(self.textviewStdOut, msg)
    
    #see http://www.scipy.org/Cookbook/LinearRegression for alternative routine
    def showWadati(self):
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
            appendTextview(self.textviewStdErr, err)
            return
        pTimes = []
        spTimes = []
        stations = []
        for i in range(len(self.dicts)):
            if self.dicts[i].has_key('P') and self.dicts[i].has_key('S'):
                p = self.streams[i][0].stats.starttime
                p += self.dicts[i]['P']
                p = "%.3f" % p.getTimeStamp()
                p = float(p[-7:])
                pTimes.append(p)
                sp = self.dicts[i]['S'] - self.dicts[i]['P']
                spTimes.append(sp)
                stations.append(self.dicts[i]['Station'])
            else:
                continue
        if len(pTimes) < 2:
            err = "Error: Less than 2 P-S Pairs!"
            appendTextview(self.textviewStdErr, err)
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
        fig = plt.figure(1001)
        fig.canvas.set_window_title("Wadati Diagram")
        ax = fig.add_subplot(111)
        ax.scatter(pTimes, spTimes)
        for i in range(len(stations)):
            ax.text(pTimes[i], spTimes[i], stations[i], va = "top")
        ax.plot([x0, x1], [y0, y1])
        ax.axhline(0, color = "blue", ls = ":")
        # origin time estimated by wadati plot
        ax.axvline(x0, color = "blue", ls = ":", label = "origin time from wadati diagram")
        # origin time from event location
        if self.dictOrigin.has_key('Time'):
            otime = "%.3f" % self.dictOrigin['Time'].getTimeStamp()
            otime = float(otime[-7:])
            ax.axvline(otime, color = "red", ls = ":", label = "origin time from event location")
        ax.text(0.1, 0.7, "Vp/Vs: %.2f\nSum of squared residuals: %.3f" % (vpvs, ressqrsum), transform = ax.transAxes)
        ax.text(0.1, 0.1, "Origin time from event location", color = "red", transform = ax.transAxes)
        #ax.axis("auto")
        ax.set_xlim(min(x0 - 1, otime - 1), max(pTimes) + 1)
        ax.set_ylim(-1, max(spTimes) + 1)
        ax.set_xlabel("absolute P times (julian seconds, truncated)")
        ax.set_xlabel("P-S times (seconds)")
        fig.canvas.draw()
        plt.show()

    def showEventMap(self):
        if self.dictOrigin == {}:
            err = "Error: No hypocenter data!"
            appendTextview(self.textviewStdErr, err)
            return
        self.figEventMap = plt.figure(1000)
        self.figEventMap.canvas.set_window_title("Event Map")
        self.axEventMap = self.figEventMap.add_subplot(111)
        self.axEventMap.scatter([self.dictOrigin['Longitude']], [self.dictOrigin['Latitude']],
                             30, color = 'red', marker = 'o')
        errLon, errLat = utlLonLat(self.dictOrigin['Longitude'], self.dictOrigin['Latitude'],
                               self.dictOrigin['Longitude Error'], self.dictOrigin['Latitude Error'])
        errLon -= self.dictOrigin['Longitude']
        errLat -= self.dictOrigin['Latitude']
        self.axEventMap.text(self.dictOrigin['Longitude'],
                             self.dictOrigin['Latitude'],
                             ' %7.3f +/- %0.2fkm\n' % \
                             (self.dictOrigin['Longitude'],
                              self.dictOrigin['Longitude Error']) + \
                             ' %7.3f +/- %0.2fkm\n' % \
                             (self.dictOrigin['Latitude'],
                              self.dictOrigin['Latitude Error']) + \
                             '  %.1fkm +/- %.1fkm' % \
                             (self.dictOrigin['Depth'],
                              self.dictOrigin['Depth Error']),
                             va = 'top', family = 'monospace')
        try:
            self.netMagLabel = '\n\n\n\n  %.2f (Var: %.2f)' % (self.dictMagnitude['Magnitude'], self.dictMagnitude['Uncertainty'])
            self.netMagText = self.axEventMap.text(self.dictOrigin['Longitude'], self.dictOrigin['Latitude'],
                              self.netMagLabel,
                              va = 'top',
                              color = 'green',
                              family = 'monospace')
        except:
            pass
        errorell = Ellipse(xy = [self.dictOrigin['Longitude'], self.dictOrigin['Latitude']],
                      width = errLon, height = errLat, angle = 0, fill = False)
        self.axEventMap.add_artist(errorell)
        self.scatterMagIndices = []
        self.scatterMagLon = []
        self.scatterMagLat = []
        for i in range(len(self.streams)):
            # determine which stations are used in location
            if self.dicts[i].has_key('Pres') or self.dicts[i].has_key('Sres'):
                stationColor = 'black'
            else:
                stationColor = 'gray'
            # plot stations at respective coordinates with names
            self.axEventMap.scatter([self.dicts[i]['StaLon']],
                                    [self.dicts[i]['StaLat']], s = 150,
                                    marker = 'v', color = '',
                                    edgecolor = stationColor)
            self.axEventMap.text(self.dicts[i]['StaLon'],
                                 self.dicts[i]['StaLat'],
                                 '  ' + self.dicts[i]['Station'],
                                 color = stationColor,
                                 va = 'top', family = 'monospace')
            if self.dicts[i].has_key('Pres'):
                presinfo = '\n\n %+0.3fs' % self.dicts[i]['Pres']
                if self.dicts[i].has_key('PPol'):
                    presinfo += '  %s' % self.dicts[i]['PPol']
                self.axEventMap.text(self.dicts[i]['StaLon'],
                                     self.dicts[i]['StaLat'],
                                     presinfo, va = 'top',
                                     family = 'monospace',
                                     color = self.dictPhaseColors['P'])
            if self.dicts[i].has_key('Sres'):
                sresinfo = '\n\n\n %+0.3fs' % self.dicts[i]['Sres']
                if self.dicts[i].has_key('SPol'):
                    sresinfo += '  %s' % self.dicts[i]['SPol']
                self.axEventMap.text(self.dicts[i]['StaLon'],
                                     self.dicts[i]['StaLat'],
                                     sresinfo, va = 'top',
                                     family = 'monospace',
                                     color = self.dictPhaseColors['S'])
            if self.dicts[i].has_key('Mag'):
                self.scatterMagIndices.append(i)
                self.scatterMagLon.append(self.dicts[i]['StaLon'])
                self.scatterMagLat.append(self.dicts[i]['StaLat'])
                self.axEventMap.text(self.dicts[i]['StaLon'], self.dicts[i]['StaLat'],
                                  '  ' + self.dicts[i]['Station'], va = 'top',
                                  family = 'monospace')
                self.axEventMap.text(self.dicts[i]['StaLon'], self.dicts[i]['StaLat'],
                                  '\n\n\n\n  %0.2f (%s)' % (self.dicts[i]['Mag'],
                                  self.dicts[i]['MagChannel']), va = 'top',
                                  family = 'monospace',
                                  color = self.dictPhaseColors['Mag'])
            if len(self.scatterMagLon) > 0 :
                self.scatterMag = self.axEventMap.scatter(self.scatterMagLon, self.scatterMagLat, s = 150,
                                     marker = 'v', color = '', edgecolor = 'black', picker = 10)
                
        self.axEventMap.set_xlabel('Longitude')
        self.axEventMap.set_ylabel('Latitude')
        self.axEventMap.set_title(self.dictOrigin['Time'])
        self.axEventMap.axis('equal')
        #XXX disabled because it plots the wrong info if the event was
        # fetched from seishub
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
        self.figEventMap.canvas.mpl_connect('pick_event', self.selectMagnitudes)
        try:
            self.scatterMag.set_facecolors(self.eventMapColors)
        except:
            pass
        plt.show()

    def selectMagnitudes(self, event):
        if event.artist != self.scatterMag:
            return
        i = self.scatterMagIndices[event.ind[0]]
        j = event.ind[0]
        self.dicts[i]['MagUse'] = not self.dicts[i]['MagUse']
        #print event.ind[0]
        #print i
        #print event.artist
        #for di in self.dicts:
        #    print di['MagUse']
        #print i
        #print self.dicts[i]['MagUse']
        if self.dicts[i]['MagUse']:
            self.eventMapColors[j] = (0.,  1.,  0.,  1.)
        else:
            self.eventMapColors[j] = (0.,  0.,  0.,  0.)
        #print self.eventMapColors
        self.scatterMag.set_facecolors(self.eventMapColors)
        #print self.scatterMag.get_facecolors()
        #event.artist.set_facecolors(self.eventMapColors)
        self.updateNetworkMag()
        self.figEventMap.canvas.draw()

    def dicts2XML(self):
        """
        Returns information of all dictionaries as xml file
        """
        xml =  Element("event")
        Sub(Sub(xml, "event_id"), "value").text = self.dictEvent['xmlEventID']
        event_type = Sub(xml, "event_type")
        Sub(event_type, "value").text = "manual"
        Sub(event_type, "user").text = self.username
        Sub(event_type, "public").text = "%s" % \
                self.checkbuttonPublicEvent.get_active()
        
        # XXX standard values for unset keys!!!???!!!???
        epidists = []
        # go through all stream-dictionaries and look for picks
        for i in range(len(self.streams)):
            d = self.dicts[i]
            st = self.streams[i]

            # write P Pick info
            if 'P' in d:
                pick = Sub(xml, "pick")
                wave = Sub(pick, "waveform")
                wave.set("networkCode", st[0].stats.network) 
                wave.set("stationCode", st[0].stats.station) 
                wave.set("channelCode", st[0].stats.channel) 
                wave.set("locationCode", st[0].stats.location) 
                date = Sub(pick, "time")
                # prepare time of pick
                picktime = st[0].stats.starttime
                picktime += d['P']
                Sub(date, "value").text = picktime.isoformat() # + '.%06i' % picktime.microsecond)
                if 'PErr1' in d and 'PErr2' in d:
                    temp = d['PErr2'] - d['PErr1']
                    Sub(date, "uncertainty").text = str(temp)
                else:
                    Sub(date, "uncertainty")
                Sub(pick, "phaseHint").text = "P"
                phase_compu = ""
                if 'POnset' in d:
                    Sub(pick, "onset").text = d['POnset']
                    if d['POnset'] == "impulsive":
                        phase_compu += "I"
                    elif d['POnset'] == "emergent":
                        phase_compu += "E"
                else:
                    Sub(pick, "onset")
                    phase_compu += "?"
                phase_compu += "P"
                if 'PPol' in d:
                    Sub(pick, "polarity").text = d['PPol']
                    if d['PPol'] == 'up':
                        phase_compu += "U"
                    elif d['PPol'] == 'poorup':
                        phase_compu += "+"
                    elif d['PPol'] == 'down':
                        phase_compu += "D"
                    elif d['PPol'] == 'poordown':
                        phase_compu += "-"
                else:
                    Sub(pick, "polarity")
                    phase_compu += "?"
                if 'PWeight' in d:
                    Sub(pick, "weight").text = '%i' % d['PWeight']
                    phase_compu += "%1i" % d['PWeight']
                else:
                    Sub(pick, "weight")
                    phase_compu += "?"
                Sub(Sub(pick, "min_amp"), "value") #XXX what is min_amp???
                
                if 'Psynth' in d:
                    Sub(pick, "phase_compu").text = phase_compu
                    Sub(Sub(pick, "phase_res"), "value").text = str(d['Pres'])
                    if self.dictOrigin['Program'] == "hyp2000" and \
                       'PsynthWeight' in d:
                        Sub(Sub(pick, "phase_weight"), "value").text = \
                                str(d['PsynthWeight'])
                    else:
                        Sub(Sub(pick, "phase_weight"), "value")
                    Sub(Sub(pick, "phase_delay"), "value")
                    Sub(Sub(pick, "azimuth"), "value").text = str(d['PAzim'])
                    Sub(Sub(pick, "incident"), "value").text = str(d['PInci'])
                    Sub(Sub(pick, "epi_dist"), "value").text = \
                            str(d['distEpi'])
                    Sub(Sub(pick, "hyp_dist"), "value").text = \
                            str(d['distHypo'])
        
            # write S Pick info
            if 'S' in d:
                axind = d['Saxind']
                pick = Sub(xml, "pick")
                wave = Sub(pick, "waveform")
                wave.set("networkCode", st[axind].stats.network) 
                wave.set("stationCode", st[axind].stats.station) 
                wave.set("channelCode", st[axind].stats.channel) 
                wave.set("locationCode", st[axind].stats.location) 
                date = Sub(pick, "time")
                # prepare time of pick
                picktime = st[axind].stats.starttime
                picktime += d['S']
                Sub(date, "value").text = picktime.isoformat() # + '.%06i' % picktime.microsecond)
                if 'SErr1' in d and 'SErr2' in d:
                    temp = d['SErr2'] - d['SErr1']
                    Sub(date, "uncertainty").text = str(temp)
                else:
                    Sub(date, "uncertainty")
                Sub(pick, "phaseHint").text = "S"
                phase_compu = ""
                if 'SOnset' in d:
                    Sub(pick, "onset").text = d['SOnset']
                    if d['SOnset'] == "impulsive":
                        phase_compu += "I"
                    elif d['SOnset'] == "emergent":
                        phase_compu += "E"
                else:
                    Sub(pick, "onset")
                    phase_compu += "?"
                phase_compu += "S"
                if 'SPol' in d:
                    Sub(pick, "polarity").text = d['SPol']
                    if d['SPol'] == 'up':
                        phase_compu += "U"
                    elif d['SPol'] == 'poorup':
                        phase_compu += "+"
                    elif d['SPol'] == 'down':
                        phase_compu += "D"
                    elif d['SPol'] == 'poordown':
                        phase_compu += "-"
                else:
                    Sub(pick, "polarity")
                    phase_compu += "?"
                if 'SWeight' in d:
                    Sub(pick, "weight").text = '%i' % d['SWeight']
                    phase_compu += "%1i" % d['SWeight']
                else:
                    Sub(pick, "weight")
                    phase_compu += "?"
                Sub(Sub(pick, "min_amp"), "value") #XXX what is min_amp???
                
                if 'Ssynth' in d:
                    Sub(pick, "phase_compu").text = phase_compu
                    Sub(Sub(pick, "phase_res"), "value").text = '%s' % self.dicts[i]['Sres']
                    if self.dictOrigin['Program'] == "hyp2000" and \
                       'SsynthWeight' in d:
                        Sub(Sub(pick, "phase_weight"), "value").text = \
                                str(d['SsynthWeight'])
                    else:
                        Sub(Sub(pick, "phase_weight"), "value")
                    Sub(Sub(pick, "phase_delay"), "value")
                    Sub(Sub(pick, "azimuth"), "value").text = str(d['SAzim'])
                    Sub(Sub(pick, "incident"), "value").text = str(d['SInci'])
                    Sub(Sub(pick, "epi_dist"), "value").text = \
                            str(d['distEpi'])
                    Sub(Sub(pick, "hyp_dist"), "value").text = \
                            str(d['distHypo'])

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
            for i in range(len(self.streams)):
                d = self.dicts[i]
                st = self.streams[i]
                if 'Mag' in d:
                    stationMagnitude = Sub(xml, "stationMagnitude")
                    mag = Sub(stationMagnitude, 'mag')
                    Sub(mag, 'value').text = str(d['Mag'])
                    Sub(mag, 'uncertainty').text
                    Sub(stationMagnitude, 'station').text = str(d['Station'])
                    if d['MagUse']:
                        Sub(stationMagnitude, 'weight').text = str(1. / dM['Station Count'])
                    else:
                        Sub(stationMagnitude, 'weight').text = "0"
                    Sub(stationMagnitude, 'channels').text = str(d['MagChannel'])
        
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
        userid = "admin"
        passwd = "admin"

        auth = 'Basic ' + (base64.encodestring(userid + ':' + passwd)).strip()

        path = '/xml/seismology/event'
        
        # overwrite the same xml file always when using option local
        # which is intended for testing purposes only
        if self.options.local:
            self.dictEvent['xmlEventID'] = '19700101000000'
        name = "obspyck_%s" % (self.dictEvent['xmlEventID']) #XXX id of the file
        # create XML and also save in temporary directory for inspection purposes
        msg = "creating xml..."
        appendTextview(self.textviewStdOut, msg)
        data = self.dicts2XML()
        tmpfile = self.tmp_dir + name + ".xml"
        msg = "writing xml as %s (for debugging purposes only!)" % tmpfile
        appendTextview(self.textviewStdOut, msg)
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
        if statuscode!=201:
            msg = "User: %s" % self.username
            msg += "\nName: %s" % name
            msg += "\nServer: %s%s" % (self.server['Server'], path)
            msg += "\nResponse: %s %s" % (statuscode, statusmessage)
            msg += "\nHeader:"
            msg += "\n%s" % str(header).strip()
            appendTextview(self.textviewStdOut, msg)
        else:
            err = "Warning/Error: Got HTTP status code 201!?!"
            appendTextview(self.textviewStdErr, err)
    
    def clearDictionaries(self):
        msg = "Clearing previous data."
        appendTextview(self.textviewStdOut, msg)
        for i in range(len(self.dicts)):
            for k in self.dicts[i].keys():
                if k != 'Station' and k != 'StaLat' and k != 'StaLon' and \
                   k != 'StaEle' and k != 'pazZ' and k != 'pazN' and \
                   k != 'pazE':
                    del self.dicts[i][k]
            self.dicts[i]['MagUse'] = True
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
        appendTextview(self.textviewStdOut, msg)
        # we need to delete all station magnitude information from all dicts
        for i in range(len(self.dicts)):
            for k in self.dicts[i].keys():
                if k != 'Station' and k != 'StaLat' and k != 'StaLon' and \
                   k != 'StaEle' and k != 'pazZ' and k != 'pazN' and \
                   k != 'pazE' and k != 'P' and k != 'PErr1' and \
                   k != 'PErr2' and k != 'POnset' and k != 'PPol' and \
                   k != 'PWeight' and k != 'S' and k != 'SErr1' and \
                   k != 'SErr2' and k != 'SOnset' and k != 'SPol' and \
                   k != 'SWeight' and k != 'Saxind':
                    del self.dicts[i][k]
            self.dicts[i]['MagUse'] = True
        self.dictOrigin = {}
        self.dictMagnitude = {}
        self.dictEvent = {}
        self.dictEvent['xmlEventID'] = None

    def clearFocmecDictionary(self):
        msg = "Clearing previous focal mechanism data."
        appendTextview(self.textviewStdOut, msg)
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

    def getNextEventFromSeishub(self, starttime, endtime):
        """
        Updates dictionary with pick data for first event which origin time
        is between startime and endtime.
        Warning:
         * When using the stream starttime an interesting event may not be
           found because the origin time may be before the stream starttime!
         * If more than one event is found in given time range only the first
           one is used, all others are disregarded!

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
        auth = base64.encodestring('%s:%s' % ("admin", "admin"))[:-1]
        req.add_header("Authorization", "Basic %s" % auth)

        f = urllib2.urlopen(req)
        xml = parse(f)
        f.close()

        picklist = []

        # iterate the counter that indicates which event to fetch
        if not self.seishubEventCount:
            self.seishubEventCount = len(xml.xpath(u".//resource_name"))
            self.seishubEventCurrent = 0
            msg = "%i events are available from Seishub" % self.seishubEventCount
            appendTextview(self.textviewStdOut, msg)
            if self.seishubEventCount == 0:
                return
        else:
            self.seishubEventCurrent = (self.seishubEventCurrent + 1) % \
                                       self.seishubEventCount

        # define which event data we will fetch
        node = xml.xpath(u".//resource_name")[self.seishubEventCurrent]
        #document = xml.xpath(".//document_id")
        #document_id = document[self.seishubEventCurrent].text
        # Hack to show xml resource as document id
        document_id = node.text
        
        resource_url = self.server['BaseUrl'] + "/xml/seismology/event/" + \
                       node.text
        resource_req = urllib2.Request(resource_url)
        resource_req.add_header("Authorization", "Basic %s" % auth)
        fp = urllib2.urlopen(resource_req)
        resource_xml = parse(fp)
        fp.close()
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
            for i in range(len(self.streams)):
                if station.strip() != self.dicts[i]['Station']:
                    continue
                else:
                    streamnum = i
                    break
            if streamnum == None:
                message = "Did not find matching stream for pick data " + \
                          "with station id: \"%s\"" % station.strip()
                warnings.warn(message)
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
            if pick.xpath(".//phaseHint")[0].text == "P":
                self.dicts[streamnum]['P'] = time
                if uncertainty:
                    self.dicts[streamnum]['PErr1'] = time - uncertainty
                    self.dicts[streamnum]['PErr2'] = time + uncertainty
                if onset:
                    self.dicts[streamnum]['POnset'] = onset
                if polarity:
                    self.dicts[streamnum]['PPol'] = polarity
                if weight:
                    self.dicts[streamnum]['PWeight'] = int(weight)
                if phase_res:
                    self.dicts[streamnum]['Psynth'] = time + float(phase_res)
                    self.dicts[streamnum]['Pres'] = float(phase_res)
                # hypo2000 uses this weight internally during the inversion
                # this is not the same as the weight assigned during picking
                if phase_weight:
                    self.dicts[streamnum]['PsynthWeight'] = phase_weight
                if azimuth:
                    self.dicts[streamnum]['PAzim'] = float(azimuth)
                if incident:
                    self.dicts[streamnum]['PInci'] = float(incident)
            if pick.xpath(".//phaseHint")[0].text == "S":
                self.dicts[streamnum]['S'] = time
                # XXX maybe dangerous to check last character:
                if channel.endswith('N'):
                    self.dicts[streamnum]['Saxind'] = 1
                if channel.endswith('E'):
                    self.dicts[streamnum]['Saxind'] = 2
                if uncertainty:
                    self.dicts[streamnum]['SErr1'] = time - uncertainty
                    self.dicts[streamnum]['SErr2'] = time + uncertainty
                if onset:
                    self.dicts[streamnum]['SOnset'] = onset
                if polarity:
                    self.dicts[streamnum]['SPol'] = polarity
                if weight:
                    self.dicts[streamnum]['SWeight'] = int(weight)
                if phase_res:
                    self.dicts[streamnum]['Ssynth'] = time + float(phase_res)
                    self.dicts[streamnum]['Sres'] = float(phase_res)
                # hypo2000 uses this weight internally during the inversion
                # this is not the same as the weight assigned during picking
                if phase_weight:
                    self.dicts[streamnum]['SsynthWeight'] = phase_weight
                if azimuth:
                    self.dicts[streamnum]['SAzim'] = float(azimuth)
                if incident:
                    self.dicts[streamnum]['SInci'] = float(incident)
            if epi_dist:
                self.dicts[streamnum]['distEpi'] = float(epi_dist)
            if hyp_dist:
                self.dicts[streamnum]['distHypo'] = float(hyp_dist)

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
                self.netMagLabel = '\n\n\n\n  %.2f (Var: %.2f)' % (self.dictMagnitude['Magnitude'], self.dictMagnitude['Uncertainty'])
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
        msg = "Fetched event %i of %i (event_id: %s, user: %s)" % \
              (self.seishubEventCurrent + 1, self.seishubEventCount,
               document_id, user)
        appendTextview(self.textviewStdOut, msg)

def main():
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
    parser.add_option("-s", "--servername", dest="servername",
                      help="Servername of the seishub server",
                      default='teide')
    parser.add_option("-p", "--port", type="int", dest="port",
                      help="Port of the seishub server",
                      default=8080)
    parser.add_option("--user", dest="user", default='admin',
                      help="Username for seishub server")
    parser.add_option("--password", dest="password", default='admin',
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
    (options, args) = parser.parse_args()
    for req in ['-d','-t','-i']:
        if not getattr(parser.values,parser.get_option(req).dest):
            parser.print_help()
            return
    
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
    else:
        try:
            t = UTCDateTime(options.time)
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
            for sta in client.waveform.getStationIds(network_id=net):
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
                                                     t + options.duration)
                    print net_sta, "fetched successfully."
                    sta_fetched.add(net_sta)
                except:
                    print net_sta, "could not be retrieved. Skipping!"
                    continue
                st.sort()
                st.reverse()
                streams.append(st)

    PickingGUI(client, streams, options)

if __name__ == "__main__":
    main()
