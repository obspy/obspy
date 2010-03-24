#!/usr/bin/env python

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

#sys.path.append('/baysoft/obspy/obspy/branches/symlink')
#os.chdir("/baysoft/obspyck/")
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
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter

#gtk
import gtk
import gobject #we use this only for redirecting StdOut and StdErr
import gtk.glade
try:
    import pango #we use this only for changing the font in the textviews
except:
    pass
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
    for widg in children:
        nofocus_recursive(widg)

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
    
    #def set_visible(self, boolean):
    #    for line in self.lines:
    #        line.set_visible(boolean)
 
# we pimp our gtk textview elements, so that they are accesible via a write()
# method. we use this to redirect stdout and stderr to our textviews
# See: http://cssed.sourceforge.net/docs/
#      pycssed_developers_guide-html-0.1/x139.html
class WritableTextView:
    def __init__(self, textview):
        self.textview = textview
    def write(self, string):        
        buffer = self.textview.get_buffer()
        iter = buffer.get_end_iter()
        buffer.insert(iter,string)
   
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
        #self.drawAllItems()
        #self.redraw()
        self.togglebuttonShowMap.set_active(True)

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
        #self.drawAllItems()
        #self.redraw()
        self.togglebuttonShowMap.set_active(True)

    def on_buttonCalcMag_clicked(self, event):
        self.calculateEpiHypoDists()
        self.dictMagnitude['Program'] = "obspy"
        self.calculateStationMagnitudes()
        self.updateNetworkMag()

    def on_buttonDoFocmec_clicked(self, event):
        self.clearFocmecDictionary()
        self.dictFocalMechanism['Program'] = "focmec"
        self.doFocmec()

    def on_togglebuttonShowMap_clicked(self, event):
        buttons_deactivate = [self.buttonClearAll, self.buttonClearOrigMag,
                              self.buttonClearFocMec, self.buttonDoHyp2000,
                              self.buttonDo3dloc, self.buttonCalcMag,
                              self.buttonDoFocmec, self.togglebuttonShowFocMec,
                              self.buttonNextFocMec, self.togglebuttonShowWadati,
                              self.buttonGetNextEvent, self.buttonSendEvent,
                              self.checkbuttonPublicEvent,
                              self.buttonPreviousStream, self.buttonNextStream,
                              self.comboboxPhaseType, self.togglebuttonFilter,
                              self.comboboxFilterType,
                              self.checkbuttonZeroPhase,
                              self.spinbuttonHighpass, self.spinbuttonLowpass,
                              self.togglebuttonSpectrogram]
        state = self.togglebuttonShowMap.get_active()
        for button in buttons_deactivate:
            button.set_sensitive(not state)
        if state:
            self.delAxes()
            self.fig.clear()
            self.drawEventMap()
            self.multicursor.visible = False
            self.toolbar.pan(True)
            self.toolbar.update()
            self.canv.draw()
        else:
            self.delEventMap()
            self.fig.clear()
            self.drawAxes()
            self.toolbar.update()
            self.drawSavedPicks()
            self.multicursorReinit()
            self.updatePlot()
            self.updateStreamLabels()
            self.canv.draw()

    def on_togglebuttonShowFocMec_clicked(self, event):
        buttons_deactivate = [self.buttonClearAll, self.buttonClearOrigMag,
                              self.buttonClearFocMec, self.buttonDoHyp2000,
                              self.buttonDo3dloc, self.buttonCalcMag,
                              self.buttonDoFocmec, self.togglebuttonShowMap,
                              self.togglebuttonShowWadati,
                              self.buttonGetNextEvent, self.buttonSendEvent,
                              self.checkbuttonPublicEvent,
                              self.buttonPreviousStream, self.buttonNextStream,
                              self.comboboxPhaseType, self.togglebuttonFilter,
                              self.comboboxFilterType,
                              self.checkbuttonZeroPhase,
                              self.spinbuttonHighpass, self.spinbuttonLowpass,
                              self.togglebuttonSpectrogram]
        state = self.togglebuttonShowFocMec.get_active()
        for button in buttons_deactivate:
            button.set_sensitive(not state)
        if state:
            self.delAxes()
            self.fig.clear()
            self.drawFocMec()
            self.multicursor.visible = False
            self.toolbar.pan()
            self.toolbar.zoom()
            self.toolbar.zoom()
            self.toolbar.update()
            self.canv.draw()
        else:
            self.delFocMec()
            self.fig.clear()
            self.drawAxes()
            self.toolbar.update()
            self.drawSavedPicks()
            self.multicursorReinit()
            self.updatePlot()
            self.updateStreamLabels()
            self.canv.draw()

    def on_buttonNextFocMec_clicked(self, event):
        self.nextFocMec()
        if self.togglebuttonShowFocMec.get_active():
            self.delFocMec()
            self.fig.clear()
            self.drawFocMec()
            self.canv.draw()

    def on_togglebuttonShowWadati_clicked(self, event):
        buttons_deactivate = [self.buttonClearAll, self.buttonClearOrigMag,
                              self.buttonClearFocMec, self.buttonDoHyp2000,
                              self.buttonDo3dloc, self.buttonCalcMag,
                              self.buttonDoFocmec, self.togglebuttonShowFocMec,
                              self.buttonNextFocMec, self.togglebuttonShowMap,
                              self.buttonGetNextEvent, self.buttonSendEvent,
                              self.checkbuttonPublicEvent,
                              self.buttonPreviousStream, self.buttonNextStream,
                              self.comboboxPhaseType, self.togglebuttonFilter,
                              self.comboboxFilterType,
                              self.checkbuttonZeroPhase,
                              self.spinbuttonHighpass, self.spinbuttonLowpass,
                              self.togglebuttonSpectrogram]
        state = self.togglebuttonShowWadati.get_active()
        for button in buttons_deactivate:
            button.set_sensitive(not state)
        if state:
            self.delAxes()
            self.fig.clear()
            self.drawWadati()
            self.multicursor.visible = False
            self.toolbar.pan()
            self.toolbar.update()
            self.canv.draw()
        else:
            self.delWadati()
            self.fig.clear()
            self.drawAxes()
            self.toolbar.update()
            self.drawSavedPicks()
            self.multicursorReinit()
            self.updatePlot()
            self.updateStreamLabels()
            self.canv.draw()

    def on_buttonGetNextEvent_clicked(self, event):
        self.delAllItems()
        self.clearDictionaries()
        # check if event list is empty and force an update if this is the case
        if not hasattr(self, "seishubEventList"):
            self.updateEventListFromSeishub(self.streams[0][0].stats.starttime,
                                            self.streams[0][0].stats.endtime)
        # iterate event number to fetch
        self.seishubEventCurrent = (self.seishubEventCurrent + 1) % \
                                   self.seishubEventCount
        resource_name = self.seishubEventList[self.seishubEventCurrent].text
        self.getEventFromSeishub(resource_name)
        #self.getNextEventFromSeishub(self.streams[0][0].stats.starttime, 
        #                             self.streams[0][0].stats.endtime)
        self.drawAllItems()
        self.redraw()
        
        #XXX 

    def on_buttonUpdateEventList_clicked(self, event):
        self.updateEventListFromSeishub(self.streams[0][0].stats.starttime,
                                        self.streams[0][0].stats.endtime)

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
        self.fig.clear()
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
        self.delAllItems()
        self.delAxes()
        self.fig.clear()
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
        sys.stdout = self.textviewStdOutWritable
        sys.stderr = self.textviewStdErrWritable

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

        # we have to control which binaries to use depending on architecture...
        architecture = platform.architecture()[0]
        if architecture == '32bit':
            self.threeDlocBinaryName = '3dloc_pitsa_32bit'
            self.hyp2000BinaryName = 'hyp2000_32bit'
        elif architecture == '64bit':
            self.threeDlocBinaryName = '3dloc_pitsa_64bit'
            self.hyp2000BinaryName = 'hyp2000_64bit'
        else:
            msg = "Warning: Could not determine architecture (32/64bit). " + \
                  "Using 32bit 3dloc binary."
            warnings.warn(msg)
            self.threeDlocBinaryName = '3dloc_pitsa_32bit'
            self.hyp2000BinaryName = 'hyp2000_32bit'

        self.threeDlocPath = self.options.pluginpath + '/3dloc/'
        self.threeDlocPath_D3_VELOCITY = self.threeDlocPath + 'D3_VELOCITY'
        self.threeDlocPath_D3_VELOCITY_2 = self.threeDlocPath + 'D3_VELOCITY_2'
        self.threeDlocOutfile = self.tmp_dir + '3dloc-out'
        self.threeDlocInfile = self.tmp_dir + '3dloc-in'
        # copy 3dloc files to temp directory
        subprocess.call('cp %s/* %s &> /dev/null' % \
                (self.threeDlocPath, self.tmp_dir), shell=True)
        self.threeDlocPreCall = 'rm %s %s &> /dev/null' \
                % (self.threeDlocOutfile, self.threeDlocInfile)
        self.threeDlocCall = 'export D3_VELOCITY=%s/;' % \
                self.threeDlocPath_D3_VELOCITY + \
                'export D3_VELOCITY_2=%s/;' % \
                self.threeDlocPath_D3_VELOCITY_2 + \
                'cd %s; ./%s' % (self.tmp_dir, self.threeDlocBinaryName)
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
                           './%s < bay2000.inp &> /dev/null' % \
                           self.hyp2000BinaryName
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
        
        # If keybindings option is set only show keybindings and exit
        if self.options.keybindings:
            for key, value in self.dictKeybindings.iteritems():
                print "%s: \"%s\"" % (key, value)
            return

        # Return, if no streams are given
        if not streams:
            return

        # Define some forbidden scenarios.
        # We assume there are:
        # - either one Z or three ZNE traces
        # - no two streams for any station (of same network)
        sta_list = set()
        # we need to go through streams/dicts backwards in order not to get
        # problems because of the pop() statement
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
                print 'Warning: All streams must have either one Z trace or a set of three ZNE traces.'
                print 'Stream %s discarded. Reason: Number of traces != (1 or 3)' % net_sta
                for j, tr in enumerate(st.traces):
                    print 'Trace no. %i in Stream: %s' % (j + 1, tr.stats.channel)
                    print tr.stats
                self.streams.pop(i)
                continue
            if len(st.traces) == 1 and st[0].stats.channel[-1] != 'Z':
                print 'Warning: All streams must have either one Z trace or a set of three ZNE traces.'
                print 'Stream %s discarded. Reason: Exactly one trace present but this is no Z trace' % net_sta
                for j, tr in enumerate(st.traces):
                    print 'Trace no. %i in Stream: %s' % (j + 1, tr.stats.channel)
                    print tr.stats
                self.streams.pop(i)
                continue
            if len(st.traces) == 3 and (st[0].stats.channel[-1] != 'Z' or
                                        st[1].stats.channel[-1] != 'N' or
                                        st[2].stats.channel[-1] != 'E' or
                                        st[0].stats.station.strip() !=
                                        st[1].stats.station.strip() or
                                        st[0].stats.station.strip() !=
                                        st[2].stats.station.strip()):
                print 'Warning: All streams must have either one Z trace or a set of three ZNE traces.'
                print 'Stream %s discarded. Reason: Exactly three traces present but they are not ZNE' % net_sta
                for j, tr in enumerate(st.traces):
                    print 'Trace no. %i in Stream: %s' % (j + 1, tr.stats.channel)
                    print tr.stats
                self.streams.pop(i)
                continue
            sta_list.add(net_sta)

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
                dict['StaLon'] = lon
                dict['StaLat'] = lat
                dict['StaEle'] = ele / 1000. # all depths in km!
                print dict['StaLon'], dict['StaLat'], dict['StaEle']
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
        self.stNum = len(self.streams)
        self.stPt = 0
    
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
        #self.win.set_title("ObsPyck")
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
        self.buttonClearAll = self.gla.get_widget("buttonClearAll")
        self.buttonClearOrigMag = self.gla.get_widget("buttonClearOrigMag")
        self.buttonClearFocMec = self.gla.get_widget("buttonClearFocMec")
        self.buttonDoHyp2000 = self.gla.get_widget("buttonDoHyp2000")
        self.buttonDo3dloc = self.gla.get_widget("buttonDo3dloc")
        self.buttonCalcMag = self.gla.get_widget("buttonCalcMag")
        self.buttonDoFocmec = self.gla.get_widget("buttonDoFocmec")
        self.togglebuttonShowMap = self.gla.get_widget("togglebuttonShowMap")
        self.togglebuttonShowFocMec = self.gla.get_widget("togglebuttonShowFocMec")
        self.buttonNextFocMec = self.gla.get_widget("buttonNextFocMec")
        self.togglebuttonShowWadati = self.gla.get_widget("togglebuttonShowWadati")
        self.buttonGetNextEvent = self.gla.get_widget("buttonGetNextEvent")
        self.buttonSendEvent = self.gla.get_widget("buttonSendEvent")
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
        self.multicursorReinit()
        self.canv.show()

        # redirect stdout and stderr
        # first we need to create a new subinstance with write method
        self.textviewStdOutWritable = WritableTextView(self.textviewStdOut)
        self.textviewStdErrWritable = WritableTextView(self.textviewStdErr)
        self.stdout_backup = sys.stdout
        self.stderr_backup = sys.stderr
        sys.stdout = self.textviewStdOutWritable
        sys.stderr = self.textviewStdErrWritable

        # change fonts of textview
        # see http://www.pygtk.org/docs/pygtk/class-pangofontdescription.html
        try:
            fontDescription = pango.FontDescription("monospace condensed 9")
            self.textviewStdOut.modify_font(fontDescription)
            self.textviewStdErr.modify_font(fontDescription)
        except NameError:
            pass

        gtk.main()

    
    ## Trim all to same length, us Z as reference
    #start, end = stZ[0].stats.starttime, stZ[0].stats.endtime
    #stN.trim(start, end)
    #stE.trim(start, end)
    
    
    def drawAxes(self):
        st = self.streams[self.stPt]
        #we start all our x-axes at 0 with the starttime of the first (Z) trace
        starttime_global = st[0].stats.starttime
        self.axs = []
        self.plts = []
        self.trans = []
        self.t = []
        trNum = len(st.traces)
        for i in range(trNum):
            npts = st[i].stats.npts
            smprt = st[i].stats.sampling_rate
            #make sure that the relative times of the x-axes get mapped to our
            #global stream (absolute) starttime (starttime of first (Z) trace)
            starttime_local = st[i].stats.starttime - starttime_global
            dt = 1. / smprt
            self.t.append(np.arange(starttime_local,
                                    starttime_local + (dt * npts),
                                    dt))
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
            self.axs[i].xaxis.set_major_formatter(FuncFormatter(formatXTicklabels))
            if not self.flagSpectrogram:
                self.plts.append(self.axs[i].plot(self.t[i], st[i].data, color='k',zorder=1000)[0])
            else:
                spectrogram(st[i].data, st[i].stats.sampling_rate,
                            axis=self.axs[i],
                            nwin=st[i].stats.npts * 4 / st[i].stats.sampling_rate)
        self.supTit = self.fig.suptitle("%s.%03d -- %s.%03d" % (st[0].stats.starttime.strftime("%Y-%m-%d  %H:%M:%S"),
                                                         st[0].stats.starttime.microsecond / 1e3 + 0.5,
                                                         st[0].stats.endtime.strftime("%H:%M:%S"),
                                                         st[0].stats.endtime.microsecond / 1e3 + 0.5), ha="left", va="bottom", x=0.01, y=0.01)
        self.xMin, self.xMax=self.axs[0].get_xlim()
        self.yMin, self.yMax=self.axs[0].get_ylim()
        #self.fig.subplots_adjust(bottom=0.04, hspace=0.01, right=0.999, top=0.94, left=0.06)
        self.fig.subplots_adjust(bottom=0.001, hspace=0.000, right=0.999, top=0.999, left=0.001)
        self.toolbar.update()
        self.toolbar.pan(False)
        self.toolbar.zoom(True)
    
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
        if hasattr(self, "PLines"):
            self.delPLine()
        dict = self.dicts[self.stPt]
        if not 'P' in dict:
            return
        self.PLines = []
        for ax in self.axs:
            line = ax.axvline(dict['P'], color=self.dictPhaseColors['P'],
                              linewidth=self.axvlinewidths,
                              linestyle=self.dictPhaseLinestyles['P'])
            self.PLines.append(line)
    
    def delPLine(self):
        if not hasattr(self, "PLines"):
            return
        for i, ax in enumerate(self.axs):
            if self.PLines[i] in ax.lines:
                ax.lines.remove(self.PLines[i])
        del self.PLines
    
    def drawPsynthLine(self):
        if hasattr(self, "PsynthLines"):
            self.delPsynthLine()
        dict = self.dicts[self.stPt]
        if not 'Psynth' in dict:
            return
        self.PsynthLines = []
        for ax in self.axs:
            line = ax.axvline(dict['Psynth'], linewidth=self.axvlinewidths,
                              color=self.dictPhaseColors['Psynth'],
                              linestyle=self.dictPhaseLinestyles['Psynth'])
            self.PsynthLines.append(line)
    
    def delPsynthLine(self):
        if not hasattr(self, "PsynthLines"):
            return
        for i, ax in enumerate(self.axs):
            if self.PsynthLines[i] in ax.lines:
                ax.lines.remove(self.PsynthLines[i])
        del self.PsynthLines
    
    def drawPLabel(self):
        dict = self.dicts[self.stPt]
        if not 'P' in dict:
            return
        label = 'P:'
        if 'POnset' in dict:
            if dict['POnset'] == 'impulsive':
                label += 'I'
            elif dict['POnset'] == 'emergent':
                label += 'E'
            else:
                label += '?'
        else:
            label += '_'
        if 'PPol' in dict:
            if dict['PPol'] == 'up':
                label += 'U'
            elif dict['PPol'] == 'poorup':
                label += '+'
            elif dict['PPol'] == 'down':
                label += 'D'
            elif dict['PPol'] == 'poordown':
                label += '-'
            else:
                label += '?'
        else:
            label += '_'
        if 'PWeight' in dict:
            label += str(dict['PWeight'])
        else:
            label += '_'
        self.PLabel = self.axs[0].text(dict['P'], 1 - 0.01 * len(self.axs),
                                       '  ' + label, transform = self.trans[0],
                                       color = self.dictPhaseColors['P'],
                                       family = 'monospace', va="top")
    
    def delPLabel(self):
        if not hasattr(self, "PLabel"):
            return
        if self.PLabel in self.axs[0].texts:
            self.axs[0].texts.remove(self.PLabel)
        del self.PLabel
    
    def drawPsynthLabel(self):
        dict = self.dicts[self.stPt]
        if not 'Psynth' in dict:
            return
        label = 'Psynth: %+.3fs' % dict['Pres']
        self.PsynthLabel = self.axs[0].text(dict['Psynth'],
                1 - 0.03 * len(self.axs), '  ' + label, va="top",
                transform=self.trans[0], color=self.dictPhaseColors['Psynth'])
    
    def delPsynthLabel(self):
        if not hasattr(self, "PsynthLabel"):
            return
        if self.PsynthLabel in self.axs[0].texts:
            self.axs[0].texts.remove(self.PsynthLabel)
        del self.PsynthLabel
    
    def drawPErr1Line(self):
        if hasattr(self, "PErr1Lines"):
            self.delPErr1Line()
        dict = self.dicts[self.stPt]
        if not 'P' in dict or not 'PErr1' in dict:
            return
        self.PErr1Lines = []
        for ax in self.axs:
            line = ax.axvline(dict['PErr1'], ymin=0.25, ymax=0.75,
                              color=self.dictPhaseColors['P'],
                              linewidth=self.axvlinewidths)
            self.PErr1Lines.append(line)
    
    def delPErr1Line(self):
        if not hasattr(self, "PErr1Lines"):
            return
        for i, ax in enumerate(self.axs):
            if self.PErr1Lines[i] in ax.lines:
                ax.lines.remove(self.PErr1Lines[i])
        del self.PErr1Lines
    
    def drawPErr2Line(self):
        if hasattr(self, "PErr2Lines"):
            self.delPErr2Line()
        dict = self.dicts[self.stPt]
        if not 'P' in dict or not 'PErr2' in dict:
            return
        self.PErr2Lines = []
        for ax in self.axs:
            line = ax.axvline(dict['PErr2'], ymin=0.25, ymax=0.75,
                              color=self.dictPhaseColors['P'],
                              linewidth=self.axvlinewidths)
            self.PErr2Lines.append(line)
    
    def delPErr2Line(self):
        if not hasattr(self, "PErr2Lines"):
            return
        for i, ax in enumerate(self.axs):
            if self.PErr2Lines[i] in ax.lines:
                ax.lines.remove(self.PErr2Lines[i])
        del self.PErr2Lines

    def drawSLine(self):
        if hasattr(self, "SLines"):
            self.delSLine()
        dict = self.dicts[self.stPt]
        if not 'S' in dict:
            return
        self.SLines = []
        for ax in self.axs:
            line = ax.axvline(dict['S'], color=self.dictPhaseColors['S'],
                              linewidth=self.axvlinewidths,
                              linestyle=self.dictPhaseLinestyles['S'])
            self.SLines.append(line)
    
    def delSLine(self):
        if not hasattr(self, "SLines"):
            return
        for i, ax in enumerate(self.axs):
            if self.SLines[i] in ax.lines:
                ax.lines.remove(self.SLines[i])
        del self.SLines
    
    def drawSsynthLine(self):
        if hasattr(self, "SsynthLines"):
            self.delSsynthLine()
        dict = self.dicts[self.stPt]
        if not 'Ssynth' in dict:
            return
        self.SsynthLines = []
        for ax in self.axs:
            line = ax.axvline(dict['Ssynth'], linewidth=self.axvlinewidths,
                              color=self.dictPhaseColors['Ssynth'],
                              linestyle=self.dictPhaseLinestyles['Ssynth'])
            self.SsynthLines.append(line)
    
    def delSsynthLine(self):
        if not hasattr(self, "SsynthLines"):
            return
        for i, ax in enumerate(self.axs):
            if self.SsynthLines[i] in ax.lines:
                ax.lines.remove(self.SsynthLines[i])
        del self.SsynthLines
    
    def drawSLabel(self):
        dict = self.dicts[self.stPt]
        if not 'S' in dict:
            return
        label = 'S:'
        if 'SOnset' in dict:
            if dict['SOnset'] == 'impulsive':
                label += 'I'
            elif dict['SOnset'] == 'emergent':
                label += 'E'
            else:
                label += '?'
        else:
            label += '_'
        if 'SPol' in dict:
            if dict['SPol'] == 'up':
                label += 'U'
            elif dict['SPol'] == 'poorup':
                label += '+'
            elif dict['SPol'] == 'down':
                label += 'D'
            elif dict['SPol'] == 'poordown':
                label += '-'
            else:
                label += '?'
        else:
            label += '_'
        if 'SWeight' in dict:
            label += str(dict['SWeight'])
        else:
            label += '_'
        self.SLabel = self.axs[0].text(dict['S'], 1 - 0.01 * len(self.axs),
                                       '  ' + label, transform = self.trans[0],
                                       color = self.dictPhaseColors['S'],
                                       family = 'monospace', va="top")
    
    def delSLabel(self):
        if not hasattr(self, "SLabel"):
            return
        if self.SLabel in self.axs[0].texts:
            self.axs[0].texts.remove(self.SLabel)
        del self.SLabel
    
    def drawSsynthLabel(self):
        dict = self.dicts[self.stPt]
        if not 'Ssynth' in dict:
            return
        label = 'Ssynth: %+.3fs' % dict['Sres']
        self.SsynthLabel = self.axs[0].text(dict['Ssynth'],
                1 - 0.03 * len(self.axs), '  ' + label, va="top",
                transform=self.trans[0], color=self.dictPhaseColors['Ssynth'])
    
    def delSsynthLabel(self):
        if not hasattr(self, "SsynthLabel"):
            return
        if self.SsynthLabel in self.axs[0].texts:
            self.axs[0].texts.remove(self.SsynthLabel)
        del self.SsynthLabel
    
    def drawSErr1Line(self):
        if hasattr(self, "SErr1Lines"):
            self.delSErr1Line()
        dict = self.dicts[self.stPt]
        if not 'S' in dict or not 'SErr1' in dict:
            return
        self.SErr1Lines = []
        for ax in self.axs:
            line = ax.axvline(dict['SErr1'], ymin=0.25, ymax=0.75,
                              color=self.dictPhaseColors['S'],
                              linewidth=self.axvlinewidths)
            self.SErr1Lines.append(line)
    
    def delSErr1Line(self):
        if not hasattr(self, "SErr1Lines"):
            return
        for i, ax in enumerate(self.axs):
            if self.SErr1Lines[i] in ax.lines:
                ax.lines.remove(self.SErr1Lines[i])
        del self.SErr1Lines
    
    def drawSErr2Line(self):
        if hasattr(self, "SErr2Lines"):
            self.delSErr2Line()
        dict = self.dicts[self.stPt]
        if not 'S' in dict or not 'SErr2' in dict:
            return
        self.SErr2Lines = []
        for ax in self.axs:
            line = ax.axvline(dict['SErr2'], ymin=0.25, ymax=0.75,
                              color=self.dictPhaseColors['S'],
                              linewidth=self.axvlinewidths)
            self.SErr2Lines.append(line)
    
    def delSErr2Line(self):
        if not hasattr(self, "SErr2Lines"):
            return
        for i, ax in enumerate(self.axs):
            if self.SErr2Lines[i] in ax.lines:
                ax.lines.remove(self.SErr2Lines[i])
        del self.SErr2Lines

    def drawMagMinCross1(self):
        dict = self.dicts[self.stPt]
        if not 'MagMin1' in dict or len(self.axs) < 2:
            return
        # we have to force the graph to the old axes limits because of the
        # completely new line object creation
        xlims = list(self.axs[0].get_xlim())
        ylims = list(self.axs[0].get_ylim())
        self.MagMinCross1 = self.axs[1].plot([dict['MagMin1T']],
                [dict['MagMin1']], markersize=self.magMarkerSize,
                markeredgewidth=self.magMarkerEdgeWidth,
                color=self.dictPhaseColors['Mag'], marker=self.magMinMarker,
                zorder=2000)[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delMagMinCross1(self):
        if not hasattr(self, "MagMinCross1"):
            return
        ax = self.axs[1]
        if self.MagMinCross1 in ax.lines:
            ax.lines.remove(self.MagMinCross1)
    
    def drawMagMaxCross1(self):
        dict = self.dicts[self.stPt]
        if not 'MagMax1' in dict or len(self.axs) < 2:
            return
        # we have to force the graph to the old axes limits because of the
        # completely new line object creation
        xlims = list(self.axs[0].get_xlim())
        ylims = list(self.axs[0].get_ylim())
        self.MagMaxCross1 = self.axs[1].plot([dict['MagMax1T']],
                [dict['MagMax1']], markersize=self.magMarkerSize,
                markeredgewidth=self.magMarkerEdgeWidth,
                color=self.dictPhaseColors['Mag'], marker=self.magMinMarker,
                zorder=2000)[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delMagMaxCross1(self):
        if not hasattr(self, "MagMaxCross1"):
            return
        ax = self.axs[1]
        if self.MagMaxCross1 in ax.lines:
            ax.lines.remove(self.MagMaxCross1)
    
    def drawMagMinCross2(self):
        dict = self.dicts[self.stPt]
        if not 'MagMin2' in dict or len(self.axs) < 3:
            return
        # we have to force the graph to the old axes limits because of the
        # completely new line object creation
        xlims = list(self.axs[0].get_xlim())
        ylims = list(self.axs[0].get_ylim())
        self.MagMinCross2 = self.axs[2].plot([dict['MagMin2T']],
                [dict['MagMin2']], markersize=self.magMarkerSize,
                markeredgewidth=self.magMarkerEdgeWidth,
                color=self.dictPhaseColors['Mag'], marker=self.magMinMarker,
                zorder=2000)[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delMagMinCross2(self):
        if not hasattr(self, "MagMinCross2"):
            return
        ax = self.axs[2]
        if self.MagMinCross2 in ax.lines:
            ax.lines.remove(self.MagMinCross2)
    
    def drawMagMaxCross2(self):
        dict = self.dicts[self.stPt]
        if not 'MagMax2' in dict or len(self.axs) < 3:
            return
        # we have to force the graph to the old axes limits because of the
        # completely new line object creation
        xlims = list(self.axs[0].get_xlim())
        ylims = list(self.axs[0].get_ylim())
        self.MagMaxCross2 = self.axs[2].plot([dict['MagMax2T']],
                [dict['MagMax2']], markersize=self.magMarkerSize,
                markeredgewidth=self.magMarkerEdgeWidth,
                color=self.dictPhaseColors['Mag'], marker=self.magMinMarker,
                zorder=2000)[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delMagMaxCross2(self):
        if not hasattr(self, "MagMaxCross2"):
            return
        ax = self.axs[2]
        if self.MagMaxCross2 in ax.lines:
            ax.lines.remove(self.MagMaxCross2)
    
    def delP(self):
        dict = self.dicts[self.stPt]
        if not 'P' in dict:
            return
        del dict['P']
        msg = "P Pick deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delPsynth(self):
        dict = self.dicts[self.stPt]
        if not 'Psynth' in dict:
            return
        del dict['Psynth']
        msg = "synthetic P Pick deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delPWeight(self):
        dict = self.dicts[self.stPt]
        if not 'PWeight' in dict:
            return
        del dict['PWeight']
        msg = "P Pick weight deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delPPol(self):
        dict = self.dicts[self.stPt]
        if not 'PPol' in dict:
            return
        del dict['PPol']
        msg = "P Pick polarity deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delPOnset(self):
        dict = self.dicts[self.stPt]
        if not 'POnset' in dict:
            return
        del dict['POnset']
        msg = "P Pick onset deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delPErr1(self):
        dict = self.dicts[self.stPt]
        if not 'PErr1' in dict:
            return
        del dict['PErr1']
        msg = "PErr1 Pick deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delPErr2(self):
        dict = self.dicts[self.stPt]
        if not 'PErr2' in dict:
            return
        del dict['PErr2']
        msg = "PErr2 Pick deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delS(self):
        dict = self.dicts[self.stPt]
        if not 'S' in dict:
            return
        del dict['S']
        if 'Saxind' in dict:
            del dict['Saxind']
        msg = "S Pick deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delSsynth(self):
        dict = self.dicts[self.stPt]
        if not 'Ssynth' in dict:
            return
        del dict['Ssynth']
        msg = "synthetic S Pick deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delSWeight(self):
        dict = self.dicts[self.stPt]
        if not 'SWeight' in dict:
            return
        del dict['SWeight']
        msg = "S Pick weight deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delSPol(self):
        dict = self.dicts[self.stPt]
        if not 'SPol' in dict:
            return
        del dict['SPol']
        msg = "S Pick polarity deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delSOnset(self):
        dict = self.dicts[self.stPt]
        if not 'SOnset' in dict:
            return
        del dict['SOnset']
        msg = "S Pick onset deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delSErr1(self):
        dict = self.dicts[self.stPt]
        if not 'SErr1' in dict:
            return
        del dict['SErr1']
        msg = "SErr1 Pick deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delSErr2(self):
        dict = self.dicts[self.stPt]
        if not 'SErr2' in dict:
            return
        del dict['SErr2']
        msg = "SErr2 Pick deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delMagMin1(self):
        dict = self.dicts[self.stPt]
        if not 'MagMin1' in dict:
            return
        del dict['MagMin1']
        del dict['MagMin1T']
        msg = "Magnitude Minimum Estimation Pick deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delMagMax1(self):
        dict = self.dicts[self.stPt]
        if not 'MagMax1' in dict:
            return
        del dict['MagMax1']
        del dict['MagMax1T']
        msg = "Magnitude Maximum Estimation Pick deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delMagMin2(self):
        dict = self.dicts[self.stPt]
        if not 'MagMin2' in dict:
            return
        del dict['MagMin2']
        del dict['MagMin2T']
        msg = "Magnitude Minimum Estimation Pick deleted"
        appendTextview(self.textviewStdOut, msg)
            
    def delMagMax2(self):
        dict = self.dicts[self.stPt]
        if not 'MagMax2' in dict:
            return
        del dict['MagMax2']
        del dict['MagMax2T']
        msg = "Magnitude Maximum Estimation Pick deleted"
        appendTextview(self.textviewStdOut, msg)
    
    def delAxes(self):
        for ax in self.axs:
            if ax in self.fig.axes: 
                self.fig.delaxes(ax)
            del ax
        if self.supTit in self.fig.texts:
            self.fig.texts.remove(self.supTit)
    
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
                    err = "Error: Unrecognized Filter Option. Showing " + \
                          "unfiltered data."
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
        if self.togglebuttonShowMap.get_active():
            return
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
                if len(self.axs) < 2:
                    err = "Error: Magnitude picking only supported with a " + \
                          "minimum of 2 axes."
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
                if len(self.axs) < 2:
                    err = "Error: Magnitude picking only supported with a " + \
                          "minimum of 2 axes."
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
        if self.togglebuttonShowMap.get_active():
            return
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
        if self.togglebuttonShowMap.get_active():
            return
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
        if self.togglebuttonShowMap.get_active():
            return
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
                        appendTextview(self.textviewStdErr, err)
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
        sys.stdout.write("".join(sub.stdout.readlines()))
        sys.stderr.write("".join(sub.stderr.readlines()))
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
            if 'P' in dict:
                t = st[0].stats.starttime
                t += dict['P']
                date = t.strftime("%Y %m %d %H %M %S")
                date += ".%03d" % (t.microsecond / 1e3 + 0.5)
                delta = dict['PErr2'] - dict['PErr1']
                f.write(fmt % (dict['Station'], 'P', date, delta, lon, lat,
                               ele / 1e3))
            if 'S' in dict:
                t = st[0].stats.starttime
                t += dict['S']
                date = t.strftime("%Y %m %d %H %M %S")
                date += ".%03d" % (t.microsecond / 1e3 + 0.5)
                delta = dict['SErr2'] - dict['SErr1']
                f.write(fmt % (dict['Station'], 'S', date, delta, lon, lat,
                               ele / 1e3))
        f.close()
        msg = 'Phases for 3Dloc:'
        appendTextview(self.textviewStdOut, msg)
        self.catFile(self.threeDlocInfile)
        #subprocess.call(self.threeDlocCall, shell=True)
        sub = subprocess.Popen(self.threeDlocCall, shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sys.stdout.write("".join(sub.stdout.readlines()))
        sys.stderr.write("".join(sub.stderr.readlines()))
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
        appendTextview(self.textviewStdOut, msg)
        self.catFile(self.focmecPhasefile)
        sub = subprocess.Popen(self.focmecCall, shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sys.stdout.write("".join(sub.stdout.readlines()))
        sys.stderr.write("".join(sub.stderr.readlines()))
        if sub.returncode == 1:
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
        dF = self.dictFocalMechanism
        msg = "Dip: %6.2f  Strike: %6.2f  Rake: %6.2f  Errors: %i/%i" % \
                (dF['Dip'], dF['Strike'], dF['Rake'], dF['Errors'],
                 dF['Station Polarity Count'])
        appendTextview(self.textviewStdOut, msg)

    def nextFocMec(self):
        if self.focMechCount is None:
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
    
    #XXX replace with drawFocMec
    def drawFocMec(self):
        if self.dictFocalMechanism == {}:
            err = "Error: No focal mechanism data!"
            appendTextview(self.textviewStdErr, err)
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
                    #azim = -180. + dict['PAzim']
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
        sub = subprocess.Popen(self.hyp2000PreCall, shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sys.stdout.write("".join(sub.stdout.readlines()))
        sys.stderr.write("".join(sub.stderr.readlines()))
        f = open(self.hyp2000Phasefile, 'w')
        f2 = open(self.hyp2000Stationsfile, 'w')
        network = "BW"
        #fmt = "RWMOIP?0 091229124412.22       13.99IS?0"
        fmtP = "%4s%1sP%1s%1i %15s"
        fmtS = "%12s%1sS%1s%1i\n"
        #fmt2 = "  BGLD4739.14N01300.75E 930"
        fmt2 = "%6s%02i%05.2fN%03i%05.2fE%4i\n"
        #self.coords = []
        for i, dict in enumerate(self.dicts):
            sta = dict['Station']
            lon = dict['StaLon']
            lon_deg = int(lon)
            lon_min = (lon - lon_deg) * 60.
            lat = dict['StaLat']
            lat_deg = int(lat)
            lat_min = (lat - lat_deg) * 60.
            ele = dict['StaEle'] * 1000
            f2.write(fmt2 % (sta, lat_deg, lat_min, lon_deg, lon_min, ele))
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
                f.write(fmtP % (sta, onset, polarity, weight, date))
            if 'S' in dict:
                if not 'P' in dict:
                    err = "Warning: Trying to print a Hypo2000 phase file " + \
                          "with an S phase without P phase.\n" + \
                          "This case might not be covered correctly and " + \
                          "could screw our file up!"
                    appendTextview(self.textviewStdErr, err)
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
                    appendTextview(self.textviewStdErr, err)
                    f.write("\n")
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
        sub = subprocess.Popen(self.hyp2000Call, shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sys.stdout.write("".join(sub.stdout.readlines()))
        sys.stderr.write("".join(sub.stderr.readlines()))
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
                appendTextview(self.textviewStdErr, err)
                continue
            
            # assign synthetic phase info
            dict = self.dicts[streamnum]
            if type == "P":
                dO['used P Count'] += 1
                dict['Psynth'] = res + dict['P']
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
                dict['Ssynth'] = res + dict['S']
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
        appendTextview(self.textviewStdOut, msg)
        dM = self.dictMagnitude
        dM['Station Count'] = 0
        dM['Magnitude'] = 0
        staMags = []
        for dict in self.dicts:
            if dict['MagUse'] and 'Mag' in dict:
                msg = "%s: %.1f" % (dict['Station'], dict['Mag'])
                appendTextview(self.textviewStdOut, msg)
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
        appendTextview(self.textviewStdOut, msg)
        self.netMagLabel = '\n\n\n\n  %.2f (Var: %.2f)' % (dM['Magnitude'],
                                                           dM['Uncertainty'])
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
                appendTextview(self.textviewStdOut, msg)
            
            elif 'MagMin1' in dict and 'MagMax1' in dict:
                amp = dict['MagMax1'] - dict['MagMin1']
                timedelta = abs(dict['MagMax1T'] - dict['MagMin1T'])
                mag = estimateMagnitude(dict['pazN'], amp, timedelta,
                                        dict['distHypo'])
                dict['Mag'] = mag
                dict['MagChannel'] = '%s' % st[1].stats.channel
                msg = 'calculated new magnitude for %s: %0.2f (channels: %s)' \
                      % (dict['Station'], dict['Mag'], dict['MagChannel'])
                appendTextview(self.textviewStdOut, msg)
            
            elif 'MagMin2' in dict and 'MagMax2' in dict:
                amp = dict['MagMax2'] - dict['MagMin2']
                timedelta = abs(dict['MagMax2T'] - dict['MagMin2T'])
                mag = estimateMagnitude(dict['pazE'], amp, timedelta,
                                        dict['distHypo'])
                dict['Mag'] = mag
                dict['MagChannel'] = '%s' % st[2].stats.channel
                msg = 'calculated new magnitude for %s: %0.2f (channels: %s)' \
                      % (dict['Station'], dict['Mag'], dict['MagChannel'])
                appendTextview(self.textviewStdOut, msg)
    
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
            appendTextview(self.textviewStdErr, err)
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

    def drawEventMap(self):
        dM = self.dictMagnitude
        dO = self.dictOrigin
        if dO == {}:
            err = "Error: No hypocenter data!"
            appendTextview(self.textviewStdErr, err)
            return
        #toolbar.pan()
        #XXX self.figEventMap.canvas.widgetlock.release(toolbar)
        self.axEventMap = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=0.07, top=0.95, left=0.07, right=0.98)
        self.axEventMap.scatter([dO['Longitude']], [dO['Latitude']], 30,
                                color='red', marker='o')
        errLon, errLat = utlLonLat(dO['Longitude'], dO['Latitude'],
                                   dO['Longitude Error'], dO['Latitude Error'])
        errLon -= dO['Longitude']
        errLat -= dO['Latitude']
        self.axEventMap.text(dO['Longitude'], dO['Latitude'],
                             ' %7.3f +/- %0.2fkm\n' % \
                             (dO['Longitude'], dO['Longitude Error']) + \
                             ' %7.3f +/- %0.2fkm\n' % \
                             (dO['Latitude'], dO['Latitude Error']) + \
                             '  %.1fkm +/- %.1fkm' % \
                             (dO['Depth'], dO['Depth Error']),
                             va='top', family='monospace')
        try:
            self.netMagLabel = '\n\n\n\n  %.2f (Var: %.2f)' % \
                    (dM['Magnitude'], dM['Uncertainty'])
            self.netMagText = self.axEventMap.text(dO['Longitude'],
                    dO['Latitude'], self.netMagLabel, va='top',
                    color='green', family = 'monospace')
        except:
            pass
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
        self.axEventMap.axis('equal')
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

    def delEventMap(self):
        try:
            self.canv.mpl_disconnect(self.eventMapPickEvent)
        except AttributeError:
            pass
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
                    if self.dictOrigin['Program'] == "hyp2000" and \
                       'PsynthWeight' in dict:
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
                    Sub(Sub(pick, "phase_res"), "value").text = '%s' % dict['Sres']
                    if self.dictOrigin['Program'] == "hyp2000" and \
                       'SsynthWeight' in dict:
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
        # if we did no location at all, and only picks hould be saved the
        # EventID ist still not set, so we have to do this now.
        if self.dictEvent['xmlEventID'] is None:
            self.setXMLEventID()
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
        msg = "User: %s" % self.username
        msg += "\nName: %s" % name
        msg += "\nServer: %s%s" % (self.server['Server'], path)
        msg += "\nResponse: %s %s" % (statuscode, statusmessage)
        #msg += "\nHeader:"
        #msg += "\n%s" % str(header).strip()
        appendTextview(self.textviewStdOut, msg)
    
    def clearDictionaries(self):
        msg = "Clearing previous data."
        appendTextview(self.textviewStdOut, msg)
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
        appendTextview(self.textviewStdOut, msg)
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
    
    def getEventFromSeishub(self, resource_name):
        #document = xml.xpath(".//document_id")
        #document_id = document[self.seishubEventCurrent].text
        # Hack to show xml resource as document id
        resource_url = self.server['BaseUrl'] + "/xml/seismology/event/" + \
                       resource_name
        resource_req = urllib2.Request(resource_url)
        auth = base64.encodestring('%s:%s' % ("admin", "admin"))[:-1]
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
            for i, dict in enumerate(self.dicts):
                if station.strip() != dict['Station']:
                    continue
                else:
                    streamnum = i
                    break
            if streamnum is None:
                err = "Warning: Did not find matching stream for pick " + \
                      "data with station id: \"%s\"" % station.strip()
                appendTextview(self.textviewStdErr, err)
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
                    dict['Psynth'] = time + float(phase_res)
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
                    dict['Ssynth'] = time + float(phase_res)
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
               resource_name, user)
        appendTextview(self.textviewStdOut, msg)

    def updateEventListFromSeishub(self, starttime, endtime):
        """
        Searches for events in the database and stores a list of resource
        names. All events with at least one pick set in between start- and
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
        auth = base64.encodestring('%s:%s' % ("admin", "admin"))[:-1]
        req.add_header("Authorization", "Basic %s" % auth)

        f = urllib2.urlopen(req)
        xml = parse(f)
        f.close()

        # populate list with resource names of all available events
        self.seishubEventList = xml.xpath(u".//resource_name")

        self.seishubEventCount = len(self.seishubEventList)
        # we set the current event-pointer to the last list element, because we
        # iterate the counter immediately when fetching the first event...
        self.seishubEventCurrent = self.seishubEventCount - 1
        msg = "%i events are available from Seishub" % self.seishubEventCount
        for event in self.seishubEventList:
            msg += "\n  - %s" % event.text
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

    PickingGUI(client, streams, options)

if __name__ == "__main__":
    main()
