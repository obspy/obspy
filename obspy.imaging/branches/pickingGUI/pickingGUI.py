#!/usr/bin/env python

#check for textboxes and other stuff:
#http://code.enthought.com/projects/traits/docs/html/tutorials/traits_ui_scientific_app.html

#matplotlib.use('gtkagg')

from lxml.etree import SubElement as Sub, parse, tostring
from lxml.etree import fromstring, Element
from optparse import OptionParser
import numpy as np
import sys
import subprocess
import httplib
import base64
import time

from obspy.core import read, UTCDateTime
from obspy.seishub import Client
from obspy.signal.filter import bandpass, bandpassZPHSH, bandstop, bandstopZPHSH
from obspy.signal.filter import lowpass, lowpassZPHSH, highpass, highpassZPHSH
from obspy.signal.util import utlLonLat

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor as mplMultiCursor
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.patches import Ellipse

#imports for the buttons
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.mathtext as mathtext
import matplotlib.artist as artist
import matplotlib.image as image

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

#Some class definitions for the menu buttons
#code from: http://matplotlib.sourceforge.net/examples/widgets/menu.html
class ItemProperties:
    def __init__(self, fontsize=12, labelcolor='black', bgcolor='yellow', alpha=1.0):
        self.fontsize = fontsize
        self.labelcolor = labelcolor
        self.bgcolor = bgcolor
        self.alpha = alpha
        self.labelcolor_rgb = colors.colorConverter.to_rgb(labelcolor)
        self.bgcolor_rgb = colors.colorConverter.to_rgb(bgcolor)

class MenuItem(artist.Artist):
    parser = mathtext.MathTextParser("Bitmap")
    padx = 5
    pady = 5
    def __init__(self, fig, labelstr, props=None, hoverprops=None, on_select=None):
        artist.Artist.__init__(self)
        self.set_figure(fig)
        self.labelstr = labelstr
        if props is None:
            props = ItemProperties()
        if hoverprops is None:
            hoverprops = ItemProperties()
        self.props = props
        self.hoverprops = hoverprops
        self.on_select = on_select
        x, self.depth = self.parser.to_mask(
            labelstr, fontsize=props.fontsize, dpi=fig.dpi)
        if props.fontsize!=hoverprops.fontsize:
            raise NotImplementedError('support for different font sizes not implemented')
        self.labelwidth = x.shape[1]
        self.labelheight = x.shape[0]
        self.labelArray = np.zeros((x.shape[0], x.shape[1], 4))
        self.labelArray[:,:,-1] = x/255.
        self.label = image.FigureImage(fig, origin='upper')
        self.label.set_array(self.labelArray)
        # we'll update these later
        self.rect = patches.Rectangle((0,0), 1,1)
        self.set_hover_props(False)
        fig.canvas.mpl_connect('button_release_event', self.check_select)

    def check_select(self, event):
        over, junk = self.rect.contains(event)
        if not over:
            return
        if self.on_select is not None:
            self.on_select(self)

    def set_extent(self, x, y, w, h):
        #print x, y, w, h
        self.rect.set_x(x)
        self.rect.set_y(y)
        self.rect.set_width(w)
        self.rect.set_height(h)
        self.label.ox = x+self.padx
        self.label.oy = y-self.depth+self.pady/2.
        self.rect._update_patch_transform()
        self.hover = False

    def draw(self, renderer):
        self.rect.draw(renderer)
        self.label.draw(renderer)

    def set_hover_props(self, b):
        if b:
            props = self.hoverprops
        else:
            props = self.props
        r, g, b = props.labelcolor_rgb
        self.labelArray[:,:,0] = r
        self.labelArray[:,:,1] = g
        self.labelArray[:,:,2] = b
        self.label.set_array(self.labelArray)
        self.rect.set(facecolor=props.bgcolor, alpha=props.alpha)

    def set_hover(self, event):
        'check the hover status of event and return true if status is changed'
        b,junk = self.rect.contains(event)
        changed = (b != self.hover)
        if changed:
            self.set_hover_props(b)
        self.hover = b
        return changed

class Menu:
    def __init__(self, fig, menuitems):
        self.figure = fig
        fig.suppressComposite = True
        self.menuitems = menuitems
        self.numitems = len(menuitems)
        maxw = max([item.labelwidth for item in menuitems])
        maxh = max([item.labelheight for item in menuitems])
        totalh = self.numitems*maxh + (self.numitems+1)*2*MenuItem.pady
        x0 = 5
        y0 = 5
        y1 = y0 + (self.numitems-1)*(maxh + MenuItem.pady)
        width = maxw + 2*MenuItem.padx
        height = maxh+MenuItem.pady
        for item in menuitems:
            left = x0
            #bottom = y0-maxh-MenuItem.pady
            bottom = y1
            item.set_extent(left, bottom, width, height)
            fig.artists.append(item)
            y1 -= maxh + MenuItem.pady
        fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        draw = False
        for item in self.menuitems:
            draw = item.set_hover(event)
            if draw:
                self.figure.canvas.draw()
                break
    
def getCoord(network, station):
    """
    Returns longitude, latitude and elevation of given station
    """
    client = Client()
    coord = []

    resource = "dataless.seed.%s_%s.xml" % (network, station)
    xml = fromstring(client.station.getResource(resource, format='metadata'))

    for attrib in [u'Longitude (\xb0)', u'Latitude (\xb0)',  u'Elevation (m)']:
        node =  xml.xpath(u".//item[@title='%s']" % attrib)[0]
        value = float(node.getchildren()[0].attrib['text'])
        coord.append(value)

    return coord

class PickingGUI:

    def __init__(self, streams = None):
        self.streams = streams
        #Define some flags, dictionaries and plotting options
        self.flagFilt=False #False:no filter  True:filter
        self.flagFiltTyp=0 #0: bandpass 1: bandstop 2:lowpass 3: highpass
        self.dictFiltTyp={'Bandpass':0, 'Bandstop':1, 'Lowpass':2, 'Highpass':3}
        self.flagFiltZPH=False #False: no zero-phase True: zero-phase filtering
        self.valFiltLow=np.NaN # These are overridden with low/high estimated from sampling rate
        self.valFiltHigh=np.NaN
        self.flagWheelZoom=True #Switch use of mousewheel for zooming
        self.flagPhase=0 #0:P 1:S 2:Magnitude
        self.dictPhase={'P':0, 'S':1, 'Mag':2}
        self.dictPhaseInverse = {} # We need the reverted dictionary for switching throug the Phase radio button
        for i in self.dictPhase.items():
            self.dictPhaseInverse[i[1]] = i[0]
        self.dictPhaseColors={'P':'red', 'S':'blue', 'Psynth':'black', 'Ssynth':'black', 'Mag':'green'}
        self.dictPhaseLinestyles={'P':'-', 'S':'-', 'Psynth':'--', 'Ssynth':'--'}
        self.pickingColor = self.dictPhaseColors['P']
        self.magPickWindow=10 #Estimating the maximum/minimum in a sample-window around click
        self.magMinMarker='x'
        self.magMaxMarker='x'
        self.magMarkerEdgeWidth=1.8
        self.magMarkerSize=20
        self.axvlinewidths=1.2
        #dictionary for key-bindings
        self.dictKeybindings = {'setPick': 'alt', 'setPickError': ' ', 'delPick': 'escape',
                           'setMagMin': 'alt', 'setMagMax': ' ', 'switchPhase': 'control',
                           'delMagMinMax': 'escape', 'switchWheelZoom': 'z',
                           'switchPan': 'p', 'prevStream': 'y', 'nextStream': 'x',
                           'setPWeight0': '0', 'setPWeight1': '1', 'setPWeight2': '2',
                           'setPWeight3': '3', # 'setPWeight4': '4', 'setPWeight5': '5',
                           'setSWeight0': '0', 'setSWeight1': '1', 'setSWeight2': '2',
                           'setSWeight3': '3', # 'setSWeight4': '4', 'setSWeight5': '5',
                           'setPPolUp': 'u', 'setPPolPoorUp': '+',
                           'setPPolDown': 'd', 'setPPolPoorDown': '-',
                           'setSPolUp': 'u', 'setSPolPoorUp': '+',
                           'setSPolDown': 'd', 'setSPolPoorDown': '-',
                           'setPOnsetImpulsive': 'i', 'setPOnsetEmergent': 'e',
                           'setSOnsetImpulsive': 'i', 'setSOnsetEmergent': 'e',}
        self.threeDlocOutfile = './3dloc-out'
        self.threeDlocInfile = './3dloc-in'
        self.xmlEventID = None
        
        # Return, if no streams are given
        if not streams:
            return

        # Define some forbidden scenarios.
        # We assume there are:
        # - either one Z or three ZNE traces
        # - no two streams for any station
        self.stationlist=[]
        for st in streams:
            if not (len(st.traces) == 1 or len(st.traces) == 3):
                print 'Error: All streams must have either one Z trace or a set of three ZNE traces'
                return
            if len(st.traces) == 1 and st[0].stats.channel[-1] != 'Z':
                print 'Error: All streams must have either one Z trace or a set of three ZNE traces'
                return
            if len(st.traces) == 3 and (st[0].stats.channel[-1] != 'Z' or
                                        st[1].stats.channel[-1] != 'N' or
                                        st[2].stats.channel[-1] != 'E' or
                                        st[0].stats.station.strip() !=
                                        st[1].stats.station.strip() or
                                        st[0].stats.station.strip() !=
                                        st[2].stats.station.strip()):
                print 'Error: All streams must have either one Z trace or a set of ZNE traces (from the same station)'
                return
            self.stationlist.append(st[0].stats.station.strip())
        if len(self.stationlist) != len(set(self.stationlist)):
            print 'Error: Found two streams for one station'
            return

        #set up a list of dictionaries to store all picking data
        # set all station magnitude use-flags False
        self.dicts = []
        self.eventMapColors = []
        for i in range(len(self.streams)):
            self.dicts.append({})
            self.dicts[i]['MagUse'] = True
            self.dicts[i]['Station'] = streams[i][0].stats.station.rstrip()
            self.eventMapColors.append((0.,  1.,  0.,  1.))
            #XXX uncomment following lines for use with dynamically acquired data from seishub!
            #lon, lat, ele = getCoord(network, self.stationlist[i])
            #self.dicts[i]['Station'] = self.stationlist[i]
            #self.dicts[i]['StaLon'] = lon
            #self.dicts[i]['StaLat'] = lat
            #self.dicts[i]['StaEle'] = ele

        #XXX Remove lines for use with dynamically acquired data from seishub!
        self.dicts[0]['StaLon'] = 12.795714
        self.dicts[1]['StaLon'] = 12.864466
        self.dicts[2]['StaLon'] = 12.867100
        self.dicts[3]['StaLon'] = 12.824082
        self.dicts[4]['StaLon'] = 12.729887
        self.dicts[0]['StaLat'] = 47.737167
        self.dicts[1]['StaLat'] = 47.761658
        self.dicts[2]['StaLat'] = 47.740501
        self.dicts[3]['StaLat'] = 47.745098
        self.dicts[4]['StaLat'] = 47.744171
        self.dicts[0]['StaEle'] = 0.860000
        self.dicts[1]['StaEle'] = 0.815000
        self.dicts[2]['StaEle'] = 0.555000
        self.dicts[3]['StaEle'] = 1.162000
        self.dicts[4]['StaEle'] = 0.763000

        #XXX only for testing purposes
        self.dicts[0]['Mag'] = 1.34
        self.dicts[1]['Mag'] = 1.03
        #self.dicts[2]['Mag'] = 1.22
        self.dicts[3]['Mag'] = 0.65
        self.dicts[4]['Mag'] = 0.96

        #Define a pointer to navigate through the streams
        self.stNum=len(streams)
        self.stPt=0
    
        # Set up initial plot
        self.fig = plt.figure()
        self.drawAxes()
        self.addFiltButtons()
        self.addPhaseButtons()
        self.addSliders()
        #redraw()
        self.fig.canvas.draw()
        # Activate all mouse/key/Cursor-events
        self.keypress = self.fig.canvas.mpl_connect('key_press_event', self.pick)
        self.keypressWheelZoom = self.fig.canvas.mpl_connect('key_press_event', self.switchWheelZoom)
        self.keypressPan = self.fig.canvas.mpl_connect('key_press_event', self.switchPan)
        self.keypressNextPrev = self.fig.canvas.mpl_connect('key_press_event', self.switchStream)
        self.keypressSwitchPhase = self.fig.canvas.mpl_connect('key_press_event', self.switchPhase)
        self.buttonpressBlockRedraw = self.fig.canvas.mpl_connect('button_press_event', self.blockRedraw)
        self.buttonreleaseAllowRedraw = self.fig.canvas.mpl_connect('button_release_event', self.allowRedraw)
        self.scroll = self.fig.canvas.mpl_connect('scroll_event', self.zoom)
        self.scroll_button = self.fig.canvas.mpl_connect('button_press_event', self.zoom_reset)
        self.fig.canvas.toolbar.zoom()
        self.fig.canvas.widgetlock.release(self.fig.canvas.toolbar)
        #multicursor = mplMultiCursor(fig.canvas,axs, useblit=True, color='black', linewidth=1, ls='dotted')
        self.multicursor = MultiCursor(self.fig.canvas,self.axs, useblit=True, color=self.dictPhaseColors['P'], linewidth=1, ls='dotted')
        for l in self.multicursor.lines:
            l.set_color(self.dictPhaseColors['P'])
        self.radioPhase.circles[0].set_facecolor(self.dictPhaseColors['P'])
        #add menu buttons:
        props = ItemProperties(labelcolor='black', bgcolor='yellow', fontsize=12, alpha=0.2)
        hoverprops = ItemProperties(labelcolor='white', bgcolor='blue', fontsize=12, alpha=0.2)
        menuitems = []
        for label in ('do3dloc', 'showMap', 'save', 'quit'):
            def on_select(item):
                print '--> ', item.labelstr
                if item.labelstr == 'quit':
                    plt.close()
                elif item.labelstr == 'do3dloc':
                    self.do3dLoc()
                elif item.labelstr == 'showMap':
                    self.load3dlocData()
                    self.show3dlocEventMap()
            item = MenuItem(self.fig, label, props=props, hoverprops=hoverprops, on_select=on_select)
            menuitems.append(item)
        self.menu = Menu(self.fig, menuitems)
        
        
        
        plt.show()
    
    
    def switch_flagFilt(self):
        self.flagFilt=not self.flagFilt
    def switch_flagFiltZPH(self):
        self.flagFiltZPH=not self.flagFiltZPH
    
    ## Trim all to same length, us Z as reference
    #start, end = stZ[0].stats.starttime, stZ[0].stats.endtime
    #stN.trim(start, end)
    #stE.trim(start, end)
    
    
    def drawAxes(self):
        self.t = np.arange(self.streams[self.stPt][0].stats.npts)
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
                self.axs.append(self.fig.add_subplot(trNum,1,i+1,sharex=self.axs[0],sharey=self.axs[0]))
                self.trans.append(matplotlib.transforms.blended_transform_factory(self.axs[i].transData,
                                                                             self.axs[i].transAxes))
            self.axs[i].set_ylabel(self.streams[self.stPt][i].stats.station+" "+self.streams[self.stPt][i].stats.channel)
            self.plts.append(self.axs[i].plot(self.t, self.streams[self.stPt][i].data, color='k',zorder=1000)[0])
        self.supTit=self.fig.suptitle("%s -- %s, %s" % (self.streams[self.stPt][0].stats.starttime, self.streams[self.stPt][0].stats.endtime, self.streams[self.stPt][0].stats.station))
        self.xMin, self.xMax=self.axs[0].get_xlim()
        self.yMin, self.yMax=self.axs[0].get_ylim()
        self.fig.subplots_adjust(bottom=0.20,hspace=0,right=0.999,top=0.95)
    
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
                PLabelString += 'i'
            elif self.dicts[self.stPt]['POnset'] == 'emergent':
                PLabelString += 'e'
        if not self.dicts[self.stPt].has_key('PPol'):
            PLabelString += '_'
        else:
            if self.dicts[self.stPt]['PPol'] == 'Up':
                PLabelString += 'u'
            elif self.dicts[self.stPt]['PPol'] == 'PoorUp':
                PLabelString += '+'
            elif self.dicts[self.stPt]['PPol'] == 'Down':
                PLabelString += 'd'
            elif self.dicts[self.stPt]['PPol'] == 'PoorDown':
                PLabelString += '-'
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
        PsynthLabelString = 'Psynth: %+.3f' % self.dicts[self.stPt]['Pres']
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
                SLabelString += 'i'
            elif self.dicts[self.stPt]['SOnset'] == 'emergent':
                SLabelString += 'e'
        if not self.dicts[self.stPt].has_key('SPol'):
            SLabelString += '_'
        else:
            if self.dicts[self.stPt]['SPol'] == 'Up':
                SLabelString += 'u'
            elif self.dicts[self.stPt]['SPol'] == 'PoorUp':
                SLabelString += '+'
            elif self.dicts[self.stPt]['SPol'] == 'Down':
                SLabelString += 'd'
            elif self.dicts[self.stPt]['SPol'] == 'PoorDown':
                SLabelString += '-'
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
        SsynthLabelString = 'Ssynth: %+.3f' % self.dicts[self.stPt]['Sres']
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
            print "P Pick deleted"
        except:
            pass
            
    def delPsynth(self):
        try:
            del self.dicts[self.stPt]['Psynth']
            print "synthetic P Pick deleted"
        except:
            pass
            
    def delPWeight(self):
        try:
            del self.dicts[self.stPt]['PWeight']
            print "P Pick weight deleted"
        except:
            pass
            
    def delPPol(self):
        try:
            del self.dicts[self.stPt]['PPol']
            print "P Pick polarity deleted"
        except:
            pass
            
    def delPOnset(self):
        try:
            del self.dicts[self.stPt]['POnset']
            print "P Pick onset deleted"
        except:
            pass
            
    def delPErr1(self):
        try:
            del self.dicts[self.stPt]['PErr1']
            print "PErr1 Pick deleted"
        except:
            pass
            
    def delPErr2(self):
        try:
            del self.dicts[self.stPt]['PErr2']
            print "PErr2 Pick deleted"
        except:
            pass
            
    def delS(self):
        try:
            del self.dicts[self.stPt]['S']
            print "S Pick deleted"
        except:
            pass
            
    def delSsynth(self):
        try:
            del self.dicts[self.stPt]['Ssynth']
            print "synthetic S Pick deleted"
        except:
            pass
            
    def delSWeight(self):
        try:
            del self.dicts[self.stPt]['SWeight']
            print "S Pick weight deleted"
        except:
            pass
            
    def delSPol(self):
        try:
            del self.dicts[self.stPt]['SPol']
            print "S Pick polarity deleted"
        except:
            pass
            
    def delSOnset(self):
        try:
            del self.dicts[self.stPt]['SOnset']
            print "S Pick onset deleted"
        except:
            pass
            
    def delSErr1(self):
        try:
            del self.dicts[self.stPt]['SErr1']
            print "SErr1 Pick deleted"
        except:
            pass
            
    def delSErr2(self):
        try:
            del self.dicts[self.stPt]['SErr2']
            print "SErr2 Pick deleted"
        except:
            pass
            
    def delMagMin1(self):
        try:
            del self.dicts[self.stPt]['MagMin1']
            del self.dicts[self.stPt]['MagMin1T']
            print "Magnitude Minimum Estimation Pick deleted"
        except:
            pass
            
    def delMagMax1(self):
        try:
            del self.dicts[self.stPt]['MagMax1']
            del self.dicts[self.stPt]['MagMax1T']
            print "Magnitude Maximum Estimation Pick deleted"
        except:
            pass
            
    def delMagMin2(self):
        try:
            del self.dicts[self.stPt]['MagMin2']
            del self.dicts[self.stPt]['MagMin2T']
            print "Magnitude Minimum Estimation Pick deleted"
        except:
            pass
            
    def delMagMax2(self):
        try:
            del self.dicts[self.stPt]['MagMax2']
            del self.dicts[self.stPt]['MagMax2T']
            print "Magnitude Maximum Estimation Pick deleted"
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
    
    def addFiltButtons(self):
        #add filter buttons
        self.axFilt = self.fig.add_axes([0.22, 0.02, 0.15, 0.15],frameon=False,axisbg='lightgrey')
        self.check = CheckButtons(self.axFilt, ('Filter','Zero-Phase'),(self.flagFilt,self.flagFiltZPH))
        self.check.on_clicked(self.funcFilt)
        self.axFiltTyp = self.fig.add_axes([0.40, 0.02, 0.15, 0.15],frameon=False,axisbg='lightgrey')
        self.radio = RadioButtons(self.axFiltTyp, ('Bandpass', 'Bandstop', 'Lowpass', 'Highpass'),activecolor='k')
        self.radio.on_clicked(self.funcFiltTyp)
        
    def addPhaseButtons(self):
        #add phase buttons
        self.axPhase = self.fig.add_axes([0.10, 0.02, 0.10, 0.15],frameon=False,axisbg='lightgrey')
        self.radioPhase = RadioButtons(self.axPhase, ('P', 'S', 'Mag'),activecolor='k')
        self.radioPhase.on_clicked(self.funcPhase)
        
    def updateLow(self,val):
        if not self.flagFilt or self.flagFiltTyp == 2:
            return
        else:
            self.updatePlot()
    
    def updateHigh(self,val):
        if not self.flagFilt or self.flagFiltTyp == 3:
            return
        else:
            self.updatePlot()
    
    def delSliders(self):
        self.valFiltLow = self.slideLow.val
        self.valFiltHigh = self.slideHigh.val
        try:
            self.fig.delaxes(self.axLowcut)
            self.fig.delaxes(self.axHighcut)
        except:
            return
    
    def addSliders(self):
        #add filter slider
        self.axLowcut = self.fig.add_axes([0.63, 0.05, 0.30, 0.03], xscale='log')
        self.axHighcut  = self.fig.add_axes([0.63, 0.10, 0.30, 0.03], xscale='log')
        low  = 1.0/ (self.streams[self.stPt][0].stats.npts/float(self.streams[self.stPt][0].stats.sampling_rate))
        high = self.streams[self.stPt][0].stats.sampling_rate/2.0
        self.valFiltLow = max(low,self.valFiltLow)
        self.valFiltHigh = min(high,self.valFiltHigh)
        self.slideLow = Slider(self.axLowcut, 'Lowcut', low, high, valinit=self.valFiltLow, facecolor='darkgrey', edgecolor='k', linewidth=1.7)
        self.slideHigh = Slider(self.axHighcut, 'Highcut', low, high, valinit=self.valFiltHigh, facecolor='darkgrey', edgecolor='k', linewidth=1.7)
        self.slideLow.on_changed(self.updateLow)
        self.slideHigh.on_changed(self.updateHigh)
        
    
    def redraw(self):
        for line in self.multicursor.lines:
            line.set_visible(False)
        self.fig.canvas.draw()
    
    def updatePlot(self):
        filt=[]
        #filter data
        if self.flagFilt==True:
            if self.flagFiltZPH:
                if self.flagFiltTyp==0:
                    for tr in self.streams[self.stPt].traces:
                        filt.append(bandpassZPHSH(tr.data,self.slideLow.val,self.slideHigh.val,df=tr.stats.sampling_rate))
                    print "Zero-Phase Bandpass: %.2f-%.2f Hz"%(self.slideLow.val,self.slideHigh.val)
                if self.flagFiltTyp==1:
                    for tr in self.streams[self.stPt].traces:
                        filt.append(bandstopZPHSH(tr.data,self.slideLow.val,self.slideHigh.val,df=tr.stats.sampling_rate))
                    print "Zero-Phase Bandstop: %.2f-%.2f Hz"%(self.slideLow.val,self.slideHigh.val)
                if self.flagFiltTyp==2:
                    for tr in self.streams[self.stPt].traces:
                        filt.append(lowpassZPHSH(tr.data,self.slideHigh.val,df=tr.stats.sampling_rate))
                    print "Zero-Phase Lowpass: %.2f Hz"%(self.slideHigh.val)
                if self.flagFiltTyp==3:
                    for tr in self.streams[self.stPt].traces:
                        filt.append(highpassZPHSH(tr.data,self.slideLow.val,df=tr.stats.sampling_rate))
                    print "Zero-Phase Highpass: %.2f Hz"%(self.slideLow.val)
            else:
                if self.flagFiltTyp==0:
                    for tr in self.streams[self.stPt].traces:
                        filt.append(bandpass(tr.data,self.slideLow.val,self.slideHigh.val,df=tr.stats.sampling_rate))
                    print "One-Pass Bandpass: %.2f-%.2f Hz"%(self.slideLow.val,self.slideHigh.val)
                if self.flagFiltTyp==1:
                    for tr in self.streams[self.stPt].traces:
                        filt.append(bandstop(tr.data,self.slideLow.val,self.slideHigh.val,df=tr.stats.sampling_rate))
                    print "One-Pass Bandstop: %.2f-%.2f Hz"%(self.slideLow.val,self.slideHigh.val)
                if self.flagFiltTyp==2:
                    for tr in self.streams[self.stPt].traces:
                        filt.append(lowpass(tr.data,self.slideHigh.val,df=tr.stats.sampling_rate))
                    print "One-Pass Lowpass: %.2f Hz"%(self.slideHigh.val)
                if self.flagFiltTyp==3:
                    for tr in self.streams[self.stPt].traces:
                        filt.append(highpass(tr.data,self.slideLow.val,df=tr.stats.sampling_rate))
                    print "One-Pass Highpass: %.2f Hz"%(self.slideLow.val)
            #make new plots
            for i in range(len(self.plts)):
                self.plts[i].set_data(self.t, filt[i])
        else:
            #make new plots
            for i in range(len(self.plts)):
                self.plts[i].set_data(self.t, self.streams[self.stPt][i].data)
            print "Unfiltered Traces"
        # Update all subplots
        self.redraw()
    
    def funcFilt(self, label):
        if label=='Filter':
            self.switch_flagFilt()
            self.updatePlot()
        elif label=='Zero-Phase':
            self.switch_flagFiltZPH()
            if self.flagFilt:
                self.updatePlot()
    
    def funcFiltTyp(self, label):
        self.flagFiltTyp=self.dictFiltTyp[label]
        if self.flagFilt:
            self.updatePlot()
    
    def funcPhase(self, label):
        self.flagPhase=self.dictPhase[label]
        self.pickingColor=self.dictPhaseColors[label]
        for l in self.multicursor.lines:
            l.set_color(self.pickingColor)
        self.radioPhase.circles[self.flagPhase].set_facecolor(self.pickingColor)
        self.redraw()
    
    def funcSwitchPhase(self):
        self.radioPhase.circles[self.flagPhase].set_facecolor(self.axPhase._axisbg)
        self.flagPhase=(self.flagPhase+1)%len(self.dictPhase)
        self.pickingColor=self.dictPhaseColors[self.dictPhaseInverse[self.flagPhase]]
        for l in self.multicursor.lines:
            l.set_color(self.pickingColor)
        self.radioPhase.circles[self.flagPhase].set_facecolor(self.pickingColor)
        self.redraw()
    
    
    
    
    # Define the event that handles the setting of P- and S-wave picks
    def pick(self, event):
        # Set new P Pick
        if self.flagPhase==0 and event.key==self.dictKeybindings['setPick']:
            self.delPLine()
            self.delPLabel()
            self.delPsynthLine()
            self.dicts[self.stPt]['P']=int(round(event.xdata))
            self.drawPLine()
            self.drawPLabel()
            self.drawPsynthLine()
            self.drawPsynthLabel()
            #check if the new P pick lies outside of the Error Picks
            try:
                if self.dicts[self.stPt]['P']<self.dicts[self.stPt]['PErr1']:
                    self.delPErr1Line()
                    self.delPErr1()
            except:
                pass
            try:
                if self.dicts[self.stPt]['P']>self.dicts[self.stPt]['PErr2']:
                    self.delPErr2Line()
                    self.delPErr2()
            except:
                pass
            # Update all subplots
            self.redraw()
            # Console output
            print "P Pick set at %i"%self.dicts[self.stPt]['P']
        # Set P Pick weight
        if self.dicts[self.stPt].has_key('P'):
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPWeight0']:
                self.delPLabel()
                self.dicts[self.stPt]['PWeight']=0
                self.drawPLabel()
                self.redraw()
                print "P Pick weight set to %i"%self.dicts[self.stPt]['PWeight']
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPWeight1']:
                self.delPLabel()
                self.dicts[self.stPt]['PWeight']=1
                print "P Pick weight set to %i"%self.dicts[self.stPt]['PWeight']
                self.drawPLabel()
                self.redraw()
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPWeight2']:
                self.delPLabel()
                self.dicts[self.stPt]['PWeight']=2
                print "P Pick weight set to %i"%self.dicts[self.stPt]['PWeight']
                self.drawPLabel()
                self.redraw()
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPWeight3']:
                self.delPLabel()
                self.dicts[self.stPt]['PWeight']=3
                print "P Pick weight set to %i"%self.dicts[self.stPt]['PWeight']
                self.drawPLabel()
                self.redraw()
        # Set P Pick polarity
        if self.dicts[self.stPt].has_key('P'):
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPPolUp']:
                self.delPLabel()
                self.dicts[self.stPt]['PPol']='Up'
                self.drawPLabel()
                self.redraw()
                print "P Pick polarity set to %s"%self.dicts[self.stPt]['PPol']
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPPolPoorUp']:
                self.delPLabel()
                self.dicts[self.stPt]['PPol']='PoorUp'
                self.drawPLabel()
                self.redraw()
                print "P Pick polarity set to %s"%self.dicts[self.stPt]['PPol']
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPPolDown']:
                self.delPLabel()
                self.dicts[self.stPt]['PPol']='Down'
                self.drawPLabel()
                self.redraw()
                print "P Pick polarity set to %s"%self.dicts[self.stPt]['PPol']
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPPolPoorDown']:
                self.delPLabel()
                self.dicts[self.stPt]['PPol']='PoorDown'
                self.drawPLabel()
                self.redraw()
                print "P Pick polarity set to %s"%self.dicts[self.stPt]['PPol']
        # Set P Pick onset
        if self.dicts[self.stPt].has_key('P'):
            if self.flagPhase == 0 and event.key == self.dictKeybindings['setPOnsetImpulsive']:
                self.delPLabel()
                self.dicts[self.stPt]['POnset'] = 'impulsive'
                self.drawPLabel()
                self.redraw()
                print "P pick onset set to %s" % self.dicts[self.stPt]['POnset']
            elif self.flagPhase == 0 and event.key == self.dictKeybindings['setPOnsetEmergent']:
                self.delPLabel()
                self.dicts[self.stPt]['POnset'] = 'emergent'
                self.drawPLabel()
                self.redraw()
                print "P pick onset set to %s" % self.dicts[self.stPt]['POnset']
        # Set new S Pick
        if self.flagPhase==1 and event.key==self.dictKeybindings['setPick']:
            self.delSLine()
            self.delSLabel()
            self.delSsynthLine()
            self.dicts[self.stPt]['S']=int(round(event.xdata))
            self.drawSLine()
            self.drawSLabel()
            self.drawSsynthLine()
            self.drawSsynthLabel()
            #check if the new S pick lies outside of the Error Picks
            try:
                if self.dicts[self.stPt]['S']<self.dicts[self.stPt]['SErr1']:
                    self.delSErr1Line()
                    self.delSErr1()
            except:
                pass
            try:
                if self.dicts[self.stPt]['S']>self.dicts[self.stPt]['SErr2']:
                    self.delSErr2Line()
                    self.delSErr2()
            except:
                pass
            # Update all subplots
            self.redraw()
            # Console output
            print "S Pick set at %i"%self.dicts[self.stPt]['S']
        # Set S Pick weight
        if self.dicts[self.stPt].has_key('S'):
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSWeight0']:
                self.delSLabel()
                self.dicts[self.stPt]['SWeight']=0
                self.drawSLabel()
                self.redraw()
                print "S Pick weight set to %i"%self.dicts[self.stPt]['SWeight']
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSWeight1']:
                self.delSLabel()
                self.dicts[self.stPt]['SWeight']=1
                self.drawSLabel()
                self.redraw()
                print "S Pick weight set to %i"%self.dicts[self.stPt]['SWeight']
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSWeight2']:
                self.delSLabel()
                self.dicts[self.stPt]['SWeight']=2
                self.drawSLabel()
                self.redraw()
                print "S Pick weight set to %i"%self.dicts[self.stPt]['SWeight']
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSWeight3']:
                self.delSLabel()
                self.dicts[self.stPt]['SWeight']=3
                self.drawSLabel()
                self.redraw()
                print "S Pick weight set to %i"%self.dicts[self.stPt]['SWeight']
        # Set S Pick polarity
        if self.dicts[self.stPt].has_key('S'):
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSPolUp']:
                self.delSLabel()
                self.dicts[self.stPt]['SPol']='Up'
                self.drawSLabel()
                self.redraw()
                print "S Pick polarity set to %s"%self.dicts[self.stPt]['SPol']
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSPolPoorUp']:
                self.delSLabel()
                self.dicts[self.stPt]['SPol']='PoorUp'
                self.drawSLabel()
                self.redraw()
                print "S Pick polarity set to %s"%self.dicts[self.stPt]['SPol']
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSPolDown']:
                self.delSLabel()
                self.dicts[self.stPt]['SPol']='Down'
                self.drawSLabel()
                self.redraw()
                print "S Pick polarity set to %s"%self.dicts[self.stPt]['SPol']
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSPolPoorDown']:
                self.delSLabel()
                self.dicts[self.stPt]['SPol']='PoorDown'
                self.drawSLabel()
                self.redraw()
                print "S Pick polarity set to %s"%self.dicts[self.stPt]['SPol']
        # Set S Pick onset
        if self.dicts[self.stPt].has_key('S'):
            if self.flagPhase == 0 and event.key == self.dictKeybindings['setSOnsetImpulsive']:
                self.delSLabel()
                self.dicts[self.stPt]['SOnset'] = 'impulsive'
                self.drawSLabel()
                self.redraw()
                print "S pick onset set to %s" % self.dicts[self.stPt]['SOnset']
            elif self.flagPhase == 0 and event.key == self.dictKeybindings['setSOnsetEmergent']:
                self.delSLabel()
                self.dicts[self.stPt]['SOnset'] = 'emergent'
                self.drawSLabel()
                self.redraw()
                print "S pick onset set to %s" % self.dicts[self.stPt]['SOnset']
        # Remove P Pick
        if self.flagPhase==0 and event.key==self.dictKeybindings['delPick']:
            # Try to remove all existing Pick lines and P Pick variable
            self.delPLine()
            self.delP()
            self.delPWeight()
            self.delPPol()
            self.delPOnset()
            self.delPLabel()
            # Try to remove existing Pick Error 1 lines and variable
            self.delPErr1Line()
            self.delPErr1()
            # Try to remove existing Pick Error 2 lines and variable
            self.delPErr2Line()
            self.delPErr2()
            # Update all subplots
            self.redraw()
        # Remove S Pick
        if self.flagPhase==1 and event.key==self.dictKeybindings['delPick']:
            # Try to remove all existing Pick lines and P Pick variable
            self.delSLine()
            self.delS()
            self.delSWeight()
            self.delSPol()
            self.delSOnset()
            self.delSLabel()
            # Try to remove existing Pick Error 1 lines and variable
            self.delSErr1Line()
            self.delSErr1()
            # Try to remove existing Pick Error 2 lines and variable
            self.delSErr2Line()
            self.delSErr2()
            # Update all subplots
            self.redraw()
        # Set new P Pick uncertainties
        if self.flagPhase==0 and event.key==self.dictKeybindings['setPickError']:
            # Set Flag to determine scenario
            try:
                # Set left Error Pick
                if event.xdata<self.dicts[self.stPt]['P']:
                    errFlag=1
                # Set right Error Pick
                else:
                    errFlag=2
            # Set no Error Pick (no P Pick yet)
            except:
                errFlag=0
            # Case 1
            if errFlag==1:
                # Define global variables seen outside
                # Remove old lines from the plot before plotting the new ones
                self.delPErr1Line()
                # Save sample value of error pick (round to integer sample value)
                self.dicts[self.stPt]['PErr1']=int(round(event.xdata))
                # Plot the lines for the P Error pick in all three traces
                self.drawPErr1Line()
                # Update all subplots
                self.redraw()
                # Console output
                print "P Error Pick 1 set at %i"%self.dicts[self.stPt]['PErr1']
            # Case 2
            if errFlag==2:
                # Define global variables seen outside
                # Remove old lines from the plot before plotting the new ones
                self.delPErr2Line()
                # Save sample value of error pick (round to integer sample value)
                self.dicts[self.stPt]['PErr2']=int(round(event.xdata))
                # Plot the lines for the P Error pick in all three traces
                self.drawPErr2Line()
                # Update all subplots
                self.redraw()
                # Console output
                print "P Error Pick 2 set at %i"%self.dicts[self.stPt]['PErr2']
        # Set new S Pick uncertainties
        if self.flagPhase==1 and event.key==self.dictKeybindings['setPickError']:
            # Set Flag to determine scenario
            try:
                # Set left Error Pick
                if event.xdata<self.dicts[self.stPt]['S']:
                    errFlag=1
                # Set right Error Pick
                else:
                    errFlag=2
            # Set no Error Pick (no S Pick yet)
            except:
                errFlag=0
            # Case 1
            if errFlag==1:
                # Define global variables seen outside
                # Remove old lines from the plot before plotting the new ones
                self.delSErr1Line()
                # Save sample value of error pick (round to integer sample value)
                self.dicts[self.stPt]['SErr1']=int(round(event.xdata))
                # Plot the lines for the S Error pick in all three traces
                self.drawSErr1Line()
                # Update all subplots
                self.redraw()
                # Console output
                print "S Error Pick 1 set at %i"%self.dicts[self.stPt]['SErr1']
            # Case 2
            if errFlag==2:
                # Define global variables seen outside
                # Remove old lines from the plot before plotting the new ones
                self.delSErr2Line()
                # Save sample value of error pick (round to integer sample value)
                self.dicts[self.stPt]['SErr2']=int(round(event.xdata))
                # Plot the lines for the S Error pick in all three traces
                self.drawSErr2Line()
                # Update all subplots
                self.redraw()
                # Console output
                print "S Error Pick 2 set at %i"%self.dicts[self.stPt]['SErr2']
        # Magnitude estimation picking:
        if self.flagPhase==2 and event.key==self.dictKeybindings['setMagMin'] and len(self.axs) > 2:
            if event.inaxes == self.axs[1]:
                self.delMagMinCross1()
                xpos=int(event.xdata)
                ydata=event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
                cutoffSamples=xpos-self.magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
                self.dicts[self.stPt]['MagMin1']=np.min(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                self.dicts[self.stPt]['MagMin1T']=cutoffSamples+np.argmin(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                #delete old MagMax Pick, if new MagMin Pick is higher
                try:
                    if self.dicts[self.stPt]['MagMin1'] > self.dicts[self.stPt]['MagMax1']:
                        self.delMagMaxCross1()
                        self.delMagMax1()
                except:
                    pass
                self.drawMagMinCross1()
                self.redraw()
                print "Minimum for magnitude estimation set: %s at %s"%(self.dicts[self.stPt]['MagMin1'],self.dicts[self.stPt]['MagMin1T'])
            elif event.inaxes == self.axs[2]:
                self.delMagMinCross2()
                xpos=int(event.xdata)
                ydata=event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
                cutoffSamples=xpos-self.magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
                self.dicts[self.stPt]['MagMin2']=np.min(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                self.dicts[self.stPt]['MagMin2T']=cutoffSamples+np.argmin(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                #delete old MagMax Pick, if new MagMin Pick is higher
                try:
                    if self.dicts[self.stPt]['MagMin2'] > self.dicts[self.stPt]['MagMax2']:
                        self.delMagMaxCross2()
                        self.delMagMax2()
                except:
                    pass
                self.drawMagMinCross2()
                self.redraw()
                print "Minimum for magnitude estimation set: %s at %s"%(self.dicts[self.stPt]['MagMin2'],self.dicts[self.stPt]['MagMin2T'])
        if self.flagPhase==2 and event.key==self.dictKeybindings['setMagMax'] and len(self.axs) > 2:
            if event.inaxes == self.axs[1]:
                self.delMagMaxCross1()
                xpos=int(event.xdata)
                ydata=event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
                cutoffSamples=xpos-self.magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
                self.dicts[self.stPt]['MagMax1']=np.max(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                self.dicts[self.stPt]['MagMax1T']=cutoffSamples+np.argmax(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                #delete old MagMax Pick, if new MagMax Pick is higher
                try:
                    if self.dicts[self.stPt]['MagMin1'] > self.dicts[self.stPt]['MagMax1']:
                        self.delMagMinCross1()
                        self.delMagMin1()
                except:
                    pass
                self.drawMagMaxCross1()
                self.redraw()
                print "Maximum for magnitude estimation set: %s at %s"%(self.dicts[self.stPt]['MagMax1'],self.dicts[self.stPt]['MagMax1T'])
            elif event.inaxes == self.axs[2]:
                self.delMagMaxCross2()
                xpos=int(event.xdata)
                ydata=event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
                cutoffSamples=xpos-self.magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
                self.dicts[self.stPt]['MagMax2']=np.max(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                self.dicts[self.stPt]['MagMax2T']=cutoffSamples+np.argmax(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                #delete old MagMax Pick, if new MagMax Pick is higher
                try:
                    if self.dicts[self.stPt]['MagMin2'] > self.dicts[self.stPt]['MagMax2']:
                        self.delMagMinCross2()
                        self.delMagMin2()
                except:
                    pass
                self.drawMagMaxCross2()
                self.redraw()
                print "Maximum for magnitude estimation set: %s at %s"%(self.dicts[self.stPt]['MagMax2'],self.dicts[self.stPt]['MagMax2T'])
        if self.flagPhase == 2 and event.key == self.dictKeybindings['delMagMinMax']:
            if event.inaxes == self.axs[1]:
                self.delMagMaxCross1()
                self.delMagMinCross1()
                self.delMagMin1()
                self.delMagMax1()
            elif event.inaxes == self.axs[2]:
                self.delMagMaxCross2()
                self.delMagMinCross2()
                self.delMagMin2()
                self.delMagMax2()
            else:
                return
            self.redraw()
    
    # Define zoom events for the mouse scroll wheel
    def zoom(self,event):
        # Zoom in on scroll-up
        if event.button=='up' and self.flagWheelZoom:
            # Calculate and set new axes boundaries from old ones
            (left,right)=self.axs[0].get_xbound()
            left+=(event.xdata-left)/2
            right-=(right-event.xdata)/2
            self.axs[0].set_xbound(lower=left,upper=right)
            # Update all subplots
            self.redraw()
        # Zoom out on scroll-down
        if event.button=='down' and self.flagWheelZoom:
            # Calculate and set new axes boundaries from old ones
            (left,right)=self.axs[0].get_xbound()
            left-=(event.xdata-left)/2
            right+=(right-event.xdata)/2
            self.axs[0].set_xbound(lower=left,upper=right)
            # Update all subplots
            self.redraw()
    
    # Define zoom reset for the mouse button 2 (always scroll wheel!?)
    def zoom_reset(self,event):
        if event.button==2:
            # Use Z trace limits as boundaries
            self.axs[0].set_xbound(lower=self.xMin,upper=self.xMax)
            self.axs[0].set_ybound(lower=self.yMin,upper=self.yMax)
            # Update all subplots
            self.redraw()
            print "Resetting axes"
    
    def switchWheelZoom(self,event):
        if event.key==self.dictKeybindings['switchWheelZoom']:
            self.flagWheelZoom=not self.flagWheelZoom
            if self.flagWheelZoom:
                print "Mouse wheel zooming activated"
            else:
                print "Mouse wheel zooming deactivated"
    
    def switchPan(self,event):
        if event.key==self.dictKeybindings['switchPan']:
            self.fig.canvas.toolbar.pan()
            self.fig.canvas.widgetlock.release(self.fig.canvas.toolbar)
            self.redraw()
            print "Switching pan mode"
    
    #lookup multicursor source: http://matplotlib.sourcearchive.com/documentation/0.98.1/widgets_8py-source.html
    def multicursorReinit(self):
        self.fig.canvas.mpl_disconnect(self.multicursor.id1)
        self.fig.canvas.mpl_disconnect(self.multicursor.id2)
        self.multicursor.__init__(self.fig.canvas,self.axs, useblit=True, color='black', linewidth=1, ls='dotted')
        #fig.canvas.draw_idle()
        #multicursor._update()
        #multicursor.needclear=True
        #multicursor.background = fig.canvas.copy_from_bbox(fig.canvas.figure.bbox)
        #fig.canvas.restore_region(multicursor.background)
        #fig.canvas.blit(fig.canvas.figure.bbox)
        for l in self.multicursor.lines:
            l.set_color(self.pickingColor)
    
    def switchPhase(self, event):
        if event.key==self.dictKeybindings['switchPhase']:
            self.funcSwitchPhase()
            print "Switching Phase button"
            
    def switchStream(self, event):
        if event.key==self.dictKeybindings['prevStream']:
            self.stPt=(self.stPt-1)%self.stNum
            self.delAxes()
            self.drawAxes()
            self.drawSavedPicks()
            self.delSliders()
            self.addSliders()
            self.multicursorReinit()
            self.updatePlot()
            print "Going to previous stream"
        if event.key==self.dictKeybindings['nextStream']:
            self.stPt=(self.stPt+1)%self.stNum
            self.delAxes()
            self.drawAxes()
            self.drawSavedPicks()
            self.delSliders()
            self.addSliders()
            self.multicursorReinit()
            self.updatePlot()
            print "Going to next stream"
            
    def blockRedraw(self, event):
        if event.button==1 or event.button==3:
            self.multicursor.visible=False
            self.fig.canvas.widgetlock(self.fig.canvas.toolbar)
            
    def allowRedraw(self, event):
        if event.button==1 or event.button==3:
            self.multicursor.visible=True
            self.fig.canvas.widgetlock.release(self.fig.canvas.toolbar)
    
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
                        continue
                    else:
                        # phSamps is the number of samples after the stream-
                        # starttime at which the time of the synthetic phase
                        # is located
                        phSamps = phUTCTime - self.streams[i][0].stats.starttime
                        phSamps = int(round(phSamps *
                                            self.streams[i][0].stats.sampling_rate))
                        if phType == 'P':
                            self.dicts[i]['Psynth'] = phSamps
                            self.dicts[i]['Pres'] = phResid
                        elif phType == 'S':
                            self.dicts[i]['Ssynth'] = phSamps
                            self.dicts[i]['Sres'] = phResid
        self.drawPsynthLine()
        self.drawPsynthLabel()
        self.drawSsynthLine()
        self.drawSsynthLabel()
        self.redraw()

    def do3dLoc(self):
        self.xmlEventID = '%i' % time.time()
        f = open(self.threeDlocInfile, 'w')
        network = "BW"
        fmt = "%04s  %s        %s %5.3f -999.0 0.000 -999. 0.000 T__DR_ %9.6f %9.6f %8.6f\n"
        self.coords = []
        for i in range(len(self.streams)):
            #lon, lat, ele = getCoord(network, self.stationlist[i])
            lon = self.dicts[i]['StaLon']
            lat = self.dicts[i]['StaLat']
            ele = self.dicts[i]['StaEle']
            self.coords.append([lon, lat])
            if self.dicts[i].has_key('P'):
                t = self.streams[i][0].stats.starttime
                t += self.dicts[i]['P'] / self.streams[i][0].stats.sampling_rate
                date = t.strftime("%Y %m %d %H %M %S")
                date += ".%03d" % (t.microsecond / 1e3 + 0.5)
                delta = self.dicts[i]['PErr2'] - self.dicts[i]['PErr1']
                delta /= self.streams[i][0].stats.sampling_rate
                f.write(fmt % (self.stationlist[i], 'P', date, delta,
                               lon, lat, ele / 1e3))
            if self.dicts[i].has_key('S'):
                t = self.streams[i][0].stats.starttime
                t += self.dicts[i]['S'] / self.streams[i][0].stats.sampling_rate
                date = t.strftime("%Y %m %d %H %M %S")
                date += ".%03d" % (t.microsecond / 1e3 + 0.5)
                delta = self.dicts[i]['SErr2'] - self.dicts[i]['SErr1']
                delta /= self.streams[i][0].stats.sampling_rate
                f.write(fmt % (self.stationlist[i], 'S', date, delta,
                               lon, lat, ele / 1e3))
        f.close()
        self.cat3dlocIn()
        subprocess.call("D3_VELOCITY=/scratch/rh_vel/vp_5836/ D3_VELOCITY_2=/scratch/rh_vel/vs_32220/ 3dloc_pitsa", shell = True)
        print '--> 3dloc finished'
        self.cat3dlocOut()
        self.load3dlocSyntheticPhases()
        self.redraw()

    def cat3dlocIn(self):
        lines = open(self.threeDlocInfile).readlines()
        for line in lines:
            print line.strip()

    def cat3dlocOut(self):
        lines = open(self.threeDlocOutfile).readlines()
        for line in lines:
            print line.strip()

    def load3dlocData(self):
        #self.load3dlocSyntheticPhases()
        event = open(self.threeDlocOutfile).readline().split()
        self.threeDlocEventLon = float(event[8])
        self.threeDlocEventLat = float(event[9])
        self.threeDlocEventErrX = float(event[11])
        self.threeDlocEventErrY = float(event[12])
        self.threeDlocEventTime = UTCDateTime(int(event[2]), int(event[3]),
                                              int(event[4]), int(event[5]),
                                              int(event[6]), float(event[7]))
        #XXX aufraeumen!! Die meisten Listen hier werden nicht mehr gebraucht
        self.threeDlocPLons = []
        self.threeDlocPLats = []
        self.threeDlocSLons = []
        self.threeDlocSLats = []
        self.threeDlocPNames = []
        self.threeDlocSNames = []
        self.threeDlocPResInfo = []
        self.threeDlocSResInfo = []
        lines = open(self.threeDlocInfile).readlines()
        for line in lines:
            pick = line.split()
            for i in range(len(self.streams)):
                if pick[0].strip() == self.streams[i][0].stats.station.strip():
                    if pick[1] == 'P':
                        self.dicts[i]['3DlocPLon'] = float(pick[14])
                        self.dicts[i]['3DlocPLat'] = float(pick[15])
                    elif pick[1] == 'S':
                        self.dicts[i]['3DlocSLon'] = float(pick[14])
                        self.dicts[i]['3DlocSLat'] = float(pick[15])
                    break
        lines = open(self.threeDlocOutfile).readlines()
        for line in lines[1:]:
            pick = line.split()
            for i in range(len(self.streams)):
                if pick[0].strip() == self.streams[i][0].stats.station.strip():
                    if pick[1] == 'P':
                        self.dicts[i]['3DlocPResInfo'] = '\n\n %+0.3fs' % float(pick[8])
                    elif pick[1] == 'S':
                        self.dicts[i]['3DlocSResInfo'] = '\n\n\n %+0.3fs' % float(pick[8])
                    break
    
    def updateNetworkMag(self):
        count = 0
        self.netMag = 0
        for i in range(len(self.streams)):
            if self.dicts[i]['MagUse'] and self.dicts[i].has_key('Mag'):
                count += 1
                self.netMag += self.dicts[i]['Mag']
        if count == 0:
            self.netMag = np.NaN
        else:
            self.netMag /= count
        self.netMagLabel = '\n\n\n  %.2f' % self.netMag
        try:
            self.netMagText.set_text(self.netMagLabel)
        except:
            pass

    def show3dlocEventMap(self):
        self.load3dlocData()
        self.updateNetworkMag()
        #print self.dicts[0]
        self.fig3dloc = plt.figure()
        self.ax3dloc = self.fig3dloc.add_subplot(111)
        self.ax3dloc.scatter([self.threeDlocEventLon], [self.threeDlocEventLat],
                             30, color = 'red', marker = 'o')
        errLon, errLat = utlLonLat(self.threeDlocEventLon, self.threeDlocEventLat,
                               self.threeDlocEventErrX, self.threeDlocEventErrY)
        errLon -= self.threeDlocEventLon
        errLat -= self.threeDlocEventLat
        self.ax3dloc.text(self.threeDlocEventLon, self.threeDlocEventLat,
                          ' %2.3f +/- %0.2fkm\n %2.3f +/- %0.2fkm' % (self.threeDlocEventLon,
                          self.threeDlocEventErrX, self.threeDlocEventLat,
                          self.threeDlocEventErrY), va = 'top',
                          family = 'monospace')
        self.netMagText = self.ax3dloc.text(self.threeDlocEventLon, self.threeDlocEventLat,
                          self.netMagLabel,
                          va = 'top',
                          color = 'green',
                          family = 'monospace')
        errorell = Ellipse(xy = [self.threeDlocEventLon, self.threeDlocEventLat],
                      width = errLon, height = errLat, angle = 0, fill = False)
        self.ax3dloc.add_artist(errorell)
        self.scatterMagIndices = []
        self.scatterMagLon = []
        self.scatterMagLat = []
        for i in range(len(self.streams)):
            if self.dicts[i].has_key('3DlocPLon'):
                self.ax3dloc.scatter([self.dicts[i]['3DlocPLon']], [self.dicts[i]['3DlocPLat']], s = 150,
                                     marker = 'v', color = '', edgecolor = 'black')
                self.ax3dloc.text(self.dicts[i]['StaLon'], self.dicts[i]['StaLat'],
                                  '  ' + self.dicts[i]['Station'], va = 'top',
                                  family = 'monospace')
                self.ax3dloc.text(self.dicts[i]['StaLon'], self.dicts[i]['StaLat'],
                                  self.dicts[i]['3DlocPResInfo'], va = 'top',
                                  family = 'monospace',
                                  color = self.dictPhaseColors['P'])
            if self.dicts[i].has_key('3DlocSLon'):
                self.ax3dloc.scatter([self.dicts[i]['StaLon']], [self.dicts[i]['StaLat']], s = 150,
                                     marker = 'v', color = '', edgecolor = 'black')
                self.ax3dloc.text(self.dicts[i]['StaLon'], self.dicts[i]['StaLat'],
                                  '  ' + self.dicts[i]['Station'], va = 'top',
                                  family = 'monospace')
                self.ax3dloc.text(self.dicts[i]['StaLon'], self.dicts[i]['StaLat'],
                                  self.dicts[i]['3DlocSResInfo'], va = 'top',
                                  family = 'monospace',
                                  color = self.dictPhaseColors['S'])
            if self.dicts[i].has_key('Mag'):
                self.scatterMagIndices.append(i)
                self.scatterMagLon.append(self.dicts[i]['StaLon'])
                self.scatterMagLat.append(self.dicts[i]['StaLat'])
                self.ax3dloc.text(self.dicts[i]['StaLon'], self.dicts[i]['StaLat'],
                                  '  ' + self.dicts[i]['Station'], va = 'top',
                                  family = 'monospace')
                self.ax3dloc.text(self.dicts[i]['StaLon'], self.dicts[i]['StaLat'],
                                  '\n\n\n\n  %s' % self.dicts[i]['Mag'], va = 'top',
                                  family = 'monospace',
                                  color = self.dictPhaseColors['Mag'])
            self.scatterMag = self.ax3dloc.scatter(self.scatterMagLon, self.scatterMagLat, s = 150,
                                     marker = 'v', color = '', edgecolor = 'black', picker = 10)
                
        #if len(self.threeDlocSNames) > 0:
        #    #self.ax3dloc.scatter(self.threeDlocSLons, self.threeDlocSLats, s = 440,
        #    #                     color = self.dictPhaseColors['S'], marker = 'v',
        #    #                     edgecolor = 'black')
        #    for i in range(len(self.threeDlocSNames)):
        #        self.ax3dloc.scatter(self.threeDlocSLons, self.threeDlocSLats, s = 150,
        #                             marker = 'v', color = '', edgecolor = 'black')
        #        #self.ax3dloc.text(self.threeDlocSLons[i], self.threeDlocSLats[i],
        #        #                  '  ' + self.threeDlocSNames[i], va = 'top')
        #        self.ax3dloc.text(self.threeDlocSLons[i], self.threeDlocSLats[i],
        #                          self.threeDlocSNames[i], va = 'top',
        #                          family = 'monospace')
        #        self.ax3dloc.text(self.threeDlocSLons[i], self.threeDlocSLats[i],
        #                          self.threeDlocSResInfo[i], va = 'top',
        #                          family = 'monospace',
        #                          color = self.dictPhaseColors['S'])
        #if len(self.threeDlocPNames) > 0:
        #    #self.ax3dloc.scatter(self.threeDlocPLons, self.threeDlocPLats, s = 180,
        #    #                     color = self.dictPhaseColors['P'], marker = 'v',
        #    #                     edgecolor = 'black')
        #    for i in range(len(self.threeDlocPNames)):
        #        self.ax3dloc.scatter(self.threeDlocPLons, self.threeDlocPLats, s = 150,
        #                             marker = 'v', color = '', edgecolor = 'black', picker = 10)
        #        #self.ax3dloc.scatter(self.threeDlocPLons, self.threeDlocPLats, s = 150,
        #        #                              marker = 'v', color = '', edgecolor = 'black', picker = True)
        #        #self.ax3dloc.text(self.threeDlocPLons[i], self.threeDlocPLats[i],
        #        #                  '  ' + self.threeDlocPNames[i], va = 'top')
        #        self.ax3dloc.text(self.threeDlocPLons[i], self.threeDlocPLats[i],
        #                          self.threeDlocPNames[i], va = 'top',
        #                          family = 'monospace')
        #        self.ax3dloc.text(self.threeDlocPLons[i], self.threeDlocPLats[i],
        #                          self.threeDlocPResInfo[i], va = 'top',
        #                          family = 'monospace',
        #                          color = self.dictPhaseColors['P'])
        self.ax3dloc.set_xlabel('Longitude')
        self.ax3dloc.set_ylabel('Latitude')
        self.ax3dloc.set_title(self.threeDlocEventTime)
        lines = open(self.threeDlocOutfile).readlines()
        infoEvent = lines[0].rstrip()
        infoPicks = ''
        for line in lines[1:]:
            infoPicks += line
        self.ax3dloc.text(0.02, 0.95, infoEvent, transform = self.ax3dloc.transAxes,
                          fontsize = 12, verticalalignment = 'top',
                          family = 'monospace')
        self.ax3dloc.text(0.02, 0.90, infoPicks, transform = self.ax3dloc.transAxes,
                          fontsize = 10, verticalalignment = 'top',
                          family = 'monospace')
        self.fig3dloc.canvas.mpl_connect('pick_event', self.selectMagnitudes)
        self.scatterMag.set_facecolors(self.eventMapColors)
        plt.show()

    def selectMagnitudes(self, event):
        if event.artist != self.scatterMag:
            return
        i = self.scatterMagIndices[event.ind[0]]
        j = event.ind[0]
        self.dicts[i]['MagUse'] = not self.dicts[i]['MagUse']
        print event.ind[0]
        print i
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
        self.fig3dloc.canvas.draw()

    def threeDLoc2XML(self):
        """
        Returns output of 3dloc as xml file
        """
        xml =  Element("event")
        Sub(Sub(xml, "event_id"), "value").text = self.xmlEventID
        Sub(Sub(xml, "event_type"), "value").text = "manual"
        
        # we save P picks on Z-component and S picks on N-component
        # XXX standard values for unset keys!!!???!!!???
        for i in range(len(self.streams)):
            if self.dicts[i].has_key('P'):
                pick = Sub(xml, "pick")
                wave = Sub(pick, "waveform")
                wave.set("networkCode", self.streams[i][0].stats.network) 
                wave.set("stationCode", self.streams[i][0].stats.station) 
                wave.set("channelCode", self.streams[i][0].stats.channel) 
                wave.set("locationCode", "") 
                date = Sub(pick, "time")
                # prepare time of pick
                picktime = self.streams[i][0].stats.starttime
                picktime += (self.dicts[i]['P'] /
                             self.streams[i][0].stats.sampling_rate)
                Sub(date, "value").text = (picktime.isoformat() + '.' +
                                           picktime.microsecond)
                Sub(date, "uncertainty") #XXX what does this line mean???
                Sub(pick, "phaseHint").text = "P"
                if self.dicts[i]['POnset'] == 'impulsive':
                    Sub(pick, "onset").text = 'impulsive'
                elif self.dicts[i]['POnset'] == 'emergent':
                    Sub(pick, "onset").text = 'emergent'
                else:
                    Sub(pick, "onset").text = ''
                if self.dicts[i]['PPol'] == 'Up' or self.dicts[i]['PPol'] == 'PoorUp':
                    Sub(pick, "polarity").text = 'positiv'
                elif self.dicts[i]['PPol'] == 'Down' or self.dicts[i]['PPol'] == 'PoorDown':
                    Sub(pick, "polarity").text = 'negativ'
                else:
                    Sub(pick, "polarity").text = ''
                if self.dicts[i].has_key('PWeight'):
                    Sub(pick, "weight").text = '%i' % self.dicts[i]['PWeight']
                else:
                    Sub(pick, "weight").text = ''
                Sub(Sub(pick, "min_amp"), "value").text = "0.00000" #XXX what is min_amp???
                Sub(pick, "phase_compu").text = "IPU0"
                Sub(Sub(pick, "phase_res"), "value").text = "0.17000"
                Sub(Sub(pick, "phase_weight"), "value").text = "1.00000"
                Sub(Sub(pick, "phase_delay"), "value").text = "0.00000"
                Sub(Sub(pick, "azimuth"), "value").text = "1.922043"
                Sub(Sub(pick, "incident"), "value").text = "96.00000"
                Sub(Sub(pick, "epi_dist"), "value").text = "44.938843"
                Sub(Sub(pick, "hyp_dist"), "value").text = "45.30929"

        origin = Sub(xml, "origin")
        date = Sub(origin, "time")
        Sub(date, "value").text = "2010-02-09T19:19:13.550"
        Sub(date, "uncertainty")
        lat = Sub(origin, "latitude")
        Sub(lat, "value").text = "50.579498"
        Sub(lat, "uncertainty").text = "2.240000"
        lon = Sub(origin, "latitude")
        Sub(lon, "value").text = "12.243500"
        Sub(lon, "uncertainty").text = "2.240000"
        depth = Sub(origin, "latitude")
        Sub(depth, "value").text = "12.243500"
        Sub(depth, "uncertainty").text = "2.240000"
        Sub(origin, "depth_type").text = "from location program"
        Sub(origin, "earth_mod").text = "VOG"
        Sub(origin, "originUncertainty")
        quality = Sub(origin, "originQuality")
        Sub(quality, "P_usedPhaseCount").text = "7"
        Sub(quality, "S_usedPhaseCount").text = "7"
        Sub(quality, "usedPhaseCount").text = "14"
        Sub(quality, "associatedPhaseCount").text = "14"
        Sub(quality, "associatedStationCount").text = "14"
        Sub(quality, "depthPhaseCount").text = "0"
        Sub(quality, "standardError").text = "0.170000"
        Sub(quality, "secondaryAzimuthalGap").text = "343.00000"
        Sub(quality, "groundTruthLevel")
        Sub(quality, "minimumDistance").text = "45.309029"
        Sub(quality, "maximumDistance").text = "90.594482"
        Sub(quality, "medianDistance").text = "64.749686"
        magnitude = Sub(xml, "magnitude")
        mag = Sub(magnitude, "mag")
        Sub(mag, "value").text = "1.823500"
        Sub(mag, "uncertainty").text = "0.1000000"
        Sub(magnitude, "type").text = "Ml"
        Sub(magnitude, "stationCount").text = "14"
        return tostring(xml,pretty_print=True,xml_declaration=True)


    def uploadSeishub(self):
        """
        Upload xml file to seishub
        """
        userid = "admin"
        passwd = "admin"

        auth = 'Basic ' + (base64.encodestring(userid + ':' + passwd)).strip()

        servername = 'teide:8080'
        path = '/xml/seismology/event'

        data = self.threedLoc2XML()
        #XXX remove later
        self.xmlEventID = '%i' % 1265906465.2780671
        name = "baynet_%s" % (self.xmlEventID) #XXX id of the file

        #construct and send the header
        webservice = httplib.HTTP(servername)
        webservice.putrequest("PUT", path + '/' + name)
        webservice.putheader('Authorization', auth )
        webservice.putheader("Host", "localhost")
        webservice.putheader("User-Agent", "pickingGUI.py")
        webservice.putheader("Content-type", "text/xml; charset=\"UTF-8\"")
        webservice.putheader("Content-length", "%d" % len(data))
        webservice.endheaders()
        webservice.send(data)

        # get the response
        statuscode, statusmessage, header = webservice.getreply()
        if statuscode!=201:
           print "Server: ", servername, path
           print "Response: ", statuscode, statusmessage
           print "Headers: ", header


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
                      help="Ids to retrieve, e.g. "
                           "'BW.RJOB..EH*,BW.RMOA..EH*'",
                      default='BW.RJOB..EH*,BW.RMOA..EH*')
    parser.add_option("-l", "--local", action="store_true", dest="local",
                      default=False,
                      help="use local files for design purposes")
    #parser.add_option("-k", "--keys", action="store_true", dest="keybindings",
    #                  default=False, help="Show keybindings and quit")
    (options, args) = parser.parse_args()
    for req in ['-d','-t','-i']:
        if not getattr(parser.values,parser.get_option(req).dest):
            parser.print_help()
            return
    
    #if options.keybindings:
    #    PickingGUI()
    #    for i in self.dictKeybindings.items():
    #        print i
    #    return

    if options.local:
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
        #streams=[]
        #streams.append(read('RJOB_061005_072159.ehz.new'))
        #streams[0].append(read('RJOB_061005_072159.ehn.new')[0])
        #streams[0].append(read('RJOB_061005_072159.ehe.new')[0])
        #streams.append(read('RNON_160505_000059.ehz.new'))
        #streams.append(read('RMOA_160505_014459.ehz.new'))
        #streams[2].append(read('RMOA_160505_014459.ehn.new')[0])
        #streams[2].append(read('RMOA_160505_014459.ehe.new')[0])
    else:
        try:
            t = UTCDateTime(options.time)
            client = Client()
            streams = []
            for id in options.ids.split(","):
                net, sta, loc, cha = id.split(".")
                st = client.waveform.getWaveform(net, sta, loc, cha, 
                                                 t, t + options.duration)
                st.sort()
                st.reverse()
                streams.append(st)
        except:
            parser.print_help()
            raise

    PickingGUI(streams)

if __name__ == "__main__":
    main()
