#!/usr/bin/env python

#check for textboxes and other stuff:
#http://code.enthought.com/projects/traits/docs/html/tutorials/traits_ui_scientific_app.html

import matplotlib
#matplotlib.use('gtkagg')

from obspy.core import read
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.widgets import MultiCursor as mplMultiCursor
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

from optparse import OptionParser
from obspy.seishub import Client
from obspy.core import UTCDateTime
from obspy.signal.filter import bandpass,bandpassZPHSH,bandstop,bandstopZPHSH,lowpass,lowpassZPHSH,highpass,highpassZPHSH

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
    
class PickingGUI:

    def __init__(self, streams = None):
        self.streams = streams
        #Define some flags, dictionaries and plotting options
        self.flagFilt=False #False:no filter  True:filter
        self.flagFiltTyp=0 #0: bandpass 1: bandstop 2:lowpass 3: highpass
        self.dictFiltTyp={'Bandpass':0, 'Bandstop':1, 'Lowpass':2, 'Highpass':3}
        self.flagFiltZPH=True #False: no zero-phase True: zero-phase filtering
        self.valFiltLow=np.NaN # These are overridden with low/high estimated from sampling rate
        self.valFiltHigh=np.NaN
        self.flagWheelZoom=True #Switch use of mousewheel for zooming
        self.flagPhase=0 #0:P 1:S 2:Magnitude
        self.dictPhase={'P':0, 'S':1, 'Mag':2}
        self.dictPhaseInverse = {} # We need the reverted dictionary for switching throug the Phase radio button
        for i in self.dictPhase.items():
            self.dictPhaseInverse[i[1]] = i[0]
        self.dictPhaseColors={'P':'red', 'S':'blue', 'Mag':'green'}
        self.pickingColor = self.dictPhaseColors['P']
        self.magPickWindow=10 #Estimating the maximum/minimum in a sample-window around click
        self.magMinMarker='x'
        self.magMaxMarker='x'
        self.magMarkerEdgeWidth=1.8
        self.magMarkerSize=20
        self.dictPolMarker = {'Up': '^', 'Down': 'v'}
        self.polMarkerSize=8
        self.axvlinewidths=1.2
        #dictionary for key-bindings
        self.dictKeybindings = {'setPick':'alt', 'setPickError':' ', 'delPick':'escape',
                           'setMagMin':'alt', 'setMagMax':' ', 'switchPhase':'control',
                           'delMagMinMax':'escape', 'switchWheelZoom':'z',
                           'switchPan':'p', 'prevStream':'y', 'nextStream':'x',
                           'setPWeight0':'0', 'setPWeight1':'1', 'setPWeight2':'2',
                           'setPWeight3':'3',# 'setPWeight4':'4', 'setPWeight5':'5',
                           'setSWeight0':'0', 'setSWeight1':'1', 'setSWeight2':'2',
                           'setSWeight3':'3',# 'setSWeight4':'4', 'setSWeight5':'5',
                           'setPPolUp':'q', 'setPPolPoorUp':'w',
                           'setPPolDown':'a', 'setPPolPoorDown':'s',
                           'setSPolUp':'q', 'setSPolPoorUp':'w',
                           'setSPolDown':'a', 'setSPolPoorDown':'s'}
        
        # Return, if no streams are given
        if not streams:
            return

        #set up a list of dictionaries to store all picking data
        self.dicts=[]
        for i in range(len(streams)):
            self.dicts.append({})
        
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
        self.fig.canvas.toolbar.pan()
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
        for label in ('save', 'quit'):
            def on_select(item):
                if item.labelstr == 'quit':
                    plt.close()
                print 'you selected', item.labelstr
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
        self.drawPWeightLabel()
        self.drawPPolMarker()
        self.drawPErr1Line()
        self.drawPErr2Line()
        self.drawSLine()
        self.drawSLabel()
        self.drawSWeightLabel()
        self.drawSPolMarker()
        self.drawSErr1Line()
        self.drawSErr2Line()
        self.drawMagMinCross1()
        self.drawMagMaxCross1()
        self.drawMagMinCross2()
        self.drawMagMaxCross2()
    
    def drawPLine(self):
        if not self.dicts[self.stPt].has_key('P'):
            return
        self.PLines=[]
        for i in range(len(self.axs)):
            self.PLines.append(self.axs[i].axvline(self.dicts[self.stPt]['P'],color=self.dictPhaseColors['P'],linewidth=self.axvlinewidths,label='P'))
    
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
    
    def drawPLabel(self):
        if not self.dicts[self.stPt].has_key('P'):
            return
        self.PLabel = self.axs[0].text(self.dicts[self.stPt]['P'], 1 - 0.04 * len(self.axs), '  P',
                             transform = self.trans[0], color=self.dictPhaseColors['P'])
    
    def delPLabel(self):
        try:
            self.axs[0].texts.remove(self.PLabel)
        except:
            pass
        try:
            del self.PLabel
        except:
            pass
    
    def drawPWeightLabel(self):
        if not self.dicts[self.stPt].has_key('P') or not self.dicts[self.stPt].has_key('PWeight'):
            return
        self.PWeightLabel = self.axs[0].text(self.dicts[self.stPt]['P'], 1 - 0.06 * len(self.axs), '  %s'%self.dicts[self.stPt]['PWeight'],
                                   transform = self.trans[0], color = self.dictPhaseColors['P'])
    
    def delPWeightLabel(self):
        try:
            self.axs[0].texts.remove(self.PWeightLabel)
        except:
            pass
        try:
            del self.PWeightLabel
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

    def drawPPolMarker(self):
        if not self.dicts[self.stPt].has_key('P') or not self.dicts[self.stPt].has_key('PPol'):
            return
        #we have to force the graph to the old axes limits because of the completely new line object creation
        xlims=list(self.axs[0].get_xlim())
        ylims=list(self.axs[0].get_ylim())
        if self.dicts[self.stPt]['PPol'] == 'Up':
            self.PPolMarker = self.axs[0].plot([self.dicts[self.stPt]['P']], [1 - 0.01 * len(self.axs)],
                                     zorder = 4000, linewidth = 0,
                                     transform = self.trans[0],
                                     markerfacecolor = self.dictPhaseColors['P'],
                                     marker = self.dictPolMarker['Up'], 
                                     markersize = self.polMarkerSize,
                                     markeredgecolor = self.dictPhaseColors['P'])[0]
        if self.dicts[self.stPt]['PPol'] == 'PoorUp':
            self.PPolMarker = self.axs[0].plot([self.dicts[self.stPt]['P']], [1 - 0.01 * len(self.axs)],
                                     zorder = 4000, linewidth = 0,
                                     transform = self.trans[0],
                                     markerfacecolor = self.axs[0]._axisbg,
                                     marker = self.dictPolMarker['Up'], 
                                     markersize = self.polMarkerSize,
                                     markeredgecolor = self.dictPhaseColors['P'])[0]
        if self.dicts[self.stPt]['PPol'] == 'Down':
            self.PPolMarker = self.axs[-1].plot([self.dicts[self.stPt]['P']], [0.01 * len(self.axs)],
                                      zorder = 4000, linewidth = 0,
                                      transform = self.trans[-1],
                                      markerfacecolor = self.dictPhaseColors['P'],
                                      marker = self.dictPolMarker['Down'], 
                                      markersize = self.polMarkerSize,
                                      markeredgecolor = self.dictPhaseColors['P'])[0]
        if self.dicts[self.stPt]['PPol'] == 'PoorDown':
            self.PPolMarker = self.axs[-1].plot([self.dicts[self.stPt]['P']], [0.01 * len(self.axs)],
                                      zorder = 4000, linewidth = 0,
                                      transform = self.trans[-1],
                                      markerfacecolor = self.axs[-1]._axisbg,
                                      marker = self.dictPolMarker['Down'], 
                                      markersize = self.polMarkerSize,
                                      markeredgecolor = self.dictPhaseColors['P'])[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delPPolMarker(self):
        try:
            self.axs[0].lines.remove(self.PPolMarker)
        except:
            pass
        try:
            self.axs[-1].lines.remove(self.PPolMarker)
        except:
            pass
        try:
            del self.PPolMarker
        except:
            pass
    
    def drawSPolMarker(self):
        if not self.dicts[self.stPt].has_key('S') or not self.dicts[self.stPt].has_key('SPol'):
            return
        #we have to force the graph to the old axes limits because of the completely new line object creation
        xlims=list(self.axs[0].get_xlim())
        ylims=list(self.axs[0].get_ylim())
        if self.dicts[self.stPt]['SPol'] == 'Up':
            self.SPolMarker = self.axs[0].plot([self.dicts[self.stPt]['S']], [1 - 0.01 * len(self.axs)],
                                     zorder = 4000, linewidth = 0,
                                     transform = self.trans[0],
                                     markerfacecolor = self.dictPhaseColors['S'],
                                     marker = self.dictPolMarker['Up'], 
                                     markersize = self.polMarkerSize,
                                     markeredgecolor = self.dictPhaseColors['S'])[0]
        if self.dicts[self.stPt]['SPol'] == 'PoorUp':
            self.SPolMarker = self.axs[0].plot([self.dicts[self.stPt]['S']], [1 - 0.01 * len(self.axs)],
                                     zorder = 4000, linewidth = 0,
                                     transform = self.trans[0],
                                     markerfacecolor = self.axs[0]._axisbg,
                                     marker = self.dictPolMarker['Up'], 
                                     markersize = self.polMarkerSize,
                                     markeredgecolor = self.dictPhaseColors['S'])[0]
        if self.dicts[self.stPt]['SPol'] == 'Down':
            self.SPolMarker = self.axs[-1].plot([self.dicts[self.stPt]['S']], [0.01 * len(self.axs)],
                                      zorder = 4000, linewidth = 0,
                                      transform = self.trans[-1],
                                      markerfacecolor = self.dictPhaseColors['S'],
                                      marker = self.dictPolMarker['Down'], 
                                      markersize = self.polMarkerSize,
                                      markeredgecolor = self.dictPhaseColors['S'])[0]
        if self.dicts[self.stPt]['SPol'] == 'PoorDown':
            self.SPolMarker = self.axs[-1].plot([self.dicts[self.stPt]['S']], [0.01 * len(self.axs)],
                                      zorder = 4000, linewidth = 0,
                                      transform = self.trans[-1],
                                      markerfacecolor = self.axs[-1]._axisbg,
                                      marker = self.dictPolMarker['Down'], 
                                      markersize = self.polMarkerSize,
                                      markeredgecolor = self.dictPhaseColors['S'])[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delSPolMarker(self):
        try:
            self.axs[0].lines.remove(self.SPolMarker)
        except:
            pass
        try:
            self.axs[-1].lines.remove(self.SPolMarker)
        except:
            pass
        try:
            del self.SPolMarker
        except:
            pass
    
    def drawSLine(self):
        if not self.dicts[self.stPt].has_key('S'):
            return
        self.SLines=[]
        for i in range(len(self.axs)):
            self.SLines.append(self.axs[i].axvline(self.dicts[self.stPt]['S'],color=self.dictPhaseColors['S'],linewidth=self.axvlinewidths,label='S'))
    
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
    
    def drawSLabel(self):
        if not self.dicts[self.stPt].has_key('S'):
            return
        self.SLabel = self.axs[0].text(self.dicts[self.stPt]['S'], 1 - 0.04 * len(self.axs), '  S',
                             transform = self.trans[0], color=self.dictPhaseColors['S'])
    
    def delSLabel(self):
        try:
            self.axs[0].texts.remove(self.SLabel)
        except:
            pass
        try:
            del self.SLabel
        except:
            pass
    
    def drawSWeightLabel(self):
        if not self.dicts[self.stPt].has_key('S') or not self.dicts[self.stPt].has_key('SWeight'):
            return
        self.SWeightLabel = self.axs[0].text(self.dicts[self.stPt]['S'], 1 - 0.06 * len(self.axs), '  %s'%self.dicts[self.stPt]['SWeight'],
                                   transform = self.trans[0], color = self.dictPhaseColors['S'])
    
    def delSWeightLabel(self):
        try:
            self.axs[0].texts.remove(self.SWeightLabel)
        except:
            pass
        try:
            del self.SWeightLabel
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
        self.check = CheckButtons(self.axFilt, ('Filter','Zero-Phase'),(False,True))
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
            self.delPPolMarker()
            self.delPLabel()
            self.delPWeightLabel()
            self.dicts[self.stPt]['P']=int(round(event.xdata))
            self.drawPLine()
            self.drawPPolMarker()
            self.drawPLabel()
            self.drawPWeightLabel()
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
                self.delPWeightLabel()
                self.dicts[self.stPt]['PWeight']=0
                self.drawPWeightLabel()
                self.redraw()
                print "P Pick weight set to %i"%self.dicts[self.stPt]['PWeight']
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPWeight1']:
                self.delPWeightLabel()
                self.dicts[self.stPt]['PWeight']=1
                print "P Pick weight set to %i"%self.dicts[self.stPt]['PWeight']
                self.drawPWeightLabel()
                self.redraw()
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPWeight2']:
                self.delPWeightLabel()
                self.dicts[self.stPt]['PWeight']=2
                print "P Pick weight set to %i"%self.dicts[self.stPt]['PWeight']
                self.drawPWeightLabel()
                self.redraw()
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPWeight3']:
                self.delPWeightLabel()
                self.dicts[self.stPt]['PWeight']=3
                print "P Pick weight set to %i"%self.dicts[self.stPt]['PWeight']
                self.drawPWeightLabel()
                self.redraw()
            #if flagPhase==0 and event.key==dictKeybindings['setPWeight4']:
            #    delPWeightLabel()
            #    dicts[self.stPt]['PWeight']=4
            #    print "P Pick weight set to %i"%dicts[self.stPt]['PWeight']
            #    drawPWeightLabel()
            #    redraw()
            #if flagPhase==0 and event.key==dictKeybindings['setPWeight5']:
            #    delPWeightLabel()
            #    dicts[self.stPt]['PWeight']=5
            #    print "P Pick weight set to %i"%dicts[self.stPt]['PWeight']
            #    drawPWeightLabel()
            #    redraw()
        # Set P Pick polarity
        if self.dicts[self.stPt].has_key('P'):
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPPolUp']:
                self.delPPolMarker()
                self.dicts[self.stPt]['PPol']='Up'
                self.drawPPolMarker()
                self.redraw()
                print "P Pick polarity set to %s"%self.dicts[self.stPt]['PPol']
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPPolPoorUp']:
                self.delPPolMarker()
                self.dicts[self.stPt]['PPol']='PoorUp'
                self.drawPPolMarker()
                self.redraw()
                print "P Pick polarity set to %s"%self.dicts[self.stPt]['PPol']
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPPolDown']:
                self.delPPolMarker()
                self.dicts[self.stPt]['PPol']='Down'
                self.drawPPolMarker()
                self.redraw()
                print "P Pick polarity set to %s"%self.dicts[self.stPt]['PPol']
            if self.flagPhase==0 and event.key==self.dictKeybindings['setPPolPoorDown']:
                self.delPPolMarker()
                self.dicts[self.stPt]['PPol']='PoorDown'
                self.drawPPolMarker()
                self.redraw()
                print "P Pick polarity set to %s"%self.dicts[self.stPt]['PPol']
        # Set new S Pick
        if self.flagPhase==1 and event.key==self.dictKeybindings['setPick']:
            self.delSLine()
            self.delSPolMarker()
            self.delSLabel()
            self.delSWeightLabel()
            self.dicts[self.stPt]['S']=int(round(event.xdata))
            self.drawSLine()
            self.drawSPolMarker()
            self.drawSLabel()
            self.drawSWeightLabel()
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
                self.delSWeightLabel()
                self.dicts[self.stPt]['SWeight']=0
                self.drawSWeightLabel()
                self.redraw()
                print "S Pick weight set to %i"%self.dicts[self.stPt]['SWeight']
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSWeight1']:
                self.delSWeightLabel()
                self.dicts[self.stPt]['SWeight']=1
                self.drawSWeightLabel()
                self.redraw()
                print "S Pick weight set to %i"%self.dicts[self.stPt]['SWeight']
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSWeight2']:
                self.delSWeightLabel()
                self.dicts[self.stPt]['SWeight']=2
                self.drawSWeightLabel()
                self.redraw()
                print "S Pick weight set to %i"%self.dicts[self.stPt]['SWeight']
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSWeight3']:
                self.delSWeightLabel()
                self.dicts[self.stPt]['SWeight']=3
                self.drawSWeightLabel()
                self.redraw()
                print "S Pick weight set to %i"%self.dicts[self.stPt]['SWeight']
            #if flagPhase==1 and event.key==dictKeybindings['setSWeight4']:
            #    delSWeightLabel()
            #    dicts[self.stPt]['SWeight']=4
            #    drawSWeightLabel()
            #    redraw()
            #    print "S Pick weight set to %i"%dicts[self.stPt]['SWeight']
            #if flagPhase==1 and event.key==dictKeybindings['setSWeight5']:
            #    delSWeightLabel()
            #    dicts[self.stPt]['SWeight']=5
            #    drawSWeightLabel()
            #    redraw()
            #    print "S Pick weight set to %i"%dicts[self.stPt]['SWeight']
        # Set S Pick polarity
        if self.dicts[self.stPt].has_key('S'):
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSPolUp']:
                self.delSPolMarker()
                self.dicts[self.stPt]['SPol']='Up'
                self.drawSPolMarker()
                self.redraw()
                print "S Pick polarity set to %s"%self.dicts[self.stPt]['SPol']
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSPolPoorUp']:
                self.delSPolMarker()
                self.dicts[self.stPt]['SPol']='PoorUp'
                self.drawSPolMarker()
                self.redraw()
                print "S Pick polarity set to %s"%self.dicts[self.stPt]['SPol']
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSPolDown']:
                self.delSPolMarker()
                self.dicts[self.stPt]['SPol']='Down'
                self.drawSPolMarker()
                self.redraw()
                print "S Pick polarity set to %s"%self.dicts[self.stPt]['SPol']
            if self.flagPhase==1 and event.key==self.dictKeybindings['setSPolPoorDown']:
                self.delSPolMarker()
                self.dicts[self.stPt]['SPol']='PoorDown'
                self.drawSPolMarker()
                self.redraw()
                print "S Pick polarity set to %s"%self.dicts[self.stPt]['SPol']
        # Remove P Pick
        if self.flagPhase==0 and event.key==self.dictKeybindings['delPick']:
            # Try to remove all existing Pick lines and P Pick variable
            self.delPLine()
            self.delP()
            self.delPWeight()
            self.delPPolMarker()
            self.delPPol()
            self.delPLabel()
            self.delPWeightLabel()
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
            self.delSPolMarker()
            self.delSPol()
            self.delSLabel()
            self.delSWeightLabel()
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
        streams.append(read('RJOB_061005_072159.ehz.new'))
        streams[0].append(read('RJOB_061005_072159.ehn.new')[0])
        streams[0].append(read('RJOB_061005_072159.ehe.new')[0])
        streams.append(read('RNON_160505_000059.ehz.new'))
        streams.append(read('RMOA_160505_014459.ehz.new'))
        streams[2].append(read('RMOA_160505_014459.ehn.new')[0])
        streams[2].append(read('RMOA_160505_014459.ehe.new')[0])
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
