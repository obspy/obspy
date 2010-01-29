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


#==============================================================================
#Prepare the example streams, this should be done by seishub beforehand in the future
#streams=[]
#streams.append(read('RJOB_061005_072159.ehz.new'))
#streams[0].append(read('RJOB_061005_072159.ehn.new')[0])
#streams[0].append(read('RJOB_061005_072159.ehe.new')[0])
#streams.append(read('RNON_160505_000059.ehz.new'))
#streams.append(read('RMOA_160505_014459.ehz.new'))
#streams[2].append(read('RMOA_160505_014459.ehn.new')[0])
#streams[2].append(read('RMOA_160505_014459.ehe.new')[0])
#===============================================================================


def picker(streams = None):
    global flagFilt
    global flagFiltTyp
    global dictFiltTyp
    global flagFiltZPH
    global flagWheelZoom
    global flagPhase
    global dictPhase
    global dictPhaseColors
    global pickingColor
    global magPickWindow
    global magMinMarker
    global magMaxMarker
    global magMarkerEdgeWidth
    global magMarkerSize
    global axvlinewidths
    global dictKeybindings
    global dicts
    global stNum
    global stPt
    global fig
    global keypress
    global keypressWheelZoom
    global keypressPan
    global keypressNextPrev
    global buttonpressBlockRedraw
    global buttonreleaseAllowRedraw
    global scroll
    global scroll_button
    global multicursor
    global props
    global hoverprops
    global menuitems
    global item
    global menu
    global valFiltLow
    global valFiltHigh
    global trans
    #Define some flags, dictionaries and plotting options
    flagFilt=False #False:no filter  True:filter
    flagFiltTyp=0 #0: bandpass 1: bandstop 2:lowpass 3: highpass
    dictFiltTyp={'Bandpass':0, 'Bandstop':1, 'Lowpass':2, 'Highpass':3}
    flagFiltZPH=True #False: no zero-phase True: zero-phase filtering
    valFiltLow=np.NaN # These are overridden with low/high estimated from sampling rate
    valFiltHigh=np.NaN
    flagWheelZoom=True #Switch use of mousewheel for zooming
    flagPhase=0 #0:P 1:S 2:Magnitude
    dictPhase={'P':0, 'S':1, 'Mag':2}
    dictPhaseColors={'P':'red', 'S':'blue', 'Mag':'green'}
    pickingColor=dictPhaseColors['P']
    magPickWindow=10 #Estimating the maximum/minimum in a sample-window around click
    magMinMarker='x'
    magMaxMarker='x'
    magMarkerEdgeWidth=1.8
    magMarkerSize=20
    dictPolMarker = {'Up': '^', 'Down': 'v'}
    polMarkerSize=8
    axvlinewidths=1.2
    #dictionary for key-bindings
    dictKeybindings = {'setPick':'alt', 'setPickError':' ', 'delPick':'escape',
                       'setMagMin':'alt', 'setMagMax':' ',
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
    
    #set up a list of dictionaries to store all picking data
    dicts=[]
    for i in range(len(streams)):
        dicts.append({})
    
    #Define a pointer to navigate through the streams
    stNum=len(streams)
    stPt=0
    
    
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
        def __init__(self, fontsize=14, labelcolor='black', bgcolor='yellow', alpha=1.0):
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
    
    
    
    def switch_flagFilt():
        global flagFilt
        flagFilt=not flagFilt
    def switch_flagFiltZPH():
        global flagFiltZPH
        flagFiltZPH=not flagFiltZPH
    
    ## Trim all to same length, us Z as reference
    #start, end = stZ[0].stats.starttime, stZ[0].stats.endtime
    #stN.trim(start, end)
    #stE.trim(start, end)
    
    
    def drawAxes():
        global axs
        global t
        global plts
        global multicursor
        global supTit
        global xMin
        global xMax
        global yMin
        global yMax
        global trans
        t = np.arange(streams[stPt][0].stats.npts)
        axs = []
        plts = []
        trans = []
        trNum = len(streams[stPt].traces)
        for i in range(trNum):
            if i == 0:
                axs.append(fig.add_subplot(trNum,1,i+1))
                trans.append(matplotlib.transforms.blended_transform_factory(axs[i].transData,
                                                                             axs[i].transAxes))
            else:
                axs.append(fig.add_subplot(trNum,1,i+1,sharex=axs[0],sharey=axs[0]))
                trans.append(matplotlib.transforms.blended_transform_factory(axs[i].transData,
                                                                             axs[i].transAxes))
            axs[i].set_ylabel(streams[stPt][i].stats.station+" "+streams[stPt][i].stats.channel)
            plts.append(axs[i].plot(t, streams[stPt][i].data, color='k',zorder=1000)[0])
        supTit=fig.suptitle("%s -- %s, %s" % (streams[stPt][0].stats.starttime, streams[stPt][0].stats.endtime, streams[stPt][0].stats.station))
        xMin,xMax=axs[0].get_xlim()
        yMin,yMax=axs[0].get_ylim()
        fig.subplots_adjust(bottom=0.25,hspace=0,right=0.999,top=0.95)
    
    def drawSavedPicks():
        drawPLine()
        drawPLabel()
        drawPWeightLabel()
        drawPPolMarker()
        drawPErr1Line()
        drawPErr2Line()
        drawSLine()
        drawSLabel()
        drawSWeightLabel()
        drawSPolMarker()
        drawSErr1Line()
        drawSErr2Line()
        drawMagMinCross()
        drawMagMaxCross()
    
    def drawPLine():
        global PLines
        if not dicts[stPt].has_key('P'):
            return
        PLines=[]
        for i in range(len(axs)):
            PLines.append(axs[i].axvline(dicts[stPt]['P'],color=dictPhaseColors['P'],linewidth=axvlinewidths,label='P'))
    
    def delPLine():
        global PLines
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(PLines[i])
        except:
            pass
        try:
            del PLines
        except:
            pass
    
    def drawPLabel():
        global PLabel
        if not dicts[stPt].has_key('P'):
            return
        PLabel = axs[0].text(dicts[stPt]['P'], 0.87, '  P',
                             transform = trans[0], color=dictPhaseColors['P'])
    
    def delPLabel():
        global PLabel
        try:
            axs[0].texts.remove(PLabel)
        except:
            pass
        try:
            del PLabel
        except:
            pass
    
    def drawPWeightLabel():
        global PWeightLabel
        if not dicts[stPt].has_key('P') or not dicts[stPt].has_key('PWeight'):
            return
        PWeightLabel = axs[0].text(dicts[stPt]['P'], 0.77, '  %s'%dicts[stPt]['PWeight'],
                                   transform = trans[0], color = dictPhaseColors['P'])
    
    def delPWeightLabel():
        global PWeightLabel
        try:
            axs[0].texts.remove(PWeightLabel)
        except:
            pass
        try:
            del PWeightLabel
        except:
            pass
    
    def drawPErr1Line():
        global PErr1Lines
        if not dicts[stPt].has_key('P') or not dicts[stPt].has_key('PErr1'):
            return
        PErr1Lines=[]
        for i in range(len(axs)):
            PErr1Lines.append(axs[i].axvline(dicts[stPt]['PErr1'],ymin=0.25,ymax=0.75,color=dictPhaseColors['P'],linewidth=axvlinewidths,label='PErr1'))
    
    def delPErr1Line():
        global PErr1Lines
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(PErr1Lines[i])
        except:
            pass
        try:
            del PErr1Lines
        except:
            pass
    
    def drawPErr2Line():
        global PErr2Lines
        if not dicts[stPt].has_key('P') or not dicts[stPt].has_key('PErr2'):
            return
        PErr2Lines=[]
        for i in range(len(axs)):
            PErr2Lines.append(axs[i].axvline(dicts[stPt]['PErr2'],ymin=0.25,ymax=0.75,color=dictPhaseColors['P'],linewidth=axvlinewidths,label='PErr2'))
    
    def delPErr2Line():
        global PErr2Lines
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(PErr2Lines[i])
        except:
            pass
        try:
            del PErr2Lines
        except:
            pass

    def drawPPolMarker():
        global PPolMarker
        if not dicts[stPt].has_key('P') or not dicts[stPt].has_key('PPol'):
            return
        #we have to force the graph to the old axes limits because of the completely new line object creation
        xlims=list(axs[0].get_xlim())
        ylims=list(axs[0].get_ylim())
        if dicts[stPt]['PPol'] == 'Up':
            PPolMarker = axs[0].plot([dicts[stPt]['P']], [0.97], zorder = 4000,
                                     linewidth = 0, transform = trans[0],
                                     markerfacecolor = dictPhaseColors['P'],
                                     marker = dictPolMarker['Up'], 
                                     markersize = polMarkerSize,
                                     markeredgecolor = dictPhaseColors['P'])[0]
        if dicts[stPt]['PPol'] == 'PoorUp':
            PPolMarker = axs[0].plot([dicts[stPt]['P']], [0.97], zorder = 4000,
                                     linewidth = 0, transform = trans[0],
                                     markerfacecolor = axs[0]._axisbg,
                                     marker = dictPolMarker['Up'], 
                                     markersize = polMarkerSize,
                                     markeredgecolor = dictPhaseColors['P'])[0]
        if dicts[stPt]['PPol'] == 'Down':
            PPolMarker = axs[-1].plot([dicts[stPt]['P']], [0.03], zorder = 4000,
                                     linewidth = 0, transform = trans[-1],
                                     markerfacecolor = dictPhaseColors['P'],
                                     marker = dictPolMarker['Down'], 
                                     markersize = polMarkerSize,
                                     markeredgecolor = dictPhaseColors['P'])[0]
        if dicts[stPt]['PPol'] == 'PoorDown':
            PPolMarker = axs[-1].plot([dicts[stPt]['P']], [0.03], zorder = 4000,
                                     linewidth = 0, transform = trans[-1],
                                     markerfacecolor = axs[-1]._axisbg,
                                     marker = dictPolMarker['Down'], 
                                     markersize = polMarkerSize,
                                     markeredgecolor = dictPhaseColors['P'])[0]
        axs[0].set_xlim(xlims)
        axs[0].set_ylim(ylims)
    
    def delPPolMarker():
        global PPolMarker
        try:
            axs[0].lines.remove(PPolMarker)
        except:
            pass
        try:
            axs[-1].lines.remove(PPolMarker)
        except:
            pass
        try:
            del PPolMarker
        except:
            pass
    
    def drawSPolMarker():
        global SPolMarker
        if not dicts[stPt].has_key('S') or not dicts[stPt].has_key('SPol'):
            return
        #we have to force the graph to the old axes limits because of the completely new line object creation
        xlims=list(axs[0].get_xlim())
        ylims=list(axs[0].get_ylim())
        if dicts[stPt]['SPol'] == 'Up':
            SPolMarker = axs[0].plot([dicts[stPt]['S']], [0.97], zorder = 4000,
                                     linewidth = 0, transform = trans[0],
                                     markerfacecolor = dictPhaseColors['S'],
                                     marker = dictPolMarker['Up'], 
                                     markersize = polMarkerSize,
                                     markeredgecolor = dictPhaseColors['S'])[0]
        if dicts[stPt]['SPol'] == 'PoorUp':
            SPolMarker = axs[0].plot([dicts[stPt]['S']], [0.97], zorder = 4000,
                                     linewidth = 0, transform = trans[0],
                                     markerfacecolor = axs[0]._axisbg,
                                     marker = dictPolMarker['Up'], 
                                     markersize = polMarkerSize,
                                     markeredgecolor = dictPhaseColors['S'])[0]
        if dicts[stPt]['SPol'] == 'Down':
            SPolMarker = axs[-1].plot([dicts[stPt]['S']], [0.03], zorder = 4000,
                                     linewidth = 0, transform = trans[-1],
                                     markerfacecolor = dictPhaseColors['S'],
                                     marker = dictPolMarker['Down'], 
                                     markersize = polMarkerSize,
                                     markeredgecolor = dictPhaseColors['S'])[0]
        if dicts[stPt]['SPol'] == 'PoorDown':
            SPolMarker = axs[-1].plot([dicts[stPt]['S']], [0.03], zorder = 4000,
                                     linewidth = 0, transform = trans[-1],
                                     markerfacecolor = axs[-1]._axisbg,
                                     marker = dictPolMarker['Down'], 
                                     markersize = polMarkerSize,
                                     markeredgecolor = dictPhaseColors['S'])[0]
        axs[0].set_xlim(xlims)
        axs[0].set_ylim(ylims)
    
    def delSPolMarker():
        global SPolMarker
        try:
            axs[0].lines.remove(SPolMarker)
        except:
            pass
        try:
            axs[-1].lines.remove(SPolMarker)
        except:
            pass
        try:
            del SPolMarker
        except:
            pass
    
    def drawSLine():
        global SLines
        if not dicts[stPt].has_key('S'):
            return
        SLines=[]
        for i in range(len(axs)):
            SLines.append(axs[i].axvline(dicts[stPt]['S'],color=dictPhaseColors['S'],linewidth=axvlinewidths,label='S'))
    
    def delSLine():
        global SLines
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(SLines[i])
        except:
            pass
        try:
            del SLines
        except:
            pass
    
    def drawSLabel():
        global SLabel
        if not dicts[stPt].has_key('S'):
            return
        SLabel = axs[0].text(dicts[stPt]['S'], 0.87, '  S',
                             transform = trans[0], color=dictPhaseColors['S'])
    
    def delSLabel():
        global SLabel
        try:
            axs[0].texts.remove(SLabel)
        except:
            pass
        try:
            del SLabel
        except:
            pass
    
    def drawSWeightLabel():
        global SWeightLabel
        if not dicts[stPt].has_key('S') or not dicts[stPt].has_key('SWeight'):
            return
        SWeightLabel = axs[0].text(dicts[stPt]['S'], 0.77, '  %s'%dicts[stPt]['SWeight'],
                                   transform = trans[0], color = dictPhaseColors['S'])
    
    def delSWeightLabel():
        global SWeightLabel
        try:
            axs[0].texts.remove(SWeightLabel)
        except:
            pass
        try:
            del SWeightLabel
        except:
            pass
    
    def drawSErr1Line():
        global SErr1Lines
        if not dicts[stPt].has_key('S') or not dicts[stPt].has_key('SErr1'):
            return
        SErr1Lines=[]
        for i in range(len(axs)):
            SErr1Lines.append(axs[i].axvline(dicts[stPt]['SErr1'],ymin=0.25,ymax=0.75,color=dictPhaseColors['S'],linewidth=axvlinewidths,label='SErr1'))
    
    def delSErr1Line():
        global SErr1Lines
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(SErr1Lines[i])
        except:
            pass
        try:
            del SErr1Lines
        except:
            pass
    
    def drawSErr2Line():
        global SErr2Lines
        if not dicts[stPt].has_key('S') or not dicts[stPt].has_key('SErr2'):
            return
        SErr2Lines=[]
        for i in range(len(axs)):
            SErr2Lines.append(axs[i].axvline(dicts[stPt]['SErr2'],ymin=0.25,ymax=0.75,color=dictPhaseColors['S'],linewidth=axvlinewidths,label='SErr2'))
    
    def delSErr2Line():
        global SErr2Lines
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(SErr2Lines[i])
        except:
            pass
        try:
            del SErr2Lines
        except:
            pass
    
    def drawMagMinCross():
        global MagMinCross
        if not dicts[stPt].has_key('MagMin'):
            return
        #we have to force the graph to the old axes limits because of the completely new line object creation
        xlims=list(axs[0].get_xlim())
        ylims=list(axs[0].get_ylim())
        MagMinCross=axs[dicts[stPt]['MagMinAxIndex']].plot([dicts[stPt]['MagMinT']],[dicts[stPt]['MagMin']],markersize=magMarkerSize,markeredgewidth=magMarkerEdgeWidth,color=dictPhaseColors['Mag'],marker=magMinMarker,zorder=2000)[0]
        axs[0].set_xlim(xlims)
        axs[0].set_ylim(ylims)
    
    def delMagMinCross():
        global MagMinCross
        try:
            axs[dicts[stPt]['MagMinAxIndex']].lines.remove(MagMinCross)
        except:
            pass
    
    def drawMagMaxCross():
        global MagMaxCross
        if not dicts[stPt].has_key('MagMax'):
            return
        #we have to force the graph to the old axes limits because of the completely new line object creation
        xlims=list(axs[0].get_xlim())
        ylims=list(axs[0].get_ylim())
        MagMaxCross=axs[dicts[stPt]['MagMaxAxIndex']].plot([dicts[stPt]['MagMaxT']],[dicts[stPt]['MagMax']],markersize=magMarkerSize,markeredgewidth=magMarkerEdgeWidth,color=dictPhaseColors['Mag'],marker=magMaxMarker,zorder=2000)[0]
        axs[0].set_xlim(xlims)
        axs[0].set_ylim(ylims)
    
    def delMagMaxCross():
        global MagMaxCross
        try:
            axs[dicts[stPt]['MagMaxAxIndex']].lines.remove(MagMaxCross)
        except:
            pass
    
    def delP():
        global dicts
        try:
            del dicts[stPt]['P']
            print "P Pick deleted"
        except:
            pass
            
    def delPWeight():
        global dicts
        try:
            del dicts[stPt]['PWeight']
            print "P Pick weight deleted"
        except:
            pass
            
    def delPPol():
        global dicts
        try:
            del dicts[stPt]['PPol']
            print "P Pick polarity deleted"
        except:
            pass
            
    def delPErr1():
        global dicts
        try:
            del dicts[stPt]['PErr1']
            print "PErr1 Pick deleted"
        except:
            pass
            
    def delPErr2():
        global dicts
        try:
            del dicts[stPt]['PErr2']
            print "PErr2 Pick deleted"
        except:
            pass
            
    def delS():
        global dicts
        try:
            del dicts[stPt]['S']
            print "S Pick deleted"
        except:
            pass
            
    def delSWeight():
        global dicts
        try:
            del dicts[stPt]['SWeight']
            print "S Pick weight deleted"
        except:
            pass
            
    def delSPol():
        global dicts
        try:
            del dicts[stPt]['SPol']
            print "S Pick polarity deleted"
        except:
            pass
            
    def delSErr1():
        global dicts
        try:
            del dicts[stPt]['SErr1']
            print "SErr1 Pick deleted"
        except:
            pass
            
    def delSErr2():
        global dicts
        try:
            del dicts[stPt]['SErr2']
            print "SErr2 Pick deleted"
        except:
            pass
            
    def delMagMin():
        global dicts
        try:
            del dicts[stPt]['MagMin']
            del dicts[stPt]['MagMinT']
            del dicts[stPt]['MagMinAxIndex']
            print "Magnitude Minimum Estimation Pick deleted"
        except:
            pass
            
    def delMagMax():
        global dicts
        try:
            del dicts[stPt]['MagMax']
            del dicts[stPt]['MagMaxT']
            del dicts[stPt]['MagMaxAxIndex']
            print "Magnitude Maximum Estimation Pick deleted"
        except:
            pass
            
    
    def delAxes():
        global axs
        for a in axs:
            try:
                fig.delaxes(a)
                del a
            except:
                pass
        try:
            fig.texts.remove(supTit)
        except:
            pass
    
    def addFiltButtons():
        global axFilt
        global check
        global axFiltTyp
        global radio
        #add filter buttons
        axFilt = fig.add_axes([0.22, 0.02, 0.15, 0.15],frameon=False,axisbg='lightgrey')
        check = CheckButtons(axFilt, ('Filter','Zero-Phase'),(False,True))
        check.on_clicked(funcFilt)
        axFiltTyp = fig.add_axes([0.40, 0.02, 0.15, 0.15],frameon=False,axisbg='lightgrey')
        radio = RadioButtons(axFiltTyp, ('Bandpass', 'Bandstop', 'Lowpass', 'Highpass'),activecolor='k')
        radio.on_clicked(funcFiltTyp)
        
    def addPhaseButtons():
        global axPhase
        global radioPhase
        #add phase buttons
        axPhase = fig.add_axes([0.10, 0.02, 0.10, 0.15],frameon=False,axisbg='lightgrey')
        radioPhase = RadioButtons(axPhase, ('P', 'S', 'Mag'),activecolor='k')
        radioPhase.on_clicked(funcPhase)
        
    def updateLow(val):
        if not flagFilt or flagFiltTyp == 2:
            return
        else:
            updatePlot()
    
    def updateHigh(val):
        if not flagFilt or flagFiltTyp == 3:
            return
        else:
            updatePlot()
    
    def delSliders():
        global valFiltLow
        global valFiltHigh
        valFiltLow = slideLow.val
        valFiltHigh = slideHigh.val
        try:
            fig.delaxes(axLowcut)
            fig.delaxes(axHighcut)
        except:
            return
    
    def addSliders():
        global axLowcut
        global axHighcut
        global slideLow
        global slideHigh
        global valFiltLow
        global valFiltHigh
        #add filter slider
        axLowcut = fig.add_axes([0.63, 0.05, 0.30, 0.03], xscale='log')
        axHighcut  = fig.add_axes([0.63, 0.10, 0.30, 0.03], xscale='log')
        low  = 1.0/ (streams[stPt][0].stats.npts/float(streams[stPt][0].stats.sampling_rate))
        high = streams[stPt][0].stats.sampling_rate/2.0
        valFiltLow = max(low,valFiltLow)
        valFiltHigh = min(high,valFiltHigh)
        slideLow = Slider(axLowcut, 'Lowcut', low, high, valinit=valFiltLow, facecolor='darkgrey', edgecolor='k', linewidth=1.7)
        slideHigh = Slider(axHighcut, 'Highcut', low, high, valinit=valFiltHigh, facecolor='darkgrey', edgecolor='k', linewidth=1.7)
        slideLow.on_changed(updateLow)
        slideHigh.on_changed(updateHigh)
        
    
    def redraw():
        global multicursor
        for line in multicursor.lines:
            line.set_visible(False)
        fig.canvas.draw()
    
    def updatePlot():
        filt=[]
        #filter data
        if flagFilt==True:
            if flagFiltZPH==True:
                if flagFiltTyp==0:
                    for tr in streams[stPt].traces:
                        filt.append(bandpassZPHSH(tr.data,slideLow.val,slideHigh.val,df=tr.stats.sampling_rate))
                    print "Zero-Phase Bandpass: %.2f-%.2f Hz"%(slideLow.val,slideHigh.val)
                if flagFiltTyp==1:
                    for tr in streams[stPt].traces:
                        filt.append(bandstopZPHSH(tr.data,slideLow.val,slideHigh.val,df=tr.stats.sampling_rate))
                    print "Zero-Phase Bandstop: %.2f-%.2f Hz"%(slideLow.val,slideHigh.val)
                if flagFiltTyp==2:
                    for tr in streams[stPt].traces:
                        filt.append(lowpassZPHSH(tr.data,slideHigh.val,df=tr.stats.sampling_rate))
                    print "Zero-Phase Lowpass: %.2f Hz"%(slideHigh.val)
                if flagFiltTyp==3:
                    for tr in streams[stPt].traces:
                        filt.append(highpassZPHSH(tr.data,slideLow.val,df=tr.stats.sampling_rate))
                    print "Zero-Phase Highpass: %.2f Hz"%(slideLow.val)
            elif flagFiltZPH==False:
                if flagFiltTyp==0:
                    for tr in streams[stPt].traces:
                        filt.append(bandpass(tr.data,slideLow.val,slideHigh.val,df=tr.stats.sampling_rate))
                    print "One-Pass Bandpass: %.2f-%.2f Hz"%(slideLow.val,slideHigh.val)
                if flagFiltTyp==1:
                    for tr in streams[stPt].traces:
                        filt.append(bandstop(tr.data,slideLow.val,slideHigh.val,df=tr.stats.sampling_rate))
                    print "One-Pass Bandstop: %.2f-%.2f Hz"%(slideLow.val,slideHigh.val)
                if flagFiltTyp==2:
                    for tr in streams[stPt].traces:
                        filt.append(lowpass(tr.data,slideHigh.val,df=tr.stats.sampling_rate))
                    print "One-Pass Lowpass: %.2f Hz"%(slideHigh.val)
                if flagFiltTyp==3:
                    for tr in streams[stPt].traces:
                        filt.append(highpass(tr.data,slideLow.val,df=tr.stats.sampling_rate))
                    print "One-Pass Highpass: %.2f Hz"%(slideLow.val)
            #make new plots
            for i in range(len(plts)):
                plts[i].set_data(t, filt[i])
        else:
            #make new plots
            for i in range(len(plts)):
                plts[i].set_data(t, streams[stPt][i].data)
            print "Unfiltered Traces"
        # Update all subplots
        redraw()
    
    def funcFilt(label):
        if label=='Filter':
            switch_flagFilt()
            updatePlot()
        elif label=='Zero-Phase':
            switch_flagFiltZPH()
            if flagFilt==True:
                updatePlot()
    
    def funcFiltTyp(label):
        global flagFiltTyp
        flagFiltTyp=dictFiltTyp[label]
        if flagFilt==True:
            updatePlot()
    
    def funcPhase(label):
        global flagPhase
        global pickingColor
        flagPhase=dictPhase[label]
        pickingColor=dictPhaseColors[label]
        for l in multicursor.lines:
            l.set_color(pickingColor)
        radioPhase.circles[flagPhase].set_facecolor(pickingColor)
        redraw()
    
    
    
    
    # Define the event that handles the setting of P- and S-wave picks
    def pick(event):
        global dicts
        # Set new P Pick
        if flagPhase==0 and event.key==dictKeybindings['setPick']:
            delPLine()
            delPPolMarker()
            delPLabel()
            delPWeightLabel()
            dicts[stPt]['P']=int(round(event.xdata))
            drawPLine()
            drawPPolMarker()
            drawPLabel()
            drawPWeightLabel()
            #check if the new P pick lies outside of the Error Picks
            try:
                if dicts[stPt]['P']<dicts[stPt]['PErr1']:
                    delPErr1Line()
                    delPErr1()
            except:
                pass
            try:
                if dicts[stPt]['P']>dicts[stPt]['PErr2']:
                    delPErr2Line()
                    delPErr2()
            except:
                pass
            # Update all subplots
            redraw()
            # Console output
            print "P Pick set at %i"%dicts[stPt]['P']
        # Set P Pick weight
        if dicts[stPt].has_key('P'):
            if flagPhase==0 and event.key==dictKeybindings['setPWeight0']:
                delPWeightLabel()
                dicts[stPt]['PWeight']=0
                drawPWeightLabel()
                redraw()
                print "P Pick weight set to %i"%dicts[stPt]['PWeight']
            if flagPhase==0 and event.key==dictKeybindings['setPWeight1']:
                delPWeightLabel()
                dicts[stPt]['PWeight']=1
                print "P Pick weight set to %i"%dicts[stPt]['PWeight']
                drawPWeightLabel()
                redraw()
            if flagPhase==0 and event.key==dictKeybindings['setPWeight2']:
                delPWeightLabel()
                dicts[stPt]['PWeight']=2
                print "P Pick weight set to %i"%dicts[stPt]['PWeight']
                drawPWeightLabel()
                redraw()
            if flagPhase==0 and event.key==dictKeybindings['setPWeight3']:
                delPWeightLabel()
                dicts[stPt]['PWeight']=3
                print "P Pick weight set to %i"%dicts[stPt]['PWeight']
                drawPWeightLabel()
                redraw()
            #if flagPhase==0 and event.key==dictKeybindings['setPWeight4']:
            #    delPWeightLabel()
            #    dicts[stPt]['PWeight']=4
            #    print "P Pick weight set to %i"%dicts[stPt]['PWeight']
            #    drawPWeightLabel()
            #    redraw()
            #if flagPhase==0 and event.key==dictKeybindings['setPWeight5']:
            #    delPWeightLabel()
            #    dicts[stPt]['PWeight']=5
            #    print "P Pick weight set to %i"%dicts[stPt]['PWeight']
            #    drawPWeightLabel()
            #    redraw()
        # Set P Pick polarity
        if dicts[stPt].has_key('P'):
            if flagPhase==0 and event.key==dictKeybindings['setPPolUp']:
                delPPolMarker()
                dicts[stPt]['PPol']='Up'
                drawPPolMarker()
                redraw()
                print "P Pick polarity set to %s"%dicts[stPt]['PPol']
            if flagPhase==0 and event.key==dictKeybindings['setPPolPoorUp']:
                delPPolMarker()
                dicts[stPt]['PPol']='PoorUp'
                drawPPolMarker()
                redraw()
                print "P Pick polarity set to %s"%dicts[stPt]['PPol']
            if flagPhase==0 and event.key==dictKeybindings['setPPolDown']:
                delPPolMarker()
                dicts[stPt]['PPol']='Down'
                drawPPolMarker()
                redraw()
                print "P Pick polarity set to %s"%dicts[stPt]['PPol']
            if flagPhase==0 and event.key==dictKeybindings['setPPolPoorDown']:
                delPPolMarker()
                dicts[stPt]['PPol']='PoorDown'
                drawPPolMarker()
                redraw()
                print "P Pick polarity set to %s"%dicts[stPt]['PPol']
        # Set new S Pick
        if flagPhase==1 and event.key==dictKeybindings['setPick']:
            delSLine()
            delSPolMarker()
            delSLabel()
            delSWeightLabel()
            dicts[stPt]['S']=int(round(event.xdata))
            drawSLine()
            drawSPolMarker()
            drawSLabel()
            drawSWeightLabel()
            #check if the new S pick lies outside of the Error Picks
            try:
                if dicts[stPt]['S']<dicts[stPt]['SErr1']:
                    delSErr1Line()
                    delSErr1()
            except:
                pass
            try:
                if dicts[stPt]['S']>dicts[stPt]['SErr2']:
                    delSErr2Line()
                    delSErr2()
            except:
                pass
            # Update all subplots
            redraw()
            # Console output
            print "S Pick set at %i"%dicts[stPt]['S']
        # Set S Pick weight
        if dicts[stPt].has_key('S'):
            if flagPhase==1 and event.key==dictKeybindings['setSWeight0']:
                delSWeightLabel()
                dicts[stPt]['SWeight']=0
                drawSWeightLabel()
                redraw()
                print "S Pick weight set to %i"%dicts[stPt]['SWeight']
            if flagPhase==1 and event.key==dictKeybindings['setSWeight1']:
                delSWeightLabel()
                dicts[stPt]['SWeight']=1
                drawSWeightLabel()
                redraw()
                print "S Pick weight set to %i"%dicts[stPt]['SWeight']
            if flagPhase==1 and event.key==dictKeybindings['setSWeight2']:
                delSWeightLabel()
                dicts[stPt]['SWeight']=2
                drawSWeightLabel()
                redraw()
                print "S Pick weight set to %i"%dicts[stPt]['SWeight']
            if flagPhase==1 and event.key==dictKeybindings['setSWeight3']:
                delSWeightLabel()
                dicts[stPt]['SWeight']=3
                drawSWeightLabel()
                redraw()
                print "S Pick weight set to %i"%dicts[stPt]['SWeight']
            #if flagPhase==1 and event.key==dictKeybindings['setSWeight4']:
            #    delSWeightLabel()
            #    dicts[stPt]['SWeight']=4
            #    drawSWeightLabel()
            #    redraw()
            #    print "S Pick weight set to %i"%dicts[stPt]['SWeight']
            #if flagPhase==1 and event.key==dictKeybindings['setSWeight5']:
            #    delSWeightLabel()
            #    dicts[stPt]['SWeight']=5
            #    drawSWeightLabel()
            #    redraw()
            #    print "S Pick weight set to %i"%dicts[stPt]['SWeight']
        # Set S Pick polarity
        if dicts[stPt].has_key('S'):
            if flagPhase==1 and event.key==dictKeybindings['setSPolUp']:
                delSPolMarker()
                dicts[stPt]['SPol']='Up'
                drawSPolMarker()
                redraw()
                print "S Pick polarity set to %s"%dicts[stPt]['SPol']
            if flagPhase==1 and event.key==dictKeybindings['setSPolPoorUp']:
                delSPolMarker()
                dicts[stPt]['SPol']='PoorUp'
                drawSPolMarker()
                redraw()
                print "S Pick polarity set to %s"%dicts[stPt]['SPol']
            if flagPhase==1 and event.key==dictKeybindings['setSPolDown']:
                delSPolMarker()
                dicts[stPt]['SPol']='Down'
                drawSPolMarker()
                redraw()
                print "S Pick polarity set to %s"%dicts[stPt]['SPol']
            if flagPhase==1 and event.key==dictKeybindings['setSPolPoorDown']:
                delSPolMarker()
                dicts[stPt]['SPol']='PoorDown'
                drawSPolMarker()
                redraw()
                print "S Pick polarity set to %s"%dicts[stPt]['SPol']
        # Remove P Pick
        if flagPhase==0 and event.key==dictKeybindings['delPick']:
            # Try to remove all existing Pick lines and P Pick variable
            delPLine()
            delP()
            delPWeight()
            delPPolMarker()
            delPPol()
            delPLabel()
            delPWeightLabel()
            # Try to remove existing Pick Error 1 lines and variable
            delPErr1Line()
            delPErr1()
            # Try to remove existing Pick Error 2 lines and variable
            delPErr2Line()
            delPErr2()
            # Update all subplots
            redraw()
        # Remove S Pick
        if flagPhase==1 and event.key==dictKeybindings['delPick']:
            # Try to remove all existing Pick lines and P Pick variable
            delSLine()
            delS()
            delSWeight()
            delSPolMarker()
            delSPol()
            delSLabel()
            delSWeightLabel()
            # Try to remove existing Pick Error 1 lines and variable
            delSErr1Line()
            delSErr1()
            # Try to remove existing Pick Error 2 lines and variable
            delSErr2Line()
            delSErr2()
            # Update all subplots
            redraw()
        # Set new P Pick uncertainties
        if flagPhase==0 and event.key==dictKeybindings['setPickError']:
            # Set Flag to determine scenario
            try:
                # Set left Error Pick
                if event.xdata<dicts[stPt]['P']:
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
                delPErr1Line()
                # Save sample value of error pick (round to integer sample value)
                dicts[stPt]['PErr1']=int(round(event.xdata))
                # Plot the lines for the P Error pick in all three traces
                drawPErr1Line()
                # Update all subplots
                redraw()
                # Console output
                print "P Error Pick 1 set at %i"%dicts[stPt]['PErr1']
            # Case 2
            if errFlag==2:
                # Define global variables seen outside
                # Remove old lines from the plot before plotting the new ones
                delPErr2Line()
                # Save sample value of error pick (round to integer sample value)
                dicts[stPt]['PErr2']=int(round(event.xdata))
                # Plot the lines for the P Error pick in all three traces
                drawPErr2Line()
                # Update all subplots
                redraw()
                # Console output
                print "P Error Pick 2 set at %i"%dicts[stPt]['PErr2']
        # Set new S Pick uncertainties
        if flagPhase==1 and event.key==dictKeybindings['setPickError']:
            # Set Flag to determine scenario
            try:
                # Set left Error Pick
                if event.xdata<dicts[stPt]['S']:
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
                delSErr1Line()
                # Save sample value of error pick (round to integer sample value)
                dicts[stPt]['SErr1']=int(round(event.xdata))
                # Plot the lines for the S Error pick in all three traces
                drawSErr1Line()
                # Update all subplots
                redraw()
                # Console output
                print "S Error Pick 1 set at %i"%dicts[stPt]['SErr1']
            # Case 2
            if errFlag==2:
                # Define global variables seen outside
                # Remove old lines from the plot before plotting the new ones
                delSErr2Line()
                # Save sample value of error pick (round to integer sample value)
                dicts[stPt]['SErr2']=int(round(event.xdata))
                # Plot the lines for the S Error pick in all three traces
                drawSErr2Line()
                # Update all subplots
                redraw()
                # Console output
                print "S Error Pick 2 set at %i"%dicts[stPt]['SErr2']
        # Magnitude estimation picking:
        if flagPhase==2 and event.key==dictKeybindings['setMagMin']:
            delMagMinCross()
            xpos=int(event.xdata)
            ydata=event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
            cutoffSamples=xpos-magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
            dicts[stPt]['MagMin']=np.min(ydata[xpos-magPickWindow:xpos+magPickWindow])
            dicts[stPt]['MagMinT']=cutoffSamples+np.argmin(ydata[xpos-magPickWindow:xpos+magPickWindow])
            dicts[stPt]['MagMinAxIndex']=axs.index(event.inaxes)
            #delete old MagMax Pick, if new MagMin Pick is higher or if it is on another axes
            try:
                if dicts[stPt].has_key('MagMax') and dicts[stPt]['MagMinAxIndex']!=dicts[stPt]['MagMaxAxIndex']:
                    delMagMaxCross()
                    delMagMax()
            except:
                pass
            try:
                if dicts[stPt]['MagMin']>dicts[stPt]['MagMax']:
                    delMagMaxCross()
                    delMagMax()
            except:
                pass
            drawMagMinCross()
            redraw()
            print "Minimum for magnitude estimation set: %s at %s"%(dicts[stPt]['MagMin'],dicts[stPt]['MagMinT'])
        if flagPhase==2 and event.key==dictKeybindings['setMagMax']:
            delMagMaxCross()
            xpos=int(event.xdata)
            ydata=event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
            cutoffSamples=xpos-magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
            dicts[stPt]['MagMax']=np.max(ydata[xpos-magPickWindow:xpos+magPickWindow])
            dicts[stPt]['MagMaxT']=cutoffSamples+np.argmax(ydata[xpos-magPickWindow:xpos+magPickWindow])
            dicts[stPt]['MagMaxAxIndex']=axs.index(event.inaxes)
            #delete old MagMin Pick, if new MagMax Pick is lower or if it is on another axes
            try:
                if dicts[stPt].has_key('MagMin') and dicts[stPt]['MagMinAxIndex']!=dicts[stPt]['MagMaxAxIndex']:
                    delMagMinCross()
                    delMagMin()
            except:
                pass
            try:
                if dicts[stPt]['MagMin']>dicts[stPt]['MagMax']:
                    delMagMinCross()
                    delMagMin()
            except:
                pass
            drawMagMaxCross()
            redraw()
            print "Maximum for magnitude estimation set: %s at %s"%(dicts[stPt]['MagMax'],dicts[stPt]['MagMaxT'])
        if flagPhase==2 and event.key==dictKeybindings['delMagMinMax']:
            delMagMaxCross()
            delMagMinCross()
            delMagMin()
            delMagMax()
            redraw()
    
    # Define zoom events for the mouse scroll wheel
    def zoom(event):
        # Zoom in on scroll-up
        if event.button=='up' and flagWheelZoom:
            # Calculate and set new axes boundaries from old ones
            (left,right)=axs[0].get_xbound()
            left+=(event.xdata-left)/2
            right-=(right-event.xdata)/2
            axs[0].set_xbound(lower=left,upper=right)
            # Update all subplots
            redraw()
        # Zoom out on scroll-down
        if event.button=='down' and flagWheelZoom:
            # Calculate and set new axes boundaries from old ones
            (left,right)=axs[0].get_xbound()
            left-=(event.xdata-left)/2
            right+=(right-event.xdata)/2
            axs[0].set_xbound(lower=left,upper=right)
            # Update all subplots
            redraw()
    
    # Define zoom reset for the mouse button 2 (always scroll wheel!?)
    def zoom_reset(event):
        if event.button==2:
            # Use Z trace limits as boundaries
            axs[0].set_xbound(lower=xMin,upper=xMax)
            axs[0].set_ybound(lower=yMin,upper=yMax)
            # Update all subplots
            redraw()
            print "Resetting axes"
    
    def switchWheelZoom(event):
        if event.key==dictKeybindings['switchWheelZoom']:
            global flagWheelZoom
            flagWheelZoom=not flagWheelZoom
            if flagWheelZoom:
                print "Mouse wheel zooming activated"
            else:
                print "Mouse wheel zooming deactivated"
    
    def switchPan(event):
        if event.key==dictKeybindings['switchPan']:
            fig.canvas.toolbar.pan()
            fig.canvas.widgetlock.release(fig.canvas.toolbar)
            redraw()
            print "Switching pan mode"
    
    #lookup multicursor source: http://matplotlib.sourcearchive.com/documentation/0.98.1/widgets_8py-source.html
    def multicursorReinit():
        global multicursor
        fig.canvas.mpl_disconnect(multicursor.id1)
        fig.canvas.mpl_disconnect(multicursor.id2)
        multicursor.__init__(fig.canvas,axs, useblit=True, color='black', linewidth=1, ls='dotted')
        #fig.canvas.draw_idle()
        #multicursor._update()
        #multicursor.needclear=True
        #multicursor.background = fig.canvas.copy_from_bbox(fig.canvas.figure.bbox)
        #fig.canvas.restore_region(multicursor.background)
        #fig.canvas.blit(fig.canvas.figure.bbox)
        for l in multicursor.lines:
            l.set_color(pickingColor)
    
    def switchStream(event):
        global stPt
        if event.key==dictKeybindings['prevStream']:
            stPt=(stPt-1)%stNum
            delAxes()
            drawAxes()
            drawSavedPicks()
            delSliders()
            addSliders()
            multicursorReinit()
            updatePlot()
            print "Going to previous stream"
        if event.key==dictKeybindings['nextStream']:
            stPt=(stPt+1)%stNum
            delAxes()
            drawAxes()
            drawSavedPicks()
            delSliders()
            addSliders()
            multicursorReinit()
            updatePlot()
            print "Going to next stream"
            
    def blockRedraw(event):
        if event.button==1 or event.button==3:
            multicursor.visible=False
            fig.canvas.widgetlock(fig.canvas.toolbar)
            
    def allowRedraw(event):
        if event.button==1 or event.button==3:
            multicursor.visible=True
            fig.canvas.widgetlock.release(fig.canvas.toolbar)
    
    
            
    # Set up initial plot
    fig = plt.figure()
    drawAxes()
    addFiltButtons()
    addPhaseButtons()
    addSliders()
    #redraw()
    fig.canvas.draw()
    # Activate all mouse/key/Cursor-events
    keypress = fig.canvas.mpl_connect('key_press_event', pick)
    keypressWheelZoom = fig.canvas.mpl_connect('key_press_event', switchWheelZoom)
    keypressPan = fig.canvas.mpl_connect('key_press_event', switchPan)
    keypressNextPrev = fig.canvas.mpl_connect('key_press_event', switchStream)
    buttonpressBlockRedraw = fig.canvas.mpl_connect('button_press_event', blockRedraw)
    buttonreleaseAllowRedraw = fig.canvas.mpl_connect('button_release_event', allowRedraw)
    scroll = fig.canvas.mpl_connect('scroll_event', zoom)
    scroll_button = fig.canvas.mpl_connect('button_press_event', zoom_reset)
    #cursorZ = mplCursor(axZ, useblit=True, color='black', linewidth=1, ls='dotted')
    #cursorN = mplCursor(axN, useblit=True, color='black', linewidth=1, ls='dotted')
    #cursorE = mplCursor(axE, useblit=True, color='black', linewidth=1, ls='dotted')
    fig.canvas.toolbar.pan()
    fig.canvas.widgetlock.release(fig.canvas.toolbar)
    #multicursor = mplMultiCursor(fig.canvas,axs, useblit=True, color='black', linewidth=1, ls='dotted')
    multicursor = MultiCursor(fig.canvas,axs, useblit=True, color=dictPhaseColors['P'], linewidth=1, ls='dotted')
    for l in multicursor.lines:
        l.set_color(dictPhaseColors['P'])
    radioPhase.circles[0].set_facecolor(dictPhaseColors['P'])
    #add menu buttons:
    props = ItemProperties(labelcolor='black', bgcolor='yellow', fontsize=15, alpha=0.2)
    hoverprops = ItemProperties(labelcolor='white', bgcolor='blue', fontsize=15, alpha=0.2)
    menuitems = []
    for label in ('save', 'quit'):
        def on_select(item):
            print 'you selected', item.labelstr
        item = MenuItem(fig, label, props=props, hoverprops=hoverprops, on_select=on_select)
        menuitems.append(item)
    menu = Menu(fig, menuitems)
    
    
    
    plt.show()


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
    (options, args) = parser.parse_args()
    for req in ['-d','-t','-i']:
        if not getattr(parser.values,parser.get_option(req).dest):
            parser.print_help()
            return
    
    if options.local:
        streams=[]
        streams.append(read('RJOB_061005_072159.ehz.new'))
        streams[0].append(read('RJOB_061005_072159.ehn.new')[0])
        streams[0].append(read('RJOB_061005_072159.ehe.new')[0])
        streams.append(read('RNON_160505_000059.ehz.new'))
        streams.append(read('RMOA_160505_014459.ehz.new'))
        streams[2].append(read('RMOA_160505_014459.ehn.new')[0])
        streams[2].append(read('RMOA_160505_014459.ehe.new')[0])
        picker(streams)
    else:
        try:
            t = UTCDateTime(options.time)
            client = Client()
            container = []
            for id in options.ids.split(","):
                net, sta, loc, cha = id.split(".")
                st = client.waveform.getWaveform(net, sta, loc, cha, 
                                                 t, t + options.duration)
                st.sort()
                st.reverse()
                container.append(st)
        except:
            parser.print_help()
            raise

        picker(container)

if __name__ == "__main__":
    main()
