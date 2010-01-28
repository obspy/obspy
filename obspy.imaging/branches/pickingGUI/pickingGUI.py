#!/usr/bin/env python

#check for textboxes and other stuff:
#http://code.enthought.com/projects/traits/docs/html/tutorials/traits_ui_scientific_app.html

import matplotlib
matplotlib.use('gtkagg')

from obspy.core import read
import matplotlib.pyplot as plt
import numpy as np
import sys
#from matplotlib.widgets import Cursor as mplCursor
from matplotlib.widgets import MultiCursor as mplMultiCursor
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

#from matplotlib.backends.backend_wx import _load_bitmap
#import wx

from obspy.signal.filter import bandpass,bandpassZPHSH,bandstop,bandstopZPHSH,lowpass,lowpassZPHSH,highpass,highpassZPHSH


#==============================================================================
#Prepare the example streams, this should be done by seishub beforehand in the future
streams=[]
streams.append(read('RJOB_061005_072159.ehz.new'))
streams[0].append(read('RJOB_061005_072159.ehn.new')[0])
streams[0].append(read('RJOB_061005_072159.ehe.new')[0])
streams.append(read('RNON_160505_000059.ehz.new'))
streams.append(read('RMOA_160505_014459.ehz.new'))
streams[2].append(read('RMOA_160505_014459.ehn.new')[0])
streams[2].append(read('RMOA_160505_014459.ehe.new')[0])
#===============================================================================


#Define some flags, dictionaries and plotting options
flagFilt=False #False:no filter  True:filter
flagFiltTyp=0 #0: bandpass 1: bandstop 2:lowpass 3: highpass
dictFiltTyp={'Bandpass':0, 'Bandstop':1, 'Lowpass':2, 'Highpass':3}
flagFiltZPH=True #False: no zero-phase True: zero-phase filtering
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
axvlinewidths=1.2
#dictionary for key-bindings
dictKeybindings={'setPick':'alt','setPickError':' ','delPick':'escape','setMagMin':'alt','setMagMax':' ','delMagMinMax':'escape','switchWheelZoom':'z','switchPan':'p','prevStream':'y','nextStream':'x'}

#set up a list of dictionaries to store all picking data
dicts=[]
for i in range(len(streams)):
    dicts.append({})

#Define a pointer to navigate through the streams
stNum=len(streams)
stPt=0





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
    t = np.arange(streams[stPt][0].stats.npts)
    axs=[]
    plts=[]
    trNum=len(streams[stPt].traces)
    for i in range(trNum):
        if i==0:
            axs.append(fig.add_subplot(trNum,1,i+1))
        else:
            axs.append(fig.add_subplot(trNum,1,i+1,sharex=axs[0],sharey=axs[0]))
        axs[i].set_ylabel(streams[stPt][i].stats.station+" "+streams[stPt][i].stats.channel)
        plts.append(axs[i].plot(t, streams[stPt][i].data, color='k',zorder=1000)[0])
    supTit=fig.suptitle("%s -- %s, %s" % (streams[stPt][0].stats.starttime, streams[stPt][0].stats.endtime, streams[stPt][0].stats.station))
    xMin,xMax=axs[0].get_xlim()
    yMin,yMax=axs[0].get_ylim()
    fig.subplots_adjust(bottom=0.25,hspace=0,right=0.999,top=0.95)

def drawSavedPicks():
    if dicts[stPt].has_key('P'):
        drawPLine()
    if dicts[stPt].has_key('PErr1'):
        drawPErr1Line()
    if dicts[stPt].has_key('PErr2'):
        drawPErr2Line()
    if dicts[stPt].has_key('S'):
        drawSLine()
    if dicts[stPt].has_key('SErr1'):
        drawSErr1Line()
    if dicts[stPt].has_key('SErr2'):
        drawSErr2Line()
    if dicts[stPt].has_key('MagMin'):
        drawMagMinCross()
    if dicts[stPt].has_key('MagMax'):
        drawMagMaxCross()

def drawPLine():
    global PLines
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

def drawPErr1Line():
    global PErr1Lines
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

def drawSLine():
    global SLines
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

def drawSErr1Line():
    global SErr1Lines
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
        #try:
        #    while True:
        #        a.lines.pop()
        #except:
        #    pass
        try:
        #    for l in a.lines:
        #        del l
        #    del a.lines
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
    axFilt = fig.add_axes([0.02, 0.02, 0.15, 0.15],frameon=False,axisbg='lightgrey')
    check = CheckButtons(axFilt, ('Filter','Zero-Phase'),(False,True))
    check.on_clicked(funcFilt)
    axFiltTyp = fig.add_axes([0.20, 0.02, 0.15, 0.15],frameon=False,axisbg='lightgrey')
    radio = RadioButtons(axFiltTyp, ('Bandpass', 'Bandstop', 'Lowpass', 'Highpass'),activecolor='k')
    radio.on_clicked(funcFiltTyp)
    
def addPhaseButtons():
    global axPhase
    global radioPhase
    #add phase buttons
    axPhase = fig.add_axes([0.90, 0.02, 0.10, 0.15],frameon=False,axisbg='lightgrey')
    radioPhase = RadioButtons(axPhase, ('P', 'S', 'Mag'),activecolor='k')
    radioPhase.on_clicked(funcPhase)
    
def update(val):
    if flagFilt==1:
        updatePlot()
    else:
        pass

def delSliders():
    try:
        fig.delaxes(axLowcut)
        fig.delaxes(axHighcut)
    except:
        pass

def addSliders():
    global axLowcut
    global axHighcut
    global slideLow
    global slideHigh
    #add filter slider
    axLowcut = fig.add_axes([0.45, 0.05, 0.35, 0.03], xscale='log')
    axHighcut  = fig.add_axes([0.45, 0.10, 0.35, 0.03], xscale='log')
    low  = 1.0/ (streams[stPt][0].stats.npts/float(streams[stPt][0].stats.sampling_rate))
    high = streams[stPt][0].stats.sampling_rate/2.0
    slideLow = Slider(axLowcut, 'Lowcut', low, high, valinit=low, facecolor='darkgrey', edgecolor='k', linewidth=1.7)
    slideHigh = Slider(axHighcut, 'Highcut', low, high, valinit=high, facecolor='darkgrey', edgecolor='k', linewidth=1.7)
    slideLow.on_changed(update)
    slideHigh.on_changed(update)
    

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
                    filt.append(lowpassZPHSH(tr.data,slideLow.val,df=tr.stats.sampling_rate))
                print "Zero-Phase Lowpass: %.2f Hz"%(slideLow.val)
            if flagFiltTyp==3:
                for tr in streams[stPt].traces:
                    filt.append(highpassZPHSH(tr.data,slideHigh.val,df=tr.stats.sampling_rate))
                print "Zero-Phase Highpass: %.2f Hz"%(slideHigh.val)
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
                    filt.append(lowpass(tr.data,slideLow.val,df=tr.stats.sampling_rate))
                print "One-Pass Lowpass: %.2f Hz"%(slideLow.val)
            if flagFiltTyp==3:
                for tr in streams[stPt].traces:
                    filt.append(highpass(tr.data,slideHigh.val,df=tr.stats.sampling_rate))
                print "One-Pass Highpass: %.2f Hz"%(slideHigh.val)
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
    #global MagMin
    #global MagMinT
    #global MagMinCross
    #global MagMax
    #global MagMaxT
    #global MagMaxCross
    # Set new P Pick
    if flagPhase==0 and event.key==dictKeybindings['setPick']:
        # Define global variables seen outside
        # Remove old lines from the plot before plotting the new ones
        delPLine()
        # Save sample value of pick (round to integer sample value)
        dicts[stPt]['P']=int(round(event.xdata))
        # Plot the lines for the P pick in all three traces
        drawPLine()
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
    # Set new S Pick
    if flagPhase==1 and event.key==dictKeybindings['setPick']:
        # Define global variables seen outside
        # Remove old lines from the plot before plotting the new ones
        delSLine()
        # Save sample value of pick (round to integer sample value)
        dicts[stPt]['S']=int(round(event.xdata))
        # Plot the lines for the S pick in all three traces
        drawSLine()
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
    # Remove P Pick
    if flagPhase==0 and event.key==dictKeybindings['delPick']:
        # Try to remove all existing Pick lines and P Pick variable
        delPLine()
        delP()
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
        redraw()
        print "Going to previous stream"
    if event.key==dictKeybindings['nextStream']:
        stPt=(stPt+1)%stNum
        delAxes()
        drawAxes()
        drawSavedPicks()
        delSliders()
        addSliders()
        multicursorReinit()
        redraw()
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

plt.show()
