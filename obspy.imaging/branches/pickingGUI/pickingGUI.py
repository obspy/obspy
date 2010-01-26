#!/usr/bin/env python

#check for textboxes and other stuff:
#http://code.enthought.com/projects/traits/docs/html/tutorials/traits_ui_scientific_app.html

import matplotlib
matplotlib.use('gtkagg')
#matplotlib.use('wxagg')

from obspy.core import read
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.widgets import Cursor as mplCursor
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


#Define some flags and dictionaries
flagFilt=False #False:no filter  True:filter
flagFiltTyp=0 #0: bandpass 1: bandstop 2:lowpass 3: highpass
dictFiltTyp={'Bandpass':0, 'Bandstop':1, 'Lowpass':2, 'Highpass':3}
flagFiltZPH=True #False: no zero-phase True: zero-phase filtering
flagWheelZoom=True #Switch use of mousewheel for zooming
flagPhase=0 #0: P 1: A 2: Magnitude
dictPhase={'P':0, 'S':1, 'Mag':2}
dictPhaseColors={0:'red', 1:'blue', 2:'green'}
magPickWindow=10 #Estimating the maximum/minimum in a sample-window around click
magMarker='x'
magMarkerEdgeWidth=1.8
magMarkerSize=20
axvlinewidths=1.8




def switch_flagFilt():
    global flagFilt
    flagFilt=not flagFilt
def switch_flagFiltZPH():
    global flagFiltZPH
    flagFiltZPH=not flagFiltZPH

#Define a pointer to navigate through the streams
stNum=len(streams)
stPt=0

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
    t = np.arange(streams[stPt][0].stats.npts)
    axs=[]
    plts=[]
    trNum=len(streams[stPt].traces)
    for i in range(trNum):
        if i==0:
            axs.append(fig.add_subplot(trNum,1,i+1))
        else:
            axs.append(fig.add_subplot(trNum,1,i+1,sharex=axs[0],sharey=axs[0]))
        axs[i].set_ylabel(streams[stPt][0].stats.station+" "+streams[stPt][0].stats.channel)
        plts.append(axs[i].plot(t, streams[stPt][i].data, color='k')[0])
    supTit=fig.suptitle("%s -- %s, %s" % (streams[stPt][0].stats.starttime, streams[stPt][0].stats.endtime, streams[stPt][0].stats.station))
    xMin,xMax=axs[0].get_xlim()
    yMin,yMax=axs[0].get_ylim()
    fig.subplots_adjust(bottom=0.25,hspace=0)
    #multicursor = mplMultiCursor(fig.canvas,axs, useblit=True, color='black', linewidth=1, ls='dotted')

def delAxes():
    for a in axs:
        try:
            fig.delaxes(a)
        except:
            print "Warning: Could not delete axes: %s"%a
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
    radio = RadioButtons(axFiltTyp, ('Bandpass', 'Bandstop', 'Lowpass', 'Highpass'))
    radio.on_clicked(funcFiltTyp)
    
def addPhaseButtons():
    global axPhase
    global radioPhase
    #add phase buttons
    axPhase = fig.add_axes([0.90, 0.02, 0.10, 0.15],frameon=False,axisbg='lightgrey')
    radioPhase = RadioButtons(axPhase, ('P', 'S', 'Mag'))
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
    slideLow = Slider(axLowcut, 'Lowcut', low, high, valinit=low)
    slideHigh = Slider(axHighcut, 'Highcut', low, high, valinit=high)
    slideLow.on_changed(update)
    slideHigh.on_changed(update)
    

def redraw():
    #xlims=list(axs[0].get_xlim())
    #ylims=list(axs[0].get_ylim())
    for a in axs:
        a.figure.canvas.draw()
    #axs[0].set_xlim(xlims)
    #axs[0].set_ylim(ylims)
    #axs[0].figure.canvas.draw()

def updatePlot():
    global pltZ
    global pltN
    global pltE
    filt=[]
    #save current x- and y-limits
    #axZx=list(axZ.get_xlim())
    #axZy=list(axZ.get_ylim())
    #axNy=list(axN.get_ylim())
    #axEy=list(axE.get_ylim())
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
    #set to saved x- and y-limits
    #axZ.set_xlim(axZx)
    #axZ.set_ylim(axZy)
    #axN.set_ylim(axNy)
    #axE.set_ylim(axEy)
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
    flagPhase=dictPhase[label]



#def update(val):
#    global pltZ
#    global pltN
#    global pltE
#    #filter data
#    filtZ=bp(stZ[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
#    filtN=bp(stN[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
#    filtE=bp(stE[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
#    #save current x- and y-limits
#    axZx=list(axZ.get_xlim())
#    axZy=list(axZ.get_ylim())
#    axNy=list(axN.get_ylim())
#    axEy=list(axE.get_ylim())
#    #remove old plots
#    try:
#        axZ.lines.remove(pltZ)
#        axN.lines.remove(pltN)
#        axE.lines.remove(pltE)
#    except:
#        pass
#    #make new plots
#    pltZ=axZ.plot(t,filtZ,color='k')[0]
#    pltN=axN.plot(t,filtN,color='k')[0]
#    pltE=axE.plot(t,filtE,color='k')[0]
#    #set to saved x- and y-limits
#    axZ.set_xlim(axZx)
#    axZ.set_ylim(axZy)
#    axN.set_ylim(axNy)
#    axE.set_ylim(axEy)
#    # Update all subplots
#    redraw()
#    # Console output
#    print "Showing traces filtered to %.2f-%.2f Hz (Zero-Phase Bandpass)"%(slideLow.val,slideHigh.val)
#slideLow.on_changed(updatePlot)
#slideHigh.on_changed(updatePlot)

#axReset = plt.axes([0.8, 0.025, 0.1, 0.04])
#button = Button(axReset, 'Reset', hovercolor='0.975')
#def reset(event):
#    slideLow.reset()
#    slideHigh.reset()
#button.on_clicked(reset)


#axUnfilter = plt.axes([0.8, 0.025, 0.1, 0.04])
#buttonUnfilter = Button(axUnfilter, 'Original', hovercolor='0.975')
#def unfilter(event):
#    slideLow.reset()
#    slideHigh.reset()
#    global pltZ
#    global pltN
#    global pltE
#    #remove old plots
#    try:
#        axZ.lines.remove(pltZ)
#        axN.lines.remove(pltN)
#        axE.lines.remove(pltE)
#    except:
#        pass
#    #save current x- and y-limits
#    axZx=list(axZ.get_xlim())
#    axZy=list(axZ.get_ylim())
#    axNy=list(axN.get_ylim())
#    axEy=list(axE.get_ylim())
#    #make new plots
#    pltZ=axZ.plot(t,stZ[0].data,color='k')[0]
#    pltN=axN.plot(t,stN[0].data,color='k')[0]
#    pltE=axE.plot(t,stE[0].data,color='k')[0]
#    #set to saved x- and y-limits
#    axZ.set_xlim(axZx)
#    axZ.set_ylim(axZy)
#    axN.set_ylim(axNy)
#    axE.set_ylim(axEy)
#    # Update all subplots
#    redraw()
#    # Console output
#    print "Showing original unfiltered traces"
#buttonUnfilter.on_clicked(unfilter)


#global PZLine
#global PNLine
#global PELine
#global P
#global PZErr1Line
#global PNErr1Line
#global PEErr1Line
#global PErr1
#global PZErr2Line
#global PNErr2Line
#global PEErr2Line
#global PErr2
#global SZLine
#global SNLine
#global SELine
#global S
#global SZErr1Line
#global SNErr1Line
#global SEErr1Line
#global SErr1
#global SZErr2Line
#global SNErr2Line
#global SEErr2Line
#global SErr2

# Set up initial plot
fig = plt.figure()
drawAxes()
addFiltButtons()
addPhaseButtons()
addSliders()
redraw()

# Define the event for setting P- and S-wave picks
def pick(event):
    global PLines
    global P
    global SLines
    global S
    global PErr1Lines
    global PErr1
    global PErr2Lines
    global PErr2
    global SErr1Lines
    global SErr1
    global SErr2Lines
    global SErr2
    global MagMin
    global MagMinT
    global MagMinCross
    global MagMax
    global MagMaxT
    global MagMaxCross
    # Set new P Pick
    if flagPhase==0 and event.key=='alt':
        # Define global variables seen outside
        # Remove old lines from the plot before plotting the new ones
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(PLines[i])
        except:
            pass
        # Plot the lines for the P pick in all three traces
        PLines=[]
        for i in range(len(axs)):
            PLines.append(axs[i].axvline(event.xdata,color=dictPhaseColors[flagPhase],linewidth=axvlinewidths))
        # Save sample value of pick (round to integer sample value)
        P=int(round(event.xdata))
        # Update all subplots
        redraw()
        # Console output
        print "P Pick set at %i"%P
    # Set new S Pick
    if flagPhase==1 and event.key=='alt':
        # Define global variables seen outside
        # Remove old lines from the plot before plotting the new ones
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(SLines[i])
        except:
            pass
        # Plot the lines for the S pick in all three traces
        SLines=[]
        for i in range(len(axs)):
            SLines.append(axs[i].axvline(event.xdata,color=dictPhaseColors[flagPhase],linewidth=axvlinewidths))
        # Save sample value of pick (round to integer sample value)
        S=int(round(event.xdata))
        # Update all subplots
        redraw()
        # Console output
        print "S Pick set at %i"%S
    # Remove P Pick
    if flagPhase==0 and event.key=='escape':
        # Try to remove all existing Pick lines and P Pick variable
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(PLines[i])
            del PLines
            del P
            # Console output
            print "P Pick deleted"
        except:
            pass
        # Try to remove existing Pick Error 1 lines and variable
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(PErr1Lines[i])
            del PErr1Lines
            del PErr1
        except:
            pass
        # Try to remove existing Pick Error 2 lines and variable
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(PErr2Lines[i])
            del PErr2Lines
            del PErr2
        except:
            pass
        # Update all subplots
        redraw()
    # Remove S Pick
    if flagPhase==1 and event.key=='escape':
        # Try to remove all existing Pick lines and P Pick variable
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(SLines[i])
            del SLines
            del S
            # Console output
            print "S Pick deleted"
        except:
            pass
        # Try to remove existing Pick Error 1 lines and variable
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(SErr1Lines[i])
            del SErr1Lines
            del SErr1
        except:
            pass
        # Try to remove existing Pick Error 2 lines and variable
        try:
            for i in range(len(axs)):
                axs[i].lines.remove(SErr2Lines[i])
            del SErr2Lines
            del SErr2
        except:
            pass
        # Update all subplots
        redraw()
    # Set new P Pick uncertainties
    if flagPhase==0 and event.key==' ':
        # Set Flag to determine scenario
        try:
            # Set left Error Pick
            if event.xdata<P:
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
            try:
                for i in range(len(axs)):
                    axs[i].lines.remove(PErr1Lines[i])
            except:
                pass
            # Plot the lines for the P Error pick in all three traces
            PErr1Lines=[]
            for i in range(len(axs)):
                PErr1Lines.append(axs[i].axvline(event.xdata,ymin=0.25,ymax=0.75,color=dictPhaseColors[flagPhase],linewidth=axvlinewidths))
            # Save sample value of error pick (round to integer sample value)
            PErr1=int(round(event.xdata))
            # Update all subplots
            redraw()
            # Console output
            print "P Error Pick 1 set at %i"%PErr1
        # Case 2
        if errFlag==2:
            # Define global variables seen outside
            # Remove old lines from the plot before plotting the new ones
            try:
                for i in range(len(axs)):
                    axs[i].lines.remove(PErr2Lines[i])
            except:
                pass
            # Plot the lines for the P Error pick in all three traces
            PErr2Lines=[]
            for i in range(len(axs)):
                PErr2Lines.append(axs[i].axvline(event.xdata,ymin=0.25,ymax=0.75,color=dictPhaseColors[flagPhase],linewidth=axvlinewidths))
            # Save sample value of error pick (round to integer sample value)
            PErr2=int(round(event.xdata))
            # Update all subplots
            redraw()
            # Console output
            print "P Error Pick 2 set at %i"%PErr2
    # Set new S Pick uncertainties
    if flagPhase==1 and event.key==' ':
        # Set Flag to determine scenario
        try:
            # Set left Error Pick
            if event.xdata<S:
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
            try:
                for i in range(len(axs)):
                    axs[i].lines.remove(SErr1Lines[i])
            except:
                pass
            # Plot the lines for the S Error pick in all three traces
            SErr1Lines=[]
            for i in range(len(axs)):
                SErr1Lines.append(axs[i].axvline(event.xdata,ymin=0.25,ymax=0.75,color=dictPhaseColors[flagPhase],linewidth=axvlinewidths))
            # Save sample value of error pick (round to integer sample value)
            SErr1=int(round(event.xdata))
            # Update all subplots
            redraw()
            # Console output
            print "S Error Pick 1 set at %i"%SErr1
        # Case 2
        if errFlag==2:
            # Define global variables seen outside
            # Remove old lines from the plot before plotting the new ones
            try:
                for i in range(len(axs)):
                    axs[i].lines.remove(SErr2Lines[i])
            except:
                pass
            # Plot the lines for the S Error pick in all three traces
            SErr2Lines=[]
            for i in range(len(axs)):
                SErr2Lines.append(axs[i].axvline(event.xdata,ymin=0.25,ymax=0.75,color=dictPhaseColors[flagPhase],linewidth=axvlinewidths))
            # Save sample value of error pick (round to integer sample value)
            SErr2=int(round(event.xdata))
            # Update all subplots
            redraw()
            # Console output
            print "S Error Pick 2 set at %i"%SErr2
    # Magnitude estimation picking:
    if flagPhase==2 and event.key=='alt':
        for a in axs:
            try:
                a.lines.remove(MagMinCross)
            except:
                pass
        xpos=int(event.xdata)
        ydata=event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
        cutoffSamples=xpos-magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
        MagMin=np.min(ydata[xpos-magPickWindow:xpos+magPickWindow])
        MagMinT=cutoffSamples+np.argmin(ydata[xpos-magPickWindow:xpos+magPickWindow])
        MagMinCross=event.inaxes.plot([MagMinT],[MagMin],markersize=magMarkerSize,markeredgewidth=magMarkerEdgeWidth,color=dictPhaseColors[flagPhase],marker=magMarker)[0]
        redraw()
        print "Minimum for magnitude estimation set: %s at %s"%(MagMin,MagMinT)
    if flagPhase==2 and event.key==' ':
        for a in axs:
            try:
                a.lines.remove(MagMaxCross)
            except:
                pass
        xpos=int(event.xdata)
        ydata=event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
        cutoffSamples=xpos-magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
        MagMax=np.max(ydata[xpos-magPickWindow:xpos+magPickWindow])
        MagMaxT=cutoffSamples+np.argmax(ydata[xpos-magPickWindow:xpos+magPickWindow])
        #save axes info:
        #xlims=list(axs[0].get_xlim())
        #ylims=list(axs[0].get_ylim())
        MagMaxCross=event.inaxes.plot([MagMaxT],[MagMax],markersize=magMarkerSize,markeredgewidth=magMarkerEdgeWidth,color=dictPhaseColors[flagPhase],marker=magMarker)[0]
        redraw()
        #axs[0].set_xlim(xlims)
        #axs[0].set_ylim(ylims)
        #axs[0].figure.canvas.draw()
        print "Maximum for magnitude estimation set: %s at %s"%(MagMax,MagMaxT)
    if flagPhase==2 and event.key=='escape':
        for a in axs:
            try:
                a.lines.remove(MagMinCross)
            except:
                pass
            try:
                a.lines.remove(MagMaxCross)
            except:
                pass
        try:
            del MagMin
            del MagMinT
        except:
            pass
        try:
            del MagMax
            del MagMaxT
        except:
            pass
        redraw()
        print "Magnitude estimation info removed"

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
    if event.key=="z":
        global flagWheelZoom
        flagWheelZoom=not flagWheelZoom
        if flagWheelZoom:
            print "Mouse wheel zooming activated"
        else:
            print "Mouse wheel zooming deactivated"

def switchPan(event):
    if event.key=="p":
        fig.canvas.toolbar.pan()
        redraw()
        print "Switching pan mode"

def switchStream(event):
    global stPt
    if event.key=="y":
        stPt=(stPt-1)%stNum
        delAxes()
        drawAxes()
        delSliders()
        addSliders()
        redraw()
        print "Going to previous stream"
    if event.key=="x":
        stPt=(stPt+1)%stNum
        delAxes()
        drawAxes()
        delSliders()
        addSliders()
        redraw()
        print "Going to next stream"
        


#remove unwanted buttons from Toolbar
#tb=fig.canvas.toolbar
#tb.pan()
#tb.DeleteToolByPos(1)
#tb.DeleteToolByPos(1)
#tb.DeleteToolByPos(4)
#tb.DeleteToolByPos(4)

#define events for filter buttons
#def unfilterTraces(self):
#    global pltZ
#    global pltN
#    global pltE
#    #remove old plots
#    axZ.lines.remove(pltZ)
#    axN.lines.remove(pltN)
#    axE.lines.remove(pltE)
#    #save current x- and y-limits
#    axZx=list(axZ.get_xlim())
#    axZy=list(axZ.get_ylim())
#    axNy=list(axN.get_ylim())
#    axEy=list(axE.get_ylim())
#    #make new plots
#    pltZ=axZ.plot(t,stZ[0].data,color='k')[0]
#    pltN=axN.plot(t,stN[0].data,color='k')[0]
#    pltE=axE.plot(t,stE[0].data,color='k')[0]
#    #set to saved x- and y-limits
#    axZ.set_xlim(axZx)
#    axZ.set_ylim(axZy)
#    axN.set_ylim(axNy)
#    axE.set_ylim(axEy)
#    # Update all subplots
#    redraw()
#    # Console output
#    print "Showing original unfiltered traces"
#def filterTraces(self):
#    global pltZ
#    global pltN
#    global pltE
#    #filter data
#    filtZ=bp(stZ[0].data,1,3,df=stZ[0].stats.sampling_rate)
#    filtN=bp(stN[0].data,1,3,df=stZ[0].stats.sampling_rate)
#    filtE=bp(stE[0].data,1,3,df=stZ[0].stats.sampling_rate)
#    #save current x- and y-limits
#    axZx=list(axZ.get_xlim())
#    axZy=list(axZ.get_ylim())
#    axNy=list(axN.get_ylim())
#    axEy=list(axE.get_ylim())
#    #remove old plots
#    axZ.lines.remove(pltZ)
#    axN.lines.remove(pltN)
#    axE.lines.remove(pltE)
#    #make new plots
#    pltZ=axZ.plot(t,filtZ,color='k')[0]
#    pltN=axN.plot(t,filtN,color='k')[0]
#    pltE=axE.plot(t,filtE,color='k')[0]
#    #set to saved x- and y-limits
#    axZ.set_xlim(axZx)
#    axZ.set_ylim(axZy)
#    axN.set_ylim(axNy)
#    axE.set_ylim(axEy)
#    # Update all subplots
#    redraw()
#    # Console output
#    print "Showing traces filtered to 1-10 Hz (Zero-Phase Bandpass)"
#add new buttons
#idUnfilter=wx.NewId()
#tbUnfilter=tb.AddSimpleTool(idUnfilter, _load_bitmap('home.xpm'),'UnFilter', 'Show unfiltered traces')
#wx.EVT_TOOL(tb,idUnfilter,unfilterTraces)
#idFilter=wx.NewId()
#tbFilter=tb.AddSimpleTool(idFilter, _load_bitmap('hand.xpm'),'Filter', 'Filter traces (zero-phase bandpass)')
#wx.EVT_TOOL(tb,idFilter,filterTraces)

#t = np.arange(streams[stPt][0].stats.npts)
#axs=[]
#plts=[]
#trNum=len(streams[stPt].traces)
#for i in range(1,trNum+1):
#    if i=0:
#        axs.append(fig.add_subplot(trNum,1,i))
#    else:
#        axs.append(fig.add_subplot(trNum,1,i,sharex=axs[0],sharey=axs[1]))
#    axs[i].set_ylabel(streams[stPt][0].stats.station+" "+streams[stPt][0].stats.channel)
#    plts.append(axs[i].plot(t, streams[stPt][i].data, color='k'))
#fig.suptitle("%s -- %s, %s" % (streams[stPt][0].stats.starttime, streams[stPt][0].stats.endtime, streams[stPt][0].stats.station))
#xMin,xMax=axs[0].get_xlim()
#yMin,yMax=axs[0].get_ylim()
#fig.subplots_adjust(bottom=0.25,hspace=0)
# Activate all mouse/key/Cursor-events
keypress = fig.canvas.mpl_connect('key_press_event', pick)
keypressWheelZoom = fig.canvas.mpl_connect('key_press_event', switchWheelZoom)
keypressPan = fig.canvas.mpl_connect('key_press_event', switchPan)
keypressNextPrev = fig.canvas.mpl_connect('key_press_event', switchStream)
scroll = fig.canvas.mpl_connect('scroll_event', zoom)
scroll_button = fig.canvas.mpl_connect('button_press_event', zoom_reset)
#cursorZ = mplCursor(axZ, useblit=True, color='black', linewidth=1, ls='dotted')
#cursorN = mplCursor(axN, useblit=True, color='black', linewidth=1, ls='dotted')
#cursorE = mplCursor(axE, useblit=True, color='black', linewidth=1, ls='dotted')
fig.canvas.toolbar.pan()

plt.show()
