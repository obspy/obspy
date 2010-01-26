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

try:
    fileZ = 'RJOB_061005_072159.ehz.new'
    fileN = 'RJOB_061005_072159.ehn.new'
    fileE = 'RJOB_061005_072159.ehe.new'
    #fileZ = sys.argv[1]
    #fileN = sys.argv[2]
    #fileE = sys.argv[3]
except:
    print __doc__
    raise

# Read all traces
stZ = read(fileZ)
stN = read(fileN)
stE = read(fileE)

#Define some flags for filtering
flagFilt=False #False:no filter  True:filter
flagFiltTyp=0 #0: bandpass 1: bandstop 2:lowpass 3: highpass
flagFiltZPH=True #False: no zero-phase True: zero-phase filtering
flagWheelZoom=True #Switch use of mousewheel for zooming

# Trim all to same length, us Z as reference
start, end = stZ[0].stats.starttime, stZ[0].stats.endtime
stN.trim(start, end)
stE.trim(start, end)

# Plot all traces
t = np.arange(stZ[0].stats.npts)
fig = plt.figure()
axZ = fig.add_subplot(311)
pltZ = axZ.plot(t, stZ[0].data, color='k')[0]
axN = fig.add_subplot(312, sharex=axZ,sharey=axZ)
pltN = axN.plot(t, stN[0].data, color='k')[0]
axE = fig.add_subplot(313, sharex=axZ,sharey=axZ)
pltE = axE.plot(t, stE[0].data, color='k')[0]
fig.suptitle("%s -- %s, %s" % (str(start), str(end), stZ[0].stats.station))
axZ.set_ylabel(stZ[0].stats.station+" "+stZ[0].stats.channel)
axN.set_ylabel(stN[0].stats.station+" "+stN[0].stats.channel)
axE.set_ylabel(stE[0].stats.station+" "+stE[0].stats.channel)
xMin,xMax=axZ.get_xlim()
yMin,yMax=axZ.get_ylim()

def redraw():
    axZ.figure.canvas.draw()
    axN.figure.canvas.draw()
    axE.figure.canvas.draw()

def switch_flagFilt():
    global flagFilt
    flagFilt=not flagFilt
def switch_flagFiltZPH():
    global flagFiltZPH
    flagFiltZPH=not flagFiltZPH

#add filter buttons
axFilt = fig.add_axes([0.02, 0.02, 0.15, 0.15],frameon=False,axisbg='lightgrey')
check = CheckButtons(axFilt, ('Filter','Zero-Phase'),(False,True))
def funcFilt(label):
    if label=='Filter':
        switch_flagFilt()
        updatePlot()
    elif label=='Zero-Phase':
        switch_flagFiltZPH()
        if flagFilt==True:
            updatePlot()
check.on_clicked(funcFilt)

axFiltTyp = fig.add_axes([0.20, 0.02, 0.15, 0.15],frameon=False,axisbg='lightgrey')
radio = RadioButtons(axFiltTyp, ('Bandpass', 'Bandstop', 'Lowpass', 'Highpass'))
def funcFiltTyp(label):
    global flagFiltTyp
    dictFiltTyp={'Bandpass':0, 'Bandstop':1, 'Lowpass':2, 'Highpass':3}
    flagFiltTyp=dictFiltTyp[label]
    if flagFilt==True:
        updatePlot()
radio.on_clicked(funcFiltTyp)


#add filter slider
fig.subplots_adjust(bottom=0.25,hspace=0)
axLowcut = fig.add_axes([0.45, 0.05, 0.45, 0.03], xscale='log')
axHighcut  = fig.add_axes([0.45, 0.10, 0.45, 0.03], xscale='log')

low  = 1.0/ (stZ[0].stats.npts/float(stZ[0].stats.sampling_rate))
high = stZ[0].stats.sampling_rate/2.0
slideLow = Slider(axLowcut, 'Lowcut', low, high, valinit=low)
slideHigh = Slider(axHighcut, 'Highcut', low, high, valinit=high)

def updatePlot():
    global pltZ
    global pltN
    global pltE
    #save current x- and y-limits
    #axZx=list(axZ.get_xlim())
    #axZy=list(axZ.get_ylim())
    #axNy=list(axN.get_ylim())
    #axEy=list(axE.get_ylim())
    #filter data
    if flagFilt==True:
        if flagFiltZPH==True:
            if flagFiltTyp==0:
                filtZ=bandpassZPHSH(stZ[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
                filtN=bandpassZPHSH(stN[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
                filtE=bandpassZPHSH(stE[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
                print "Zero-Phase Bandpass: %.2f-%.2f Hz"%(slideLow.val,slideHigh.val)
            if flagFiltTyp==1:
                filtZ=bandstopZPHSH(stZ[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
                filtN=bandstopZPHSH(stN[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
                filtE=bandstopZPHSH(stE[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
                print "Zero-Phase Bandstop: %.2f-%.2f Hz"%(slideLow.val,slideHigh.val)
            if flagFiltTyp==2:
                filtZ=lowpassZPHSH(stZ[0].data,slideLow.val,df=stZ[0].stats.sampling_rate)
                filtN=lowpassZPHSH(stN[0].data,slideLow.val,df=stZ[0].stats.sampling_rate)
                filtE=lowpassZPHSH(stE[0].data,slideLow.val,df=stZ[0].stats.sampling_rate)
                print "Zero-Phase Lowpass: %.2f Hz"%(slideLow.val)
            if flagFiltTyp==3:
                filtZ=highpassZPHSH(stZ[0].data,slideHigh.val,df=stZ[0].stats.sampling_rate)
                filtN=highpassZPHSH(stN[0].data,slideHigh.val,df=stZ[0].stats.sampling_rate)
                filtE=highpassZPHSH(stE[0].data,slideHigh.val,df=stZ[0].stats.sampling_rate)
                print "Zero-Phase Highpass: %.2f Hz"%(slideHigh.val)
        elif flagFiltZPH==False:
            if flagFiltTyp==0:
                filtZ=bandpass(stZ[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
                filtN=bandpass(stN[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
                filtE=bandpass(stE[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
                print "One-Pass Bandpass: %.2f-%.2f Hz"%(slideLow.val,slideHigh.val)
            if flagFiltTyp==1:
                filtZ=bandstop(stZ[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
                filtN=bandstop(stN[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
                filtE=bandstop(stE[0].data,slideLow.val,slideHigh.val,df=stZ[0].stats.sampling_rate)
                print "One-Pass Bandstop: %.2f-%.2f Hz"%(slideLow.val,slideHigh.val)
            if flagFiltTyp==2:
                filtZ=lowpass(stZ[0].data,slideLow.val,df=stZ[0].stats.sampling_rate)
                filtN=lowpass(stN[0].data,slideLow.val,df=stZ[0].stats.sampling_rate)
                filtE=lowpass(stE[0].data,slideLow.val,df=stZ[0].stats.sampling_rate)
                print "One-Pass Lowpass: %.2f Hz"%(slideLow.val)
            if flagFiltTyp==3:
                filtZ=highpass(stZ[0].data,slideHigh.val,df=stZ[0].stats.sampling_rate)
                filtN=highpass(stN[0].data,slideHigh.val,df=stZ[0].stats.sampling_rate)
                filtE=highpass(stE[0].data,slideHigh.val,df=stZ[0].stats.sampling_rate)
                print "One-Pass Highpass: %.2f Hz"%(slideHigh.val)
        #make new plots
        pltZ.set_data(t, filtZ)
        pltN.set_data(t, filtN)
        pltE.set_data(t, filtE)
    else:
        #make new plots
        pltZ.set_data(t, stZ[0].data)
        pltN.set_data(t, stN[0].data)
        pltE.set_data(t, stE[0].data)
        print "Unfiltered Traces"
    #set to saved x- and y-limits
    #axZ.set_xlim(axZx)
    #axZ.set_ylim(axZy)
    #axN.set_ylim(axNy)
    #axE.set_ylim(axEy)
    # Update all subplots
    redraw()

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
def update(val):
    if flagFilt==1:
        updatePlot()
    else:
        pass

slideLow.on_changed(update)
slideHigh.on_changed(update)
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

# Define the event for setting P- and S-wave picks
def pick(event):
    global PZLine
    global PNLine
    global PELine
    global P
    global SZLine
    global SNLine
    global SELine
    global S
    global PZErr1Line
    global PNErr1Line
    global PEErr1Line
    global PErr1
    global PZErr2Line
    global PNErr2Line
    global PEErr2Line
    global PErr2
    global SZErr1Line
    global SNErr1Line
    global SEErr1Line
    global SErr1
    global SZErr2Line
    global SNErr2Line
    global SEErr2Line
    global SErr2
    # Set new P Pick
    if event.inaxes==axZ and event.key==' ':
        # Define global variables seen outside
        # Remove old lines from the plot before plotting the new ones
        try:
            axZ.lines.remove(PZLine)
            axN.lines.remove(PNLine)
            axE.lines.remove(PELine)
        except:
            pass
        # Plot the lines for the P pick in all three traces
        PZLine=axZ.axvline(event.xdata,color='red',linewidth=1.3)
        PNLine=axN.axvline(event.xdata,color='red',linestyle='dotted')
        PELine=axE.axvline(event.xdata,color='red',linestyle='dotted')
        # Save sample value of pick (round to integer sample value)
        P=int(round(event.xdata))
        # Update all subplots
        redraw()
        # Console output
        print "P Pick set at %i"%P
    # Set new S Pick
    if ( event.inaxes==axN or event.inaxes==axE ) and event.key==' ':
        # Define global variables seen outside
        # Remove old lines from the plot before plotting the new ones
        try:
            axZ.lines.remove(SZLine)
            axN.lines.remove(SNLine)
            axE.lines.remove(SELine)
        except:
            pass
        # Plot the lines for the S pick in all three traces
        SZLine=axZ.axvline(event.xdata,color='blue',linestyle='dotted')
        SNLine=axN.axvline(event.xdata,color='blue',linewidth=1.3)
        SELine=axE.axvline(event.xdata,color='blue',linestyle='dotted')
        # Save sample value of pick (round to integer sample value)
        S=int(round(event.xdata))
        # Update all subplots
        redraw()
        # Console output
        print "S Pick set at %i"%S
    # Remove P Pick
    if event.inaxes==axZ and event.key=='escape':
        # Try to remove all existing Pick lines and P Pick variable
        try:
            axZ.lines.remove(PZLine)
            axN.lines.remove(PNLine)
            axE.lines.remove(PELine)
            del PZLine
            del PNLine
            del PELine
            del P
            # Console output
            print "P Pick deleted"
        except:
            pass
        # Try to remove existing Pick Error 1 lines and variable
        try:
            axZ.lines.remove(PZErr1Line)
            axN.lines.remove(PNErr1Line)
            axE.lines.remove(PEErr1Line)
            del PZErr1Line
            del PNErr1Line
            del PEErr1Line
            del PErr1
        except:
            pass
        # Try to remove existing Pick Error 2 lines and variable
        try:
            axZ.lines.remove(PZErr2Line)
            axN.lines.remove(PNErr2Line)
            axE.lines.remove(PEErr2Line)
            del PZErr2Line
            del PNErr2Line
            del PEErr2Line
            del PErr2
        except:
            pass
        # Update all subplots
        redraw()
    # Remove S Pick
    if ( event.inaxes==axN or event.inaxes==axE ) and event.key=='escape':
        # Try to remove all existing Pick lines and P Pick variable
        try:
            axZ.lines.remove(SZLine)
            axN.lines.remove(SNLine)
            axE.lines.remove(SELine)
            del SZLine
            del SNLine
            del SELine
            del S
            # Console output
            print "S Pick deleted"
        except:
            pass
        # Try to remove existing Pick Error 1 lines and variable
        try:
            axZ.lines.remove(SZErr1Line)
            axN.lines.remove(SNErr1Line)
            axE.lines.remove(SEErr1Line)
            del SZErr1Line
            del SNErr1Line
            del SEErr1Line
            del SErr1
        except:
            pass
        # Try to remove existing Pick Error 2 lines and variable
        try:
            axZ.lines.remove(SZErr2Line)
            axN.lines.remove(SNErr2Line)
            axE.lines.remove(SEErr2Line)
            del SZErr2Line
            del SNErr2Line
            del SEErr2Line
            del SErr2
        except:
            pass
        # Update all subplots
        redraw()
    # Set new P Pick uncertainties
    if event.inaxes==axZ and event.key=='alt':
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
                axZ.lines.remove(PZErr1Line)
                axN.lines.remove(PNErr1Line)
                axE.lines.remove(PEErr1Line)
            except:
                pass
            # Plot the lines for the P Error pick in all three traces
            PZErr1Line=axZ.axvline(event.xdata,ymin=0.25,ymax=0.75,color='red')
            PNErr1Line=axN.axvline(event.xdata,ymin=0.25,ymax=0.75,color='red',linestyle='dotted')
            PEErr1Line=axE.axvline(event.xdata,ymin=0.25,ymax=0.75,color='red',linestyle='dotted')
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
                axZ.lines.remove(PZErr2Line)
                axN.lines.remove(PNErr2Line)
                axE.lines.remove(PEErr2Line)
            except:
                pass
            # Plot the lines for the P Error pick in all three traces
            PZErr2Line=axZ.axvline(event.xdata,ymin=0.25,ymax=0.75,color='red')
            PNErr2Line=axN.axvline(event.xdata,ymin=0.25,ymax=0.75,color='red',linestyle='dotted')
            PEErr2Line=axE.axvline(event.xdata,ymin=0.25,ymax=0.75,color='red',linestyle='dotted')
            # Save sample value of error pick (round to integer sample value)
            PErr2=int(round(event.xdata))
            # Update all subplots
            redraw()
            # Console output
            print "P Error Pick 2 set at %i"%PErr2
    # Set new S Pick uncertainties
    if ( event.inaxes==axN or event.inaxes==axE ) and event.key=='alt':
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
                axZ.lines.remove(SZErr1Line)
                axN.lines.remove(SNErr1Line)
                axE.lines.remove(SEErr1Line)
            except:
                pass
            # Plot the lines for the S Error pick in all three traces
            SZErr1Line=axZ.axvline(event.xdata,ymin=0.25,ymax=0.75,color='blue',linestyle='dotted')
            SNErr1Line=axN.axvline(event.xdata,ymin=0.25,ymax=0.75,color='blue')
            SEErr1Line=axE.axvline(event.xdata,ymin=0.25,ymax=0.75,color='blue',linestyle='dotted')
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
                axZ.lines.remove(SZErr2Line)
                axN.lines.remove(SNErr2Line)
                axE.lines.remove(SEErr2Line)
            except:
                pass
            # Plot the lines for the S Error pick in all three traces
            SZErr2Line=axZ.axvline(event.xdata,ymin=0.25,ymax=0.75,color='blue',linestyle='dotted')
            SNErr2Line=axN.axvline(event.xdata,ymin=0.25,ymax=0.75,color='blue')
            SEErr2Line=axE.axvline(event.xdata,ymin=0.25,ymax=0.75,color='blue',linestyle='dotted')
            # Save sample value of error pick (round to integer sample value)
            SErr2=int(round(event.xdata))
            # Update all subplots
            redraw()
            # Console output
            print "S Error Pick 2 set at %i"%SErr2

# Define zoom events for the mouse scroll wheel
def zoom(event):
    # Zoom in on scroll-up
    if event.button=='up' and flagWheelZoom:
        # Calculate and set new axes boundaries from old ones
        (left,right)=axZ.get_xbound()
        left+=(event.xdata-left)/2
        right-=(right-event.xdata)/2
        axZ.set_xbound(lower=left,upper=right)
        # Update all subplots
        redraw()
    # Zoom out on scroll-down
    if event.button=='down' and flagWheelZoom:
        # Calculate and set new axes boundaries from old ones
        (left,right)=axZ.get_xbound()
        left-=(event.xdata-left)/2
        right+=(right-event.xdata)/2
        axZ.set_xbound(lower=left,upper=right)
        # Update all subplots
        redraw()

# Define zoom reset for the mouse button 2 (always scroll wheel!?)
def zoom_reset(event):
    if event.button==2:
        # Use Z trace limits as boundaries
        axZ.set_xbound(lower=xMin,upper=xMax)
        axZ.set_ybound(lower=yMin,upper=yMax)
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

# Activate all mouse/key/Cursor-events
keypress = fig.canvas.mpl_connect('key_press_event', pick)
keypressWheelZoom = fig.canvas.mpl_connect('key_press_event', switchWheelZoom)
scroll = fig.canvas.mpl_connect('scroll_event', zoom)
scroll_button = fig.canvas.mpl_connect('button_press_event', zoom_reset)
#cursorZ = mplCursor(axZ, useblit=True, color='black', linewidth=1, ls='dotted')
#cursorN = mplCursor(axN, useblit=True, color='black', linewidth=1, ls='dotted')
#cursorE = mplCursor(axE, useblit=True, color='black', linewidth=1, ls='dotted')
fig.canvas.toolbar.pan()
multicursor = mplMultiCursor(fig.canvas,(axZ,axN,axE), useblit=True, color='black', linewidth=1, ls='dotted')


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

plt.show()
