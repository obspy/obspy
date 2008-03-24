#!/usr/bin/env python
#
######################################################################
# ext_pk_mbaer.baerPick = baerPick(...)
#   pptime = baerPick(reltrc,pfm,samplespersec,tdownmax,tupevent,thrshl1,
#                       thrshl2,preset_len,p_dur) 
#     pptime       : sample number of parrival 
#     reltrc       : timeseries as floating data, possibly filtered
#     pfm          : direction of first motion (U or D) 
#     samplespersec: no of samples per second 
#     tdownmax     : if dtime exceeds tdownmax, the trigger is examined for 
#                    validity 
#     tupevent     : min nr of samples for itrm to be accepted as a pick 
#     thrshl1      : threshold to trigger for pick (c.f. paper) 
#     thrshl2      : threshold for updating sigma  (c.f. paper) 
#     preset_len   : no of points taken for the estimation of variance of 
#                    SF(t) on preset() 
#     p_dur        : p_dur defines the time interval for which the maximum 
#                    amplitude is evaluated Originally set to 6 secs
######################################################################
# pitsa conf
#PRESETDUR
#1.
#P_DUR
#5.
#TDOWNMAX
#0.5
#TUPEVENT
#1.
#THRSHL1
#7
#THRSHL2
#12
###################

from numpy import *
from ext_pk_mbaer import *
a=fromfile("loc_RJOB20020325181100.ascii",sep="\n",dtype=float32)
t = arange(len(a),dtype=float)/200
pptime = float(baerPick(a,200,20,60,7,12,100,100)[0])/200
pfm = baerPick(a,200,20,60,7,12,100,100)[1]
print "P detection at %fs; phase quality %s" % (pptime,pfm)

from pylab import *
def minsec(x):
	return '%6.3fs' % (x)
def pick(event):
	print event.xdata, event.ydata

ax = subplot(111)
plot(t,a,'k')
ax.fmt_xdata = minsec
hold(True)
# Plot phase pick pptime
plot([pptime,pptime],array(ylim())*1/4,color='b',linewidth=1)
text(pptime,ylim()[1]*1/4,'$P_{%s}$'%pfm,color='b',fontsize=14)
connect('button_press_event', pick)
hold(False)
show()



