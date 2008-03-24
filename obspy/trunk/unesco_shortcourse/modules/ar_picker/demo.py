#!/usr/bin/env python
#
# Pitsa config of /import/ambrym-data/yogya_06/merapi_base/work/pitsa.cfg
# or /home/jowa/setup/pitsa.cfg
#AR_S
#AR_FLOW
#1.0
#AR_FHIGH
#20.0
#LTA_P
#1.0
#STA_P
#0.05
#LTA_S
#4.0
#STA_S
#1.0
#VARLEN_P
#0.1
#VARLEN_S
#0.1
#ARCOEF_P
#2
#ARCOEF_S
#8


from numpy import *
from ext_arpicker import *
a=fromfile("loc_RJOB20050801145719850.z",sep="\n",dtype=float32)
b=fromfile("loc_RJOB20050801145719850.n",sep="\n",dtype=float32)
c=fromfile("loc_RJOB20050801145719850.e",sep="\n",dtype=float32)
t = arange(len(a),dtype=float)/200

#a, b, c, sample_rate, f1, f2, lta_p, sta_p, lta_s, sta_s, m_p,
#m_s, l_p, l_s # give times in seconds!!!!
[ptime,stime] = arPick(a,b,c,200,1.0,20.0,1.0,0.1,4,1,2,8,0.1,0.2)
#ptime = float(time[0])/200
#stime = float(time[1])/200

from pylab import *
def minsec(x):
	return '%6.3fs' % (x)
def pick(event):
	if event.key=='p' and event.inaxes is not None:
		print event.xdata, event.ydata

hold(True)

ax = subplot(311)
ax.fmt_xdata = minsec
plot(t,a,'k')
# Plot phase pick pptime
plot([ptime,ptime],array(ylim())*1/4,color='b',linewidth=1)
text(ptime,ylim()[1]*1/4,'P',color='b',fontsize=12)

ax2=subplot(312, sharex=ax)
ax2.fmt_xdata = minsec
plot(t,b,'k')
plot([stime,stime],array(ylim())*1/4,color='r',linewidth=1)
text(stime,ylim()[1]*1/4,'S',color='r',fontsize=12)

ax3=subplot(313, sharex=ax)
ax3.fmt_xdata = minsec
plot(t,c,'k')
plot([stime,stime],array(ylim())*1/4,color='r',linewidth=1)
text(stime,ylim()[1]*1/4,'S',color='r',fontsize=12)

hold(False)
connect('key_press_event', pick)
show()



