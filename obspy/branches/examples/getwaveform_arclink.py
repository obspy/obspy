# -*- coding: utf-8 -*-
# 2009-11-27 Moritz

from obspy.core import UTCDateTime
from obspy.core import Stream
from obspy.arclink import Client
import matplotlib
matplotlib.rc('figure.subplot', hspace=0.8) # set default
import matplotlib.pyplot as plt

# Initialize ArcLink client
client= Client()

# Initialize data as emtpy list, set the corresponding stations
# Find available networks and stations on www.webdc.eu -> Query
# Global earthquake Sumatra
t = UTCDateTime(2004,12,26,01,00,00)
network = 'GR'
stations = ['FUR','BFO','BRG','BSEG','BUG']
channel = 'LHZ'

## Retrieve stations of network GR and immediately save waveforms
data = [] # initialize as empty list
for i, station in enumerate(stations):
    try:
        data.append(client.getWaveform(network, station, '', channel, t, t+1800))
        print "Retrieved data for station", station
    except:
        print "Cannot retrieve data for station", station


# Plot all the seismograms
m = len(stations)
plt.clf()
for i, st in enumerate(data):
    tr = st[0]
    plt.subplot(m,1,i+1) # python starts counting with 0
    plt.plot(tr.data)
    plt.title("%s %s" % (tr.stats.station,tr.stats.starttime))
plt.show()

