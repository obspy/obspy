from obspy.core import UTCDateTime
from obspy.core import Stream
from obspy.arclink import Client
import matplotlib
matplotlib.rc('figure.subplot', hspace=0.8) # set default
import matplotlib.pyplot as plt

# Initialize ArcLink client
client= Client()
t = UTCDateTime(2004,12,26,01,00,00)

# Initialize data as emtpy list, set the corresponding stations
# Find available networks and stations on www.webdc.eu -> Query
data = []
stations = ['FUR','BFO','BRG','BSEG','BUG']
m = len(stations)

## Retrieve stations of network GR and immediately save waveforms
#data = []
#for i, station in enumerate(stations):
#    data.append(client.getWaveform('GR', station, '', 'LHZ', t, t+1800))
#    print "Retrieved data from station", station
#
## Plot all the seismograms
#plt.clf()
#for i, st in enumerate(data):
#    tr = st[0]
#    plt.subplot(m,1,i+1) # python starts counting with 0
#    plt.plot(tr.data)
#    plt.title("%s %s" % (tr.stats.station,tr.stats.starttime))
#plt.show()


# Save data as Miniseed
for station in stations:
    file = "%s.%s.%s.%s.D.%s" % ('GR',station,'','LHZ',t.strftime("%Y.%j"))
    client.saveWaveform(file, 'GR', station, '', 'LHZ', t, t+(3600*5))
    print "Saved data from station", station
