# -*- coding: utf-8 -*-
# 2009-11-27 Moritz

from obspy.core import UTCDateTime
from obspy.arclink import Client

def save_all(network, stations, channel, start, end):
    """
    Save data as Miniseed for each station in stations

    @param network: The network id
    @param stations: List of stations ids
    @param channel: The channel id
    @param start: start datetime as UTCDateTime
    @param end: end datetime as UTCDateTime
    """
    client = Client()
    for station in stations:
        file = "%s.%s.%s.%s.D.%s" % (network ,station,'',channel,t.strftime("%Y.%j"))
        try:
            client.saveWaveform(file, network, station, '', channel, start, end)
            print "Saved data from station", station
        except:
            print "Cannot retieve data from station", station


#
# Sumatra earthquake and corresponding stations
#
t = UTCDateTime(2004,12,26,01,00,00)
network = 'GR'
stations = ['FUR','BFO','BRG','BSEG','BUG']
channel = 'LHZ'
save_all(network, stations, channel, t, t + (5*3600)) #5 hours

#
# Local earthquake Bavaria and corresponding stations
#
t = UTCDateTime("2008-04-17 16:00:20") 
network = 'BW'
stations = ['RJOB','RNON','RMOA']
channel = 'EHZ'
save_all(network, stations, channel, t+20, t+80) #1 minute

#
# Seismic noise, nothing in it
#
t = UTCDateTime("2008-04-16 16:00:20") 
network = 'BW'
stations = ['RJOB','RNON','RMOA']
channel = 'EHZ'
save_all(network, stations, channel, t+20, t+80) #1 minute
