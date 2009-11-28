# -*- coding: utf-8 -*-
# 2009-11-27 Moritz

from obspy.core import UTCDateTime
from obspy.arclink import Client
import pickle, time

client = Client()

def save_all(network, stations, channel, start, end):
    """
    Save data as MiniSEED and PAZ as Python dump

    @param network: The network id
    @param stations: List of stations ids
    @param channel: The channel id
    @param start: start datetime as UTCDateTime
    @param end: end datetime as UTCDateTime
    """
    for station in stations:
        mseed_file = "%s.%s.%s.%s.D.%s" % (network ,station,'',channel,t.strftime("%Y.%j"))
        paz_file = "%s.%s.%s.%s.D.paz" % (network ,station,'',channel)
        time.sleep(1) # avoid too fast requests
        try:
            client.saveWaveform(mseed_file, network, station, '', channel, start, end)
            time.sleep(1)
            paz = client.getPAZ(network, station, '', channel, start, end)
            pickle.dump(paz, open(paz_file,'wb'))
            print "Saved data and paz for station", station
        except:
            print "Cannot retieve data and/or paz for station", station


#
# Sumatra earthquake and corresponding stations
#
t = UTCDateTime(2004,12,26,01,00,00)
network = 'GR'
stations = ['FUR','BFO','BRG','BSEG','BUG']
channel = 'LHZ'
save_all(network, stations, channel, t, t + (5*3600)) #5 hours

#
# China earthquake
#
t = UTCDateTime("2008,133,5:17:48.640")
save_all('GR',['FUR'], 'LHZ', t, t+(3600*23.9))

#
# Local earthquake Bavaria and corresponding stations
#
t = UTCDateTime("2008-04-17 16:00:20") 
network = 'BW'
stations = ['RJOB','RNON','RMOA']
channel = 'EHZ'
save_all(network, stations, channel, t, t+60) #1 minute

#
# Seismic noise, nothing in it
#
t = UTCDateTime("2008-04-16 16:00:20") 
network = 'BW'
stations = ['RJOB','RNON','RMOA']
channel = 'EHZ'
save_all(network, stations, channel, t, t+60) #1 minute

