#!/usr/bin/env python
"""
Recursive STA/LTA trigger for Unterhaching subnet.
"""
# 2009-07-23 Moritz; PYTHON2.5 REQUIRED
# 2009-11-25 Moritz
# 2010-09 Tobi

import matplotlib
matplotlib.use("AGG")

import os
import sys
import glob
import numpy as np
from obspy.core import read, UTCDateTime, Stream
from obspy.signal import recStalta, triggerOnset, seisSim, cornFreq2Paz, bandpass
from obspy.seishub import Client
from matplotlib.mlab import detrend_linear as detrend


NET = "BW"
STATIONS = ("DHFO", "UH1", "UH2", "UH3", "UH4")
CHANNEL = "EHZ"
# search from 2h before now to 1h before now
T1 = UTCDateTime() - (60 * 60 * 2)
T2 = T1 + (60 * 60 * 1)
LOW = 10.0 # bandpass low corner
HIGH = 20.0 # bandpass high corner
STA = 0.5 # length of sta in seconds
LTA = 10 # length of lta in seconds
ON = 3.5 # trigger on threshold
OFF = 1 # trigger off threshold
ALLOWANCE = 3 # time in seconds to extend trigger-off time
MAXLEN = 10 # maximum trigger length in seconds
MIN_STATIONS = 3 # minimum of coinciding stations for alert
SUMMARY = "/scratch/uh_trigger.txt"
PLOTDIR = "/scratch/uh_trigger/"


client = Client()

st = Stream()
for station in STATIONS:
    try:
        # we request 60s more at start and end and cut them off later to avoid
        # a false trigger due to the tapering during instrument correction
        tmp = client.waveform.getWaveform(NET, station, "", CHANNEL, T1 - 60,
                                          T2 + 60, getPAZ=True,
                                          getCoordinates=True)
    except:
        pass
    st.extend(tmp)

if not st:
    pass # XXX print/mail warning

summary = []
summary.append("#" * 79)
summary.append("######## %s  ---  %s ########" % (T1, T2))
summary.append("#" * 79)

# preprocessing, backup original data for plotting at end
st.merge(0)
for tr in st:
    tr.data = detrend(tr.data)
st.simulate(paz_remove="self", paz_simulate=cornFreq2Paz(1.0), remove_sensitivity=False)
st.sort()
st_trigger = st.copy()
st_trigger.filter("bandpass", freqmin=LOW, freqmax=HIGH, corners=1, zerophase=True)
st.trim(T1, T2)
st_trigger.trim(T1, T2)
st_trigger.trigger("recstalta", sta=STA, lta=LTA)
summary.append(str(st))

# do the triggering
trigger_list = []
for tr in st_trigger:
    tr.stats.channel = "recstalta"
    max_len = MAXLEN * tr.stats.sampling_rate
    trigger_sample_list = triggerOnset(tr.data, ON, OFF, max_len=max_len)
    for on, off in trigger_sample_list:
         begin = tr.stats.starttime + float(on) / tr.stats.sampling_rate
         end = tr.stats.starttime + float(off) / tr.stats.sampling_rate
         trigger_list.append((begin.timestamp, end.timestamp, tr.stats.station))
trigger_list.sort()

# merge waveform and trigger stream for plotting
# the normalizations are done because the triggers have a completely different
# scale and would not be visible in the plot otherwise...
st.normalize(global_max=False)
st_trigger.normalize(global_max=True)
st.extend(st_trigger)

# coincidence part, work through sorted trigger list...
while len(trigger_list) > 1:
    on, off, sta = trigger_list[0]
    stations = set()
    stations.add(sta)
    for i in xrange(1, len(trigger_list)):
        tmp_on, tmp_off, tmp_sta = trigger_list[i]
        if tmp_on < off + ALLOWANCE:
            stations.add(tmp_sta)
            # allow sets of triggers that overlap only on subsets of all
            # stations (e.g. A overlaps with B and B overlaps with C => ABC)
            off = max(off, tmp_off)
        else:
            break
    # process event if enough stations reported it
    if len(stations) >= MIN_STATIONS:
        event = (UTCDateTime(on), off - on, tuple(stations))
        summary.append("%s %04.1f %s" % event)
        tmp = st.slice(UTCDateTime(on), UTCDateTime(off))
        tmp.plot(outfile="%s/%s.png" % (PLOTDIR, UTCDateTime(on)))
    # shorten trigger_list and go on
    # index i marks the index of the next non-matching pick
    trigger_list = trigger_list[i:]

summary = "\n".join(summary)
print summary
open(SUMMARY, "at").write(summary + "\n")
