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
import subprocess
import numpy as np
from obspy.core import read, UTCDateTime, Stream, AttribDict
from obspy.signal import recStalta, triggerOnset, seisSim, cornFreq2Paz, bandpass
from obspy.seishub import Client
from matplotlib.mlab import detrend_linear as detrend


NET = "BW"
STATIONS = ("DHFO", "UH1", "UH2", "UH3", "UH4")
CHANNEL = "EHZ"
# search from 2h before now to 1h before now
T1 = UTCDateTime() - (60 * 60 * 2)
T2 = T1 + (60 * 60 * 1) + 30
PAR = dict(LOW=10.0, # bandpass low corner
           HIGH=20.0, # bandpass high corner
           STA=0.5, # length of sta in seconds
           LTA=10, # length of lta in seconds
           ON=3.5, # trigger on threshold
           OFF=1, # trigger off threshold
           ALLOWANCE=1, # time in seconds to extend trigger-off time
           MAXLEN=10, # maximum trigger length in seconds
           MIN_STATIONS=3) # minimum of coinciding stations for alert
PAR = AttribDict(PAR)
SUMMARY = "/scratch/uh_trigger/uh_trigger.txt"
PLOTDIR = "/scratch/uh_trigger/"
MAILTO = ("megies",)


client = Client()

st = Stream()
num_stations = 0
for station in STATIONS:
    try:
        # we request 60s more at start and end and cut them off later to avoid
        # a false trigger due to the tapering during instrument correction
        tmp = client.waveform.getWaveform(NET, station, "", CHANNEL, T1 - 60,
                                          T2 + 60, getPAZ=True,
                                          getCoordinates=True)
    except:
        continue
    st.extend(tmp)
    num_stations += 1

if not st:
    pass # XXX print/mail warning

summary = []
summary.append("#" * 79)
summary.append("######## %s  ---  %s ########" % (T1, T2))
summary.append("#" * 79)
summary.append(str(st))

# preprocessing, backup original data for plotting at end
st.merge(0)
for tr in st:
    tr.data = detrend(tr.data)
st.simulate(paz_remove="self", paz_simulate=cornFreq2Paz(1.0), remove_sensitivity=False)
st.sort()
st_trigger = st.copy()
st_trigger.filter("bandpass", freqmin=PAR.LOW, freqmax=PAR.HIGH, corners=1, zerophase=True)
st.trim(T1, T2)
st_trigger.trim(T1, T2)
st_trigger.trigger("recstalta", sta=PAR.STA, lta=PAR.LTA)

# do the triggering
trigger_list = []
for tr in st_trigger:
    tr.stats.channel = "recstalta"
    max_len = PAR.MAXLEN * tr.stats.sampling_rate
    trigger_sample_list = triggerOnset(tr.data, PAR.ON, PAR.OFF, max_len=max_len)
    for on, off in trigger_sample_list:
         begin = tr.stats.starttime + float(on) / tr.stats.sampling_rate
         end = tr.stats.starttime + float(off) / tr.stats.sampling_rate
         trigger_list.append((begin.timestamp, end.timestamp, tr.stats.station))
trigger_list.sort()

# merge waveform and trigger stream for plotting
# the normalizations are done because the triggers have a completely different
# scale and would not be visible in the plot otherwise...
st.filter("bandpass", freqmin=1.0, freqmax=20.0, corners=1, zerophase=True)
st.normalize(global_max=False)
st_trigger.normalize(global_max=True)
st.extend(st_trigger)

# coincidence part, work through sorted trigger list...
mutt = ["mutt", "-s", "UH Alert  %s -- %s" % (T1, T2)]
last_off_time = 0
while len(trigger_list) > 1:
    on, off, sta = trigger_list[0]
    stations = []
    stations.append(sta)
    for i in xrange(1, len(trigger_list)):
        tmp_on, tmp_off, tmp_sta = trigger_list[i]
        # skip retriggering of already present station in current trigger
        if tmp_sta in stations:
            continue
        if tmp_on < off + PAR.ALLOWANCE:
            stations.append(tmp_sta)
            # allow sets of triggers that overlap only on subsets of all
            # stations (e.g. A overlaps with B and B overlaps with C => ABC)
            off = max(off, tmp_off)
        else:
            break
    # process event if enough stations reported it
    if len(set(stations)) >= PAR.MIN_STATIONS:
        if off != last_off_time:
            event = (UTCDateTime(on), off - on, stations)
            summary.append("%s %04.1f %s" % event)
            tmp = st.slice(UTCDateTime(on) - 1, UTCDateTime(off))
            outfilename = "%s/%s_%.1f_%s-%s_%s.png" % (PLOTDIR, UTCDateTime(on), off - on, len(stations), num_stations, "-".join(stations))
            tmp.plot(outfile=outfilename)
            mutt += ("-a", outfilename)
            last_off_time = off
    # shorten trigger_list and go on
    # index i marks the index of the next non-matching pick
    trigger_list = trigger_list[1:]

summary = "\n".join(summary)
summary += "\n" + "\n".join(("%s=%s" % (k, v) for k, v in PAR.items()))
#print summary
open(SUMMARY, "at").write(summary + "\n")
# send emails
if MAILTO:
    mutt += MAILTO
    subprocess.Popen(mutt, stdin=subprocess.PIPE).communicate(summary)
