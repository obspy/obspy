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
import matplotlib.pyplot as plt
from matplotlib.mlab import detrend_linear as detrend
from obspy.core import read, UTCDateTime, Stream, AttribDict
from obspy.signal import recStalta, triggerOnset, seisSim, cornFreq2Paz, bandpass
from obspy.seishub import Client

# XXX START = UTCDateTime("2010-01-26T00:00:00")
START = UTCDateTime("2010-06-22T07:00:00.000000Z") # XXX
END = UTCDateTime("2010-09-26T00:00:00")

NET = "BW"
STATIONS = ("DHFO", "UH1", "UH2", "UH3", "UH4")
CHANNEL = "EHZ"
PAR = dict(LOW=10.0, # bandpass low corner
           HIGH=20.0, # bandpass high corner
           STA=0.5, # length of sta in seconds
           LTA=10, # length of lta in seconds
           ON=3.5, # trigger on threshold
           OFF=1, # trigger off threshold
           ALLOWANCE=2, # time in seconds to extend trigger-off time
           MAXLEN=10, # maximum trigger length in seconds
           MIN_STATIONS=3, # minimum of coinciding stations for alert
           ALLOW_LESS_STATIONS=True) # allow trigger with less stations than MIN_STATIONS if all stations trigger
PAR = AttribDict(PAR)
SUMMARY = "/scratch/uh_trigger_extra/uh_trigger.txt"
PLOTDIR = "/scratch/uh_trigger_extra/"
MAILTO = ()


client = Client()

# search given timespan one hour at a time, set initial T1 one hour earlier
T1 = START - (60 * 60 * 1)
while T1 < END:
    T1 += (60 * 60 * 1)
    T2 = T1 + (60 * 60 * 1)

    st = Stream()
    num_stations = 0
    for station in STATIONS:
        try:
            # we request 60s more at start and end and cut them off later to avoid
            # a false trigger due to the tapering during instrument correction
            tmp = client.waveform.getWaveform(NET, station, "", CHANNEL, T1 - 60,
                                              T2 + 60, getPAZ=True,
                                              getCoordinates=True)
            st.extend(tmp)
            num_stations += 1
        except Exception, e:
            if "No waveform data available" in str(e):
                continue
            raise

    summary = []
    summary.append("#" * 79)
    summary.append("######## %s  ---  %s ########" % (T1, T2))
    summary.append("#" * 79)
    summary.append(str(st))

    if not st:
        summary = "\n".join(summary)
        #summary += "\n" + "\n".join(("%s=%s" % (k, v) for k, v in PAR.items()))
        open(SUMMARY, "at").write(summary + "\n")
        continue
    
    # merging
    try:
        st.merge(0)
    except Exception, e:
        summary.append("Error while merging:")
        summary.append(str(e))
        summary = "\n".join(summary)
        summary += "\n" + "\n".join(("%s=%s" % (k, v) for k, v in PAR.items()))
        open(SUMMARY, "at").write(summary + "\n")
        continue

    # preprocessing, keep original data for plotting at end
    for tr in st:
        tr.data = detrend(tr.data)
    st.simulate(paz_remove="self", paz_simulate=cornFreq2Paz(1.0), remove_sensitivity=False)
    st.sort()
    st_trigger = st.copy()
    st_trigger.filter("bandpass", freqmin=PAR.LOW, freqmax=PAR.HIGH, corners=1, zerophase=True)
    st.trim(T1, T2)
    st_trigger.trim(T1, T2)
    st_trigger.trigger("recstalta", sta=PAR.STA, lta=PAR.LTA)
    summary.append(str(st))

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
    while len(trigger_list) > 1:
        on, off, sta = trigger_list[0]
        stations = set()
        stations.add(sta)
        for i in xrange(1, len(trigger_list)):
            tmp_on, tmp_off, tmp_sta = trigger_list[i]
            if tmp_on < off + PAR.ALLOWANCE:
                stations.add(tmp_sta)
                # allow sets of triggers that overlap only on subsets of all
                # stations (e.g. A overlaps with B and B overlaps with C => ABC)
                off = max(off, tmp_off)
            else:
                break
        # process event if enough stations reported it.
        # a minimum of PAR.MIN_STATIONS must have triggered together.
        # if PAR.ALLOW_LESS_STATIONS is set, then an event also is issued if
        # ALL stations have triggered but the number of available stations
        # falls short of PAR.MIN_STATIONS.
        if len(stations) >= (PAR.ALLOW_LESS_STATIONS and min(num_stations, PAR.MIN_STATIONS) or PAR.MIN_STATIONS):
            event = (UTCDateTime(on), off - on, tuple(stations))
            summary.append("%s %04.1f %s" % event)
            tmp = st.slice(UTCDateTime(on), UTCDateTime(off))
            outfilename = "%s/%s_%s-%s_%s.png" % (PLOTDIR, UTCDateTime(on), len(stations), num_stations, "-".join(stations))
            tmp.plot(outfile=outfilename)
            mutt += ("-a", outfilename)
        # shorten trigger_list and go on
        # index i marks the index of the next non-matching pick
        trigger_list = trigger_list[i:]

    summary = "\n".join(summary)
    summary += "\n" + "\n".join(("%s=%s" % (k, v) for k, v in PAR.items()))
    #print summary
    open(SUMMARY, "at").write(summary + "\n")
    # send emails
    if MAILTO:
        mutt += MAILTO
        subprocess.Popen(mutt, stdin=subprocess.PIPE).communicate(summary)

    plt.close('all')
    del st
    del tmp
    del st_trigger
    del summary
    del tr
    del trigger_list
    del trigger_sample_list
    del mutt
