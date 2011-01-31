#!/usr/bin/env python
"""
Recursive STA/LTA trigger for Unterhaching subnet.
Scans yesterdays files in mseed archive for events.
"""
# 2009-07-23 Moritz; PYTHON2.5 REQUIRED
# 2009-11-25 Moritz
# 2010-09 Tobi

import os
import sys
import glob
import numpy as np
from obspy.core import read, UTCDateTime
from obspy.signal import recStalta, triggerOnset, seisSim, cornFreq2Paz, bandpass
from obspy.xseed import Parser
from matplotlib.mlab import detrend_linear as detrend

def s2p(sec, trace):
    """
    Convert seconds to samples with the sampling rate of trace object
    """
    return int(sec * trace.stats.sampling_rate)

def trId(stats):
    return stats.endtime, "%s%s%s%f" % (stats.network, stats.station,
                                        stats.channel, stats.sampling_rate)

BASEDIR = "/bay200/mseed_online/archive/"
BASEDIR_DATALESS = "/bay200/dataless/"
NET = "BW"
STATIONS = ("DHFO", "UH1", "UH2", "UH3", "UH4")
CHANNEL = "EHZ"
TIME = UTCDateTime() - (60 * 60 * 24) # search yesterday
TIME = UTCDateTime("2010-09-20T12:00:00") # XXX
LOW = 10.0 # bandpass low corner
HIGH = 20.0 # bandpass high corner
STA = 0.5 # length of sta in seconds
LTA = 10 # length of lta in seconds
ON = 3.5 # trigger on threshold
OFF = 1 # trigger off threshold
ALLOWANCE = 3
MIN_STATIONS = 3 # minimum of coincident stations for alert
SUMMARY = "/scratch/uh_trigger.txt"


mseed_files = []
parsers = []
for station in STATIONS:
    # waveforms
    dir = os.path.join(BASEDIR, str(TIME.year), NET, station, CHANNEL)
    # XXX maybe read the day before/after to make sure we dont miss data around
    # 00:00
    files = glob.glob("%s*/*.%s" % (dir, TIME.julday))
    mseed_files.extend(files)
    # metadata
    files = glob.glob("%s/dataless*%s" % (BASEDIR_DATALESS, station))
    for file in files:
        parsers.append(Parser(file))

if not mseed_files:
    pass # XXX print/mail warning

inst = cornFreq2Paz(1.0)
nfft = 4194304 # next nfft of 5h
last_endtime = 0
last_id = "--"

trigger_list = []
summary = []
summary.append("#" * 79)
for file in mseed_files:
    summary.append(file)
    try:
        st = read(file, "MSEED")
        T1 = UTCDateTime("2010-09-20T00:00:00") # XXX
        T2 = UTCDateTime("2010-09-20T04:00:00") # XXX
        st.trim(T1, T2) # XXX
        #st.trim(endtime=st[0].stats.starttime+5000) # XXX
        summary.append(str(st))
    except:
        summary.append("skipped!")
        continue
    for tr in st:
        stats = tr.stats
        for parser in parsers:
            try:
                tr.stats.paz = parser.getPAZ(tr.id, tr.stats.starttime)
                tr.stats.coordinates = parser.getCoordinates(tr.id, tr.stats.starttime)
                break
            except:
                pass
        if not getattr(tr.stats, "paz", None):
            summary.append("found no metadata for %s. skipping!" % tr.id)
            continue
        # Cannot process a whole day file, split it in smaller junks
        overlap = s2p(30.0, tr)
        olap = overlap
        samp = 0
        df = tr.stats.sampling_rate
        if trId(tr.stats)[1] != last_id or tr.stats.starttime - last_endtime > 1.0 / df:
            data_buf = np.array([], dtype='float64')
            olap = 0
        while samp < tr.stats.npts:
            data = tr.data[samp:samp + nfft - olap].astype('float64')
            data = np.concatenate((data_buf, data))
            data = detrend(data)
            # Correct for frequency response of instrument
            data = seisSim(data, df, paz_remove=tr.stats.paz, paz_simulate=inst, remove_sensitivity=True)
            # XXX is removed in seisSim... ?!
            # XXX data /= (paz['sensitivity'] / 1e9)  #V/nm/s correct for overall sensitivity
            data = bandpass(data, LOW, HIGH, df)
            data = recStalta(data, s2p(STA, tr), s2p(LTA, tr))
            picked_values = triggerOnset(data, ON, OFF, max_len=overlap)
            #
            for i, j in picked_values:
                 begin = tr.stats.starttime + float(i + samp - olap) / df
                 end = tr.stats.starttime + float(j + samp - olap) / df
                 trigger_list.append((begin.timestamp, end.timestamp, tr.stats.station))
            olap = overlap # only needed for first time in loop
            samp += nfft - overlap
            data_buf = data[-overlap:]
        last_endtime, last_id = trId(tr.stats)

###############################################################################
# start of coincidence part
###############################################################################
trigger_list.sort()
#print [(UTCDateTime(i[0]).isoformat(), UTCDateTime(i[1]).isoformat(), i[2]) for i in trigger_list]
import pprint
pprint.pprint([(UTCDateTime(i[0]).isoformat(), UTCDateTime(i[1]).isoformat(), i[2]) for i in trigger_list])

while len(trigger_list) > 1:
    coinc = set()
    on, off, sta = trigger_list[0]
    coinc.add(sta)
    for i in range(1, len(trigger_list)):
        tmp_on, tmp_off, tmp_sta = trigger_list[i]
        if tmp_on < off + ALLOWANCE:
            #import ipdb; ipdb.set_trace()
            coinc.add(tmp_sta)
            off = max(off, tmp_off)
        else:
            break
    if len(coinc) >= MIN_STATIONS:
        summary.append("%s %04.1f %s" % (UTCDateTime(on), off - on, coinc))
    # index i marks the index of the next non-matching pick
    trigger_list = trigger_list[i:]

summary = "\n".join(summary)
print summary
#open(SUMMARY, "at").write(summary)
