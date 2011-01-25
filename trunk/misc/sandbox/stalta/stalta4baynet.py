#!/usr/bin/env python
"""
USAGE: stalta4baynet.py mseedfile1 mseedfile2 mseedfile3 ...

STA/LTA trigger for Baynet. \n E.g file wildcard:
/bay200/mseed_online/archive/2009/BW/R*/EHZ.D/BW.R*..EHZ.D.2009.198

obspy must be in the PYTHONPATH
"""
# 2009-07-23 Moritz; PYTHON2.5 REQUIRED
# 2009-11-25 Moritz

import sys
from obspy.core import read
from obspy.signal import recStalta, triggerOnset, seisSim, pazToFreqResp
from obspy.signal import cornFreq2Paz
from obspy.seishub.client import Client
import numpy as np
from matplotlib.mlab import detrend_linear as detrend

def s2p(sec, trace):
    """Convert seconds to samples with the sampling rate of trace object"""
    return int(sec * trace.stats.sampling_rate)

def trId(stats):
    return stats.endtime, "%s%s%s%f" % (stats.network, stats.station,
                                        stats.channel, stats.sampling_rate)

mseed_files = sys.argv[1:]
if mseed_files == []:
    print __doc__
    sys.exit(1)

client = Client()

inst = cornFreq2Paz(1.0)
nfft = 4194304 # next nfft of 5h
station_list = []
last_endtime, last_id = 0, "--"
for file in mseed_files:
    print "\n", file,
    try:
        stream = read(file)
    except:
        continue
    stats = stream[0].stats
    pick_file = "%s_%s_%s.picks" % (stats.starttime.year,
                                    stats.starttime.strftime("%j"),
                                    stats.station)
    if not stats.station in station_list:
        station_list.append(stats.station)
    f = open(pick_file, 'w')
    for tr in stream:
        try:
            paz = client.station.getPAZ(tr.stats.network, tr.stats.station,
                                        tr.stats.starttime)
        except ValueError:
            print "Cannot process station %s, no RESP file given" % tr.stats.station
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
            data = seisSim(data, tr.stats.sampling_rate, paz, inst_sim=inst)
            data /= (paz['sensitivity'] / 1e9)  #V/nm/s correct for overall sensitivity
            data = recStalta(data, s2p(2.5, tr), s2p(10.0, tr))
            picked_values = triggerOnset(data, 3.0, 0.5, max_len=overlap)
            #
            for i, j in picked_values:
                 begin = tr.stats.starttime + float(i + samp - olap) / df
                 end = tr.stats.starttime + float(j + samp - olap) / df
                 f.write("%s,%s,%s\n" % (str(begin), str(end), tr.stats.station))
            olap = overlap # only needed for first time in loop
            samp += nfft - overlap
            data_buf = data[-overlap:]
            print '.', # Progress Bar
        last_endtime, last_id = trId(tr.stats)
    f.close()
