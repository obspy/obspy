# -*- coding: utf-8 -*-

import numpy as np
from obspy.core import read
import obspy.realtime.rttrace as rt
from obspy.realtime.signal.util import calculateMwpMag
import os

# read data file

data_stream = read(os.path.join(os.path.dirname(rt.__file__),
                   os.path.join('tests', 'data'), 'II.TLY.BHZ.SAC'))
data_trace = data_stream[0]
ref_time_offest = data_trace.stats['sac']['a']
print 'ref_time_offest (sac.a):' + str(ref_time_offest)
epicentral_distance = data_trace.stats['sac']['gcarc']
print 'epicentral_distance (sac.gcarc):' + str(epicentral_distance)

# create set of contiguous packet data in an array of Trace objects

total_length = np.size(data_trace.data)
num_pakets = 3
packet_length = int(total_length / num_pakets)  # may give int truncate
delta_time = 1.0 / data_trace.stats.sampling_rate
tstart = data_trace.stats.starttime
tend = tstart + delta_time * packet_length
traces = []
for i in range(num_pakets):
    tr = data_trace.copy()
    tr = tr.slice(tstart, tend)
    traces.append(tr)
    tstart = tend + delta_time
    tend = tstart + delta_time * packet_length

# assemble realtime trace

rt_trace = rt.RtTrace()
rt_trace.registerRtProcess('integrate')
rt_trace.registerRtProcess('mwpIntegral', mem_time=240,
                           ref_time=(data_trace.stats.starttime + \
                           ref_time_offest),
                           max_time=120, gain=1.610210e+09)

# append packet data to RtTrace

for i in range(num_pakets):
    appended_trace = rt_trace.append(traces[i], gap_overlap_check=True)

# post processing to get Mwp

peak = np.amax(np.abs(rt_trace.data))
print 'mwpIntegral peak = ', peak
print 'epicentral_distance = ', epicentral_distance
mwp = calculateMwpMag(peak, epicentral_distance)
print 'Mwp = ', mwp
assert(int(mwp * 1000) == int(8.78902911791 * 1000))
