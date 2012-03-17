# -*- coding: utf-8 -*-

from obspy.core import read
from obspy.realtime.signal import calculateMwpMag, tauc, scale, boxcar, \
    integrate, differentiate, mwpIntegral
import math
import numpy as np
import obspy.realtime.rttrace as rt
import os


SAC_DATA_FILE = os.path.join(os.path.dirname(rt.__file__),
                             os.path.join('tests', 'data'), 'II.TLY.BHZ.SAC')
GAIN = 1.610210e+09

PLOT_ORIGINAL_MERGED = False

# =================================
# select processing sequence to run
# =================================
#
if 0:
    process_list = ['scale']
    SCALE_FACTOR = 1000
#
elif 0:
    process_list = ['tauc']
    TAUC_WINDOW_WIDTH = 60
#
elif 1:
    process_list = ['boxcar']
    BOXCAR_WIDTH = 500
    PLOT_ORIGINAL_MERGED = True
#
elif 0:
    process_list = ['integrate']
#
elif 0:
    process_list = ['differentiate']
#
elif 0:
    process_list = ['integrate', 'mwpIntegral']
    MWP_MAX_TIME = 120
#
elif 0:
    process_list = ['np.abs']
    PLOT_ORIGINAL_MERGED = True
#
elif 0:
    process_list = ['np.square']


# display processing functions doc
#print rt.RtTrace.rtProcessFunctionsToString()


#st = read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')
print 'Reading: ', SAC_DATA_FILE
st = read(SAC_DATA_FILE)
orig_data = st[0].copy().data
st[0].write('trace_orig.sac', format='SAC')
# set needed values
ref_time_offest = st[0].stats['sac']['a']
print '   sac.a:' + str(ref_time_offest)
if math.fabs(ref_time_offest - -12345.0) < 0.001:
    print 'Error: sac.a value not set.'
epicentral_distance = st[0].stats['sac']['gcarc']
print '   sac.gcarc:' + str(epicentral_distance)
if math.fabs(epicentral_distance - -12345.0) < 0.001:
    print 'Error: sac.gcarc value not set.'

print 'Processing is ', process_list

# apply normal ObsPy processing to original trace
# Filtering the Stream object
st_filt = st[0].copy()
delta_time = 1.0 / st[0].stats.sampling_rate
for process in process_list:
    if process == 'scale':
        st_filt.data = scale(st_filt, factor=SCALE_FACTOR)
    elif process == 'tauc':
        st_filt.data = tauc(st_filt, width=TAUC_WINDOW_WIDTH)
    elif process == 'boxcar':
        st_filt.data = boxcar(st_filt, width=BOXCAR_WIDTH)
    elif process == 'integrate':
        st_filt.data = integrate(st_filt)
    elif process == 'differentiate':
        st_filt.data = differentiate(st_filt)
    elif process == 'mwpIntegral':
        st_filt.data = mwpIntegral(st_filt, mem_time=MWP_MAX_TIME,
             ref_time=(st_filt.stats.starttime + ref_time_offest),
             max_time=MWP_MAX_TIME, gain=GAIN)
    elif process == 'np.abs':
        st_filt.data = np.abs(st_filt.data)
    elif process == 'np.square':
        st_filt.data = np.square(st_filt.data)
    else:
        print 'Warning: process:', process, ': not supported by this function'
# save processed trace to disk
st_filt.write('trace.sac', format='SAC')


# create set of contiguous packet data in an array of Trace objects
total_length = st[0].stats.endtime - st[0].stats.starttime
num_pakets = 3
packet_length = total_length / num_pakets
tstart = st[0].stats.starttime
tend = tstart + packet_length
traces = []
for i in range(num_pakets):
    tr = st[0].copy()
    tr = tr.slice(tstart, tend)
    traces.append(tr)
    tstart = tend + 1.0 / st[0].stats.sampling_rate
    tend = tstart + packet_length


# assemble realtime trace
rt_trace = rt.RtTrace()
#rt_trace = rt.RtTrace(max_length=600)
#
for process in process_list:
    if process == 'scale':
        rt_trace.registerRtProcess('scale', factor=SCALE_FACTOR)
    elif process == 'tauc':
        rt_trace.registerRtProcess('tauc', width=TAUC_WINDOW_WIDTH)
    elif process == 'boxcar':
        rt_trace.registerRtProcess('boxcar', width=BOXCAR_WIDTH)
    elif process == 'integrate':
        rt_trace.registerRtProcess('int')
    elif process == 'differentiate':
        rt_trace.registerRtProcess('diff')
    elif process == 'mwpIntegral':
        rt_trace.registerRtProcess('mwpIntegral', mem_time=MWP_MAX_TIME,
             ref_time=(st[0].stats.starttime + ref_time_offest),
             max_time=MWP_MAX_TIME, gain=GAIN)
    elif process == 'np.abs':
        rt_trace.registerRtProcess('np.abs')
    elif process == 'np.square':
        rt_trace.registerRtProcess('np.square')
    else:
        rt_trace.registerRtProcess(process)


# append packet data to RtTrace
for i in range(num_pakets):
    print 'Appending packet: ', i
    appended_trace = rt_trace.append(traces[i])
    appended_trace.write('appended_trace%d.sac' % (i), format='SAC')
rt_trace.write('rt_trace.sac', format='SAC')


# post processing
if process_list[-1] == 'mwpIntegral':
    print 'Post-processing for ', process_list[-1].strip(), ':'
    peak = np.amax(np.abs(rt_trace.data))
    print '   mwpIntegral max = ', peak
    print '   epicentral_distance = ', epicentral_distance
    print '   mwp = ', calculateMwpMag(peak, epicentral_distance)
elif process_list[-1] == 'tauc':
    print 'Post-processing for ', process_list[-1].strip(), ':'
    peak = np.amax(np.abs(rt_trace.data))
    print '   tauc max = ', peak


# plot
if 1:
    plt_stream = read(SAC_DATA_FILE)
    if PLOT_ORIGINAL_MERGED:
        plt_stream = read(SAC_DATA_FILE)
        plt_stream += read('trace.sac')
    else:
        print 'This plot shows the original, unprocessed data trace.'
        plt_stream.plot(automerge=False, size=(800, 1000))
        plt_stream = read('trace.sac')
    plt_stream += read('rt_trace.sac')
    for i in range(num_pakets):
        plt_stream += read('appended_trace%d.sac' % (i))
    print 'This plot shows:'
    if PLOT_ORIGINAL_MERGED:
        print '   the original, unprocessed data trace'
    print '   the original data trace processed with', process_list
    print '   the realtime data trace composed of', num_pakets, \
        ' appended packets, processed with', process_list
    print '   the', num_pakets, 'processed packets individually'
    plt_stream.plot(automerge=False, size=(800, 1000), color='blue')
