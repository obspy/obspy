#!/usr/bin/env python

import unittest

class SonicTestCase(unittest.TestCase):
    def test_empty(self):
        pass

def suite():
    return unittest.makeSuite(SonicTestCase, 'test')

if __name__ == '__main__':
    import numpy as np
    from obspy.core import UTCDateTime
    from obspy.signal.array_analysis import sonic
    from obspy.core import Trace, Stream
        
    geometry = np.array([[   0.0,  0.0, 0.0],
                         [  -5.0,  7.0, 0.0],
                         [   5.0,  7.0, 0.0],
                         [  10.0,  0.0, 0.0],
                         [   5.0, -7.0, 0.0],
                         [  -5.0, -7.0, 0.0],
                         [ -10.0,  0.0, 0.0]])
    
    geometry /= 100 # in km
    slowness = 1.3 # in s/km
    baz = 90.0 # 0.0 > source in x direction
    baz *= 2. * np.pi / 360.
    df = 100 # samplerate
    SNR = 10. # Sound to noise ratio
    amp = .00001 # amplitude of coherent wave
    length = 1000 # signal length in samples

#     coherent_wave = amp * np.exp(-1 * np.square(np.linspace(-2, 2, length)))\
#         * np.sin(np.linspace(-100 * np.pi, 100 * np.pi, length)) 
    # maybe better not use a sine signal because of periodicity??
    # lets try a random signal, its lowpassed in the next step
    coherent_wave = amp * np.random.randn(length)
    
    # time offsets in samples
    dt = -1 * df * slowness * (np.cos(baz) * geometry[:,0] + np.sin(baz) * geometry[:,1])
    dt = np.int32(dt)
    # print dt
    max_dt = np.max(dt) + 1
    min_dt = np.min(dt) - 1
    trl = list()
    for i in xrange(len(geometry)):
        # print i
        # print dt[i]
        # print len(coherent_wave[-min_dt + dt[i]: -max_dt + dt[i]])
        # print (length - abs(min_dt) - abs(max_dt))
        tr = Trace(coherent_wave[-min_dt + dt[i]: -max_dt + dt[i]] \
            + amp / SNR * np.random.randn(length - abs(min_dt) - abs(max_dt)))
        tr.stats.sampling_rate = df
        tr.filter("lowpass", {'freq': df/4.})
        trl.append(tr)

    st = Stream(trl)

    stime = UTCDateTime(1970, 1, 1, 0, 0) + 1
    etime = UTCDateTime(1970, 1, 1, 0, 0) + (length -  abs(min_dt) - abs(max_dt))/df -1

    win_len = 1.0
    step_frac = 0.05
    sll_x = -3.0
    slm_x = 3.0
    sll_y = -3.0
    slm_y = 3.0
    sl_s = 0.1

    frqlow = 1.0
    frqhigh = 8.0
    prewhiten = 0

    semb_thres = -1e99
    vel_thres = -1e99

    print "executing sonic"
    out = sonic(st, win_len, step_frac, sll_x, slm_x, sll_y, slm_y, sl_s,
                semb_thres, vel_thres, frqlow, frqhigh, stime, etime,
                prewhiten, geometry=geometry)

#     from obspy.core import read, UTCDateTime
# 
#     path = "/scratch/AGFA/orig_spreng"
#     #path = "muh"
#     # BW01
#     st = read("%s/20080217_110000_Z.A060" % path)
#     st[-1].stats.lon = 11.582967
#     st[-1].stats.lat = 48.108589
#     st[-1].stats.elev = 0.450
#     # BW02
#     st += read("%s/20080217_110000_1.bw02" % path)
#     st[-1].stats.lon = 48.108192
#     st[-1].stats.lat = 11.583120
#     st[-1].stats.elev = 0.450
#     # BW03
#     st += read("%s/20080217_110000_1.bw03" % path)
#     st[-1].stats.lon = 11.583414
#     st[-1].stats.lat = 48.108692
#     st[-1].stats.elev = 0.450
#     # BW07
#     st += read("%s/20080217_110000_Z.BW07" % path)
#     st[-1].stats.lon = 11.583049
#     st[-1].stats.lat = 48.108456
#     st[-1].stats.elev = 0.450
#     # BW08
#     st += read("%s/20080217_110000_Z.A0D0" % path)
#     st[-1].stats.lon = 11.583157
#     st[-1].stats.lat = 48.108730
#     st[-1].stats.elev = 0.450
# 
#     # we do this instead of instrument simulation, take flat part of
#     # response spectrum
#     st.filter('bandpass', {'freqmin': 1.0, 'freqmax': 5.0, 'corners': 1, 'zerophase': False})
# 
#     stime = UTCDateTime("20080217110520")
#     etime = UTCDateTime("20080217110540")
# 
#     win_len = 1.0
#     step_frac = 0.05
#     # X min
#     sll_x = -3.0
#     # X max
#     slm_x = 3.0
#     # Y min
#     sll_y = -3.0
#     # Y max
#     slm_y = 3.0
#     # Slow Step
#     sl_s = 0.03
# 
#     frqlow = 1.0
#     frqhigh = 8.0
#     prewhiten = 0
# 
#     semb_thres = -1e99
#     vel_thres = -1e99
# 
#     st.trim(stime, etime)
# 
#     print "executing sonic"
#     out = sonic(st, win_len, step_frac, sll_x, slm_x, sll_y, slm_y, sl_s,
#                 semb_thres, vel_thres, frqlow, frqhigh, stime, etime,
#                 prewhiten)
