#!/usr/bin/env python

import unittest

class SonicTestCase(unittest.TestCase):
    def test_empty(self):
        pass

def suite():
    return unittest.makeSuite(SonicTestCase, 'test')

if __name__ == '__main__':
    from obspy.core import read, UTCDateTime
    from obspy.signal.array_analysis import sonic

    path = "/scratch/AGFA/orig_spreng"
    #path = "muh"
    # BW01
    st = read("%s/20080217_110000_Z.A060" % path)
    st[-1].stats.lon = 11.582967
    st[-1].stats.lat = 48.108589
    st[-1].stats.elev = 0.450
    # BW02
    st += read("%s/20080217_110000_1.bw02" % path)
    st[-1].stats.lon = 48.108192
    st[-1].stats.lat = 11.583120
    st[-1].stats.elev = 0.450
    # BW03
    st += read("%s/20080217_110000_1.bw03" % path)
    st[-1].stats.lon = 11.583414
    st[-1].stats.lat = 48.108692
    st[-1].stats.elev = 0.450
    # BW07
    st += read("%s/20080217_110000_Z.BW07" % path)
    st[-1].stats.lon = 11.583049
    st[-1].stats.lat = 48.108456
    st[-1].stats.elev = 0.450
    # BW08
    st += read("%s/20080217_110000_Z.A0D0" % path)
    st[-1].stats.lon = 11.583157
    st[-1].stats.lat = 48.108730
    st[-1].stats.elev = 0.450

    # we do this instead of instrument simulation, take flat part of
    # response spectrum
    st.filter('bandpass', {'freqmin': 1.0, 'freqmax': 40.0, 'corners': 1, 'zerophase': False})

    stime = UTCDateTime("20080217110520")
    etime = UTCDateTime("20080217110540")

    win_len = 1.0
    step_frac = 0.05
    # X min
    sll_x = -3.0
    # X max
    slm_x = 3.0
    # Y min
    sll_y = -3.0
    # Y max
    slm_y = 3.0
    # Slow Step
    sl_s = 0.03

    frqlow = 1.0
    frqhigh = 8.0
    prewhiten = 0

    semb_thres = -1e99
    vel_thres = -1e99

    st.trim(stime, etime)

    print "executing sonic"
    out = sonic(st, win_len, step_frac, sll_x, slm_x, sll_y, slm_y, sl_s,
                semb_thres, vel_thres, frqlow, frqhigh, stime, etime,
                prewhiten)
