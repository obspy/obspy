#!/usr/bin/env python

import unittest
import numpy as np
from obspy.core import UTCDateTime
from obspy.signal.array_analysis import sonic
from obspy.core import Trace, Stream
from obspy.core.util import AttribDict

class SonicTestCase(unittest.TestCase):
    """
    Test fk analysis, main function is sonic() in array_analysis.py
    """

    def test_sonic(self):
#        for i in xrange(100):
        np.random.seed(2348)    
        
        geometry = np.array([[   0.0,  0.0, 0.0],
                             [  -5.0,  7.0, 0.0],
                             [   5.0,  7.0, 0.0],
                             [  10.0,  0.0, 0.0],
                             [   5.0, -7.0, 0.0],
                             [  -5.0, -7.0, 0.0],
                             [ -10.0,  0.0, 0.0]])
        
        geometry /= 100      # in km
        slowness = 1.3       # in s/km
        baz_degree = 20.0           # 0.0 > source in x direction
        baz = baz_degree * np.pi / 180.
        df = 100             # samplerate
        # SNR = 100.         # signal to noise ratio
        amp = .00001         # amplitude of coherent wave
        length = 500         # signal length in samples

        coherent_wave = amp * np.random.randn(length)
        
        # time offsets in samples
        dt = df * slowness * (np.cos(baz) * geometry[:,1] + np.sin(baz) *
                              geometry[:,0])
        dt = np.round(dt)
        dt = dt.astype('int32')
        # print dt
        max_dt = np.max(dt) + 1
        min_dt = np.min(dt) - 1
        trl = list()
        for i in xrange(len(geometry)):
            tr = Trace(coherent_wave[-min_dt + dt[i]: -max_dt + dt[i]].copy())
                #+ amp / SNR * np.random.randn(length - abs(min_dt) - abs(max_dt)))
            tr.stats.sampling_rate = df 
            tr.stats.coordinates = AttribDict()
            tr.stats.coordinates.x = geometry[i,0]
            tr.stats.coordinates.y = geometry[i,1]
            tr.stats.coordinates.elevation = geometry[i,2]
            # lowpass random signal to f_nyquist / 2
            tr.filter("lowpass", {'freq': df/4.})
            trl.append(tr)

        st = Stream(trl)

        stime = UTCDateTime(1970, 1, 1, 0, 0) 
        etime = UTCDateTime(1970, 1, 1, 0, 0) + \
                (length -  abs(min_dt) - abs(max_dt)) / df

        win_len = 2.
        step_frac = 0.2
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
        
        # out returns: rel. power, abs. power, azimuth, slowness
        out = sonic(st, win_len, step_frac, sll_x, slm_x, sll_y, slm_y, sl_s,
                    semb_thres, vel_thres, frqlow, frqhigh, stime, etime,
                    prewhiten, coordsys='xy', verbose=False)
        
        # azimuth: baz_degree - 180 ~= -160
        np.testing.assert_almost_equal(out[:,3].mean(), -161.565051177)
        # slowness ~= 1.3
        np.testing.assert_almost_equal(out[:,4].mean(), 1.26491106407)


def suite():
    return unittest.makeSuite(SonicTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
