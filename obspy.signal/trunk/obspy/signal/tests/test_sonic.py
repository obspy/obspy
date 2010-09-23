#!/usr/bin/env python

from obspy.core import Trace, Stream, UTCDateTime
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import sonic, array_transff_freqslowness
from obspy.signal.array_analysis import array_transff_wavenumber
from obspy.signal.util import utlLonLat
import numpy as np
import unittest


class SonicTestCase(unittest.TestCase):
    """
    Test fk analysis, main function is sonic() in array_analysis.py
    """

    def test_sonic(self):
#        for i in xrange(100):
        np.random.seed(2348)

        geometry = np.array([[   0.0, 0.0, 0.0],
                             [  -5.0, 7.0, 0.0],
                             [   5.0, 7.0, 0.0],
                             [  10.0, 0.0, 0.0],
                             [   5.0, -7.0, 0.0],
                             [  -5.0, -7.0, 0.0],
                             [-10.0, 0.0, 0.0]])

        geometry /= 100      # in km
        slowness = 1.3       # in s/km
        baz_degree = 20.0    # 0.0 > source in x direction
        baz = baz_degree * np.pi / 180.
        df = 100             # samplerate
        # SNR = 100.         # signal to noise ratio
        amp = .00001         # amplitude of coherent wave
        length = 500         # signal length in samples

        coherent_wave = amp * np.random.randn(length)

        # time offsets in samples
        dt = df * slowness * (np.cos(baz) * geometry[:, 1] + np.sin(baz) *
                              geometry[:, 0])
        dt = np.round(dt)
        dt = dt.astype('int32')
        # print dt
        max_dt = np.max(dt) + 1
        min_dt = np.min(dt) - 1
        trl = list()
        for i in xrange(len(geometry)):
            tr = Trace(coherent_wave[-min_dt + dt[i]:-max_dt + dt[i]].copy())
                #+ amp / SNR * np.random.randn(length - abs(min_dt) - abs(max_dt)))
            tr.stats.sampling_rate = df
            tr.stats.coordinates = AttribDict()
            tr.stats.coordinates.x = geometry[i, 0]
            tr.stats.coordinates.y = geometry[i, 1]
            tr.stats.coordinates.elevation = geometry[i, 2]
            # lowpass random signal to f_nyquist / 2
            tr.filter("lowpass", {'freq': df / 4.})
            trl.append(tr)

        st = Stream(trl)

        stime = UTCDateTime(1970, 1, 1, 0, 0)
        etime = UTCDateTime(1970, 1, 1, 0, 0) + \
                (length - abs(min_dt) - abs(max_dt)) / df

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

        # out returns: rel. power, abs. power, backazimuth, slowness
        out = sonic(st, win_len, step_frac, sll_x, slm_x, sll_y, slm_y, sl_s,
                    semb_thres, vel_thres, frqlow, frqhigh, stime, etime,
                    prewhiten, coordsys='xy', verbose=False)

        # returns baz
        np.testing.assert_almost_equal(out[:, 3].mean(), 18.434948822922024)
        # slowness ~= 1.3
        np.testing.assert_almost_equal(out[:, 4].mean(), 1.26491106407)

    
    def test_array_transff_freqslowness(self):
        
        coords = np.array([[  10.,   60., 0.],
                           [ 200.,   50., 0.],
                           [-120.,  170., 0.],
                           [-100., -150., 0.],
                           [  30., -220., 0.]])

        coords /= 1000.
       
        coordsll = np.zeros(coords.shape)
        for i in np.arange(len(coords)):
            coordsll[i,0], coordsll[i,1] = utlLonLat(0., 0., coords[i,0], coords[i,1])

        slim = 40.
        fmin = 1.
        fmax = 10.
        fstep = 1.

        sstep = slim/2.

        transff = array_transff_freqslowness(coords, slim, sstep, fmin, fmax,
            fstep, coordsys='xy')

        transffll = array_transff_freqslowness(coordsll, slim, sstep, fmin, fmax,
            fstep, coordsys='lonlat')

        transffth = np.array(
            [[ 0.17074014,  0.33333333,  0.11102387, 0.12237009,  0.3278369 ],
            [ 0.08828665,  0.21897583,  0.18096393, 0.34051651,  0.22606102], 
            [ 0.1118241 ,  0.08828665,  1.        , 0.08828665,  0.1118241 ], 
            [ 0.22606102,  0.34051651,  0.18096393, 0.21897583,  0.08828665], 
            [ 0.3278369 ,  0.12237009,  0.11102387, 0.33333333,  0.17074014]])

        np.testing.assert_array_almost_equal(transff, transffth, decimal=6)
        np.testing.assert_array_almost_equal(transffll, transffth, decimal=6)
    
    
    def test_array_transff_wavenumber(self):
        
        coords = np.array([[  10.,   60., 0.],
                           [ 200.,   50., 0.],
                           [-120.,  170., 0.],
                           [-100., -150., 0.],
                           [  30., -220., 0.]])

        coords /= 1000.

        coordsll = np.zeros(coords.shape)
        for i in np.arange(len(coords)):
            coordsll[i,0], coordsll[i,1] = utlLonLat(0., 0., coords[i,0], coords[i,1])
        
        klim = 40.
        kstep = klim/2.

        transff = array_transff_wavenumber(coords, klim, kstep, coordsys='xy')
        transffll = array_transff_wavenumber(coordsll, klim, kstep, coordsys='lonlat')

        transffth = np.array(
            [[  2.51325969e-01,   3.68857351e-02,   5.87486028e-02,
                3.07845295e-02,   3.35042906e-04],
            [  4.22331189e-02,   1.70231459e-03,   6.54354311e-04,
                3.91531577e-02,   3.95319293e-01],
            [  6.46052074e-02,   1.75467603e-01,   1.00000000e+00,
                1.75467603e-01,   6.46052074e-02],
            [  3.95319293e-01,   3.91531577e-02,   6.54354311e-04,
                1.70231459e-03,   4.22331189e-02],
            [  3.35042906e-04,   3.07845295e-02,   5.87486028e-02,
                3.68857351e-02,   2.51325969e-01]])

        np.testing.assert_array_almost_equal(transff, transffth, decimal=6)
        np.testing.assert_array_almost_equal(transffll, transffth, decimal=6)

def suite():
    return unittest.makeSuite(SonicTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
