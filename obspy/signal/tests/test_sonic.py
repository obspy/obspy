#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import unittest

import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import (array_processing,
                                         array_transff_freqslowness,
                                         array_transff_wavenumber, get_spoint)
from obspy.signal.util import util_lon_lat


class SonicTestCase(unittest.TestCase):
    """
    Test fk analysis, main function is sonic() in array_analysis.py
    """

    def array_processing(self, prewhiten, method):
        np.random.seed(2348)

        geometry = np.array([[0.0, 0.0, 0.0],
                             [-5.0, 7.0, 0.0],
                             [5.0, 7.0, 0.0],
                             [10.0, 0.0, 0.0],
                             [5.0, -7.0, 0.0],
                             [-5.0, -7.0, 0.0],
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
        dt = dt.astype(np.int32)
        max_dt = np.max(dt) + 1
        min_dt = np.min(dt) - 1
        trl = list()
        for i in range(len(geometry)):
            tr = Trace(coherent_wave[-min_dt + dt[i]:-max_dt + dt[i]].copy())
            # + amp / SNR * \
            # np.random.randn(length - abs(min_dt) - abs(max_dt)))
            tr.stats.sampling_rate = df
            tr.stats.coordinates = AttribDict()
            tr.stats.coordinates.x = geometry[i, 0]
            tr.stats.coordinates.y = geometry[i, 1]
            tr.stats.coordinates.elevation = geometry[i, 2]
            # lowpass random signal to f_nyquist / 2
            tr.filter("lowpass", freq=df / 4.)
            trl.append(tr)

        st = Stream(trl)

        stime = UTCDateTime(1970, 1, 1, 0, 0)
        etime = UTCDateTime(1970, 1, 1, 0, 0) + 4.0
        # TODO: check why this does not work any more
        #    (length - abs(min_dt) - abs(max_dt)) / df

        win_len = 2.
        step_frac = 0.2
        sll_x = -3.0
        slm_x = 3.0
        sll_y = -3.0
        slm_y = 3.0
        sl_s = 0.1

        frqlow = 1.0
        frqhigh = 8.0

        semb_thres = -1e99
        vel_thres = -1e99

        args = (st, win_len, step_frac, sll_x, slm_x, sll_y, slm_y, sl_s,
                semb_thres, vel_thres, frqlow, frqhigh, stime, etime)
        kwargs = dict(prewhiten=prewhiten, coordsys='xy', verbose=False,
                      method=method)
        out = array_processing(*args, **kwargs)
        if False:  # 1 for debugging
            print('\n', out[:, 1:])
        return out

    def test_sonic_bf(self):
        out = self.array_processing(prewhiten=0, method=0)
        raw = """
9.68742255e-01 1.95739086e-05 1.84349488e+01 1.26491106e+00
9.60822403e-01 1.70468277e-05 1.84349488e+01 1.26491106e+00
9.61689241e-01 1.35971034e-05 1.84349488e+01 1.26491106e+00
9.64670470e-01 1.35565806e-05 1.84349488e+01 1.26491106e+00
9.56880885e-01 1.16028992e-05 1.84349488e+01 1.26491106e+00
9.49584782e-01 9.67131311e-06 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(io.StringIO(raw), dtype=np.float32)
        self.assertTrue(np.allclose(ref, out[:, 1:], rtol=1e-6))

    def test_sonic_bf_prew(self):
        out = self.array_processing(prewhiten=1, method=0)
        raw = """
1.40997967e-01 1.95739086e-05 1.84349488e+01 1.26491106e+00
1.28566503e-01 1.70468277e-05 1.84349488e+01 1.26491106e+00
1.30517975e-01 1.35971034e-05 1.84349488e+01 1.26491106e+00
1.34614854e-01 1.35565806e-05 1.84349488e+01 1.26491106e+00
1.33609938e-01 1.16028992e-05 1.84349488e+01 1.26491106e+00
1.32638966e-01 9.67131311e-06 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(io.StringIO(raw), dtype=np.float32)
        self.assertTrue(np.allclose(ref, out[:, 1:]))

    def test_sonic_capon(self):
        out = self.array_processing(prewhiten=0, method=1)
        raw = """
9.06938200e-01 9.06938200e-01  1.49314172e+01  1.55241747e+00
8.90494375e+02 8.90494375e+02 -9.46232221e+00  1.21655251e+00
3.07129784e+03 3.07129784e+03 -4.95739213e+01  3.54682957e+00
5.00019137e+03 5.00019137e+03 -1.35000000e+02  1.41421356e-01
7.94530414e+02 7.94530414e+02 -1.65963757e+02  2.06155281e+00
6.08349575e+03 6.08349575e+03  1.77709390e+02  2.50199920e+00
        """
        ref = np.loadtxt(io.StringIO(raw), dtype=np.float32)
        # XXX relative tolerance should be lower!
        self.assertTrue(np.allclose(ref, out[:, 1:], rtol=5e-3))

    def test_sonic_capon_prew(self):
        out = self.array_processing(prewhiten=1, method=1)
        raw = """
1.30482688e-01 9.06938200e-01  1.49314172e+01  1.55241747e+00
8.93029978e-03 8.90494375e+02 -9.46232221e+00  1.21655251e+00
9.55393634e-03 1.50655072e+01  1.42594643e+02  2.14009346e+00
8.85762420e-03 7.27883670e+01  1.84349488e+01  1.26491106e+00
1.51510617e-02 6.54541771e-01  6.81985905e+01  2.15406592e+00
3.10761699e-02 7.38667657e+00  1.13099325e+01  1.52970585e+00
        """
        ref = np.loadtxt(io.StringIO(raw), dtype=np.float32)
        # XXX relative tolerance should be lower!
        self.assertTrue(np.allclose(ref, out[:, 1:], rtol=4e-5))

    def test_get_spoint(self):
        stime = UTCDateTime(1970, 1, 1, 0, 0)
        etime = UTCDateTime(1970, 1, 1, 0, 0) + 10
        data = np.empty(20)
        # sampling rate defaults to 1 Hz
        st = Stream([
            Trace(data, {'starttime': stime - 1}),
            Trace(data, {'starttime': stime - 4}),
            Trace(data, {'starttime': stime - 2}),
        ])
        spoint, epoint = get_spoint(st, stime, etime)
        self.assertTrue(np.allclose([1, 4, 2], spoint))
        self.assertTrue(np.allclose([8, 5, 7], epoint))

    def test_array_transff_freqslowness(self):
        coords = np.array([[10., 60., 0.],
                           [200., 50., 0.],
                           [-120., 170., 0.],
                           [-100., -150., 0.],
                           [30., -220., 0.]])

        coords /= 1000.

        coordsll = np.zeros(coords.shape)
        for i in np.arange(len(coords)):
            coordsll[i, 0], coordsll[i, 1] = util_lon_lat(0., 0., coords[i, 0],
                                                          coords[i, 1])

        slim = 40.
        fmin = 1.
        fmax = 10.
        fstep = 1.

        sstep = slim / 2.

        transff = array_transff_freqslowness(coords, slim, sstep, fmin, fmax,
                                             fstep, coordsys='xy')

        transffll = array_transff_freqslowness(coordsll, slim, sstep, fmin,
                                               fmax, fstep, coordsys='lonlat')

        transffth = np.array(
            [[0.41915119, 0.33333333, 0.32339525, 0.24751548, 0.67660475],
             [0.25248452, 0.41418215, 0.34327141, 0.65672859, 0.33333333],
             [0.24751548, 0.25248452, 1.00000000, 0.25248452, 0.24751548],
             [0.33333333, 0.65672859, 0.34327141, 0.41418215, 0.25248452],
             [0.67660475, 0.24751548, 0.32339525, 0.33333333, 0.41915119]])

        np.testing.assert_array_almost_equal(transff, transffth, decimal=6)
        np.testing.assert_array_almost_equal(transffll, transffth, decimal=6)

    def test_array_transff_wavenumber(self):
        coords = np.array([[10., 60., 0.],
                           [200., 50., 0.],
                           [-120., 170., 0.],
                           [-100., -150., 0.],
                           [30., -220., 0.]])

        coords /= 1000.

        coordsll = np.zeros(coords.shape)
        for i in np.arange(len(coords)):
            coordsll[i, 0], coordsll[i, 1] = util_lon_lat(0., 0., coords[i, 0],
                                                          coords[i, 1])

        klim = 40.
        kstep = klim / 2.

        transff = array_transff_wavenumber(coords, klim, kstep, coordsys='xy')
        transffll = array_transff_wavenumber(coordsll, klim, kstep,
                                             coordsys='lonlat')

        transffth = np.array(
            [[3.13360360e-01, 4.23775796e-02, 6.73650243e-01,
              4.80470652e-01, 8.16891615e-04],
             [2.98941684e-01, 2.47377842e-01, 9.96352135e-02,
              6.84732871e-02, 5.57078203e-01],
             [1.26523678e-01, 2.91010683e-01, 1.00000000e+00,
              2.91010683e-01, 1.26523678e-01],
             [5.57078203e-01, 6.84732871e-02, 9.96352135e-02,
              2.47377842e-01, 2.98941684e-01],
             [8.16891615e-04, 4.80470652e-01, 6.73650243e-01,
              4.23775796e-02, 3.13360360e-01]])

        np.testing.assert_array_almost_equal(transff, transffth, decimal=6)
        np.testing.assert_array_almost_equal(transffll, transffth, decimal=6)
