#!/usr/bin/env python

from obspy.core import Trace, Stream, UTCDateTime
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import sonic, array_transff_freqslowness, \
  array_processing
from obspy.signal.array_analysis import array_transff_wavenumber
from obspy.signal.util import utlLonLat
import numpy as np
import unittest
from cStringIO import StringIO


class SonicTestCase(unittest.TestCase):
    """
    Test fk analysis, main function is sonic() in array_analysis.py
    """

    def sonicArgs(self):
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
        dt = dt.astype('int32')
        max_dt = np.max(dt) + 1
        min_dt = np.min(dt) - 1
        trl = list()
        for i in xrange(len(geometry)):
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

        semb_thres = -1e99
        vel_thres = -1e99

        args = (st, win_len, step_frac, sll_x, slm_x, sll_y, slm_y, sl_s,
                semb_thres, vel_thres, frqlow, frqhigh, stime, etime)
        return args


    def test_sonicBbfk(self):
        args = self.sonicArgs()
        kwargs = dict(prewhiten=0, coordsys='xy', verbose=False)
        out = sonic(*args, **kwargs)
        raw = """
7.19162000e+05 9.65490401e-01 8.70780661e-13 1.84349488e+01 1.26491106e+00
7.19162000e+05 9.59400535e-01 7.54453737e-13 1.84349488e+01 1.26491106e+00
7.19162000e+05 9.59788144e-01 6.01638964e-13 1.84349488e+01 1.26491106e+00
7.19162000e+05 9.64473665e-01 6.02637731e-13 1.84349488e+01 1.26491106e+00
7.19162000e+05 9.54958439e-01 5.13858053e-13 1.84349488e+01 1.26491106e+00
7.19162000e+05 9.53330994e-01 4.29547646e-13 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(StringIO(raw), dtype='f4')
        np.testing.assert_allclose(ref, out)


    def test_sonicBbfkPrew(self):
        args = self.sonicArgs()
        kwargs = dict(prewhiten=1, coordsys='xy', verbose=False)
        out = sonic(*args, **kwargs)
        raw = """
7.19162000e+05 3.85422463e-05 0.00000000e+00 1.84349488e+01 1.26491106e+00
7.19162000e+05 3.51870804e-05 0.00000000e+00 1.84349488e+01 1.26491106e+00
7.19162000e+05 3.55861157e-05 0.00000000e+00 1.84349488e+01 1.26491106e+00
7.19162000e+05 3.67066314e-05 0.00000000e+00 1.84349488e+01 1.26491106e+00
7.19162000e+05 3.64870211e-05 0.00000000e+00 1.84349488e+01 1.26491106e+00
7.19162000e+05 3.60413214e-05 0.00000000e+00 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(StringIO(raw), dtype='f4')
        np.testing.assert_allclose(ref, out)


    def test_sonicBf(self):
        args = self.sonicArgs()
        kwargs = dict(prewhiten=0, coordsys='xy', verbose=False,
                      method='bf')
        out = array_processing(*args, **kwargs)
        raw = """
7.19162000e+05 9.67606719e-01 2.28960532e-05 1.84349488e+01 1.26491106e+00
7.19162000e+05 9.69335057e-01 1.97564749e-05 1.84349488e+01 1.26491106e+00
7.19162000e+05 9.46067852e-01 1.54123841e-05 1.84349488e+01 1.26491106e+00
7.19162000e+05 9.41080449e-01 1.43546068e-05 1.84349488e+01 1.26491106e+00
7.19162000e+05 9.67305409e-01 1.35210630e-05 1.84349488e+01 1.26491106e+00
7.19162000e+05 9.39513631e-01 1.13792741e-05 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(StringIO(raw), dtype='f4')
        np.testing.assert_allclose(ref, out)


    def test_sonicBfPrew(self):
        args = self.sonicArgs()
        kwargs = dict(prewhiten=1, coordsys='xy', verbose=False,
                      method='bf')
        out = array_processing(*args, **kwargs)
        raw = """
7.19162000e+05 1.40348207e-01 2.28960532e-05 1.84349488e+01 1.26491106e+00
7.19162000e+05 1.30415347e-01 1.97564749e-05 1.84349488e+01 1.26491106e+00
7.19162000e+05 1.28008783e-01 1.54123841e-05 1.84349488e+01 1.26491106e+00
7.19162000e+05 1.25878335e-01 1.43546068e-05 1.84349488e+01 1.26491106e+00
7.19162000e+05 1.39312546e-01 1.35210630e-05 1.84349488e+01 1.26491106e+00
7.19162000e+05 1.32947973e-01 1.13792741e-05 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(StringIO(raw), dtype='f4')
        np.testing.assert_allclose(ref, out)


    def test_sonicCapon(self):
        args = self.sonicArgs()
        kwargs = dict(prewhiten=0, coordsys='xy', verbose=False,
                      method='capon')
        out = array_processing(*args, **kwargs)
        raw = """
7.19162000e+05 1.17234469e+01 1.17234469e+01 1.35000000e+02 8.48528137e-01
7.19162000e+05 2.40054037e+03 2.40054037e+03 -1.69380345e+02 1.62788206e+00
7.19162000e+05 4.49490862e+00 4.49490862e+00 -1.53434949e+02 4.47213595e-01
7.19162000e+05 4.10257370e+03 4.10257370e+03 0.00000000e+00 1.00000000e-01
7.19162000e+05 7.03232341e+02 7.03232341e+02 4.19872125e+01 2.69072481e+00
7.19162000e+05 9.76931058e+01 9.76931058e+01 -3.86598083e+01 2.56124969e+00
        """
        ref = np.loadtxt(StringIO(raw), dtype='f4')
        np.testing.assert_allclose(ref, out, rtol=1e-6)


    def test_sonicCaponPrew(self):
        args = self.sonicArgs()
        kwargs = dict(prewhiten=1, coordsys='xy', verbose=False,
                      method='capon')
        out = array_processing(*args, **kwargs)
        raw = """
7.19162000e+05 1.12733499e-01 0.00000000e+00 3.36900675e+01 7.21110255e-01
7.19162000e+05 1.77112262e-02 0.00000000e+00 2.65650512e+01 2.23606798e-01
7.19162000e+05 8.14214170e-02 0.00000000e+00 1.66992442e+01 1.04403065e+00
7.19162000e+05 1.27112655e-02 0.00000000e+00 1.84349488e+01 1.58113883e+00
7.19162000e+05 1.74286539e-02 0.00000000e+00 1.40362435e+01 8.24621125e-01
7.19162000e+05 4.49519407e-02 0.00000000e+00 1.96538241e+01 1.48660687e+00
        """
        ref = np.loadtxt(StringIO(raw), dtype='f4')
        np.testing.assert_allclose(ref, out, rtol=1e-6)


    def test_array_transff_freqslowness(self):

        coords = np.array([[10., 60., 0.],
                           [200., 50., 0.],
                           [-120., 170., 0.],
                           [-100., -150., 0.],
                           [30., -220., 0.]])

        coords /= 1000.

        coordsll = np.zeros(coords.shape)
        for i in np.arange(len(coords)):
            coordsll[i, 0], coordsll[i, 1] = utlLonLat(0., 0., coords[i, 0],
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
            coordsll[i, 0], coordsll[i, 1] = utlLonLat(0., 0., coords[i, 0],
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


def suite():
    return unittest.makeSuite(SonicTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
