#!/usr/bin/env python

from obspy.core import Trace, Stream, UTCDateTime
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import array_transff_freqslowness, \
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

    def arrayProcessing(self, prewhiten, method):
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
        kwargs = dict(prewhiten=prewhiten, coordsys='xy', verbose=False,
                      method=method)
        return array_processing(*args, **kwargs)

    def test_sonicBbfk(self):
        out = self.arrayProcessing(prewhiten=0, method='bbfk')
        raw = """
9.69512105e-01 8.64673947e-13 1.84349488e+01 1.26491106e+00
9.59917307e-01 7.46868442e-13 1.84349488e+01 1.26491106e+00
9.63363886e-01 6.03434783e-13 1.84349488e+01 1.26491106e+00
9.65261817e-01 5.98235383e-13 1.84349488e+01 1.26491106e+00
9.56272662e-01 5.09821189e-13 1.84349488e+01 1.26491106e+00
9.49727714e-01 4.28060799e-13 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(StringIO(raw), dtype='f4')
        np.testing.assert_allclose(ref, out[:, 1:])

    def test_sonicBbfkPrew(self):
        out = self.arrayProcessing(prewhiten=1, method='bbfk')
        raw = """
3.85865678e-05 0.00000000e+00 1.84349488e+01 1.26491106e+00
3.51731906e-05 0.00000000e+00 1.84349488e+01 1.26491106e+00
3.56623459e-05 0.00000000e+00 1.84349488e+01 1.26491106e+00
3.64871194e-05 0.00000000e+00 1.84349488e+01 1.26491106e+00
3.62874234e-05 0.00000000e+00 1.84349488e+01 1.26491106e+00
3.60890990e-05 0.00000000e+00 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(StringIO(raw), dtype='f4')
        np.testing.assert_allclose(ref, out[:, 1:])

    def test_sonicBf(self):
        out = self.arrayProcessing(prewhiten=0, method='bf')
        raw = """
9.71158129e-01 2.12938657e-05 1.84349488e+01 1.26491106e+00
9.63377268e-01 1.82054523e-05 1.84349488e+01 1.26491106e+00
9.64157527e-01 1.43237060e-05 1.84349488e+01 1.26491106e+00
9.64274905e-01 1.37873679e-05 1.84349488e+01 1.26491106e+00
9.57727890e-01 1.19452222e-05 1.84349488e+01 1.26491106e+00
9.55351246e-01 1.03071695e-05 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(StringIO(raw), dtype='f4')
        np.testing.assert_allclose(ref, out[:, 1:])

    def test_sonicBfPrew(self):
        out = self.arrayProcessing(prewhiten=1, method='bf')
        raw = """
1.41150417e-01 2.12938657e-05 1.84349488e+01 1.26491106e+00
1.28985966e-01 1.82054523e-05 1.84349488e+01 1.26491106e+00
1.30376049e-01 1.43237060e-05 1.84349488e+01 1.26491106e+00
1.33274120e-01 1.37873679e-05 1.84349488e+01 1.26491106e+00
1.32684786e-01 1.19452222e-05 1.84349488e+01 1.26491106e+00
1.32034559e-01 1.03071695e-05 1.84349488e+01 1.26491106e+00
        """
        ref = np.loadtxt(StringIO(raw), dtype='f4')
        np.testing.assert_allclose(ref, out[:, 1:])

    def test_sonicCapon(self):
        out = self.arrayProcessing(prewhiten=0, method='capon')
        raw = """
3.38665915e+00 3.38665915e+00   1.40362435e+01 2.06155281e+00
9.23938685e+02 9.23938685e+02  -1.38012788e+02 2.69072481e+00
2.19122428e+03 2.19122428e+03   3.02564372e+01 1.38924440e+00
2.85527020e+03 2.85527020e+03   6.44400348e+01 2.54950976e+00
6.39601952e+02 6.39601952e+02  -1.84349488e+01 2.52982213e+00
8.24093589e+01 8.24093589e+01   1.64054604e+02 7.28010989e-01
        """
        ref = np.loadtxt(StringIO(raw), dtype='f4')
        np.testing.assert_allclose(ref, out[:, 1:], rtol=1e-6)

    def test_sonicCaponPrew(self):
        out = self.arrayProcessing(prewhiten=1, method='capon')
        raw = """
5.60085245e-02 0.00000000e+00   1.55241110e+01 1.86815417e+00
8.45669383e-03 0.00000000e+00   2.65650512e+01 1.34164079e+00
9.76986611e-03 0.00000000e+00   6.49831065e+01 3.31058907e+00
8.22576235e-03 0.00000000e+00  -4.50000000e+01 3.95979797e+00
1.55732220e-02 0.00000000e+00   2.23801351e+01 1.83847763e+00
1.28357278e-02 0.00000000e+00   9.78240703e+00 2.94278779e+00
        """
        ref = np.loadtxt(StringIO(raw), dtype='f4')
        np.testing.assert_allclose(ref, out[:, 1:], rtol=1e-6)

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
