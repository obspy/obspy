# -*- coding: utf-8 -*-
"""
The cross correlation test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import ctypes as C
import numpy as np
import os
import unittest

from obspy import UTCDateTime, read
from obspy.core.util.libnames import _load_cdll
from obspy.core.util.testing import ImageComparison
from obspy.signal.cross_correlation import (xcorr, xcorr_pick_correction,
                                            xcorr_3c, xcorr_max,
                                            _xcorr_padzeros, _xcorr_slice)


class CrossCorrelationTestCase(unittest.TestCase):

    """
    Cross corrrelation test case
    """

    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        self.path_images = os.path.join(os.path.dirname(__file__), 'images')
        self.a = np.sin(np.linspace(0, 10, 101))
        self.b = 5 * np.roll(self.a, 5)
        self.c = 5 * np.roll(self.a[:81], 5)

    def test_xcorr(self):
        """
        """
        # example 1 - all samples are equal
        np.random.seed(815)  # make test reproducible
        tr1 = np.random.randn(10000).astype(np.float32)
        tr2 = tr1.copy()
        shift, corr, _ = xcorr(tr1, tr2, 100, full_xcorr=True)
        self.assertEqual(shift, 0)
        self.assertAlmostEqual(corr, 1, 2)
        # example 2 - all samples are different
        tr1 = np.ones(10000, dtype=np.float32)
        tr2 = np.zeros(10000, dtype=np.float32)
        shift, corr, _ = xcorr(tr1, tr2, 100, full_xcorr=True)
        # comment next line, because shift is not unique
        # self.assertEqual(shift, 0)
        self.assertAlmostEqual(corr, 0, 2)
        # example 3 - shift of 10 samples
        tr1 = np.random.randn(10000).astype(np.float32)
        tr2 = np.concatenate((np.zeros(10), tr1[0:-10]))
        shift, corr, _ = xcorr(tr1, tr2, 100, full_xcorr=True)
        self.assertEqual(shift, -10)
        self.assertAlmostEqual(corr, 1, 2)
        shift, corr, _ = xcorr(tr2, tr1, 100, full_xcorr=True)
        self.assertEqual(shift, 10)
        self.assertAlmostEqual(corr, 1, 2)
        # example 4 - shift of 10 samples + small sine disturbance
        tr1 = (np.random.randn(10000) * 100).astype(np.float32)
        var = np.sin(np.arange(10000, dtype=np.float32) * 0.1)
        tr2 = np.concatenate((np.zeros(10), tr1[0:-10])) * 0.9
        tr2 += var
        shift, corr, _ = xcorr(tr1, tr2, 100, full_xcorr=True)
        self.assertEqual(shift, -10)
        self.assertAlmostEqual(corr, 1, 2)
        shift, corr, _ = xcorr(tr2, tr1, 100, full_xcorr=True)
        self.assertEqual(shift, 10)
        self.assertAlmostEqual(corr, 1, 2)

    def test_srl_xcorr(self):
        """
        Tests if example in ObsPy paper submitted to the Electronic
        Seismologist section of SRL is still working. The test shouldn't be
        changed because the reference gets wrong.
        """
        np.random.seed(815)
        data1 = np.random.randn(1000).astype(np.float32)
        data2 = data1.copy()

        window_len = 100
        corp = np.empty(2 * window_len + 1, dtype=np.float64)

        lib = _load_cdll("signal")
        #
        shift = C.c_int()
        coe_p = C.c_double()
        res = lib.X_corr(data1.ctypes.data_as(C.c_void_p),
                         data2.ctypes.data_as(C.c_void_p),
                         corp.ctypes.data_as(C.c_void_p),
                         window_len, len(data1), len(data2),
                         C.byref(shift), C.byref(coe_p))

        self.assertEqual(0, res)
        self.assertAlmostEqual(0.0, shift.value)
        self.assertAlmostEqual(1.0, coe_p.value)

    def test_xcorr_depreciation_full_xcorr(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shift, corr = xcorr(self.a, self.b, 15)
            self.assertEqual(len(w), 1)
            self.assertIn('full_xcorr', str(w[-1].message))

    def test_xcorr_vs_old_implementation(self):
        """
        Test against output of xcorr from ObsPy<1.1
        """
        # Results of xcorr(self.a, self.b, 15, full_xcorr=True)
        # for ObsPy==1.0.2:
        # -5, 0.9651607597888241
        x = [0.53555336, 0.60748967, 0.67493495, 0.73707491, 0.79313226,
             0.84237607, 0.88413089, 0.91778536, 0.94280034, 0.95871645,
             0.96516076, 0.96363672, 0.95043933, 0.92590109, 0.89047807,
             0.84474328, 0.78377236, 0.71629895, 0.64316805, 0.56526677,
             0.48351386, 0.39884904, 0.31222231, 0.22458339, 0.13687123,
             0.05000401, -0.03513057, -0.11768441, -0.19685756, -0.27190599,
             -0.34214866]
        shift, corr, corr_fun = xcorr(self.a, self.b, 15, full_xcorr=True)
        np.testing.assert_allclose(corr_fun, x)
        self.assertAlmostEqual(corr, 0.96516076)
        self.assertEqual(shift, -5)

    def test_xcorr_different_length_of_signals(self):
        # Signals are aligned around the middle
        shift, _, _ = xcorr(self.a, self.c, 50, full_xcorr=True)
        self.assertEqual(shift, -5 - (len(self.a) - len(self.c)) // 2)

    def test_xcorr_different_implementations(self):
        """
        Test correct length and different implementations against each other
        """
        xcorrs1 = []
        xcorrs2 = []
        for xcorr_func in (_xcorr_padzeros, _xcorr_slice):
            for domain in ('freq', 'time'):
                x = xcorr_func(self.a, self.b, 40, domain=domain)
                y = xcorr_func(self.a, self.b[:-1], 40, domain=domain)
                self.assertEqual((len(self.a) - len(self.b)) % 2, 0)
                self.assertEqual(len(x), 2 * 40 + 1)
                self.assertEqual(len(y), 2 * 40)
                xcorrs1.append(x)
                xcorrs2.append(y)
        for x_other in xcorrs1[1:]:
            np.testing.assert_allclose(x_other, xcorrs1[0])
        for x_other in xcorrs2[1:]:
            np.testing.assert_allclose(x_other, xcorrs2[0])

    def test_xcorr_max(self):
        shift, value = xcorr_max((1, 3, -5))
        self.assertEqual(shift, 1)
        self.assertEqual(value, -5)
        shift, value = xcorr_max((3., -5.), abs_max=False)
        self.assertEqual(shift, -0.5)
        self.assertEqual(value, 3.)

    def test_xcorr_3c(self):
        st = read()
        st2 = read()
        for tr in st2:
            tr.data = -5*np.roll(tr.data, 50)
        shift, value, x = xcorr_3c(st, st2, 200, full_xcorr=True)
        self.assertEqual(shift, -50)
        self.assertAlmostEqual(value, -0.998, 3)

    def test_xcorr_pick_correction(self):
        """
        Test cross correlation pick correction on a set of two small local
        earthquakes.
        """
        st1 = read(os.path.join(self.path,
                                'BW.UH1._.EHZ.D.2010.147.a.slist.gz'))
        st2 = read(os.path.join(self.path,
                                'BW.UH1._.EHZ.D.2010.147.b.slist.gz'))

        tr1 = st1.select(component="Z")[0]
        tr2 = st2.select(component="Z")[0]
        tr1_copy = tr1.copy()
        tr2_copy = tr2.copy()
        t1 = UTCDateTime("2010-05-27T16:24:33.315000Z")
        t2 = UTCDateTime("2010-05-27T16:27:30.585000Z")

        dt, coeff = xcorr_pick_correction(t1, tr1, t2, tr2, 0.05, 0.2, 0.1)
        self.assertAlmostEqual(dt, -0.014459080288833711)
        self.assertAlmostEqual(coeff, 0.91542878457939791)
        dt, coeff = xcorr_pick_correction(t2, tr2, t1, tr1, 0.05, 0.2, 0.1)
        self.assertAlmostEqual(dt, 0.014459080288833711)
        self.assertAlmostEqual(coeff, 0.91542878457939791)
        dt, coeff = xcorr_pick_correction(
            t1, tr1, t2, tr2, 0.05, 0.2, 0.1, filter="bandpass",
            filter_options={'freqmin': 1, 'freqmax': 10})
        self.assertAlmostEqual(dt, -0.013025086360067755)
        self.assertAlmostEqual(coeff, 0.98279277273758803)
        self.assertEqual(tr1, tr1_copy)
        self.assertEqual(tr2, tr2_copy)

    def test_xcorr_pick_correction_images(self):
        """
        Test cross correlation pick correction on a set of two small local
        earthquakes.
        """
        st1 = read(os.path.join(self.path,
                                'BW.UH1._.EHZ.D.2010.147.a.slist.gz'))
        st2 = read(os.path.join(self.path,
                                'BW.UH1._.EHZ.D.2010.147.b.slist.gz'))

        tr1 = st1.select(component="Z")[0]
        tr2 = st2.select(component="Z")[0]
        t1 = UTCDateTime("2010-05-27T16:24:33.315000Z")
        t2 = UTCDateTime("2010-05-27T16:27:30.585000Z")

        with ImageComparison(self.path_images, 'xcorr_pick_corr.png') as ic:
            dt, coeff = xcorr_pick_correction(
                t1, tr1, t2, tr2, 0.05, 0.2, 0.1, plot=True, filename=ic.name)


def suite():
    return unittest.makeSuite(CrossCorrelationTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
