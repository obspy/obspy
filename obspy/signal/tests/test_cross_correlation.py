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
import warnings

from obspy import UTCDateTime, read
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from obspy.core.util.libnames import _load_cdll
from obspy.core.util.testing import ImageComparison
from obspy.signal.cross_correlation import (correlate, correlate_template,
                                            xcorr_pick_correction,
                                            xcorr_3c, xcorr_max, xcorr,
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
        This tests the old, deprecated xcorr() function.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ObsPyDeprecationWarning)
            # example 1 - all samples are equal
            np.random.seed(815)  # make test reproducible
            tr1 = np.random.randn(10000).astype(np.float32)
            tr2 = tr1.copy()
            shift, corr = xcorr(tr1, tr2, 100)
            self.assertEqual(shift, 0)
            self.assertAlmostEqual(corr, 1, 2)
            # example 2 - all samples are different
            tr1 = np.ones(10000, dtype=np.float32)
            tr2 = np.zeros(10000, dtype=np.float32)
            shift, corr = xcorr(tr1, tr2, 100)
            self.assertEqual(shift, 0)
            self.assertAlmostEqual(corr, 0, 2)
            # example 3 - shift of 10 samples
            tr1 = np.random.randn(10000).astype(np.float32)
            tr2 = np.concatenate((np.zeros(10), tr1[0:-10]))
            shift, corr = xcorr(tr1, tr2, 100)
            self.assertEqual(shift, -10)
            self.assertAlmostEqual(corr, 1, 2)
            shift, corr = xcorr(tr2, tr1, 100)
            self.assertEqual(shift, 10)
            self.assertAlmostEqual(corr, 1, 2)
            # example 4 - shift of 10 samples + small sine disturbance
            tr1 = (np.random.randn(10000) * 100).astype(np.float32)
            var = np.sin(np.arange(10000, dtype=np.float32) * 0.1)
            tr2 = np.concatenate((np.zeros(10), tr1[0:-10])) * 0.9
            tr2 += var
            shift, corr = xcorr(tr1, tr2, 100)
            self.assertEqual(shift, -10)
            self.assertAlmostEqual(corr, 1, 2)
            shift, corr = xcorr(tr2, tr1, 100)
            self.assertEqual(shift, 10)
            self.assertAlmostEqual(corr, 1, 2)

    def test_correlate_deprecated_domain_keyword(self):

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=ObsPyDeprecationWarning)
            a = [1, 2, 3]
            b = [1, 2]
            correlate(a, b, 5, domain='freq')
            correlate(a, b, 5, domain='time')
        # on py37, scipy 1.1.0 this also catch FutureWarning from scipy
        # internals, so we need to filter the warning messages
        domain_warn = [x for x in w if 'keyword of correlate function'
                       in str(x.message)]
        self.assertEqual(len(domain_warn), 2)

    def test_correlate_normalize_true_false(self):
        a = read()[0].data[500:]
        b = a[10:]
        shift = 100
        cc1 = correlate(a, b, shift, normalize='naive')
        cc2 = correlate(a, b, shift, normalize=True)
        cc3 = correlate(a, b, shift, normalize=None)
        cc4 = correlate(a, b, shift, normalize=False)
        np.testing.assert_equal(cc1, cc2)
        np.testing.assert_equal(cc3, cc4)

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
        corr_fun = correlate(self.a, self.b, shift=15)
        shift, corr = xcorr_max(corr_fun)
        np.testing.assert_allclose(corr_fun, x)
        self.assertAlmostEqual(corr, 0.96516076)
        self.assertEqual(shift, -5)

    def test_correlate_different_length_of_signals(self):
        # Signals are aligned around the middle
        cc = correlate(self.a, self.c, 50)
        shift, _ = xcorr_max(cc)
        self.assertEqual(shift, -5 - (len(self.a) - len(self.c)) // 2)

    def test_correlate(self):
        # simple test
        a, b = [0, 1], [20, 10]
        cc = correlate(a, b, 1, demean=False, normalize=False)
        shift, value = xcorr_max(cc)
        self.assertEqual(shift, 1)
        self.assertAlmostEqual(value, 20.)
        np.testing.assert_allclose(cc, [0., 10., 20.], atol=1e-14)
        # test symetry and different length of a and b
        a, b = [0, 1, 2], [20, 10]
        cc1 = correlate(a, b, 1, demean=False, normalize=False, method='fft')
        cc2 = correlate(a, b, 1, demean=False, normalize=False,
                        method='direct')
        cc3 = correlate(b, a, 1, demean=False, normalize=False, method='fft')
        cc4 = correlate(b, a, 1, demean=False, normalize=False,
                        method='direct')
        shift1, _ = xcorr_max(cc1)
        shift2, _ = xcorr_max(cc2)
        shift3, _ = xcorr_max(cc3)
        shift4, _ = xcorr_max(cc4)
        self.assertEqual(shift1, 0.5)
        self.assertEqual(shift2, 0.5)
        self.assertEqual(shift3, -0.5)
        self.assertEqual(shift4, -0.5)
        np.testing.assert_allclose(cc1, cc2)
        np.testing.assert_allclose(cc3, cc4)
        np.testing.assert_allclose(cc1, cc3[::-1])
        # test sysmetry for method='direct' and len(a) - len(b) - 2 * num > 0
        a, b = [0, 1, 2, 3, 4, 5, 6, 7], [20, 10]
        cc1 = correlate(a, b, 2, method='direct')
        cc2 = correlate(b, a, 2, method='direct')
        np.testing.assert_allclose(cc1, cc2[::-1])

    def test_correlate_different_implementations(self):
        """
        Test correct length and different implementations against each other
        """
        xcorrs1 = []
        xcorrs2 = []
        for xcorr_func in (_xcorr_padzeros, _xcorr_slice):
            for method in ('auto', 'fft', 'direct'):
                x = xcorr_func(self.a, self.b, 40, method)
                y = xcorr_func(self.a, self.b[:-1], 40, method)
                self.assertEqual((len(self.a) - len(self.b)) % 2, 0)
                self.assertEqual(len(x), 2 * 40 + 1)
                self.assertEqual(len(y), 2 * 40)
                xcorrs1.append(x)
                xcorrs2.append(y)
        for x_other in xcorrs1[1:]:
            np.testing.assert_allclose(x_other, xcorrs1[0])
        for x_other in xcorrs2[1:]:
            np.testing.assert_allclose(x_other, xcorrs2[0])

    def test_correlate_extreme_shifts_for_freq_xcorr(self):
        """
        Also test shift=None
        """
        a, b = [1, 2, 3], [1, 2, 3]
        n = len(a) + len(b) - 1
        cc1 = correlate(a, b, 2, method='fft')
        cc2 = correlate(a, b, 3, method='fft')
        cc3 = correlate(a, b, None, method='fft')
        cc4 = correlate(a, b, None, method='direct')
        self.assertEqual(len(cc1), n)
        self.assertEqual(len(cc2), 2 + n)
        self.assertEqual(len(cc3), n)
        self.assertEqual(len(cc4), n)
        a, b = [1, 2, 3], [1, 2]
        n = len(a) + len(b) - 1
        cc1 = correlate(a, b, 2, method='fft')
        cc2 = correlate(a, b, 3, method='fft')
        cc3 = correlate(a, b, None, method='fft')
        cc4 = correlate(a, b, None, method='direct')
        self.assertEqual(len(cc1), n)
        self.assertEqual(len(cc2), 2 + n)
        self.assertEqual(len(cc3), n)
        self.assertEqual(len(cc4), n)

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
            tr.data = -5 * np.roll(tr.data, 50)
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

    def test_correlate_template_eqcorrscan(self):
        """
        Test for moving window correlations with "full" normalisation.

        Comparison result is from EQcorrscan v.0.2.7, using the following:

        from eqcorrscan.utils.correlate import get_array_xcorr
        from obspy import read

        data = read()[0].data
        template = data[400:600]
        data = data[380:620]
        eqcorrscan_func = get_array_xcorr("fftw")
        result = eqcorrscan_func(
            stream=data, templates=template.reshape(1, len(template)),
            pads=[0])[0][0]
        """
        result = [
            -2.24548906e-01,  7.10350871e-02,  2.68642932e-01,  2.75941312e-01,
            1.66854098e-01,  1.66086946e-02, -1.29057273e-01, -1.96172655e-01,
            -1.41613603e-01, -6.83271606e-03,  1.45768464e-01,  2.42143899e-01,
            1.98310092e-01,  2.16377302e-04, -2.41576880e-01, -4.00586188e-01,
            -4.32240069e-01, -2.88735539e-01,  1.26461715e-01,  7.09268868e-01,
            9.99999940e-01,  7.22769439e-01,  1.75955653e-01, -2.46459037e-01,
            -4.34027880e-01, -4.32590246e-01, -2.67131507e-01, -6.78363896e-04,
            2.08171085e-01,  2.32197508e-01,  8.64804164e-02, -1.14158235e-01,
            -2.53621429e-01, -2.62945205e-01, -1.40505865e-01,  3.35594788e-02,
            1.77415669e-01,  2.72263527e-01,  2.81718552e-01,  1.38080209e-01,
            -1.27307668e-01]
        data = read()[0].data
        template = data[400:600]
        data = data[380:620]
        cc = correlate_template(data, template)
        np.testing.assert_allclose(cc, result, atol=1e-7)
        shift, corr = xcorr_max(cc)
        self.assertAlmostEqual(corr, 1.0)
        self.assertEqual(shift, 0)

    def test_correlate_template_eqcorrscan_time(self):
        """
        Test full normalization for method='direct'.
        """
        result = [
            -2.24548906e-01,  7.10350871e-02,  2.68642932e-01,  2.75941312e-01,
            1.66854098e-01,  1.66086946e-02, -1.29057273e-01, -1.96172655e-01,
            -1.41613603e-01, -6.83271606e-03,  1.45768464e-01,  2.42143899e-01,
            1.98310092e-01,  2.16377302e-04, -2.41576880e-01, -4.00586188e-01,
            -4.32240069e-01, -2.88735539e-01,  1.26461715e-01,  7.09268868e-01,
            9.99999940e-01,  7.22769439e-01,  1.75955653e-01, -2.46459037e-01,
            -4.34027880e-01, -4.32590246e-01, -2.67131507e-01, -6.78363896e-04,
            2.08171085e-01,  2.32197508e-01,  8.64804164e-02, -1.14158235e-01,
            -2.53621429e-01, -2.62945205e-01, -1.40505865e-01,  3.35594788e-02,
            1.77415669e-01,  2.72263527e-01,  2.81718552e-01,  1.38080209e-01,
            -1.27307668e-01]
        data = read()[0].data
        template = data[400:600]
        data = data[380:620]
        cc = correlate_template(data, template, method='direct')
        np.testing.assert_allclose(cc, result, atol=1e-7)
        shift, corr = xcorr_max(cc)
        self.assertAlmostEqual(corr, 1.0)
        self.assertEqual(shift, 0)

    def test_correlate_template_different_normalizations(self):
        data = read()[0].data
        template = data[400:600]
        data = data[380:700]
        max_index = 20
        ct = correlate_template
        full_xcorr = ct(data, template, demean=False)
        naive_xcorr = ct(data, template, demean=False, normalize='naive')
        nonorm_xcorr = ct(data, template, demean=False, normalize=None)
        self.assertEqual(np.argmax(full_xcorr), max_index)
        self.assertEqual(np.argmax(naive_xcorr), max_index)
        self.assertEqual(np.argmax(nonorm_xcorr), max_index)
        self.assertAlmostEqual(full_xcorr[max_index], 1.0)
        self.assertLess(naive_xcorr[max_index], full_xcorr[max_index])
        np.testing.assert_allclose(nonorm_xcorr, np.correlate(data, template))

    def test_correlate_template_correct_alignment_of_normalization(self):
        data = read()[0].data
        template = data[400:600]
        data = data[380:620]
        # test for all combinations of odd and even length input data
        for i1, i2 in ((0, 0), (0, 1), (1, 1), (1, 0)):
            for mode in ('valid', 'same', 'full'):
                for demean in (True, False):
                    xcorr = correlate_template(data[i1:], template[i2:],
                                               mode=mode, demean=demean)
                    self.assertAlmostEqual(np.max(xcorr), 1)

    def test_correlate_template_versus_correlate(self):
        data = read()[0].data
        template = data[400:600]
        data = data[380:620]
        xcorr1 = correlate_template(data, template, normalize='naive')
        xcorr2 = correlate(data, template, 20)
        np.testing.assert_equal(xcorr1, xcorr2)

    def test_correlate_template_zeros_in_input(self):
        template = np.zeros(10)
        data = read()[0].data[380:420]
        xcorr = correlate_template(data, template)
        np.testing.assert_equal(xcorr, np.zeros(len(xcorr)))
        template[:] = data[:10]
        data[5:20] = 0
        xcorr = correlate_template(data, template)
        np.testing.assert_equal(xcorr[5:11], np.zeros(6))
        data[:] = 0
        xcorr = correlate_template(data, template)
        np.testing.assert_equal(xcorr, np.zeros(len(xcorr)))
        xcorr = correlate_template(data, template, normalize='naive')
        np.testing.assert_equal(xcorr, np.zeros(len(xcorr)))

    def test_correlate_template_different_amplitudes(self):
        """
        Check that correlations are the same independent of template amplitudes
        """
        data = np.random.randn(20000)
        template = data[1000:1200]
        template_large = template * 10e10
        template_small = template * 10e-10

        cc = correlate_template(data, template)
        cc_large = correlate_template(data, template_large)
        cc_small = correlate_template(data, template_small)
        np.testing.assert_allclose(cc, cc_large)
        np.testing.assert_allclose(cc, cc_small)

    def test_correlate_template_nodemean_fastmatchedfilter(self):
        """
        Compare non-demeaned result against FMF derived result.

        FMF result obtained by the following:

        import copy
        import numpy as np
        from fast_matched_filter import matched_filter
        from obspy import read

        data = read()[0].data
        template = copy.deepcopy(data[400:600])
        data = data[380:620]
        result = matched_filter(
            templates=template.reshape(1, 1, 1, len(template)),
            moveouts=np.array(0).reshape(1, 1, 1),
            weights=np.array(1).reshape(1, 1, 1),
            data=data.reshape(1, 1, len(data)),
            step=1, arch='cpu')[0]

        .. note::
            FastMatchedFilter doesn't use semver, but result generated by Calum
            Chamberlain on 18 Jan 2018 using up-to-date code, with the patch
            in https://github.com/beridel/fast_matched_filter/pull/12
        """
        result = [
            -1.48108244e-01,   4.71532270e-02,   1.82797655e-01,
            1.92574233e-01,   1.18700281e-01,   1.18958903e-02,
            -9.23405439e-02,  -1.40047163e-01,  -1.00863703e-01,
            -4.86961426e-03,   1.04124829e-01,   1.72662303e-01,
            1.41110823e-01,   1.53776666e-04,  -1.71214968e-01,
            -2.83201426e-01,  -3.04899812e-01,  -2.03215942e-01,
            8.88349637e-02,   5.00749528e-01,   7.18140483e-01,
            5.29728174e-01,   1.30591258e-01,  -1.83402568e-01,
            -3.22406143e-01,  -3.20676118e-01,  -1.98054180e-01,
            -5.06028766e-04,   1.56253457e-01,   1.74580097e-01,
            6.49696961e-02,  -8.56237561e-02,  -1.89858019e-01,
            -1.96504310e-01,  -1.04968190e-01,   2.51029599e-02,
            1.32686019e-01,   2.03692451e-01,   2.11983219e-01,
            0.00000000e+00,   0.00000000e+00]
        data = read()[0].data
        template = data[400:600]
        data = data[380:620]
        # FMF demeans template but does not locally demean data for
        # normalization
        template = template - template.mean()
        cc = correlate_template(data, template, demean=False)
        # FMF misses the last two elements?
        np.testing.assert_allclose(cc[0:-2], result[0:-2], atol=1e-7)
        shift, corr = xcorr_max(cc)
        self.assertEqual(shift, 0)

    def test_integer_input_equals_float_input(self):
        a = [-3, 0, 4]
        b = [-3, 4]
        c = np.array(a, dtype=float)
        d = np.array(b, dtype=float)
        for demean in (True, False):
            for normalize in (None, 'naive'):
                cc1 = correlate(a, b, 3, demean=demean, normalize=normalize,
                                method='direct')
                cc2 = correlate(c, d, 3, demean=demean, normalize=normalize)
                np.testing.assert_allclose(cc1, cc2)
            for normalize in (None, 'naive', 'full'):
                cc3 = correlate_template(a, b, demean=demean,
                                         normalize=normalize, method='direct')
                cc4 = correlate_template(c, d, demean=demean,
                                         normalize=normalize)
                np.testing.assert_allclose(cc3, cc4)


def suite():
    return unittest.makeSuite(CrossCorrelationTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
