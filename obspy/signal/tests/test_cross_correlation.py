# -*- coding: utf-8 -*-
"""
The cross correlation test suite.
"""
import ctypes as C  # NOQA
import numpy as np
import os
import warnings

import pytest

from obspy import UTCDateTime, read, Trace
from obspy.core.util import AttribDict
from obspy.core.util.libnames import _load_cdll
from obspy.signal.cross_correlation import (
    correlate, correlate_template, correlate_stream_template,
    correlation_detector,
    xcorr_pick_correction, xcorr_3c, xcorr_max,
    _xcorr_padzeros, _xcorr_slice, _find_peaks)
from obspy.signal.trigger import coincidence_trigger


class TestCrossCorrelation:

    """
    Cross correlation test case
    """
    @pytest.fixture(scope='class')
    def state(self):
        """Return test state."""
        out = AttribDict()
        out.path = os.path.join(os.path.dirname(__file__), 'data')
        out.path_images = os.path.join(os.path.dirname(__file__), 'images')
        out.a = np.sin(np.linspace(0, 10, 101))
        out.b = 5 * np.roll(out.a, 5)
        out.c = 5 * np.roll(out.a[:81], 5)
        return out

    def test_correlate_normalize_true_false(self):
        a = read()[0].data[500:]
        b = a[10:]
        shift = 100
        cc1 = correlate(a, b, shift, normalize='naive')
        cc2 = correlate(a, b, shift, normalize=True)
        cc3 = correlate(a, b, shift, normalize=None)
        cc4 = correlate(a, b, shift, normalize=False)
        np.testing.assert_allclose(cc1, cc2, rtol=1e-6)
        np.testing.assert_allclose(cc3, cc4, rtol=1e-6)

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

        assert 0 == res
        assert round(abs(0.0-shift.value), 7) == 0
        assert round(abs(1.0-coe_p.value), 7) == 0

    def test_xcorr_vs_old_implementation(self, state):
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
        corr_fun = correlate(state.a, state.b, shift=15)
        shift, corr = xcorr_max(corr_fun)
        np.testing.assert_allclose(corr_fun, x)
        assert round(abs(corr-0.96516076), 7) == 0
        assert shift == -5

    def test_correlate_different_length_of_signals(self, state):
        # Signals are aligned around the middle
        cc = correlate(state.a, state.c, 50)
        shift, _ = xcorr_max(cc)
        assert shift == -5 - (len(state.a) - len(state.c)) // 2

    def test_correlate(self):
        # simple test
        a, b = [0, 1], [20, 10]
        cc = correlate(a, b, 1, demean=False, normalize=False)
        shift, value = xcorr_max(cc)
        assert shift == 1
        assert round(abs(value-20.), 7) == 0
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
        assert shift1 == 0.5
        assert shift2 == 0.5
        assert shift3 == -0.5
        assert shift4 == -0.5
        np.testing.assert_allclose(cc1, cc2)
        np.testing.assert_allclose(cc3, cc4)
        np.testing.assert_allclose(cc1, cc3[::-1])
        # test sysmetry for method='direct' and len(a) - len(b) - 2 * num > 0
        a, b = [0, 1, 2, 3, 4, 5, 6, 7], [20, 10]
        cc1 = correlate(a, b, 2, method='direct')
        cc2 = correlate(b, a, 2, method='direct')
        np.testing.assert_allclose(cc1, cc2[::-1])

    def test_correlate_different_implementations(self, state):
        """
        Test correct length and different implementations against each other
        """
        xcorrs1 = []
        xcorrs2 = []
        for xcorr_func in (_xcorr_padzeros, _xcorr_slice):
            for method in ('auto', 'fft', 'direct'):
                x = xcorr_func(state.a, state.b, 40, method)
                y = xcorr_func(state.a, state.b[:-1], 40, method)
                assert (len(state.a) - len(state.b)) % 2 == 0
                assert len(x) == 2 * 40 + 1
                assert len(y) == 2 * 40
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
        assert len(cc1) == n
        assert len(cc2) == 2 + n
        assert len(cc3) == n
        assert len(cc4) == n
        a, b = [1, 2, 3], [1, 2]
        n = len(a) + len(b) - 1
        cc1 = correlate(a, b, 2, method='fft')
        cc2 = correlate(a, b, 3, method='fft')
        cc3 = correlate(a, b, None, method='fft')
        cc4 = correlate(a, b, None, method='direct')
        assert len(cc1) == n
        assert len(cc2) == 2 + n
        assert len(cc3) == n
        assert len(cc4) == n

    def test_xcorr_max(self):
        shift, value = xcorr_max((1, 3, -5))
        assert shift == 1
        assert value == -5
        shift, value = xcorr_max((3., -5.), abs_max=False)
        assert shift == -0.5
        assert value == 3.

    def test_xcorr_3c(self):
        st = read()
        st2 = read()
        for tr in st2:
            tr.data = -5 * np.roll(tr.data, 50)
        shift, value, x = xcorr_3c(st, st2, 200, full_xcorr=True)
        assert shift == -50
        assert round(abs(value--0.998), 3) == 0

    def test_xcorr_pick_correction(self, state):
        """
        Test cross correlation pick correction on a set of two small local
        earthquakes.
        """
        st1 = read(os.path.join(state.path,
                                'BW.UH1._.EHZ.D.2010.147.a.slist.gz'))
        st2 = read(os.path.join(state.path,
                                'BW.UH1._.EHZ.D.2010.147.b.slist.gz'))

        tr1 = st1.select(component="Z")[0]
        tr2 = st2.select(component="Z")[0]
        tr1_copy = tr1.copy()
        tr2_copy = tr2.copy()
        t1 = UTCDateTime("2010-05-27T16:24:33.315000Z")
        t2 = UTCDateTime("2010-05-27T16:27:30.585000Z")

        dt, coeff = xcorr_pick_correction(t1, tr1, t2, tr2, 0.05, 0.2, 0.1)
        assert round(abs(dt--0.014459080288833711), 7) == 0
        assert round(abs(coeff-0.91542878457939791), 7) == 0
        dt, coeff = xcorr_pick_correction(t2, tr2, t1, tr1, 0.05, 0.2, 0.1)
        assert round(abs(dt-0.014459080288833711), 7) == 0
        assert round(abs(coeff-0.91542878457939791), 7) == 0
        dt, coeff = xcorr_pick_correction(
            t1, tr1, t2, tr2, 0.05, 0.2, 0.1, filter="bandpass",
            filter_options={'freqmin': 1, 'freqmax': 10})
        assert round(abs(dt--0.013025086360067755), 7) == 0
        assert round(abs(coeff-0.98279277273758803), 7) == 0
        assert tr1 == tr1_copy
        assert tr2 == tr2_copy

    def test_xcorr_pick_correction_images(self, state, image_path):
        """
        Test cross correlation pick correction on a set of two small local
        earthquakes.
        """
        st1 = read(os.path.join(state.path,
                                'BW.UH1._.EHZ.D.2010.147.a.slist.gz'))
        st2 = read(os.path.join(state.path,
                                'BW.UH1._.EHZ.D.2010.147.b.slist.gz'))

        tr1 = st1.select(component="Z")[0]
        tr2 = st2.select(component="Z")[0]
        t1 = UTCDateTime("2010-05-27T16:24:33.315000Z")
        t2 = UTCDateTime("2010-05-27T16:27:30.585000Z")

        xcorr_pick_correction(
            t1, tr1, t2, tr2, 0.05, 0.2, 0.1, plot=True, filename=image_path)

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
            -2.24548906e-01, 7.10350871e-02, 2.68642932e-01, 2.75941312e-01,
            1.66854098e-01, 1.66086946e-02, -1.29057273e-01, -1.96172655e-01,
            -1.41613603e-01, -6.83271606e-03, 1.45768464e-01, 2.42143899e-01,
            1.98310092e-01, 2.16377302e-04, -2.41576880e-01, -4.00586188e-01,
            -4.32240069e-01, -2.88735539e-01, 1.26461715e-01, 7.09268868e-01,
            9.99999940e-01, 7.22769439e-01, 1.75955653e-01, -2.46459037e-01,
            -4.34027880e-01, -4.32590246e-01, -2.67131507e-01, -6.78363896e-04,
            2.08171085e-01, 2.32197508e-01, 8.64804164e-02, -1.14158235e-01,
            -2.53621429e-01, -2.62945205e-01, -1.40505865e-01, 3.35594788e-02,
            1.77415669e-01, 2.72263527e-01, 2.81718552e-01, 1.38080209e-01,
            -1.27307668e-01]
        data = read()[0].data
        template = data[400:600]
        data = data[380:620]
        cc = correlate_template(data, template)
        np.testing.assert_allclose(cc, result, atol=1e-7)
        shift, corr = xcorr_max(cc)
        assert round(abs(corr-1.0), 7) == 0
        assert shift == 0

    def test_correlate_template_eqcorrscan_time(self):
        """
        Test full normalization for method='direct'.
        """
        result = [
            -2.24548906e-01, 7.10350871e-02, 2.68642932e-01, 2.75941312e-01,
            1.66854098e-01, 1.66086946e-02, -1.29057273e-01, -1.96172655e-01,
            -1.41613603e-01, -6.83271606e-03, 1.45768464e-01, 2.42143899e-01,
            1.98310092e-01, 2.16377302e-04, -2.41576880e-01, -4.00586188e-01,
            -4.32240069e-01, -2.88735539e-01, 1.26461715e-01, 7.09268868e-01,
            9.99999940e-01, 7.22769439e-01, 1.75955653e-01, -2.46459037e-01,
            -4.34027880e-01, -4.32590246e-01, -2.67131507e-01, -6.78363896e-04,
            2.08171085e-01, 2.32197508e-01, 8.64804164e-02, -1.14158235e-01,
            -2.53621429e-01, -2.62945205e-01, -1.40505865e-01, 3.35594788e-02,
            1.77415669e-01, 2.72263527e-01, 2.81718552e-01, 1.38080209e-01,
            -1.27307668e-01]
        data = read()[0].data
        template = data[400:600]
        data = data[380:620]
        cc = correlate_template(data, template, method='direct')
        np.testing.assert_allclose(cc, result, atol=1e-7)
        shift, corr = xcorr_max(cc)
        assert round(abs(corr-1.0), 7) == 0
        assert shift == 0

    def test_correlate_template_different_normalizations(self):
        data = read()[0].data
        template = data[400:600]
        data = data[380:700]
        max_index = 20
        ct = correlate_template
        full_xcorr = ct(data, template, demean=False)
        naive_xcorr = ct(data, template, demean=False, normalize='naive')
        nonorm_xcorr = ct(data, template, demean=False, normalize=None)
        assert np.argmax(full_xcorr) == max_index
        assert np.argmax(naive_xcorr) == max_index
        assert np.argmax(nonorm_xcorr) == max_index
        assert round(abs(full_xcorr[max_index]-1.0), 7) == 0
        assert naive_xcorr[max_index] < full_xcorr[max_index]
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
                    assert round(abs(np.max(xcorr)-1), 7) == 0

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
            -1.48108244e-01, 4.71532270e-02, 1.82797655e-01,
            1.92574233e-01, 1.18700281e-01, 1.18958903e-02,
            -9.23405439e-02, -1.40047163e-01, -1.00863703e-01,
            -4.86961426e-03, 1.04124829e-01, 1.72662303e-01,
            1.41110823e-01, 1.53776666e-04, -1.71214968e-01,
            -2.83201426e-01, -3.04899812e-01, -2.03215942e-01,
            8.88349637e-02, 5.00749528e-01, 7.18140483e-01,
            5.29728174e-01, 1.30591258e-01, -1.83402568e-01,
            -3.22406143e-01, -3.20676118e-01, -1.98054180e-01,
            -5.06028766e-04, 1.56253457e-01, 1.74580097e-01,
            6.49696961e-02, -8.56237561e-02, -1.89858019e-01,
            -1.96504310e-01, -1.04968190e-01, 2.51029599e-02,
            1.32686019e-01, 2.03692451e-01, 2.11983219e-01,
            0.00000000e+00, 0.00000000e+00]
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
        assert shift == 0

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

    def test_correlate_stream_template_and_correlation_detector(self):
        template = read().filter('highpass', freq=5).normalize()
        pick = UTCDateTime('2009-08-24T00:20:07.73')
        template.trim(pick, pick + 10)
        n1 = len(template[0])
        n2 = 100 * 3600  # 1 hour
        dt = template[0].stats.delta
        # shift one template Trace
        template[1].stats.starttime += 5
        stream = template.copy()
        np.random.seed(42)
        for tr, trt in zip(stream, template):
            tr.stats.starttime += 24 * 3600
            tr.data = np.random.random(n2) - 0.5  # noise
            if tr.stats.channel[-1] == 'Z':
                tr.data[n1:2 * n1] += 10 * trt.data
                tr.data = tr.data[:-n1]
            tr.data[5 * n1:6 * n1] += 100 * trt.data
            tr.data[20 * n1:21 * n1] += 2 * trt.data
        # make one template trace a bit shorter
        template[2].data = template[2].data[:-n1 // 5]
        # make two stream traces a bit shorter
        stream[0].trim(5, None)
        stream[1].trim(1, 20)
        # second template
        pick2 = stream[0].stats.starttime + 20 * n1 * dt
        template2 = stream.slice(pick2 - 5, pick2 + 5)
        # test cross correlation
        stream_orig = stream.copy()
        template_orig = template.copy()
        ccs = correlate_stream_template(stream, template)
        assert len(ccs) == len(stream)
        assert stream[1].stats.starttime == ccs[0].stats.starttime
        assert stream_orig == stream
        assert template_orig == template
        # test if traces with not matching seed ids are discarded
        ccs = correlate_stream_template(stream[:2], template[1:])
        assert len(ccs) == 1
        assert stream_orig == stream
        assert template_orig == template
        # test template_time parameter
        ccs1 = correlate_stream_template(stream, template)
        template_time = template[0].stats.starttime + 100
        ccs2 = correlate_stream_template(stream, template,
                                         template_time=template_time)
        assert len(ccs2) == len(ccs1)
        delta = ccs2[0].stats.starttime - ccs1[0].stats.starttime
        assert round(abs(delta-100), 7) == 0
        # test if all three events found
        detections, sims = correlation_detector(stream, template, 0.2, 30)
        assert len(detections) == 3
        dtime = pick + n1 * dt + 24 * 3600
        assert round(abs(detections[0]['time']-dtime), 7) == 0
        assert len(sims) == 1
        assert stream_orig == stream
        assert template_orig == template
        # test if xcorr stream is suitable for coincidence_trigger
        # result should be the same, return values related
        ccs = correlate_stream_template(stream, template)
        triggers = coincidence_trigger(None, 0.2, -1, ccs, 2,
                                       max_trigger_length=30, details=True)
        assert len(triggers) == 2
        for d, t in zip(detections[1:], triggers):
            assert round(abs(np.mean(t['cft_peaks'])-d['similarity']), 7) == 0
        # test template_magnitudes
        detections, _ = correlation_detector(stream, template, 0.2, 30,
                                             template_magnitudes=1)
        assert abs(detections[1]['amplitude_ratio']-100) < 1
        assert abs(detections[1]['magnitude'] - (1 + 8 / 3)) < 0.01
        assert abs(detections[2]['amplitude_ratio']-2) < 2
        detections, _ = correlation_detector(stream, template, 0.2, 30,
                                             template_magnitudes=True)
        assert abs(detections[1]['amplitude_ratio']-100) < 1
        assert 'magnitude' not in detections[1]
        assert stream_orig == stream
        assert template_orig == template
        # test template names
        detections, _ = correlation_detector(stream, template, 0.2, 30,
                                             template_names='eq')
        assert detections[0]['template_name'] == 'eq'
        detections, _ = correlation_detector(stream, template, 0.2, 30,
                                             template_names=['eq'], plot=True)
        assert detections[0]['template_name'] == 'eq'
        # test similarity parameter with additional constraints
        # test details=True

        def simf(ccs):
            ccmatrix = np.array([tr.data for tr in ccs])
            comp_thres = np.sum(ccmatrix > 0.2, axis=0) > 1
            similarity = ccs[0].copy()
            similarity.data = np.mean(ccmatrix, axis=0) * comp_thres
            return similarity
        detections, _ = correlation_detector(stream, template, 0.1, 30,
                                             similarity_func=simf,
                                             details=True)
        assert len(detections) == 2
        for d in detections:
            mean_val = np.mean(list(d['cc_values'].values()))
            assert round(abs(mean_val - d['similarity']), 7) == 0
        # test if properties from find_peaks function are returned
        detections, sims = correlation_detector(stream, template, 0.1, 30,
                                                threshold=0.16, details=True,
                                                similarity_func=simf)
        try:
            from scipy.signal import find_peaks  # noqa
        except ImportError:
            assert len(detections) == 2
            assert 'left_threshold' not in detections[0]
        else:
            assert len(detections) == 1
            assert 'left_threshold' in detections[0]
        # also check the _find_peaks function
        distance = int(round(30 / sims[0].stats.delta))
        indices = _find_peaks(sims[0].data, 0.1, distance, distance)
        assert len(indices) == 2
        # test distance parameter
        detections, _ = correlation_detector(stream, template, 0.2, 500)
        assert len(detections) == 1
        # test more than one template
        # just 2 detections for first template, because second template has
        # a higher similarity for third detection
        templates = (template, template2)
        templatetime2 = pick2 - 10
        template_times = (template[0].stats.starttime, templatetime2)
        detections, _ = correlation_detector(stream, templates, (0.2, 0.3), 30,
                                             plot=stream,
                                             template_times=template_times,
                                             template_magnitudes=(2, 5))
        assert len(detections) > 0
        assert 'template_id' in detections[0]
        detections0 = [d for d in detections if d['template_id'] == 0]
        assert len(detections0) == 2
        assert len(detections) == 3
        assert round(abs(detections[2]['similarity']-1), 7) == 0
        assert round(abs(detections[2]['magnitude']-5), 7) == 0
        assert detections[2]['time'] == templatetime2
        # test if everything is correct if template2 and stream do not have
        # any ids in common
        templates = (template, template2[2:])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detections, sims = correlation_detector(
                stream[:1], templates, 0.2, 30, plot=True,
                template_times=templatetime2, template_magnitudes=2)
        detections0 = [d for d in detections if d['template_id'] == 0]
        assert len(detections0) == 3
        assert len(detections) == 3
        assert len(sims) == 2
        assert isinstance(sims[0], Trace)
        assert sims[1] is None
