# -*- coding: utf-8 -*-
"""
The obspy.signal.trigger test suite.
"""
import gzip
import re
import warnings
from ctypes import ArgumentError

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from obspy import Stream, UTCDateTime, read
from obspy.signal.trigger import (
    ar_pick, classic_sta_lta, classic_sta_lta_py, coincidence_trigger, pk_baer,
    recursive_sta_lta, recursive_sta_lta_py, trigger_onset, aic_simple,
    energy_ratio, modified_energy_ratio)
from obspy.signal.util import clibsignal


def aic_simple_python(a):
    if len(a) <= 2:
        return np.zeros(len(a), dtype=np.float64)
    a = np.asarray(a)
    aic_cf = np.zeros(a.size - 1, dtype=np.float64)
    with np.errstate(divide='ignore'):
        aic_cf[0] = (a.size - 2) * np.log(np.var(a[1:]))
        aic_cf[-1] = (a.size - 1) * np.log(np.var(a[:-1]))
        for ii in range(2, a.size - 1):
            var1 = np.log(np.var(a[:ii]))
            var2 = np.log(np.var(a[ii:]))
            val1 = ii * var1
            val2 = (a.size - ii - 1) * var2
            aic_cf[ii - 1] = (val1 + val2)
    aic_cf = np.r_[aic_cf, aic_cf[-1]]
    return aic_cf


class TestTrigger():
    """
    Test cases for obspy.trigger
    """
    @classmethod
    def setup_class(cls):
        # random seed data
        np.random.seed(815)
        cls.data = np.random.randn(int(1e5))

    def test_rec_sta_lta_c(self):
        """
        Test case for ctypes version of recursive_sta_lta
        """
        nsta, nlta = 5, 10
        c1 = recursive_sta_lta(self.data, nsta, nlta)
        assert round(abs(c1[99]-0.80810165), 7) == 0
        assert round(abs(c1[100]-0.75939449), 7) == 0
        assert round(abs(c1[101]-0.91763978), 7) == 0
        assert round(abs(c1[102]-0.97465004), 7) == 0

    def test_rec_sta_lta_python(self):
        """
        Test case for python version of recursive_sta_lta
        """
        nsta, nlta = 5, 10
        c2 = recursive_sta_lta_py(self.data, nsta, nlta)
        assert round(abs(c2[99]-0.80810165), 7) == 0
        assert round(abs(c2[100]-0.75939449), 7) == 0
        assert round(abs(c2[101]-0.91763978), 7) == 0
        assert round(abs(c2[102]-0.97465004), 7) == 0

    def test_rec_sta_lta_raise(self):
        """
        Type checking recursive_sta_lta
        """
        ndat = 1
        charfct = np.empty(ndat, dtype=np.float64)
        with pytest.raises(ArgumentError):
            clibsignal.recstalta(
                [1], charfct, ndat, 5, 10)
        with pytest.raises(ArgumentError):
            clibsignal.recstalta(
                np.array([1], dtype=np.int32), charfct, ndat, 5, 10)

    def test_pk_baer(self, testdata):
        """
        Test pk_baer against implementation for UNESCO short course
        """
        with gzip.open(testdata['manz_waldk.a01.gz']) as f:
            data = np.loadtxt(f, dtype=np.float32)
        df, ntdownmax, ntupevent, thr1, thr2, npreset_len, np_dur = \
            (200.0, 20, 60, 7.0, 12.0, 100, 100)
        nptime, pfm = pk_baer(data, df, ntdownmax, ntupevent,
                              thr1, thr2, npreset_len, np_dur)
        assert nptime == 17545
        assert pfm == 'IPU0'

    def test_pk_baer_cf(self, testdata):
        """
        Test pk_baer against implementation for UNESCO short course
        """
        with gzip.open(testdata['manz_waldk.a01.gz']) as f:
            data = np.loadtxt(f, dtype=np.float32)
        df, ntdownmax, ntupevent, thr1, thr2, npreset_len, np_dur = \
            (200.0, 20, 60, 7.0, 12.0, 100, 100)
        nptime, pfm, cf = pk_baer(data, df, ntdownmax, ntupevent,
                                  thr1, thr2, npreset_len, np_dur,
                                  return_cf=True)
        assert nptime == 17545
        assert pfm == 'IPU0'
        assert len(cf) == 119999

    def test_aic_simple_constant_data(self):
        data = [1] * 10
        # all negative inf
        assert_array_equal(aic_simple(data), -np.inf)

    def test_aic_simple_small_size(self):
        data = [3, 4]
        assert_array_equal(aic_simple(data), [0, 0])

    def test_aic_simple(self):
        np.random.seed(0)
        data = np.random.rand(100)
        aic = aic_simple(data)
        assert len(aic) == len(data)
        aic_true = aic_simple_python(data)
        assert_array_almost_equal(aic, aic_true)

    def test_ar_pick(self, testdata):
        """
        Test ar_pick against implementation for UNESCO short course
        """
        data = []
        for channel in ['z', 'n', 'e']:
            filename = 'loc_RJOB20050801145719850.' + channel
            data.append(np.loadtxt(testdata[filename], dtype=np.float32))
        # some default arguments
        samp_rate, f1, f2, lta_p, sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s = \
            200.0, 1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2
        ptime, stime = ar_pick(data[0], data[1], data[2], samp_rate, f1, f2,
                               lta_p, sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s)
        assert round(abs(ptime-30.6350002289), 7) == 0
        # seems to be strongly machine dependent, go for int for 64 bit
        # self.assertEqual(int(stime + 0.5), 31)
        assert abs(stime-31.165) < 0.05

        # All three arrays must have the same length, otherwise an error is
        # raised.
        msg = "All three data arrays must have the same length."
        with pytest.raises(ValueError, match=msg):
            ar_pick(data[0], data[1], np.zeros(1), samp_rate, f1, f2, lta_p,
                    sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s)

    def test_ar_pick_low_amplitude(self, testdata):
        """
        Test ar_pick with low amplitude data
        """
        data = []
        for channel in ['z', 'n', 'e']:
            filename = 'loc_RJOB20050801145719850.' + channel
            data.append(np.loadtxt(testdata[filename], dtype=np.float32))

        # articially reduce signal amplitude
        for d in data:
            d /= 10.0 * d.max()

        # some default arguments
        samp_rate, f1, f2, lta_p, sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s = \
            200.0, 1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2
        ptime, stime = ar_pick(data[0], data[1], data[2], samp_rate, f1, f2,
                               lta_p, sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s)
        assert round(abs(ptime-30.6350002289), 7) == 0
        # seems to be strongly machine dependent, go for int for 64 bit
        # self.assertAlmostEqual(stime, 31.2800006866)
        assert int(stime + 0.5) == 31

    def test_ar_pick_3552(self, testdata):
        """
        Regression test for PR 3552
        """
        tr = read(testdata['arpick_pr3552.mseed'], 'MSEED')[0]
        # no need to test any values etc, the following was crashing
        # obspy/python/pytest with a segfault
        ar_pick(tr.data, np.arange(len(tr)), np.arange(len(tr)),
                tr.stats.sampling_rate, 1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8,
                0.1, 0.2)

    def test_trigger_onset(self):
        """
        Test trigger onset function
        """
        on_of = np.array([[6.0, 31], [69, 94], [131, 181], [215, 265],
                          [278, 315], [480, 505], [543, 568], [605, 631]])
        cft = np.concatenate((np.sin(np.arange(0, 5 * np.pi, 0.1)) + 1,
                              np.sin(np.arange(0, 5 * np.pi, 0.1)) + 2.1,
                              np.sin(np.arange(0, 5 * np.pi, 0.1)) + 0.4,
                              np.sin(np.arange(0, 5 * np.pi, 0.1)) + 1))
        picks = trigger_onset(cft, 1.5, 1.0, max_len=50)
        np.testing.assert_array_equal(picks, on_of)
        # check that max_len_delete drops the picks
        picks_del = trigger_onset(cft, 1.5, 1.0, max_len=50,
                                  max_len_delete=True)
        np.testing.assert_array_equal(
            picks_del, on_of[np.array([0, 1, 5, 6, 7])])
        #
        # set True for visual understanding the tests
        if False:  # pragma: no cover
            import matplotlib.pyplot as plt
            plt.plot(cft)
            plt.hlines([1.5, 1.0], 0, len(cft))
            on_of = np.array(on_of)
            plt.vlines(picks[:, 0], 1.0, 2.0, color='g', linewidth=2,
                       label="ON max_len")
            plt.vlines(picks[:, 1], 0.5, 1.5, color='r', linewidth=2,
                       label="OF max_len")
            plt.vlines(picks_del[:, 0] + 2, 1.0, 2.0, color='y', linewidth=2,
                       label="ON max_len_delete")
            plt.vlines(picks_del[:, 1] + 2, 0.5, 1.5, color='b', linewidth=2,
                       label="OF max_len_delete")
            plt.legend()
            plt.show()

    def test_trigger_onset_issue_2891(self, testdata):
        """
        Regression test for issue 2891

        This used to raise an error if a trigger was activated near the end of
        the trace, and all sample values after that trigger on threshold are
        above the designated off threshold. So basically this can only happen
        if the on threshold is below the off threshold, which is kind of
        unusual, but we fixed it nevertheless, since people can run into this
        playing around with different threshold settings
        """
        tr = read(testdata['BW.UH1._.EHZ.D.2010.147.a.slist.gz'])[0]
        cft = recursive_sta_lta(tr.data, 5, 30)
        trigger_onset(cft, 2.5, 3.2)

    def test_coincidence_trigger(self, testdata):
        """
        Test network coincidence trigger.
        """
        st = Stream()
        files = ["BW.UH1._.SHZ.D.2010.147.cut.slist.gz",
                 "BW.UH2._.SHZ.D.2010.147.cut.slist.gz",
                 "BW.UH3._.SHZ.D.2010.147.cut.slist.gz",
                 "BW.UH4._.EHZ.D.2010.147.cut.slist.gz"]
        for filename in files:
            st += read(testdata[filename])
        # some prefiltering used for UH network
        st.filter('bandpass', freqmin=10, freqmax=20)

        # 1. no weighting, no stations specified, good settings
        # => 3 events, no false triggers
        # for the first test we make some additional tests regarding types
        res = coincidence_trigger("recstalta", 3.5, 1, st.copy(), 3, sta=0.5,
                                  lta=10)
        assert isinstance(res, list)
        assert len(res) == 3
        expected_keys = ['time', 'coincidence_sum', 'duration', 'stations',
                         'trace_ids']
        expected_types = [UTCDateTime, float, float, list, list]
        for item in res:
            assert isinstance(item, dict)
            for key, _type in zip(expected_keys, expected_types):
                assert key in item
                assert isinstance(item[key], _type)
        assert res[0]['time'] > UTCDateTime("2010-05-27T16:24:31")
        assert res[0]['time'] < UTCDateTime("2010-05-27T16:24:35")
        assert 4.2 < res[0]['duration'] < 4.8
        assert res[0]['stations'] == ['UH3', 'UH2', 'UH1', 'UH4']
        assert res[0]['coincidence_sum'] == 4
        assert res[1]['time'] > UTCDateTime("2010-05-27T16:26:59")
        assert res[1]['time'] < UTCDateTime("2010-05-27T16:27:03")
        assert 3.2 < res[1]['duration'] < 3.7
        assert res[1]['stations'] == ['UH2', 'UH3', 'UH1']
        assert res[1]['coincidence_sum'] == 3
        assert res[2]['time'] > UTCDateTime("2010-05-27T16:27:27")
        assert res[2]['time'] < UTCDateTime("2010-05-27T16:27:33")
        assert 4.2 < res[2]['duration'] < 4.4
        assert res[2]['stations'] == ['UH3', 'UH2', 'UH1', 'UH4']
        assert res[2]['coincidence_sum'] == 4

        # 2. no weighting, station selection
        # => 2 events, no false triggers
        trace_ids = ['BW.UH1..SHZ', 'BW.UH3..SHZ', 'BW.UH4..EHZ']
        # raises "UserWarning: At least one trace's ID was not found"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', UserWarning)
            re = coincidence_trigger("recstalta", 3.5, 1, st.copy(), 3,
                                     trace_ids=trace_ids, sta=0.5, lta=10)
            assert len(w) == 1
            assert "At least one trace's ID was not" in str(w[0])
        assert len(re) == 2
        assert re[0]['time'] > UTCDateTime("2010-05-27T16:24:31")
        assert re[0]['time'] < UTCDateTime("2010-05-27T16:24:35")
        assert 4.2 < re[0]['duration'] < 4.8
        assert re[0]['stations'] == ['UH3', 'UH1', 'UH4']
        assert re[0]['coincidence_sum'] == 3
        assert re[1]['time'] > UTCDateTime("2010-05-27T16:27:27")
        assert re[1]['time'] < UTCDateTime("2010-05-27T16:27:33")
        assert 4.2 < re[1]['duration'] < 4.4
        assert re[1]['stations'] == ['UH3', 'UH1', 'UH4']
        assert re[1]['coincidence_sum'] == 3

        # 3. weighting, station selection
        # => 3 events, no false triggers
        trace_ids = {'BW.UH1..SHZ': 0.4, 'BW.UH2..SHZ': 0.35,
                     'BW.UH3..SHZ': 0.4, 'BW.UH4..EHZ': 0.25}
        res = coincidence_trigger("recstalta", 3.5, 1, st.copy(), 1.0,
                                  trace_ids=trace_ids, sta=0.5, lta=10)
        assert len(res) == 3
        assert res[0]['time'] > UTCDateTime("2010-05-27T16:24:31")
        assert res[0]['time'] < UTCDateTime("2010-05-27T16:24:35")
        assert 4.2 < res[0]['duration'] < 4.8
        assert res[0]['stations'] == ['UH3', 'UH2', 'UH1', 'UH4']
        assert res[0]['coincidence_sum'] == 1.4
        assert res[1]['time'] > UTCDateTime("2010-05-27T16:26:59")
        assert res[1]['time'] < UTCDateTime("2010-05-27T16:27:03")
        assert 3.2 < res[1]['duration'] < 3.7
        assert res[1]['stations'] == ['UH2', 'UH3', 'UH1']
        assert res[1]['coincidence_sum'] == 1.15
        assert res[2]['time'] > UTCDateTime("2010-05-27T16:27:27")
        assert res[2]['time'] < UTCDateTime("2010-05-27T16:27:33")
        assert 4.2 < res[2]['duration'] < 4.4
        assert res[2]['stations'] == ['UH3', 'UH2', 'UH1', 'UH4']
        assert res[2]['coincidence_sum'] == 1.4

        # 4. weighting, station selection, max_len
        # => 2 events, no false triggers, small event does not overlap anymore
        trace_ids = {'BW.UH1..SHZ': 0.6, 'BW.UH2..SHZ': 0.6}
        # raises "UserWarning: At least one trace's ID was not found"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', UserWarning)
            re = coincidence_trigger("recstalta", 3.5, 1, st.copy(), 1.2,
                                     trace_ids=trace_ids,
                                     max_trigger_length=0.13, sta=0.5, lta=10)
            assert len(w) == 2
            assert "At least one trace's ID was not" in str(w[0])
            assert "At least one trace's ID was not" in str(w[1])
        assert len(re) == 2
        assert re[0]['time'] > UTCDateTime("2010-05-27T16:24:31")
        assert re[0]['time'] < UTCDateTime("2010-05-27T16:24:35")
        assert 0.2 < re[0]['duration'] < 0.3
        assert re[0]['stations'] == ['UH2', 'UH1']
        assert re[0]['coincidence_sum'] == 1.2
        assert re[1]['time'] > UTCDateTime("2010-05-27T16:27:27")
        assert re[1]['time'] < UTCDateTime("2010-05-27T16:27:33")
        assert 0.18 < re[1]['duration'] < 0.2
        assert re[1]['stations'] == ['UH2', 'UH1']
        assert re[1]['coincidence_sum'] == 1.2

        # 5. station selection, extremely sensitive settings
        # => 4 events, 1 false triggers
        # raises "UserWarning: At least one trace's ID was not found"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', UserWarning)
            res = coincidence_trigger("recstalta", 2.5, 1, st.copy(), 2,
                                      trace_ids=['BW.UH1..SHZ', 'BW.UH3..SHZ'],
                                      sta=0.3, lta=5)
            assert len(w) == 2
            assert "At least one trace's ID was not" in str(w[0])
            assert "At least one trace's ID was not" in str(w[1])
        assert len(res) == 5
        assert res[3]['time'] > UTCDateTime("2010-05-27T16:27:01")
        assert res[3]['time'] < UTCDateTime("2010-05-27T16:27:02")
        assert 1.5 < res[3]['duration'] < 1.7
        assert res[3]['stations'] == ['UH3', 'UH1']
        assert res[3]['coincidence_sum'] == 2.0

        # 6. same as 5, gappy stream
        # => same as 5 (almost, duration of 1 event changes by 0.02s)
        st2 = st.copy()
        tr1 = st2.pop(0)
        t1 = tr1.stats.starttime
        t2 = tr1.stats.endtime
        td = t2 - t1
        tr1a = tr1.slice(starttime=t1, endtime=t1 + 0.45 * td)
        tr1b = tr1.slice(starttime=t1 + 0.6 * td, endtime=t1 + 0.94 * td)
        st2.insert(1, tr1a)
        st2.insert(3, tr1b)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', UserWarning)
            res = coincidence_trigger("recstalta", 2.5, 1, st2, 2,
                                      trace_ids=['BW.UH1..SHZ', 'BW.UH3..SHZ'],
                                      sta=0.3, lta=5)
            assert len(w) == 2
            assert "At least one trace's ID was not" in str(w[0])
            assert "At least one trace's ID was not" in str(w[1])
        assert len(res) == 5
        assert res[3]['time'] > UTCDateTime("2010-05-27T16:27:01")
        assert res[3]['time'] < UTCDateTime("2010-05-27T16:27:02")
        assert 1.5 < res[3]['duration'] < 1.7
        assert res[3]['stations'] == ['UH3', 'UH1']
        assert res[3]['coincidence_sum'] == 2.0

        # 7. same as 3 but modify input trace ids and check output of trace_ids
        # and other additional information with ``details=True``
        st2 = st.copy()
        st2[0].stats.network = "XX"
        st2[1].stats.location = "99"
        st2[1].stats.network = ""
        st2[1].stats.location = "99"
        st2[1].stats.channel = ""
        st2[2].stats.channel = "EHN"
        st2[3].stats.network = ""
        st2[3].stats.channel = ""
        st2[3].stats.station = ""
        trace_ids = {'XX.UH1..SHZ': 0.4, '.UH2.99.': 0.35,
                     'BW.UH3..EHN': 0.4, '...': 0.25}
        res = coincidence_trigger("recstalta", 3.5, 1, st2, 1.0,
                                  trace_ids=trace_ids, details=True,
                                  sta=0.5, lta=10)
        assert len(res) == 3
        assert res[0]['time'] > UTCDateTime("2010-05-27T16:24:31")
        assert res[0]['time'] < UTCDateTime("2010-05-27T16:24:35")
        assert 4.2 < res[0]['duration'] < 4.8
        assert res[0]['stations'] == ['UH3', 'UH2', 'UH1', '']
        assert res[0]['trace_ids'][0] == st2[2].id
        assert res[0]['trace_ids'][1] == st2[1].id
        assert res[0]['trace_ids'][2] == st2[0].id
        assert res[0]['trace_ids'][3] == st2[3].id
        assert res[0]['coincidence_sum'] == 1.4
        assert res[1]['time'] > UTCDateTime("2010-05-27T16:26:59")
        assert res[1]['time'] < UTCDateTime("2010-05-27T16:27:03")
        assert 3.2 < res[1]['duration'] < 3.7
        assert res[1]['stations'] == ['UH2', 'UH3', 'UH1']
        assert res[1]['trace_ids'][0] == st2[1].id
        assert res[1]['trace_ids'][1] == st2[2].id
        assert res[1]['trace_ids'][2] == st2[0].id
        assert res[1]['coincidence_sum'] == 1.15
        assert res[2]['time'] > UTCDateTime("2010-05-27T16:27:27")
        assert res[2]['time'] < UTCDateTime("2010-05-27T16:27:33")
        assert 4.2 < res[2]['duration'] < 4.4
        assert res[2]['stations'] == ['UH3', 'UH2', 'UH1', '']
        assert res[2]['trace_ids'][0] == st2[2].id
        assert res[2]['trace_ids'][1] == st2[1].id
        assert res[2]['trace_ids'][2] == st2[0].id
        assert res[2]['trace_ids'][3] == st2[3].id
        assert res[2]['coincidence_sum'] == 1.4
        expected_keys = ['cft_peak_wmean', 'cft_std_wmean', 'cft_peaks',
                         'cft_stds']
        expected_types = [float, float, list, list]
        for item in res:
            for key, _type in zip(expected_keys, expected_types):
                assert key in item
                assert isinstance(item[key], _type)
        # check some of the detailed info
        ev = res[-1]
        assert round(abs(ev['cft_peak_wmean']-18.101139518271076), 5) == 0
        assert round(abs(ev['cft_std_wmean']-4.800051726246676), 5) == 0
        assert round(abs(ev['cft_peaks'][0]-18.985548683223936), 5) == 0
        assert round(abs(ev['cft_peaks'][1]-16.852175794415011), 5) == 0
        assert round(abs(ev['cft_peaks'][2]-18.64005853900883), 5) == 0
        assert round(abs(ev['cft_peaks'][3]-17.572363634564621), 5) == 0
        assert round(abs(ev['cft_stds'][0]-4.8909448258821362), 5) == 0
        assert round(abs(ev['cft_stds'][1]-4.4446373508521804), 5) == 0
        assert round(abs(ev['cft_stds'][2]-5.3499401252675964), 5) == 0
        assert round(abs(ev['cft_stds'][3]-4.2723814539487703), 5) == 0

    def test_coincidence_trigger_with_similarity_checking(self, testdata):
        """
        Test network coincidence trigger with cross correlation similarity
        checking of given event templates.
        """
        st = Stream()
        files = ["BW.UH1._.SHZ.D.2010.147.cut.slist.gz",
                 "BW.UH2._.SHZ.D.2010.147.cut.slist.gz",
                 "BW.UH3._.SHZ.D.2010.147.cut.slist.gz",
                 "BW.UH3._.SHN.D.2010.147.cut.slist.gz",
                 "BW.UH3._.SHE.D.2010.147.cut.slist.gz",
                 "BW.UH4._.EHZ.D.2010.147.cut.slist.gz"]
        for filename in files:
            st += read(testdata[filename])
        # some prefiltering used for UH network
        st.filter('bandpass', freqmin=10, freqmax=20)
        # set up template event streams
        times = ["2010-05-27T16:24:33.095000", "2010-05-27T16:27:30.370000"]
        templ = {}
        for t in times:
            t = UTCDateTime(t)
            st_ = st.select(station="UH3").slice(t, t + 2.5).copy()
            templ.setdefault("UH3", []).append(st_)
        times = ["2010-05-27T16:27:30.574999"]
        for t in times:
            t = UTCDateTime(t)
            st_ = st.select(station="UH1").slice(t, t + 2.5).copy()
            templ.setdefault("UH1", []).append(st_)
        # add another template with different SEED ID, it should be ignored
        # (this can happen when using many templates over a long time period
        # and instrument changes over time)
        st_ = st_.copy()
        for tr in st_:
            tr.stats.channel = 'X' + tr.stats.channel[1:]
        templ['UH1'].insert(0, st_)
        trace_ids = {"BW.UH1..SHZ": 1,
                     "BW.UH2..SHZ": 1,
                     "BW.UH3..SHZ": 1,
                     "BW.UH4..EHZ": 1}
        similarity_thresholds = {"UH1": 0.8, "UH3": 0.7}
        with warnings.catch_warnings(record=True) as w:
            # avoid getting influenced by the warning filters getting set up
            # differently in obspy-runtests.
            # (e.g. depending on options "-v" and "-q")
            warnings.resetwarnings()
            trig = coincidence_trigger(
                "classicstalta", 5, 1, st.copy(), 4, sta=0.5, lta=10,
                trace_ids=trace_ids, event_templates=templ,
                similarity_threshold=similarity_thresholds)
        # four warnings get raised
        assert len(w) == 4
        assert str(w[0].message) == \
            "At least one trace's ID was not found in the trace ID list and " \
            "was disregarded (BW.UH3..SHN)"
        assert str(w[1].message) == \
            "At least one trace's ID was not found in the trace ID list and " \
            "was disregarded (BW.UH3..SHE)"
        assert str(w[2].message) == \
            ('Skipping trace BW.UH1..XHZ in template correlation (not present '
             'in stream to check).')
        assert str(w[3].message) == \
            ("Skipping template(s) for station 'UH1': No common SEED IDs when "
             "comparing template (BW.UH1..XHZ) and data streams (BW.UH1..SHZ, "
             "BW.UH2..SHZ, BW.UH3..SHE, BW.UH3..SHN, BW.UH3..SHZ, "
             "BW.UH4..EHZ).")
        # check floats in resulting dictionary separately
        assert round(abs(trig[0].pop('duration')-3.96), 6) == 0
        assert round(abs(trig[1].pop('duration')-1.99), 6) == 0
        assert round(abs(trig[2].pop('duration')-1.92), 6) == 0
        assert round(abs(trig[3].pop('duration')-3.92), 6) == 0
        assert round(abs(trig[0]['similarity'].pop('UH1')-0.94149447384), 6) \
            == 0
        assert round(abs(trig[0]['similarity'].pop('UH3')-1), 6) == 0
        assert round(abs(trig[1]['similarity'].pop('UH1')-0.65228204570), 6) \
            == 0
        assert round(abs(trig[1]['similarity'].pop('UH3')-0.72679293429), 6) \
            == 0
        assert round(abs(trig[2]['similarity'].pop('UH1')-0.89404458774), 6) \
            == 0
        assert round(abs(trig[2]['similarity'].pop('UH3')-0.74581409371), 6) \
            == 0
        assert round(abs(trig[3]['similarity'].pop('UH1')-1), 6) == 0
        assert round(abs(trig[3]['similarity'].pop('UH3')-1), 6) == 0
        remaining_results = \
            [{'coincidence_sum': 4.0,
              'similarity': {},
              'stations': ['UH3', 'UH2', 'UH1', 'UH4'],
              'time': UTCDateTime(2010, 5, 27, 16, 24, 33, 210000),
              'trace_ids': ['BW.UH3..SHZ', 'BW.UH2..SHZ', 'BW.UH1..SHZ',
                            'BW.UH4..EHZ']},
             {'coincidence_sum': 3.0,
              'similarity': {},
              'stations': ['UH3', 'UH1', 'UH2'],
              'time': UTCDateTime(2010, 5, 27, 16, 25, 26, 710000),
              'trace_ids': ['BW.UH3..SHZ', 'BW.UH1..SHZ', 'BW.UH2..SHZ']},
             {'coincidence_sum': 3.0,
              'similarity': {},
              'stations': ['UH2', 'UH1', 'UH3'],
              'time': UTCDateTime(2010, 5, 27, 16, 27, 2, 260000),
              'trace_ids': ['BW.UH2..SHZ', 'BW.UH1..SHZ', 'BW.UH3..SHZ']},
             {'coincidence_sum': 4.0,
              'similarity': {},
              'stations': ['UH3', 'UH2', 'UH1', 'UH4'],
              'time': UTCDateTime(2010, 5, 27, 16, 27, 30, 510000),
              'trace_ids': ['BW.UH3..SHZ', 'BW.UH2..SHZ', 'BW.UH1..SHZ',
                            'BW.UH4..EHZ']}]
        assert trig == remaining_results

    def test_classic_sta_lta_c_python(self):
        """
        Test case for ctypes version of recursive_sta_lta
        """
        nsta, nlta = 5, 10
        c1 = classic_sta_lta(self.data, nsta, nlta)
        c2 = classic_sta_lta_py(self.data, nsta, nlta)
        assert np.allclose(c1, c2, rtol=1e-10)
        ref = np.array([0.38012302, 0.37704431, 0.47674533, 0.67992292])
        assert np.allclose(ref, c2[99:103])


class TestEnergyRatio():
    # parameterize ranges are based on chosen value of "a" with length 10
    @pytest.fixture(autouse=True, scope="function")
    def setup(self):
        self.a = np.empty(10)

    @pytest.mark.parametrize('nsta', (1, 2, 3, 4))
    def test_all_zero(self, nsta):
        self.a.fill(0)
        er = energy_ratio(self.a, nsta=nsta)
        assert_array_equal(er, 0)

    def test_arange(self):
        self.a = np.arange(10)
        er = energy_ratio(self.a, nsta=3)
        # Taken as the function output to keep track of regression bugs
        er_expected = [0., 0., 0., 10., 5.5, 3.793103, 2.98, 2.519481, 0., 0.]
        assert_array_almost_equal(er, er_expected)

    @pytest.mark.parametrize('nsta', (1, 2, 3, 4, 5))
    def test_all_ones(self, nsta):
        self.a.fill(1)
        # Forward and backward entries are symmetric -> expecting output '1'
        # Fill nsta on both sides with zero to return same length
        er = energy_ratio(self.a, nsta=nsta)
        er_exp = np.zeros_like(self.a)
        er_exp[nsta: len(self.a) - nsta + 1] = 1
        assert_array_equal(er, er_exp)

    @pytest.mark.parametrize('nsta', (6, 10, 20))
    def test_nsta_too_large(self, nsta):
        expected_msg = re.escape(
            f'nsta ({nsta}) must not be larger than half the length of '
            f'the data (10 samples).')
        with pytest.raises(ValueError, match=expected_msg):
            energy_ratio(self.a, nsta)

    @pytest.mark.parametrize('nsta', (0, -1, -10))
    def test_nsta_zero_or_less(self, nsta):
        expected_msg = re.escape(
            f'nsta ({nsta}) must not be equal to or less than zero.')
        with pytest.raises(ValueError, match=expected_msg):
            energy_ratio(self.a, nsta)


class TestModifiedEnergyRatio():
    # parameterize ranges are based on chosen value of "a" with length 10
    @pytest.fixture(autouse=True, scope="function")
    def setup(self):
        self.a = np.empty(10)

    @pytest.mark.parametrize('nsta', (1, 2, 3, 4))
    def test_all_zero(self, nsta):
        self.a.fill(0)
        er = modified_energy_ratio(self.a, nsta=nsta)
        assert_array_equal(er, 0)

    def test_arange(self):
        self.a = np.arange(10)
        er = modified_energy_ratio(self.a, nsta=3)
        # Taken as the function output to keep track of regression bugs
        er_expected = [0., 0., 0., 27000., 10648., 6821.722908, 5716.135872,
                       5485.637866, 0., 0.]
        assert_array_almost_equal(er, er_expected)

    @pytest.mark.parametrize('nsta', (1, 2, 3, 4, 5))
    def test_all_ones(self, nsta):
        self.a.fill(1)
        # Forward and backward entries are symmetric -> expecting output '1'
        # Fill nsta on both sides with zero to return same length
        er = modified_energy_ratio(self.a, nsta=nsta)
        er_exp = np.zeros_like(self.a)
        er_exp[nsta: len(self.a) - nsta + 1] = 1
        assert_array_equal(er, er_exp)

    @pytest.mark.parametrize('nsta', (6, 10, 20))
    def test_nsta_too_large(self, nsta):
        expected_msg = re.escape(
            f'nsta ({nsta}) must not be larger than half the length of '
            f'the data (10 samples).')
        with pytest.raises(ValueError, match=expected_msg):
            energy_ratio(self.a, nsta)

    @pytest.mark.parametrize('nsta', (0, -1, -10))
    def test_nsta_zero_or_less(self, nsta):
        expected_msg = re.escape(
            f'nsta ({nsta}) must not be equal to or less than zero.')
        with pytest.raises(ValueError, match=expected_msg):
            energy_ratio(self.a, nsta)
