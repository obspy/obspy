# -*- coding: utf-8 -*-
"""
The obspy.signal.trigger test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import gzip
import os
import unittest
import warnings
from ctypes import ArgumentError

import numpy as np

from obspy import Stream, UTCDateTime, read
from obspy.signal.trigger import (
    ar_pick, classic_STALTA, classic_STALTA_py, coincidence_trigger, pk_baer,
    recursive_STALTA, recursive_STALTA_py, trigger_onset)
from obspy.signal.util import clibsignal


class TriggerTestCase(unittest.TestCase):
    """
    Test cases for obspy.trigger
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        # random seed data
        np.random.seed(815)
        self.data = np.random.randn(int(1e5))

    def test_recSTALTAC(self):
        """
        Test case for ctypes version of recursive_STALTA
        """
        nsta, nlta = 5, 10
        c1 = recursive_STALTA(self.data, nsta, nlta)
        self.assertAlmostEqual(c1[99], 0.80810165)
        self.assertAlmostEqual(c1[100], 0.75939449)
        self.assertAlmostEqual(c1[101], 0.91763978)
        self.assertAlmostEqual(c1[102], 0.97465004)

    def test_recSTALTAPy(self):
        """
        Test case for python version of recursive_STALTA
        """
        nsta, nlta = 5, 10
        c2 = recursive_STALTA_py(self.data, nsta, nlta)
        self.assertAlmostEqual(c2[99], 0.80810165)
        self.assertAlmostEqual(c2[100], 0.75939449)
        self.assertAlmostEqual(c2[101], 0.91763978)
        self.assertAlmostEqual(c2[102], 0.97465004)

    def test_recSTALTARaise(self):
        """
        Type checking recursive_STALTA
        """
        ndat = 1
        charfct = np.empty(ndat, dtype=np.float64)
        self.assertRaises(ArgumentError, clibsignal.recstalta, [1], charfct,
                          ndat, 5, 10)
        self.assertRaises(ArgumentError, clibsignal.recstalta,
                          np.array([1], dtype=np.int32), charfct, ndat, 5, 10)

    def test_pkBaer(self):
        """
        Test pk_baer against implementation for UNESCO short course
        """
        filename = os.path.join(self.path, 'manz_waldk.a01.gz')
        with gzip.open(filename) as f:
            data = np.loadtxt(f, dtype=np.float32)
        df, ntdownmax, ntupevent, thr1, thr2, npreset_len, np_dur = \
            (200.0, 20, 60, 7.0, 12.0, 100, 100)
        nptime, pfm = pk_baer(data, df, ntdownmax, ntupevent,
                              thr1, thr2, npreset_len, np_dur)
        self.assertEqual(nptime, 17545)
        self.assertEqual(pfm, 'IPU0')

    def test_arPick(self):
        """
        Test ar_pick against implementation for UNESCO short course
        """
        data = []
        for channel in ['z', 'n', 'e']:
            file = os.path.join(self.path,
                                'loc_RJOB20050801145719850.' + channel)
            data.append(np.loadtxt(file, dtype=np.float32))
        # some default arguments
        samp_rate, f1, f2, lta_p, sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s = \
            200.0, 1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2
        ptime, stime = ar_pick(data[0], data[1], data[2], samp_rate, f1, f2,
                               lta_p, sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s)
        self.assertAlmostEqual(ptime, 30.6350002289)
        # seems to be strongly machine dependent, go for int for 64 bit
        # self.assertAlmostEqual(stime, 31.2800006866)
        self.assertEqual(int(stime + 0.5), 31)

    def test_triggerOnset(self):
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

    def test_coincidenceTrigger(self):
        """
        Test network coincidence trigger.
        """
        st = Stream()
        files = ["BW.UH1._.SHZ.D.2010.147.cut.slist.gz",
                 "BW.UH2._.SHZ.D.2010.147.cut.slist.gz",
                 "BW.UH3._.SHZ.D.2010.147.cut.slist.gz",
                 "BW.UH4._.EHZ.D.2010.147.cut.slist.gz"]
        for filename in files:
            filename = os.path.join(self.path, filename)
            st += read(filename)
        # some prefiltering used for UH network
        st.filter('bandpass', freqmin=10, freqmax=20)
        # 1. no weighting, no stations specified, good settings
        # => 3 events, no false triggers
        # for the first test we make some additional tests regarding types
        res = coincidence_trigger("recstalta", 3.5, 1, st.copy(), 3, sta=0.5,
                                  lta=10)
        self.assertTrue(isinstance(res, list))
        self.assertEqual(len(res), 3)
        expected_keys = ['time', 'coincidence_sum', 'duration', 'stations',
                         'trace_ids']
        expected_types = [UTCDateTime, float, float, list, list]
        for item in res:
            self.assertTrue(isinstance(item, dict))
            for key, _type in zip(expected_keys, expected_types):
                self.assertIn(key, item)
                self.assertTrue(isinstance(item[key], _type))
        self.assertGreater(res[0]['time'], UTCDateTime("2010-05-27T16:24:31"))
        self.assertTrue(res[0]['time'] < UTCDateTime("2010-05-27T16:24:35"))
        self.assertTrue(4.2 < res[0]['duration'] < 4.8)
        self.assertEqual(res[0]['stations'], ['UH3', 'UH2', 'UH1', 'UH4'])
        self.assertEqual(res[0]['coincidence_sum'], 4)
        self.assertGreater(res[1]['time'], UTCDateTime("2010-05-27T16:26:59"))
        self.assertTrue(res[1]['time'] < UTCDateTime("2010-05-27T16:27:03"))
        self.assertTrue(3.2 < res[1]['duration'] < 3.7)
        self.assertEqual(res[1]['stations'], ['UH2', 'UH3', 'UH1'])
        self.assertEqual(res[1]['coincidence_sum'], 3)
        self.assertGreater(res[2]['time'], UTCDateTime("2010-05-27T16:27:27"))
        self.assertTrue(res[2]['time'] < UTCDateTime("2010-05-27T16:27:33"))
        self.assertTrue(4.2 < res[2]['duration'] < 4.4)
        self.assertEqual(res[2]['stations'], ['UH3', 'UH2', 'UH1', 'UH4'])
        self.assertEqual(res[2]['coincidence_sum'], 4)
        # 2. no weighting, station selection
        # => 2 events, no false triggers
        trace_ids = ['BW.UH1..SHZ', 'BW.UH3..SHZ', 'BW.UH4..EHZ']
        # ignore UserWarnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore', UserWarning)
            re = coincidence_trigger("recstalta", 3.5, 1, st.copy(), 3,
                                     trace_ids=trace_ids, sta=0.5, lta=10)
            self.assertEqual(len(re), 2)
            self.assertGreater(re[0]['time'],
                               UTCDateTime("2010-05-27T16:24:31"))
            self.assertTrue(re[0]['time'] < UTCDateTime("2010-05-27T16:24:35"))
            self.assertTrue(4.2 < re[0]['duration'] < 4.8)
            self.assertEqual(re[0]['stations'], ['UH3', 'UH1', 'UH4'])
            self.assertEqual(re[0]['coincidence_sum'], 3)
            self.assertGreater(re[1]['time'],
                               UTCDateTime("2010-05-27T16:27:27"))
            self.assertTrue(re[1]['time'] < UTCDateTime("2010-05-27T16:27:33"))
            self.assertTrue(4.2 < re[1]['duration'] < 4.4)
            self.assertEqual(re[1]['stations'], ['UH3', 'UH1', 'UH4'])
            self.assertEqual(re[1]['coincidence_sum'], 3)
        # 3. weighting, station selection
        # => 3 events, no false triggers
        trace_ids = {'BW.UH1..SHZ': 0.4, 'BW.UH2..SHZ': 0.35,
                     'BW.UH3..SHZ': 0.4, 'BW.UH4..EHZ': 0.25}
        res = coincidence_trigger("recstalta", 3.5, 1, st.copy(), 1.0,
                                  trace_ids=trace_ids, sta=0.5, lta=10)
        self.assertEqual(len(res), 3)
        self.assertGreater(res[0]['time'], UTCDateTime("2010-05-27T16:24:31"))
        self.assertTrue(res[0]['time'] < UTCDateTime("2010-05-27T16:24:35"))
        self.assertTrue(4.2 < res[0]['duration'] < 4.8)
        self.assertEqual(res[0]['stations'], ['UH3', 'UH2', 'UH1', 'UH4'])
        self.assertEqual(res[0]['coincidence_sum'], 1.4)
        self.assertGreater(res[1]['time'], UTCDateTime("2010-05-27T16:26:59"))
        self.assertTrue(res[1]['time'] < UTCDateTime("2010-05-27T16:27:03"))
        self.assertTrue(3.2 < res[1]['duration'] < 3.7)
        self.assertEqual(res[1]['stations'], ['UH2', 'UH3', 'UH1'])
        self.assertEqual(res[1]['coincidence_sum'], 1.15)
        self.assertGreater(res[2]['time'], UTCDateTime("2010-05-27T16:27:27"))
        self.assertTrue(res[2]['time'] < UTCDateTime("2010-05-27T16:27:33"))
        self.assertTrue(4.2 < res[2]['duration'] < 4.4)
        self.assertEqual(res[2]['stations'], ['UH3', 'UH2', 'UH1', 'UH4'])
        self.assertEqual(res[2]['coincidence_sum'], 1.4)
        # 4. weighting, station selection, max_len
        # => 2 events, no false triggers, small event does not overlap anymore
        trace_ids = {'BW.UH1..SHZ': 0.6, 'BW.UH2..SHZ': 0.6}
        # ignore UserWarnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore', UserWarning)
            re = coincidence_trigger("recstalta", 3.5, 1, st.copy(), 1.2,
                                     trace_ids=trace_ids,
                                     max_trigger_length=0.13, sta=0.5, lta=10)
            self.assertEqual(len(re), 2)
            self.assertGreater(re[0]['time'],
                               UTCDateTime("2010-05-27T16:24:31"))
            self.assertTrue(re[0]['time'] < UTCDateTime("2010-05-27T16:24:35"))
            self.assertTrue(0.2 < re[0]['duration'] < 0.3)
            self.assertEqual(re[0]['stations'], ['UH2', 'UH1'])
            self.assertEqual(re[0]['coincidence_sum'], 1.2)
            self.assertGreater(re[1]['time'],
                               UTCDateTime("2010-05-27T16:27:27"))
            self.assertTrue(re[1]['time'] < UTCDateTime("2010-05-27T16:27:33"))
            self.assertTrue(0.18 < re[1]['duration'] < 0.2)
            self.assertEqual(re[1]['stations'], ['UH2', 'UH1'])
            self.assertEqual(re[1]['coincidence_sum'], 1.2)
        # 5. station selection, extremely sensitive settings
        # => 4 events, 1 false triggers
        res = coincidence_trigger("recstalta", 2.5, 1, st.copy(), 2,
                                  trace_ids=['BW.UH1..SHZ', 'BW.UH3..SHZ'],
                                  sta=0.3, lta=5)
        self.assertEqual(len(res), 5)
        self.assertGreater(res[3]['time'], UTCDateTime("2010-05-27T16:27:01"))
        self.assertTrue(res[3]['time'] < UTCDateTime("2010-05-27T16:27:02"))
        self.assertTrue(1.5 < res[3]['duration'] < 1.7)
        self.assertEqual(res[3]['stations'], ['UH3', 'UH1'])
        self.assertEqual(res[3]['coincidence_sum'], 2.0)
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
        res = coincidence_trigger("recstalta", 2.5, 1, st2, 2,
                                  trace_ids=['BW.UH1..SHZ', 'BW.UH3..SHZ'],
                                  sta=0.3, lta=5)
        self.assertEqual(len(res), 5)
        self.assertGreater(res[3]['time'], UTCDateTime("2010-05-27T16:27:01"))
        self.assertTrue(res[3]['time'] < UTCDateTime("2010-05-27T16:27:02"))
        self.assertTrue(1.5 < res[3]['duration'] < 1.7)
        self.assertEqual(res[3]['stations'], ['UH3', 'UH1'])
        self.assertEqual(res[3]['coincidence_sum'], 2.0)
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
        self.assertEqual(len(res), 3)
        self.assertGreater(res[0]['time'], UTCDateTime("2010-05-27T16:24:31"))
        self.assertTrue(res[0]['time'] < UTCDateTime("2010-05-27T16:24:35"))
        self.assertTrue(4.2 < res[0]['duration'] < 4.8)
        self.assertEqual(res[0]['stations'], ['UH3', 'UH2', 'UH1', ''])
        self.assertEqual(res[0]['trace_ids'][0], st2[2].id)
        self.assertEqual(res[0]['trace_ids'][1], st2[1].id)
        self.assertEqual(res[0]['trace_ids'][2], st2[0].id)
        self.assertEqual(res[0]['trace_ids'][3], st2[3].id)
        self.assertEqual(res[0]['coincidence_sum'], 1.4)
        self.assertGreater(res[1]['time'], UTCDateTime("2010-05-27T16:26:59"))
        self.assertTrue(res[1]['time'] < UTCDateTime("2010-05-27T16:27:03"))
        self.assertTrue(3.2 < res[1]['duration'] < 3.7)
        self.assertEqual(res[1]['stations'], ['UH2', 'UH3', 'UH1'])
        self.assertEqual(res[1]['trace_ids'][0], st2[1].id)
        self.assertEqual(res[1]['trace_ids'][1], st2[2].id)
        self.assertEqual(res[1]['trace_ids'][2], st2[0].id)
        self.assertEqual(res[1]['coincidence_sum'], 1.15)
        self.assertGreater(res[2]['time'], UTCDateTime("2010-05-27T16:27:27"))
        self.assertTrue(res[2]['time'] < UTCDateTime("2010-05-27T16:27:33"))
        self.assertTrue(4.2 < res[2]['duration'] < 4.4)
        self.assertEqual(res[2]['stations'], ['UH3', 'UH2', 'UH1', ''])
        self.assertEqual(res[2]['trace_ids'][0], st2[2].id)
        self.assertEqual(res[2]['trace_ids'][1], st2[1].id)
        self.assertEqual(res[2]['trace_ids'][2], st2[0].id)
        self.assertEqual(res[2]['trace_ids'][3], st2[3].id)
        self.assertEqual(res[2]['coincidence_sum'], 1.4)
        expected_keys = ['cft_peak_wmean', 'cft_std_wmean', 'cft_peaks',
                         'cft_stds']
        expected_types = [float, float, list, list]
        for item in res:
            for key, _type in zip(expected_keys, expected_types):
                self.assertIn(key, item)
                self.assertTrue(isinstance(item[key], _type))
        # check some of the detailed info
        ev = res[-1]
        self.assertAlmostEqual(ev['cft_peak_wmean'], 18.101139518271076)
        self.assertAlmostEqual(ev['cft_std_wmean'], 4.800051726246676)
        self.assertAlmostEqual(ev['cft_peaks'][0], 18.985548683223936)
        self.assertAlmostEqual(ev['cft_peaks'][1], 16.852175794415011)
        self.assertAlmostEqual(ev['cft_peaks'][2], 18.64005853900883)
        self.assertAlmostEqual(ev['cft_peaks'][3], 17.572363634564621)
        self.assertAlmostEqual(ev['cft_stds'][0], 4.8909448258821362)
        self.assertAlmostEqual(ev['cft_stds'][1], 4.4446373508521804)
        self.assertAlmostEqual(ev['cft_stds'][2], 5.3499401252675964)
        self.assertAlmostEqual(ev['cft_stds'][3], 4.2723814539487703)

    def test_coincidenceTriggerWithSimilarityChecking(self):
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
            filename = os.path.join(self.path, filename)
            st += read(filename)
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
            # two warnings get raised
            self.assertEqual(len(w), 2)
        # check floats in resulting dictionary separately
        self.assertAlmostEqual(trig[0].pop('duration'), 3.9600000381469727)
        self.assertAlmostEqual(trig[1].pop('duration'), 1.9900000095367432)
        self.assertAlmostEqual(trig[2].pop('duration'), 1.9200000762939453)
        self.assertAlmostEqual(trig[3].pop('duration'), 3.9200000762939453)
        self.assertAlmostEqual(trig[0]['similarity'].pop('UH1'), 0.94149447384)
        self.assertAlmostEqual(trig[0]['similarity'].pop('UH3'), 1)
        self.assertAlmostEqual(trig[1]['similarity'].pop('UH1'), 0.65228204570)
        self.assertAlmostEqual(trig[1]['similarity'].pop('UH3'), 0.72679293429)
        self.assertAlmostEqual(trig[2]['similarity'].pop('UH1'), 0.89404458774)
        self.assertAlmostEqual(trig[2]['similarity'].pop('UH3'), 0.74581409371)
        self.assertAlmostEqual(trig[3]['similarity'].pop('UH1'), 1)
        self.assertAlmostEqual(trig[3]['similarity'].pop('UH3'), 1)
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
        self.assertEqual(trig, remaining_results)

    def test_classicSTALTAPyC(self):
        """
        Test case for ctypes version of recursive_STALTA
        """
        nsta, nlta = 5, 10
        c1 = classic_STALTA(self.data, nsta, nlta)
        c2 = classic_STALTA_py(self.data, nsta, nlta)
        self.assertTrue(np.allclose(c1, c2, rtol=1e-10))
        ref = np.array([0.38012302, 0.37704431, 0.47674533, 0.67992292])
        self.assertTrue(np.allclose(ref, c2[99:103]))


def suite():
    return unittest.makeSuite(TriggerTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
