# -*- coding: utf-8 -*-
import unittest
from copy import deepcopy

import numpy as np

from obspy import Trace, UTCDateTime, read
from obspy.signal.filter import (bandpass, bandstop, highpass, lowpass,
                                 lowpass_cheby_2)
from obspy.signal.invsim import simulate_seismometer


class TraceTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.trace.Trace.
    """
    def test_simulate(self):
        """
        Tests if calling simulate of trace gives the same result as using
        simulate_seismometer manually.
        """
        tr = read()[0]
        paz_sts2 = {'poles': [-0.037004 + 0.037016j, -0.037004 - 0.037016j,
                              - 251.33 + 0j, -131.04 - 467.29j,
                              - 131.04 + 467.29j],
                    'zeros': [0j, 0j],
                    'gain': 60077000.0,
                    'sensitivity': 2516778400.0}
        paz_le3d1s = {'poles': [-4.440 + 4.440j, -4.440 - 4.440j,
                                - 1.083 + 0.0j],
                      'zeros': [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                      'gain': 0.4,
                      'sensitivity': 1.0}
        data = simulate_seismometer(
            tr.data, tr.stats.sampling_rate, paz_remove=paz_sts2,
            paz_simulate=paz_le3d1s, remove_sensitivity=True,
            simulate_sensitivity=True)
        tr.simulate(paz_remove=paz_sts2, paz_simulate=paz_le3d1s)
        # There is some strange issue on Win32bit (see #2188). Thus we just
        # use assert_allclose() here instead of testing for full equality.
        np.testing.assert_allclose(tr.data, data)

    def test_filter(self):
        """
        Tests the filter method of the Trace object.

        Basically three scenarios are tested (with differing filter options):
        - filtering with in_place=False:
            - is original trace unchanged?
            - is data of filtered trace the same as if done by hand
            - is processing information present in filtered trace
        - filtering with in_place=True:
            - is data of filtered trace the same as if done by hand
            - is processing information present in filtered trace
        - filtering with bad arguments passed to trace.filter():
            - is a TypeError properly raised?
            - after all bad filter calls, is the trace still unchanged?
        """
        # create two test Traces
        traces = []
        np.random.seed(815)
        header = {'network': 'BW', 'station': 'BGLD',
                  'starttime': UTCDateTime(2007, 12, 31, 23, 59, 59, 915000),
                  'npts': 412, 'sampling_rate': 200.0,
                  'channel': 'EHE'}
        traces.append(Trace(data=np.random.randint(0, 1000, 412),
                            header=deepcopy(header)))
        header['starttime'] = UTCDateTime(2008, 1, 1, 0, 0, 4, 35000)
        header['npts'] = 824
        traces.append(Trace(data=np.random.randint(0, 1000, 824),
                            header=deepcopy(header)))
        traces_bkp = deepcopy(traces)
        # different sets of filters to run test on:
        filters = [['bandpass', {'freqmin': 1., 'freqmax': 20.}],
                   ['bandstop', {'freqmin': 5, 'freqmax': 15., 'corners': 6}],
                   ['lowpass', {'freq': 30.5, 'zerophase': True}],
                   ['highpass', {'freq': 2, 'corners': 2}]]
        filter_map = {'bandpass': bandpass, 'bandstop': bandstop,
                      'lowpass': lowpass, 'highpass': highpass}

        # tests for in_place=True
        for i, tr in enumerate(traces):
            for filt_type, filt_ops in filters:
                tr = deepcopy(traces_bkp[i])
                tr.filter(filt_type, **filt_ops)
                # test if trace was filtered as expected
                data_filt = filter_map[filt_type](
                    traces_bkp[i].data,
                    df=traces_bkp[i].stats.sampling_rate, **filt_ops)
                np.testing.assert_array_equal(tr.data, data_filt)
                self.assertIn('processing', tr.stats)
                self.assertEqual(len(tr.stats.processing), 1)
                self.assertIn("filter", tr.stats.processing[0])
                self.assertIn(filt_type, tr.stats.processing[0])
                for key, value in filt_ops.items():
                    self.assertTrue("'%s': %s" % (key, value)
                                    in tr.stats.processing[0])
                # another filter run
                tr.filter(filt_type, **filt_ops)
                data_filt = filter_map[filt_type](
                    data_filt,
                    df=traces_bkp[i].stats.sampling_rate, **filt_ops)
                np.testing.assert_array_equal(tr.data, data_filt)
                self.assertIn('processing', tr.stats)
                self.assertEqual(len(tr.stats.processing), 2)
                for proc_info in tr.stats.processing:
                    self.assertIn("filter", proc_info)
                    self.assertIn(filt_type, proc_info)
                    for key, value in filt_ops.items():
                        self.assertTrue("'%s': %s" % (key, value)
                                        in proc_info)

        # some tests that should raise an Exception
        tr = traces[0]
        bad_filters = [
            ['bandpass', {'freqmin': 1., 'XXX': 20.}],
            ['bandstop', {'freqmin': 5, 'freqmax': "XXX", 'corners': 6}],
            ['bandstop', {}],
            ['bandstop', [1, 2, 3, 4, 5]],
            ['bandstop', None],
            ['bandstop', 3],
            ['bandstop', 'XXX'],
            ['bandpass', {'freqmin': 5, 'corners': 6}],
            ['bandpass', {'freqmin': 5, 'freqmax': 20., 'df': 100.}]]
        for filt_type, filt_ops in bad_filters:
            self.assertRaises(TypeError, tr.filter, filt_type, filt_ops)
        bad_filters = [['XXX', {'freqmin': 5, 'freqmax': 20., 'corners': 6}]]
        for filt_type, filt_ops in bad_filters:
            self.assertRaises(ValueError, tr.filter, filt_type, **filt_ops)
        # test if trace is unchanged after all these bad tests
        np.testing.assert_array_equal(tr.data, traces_bkp[0].data)
        self.assertEqual(tr.stats, traces_bkp[0].stats)

    def test_decimate(self):
        """
        Tests the decimate method of the Trace object.
        """
        # create test Trace
        tr = Trace(data=np.arange(20))
        tr_bkp = deepcopy(tr)
        # some test that should fail and leave the original trace alone
        self.assertRaises(ValueError, tr.decimate, 7, strict_length=True)
        self.assertRaises(ValueError, tr.decimate, 9, strict_length=True)
        self.assertRaises(ArithmeticError, tr.decimate, 18)
        # some tests in place
        tr.decimate(4, no_filter=True)
        np.testing.assert_array_equal(tr.data, np.arange(0, 20, 4))
        self.assertEqual(tr.stats.npts, 5)
        self.assertEqual(tr.stats.sampling_rate, 0.25)
        self.assertIn("decimate", tr.stats.processing[0])
        self.assertIn("factor=4", tr.stats.processing[0])
        tr = tr_bkp.copy()
        tr.decimate(10, no_filter=True)
        np.testing.assert_array_equal(tr.data, np.arange(0, 20, 10))
        self.assertEqual(tr.stats.npts, 2)
        self.assertEqual(tr.stats.sampling_rate, 0.1)
        self.assertIn("decimate", tr.stats.processing[0])
        self.assertIn("factor=10", tr.stats.processing[0])
        # some tests with automatic prefiltering
        tr = tr_bkp.copy()
        tr2 = tr_bkp.copy()
        tr.decimate(4)
        df = tr2.stats.sampling_rate
        tr2.data, fp = lowpass_cheby_2(data=tr2.data, freq=df * 0.5 / 4.0,
                                       df=df, maxorder=12, ba=False,
                                       freq_passband=True)
        # check that iteratively determined pass band frequency is correct
        self.assertAlmostEqual(0.0811378285461, fp, places=7)
        tr2.decimate(4, no_filter=True)
        np.testing.assert_array_equal(tr.data, tr2.data)
