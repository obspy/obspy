# -*- coding: utf-8 -*-
import unittest
from copy import deepcopy
import platform

import numpy as np

from obspy import Stream, Trace, UTCDateTime, read
from obspy.signal.filter import bandpass, bandstop, highpass, lowpass


class StreamTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.stream.Stream.
    """
    def test_filter(self):
        """
        Tests the filter method of the Stream object.

        Basically three scenarios are tested (with differing filter options):
        - filtering with in_place=False:
            - is original stream unchanged?
            - is data of filtered stream's traces the same as if done by hand
            - is processing information present in filtered stream's traces
        - filtering with in_place=True:
            - is data of filtered stream's traces the same as if done by hand
            - is processing information present in filtered stream's traces
        - filtering with bad arguments passed to stream.filter():
            - is a TypeError properly raised?
            - after all bad filter calls, is the stream still unchanged?
        """
        # set specific seed value such that random numbers are reproducible
        np.random.seed(815)
        header = {'network': 'BW', 'station': 'BGLD',
                  'starttime': UTCDateTime(2007, 12, 31, 23, 59, 59, 915000),
                  'npts': 412, 'sampling_rate': 200.0,
                  'channel': 'EHE'}
        trace1 = Trace(data=np.random.randint(0, 1000, 412),
                       header=deepcopy(header))
        header['starttime'] = UTCDateTime(2008, 1, 1, 0, 0, 4, 35000)
        header['npts'] = 824
        trace2 = Trace(data=np.random.randint(0, 1000, 824),
                       header=deepcopy(header))
        header['starttime'] = UTCDateTime(2008, 1, 1, 0, 0, 10, 215000)
        trace3 = Trace(data=np.random.randint(0, 1000, 824),
                       header=deepcopy(header))
        header['starttime'] = UTCDateTime(2008, 1, 1, 0, 0, 18, 455000)
        header['npts'] = 50668
        trace4 = Trace(data=np.random.randint(0, 1000, 50668),
                       header=deepcopy(header))
        mseed_stream = Stream(traces=[trace1, trace2, trace3, trace4])
        header = {'network': '', 'station': 'RNON ', 'location': '',
                  'starttime': UTCDateTime(2004, 6, 9, 20, 5, 59, 849998),
                  'sampling_rate': 200.0, 'npts': 12000,
                  'channel': '  Z'}
        trace = Trace(data=np.random.randint(0, 1000, 12000), header=header)
        gse2_stream = Stream(traces=[trace])
        # streams to run tests on:
        streams = [mseed_stream, gse2_stream]
        # drop the longest trace of the first stream to save a second
        streams[0].pop()
        streams_bkp = deepcopy(streams)
        # different sets of filters to run test on:
        filters = [['bandpass', {'freqmin': 1., 'freqmax': 20.}],
                   ['bandstop', {'freqmin': 5, 'freqmax': 15., 'corners': 6}],
                   ['lowpass', {'freq': 30.5, 'zerophase': True}],
                   ['highpass', {'freq': 2, 'corners': 2}]]
        filter_map = {'bandpass': bandpass, 'bandstop': bandstop,
                      'lowpass': lowpass, 'highpass': highpass}

        # tests for in_place=True
        for j, st in enumerate(streams):
            st_bkp = streams_bkp[j]
            for filt_type, filt_ops in filters:
                st = deepcopy(streams_bkp[j])
                st.filter(filt_type, **filt_ops)
                # test if all traces were filtered as expected
                for i, tr in enumerate(st):
                    data_filt = filter_map[filt_type](
                        st_bkp[i].data,
                        df=st_bkp[i].stats.sampling_rate, **filt_ops)
                    np.testing.assert_array_equal(tr.data, data_filt)
                    self.assertIn('processing', tr.stats)
                    self.assertEqual(len(tr.stats.processing), 1)
                    self.assertIn("filter", tr.stats.processing[0])
                    self.assertIn(filt_type, tr.stats.processing[0])
                    for key, value in filt_ops.items():
                        self.assertTrue("'%s': %s" % (key, value)
                                        in tr.stats.processing[0])
                st.filter(filt_type, **filt_ops)
                for i, tr in enumerate(st):
                    self.assertIn('processing', tr.stats)
                    self.assertEqual(len(tr.stats.processing), 2)
                    for proc_info in tr.stats.processing:
                        self.assertIn("filter", proc_info)
                        self.assertIn(filt_type, proc_info)
                        for key, value in filt_ops.items():
                            self.assertTrue("'%s': %s" % (key, value)
                                            in proc_info)

        # some tests that should raise an Exception
        st = streams[0]
        st_bkp = streams_bkp[0]
        bad_filters = [
            ['bandpass', {'freqmin': 1., 'XXX': 20.}],
            ['bandstop', [1, 2, 3, 4, 5]],
            ['bandstop', None],
            ['bandstop', 3],
            ['bandstop', 'XXX']]
        for filt_type, filt_ops in bad_filters:
            self.assertRaises(TypeError, st.filter, filt_type, filt_ops)
        bad_filters = [
            ['bandpass', {'freqmin': 1., 'XXX': 20.}],
            ['bandstop', {'freqmin': 5, 'freqmax': "XXX", 'corners': 6}],
            ['bandstop', {}],
            ['bandpass', {'freqmin': 5, 'corners': 6}],
            ['bandpass', {'freqmin': 5, 'freqmax': 20., 'df': 100.}]]
        for filt_type, filt_ops in bad_filters:
            self.assertRaises(TypeError, st.filter, filt_type, **filt_ops)
        bad_filters = [['XXX', {'freqmin': 5, 'freqmax': 20., 'corners': 6}]]
        for filt_type, filt_ops in bad_filters:
            self.assertRaises(ValueError, st.filter, filt_type, **filt_ops)
        # test if stream is unchanged after all these bad tests
        for i, tr in enumerate(st):
            np.testing.assert_array_equal(tr.data, st_bkp[i].data)
            self.assertEqual(tr.stats, st_bkp[i].stats)

    def test_simulate(self):
        """
        Tests if calling simulate of stream gives the same result as calling
        simulate on every trace manually.
        """
        st1 = read()
        st2 = read()
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
        st1.simulate(paz_remove=paz_sts2, paz_simulate=paz_le3d1s)
        for tr in st2:
            tr.simulate(paz_remove=paz_sts2, paz_simulate=paz_le3d1s)

        # There is some strange issue on Win32bit (see #2188) and Win64bit (see
        # #2330). Thus we just use assert_allclose() here instead of testing
        # for full equality.
        if platform.system() == "Windows":  # pragma: no cover
            for tr1, tr2 in zip(st1, st2):
                self.assertEqual(tr1.stats, tr2.stats)
                np.testing.assert_allclose(tr1.data, tr2.data, rtol=1E-6,
                                           atol=1E-6 * tr1.data.ptp())
        else:
            # Added (up to ###) to debug appveyor fails
            for tr1, tr2 in zip(st1.sort(), st2.sort()):
                self.assertEqual(tr1.stats, tr2.stats)
                np.testing.assert_allclose(tr1.data, tr2.data)
            ###
            self.assertEqual(st1, st2)

    def test_decimate(self):
        """
        Tests if all traces in the stream object are handled as expected
        by the decimate method on the trace object.
        """
        # create test Stream
        st = read()
        st_bkp = st.copy()
        # test if all traces are decimated as expected
        st.decimate(10, strict_length=False)
        for i, tr in enumerate(st):
            st_bkp[i].decimate(10, strict_length=False)
            self.assertEqual(tr, st_bkp[i])
