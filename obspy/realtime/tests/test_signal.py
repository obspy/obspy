# -*- coding: utf-8 -*-
"""
The obspy.realtime.signal test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import numpy as np

from obspy import read
from obspy.core.stream import Stream
from obspy.realtime import RtTrace, signal


# some debug flags
PLOT_TRACES = False
NUM_PACKETS = 3


class RealTimeSignalTestCase(unittest.TestCase):
    """
    The obspy.realtime.signal test suite.
    """
    def __init__(self, *args, **kwargs):
        super(RealTimeSignalTestCase, self).__init__(*args, **kwargs)
        # read test data as float64
        self.orig_trace = read(os.path.join(os.path.dirname(__file__), 'data',
                                            'II.TLY.BHZ.SAC'),
                               dtype=np.float64)[0]
        # make really sure test data is float64
        self.orig_trace.data = np.require(self.orig_trace.data, np.float64)
        self.orig_trace_chunks = self.orig_trace / NUM_PACKETS

    def setUp(self):
        # clear results
        self.filt_trace_data = None
        self.rt_trace = None
        self.rt_appended_traces = []

    def tearDown(self):
        # use results for debug plots if enabled
        if PLOT_TRACES and self.filt_trace_data is not None and \
           self.rt_trace is not None and self.rt_appended_traces:
            self._plotResults()

    def test_square(self):
        """
        Testing np.square function.
        """
        trace = self.orig_trace.copy()
        # filtering manual
        self.filt_trace_data = np.square(trace)
        # filtering real time
        process_list = [(np.square, {})]
        self._runRtProcess(process_list)
        # check results
        np.testing.assert_almost_equal(self.filt_trace_data,
                                       self.rt_trace.data)

    def test_integrate(self):
        """
        Testing integrate function.
        """
        trace = self.orig_trace.copy()
        # filtering manual
        self.filt_trace_data = signal.integrate(trace)
        # filtering real time
        process_list = [('integrate', {})]
        self._runRtProcess(process_list)
        # check results
        np.testing.assert_almost_equal(self.filt_trace_data,
                                       self.rt_trace.data)

    def test_differentiate(self):
        """
        Testing differentiate function.
        """
        trace = self.orig_trace.copy()
        # filtering manual
        self.filt_trace_data = signal.differentiate(trace)
        # filtering real time
        process_list = [('differentiate', {})]
        self._runRtProcess(process_list)
        # check results
        np.testing.assert_almost_equal(self.filt_trace_data,
                                       self.rt_trace.data)

    def test_boxcar(self):
        """
        Testing boxcar function.
        """
        trace = self.orig_trace.copy()
        options = {'width': 500}
        # filtering manual
        self.filt_trace_data = signal.boxcar(trace, **options)
        # filtering real time
        process_list = [('boxcar', options)]
        self._runRtProcess(process_list)
        # check results
        peak = np.amax(np.abs(self.rt_trace.data))
        self.assertAlmostEqual(peak, 566974.214, 3)
        np.testing.assert_almost_equal(self.filt_trace_data,
                                       self.rt_trace.data)

    def test_scale(self):
        """
        Testing scale function.
        """
        trace = self.orig_trace.copy()
        options = {'factor': 1000}
        # filtering manual
        self.filt_trace_data = signal.scale(trace, **options)
        # filtering real time
        process_list = [('scale', options)]
        self._runRtProcess(process_list)
        # check results
        peak = np.amax(np.abs(self.rt_trace.data))
        self.assertEqual(peak, 1045237000.0)
        np.testing.assert_almost_equal(self.filt_trace_data,
                                       self.rt_trace.data)

    def test_offset(self):
        """
        Testing offset function.
        """
        trace = self.orig_trace.copy()
        options = {'offset': 500}
        # filtering manual
        self.filt_trace_data = signal.offset(trace, **options)
        # filtering real time
        process_list = [('offset', options)]
        self._runRtProcess(process_list)
        # check results
        diff = self.rt_trace.data - self.orig_trace.data
        self.assertEqual(np.mean(diff), 500)
        np.testing.assert_almost_equal(self.filt_trace_data,
                                       self.rt_trace.data)

    def test_kurtosis(self):
        """
        Testing kurtosis function.
        """
        trace = self.orig_trace.copy()
        options = {'win': 5}
        # filtering manual
        self.filt_trace_data = signal.kurtosis(trace, **options)
        # filtering real time
        process_list = [('kurtosis', options)]
        self._runRtProcess(process_list)
        # check results
        np.testing.assert_almost_equal(self.filt_trace_data,
                                       self.rt_trace.data)

    def test_abs(self):
        """
        Testing np.abs function.
        """
        trace = self.orig_trace.copy()
        # filtering manual
        self.filt_trace_data = np.abs(trace)
        # filtering real time
        process_list = [(np.abs, {})]
        self._runRtProcess(process_list)
        # check results
        peak = np.amax(np.abs(self.rt_trace.data))
        self.assertEqual(peak, 1045237)
        np.testing.assert_almost_equal(self.filt_trace_data,
                                       self.rt_trace.data)

    def test_tauc(self):
        """
        Testing tauc function.
        """
        trace = self.orig_trace.copy()
        options = {'width': 60}
        # filtering manual
        self.filt_trace_data = signal.tauc(trace, **options)
        # filtering real time
        process_list = [('tauc', options)]
        self._runRtProcess(process_list)
        # check results
        peak = np.amax(np.abs(self.rt_trace.data))
        self.assertAlmostEqual(peak, 114.302, 3)
        np.testing.assert_almost_equal(self.filt_trace_data,
                                       self.rt_trace.data)

    def test_mwpIntegral(self):
        """
        Testing mwpIntegral functions.
        """
        trace = self.orig_trace.copy()
        options = {'mem_time': 240,
                   'ref_time': trace.stats.starttime + 301.506,
                   'max_time': 120,
                   'gain': 1.610210e+09}
        # filtering manual
        self.filt_trace_data = signal.mwpIntegral(self.orig_trace.copy(),
                                                  **options)
        # filtering real time
        process_list = [('mwpIntegral', options)]
        self._runRtProcess(process_list)
        # check results
        np.testing.assert_almost_equal(self.filt_trace_data,
                                       self.rt_trace.data)

    def test_mwp(self):
        """
        Testing Mwp calculation using two processing functions.
        """
        trace = self.orig_trace.copy()
        epicentral_distance = 30.0855
        options = {'mem_time': 240,
                   'ref_time': trace.stats.starttime + 301.506,
                   'max_time': 120,
                   'gain': 1.610210e+09}
        # filtering manual
        trace.data = signal.integrate(trace)
        self.filt_trace_data = signal.mwpIntegral(trace, **options)
        # filtering real time
        process_list = [('integrate', {}), ('mwpIntegral', options)]
        self._runRtProcess(process_list)
        # check results
        peak = np.amax(np.abs(self.rt_trace.data))
        mwp = signal.calculateMwpMag(peak, epicentral_distance)
        self.assertAlmostEqual(mwp, 8.78902911791, 5)
        np.testing.assert_almost_equal(self.filt_trace_data,
                                       self.rt_trace.data)

    def test_combined(self):
        """
        Testing combining integrate and differentiate functions.
        """
        trace = self.orig_trace.copy()
        # filtering manual
        trace.data = signal.integrate(trace)
        self.filt_trace_data = signal.differentiate(trace)
        # filtering real time
        process_list = [('int', {}), ('diff', {})]
        self._runRtProcess(process_list)
        # check results
        trace = self.orig_trace.copy()
        np.testing.assert_almost_equal(self.filt_trace_data,
                                       self.rt_trace.data)
        np.testing.assert_almost_equal(trace.data[1:], self.rt_trace.data[1:])
        np.testing.assert_almost_equal(trace.data[1:],
                                       self.filt_trace_data[1:])

    def _runRtProcess(self, process_list, max_length=None):
        """
        Helper function to create a RtTrace, register all given process
        functions and run the real time processing.
        """
        # assemble real time trace
        self.rt_trace = RtTrace(max_length=max_length)

        for (process, options) in process_list:
            self.rt_trace.registerRtProcess(process, **options)

        # append packet data to RtTrace
        self.rt_appended_traces = []
        for trace in self.orig_trace_chunks:
            # process single trace
            result = self.rt_trace.append(trace, gap_overlap_check=True)
            # add to list of appended traces
            self.rt_appended_traces.append(result)

    def _plotResults(self):
        """
        Plots original, filtered original and real time processed traces into
        a single plot.
        """
        # plot only if test is started manually
        if __name__ != '__main__':
            return
        # create empty stream
        st = Stream()
        st.label = self._testMethodName
        # original trace
        self.orig_trace.label = "Original Trace"
        st += self.orig_trace
        # use header information of original trace with filtered trace data
        tr = self.orig_trace.copy()
        tr.data = self.filt_trace_data
        tr.label = "Filtered original Trace"
        st += tr
        # real processed chunks
        for i, tr in enumerate(self.rt_appended_traces):
            tr.label = "RT Chunk %02d" % (i + 1)
            st += tr
        # real time processed trace
        self.rt_trace.label = "RT Trace"
        st += self.rt_trace
        st.plot(automerge=False, color='blue', equal_scale=False)


def suite():
    return unittest.makeSuite(RealTimeSignalTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
