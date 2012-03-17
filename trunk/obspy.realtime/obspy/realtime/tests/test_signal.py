# -*- coding: utf-8 -*-
"""
The obspy.realtime.signal test suite.
"""
from obspy.core import read
from obspy.realtime import RtTrace, _splitTrace
from obspy.realtime.signal import calculateMwpMag, scale, tauc, boxcar, \
    integrate, differentiate, mwpIntegral
import math
import numpy as np
import os
import unittest


# some debug flags
PLOT_TRACES = False
DISPLAY_PROCESSING_FUNCTIONS_DOC = False
NUM_PAKETS = 3


class RealTimeSignalTestCase(unittest.TestCase):
    """
    The obspy.realtime.signal test suite.
    """
    trace_file_name = 'II.TLY.BHZ.SAC'
    trace_file = os.path.join(os.path.dirname(__file__), 'data',
                              trace_file_name)
    trace_gain = 1.610210e+09

    # processing constants
    scale_factor = 1000
    tauc_window_width = 60
    mwp_max_time = 120
    mwp_mem_time = 2 * mwp_max_time
    boxcar_width = 500

    @classmethod
    def setUpClass(cls):
        if DISPLAY_PROCESSING_FUNCTIONS_DOC:
            # display processing functions doc
            print RtTrace.rtProcessFunctionsToString()
        cls.data_trace = read(cls.trace_file)
        cls.data_trace[0].write('trace_orig.sac', format='SAC')
        # set needed values
        cls.ref_time_offset = \
            cls.data_trace[0].stats['sac']['a']
        if math.fabs(cls.ref_time_offset - -12345.0) < 0.001:
            print 'Error: sac.a value not set.'
        cls.epicentral_distance = \
            cls.data_trace[0].stats['sac']['gcarc']
        if math.fabs(cls.epicentral_distance - -12345.0) < 0.001:
            print 'Error: sac.gcarc value not set.'
        if PLOT_TRACES:
            cls.plotOriginal()

    @classmethod
    def tearDownClass(cls):
        # cleanup
        for file in os.listdir(os.path.dirname(__file__)):
            if file.endswith('.sac'):
                try:
                    os.remove(file)
                except:
                    pass

    def tearDown(self):
        if PLOT_TRACES:
            self.plotResults()

    def test_square(self):
        self.process_list = ['np.square']
        self._processTrace()
        np.testing.assert_array_equal(np.square(self.data_trace[0].data),
                                      self.rt_trace.data)

    def test_boxcar(self):
        self.process_list = ['boxcar']
        self._processTrace()
        peak = np.amax(np.abs(self.rt_trace.data))
        self.assertAlmostEqual(peak, 566974.187, 3)

    def test_scale(self):
        self.process_list = ['scale']
        self._processTrace()
        peak = np.amax(np.abs(self.rt_trace.data))
        self.assertEqual(peak, 1045236992)

    def test_abs(self):
        self.process_list = ['np.abs']
        self._processTrace()
        peak = np.amax(np.abs(self.rt_trace.data))
        self.assertEqual(peak, 1045237)
        np.testing.assert_array_equal(np.abs(self.data_trace[0].data),
                                      self.rt_trace.data)

    def test_tauc(self):
        self.process_list = ['tauc']
        self._processTrace()
        peak = np.amax(np.abs(self.rt_trace.data))
        self.assertAlmostEqual(peak, 114.296, 3)

    def test_mwp(self):
        self.process_list = ['integrate', 'mwpIntegral']
        self._processTrace()
        peak = np.amax(np.abs(self.rt_trace.data))
        mwp = calculateMwpMag(peak, self.epicentral_distance)
        self.assertAlmostEqual(mwp, 8.78902911791)

    def _processTrace(self):
        # apply normal obspy processing to original trace
        # Filtering the Stream object
        st_filt = self.data_trace[0].copy()
        for process in self.process_list:
            if process == 'scale':
                st_filt.data = scale(st_filt, factor=self.scale_factor)
            elif process == 'tauc':
                st_filt.data = tauc(st_filt, width=self.tauc_window_width)
            elif process == 'boxcar':
                st_filt.data = boxcar(st_filt, width=self.boxcar_width)
            elif process == 'integrate':
                st_filt.data = integrate(st_filt)
            elif process == 'differentiate':
                st_filt.data = differentiate(st_filt)
            elif process == 'mwpIntegral':
                st_filt.data = mwpIntegral(st_filt, mem_time=self.mwp_mem_time,
                    ref_time=(st_filt.stats.starttime + self.ref_time_offset),
                    max_time=self.mwp_max_time, gain=self.trace_gain)
            elif process == 'np.abs':
                st_filt.data = np.abs(st_filt.data)
            elif process == 'np.square':
                st_filt.data = np.square(st_filt.data)
            else:
                print 'Warning: process:', process, \
                    ': not supported by this function'
        # save processed trace to disk
        st_filt.write('trace.sac', format='SAC')

        # create set of contiguous packet data in an array of Trace objects
        traces = _splitTrace(self.data_trace[0], NUM_PAKETS)

        # assemble realtime trace
        self.rt_trace = RtTrace()
        #self.rt_trace = RtTrace(max_length=600)

        for process in self.process_list:
            if process == 'scale':
                self.rt_trace.registerRtProcess('scale',
                                                factor=self.scale_factor)
            elif process == 'tauc':
                self.rt_trace.registerRtProcess('tauc',
                                                width=self.tauc_window_width)
            elif process == 'boxcar':
                self.rt_trace.registerRtProcess('boxcar',
                                                width=self.boxcar_width)
            elif process == 'integrate':
                # 'int' is contained in 'integrate'
                self.rt_trace.registerRtProcess('int')
            elif process == 'differentiate':
                self.rt_trace.registerRtProcess('diff')
            elif process == 'mwpIntegral':
                self.rt_trace.registerRtProcess('mwpIntegral',
                    mem_time=self.mwp_max_time,
                    ref_time=(self.data_trace[0].stats.starttime + \
                              self.ref_time_offset),
                    max_time=self.mwp_max_time, gain=self.trace_gain)
            elif process == 'np.abs':
                self.rt_trace.registerRtProcess(np.abs)
            elif process == 'np.square':
                self.rt_trace.registerRtProcess(np.square)
            else:
                self.rt_trace.registerRtProcess(process)

        # append packet data to RtTrace
        for i, trace in enumerate(traces):
            appended_trace = \
                self.rt_trace.append(trace, gap_overlap_check=True)
            appended_trace.write('appended_trace%d.sac' % (i), format='SAC')
        self.rt_trace.write('rt_trace.sac', format='SAC')

    @classmethod
    def plotOriginal(cls):
        plt_stream = read('trace_orig.sac')
        plt_stream.plot(automerge=False, size=(800, 1000), color='blue')

    @classmethod
    def plotResults(cls):
        plt_stream = read('trace.sac')
        plt_stream += read('rt_trace.sac')
        for i in range(cls.num_pakets):
            plt_stream += read('appended_trace%d.sac' % (i))
        plt_stream.plot(automerge=False, size=(800, 1000), color='blue')


def suite():
    return unittest.makeSuite(RealTimeSignalTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
