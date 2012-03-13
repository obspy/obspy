# -*- coding: utf-8 -*-
"""
The obspyRT.realtime.rttrace test suite.
"""	

import math

import numpy as np
from obspy.core import read
import obspyRT.realtime.rttrace as rt
from obspyRT.realtime.signal.util import *
import os
import unittest

display_processing_functions_doc = True

class RtTraceTestCase(unittest.TestCase):

    plot_traces = False
    num_pakets = 3

    trace_file_name = 'II.TLY.BHZ.SAC'
    trace_file = os.path.join(os.path.dirname(__file__), 'data',
                              trace_file_name)
    trace_gain = 1.610210e+09
    data_trace = None

    process_list = []

    rt_trace = None
    epicentral_distance = None
    ref_time_offest = None

    # processing constants
    scale_factor = 1000
    tauc_window_width = 60
    mwp_max_time = 120
    mwp_mem_time = 2 * mwp_max_time
    boxcar_width = 500


    def test_boxcar(self):
        self.process_list = ['boxcar']
        self.processTrace()
        print 'Post-processing for', self.process_list[-1], ':'
        peak = np.amax(np.abs(self.rt_trace.data))
        print '   ', self.process_list[-1], 'peak = ', peak
        self.assertEqual(int(peak * 1000), int(566974.187 * 1000))
 
    def test_scale(self):
        self.process_list = ['scale']
        self.processTrace()
        print 'Post-processing for', self.process_list[-1], ':'
        peak = np.amax(np.abs(self.rt_trace.data))
        print '   ', self.process_list[-1], 'peak = ', peak
        self.assertEqual(int(peak * 1000), int(1045236992 * 1000))

    def test_abs(self):
        self.process_list = ['np.abs']
        self.processTrace()
        print 'Post-processing for', self.process_list[-1], ':'
        peak = np.amax(np.abs(self.rt_trace.data))
        print '   ', self.process_list[-1], 'peak = ', peak
        self.assertEqual(int(peak * 1000), int(1045237 * 1000))


    def test_tauc(self):
        self.process_list = ['tauc']
        self.processTrace()
        print 'Post-processing for', self.process_list[-1], ':'
        peak = np.amax(np.abs(self.rt_trace.data))
        print '   ', self.process_list[-1], 'peak = ', peak
        self.assertEqual(int(peak * 1000), int(114.296 * 1000))


    def test_mwp(self):
        self.process_list = ['integrate', 'mwpIntegral']
        self.processTrace()
        print 'Post-processing for', self.process_list[-1], ':'
        peak = np.amax(np.abs(self.rt_trace.data))
        print '   ', self.process_list[-1], 'peak = ', peak
        print '    epicentral_distance (sac.gcarc) = ', \
            self.epicentral_distance
        mwp = calculateMwpMag(peak, self.epicentral_distance)
        print '    mwp = ', mwp
        self.assertEqual(int(mwp * 1000), int(8.78902911791 * 1000))


    @classmethod
    def setUpClass(cls):
        print
        if display_processing_functions_doc:
            # display processing functions doc
            print rt.RtTrace.rtProcessFunctionsToString()
        print 'Reading: ', RtTraceTestCase.trace_file
        RtTraceTestCase.data_trace = read(RtTraceTestCase.trace_file)
        RtTraceTestCase.data_trace[0].write('trace_orig.sac', format='SAC')
        # set needed values
        RtTraceTestCase.ref_time_offest = \
            RtTraceTestCase.data_trace[0].stats['sac']['a']
        print '   ref_time_offest (sac.a):' + \
            str(RtTraceTestCase.ref_time_offest)
        if math.fabs(RtTraceTestCase.ref_time_offest - -12345.0) < 0.001:
            print 'Error: sac.a value not set.'
        RtTraceTestCase.epicentral_distance = \
            RtTraceTestCase.data_trace[0].stats['sac']['gcarc']
        print '   sac.gcarc:' + str(RtTraceTestCase.epicentral_distance)
        if math.fabs(RtTraceTestCase.epicentral_distance - -12345.0) < 0.001:
            print 'Error: sac.gcarc value not set.'
        if RtTraceTestCase.plot_traces:
            RtTraceTestCase.plotOriginal()


    def setUp(self):
        pass


    def tearDown(self):
        if RtTraceTestCase.plot_traces:
            RtTraceTestCase.plotResults()


    def processTrace(self):
        print
        print 'Test processing:', self.process_list

        # apply normal obspy processing to original trace
        # Filtering the Stream object
        st_filt = self.data_trace[0].copy()
        delta_time = 1.0 / self.data_trace[0].stats.sampling_rate
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
                st_filt.data = \
                    mwpIntegral(st_filt,
                                mem_time=self.mwp_mem_time,
                                ref_time=(st_filt.stats.starttime
                                + self.ref_time_offest),
                                max_time=self.mwp_max_time,
                                gain=self.trace_gain)
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
        total_length = np.size(self.data_trace[0].data)
        # following may give int truncate
        packet_length = int(total_length / self.num_pakets)  
        delta_time = 1.0 / self.data_trace[0].stats.sampling_rate
        print "self.data_trace[0].stats.sampling_rate:", self.data_trace[0].stats.sampling_rate
        print "delta_time:", delta_time
        tstart = self.data_trace[0].stats.starttime
        tend = tstart + delta_time * float(packet_length - 1)
        print "next_tstart, next_tend:", tstart, tend
        traces = []
        for i in range(self.num_pakets):
            tr = self.data_trace[0].copy()
            tr = tr.slice(tstart, tend)
            traces.append(tr)
            print "tr.stats.sampling_rate:", tr.stats.sampling_rate
            tstart = tr.stats.endtime + delta_time
            print "tstart, tend:", tr.stats.starttime, tr.stats.endtime
            tend = tstart + delta_time * float(packet_length - 1)
            print "next_tstart, next_tend:", tstart, tend

        # assemble realtime trace
        self.rt_trace = rt.RtTrace()
        #self.rt_trace = rt.RtTrace(max_length=600)
        #
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
                self.rt_trace.registerRtProcess('int')
            elif process == 'differentiate':
                self.rt_trace.registerRtProcess('diff')
            elif process == 'mwpIntegral':
                self.rt_trace.registerRtProcess \
                    (
                     'mwpIntegral', mem_time=self.mwp_max_time,
                     ref_time=(self.data_trace[0].stats.starttime + \
                     self.ref_time_offest),
                     max_time=self.mwp_max_time, gain=self.trace_gain)
            elif process == 'np.abs':
                self.rt_trace.registerRtProcess('np.abs')
            elif process == 'np.square':
                self.rt_trace.registerRtProcess('np.square')
            else:
                self.rt_trace.registerRtProcess(process)

        # append packet data to RtTrace
        print 'Appending packets:',
        for i in range(self.num_pakets):
            print i,
            appended_trace = \
                self.rt_trace.append(traces[i], gap_overlap_check=True)
            appended_trace.write('appended_trace%d.sac' % (i), format='SAC')
        print
        self.rt_trace.write('rt_trace.sac', format='SAC')


    @classmethod
    def plotOriginal(cls):
        plt_stream = read('trace_orig.sac')
        plt_stream.plot(automerge=False, size=(800, 1000), color='blue')


    @classmethod
    def plotResults(cls):
        plt_stream = read('trace.sac')
        plt_stream += read('rt_trace.sac')
        for i in range(RtTraceTestCase.num_pakets):
            plt_stream += read('appended_trace%d.sac' % (i))
        plt_stream.plot(automerge=False, size=(800, 1000), color='blue')


def suite():
    return unittest.makeSuite(RtTraceTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')