# -*- coding: utf-8 -*-
"""
The obspy.realtime.rttrace test suite.
"""
from obspy import Trace
from obspy.core.stream import read
from obspy.realtime import RtTrace
from obspy.realtime.rtmemory import RtMemory
from obspy.signal import filter
import numpy as np
import unittest
import warnings


class RtTraceTestCase(unittest.TestCase):

    def test_eq(self):
        """
        Testing __eq__ method.
        """
        tr = Trace()
        tr2 = RtTrace()
        tr3 = RtTrace()
        # RtTrace should never be equal with Trace objects
        self.assertFalse(tr2 == tr)
        self.assertFalse(tr2.__eq__(tr))
        self.assertTrue(tr2 == tr3)
        self.assertTrue(tr2.__eq__(tr3))

    def test_ne(self):
        """
        Testing __ne__ method.
        """
        tr = Trace()
        tr2 = RtTrace()
        tr3 = RtTrace()
        # RtTrace should never be equal with Trace objects
        self.assertTrue(tr2 != tr)
        self.assertTrue(tr2.__ne__(tr))
        self.assertFalse(tr2 != tr3)
        self.assertFalse(tr2.__ne__(tr3))

    def test_registerRtProcess(self):
        """
        Testing registerRtProcess method.
        """
        tr = RtTrace()
        # 1 - function call
        tr.registerRtProcess(np.abs)
        self.assertEqual(tr.processing, [(np.abs, {}, None)])
        # 2 - predefined RT processing algorithm
        tr.registerRtProcess('integrate', test=1, muh='maeh')
        self.assertEqual(tr.processing[1][0], 'integrate')
        self.assertEqual(tr.processing[1][1], {'test': 1, 'muh': 'maeh'})
        self.assertTrue(isinstance(tr.processing[1][2][0], RtMemory))
        # 3 - contained name of predefined RT processing algorithm
        tr.registerRtProcess('in')
        self.assertEqual(tr.processing[2][0], 'integrate')
        tr.registerRtProcess('integ')
        self.assertEqual(tr.processing[3][0], 'integrate')
        tr.registerRtProcess('integr')
        self.assertEqual(tr.processing[4][0], 'integrate')
        # 4 - unknown functions
        self.assertRaises(NotImplementedError,
                          tr.registerRtProcess, 'integrate2')
        self.assertRaises(NotImplementedError, tr.registerRtProcess, 'xyz')
        # 5 - module instead of function
        self.assertRaises(NotImplementedError, tr.registerRtProcess, np)
        # check number off all processing steps within RtTrace
        self.assertEqual(len(tr.processing), 5)
        # check tr.stats.processing
        self.assertEqual(len(tr.stats.processing), 5)
        self.assertTrue(tr.stats.processing[0].startswith("realtime_process"))
        self.assertTrue('absolute' in tr.stats.processing[0])
        for i in range(1, 5):
            self.assertTrue('integrate' in tr.stats.processing[i])
        # check kwargs
        self.assertTrue("maeh" in tr.stats.processing[1])

    def test_appendSanityChecks(self):
        """
        Testing sanity checks of append method.
        """
        rtr = RtTrace()
        ftr = Trace(data=np.array([0, 1]))
        # sanity checks need something already appended
        rtr.append(ftr)
        # 1 - differing ID
        tr = Trace(header={'network': 'xyz'})
        self.assertRaises(TypeError, rtr.append, tr)
        tr = Trace(header={'station': 'xyz'})
        self.assertRaises(TypeError, rtr.append, tr)
        tr = Trace(header={'location': 'xy'})
        self.assertRaises(TypeError, rtr.append, tr)
        tr = Trace(header={'channel': 'xyz'})
        self.assertRaises(TypeError, rtr.append, tr)
        # 2 - sample rate
        tr = Trace(header={'sampling_rate': 100.0})
        self.assertRaises(TypeError, rtr.append, tr)
        tr = Trace(header={'delta': 0.25})
        self.assertRaises(TypeError, rtr.append, tr)
        # 3 - calibration factor
        tr = Trace(header={'calib': 100.0})
        self.assertRaises(TypeError, rtr.append, tr)
        # 4 - data type
        tr = Trace(data=np.array([0.0, 1.1]))
        self.assertRaises(TypeError, rtr.append, tr)
        # 5 - only Trace objects are allowed
        self.assertRaises(TypeError, rtr.append, 1)
        self.assertRaises(TypeError, rtr.append, "2323")

    def test_appendOverlap(self):
        """
        Appending overlapping traces should raise a UserWarning/TypeError
        """
        rtr = RtTrace()
        tr = Trace(data=np.array([0, 1]))
        rtr.append(tr)
        # this raises UserWarning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            self.assertRaises(UserWarning, rtr.append, tr)
        # append with gap_overlap_check=True will raise a TypeError
        self.assertRaises(TypeError, rtr.append, tr, gap_overlap_check=True)

    def test_appendGap(self):
        """
        Appending a traces with a time gap should raise a UserWarning/TypeError
        """
        rtr = RtTrace()
        tr = Trace(data=np.array([0, 1]))
        tr2 = Trace(data=np.array([5, 6]))
        tr2.stats.starttime = tr.stats.starttime + 10
        rtr.append(tr)
        # this raises UserWarning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            self.assertRaises(UserWarning, rtr.append, tr2)
        # append with gap_overlap_check=True will raise a TypeError
        self.assertRaises(TypeError, rtr.append, tr2, gap_overlap_check=True)

    def test_copy(self):
        """
        Testing copy of RtTrace object.
        """
        rtr = RtTrace()
        rtr.copy()
        # register predefined function
        rtr.registerRtProcess('integrate', test=1, muh='maeh')
        rtr.copy()
        # register ObsPy function call
        rtr.registerRtProcess(filter.bandpass, freqmin=0, freqmax=1, df=0.1)
        rtr.copy()
        # register NumPy function call
        rtr.registerRtProcess(np.square)
        rtr.copy()

    def test_appendNotFloat32(self):
        """
        Test for not using float32.
        """
        tr = read()[0]
        tr.data = np.require(tr.data, dtype='>f4')
        traces = tr / 3
        rtr = RtTrace()
        for trace in traces:
            rtr.append(trace)

    def test_missingOrWrongArgumentInRtProcess(self):
        """
        Tests handling of missing/wrong arguments.
        """
        trace = Trace(np.arange(100))
        # 1- function scale needs no additional arguments
        rt_trace = RtTrace()
        rt_trace.registerRtProcess('scale')
        rt_trace.append(trace)
        # adding arbitrary arguments should fail
        rt_trace = RtTrace()
        rt_trace.registerRtProcess('scale', muh='maeh')
        self.assertRaises(TypeError, rt_trace.append, trace)
        # 2- function tauc has one required argument
        rt_trace = RtTrace()
        rt_trace.registerRtProcess('tauc', width=10)
        rt_trace.append(trace)
        # wrong argument should fail
        rt_trace = RtTrace()
        rt_trace.registerRtProcess('tauc', xyz='xyz')
        self.assertRaises(TypeError, rt_trace.append, trace)
        # missing argument width should raise an exception
        rt_trace = RtTrace()
        rt_trace.registerRtProcess('tauc')
        self.assertRaises(TypeError, rt_trace.append, trace)
        # adding arbitrary arguments should fail
        rt_trace = RtTrace()
        rt_trace.registerRtProcess('tauc', width=20, notexistingoption=True)
        self.assertRaises(TypeError, rt_trace.append, trace)


def suite():
    return unittest.makeSuite(RtTraceTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
