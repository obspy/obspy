# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import math
import os
import unittest
from copy import deepcopy

import numpy as np
import numpy.ma as ma

from obspy import Stream, Trace, UTCDateTime, __version__, read, read_inventory
from obspy.core import Stats
from obspy.core.compatibility import mock
from obspy.core.util.testing import ImageComparison
from obspy.io.xseed import Parser


class TraceTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.trace.Trace.
    """
    @staticmethod
    def __remove_processing(tr):
        """
        Removes all processing information in the trace object.

        Useful for testing.
        """
        if "processing" not in tr.stats:
            return
        del tr.stats.processing

    def test_init(self):
        """
        Tests the __init__ method of the Trace class.
        """
        # NumPy ndarray
        tr = Trace(data=np.arange(4))
        self.assertEqual(len(tr), 4)
        # NumPy masked array
        data = np.ma.array([0, 1, 2, 3], mask=[True, True, False, False])
        tr = Trace(data=data)
        self.assertEqual(len(tr), 4)
        # other data types will raise
        self.assertRaises(ValueError, Trace, data=[0, 1, 2, 3])
        self.assertRaises(ValueError, Trace, data=(0, 1, 2, 3))
        self.assertRaises(ValueError, Trace, data='1234')

    def test_setattr(self):
        """
        Tests the __setattr__ method of the Trace class.
        """
        # NumPy ndarray
        tr = Trace()
        tr.data = np.arange(4)
        self.assertEqual(len(tr), 4)
        # NumPy masked array
        tr = Trace()
        tr.data = np.ma.array([0, 1, 2, 3], mask=[True, True, False, False])
        self.assertEqual(len(tr), 4)
        # other data types will raise
        tr = Trace()
        self.assertRaises(ValueError, tr.__setattr__, 'data', [0, 1, 2, 3])
        self.assertRaises(ValueError, tr.__setattr__, 'data', (0, 1, 2, 3))
        self.assertRaises(ValueError, tr.__setattr__, 'data', '1234')

    def test_len(self):
        """
        Tests the __len__ and count methods of the Trace class.
        """
        trace = Trace(data=np.arange(1000))
        self.assertEqual(len(trace), 1000)
        self.assertEqual(trace.count(), 1000)

    def test_mul(self):
        """
        Tests the __mul__ method of the Trace class.
        """
        tr = Trace(data=np.arange(10))
        st = tr * 5
        self.assertEqual(len(st), 5)
        # you may only multiply using an integer
        self.assertRaises(TypeError, tr.__mul__, 2.5)
        self.assertRaises(TypeError, tr.__mul__, '1234')

    def test_div(self):
        """
        Tests the __div__ method of the Trace class.
        """
        tr = Trace(data=np.arange(1000))
        st = tr / 5
        self.assertEqual(len(st), 5)
        self.assertEqual(len(st[0]), 200)
        # you may only multiply using an integer
        self.assertRaises(TypeError, tr.__div__, 2.5)
        self.assertRaises(TypeError, tr.__div__, '1234')

    def test_ltrim(self):
        """
        Tests the ltrim method of the Trace class.
        """
        # set up
        trace = Trace(data=np.arange(1000))
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 200.0
        end = UTCDateTime(2000, 1, 1, 0, 0, 4, 995000)
        # verify
        trace.verify()
        # UTCDateTime/int/float required
        self.assertRaises(TypeError, trace._ltrim, '1234')
        self.assertRaises(TypeError, trace._ltrim, [1, 2, 3, 4])
        # ltrim 100 samples
        tr = deepcopy(trace)
        tr._ltrim(0.5)
        tr.verify()
        np.testing.assert_array_equal(tr.data[0:5],
                                      np.array([100, 101, 102, 103, 104]))
        self.assertEqual(len(tr.data), 900)
        self.assertEqual(tr.stats.npts, 900)
        self.assertEqual(tr.stats.sampling_rate, 200.0)
        self.assertEqual(tr.stats.starttime, start + 0.5)
        self.assertEqual(tr.stats.endtime, end)
        # ltrim 202 samples
        tr = deepcopy(trace)
        tr._ltrim(1.010)
        tr.verify()
        np.testing.assert_array_equal(tr.data[0:5],
                                      np.array([202, 203, 204, 205, 206]))
        self.assertEqual(len(tr.data), 798)
        self.assertEqual(tr.stats.npts, 798)
        self.assertEqual(tr.stats.sampling_rate, 200.0)
        self.assertEqual(tr.stats.starttime, start + 1.010)
        self.assertEqual(tr.stats.endtime, end)
        # ltrim to UTCDateTime
        tr = deepcopy(trace)
        tr._ltrim(UTCDateTime(2000, 1, 1, 0, 0, 1, 10000))
        tr.verify()
        np.testing.assert_array_equal(tr.data[0:5],
                                      np.array([202, 203, 204, 205, 206]))
        self.assertEqual(len(tr.data), 798)
        self.assertEqual(tr.stats.npts, 798)
        self.assertEqual(tr.stats.sampling_rate, 200.0)
        self.assertEqual(tr.stats.starttime, start + 1.010)
        self.assertEqual(tr.stats.endtime, end)
        # some sanity checks
        # negative start time as datetime
        tr = deepcopy(trace)
        tr._ltrim(start - 1, pad=True)
        tr.verify()
        self.assertEqual(tr.stats.starttime, start - 1)
        np.testing.assert_array_equal(trace.data, tr.data[200:])
        self.assertEqual(tr.stats.endtime, trace.stats.endtime)
        # negative start time as integer
        tr = deepcopy(trace)
        tr._ltrim(-100, pad=True)
        tr.verify()
        self.assertEqual(tr.stats.starttime, start - 100)
        delta = 100 * trace.stats.sampling_rate
        np.testing.assert_array_equal(trace.data, tr.data[int(delta):])
        self.assertEqual(tr.stats.endtime, trace.stats.endtime)
        # start time > end time
        tr = deepcopy(trace)
        tr._ltrim(trace.stats.endtime + 100)
        tr.verify()
        self.assertEqual(tr.stats.starttime,
                         trace.stats.endtime + 100)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        self.assertEqual(tr.stats.endtime, tr.stats.starttime)
        # start time == end time
        tr = deepcopy(trace)
        tr._ltrim(5)
        tr.verify()
        self.assertEqual(tr.stats.starttime,
                         trace.stats.starttime + 5)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        self.assertEqual(tr.stats.endtime, tr.stats.starttime)
        # start time == end time
        tr = deepcopy(trace)
        tr._ltrim(5.1)
        tr.verify()
        self.assertEqual(tr.stats.starttime,
                         trace.stats.starttime + 5.1)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        self.assertEqual(tr.stats.endtime, tr.stats.starttime)

    def test_rtrim(self):
        """
        Tests the rtrim method of the Trace class.
        """
        # set up
        trace = Trace(data=np.arange(1000))
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 200.0
        end = UTCDateTime(2000, 1, 1, 0, 0, 4, 995000)
        trace.verify()
        # UTCDateTime/int/float required
        self.assertRaises(TypeError, trace._rtrim, '1234')
        self.assertRaises(TypeError, trace._rtrim, [1, 2, 3, 4])
        # rtrim 100 samples
        tr = deepcopy(trace)
        tr._rtrim(0.5)
        tr.verify()
        np.testing.assert_array_equal(tr.data[-5:],
                                      np.array([895, 896, 897, 898, 899]))
        self.assertEqual(len(tr.data), 900)
        self.assertEqual(tr.stats.npts, 900)
        self.assertEqual(tr.stats.sampling_rate, 200.0)
        self.assertEqual(tr.stats.starttime, start)
        self.assertEqual(tr.stats.endtime, end - 0.5)
        # rtrim 202 samples
        tr = deepcopy(trace)
        tr._rtrim(1.010)
        tr.verify()
        np.testing.assert_array_equal(tr.data[-5:],
                                      np.array([793, 794, 795, 796, 797]))
        self.assertEqual(len(tr.data), 798)
        self.assertEqual(tr.stats.npts, 798)
        self.assertEqual(tr.stats.sampling_rate, 200.0)
        self.assertEqual(tr.stats.starttime, start)
        self.assertEqual(tr.stats.endtime, end - 1.010)
        # rtrim 1 minute via UTCDateTime
        tr = deepcopy(trace)
        tr._rtrim(UTCDateTime(2000, 1, 1, 0, 0, 3, 985000))
        tr.verify()
        np.testing.assert_array_equal(tr.data[-5:],
                                      np.array([793, 794, 795, 796, 797]))
        self.assertEqual(len(tr.data), 798)
        self.assertEqual(tr.stats.npts, 798)
        self.assertEqual(tr.stats.sampling_rate, 200.0)
        self.assertEqual(tr.stats.starttime, start)
        self.assertEqual(tr.stats.endtime, end - 1.010)
        # some sanity checks
        # negative end time
        tr = deepcopy(trace)
        t = UTCDateTime(1999, 12, 31)
        tr._rtrim(t)
        tr.verify()
        self.assertEqual(tr.stats.endtime, t)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        # negative end time with given seconds
        tr = deepcopy(trace)
        tr._rtrim(100)
        tr.verify()
        self.assertEqual(tr.stats.endtime, trace.stats.endtime - 100)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        self.assertEqual(tr.stats.endtime, tr.stats.starttime)
        # end time > start time
        tr = deepcopy(trace)
        t = UTCDateTime(2001)
        tr._rtrim(t)
        tr.verify()
        self.assertEqual(tr.stats.endtime, t)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        self.assertEqual(tr.stats.endtime, tr.stats.starttime)
        # end time > start time given seconds
        tr = deepcopy(trace)
        tr._rtrim(5.1)
        tr.verify()
        delta = int(math.floor(round(5.1 * trace.stats.sampling_rate, 7)))
        endtime = trace.stats.starttime + trace.stats.delta * \
            (trace.stats.npts - delta - 1)
        self.assertEqual(tr.stats.endtime, endtime)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        # end time == start time
        # returns one sample!
        tr = deepcopy(trace)
        tr._rtrim(4.995)
        tr.verify()
        np.testing.assert_array_equal(tr.data, np.array([0]))
        self.assertEqual(len(tr.data), 1)
        self.assertEqual(tr.stats.npts, 1)
        self.assertEqual(tr.stats.sampling_rate, 200.0)
        self.assertEqual(tr.stats.starttime, start)
        self.assertEqual(tr.stats.endtime, start)

    def test_rtrim_with_padding(self):
        """
        Tests the _rtrim() method of the Trace class with padding. It has
        already been tested in the two sided trimming tests. This is just to
        have an explicit test. Also tests issue #429.
        """
        # set up
        trace = Trace(data=np.arange(10))
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 1.0
        trace.verify()

        # Pad with no fill_value will mask the additional values.
        tr = trace.copy()
        end = tr.stats.endtime
        tr._rtrim(end + 10, pad=True)
        self.assertEqual(tr.stats.endtime, trace.stats.endtime + 10)
        np.testing.assert_array_equal(tr.data[0:10], np.arange(10))
        # Check that the first couple of entries are not masked.
        self.assertFalse(tr.data[0:10].mask.any())
        # All the other entries should be masked.
        self.assertTrue(tr.data[10:].mask.all())

        # Pad with fill_value.
        tr = trace.copy()
        end = tr.stats.endtime
        tr._rtrim(end + 10, pad=True, fill_value=-33)
        self.assertEqual(tr.stats.endtime, trace.stats.endtime + 10)
        # The first ten entries should not have changed.
        np.testing.assert_array_equal(tr.data[0:10], np.arange(10))
        # The rest should be filled with the fill_value.
        np.testing.assert_array_equal(tr.data[10:], np.ones(10) * -33)

    def test_trim(self):
        """
        Tests the trim method of the Trace class.
        """
        # set up
        trace = Trace(data=np.arange(1001))
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 200.0
        end = UTCDateTime(2000, 1, 1, 0, 0, 5, 0)
        trace.verify()
        # rtrim 100 samples
        trace.trim(0.5, 0.5)
        trace.verify()
        np.testing.assert_array_equal(trace.data[-5:],
                                      np.array([896, 897, 898, 899, 900]))
        np.testing.assert_array_equal(trace.data[:5],
                                      np.array([100, 101, 102, 103, 104]))
        self.assertEqual(len(trace.data), 801)
        self.assertEqual(trace.stats.npts, 801)
        self.assertEqual(trace.stats.sampling_rate, 200.0)
        self.assertEqual(trace.stats.starttime, start + 0.5)
        self.assertEqual(trace.stats.endtime, end - 0.5)
        # start time should be before end time
        self.assertRaises(ValueError, trace.trim, end, start)

    def test_trimAllDoesNotChangeDtype(self):
        """
        If a Trace is completely trimmed, e.g. no data samples are remaining,
        the dtype should remain unchanged.

        A trace with no data samples is not really senseful but the dtype
        should not be changed anyways.
        """
        # Choose non native dtype.
        tr = Trace(np.arange(100, dtype=np.int16))
        tr.trim(UTCDateTime(10000), UTCDateTime(20000))
        # Assert the result.
        self.assertEqual(len(tr.data), 0)
        self.assertEqual(tr.data.dtype, np.int16)

    def test_addTraceWithGap(self):
        """
        Tests __add__ method of the Trace class.
        """
        # set up
        tr1 = Trace(data=np.arange(1000))
        tr1.stats.sampling_rate = 200
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr2 = Trace(data=np.arange(0, 1000)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 10
        # verify
        tr1.verify()
        tr2.verify()
        # add
        trace = tr1 + tr2
        # stats
        self.assertEqual(trace.stats.starttime, start)
        self.assertEqual(trace.stats.endtime, start + 14.995)
        self.assertEqual(trace.stats.sampling_rate, 200)
        self.assertEqual(trace.stats.npts, 3000)
        # data
        self.assertEqual(len(trace), 3000)
        self.assertEqual(trace[0], 0)
        self.assertEqual(trace[999], 999)
        self.assertTrue(ma.is_masked(trace[1000]))
        self.assertTrue(ma.is_masked(trace[1999]))
        self.assertEqual(trace[2000], 999)
        self.assertEqual(trace[2999], 0)
        # verify
        trace.verify()

    def test_addTraceWithOverlap(self):
        """
        Tests __add__ method of the Trace class.
        """
        # set up
        tr1 = Trace(data=np.arange(1000))
        tr1.stats.sampling_rate = 200
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr2 = Trace(data=np.arange(0, 1000)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 4
        # add
        trace = tr1 + tr2
        # stats
        self.assertEqual(trace.stats.starttime, start)
        self.assertEqual(trace.stats.endtime, start + 8.995)
        self.assertEqual(trace.stats.sampling_rate, 200)
        self.assertEqual(trace.stats.npts, 1800)
        # data
        self.assertEqual(len(trace), 1800)
        self.assertEqual(trace[0], 0)
        self.assertEqual(trace[799], 799)
        self.assertTrue(trace[800].mask)
        self.assertTrue(trace[999].mask)
        self.assertEqual(trace[1000], 799)
        self.assertEqual(trace[1799], 0)
        # verify
        trace.verify()

    def test_addSameTrace(self):
        """
        Tests __add__ method of the Trace class.
        """
        # set up
        tr1 = Trace(data=np.arange(1001))
        # add
        trace = tr1 + tr1
        # should return exact the same values
        self.assertEqual(trace.stats, tr1.stats)
        np.testing.assert_array_equal(trace.data, tr1.data)
        # verify
        trace.verify()

    def test_addTraceWithinTrace(self):
        """
        Tests __add__ method of the Trace class.
        """
        # set up
        tr1 = Trace(data=np.arange(1001))
        tr1.stats.sampling_rate = 200
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr2 = Trace(data=np.arange(201))
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 1
        # add
        trace = tr1 + tr2
        # should return exact the same values like trace 1
        self.assertEqual(trace.stats, tr1.stats)
        mask = np.zeros(len(tr1)).astype(np.bool_)
        mask[200:401] = True
        np.testing.assert_array_equal(trace.data.mask, mask)
        np.testing.assert_array_equal(trace.data.data[:200], tr1.data[:200])
        np.testing.assert_array_equal(trace.data.data[401:], tr1.data[401:])
        # add the other way around
        trace = tr2 + tr1
        # should return exact the same values like trace 1
        self.assertEqual(trace.stats, tr1.stats)
        np.testing.assert_array_equal(trace.data.mask, mask)
        np.testing.assert_array_equal(trace.data.data[:200], tr1.data[:200])
        np.testing.assert_array_equal(trace.data.data[401:], tr1.data[401:])
        # verify
        trace.verify()

    def test_addGapAndOverlap(self):
        """
        Test order of merging traces.
        """
        # set up
        tr1 = Trace(data=np.arange(1000))
        tr1.stats.sampling_rate = 200
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr2 = Trace(data=np.arange(1000)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 4
        tr3 = Trace(data=np.arange(1000)[::-1])
        tr3.stats.sampling_rate = 200
        tr3.stats.starttime = start + 12
        # overlap
        overlap = tr1 + tr2
        self.assertEqual(len(overlap), 1800)
        mask = np.zeros(1800).astype(np.bool_)
        mask[800:1000] = True
        np.testing.assert_array_equal(overlap.data.mask, mask)
        np.testing.assert_array_equal(overlap.data.data[:800], tr1.data[:800])
        np.testing.assert_array_equal(overlap.data.data[1000:], tr2.data[200:])
        # overlap + gap
        overlap_gap = overlap + tr3
        self.assertEqual(len(overlap_gap), 3400)
        mask = np.zeros(3400).astype(np.bool_)
        mask[800:1000] = True
        mask[1800:2400] = True
        np.testing.assert_array_equal(overlap_gap.data.mask, mask)
        np.testing.assert_array_equal(overlap_gap.data.data[:800],
                                      tr1.data[:800])
        np.testing.assert_array_equal(overlap_gap.data.data[1000:1800],
                                      tr2.data[200:])
        np.testing.assert_array_equal(overlap_gap.data.data[2400:], tr3.data)
        # gap
        gap = tr2 + tr3
        self.assertEqual(len(gap), 2600)
        mask = np.zeros(2600).astype(np.bool_)
        mask[1000:1600] = True
        np.testing.assert_array_equal(gap.data.mask, mask)
        np.testing.assert_array_equal(gap.data.data[:1000], tr2.data)
        np.testing.assert_array_equal(gap.data.data[1600:], tr3.data)

    def test_addIntoGap(self):
        """
        Test __add__ method of the Trace class
        Adding a trace that fits perfectly into gap in a trace
        """
        myArray = np.arange(6, dtype=np.int32)

        stats = Stats()
        stats.network = 'VI'
        stats['starttime'] = UTCDateTime(2009, 8, 5, 0, 0, 0)
        stats['npts'] = 0
        stats['station'] = 'IKJA'
        stats['channel'] = 'EHZ'
        stats['sampling_rate'] = 1

        bigtrace = Trace(data=np.array([], dtype=np.int32), header=stats)
        bigtrace_sort = bigtrace.copy()
        stats['npts'] = len(myArray)
        myTrace = Trace(data=myArray, header=stats)

        stats['npts'] = 2
        trace1 = Trace(data=myArray[0:2].copy(), header=stats)
        stats['starttime'] = UTCDateTime(2009, 8, 5, 0, 0, 2)
        trace2 = Trace(data=myArray[2:4].copy(), header=stats)
        stats['starttime'] = UTCDateTime(2009, 8, 5, 0, 0, 4)
        trace3 = Trace(data=myArray[4:6].copy(), header=stats)

        tr1 = bigtrace
        tr2 = bigtrace_sort
        for method in [0, 1]:
            # Random
            bigtrace = tr1.copy()
            bigtrace = bigtrace.__add__(trace1, method=method)
            bigtrace = bigtrace.__add__(trace3, method=method)
            bigtrace = bigtrace.__add__(trace2, method=method)

            # Sorted
            bigtrace_sort = tr2.copy()
            bigtrace_sort = bigtrace_sort.__add__(trace1, method=method)
            bigtrace_sort = bigtrace_sort.__add__(trace2, method=method)
            bigtrace_sort = bigtrace_sort.__add__(trace3, method=method)

            for tr in (bigtrace, bigtrace_sort):
                self.assertTrue(isinstance(tr, Trace))
                self.assertFalse(isinstance(tr.data, np.ma.masked_array))

            self.assertTrue((bigtrace_sort.data == myArray).all())

            fail_pattern = "\n\tExpected %s\n\tbut got  %s"
            failinfo = fail_pattern % (myTrace, bigtrace_sort)
            failinfo += fail_pattern % (myTrace.data, bigtrace_sort.data)
            self.assertEqual(bigtrace_sort, myTrace, failinfo)

            failinfo = fail_pattern % (myArray, bigtrace.data)
            self.assertTrue((bigtrace.data == myArray).all(), failinfo)

            failinfo = fail_pattern % (myTrace, bigtrace)
            failinfo += fail_pattern % (myTrace.data, bigtrace.data)
            self.assertEqual(bigtrace, myTrace, failinfo)

            for array_ in (bigtrace.data, bigtrace_sort.data):
                failinfo = fail_pattern % (myArray.dtype, array_.dtype)
                self.assertEqual(myArray.dtype, array_.dtype, failinfo)

    def test_slice(self):
        """
        Tests the slicing of trace objects.
        """
        tr = Trace(data=np.arange(10, dtype=np.int32))
        mempos = tr.data.ctypes.data
        t = tr.stats.starttime
        tr1 = tr.slice(t + 2, t + 8)
        tr1.data[0] = 10
        self.assertEqual(tr.data[2], 10)
        self.assertEqual(tr.data.ctypes.data, mempos)
        self.assertEqual(tr.data[2:9].ctypes.data, tr1.data.ctypes.data)
        self.assertEqual(tr1.data.ctypes.data - 8, mempos)

        # Test the processing information for the slicing. The sliced trace
        # should have a processing information showing that it has been
        # trimmed. The original trace should have nothing.
        tr = Trace(data=np.arange(10, dtype=np.int32))
        tr2 = tr.slice(tr.stats.starttime)
        self.assertNotIn("processing", tr.stats)
        self.assertIn("processing", tr2.stats)
        self.assertIn("trim", tr2.stats.processing[0])

    def test_slice_noStarttimeOrEndtime(self):
        """
        Tests the slicing of trace objects with no start time or end time
        provided. Compares results against the equivalent trim() operation
        """
        tr_orig = Trace(data=np.arange(10, dtype=np.int32))
        tr = tr_orig.copy()
        # two time points outside the trace and two inside
        t1 = tr.stats.starttime - 2
        t2 = tr.stats.starttime + 2
        t3 = tr.stats.endtime - 3
        t4 = tr.stats.endtime + 2

        # test 1: only removing data at left side
        tr_trim = tr_orig.copy()
        tr_trim.trim(starttime=t2)
        self.assertEqual(tr_trim, tr.slice(starttime=t2))
        tr2 = tr.slice(starttime=t2, endtime=t4)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        self.assertEqual(tr_trim, tr2)

        # test 2: only removing data at right side
        tr_trim = tr_orig.copy()
        tr_trim.trim(endtime=t3)
        self.assertEqual(tr_trim, tr.slice(endtime=t3))
        tr2 = tr.slice(starttime=t1, endtime=t3)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        self.assertEqual(tr_trim, tr2)

        # test 3: not removing data at all
        tr_trim = tr_orig.copy()
        tr_trim.trim(starttime=t1, endtime=t4)
        tr2 = tr.slice()
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        self.assertEqual(tr_trim, tr2)

        tr2 = tr.slice(starttime=t1)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        self.assertEqual(tr_trim, tr2)

        tr2 = tr.slice(endtime=t4)
        self.__remove_processing(tr2)
        self.assertEqual(tr_trim, tr2)

        tr2 = tr.slice(starttime=t1, endtime=t4)
        self.__remove_processing(tr2)
        self.assertEqual(tr_trim, tr2)

        tr_trim.trim()
        tr2 = tr.slice()
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        self.assertEqual(tr_trim, tr2)

        tr2 = tr.slice(starttime=t1)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        self.assertEqual(tr_trim, tr2)

        tr2 = tr.slice(endtime=t4)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        self.assertEqual(tr_trim, tr2)

        tr2 = tr.slice(starttime=t1, endtime=t4)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        self.assertEqual(tr_trim, tr2)

        # test 4: removing data at left and right side
        tr_trim = tr_orig.copy()
        tr_trim.trim(starttime=t2, endtime=t3)
        self.assertEqual(tr_trim, tr.slice(t2, t3))
        self.assertEqual(tr_trim, tr.slice(starttime=t2, endtime=t3))

        # test 5: no data left after operation
        tr_trim = tr_orig.copy()
        tr_trim.trim(starttime=t4)

        tr2 = tr.slice(starttime=t4)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        self.assertEqual(tr_trim, tr2)

        tr2 = tr.slice(starttime=t4, endtime=t4 + 1)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        self.assertEqual(tr_trim, tr2)

    def test_slice_nearest_sample(self):
        """
        Tests slicing with the nearest sample flag set to on or off.
        """
        tr = Trace(data=np.arange(6))
        # Samples at:
        # 0       10       20       30       40       50
        tr.stats.sampling_rate = 0.1

        # Nearest sample flag defaults to true.
        tr2 = tr.slice(UTCDateTime(4), UTCDateTime(44))
        self.assertEqual(tr2.stats.starttime, UTCDateTime(0))
        self.assertEqual(tr2.stats.endtime, UTCDateTime(40))

        tr2 = tr.slice(UTCDateTime(8), UTCDateTime(48))
        self.assertEqual(tr2.stats.starttime, UTCDateTime(10))
        self.assertEqual(tr2.stats.endtime, UTCDateTime(50))

        # Setting it to False changes the returned values.
        tr2 = tr.slice(UTCDateTime(4), UTCDateTime(44), nearest_sample=False)
        self.assertEqual(tr2.stats.starttime, UTCDateTime(10))
        self.assertEqual(tr2.stats.endtime, UTCDateTime(40))

        tr2 = tr.slice(UTCDateTime(8), UTCDateTime(48), nearest_sample=False)
        self.assertEqual(tr2.stats.starttime, UTCDateTime(10))
        self.assertEqual(tr2.stats.endtime, UTCDateTime(40))

    def test_trimFloatingPoint(self):
        """
        Tests the slicing of trace objects.
        """
        # Create test array that allows for easy testing.
        tr = Trace(data=np.arange(11))
        org_stats = deepcopy(tr.stats)
        org_data = deepcopy(tr.data)
        # Save memory position of array.
        mem_pos = tr.data.ctypes.data
        # Just some sanity tests.
        self.assertEqual(tr.stats.starttime, UTCDateTime(0))
        self.assertEqual(tr.stats.endtime, UTCDateTime(10))
        # Create temp trace object used for testing.
        st = tr.stats.starttime
        # This is supposed to include the start and end times and should
        # therefore cut right at 2 and 8.
        temp = deepcopy(tr)
        temp.trim(st + 2.1, st + 7.1)
        # Should be identical.
        temp2 = deepcopy(tr)
        temp2.trim(st + 2.0, st + 8.0)
        self.assertEqual(temp.stats.starttime, UTCDateTime(2))
        self.assertEqual(temp.stats.endtime, UTCDateTime(7))
        self.assertEqual(temp.stats.npts, 6)
        self.assertEqual(temp2.stats.npts, 7)
        # self.assertEqual(temp.stats, temp2.stats)
        np.testing.assert_array_equal(temp.data, temp2.data[:-1])
        # Create test array that allows for easy testing.
        # Check if the data is the same.
        self.assertNotEqual(temp.data.ctypes.data, tr.data[2:9].ctypes.data)
        np.testing.assert_array_equal(tr.data[2:8], temp.data)
        # Using out of bounds times should not do anything but create
        # a copy of the stats.
        temp = deepcopy(tr)
        temp.trim(st - 2.5, st + 200)
        # The start and end times should not change.
        self.assertEqual(temp.stats.starttime, UTCDateTime(0))
        self.assertEqual(temp.stats.endtime, UTCDateTime(10))
        self.assertEqual(temp.stats.npts, 11)
        # Alter the new stats to make sure the old one stays intact.
        temp.stats.starttime = UTCDateTime(1000)
        self.assertEqual(org_stats, tr.stats)
        # Check if the data address is not the same, that is it is a copy
        self.assertNotEqual(temp.data.ctypes.data, tr.data.ctypes.data)
        np.testing.assert_array_equal(tr.data, temp.data)
        # Make sure the original Trace object did not change.
        np.testing.assert_array_equal(tr.data, org_data)
        self.assertEqual(tr.data.ctypes.data, mem_pos)
        self.assertEqual(tr.stats, org_stats)
        # Use more complicated times and sampling rate.
        tr = Trace(data=np.arange(111))
        tr.stats.starttime = UTCDateTime(111.11111)
        tr.stats.sampling_rate = 50.0
        org_stats = deepcopy(tr.stats)
        org_data = deepcopy(tr.data)
        # Save memory position of array.
        mem_pos = tr.data.ctypes.data
        # Create temp trace object used for testing.
        temp = deepcopy(tr)
        temp.trim(UTCDateTime(111.22222), UTCDateTime(112.99999),
                  nearest_sample=False)
        # Should again be identical. XXX NOT!
        temp2 = deepcopy(tr)
        temp2.trim(UTCDateTime(111.21111), UTCDateTime(113.01111),
                   nearest_sample=False)
        np.testing.assert_array_equal(temp.data, temp2.data[1:-1])
        # Check stuff.
        self.assertEqual(temp.stats.starttime, UTCDateTime(111.23111))
        self.assertEqual(temp.stats.endtime, UTCDateTime(112.991110))
        # Check if the data is the same.
        temp = deepcopy(tr)
        temp.trim(UTCDateTime(0), UTCDateTime(1000 * 1000))
        self.assertNotEqual(temp.data.ctypes.data, tr.data.ctypes.data)
        # starttime must be in conformance with sampling rate
        t = UTCDateTime(111.11111)
        self.assertEqual(temp.stats.starttime, t)
        delta = int((tr.stats.starttime - t) * tr.stats.sampling_rate + .5)
        np.testing.assert_array_equal(tr.data, temp.data[delta:delta + 111])
        # Make sure the original Trace object did not change.
        np.testing.assert_array_equal(tr.data, org_data)
        self.assertEqual(tr.data.ctypes.data, mem_pos)
        self.assertEqual(tr.stats, org_stats)

    def test_trimFloatingPointWithPadding1(self):
        """
        Tests the slicing of trace objects with the use of the padding option.
        """
        # Create test array that allows for easy testing.
        tr = Trace(data=np.arange(11))
        org_stats = deepcopy(tr.stats)
        org_data = deepcopy(tr.data)
        # Save memory position of array.
        mem_pos = tr.data.ctypes.data
        # Just some sanity tests.
        self.assertEqual(tr.stats.starttime, UTCDateTime(0))
        self.assertEqual(tr.stats.endtime, UTCDateTime(10))
        # Create temp trace object used for testing.
        st = tr.stats.starttime
        # Using out of bounds times should not do anything but create
        # a copy of the stats.
        temp = deepcopy(tr)
        temp.trim(st - 2.5, st + 200, pad=True)
        self.assertEqual(temp.stats.starttime.timestamp, -2.0)
        self.assertEqual(temp.stats.endtime.timestamp, 200)
        self.assertEqual(temp.stats.npts, 203)
        mask = np.zeros(203).astype(np.bool_)
        mask[:2] = True
        mask[13:] = True
        np.testing.assert_array_equal(temp.data.mask, mask)
        # Alter the new stats to make sure the old one stays intact.
        temp.stats.starttime = UTCDateTime(1000)
        self.assertEqual(org_stats, tr.stats)
        # Check if the data address is not the same, that is it is a copy
        self.assertNotEqual(temp.data.ctypes.data, tr.data.ctypes.data)
        np.testing.assert_array_equal(tr.data, temp.data[2:13])
        # Make sure the original Trace object did not change.
        np.testing.assert_array_equal(tr.data, org_data)
        self.assertEqual(tr.data.ctypes.data, mem_pos)
        self.assertEqual(tr.stats, org_stats)

    def test_trimFloatingPointWithPadding2(self):
        """
        Use more complicated times and sampling rate.
        """
        tr = Trace(data=np.arange(111))
        tr.stats.starttime = UTCDateTime(111.11111)
        tr.stats.sampling_rate = 50.0
        org_stats = deepcopy(tr.stats)
        org_data = deepcopy(tr.data)
        # Save memory position of array.
        mem_pos = tr.data.ctypes.data
        # Create temp trace object used for testing.
        temp = deepcopy(tr)
        temp.trim(UTCDateTime(111.22222), UTCDateTime(112.99999),
                  nearest_sample=False)
        # Should again be identical.#XXX not
        temp2 = deepcopy(tr)
        temp2.trim(UTCDateTime(111.21111), UTCDateTime(113.01111),
                   nearest_sample=False)
        np.testing.assert_array_equal(temp.data, temp2.data[1:-1])
        # Check stuff.
        self.assertEqual(temp.stats.starttime, UTCDateTime(111.23111))
        self.assertEqual(temp.stats.endtime, UTCDateTime(112.991110))
        # Check if the data is the same.
        temp = deepcopy(tr)
        temp.trim(UTCDateTime(0), UTCDateTime(1000 * 1000), pad=True)
        self.assertNotEqual(temp.data.ctypes.data, tr.data.ctypes.data)
        # starttime must be in conformance with sampling rate
        t = UTCDateTime(1969, 12, 31, 23, 59, 59, 991110)
        self.assertEqual(temp.stats.starttime, t)
        delta = int((tr.stats.starttime - t) * tr.stats.sampling_rate + .5)
        np.testing.assert_array_equal(tr.data, temp.data[delta:delta + 111])
        # Make sure the original Trace object did not change.
        np.testing.assert_array_equal(tr.data, org_data)
        self.assertEqual(tr.data.ctypes.data, mem_pos)
        self.assertEqual(tr.stats, org_stats)

    def test_add_sanity(self):
        """
        Test sanity checks in __add__ method of the Trace object.
        """
        tr = Trace(data=np.arange(10))
        # you may only add a Trace object
        self.assertRaises(TypeError, tr.__add__, 1234)
        self.assertRaises(TypeError, tr.__add__, '1234')
        self.assertRaises(TypeError, tr.__add__, [1, 2, 3, 4])
        # trace id
        tr2 = Trace()
        tr2.stats.station = 'TEST'
        self.assertRaises(TypeError, tr.__add__, tr2)
        # sample rate
        tr2 = Trace()
        tr2.stats.sampling_rate = 20
        self.assertRaises(TypeError, tr.__add__, tr2)
        # calibration factor
        tr2 = Trace()
        tr2.stats.calib = 20
        self.assertRaises(TypeError, tr.__add__, tr2)
        # data type
        tr2 = Trace()
        tr2.data = np.arange(10, dtype=np.float32)
        self.assertRaises(TypeError, tr.__add__, tr2)

    def test_addOverlapsDefaultMethod(self):
        """
        Test __add__ method of the Trace object.
        """
        # 1
        # overlapping trace with differing data
        # Trace 1: 0000000
        # Trace 2:      1111111
        tr1 = Trace(data=np.zeros(7))
        tr2 = Trace(data=np.ones(7))
        tr2.stats.starttime = tr1.stats.starttime + 5
        # 1 + 2  : 00000--11111
        tr = tr1 + tr2
        self.assertTrue(isinstance(tr.data, np.ma.masked_array))
        self.assertEqual(tr.data.tolist(),
                         [0, 0, 0, 0, 0, None, None, 1, 1, 1, 1, 1])
        # 2 + 1  : 00000--11111
        tr = tr2 + tr1
        self.assertTrue(isinstance(tr.data, np.ma.masked_array))
        self.assertEqual(tr.data.tolist(),
                         [0, 0, 0, 0, 0, None, None, 1, 1, 1, 1, 1])
        # 2
        # overlapping trace with same data
        # Trace 1: 0000000
        # Trace 2:      0000000
        tr1 = Trace(data=np.zeros(7))
        tr2 = Trace(data=np.zeros(7))
        tr2.stats.starttime = tr1.stats.starttime + 5
        # 1 + 2  : 000000000000
        tr = tr1 + tr2
        self.assertTrue(isinstance(tr.data, np.ndarray))
        np.testing.assert_array_equal(tr.data, np.zeros(12))
        # 2 + 1  : 000000000000
        tr = tr2 + tr1
        self.assertTrue(isinstance(tr.data, np.ndarray))
        np.testing.assert_array_equal(tr.data, np.zeros(12))
        # 3
        # contained trace with same data
        # Trace 1: 1111111111
        # Trace 2:      11
        tr1 = Trace(data=np.ones(10))
        tr2 = Trace(data=np.ones(2))
        tr2.stats.starttime = tr1.stats.starttime + 5
        # 1 + 2  : 1111111111
        tr = tr1 + tr2
        self.assertTrue(isinstance(tr.data, np.ndarray))
        np.testing.assert_array_equal(tr.data, np.ones(10))
        # 2 + 1  : 1111111111
        tr = tr2 + tr1
        self.assertTrue(isinstance(tr.data, np.ndarray))
        np.testing.assert_array_equal(tr.data, np.ones(10))
        # 4
        # contained trace with differing data
        # Trace 1: 0000000000
        # Trace 2:      11
        tr1 = Trace(data=np.zeros(10))
        tr2 = Trace(data=np.ones(2))
        tr2.stats.starttime = tr1.stats.starttime + 5
        # 1 + 2  : 00000--000
        tr = tr1 + tr2
        self.assertTrue(isinstance(tr.data, np.ma.masked_array))
        self.assertEqual(tr.data.tolist(),
                         [0, 0, 0, 0, 0, None, None, 0, 0, 0])
        # 2 + 1  : 00000--000
        tr = tr2 + tr1
        self.assertTrue(isinstance(tr.data, np.ma.masked_array))
        self.assertEqual(tr.data.tolist(),
                         [0, 0, 0, 0, 0, None, None, 0, 0, 0])
        # 5
        # completely contained trace with same data until end
        # Trace 1: 1111111111
        # Trace 2: 1111111111
        tr1 = Trace(data=np.ones(10))
        tr2 = Trace(data=np.ones(10))
        # 1 + 2  : 1111111111
        tr = tr1 + tr2
        self.assertTrue(isinstance(tr.data, np.ndarray))
        np.testing.assert_array_equal(tr.data, np.ones(10))
        # 6
        # completely contained trace with differing data
        # Trace 1: 0000000000
        # Trace 2: 1111111111
        tr1 = Trace(data=np.zeros(10))
        tr2 = Trace(data=np.ones(10))
        # 1 + 2  : ----------
        tr = tr1 + tr2
        self.assertTrue(isinstance(tr.data, np.ma.masked_array))
        self.assertEqual(tr.data.tolist(), [None] * 10)

    def test_addWithDifferentSamplingRates(self):
        """
        Test __add__ method of the Trace object.
        """
        # 1 - different sampling rates for the same channel should fail
        tr1 = Trace(data=np.zeros(5))
        tr1.stats.sampling_rate = 200
        tr2 = Trace(data=np.zeros(5))
        tr2.stats.sampling_rate = 50
        self.assertRaises(TypeError, tr1.__add__, tr2)
        self.assertRaises(TypeError, tr2.__add__, tr1)
        # 2 - different sampling rates for the different channels works
        tr1 = Trace(data=np.zeros(5))
        tr1.stats.sampling_rate = 200
        tr1.stats.channel = 'EHE'
        tr2 = Trace(data=np.zeros(5))
        tr2.stats.sampling_rate = 50
        tr2.stats.channel = 'EHZ'
        tr3 = Trace(data=np.zeros(5))
        tr3.stats.sampling_rate = 200
        tr3.stats.channel = 'EHE'
        tr4 = Trace(data=np.zeros(5))
        tr4.stats.sampling_rate = 50
        tr4.stats.channel = 'EHZ'
        # same sampling rate and ids should not fail
        tr1 + tr3
        tr3 + tr1
        tr2 + tr4
        tr4 + tr2

    def test_addWithDifferentDatatypesOrID(self):
        """
        Test __add__ method of the Trace object.
        """
        # 1 - different data types for the same channel should fail
        tr1 = Trace(data=np.zeros(5, dtype=np.int32))
        tr2 = Trace(data=np.zeros(5, dtype=np.float32))
        self.assertRaises(TypeError, tr1.__add__, tr2)
        self.assertRaises(TypeError, tr2.__add__, tr1)
        # 2 - different sampling rates for the different channels works
        tr1 = Trace(data=np.zeros(5, dtype=np.int32))
        tr1.stats.channel = 'EHE'
        tr2 = Trace(data=np.zeros(5, dtype=np.float32))
        tr2.stats.channel = 'EHZ'
        tr3 = Trace(data=np.zeros(5, dtype=np.int32))
        tr3.stats.channel = 'EHE'
        tr4 = Trace(data=np.zeros(5, dtype=np.float32))
        tr4.stats.channel = 'EHZ'
        # same data types and ids should not fail
        tr1 + tr3
        tr3 + tr1
        tr2 + tr4
        tr4 + tr2
        # adding traces with different ids should raise
        self.assertRaises(TypeError, tr1.__add__, tr2)
        self.assertRaises(TypeError, tr3.__add__, tr4)
        self.assertRaises(TypeError, tr2.__add__, tr1)
        self.assertRaises(TypeError, tr4.__add__, tr3)

    def test_comparisons(self):
        """
        Tests all rich comparison operators (==, !=, <, <=, >, >=)
        The latter four are not implemented due to ambiguous meaning and bounce
        an error.
        """
        # create test traces
        tr0 = Trace(np.arange(3))
        tr1 = Trace(np.arange(3))
        tr2 = Trace(np.arange(3), {'station': 'X'})
        tr3 = Trace(np.arange(3), {'processing':
                                   ["filter:lowpass:{'freq': 10}"]})
        tr4 = Trace(np.arange(5))
        tr5 = Trace(np.arange(5), {'station': 'X'})
        tr6 = Trace(np.arange(5), {'processing':
                                   ["filter:lowpass:{'freq': 10}"]})
        tr7 = Trace(np.array([1, 1, 1]))
        # tests that should raise a NotImplementedError (i.e. <=, <, >=, >)
        self.assertRaises(NotImplementedError, tr1.__lt__, tr1)
        self.assertRaises(NotImplementedError, tr1.__le__, tr1)
        self.assertRaises(NotImplementedError, tr1.__gt__, tr1)
        self.assertRaises(NotImplementedError, tr1.__ge__, tr1)
        self.assertRaises(NotImplementedError, tr1.__lt__, tr2)
        self.assertRaises(NotImplementedError, tr1.__le__, tr2)
        self.assertRaises(NotImplementedError, tr1.__gt__, tr2)
        self.assertRaises(NotImplementedError, tr1.__ge__, tr2)
        # normal tests
        self.assertEqual(tr0 == tr0, True)
        self.assertEqual(tr0 == tr1, True)
        self.assertEqual(tr0 == tr2, False)
        self.assertEqual(tr0 == tr3, False)
        self.assertEqual(tr0 == tr4, False)
        self.assertEqual(tr0 == tr5, False)
        self.assertEqual(tr0 == tr6, False)
        self.assertEqual(tr0 == tr7, False)
        self.assertEqual(tr5 == tr0, False)
        self.assertEqual(tr5 == tr1, False)
        self.assertEqual(tr5 == tr2, False)
        self.assertEqual(tr5 == tr3, False)
        self.assertEqual(tr5 == tr4, False)
        self.assertEqual(tr5 == tr5, True)
        self.assertEqual(tr5 == tr6, False)
        self.assertEqual(tr3 == tr6, False)
        self.assertEqual(tr0 != tr0, False)
        self.assertEqual(tr0 != tr1, False)
        self.assertEqual(tr0 != tr2, True)
        self.assertEqual(tr0 != tr3, True)
        self.assertEqual(tr0 != tr4, True)
        self.assertEqual(tr0 != tr5, True)
        self.assertEqual(tr0 != tr6, True)
        self.assertEqual(tr0 != tr7, True)
        self.assertEqual(tr5 != tr0, True)
        self.assertEqual(tr5 != tr1, True)
        self.assertEqual(tr5 != tr2, True)
        self.assertEqual(tr5 != tr3, True)
        self.assertEqual(tr5 != tr4, True)
        self.assertEqual(tr5 != tr5, False)
        self.assertEqual(tr5 != tr6, True)
        self.assertEqual(tr3 != tr6, True)
        # some weirder tests against non-Trace objects
        for object in [0, 1, 0.0, 1.0, "", "test", True, False, [], [tr0],
                       set(), set(tr0), {}, {"test": "test"}, [], None, ]:
            self.assertEqual(tr0 == object, False)
            self.assertEqual(tr0 != object, True)

    def test_nearestSample(self):
        """
        This test case shows that the libmseed is actually flooring the
        starttime to the next sample value, regardless if it is the nearest
        sample. The flag nearest_sample=True tries to avoids this and
        rounds it to the next actual possible sample point.
        """
        # set up
        trace = Trace(data=np.empty(10000))
        trace.stats.starttime = UTCDateTime("2010-06-20T20:19:40.000000Z")
        trace.stats.sampling_rate = 200.0
        # ltrim
        tr = deepcopy(trace)
        t = UTCDateTime("2010-06-20T20:19:51.494999Z")
        tr._ltrim(t - 3, nearest_sample=True)
        # see that it is actually rounded to the next sample point
        self.assertEqual(tr.stats.starttime,
                         UTCDateTime("2010-06-20T20:19:48.495000Z"))
        # Lots of tests follow that thoroughly check the cutting behavior
        # using nearest_sample=True/False
        # rtrim
        tr = deepcopy(trace)
        t = UTCDateTime("2010-06-20T20:19:51.494999Z")
        tr._rtrim(t + 7, nearest_sample=True)
        # see that it is actually rounded to the next sample point
        self.assertEqual(tr.stats.endtime,
                         UTCDateTime("2010-06-20T20:19:58.495000Z"))
        tr = deepcopy(trace)
        t = UTCDateTime("2010-06-20T20:19:51.495000Z")
        tr._rtrim(t + 7, nearest_sample=True)
        # see that it is actually rounded to the next sample point
        self.assertEqual(tr.stats.endtime,
                         UTCDateTime("2010-06-20T20:19:58.495000Z"))
        tr = deepcopy(trace)
        t = UTCDateTime("2010-06-20T20:19:51.495111Z")
        tr._rtrim(t + 7, nearest_sample=True)
        # see that it is actually rounded to the next sample point
        self.assertEqual(tr.stats.endtime,
                         UTCDateTime("2010-06-20T20:19:58.495000Z"))
        tr = deepcopy(trace)
        t = UTCDateTime("2010-06-20T20:19:51.497501Z")
        tr._rtrim(t + 7, nearest_sample=True)
        # see that it is actually rounded to the next sample point
        self.assertEqual(tr.stats.endtime,
                         UTCDateTime("2010-06-20T20:19:58.500000Z"))
        # rtrim
        tr = deepcopy(trace)
        t = UTCDateTime("2010-06-20T20:19:51.494999Z")
        tr._rtrim(t + 7, nearest_sample=False)
        # see that it is actually rounded to the next sample point
        self.assertEqual(tr.stats.endtime,
                         UTCDateTime("2010-06-20T20:19:58.490000Z"))
        tr = deepcopy(trace)
        t = UTCDateTime("2010-06-20T20:19:51.495000Z")
        tr._rtrim(t + 7, nearest_sample=False)
        # see that it is actually rounded to the next sample point
        self.assertEqual(tr.stats.endtime,
                         UTCDateTime("2010-06-20T20:19:58.495000Z"))
        tr = deepcopy(trace)
        t = UTCDateTime("2010-06-20T20:19:51.495111Z")
        tr._rtrim(t + 7, nearest_sample=False)
        # see that it is actually rounded to the next sample point
        self.assertEqual(tr.stats.endtime,
                         UTCDateTime("2010-06-20T20:19:58.495000Z"))
        tr = deepcopy(trace)
        t = UTCDateTime("2010-06-20T20:19:51.497500Z")
        tr._rtrim(t + 7, nearest_sample=False)
        # see that it is actually rounded to the next sample point
        self.assertEqual(tr.stats.endtime,
                         UTCDateTime("2010-06-20T20:19:58.495000Z"))

    def test_maskedArrayToString(self):
        """
        Masked arrays should be marked using __str__.
        """
        st = read()
        overlaptrace = st[0].copy()
        overlaptrace.stats.starttime += 1
        st.append(overlaptrace)
        st.merge()
        out = st[0].__str__()
        self.assertTrue(out.endswith('(masked)'))

    def test_detrend(self):
        """
        Test detrend method of trace
        """
        t = np.arange(10)
        data = 0.1 * t + 1.
        tr = Trace(data=data.copy())

        tr.detrend(type='simple')
        np.testing.assert_array_almost_equal(tr.data, np.zeros(10))

        tr.data = data.copy()
        tr.detrend(type='linear')
        np.testing.assert_array_almost_equal(tr.data, np.zeros(10))

        data = np.zeros(10)
        data[3:7] = 1.

        tr.data = data.copy()
        tr.detrend(type='simple')
        np.testing.assert_almost_equal(tr.data[0], 0.)
        np.testing.assert_almost_equal(tr.data[-1], 0.)

        tr.data = data.copy()
        tr.detrend(type='linear')
        np.testing.assert_almost_equal(tr.data[0], -0.4)
        np.testing.assert_almost_equal(tr.data[-1], -0.4)

    def test_differentiate(self):
        """
        Test differentiation method of trace
        """
        t = np.linspace(0., 1., 11)
        data = 0.1 * t + 1.
        tr = Trace(data=data)
        tr.stats.delta = 0.1
        tr.differentiate(method='gradient')
        np.testing.assert_array_almost_equal(tr.data, np.ones(11) * 0.1)

    def test_integrate(self):
        """
        Test integration method of trace
        """
        data = np.ones(101) * 0.01
        tr = Trace(data=data)
        tr.stats.delta = 0.1
        tr.integrate()
        # Assert time and length of resulting array.
        self.assertEqual(tr.stats.starttime, UTCDateTime(0))
        self.assertEqual(tr.stats.npts, 101)
        np.testing.assert_array_almost_equal(
            tr.data, np.concatenate([[0.0], np.cumsum(data)[:-1] * 0.1]))

    def test_issue317(self):
        """
        Tests times after breaking a stream into parts and merging it again.
        """
        # create a sample trace
        org_trace = Trace(data=np.arange(22487))
        org_trace.stats.starttime = UTCDateTime()
        org_trace.stats.sampling_rate = 0.999998927116
        num_pakets = 10
        # break org_trace into set of contiguous packet data
        traces = []
        packet_length = int(np.size(org_trace.data) / num_pakets)
        delta_time = org_trace.stats.delta
        tstart = org_trace.stats.starttime
        tend = tstart + delta_time * float(packet_length - 1)
        for i in range(num_pakets):
            tr = Trace(org_trace.data, org_trace.stats)
            tr = tr.slice(tstart, tend)
            traces.append(tr)
            tstart = tr.stats.endtime + delta_time
            tend = tstart + delta_time * float(packet_length - 1)
        # reconstruct original trace by adding together packet traces
        sum_trace = traces[0].copy()
        npts = traces[0].stats.npts
        for i in range(1, len(traces)):
            sum_trace = sum_trace.__add__(traces[i].copy(), method=0,
                                          interpolation_samples=0,
                                          fill_value='latest',
                                          sanity_checks=True)
            # check npts
            self.assertEqual(traces[i].stats.npts, npts)
            self.assertEqual(sum_trace.stats.npts, (i + 1) * npts)
            # check data
            np.testing.assert_array_equal(traces[i].data,
                                          np.arange(i * npts, (i + 1) * npts))
            np.testing.assert_array_equal(sum_trace.data,
                                          np.arange(0, (i + 1) * npts))
            # check delta
            self.assertEqual(traces[i].stats.delta, org_trace.stats.delta)
            self.assertEqual(sum_trace.stats.delta, org_trace.stats.delta)
            # check sampling rates
            self.assertAlmostEqual(traces[i].stats.sampling_rate,
                                   org_trace.stats.sampling_rate)
            self.assertAlmostEqual(sum_trace.stats.sampling_rate,
                                   org_trace.stats.sampling_rate)
            # check end times
            self.assertEqual(traces[i].stats.endtime, sum_trace.stats.endtime)

    def test_verify(self):
        """
        Tests verify method.
        """
        # empty Trace
        tr = Trace()
        tr.verify()
        # Trace with a single sample (issue #357)
        tr = Trace(data=np.array([1]))
        tr.verify()
        # example Trace
        tr = read()[0]
        tr.verify()

    def test_percent_in_str(self):
        """
        Tests if __str__ method is working with percent sign (%).
        """
        tr = Trace()
        tr.stats.station = '%t3u'
        self.assertTrue(tr.__str__().startswith(".%t3u.. | 1970"))

    def test_taper(self):
        """
        Test taper method of trace
        """
        data = np.ones(10)
        tr = Trace(data=data)
        tr.taper(max_percentage=0.05, type='cosine')
        for i in range(len(data)):
            self.assertLessEqual(tr.data[i], 1.)
            self.assertGreaterEqual(tr.data[i], 0.)

    def test_taper_onesided(self):
        """
        Test onesided taper method of trace
        """
        data = np.ones(11)
        tr = Trace(data=data)
        tr.taper(max_percentage=None, side="left")
        self.assertTrue(tr.data[:5].sum() < 5.)
        self.assertEqual(tr.data[6:].sum(), 5.)

        data = np.ones(11)
        tr = Trace(data=data)
        tr.taper(max_percentage=None, side="right")
        self.assertEqual(tr.data[:5].sum(), 5.)
        self.assertTrue(tr.data[6:].sum() < 5.)

    def test_taper_length(self):
        npts = 11
        type_ = "hann"

        data = np.ones(npts)
        tr = Trace(data=data, header={'sampling': 1.})
        # test an overlong taper request, should still work
        tr.taper(max_percentage=0.7, max_length=int(npts / 2) + 1)

        data = np.ones(npts)
        tr = Trace(data=data, header={'sampling': 1.})
        # first 3 samples get tapered
        tr.taper(max_percentage=None, type=type_, side="left", max_length=3)
        # last 5 samples get tapered
        tr.taper(max_percentage=0.5, type=type_, side="right", max_length=None)
        self.assertTrue(np.all(tr.data[:3] < 1.))
        self.assertTrue(np.all(tr.data[3:6] == 1.))
        self.assertTrue(np.all(tr.data[6:] < 1.))

        data = np.ones(npts)
        tr = Trace(data=data, header={'sampling': 1.})
        # first 3 samples get tapered
        tr.taper(max_percentage=0.5, type=type_, side="left", max_length=3)
        # last 3 samples get tapered
        tr.taper(max_percentage=0.3, type=type_, side="right", max_length=5)
        self.assertTrue(np.all(tr.data[:3] < 1.))
        self.assertTrue(np.all(tr.data[3:8] == 1.))
        self.assertTrue(np.all(tr.data[8:] < 1.))

    def test_times(self):
        """
        Test if the correct times array is returned for normal traces and
        traces with gaps.
        """
        tr = Trace(data=np.ones(100))
        tr.stats.sampling_rate = 20
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr.stats.starttime = start
        tm = tr.times()
        self.assertAlmostEqual(tm[-1], tr.stats.endtime - tr.stats.starttime)
        tr.data = np.ma.ones(100)
        tr.data[30:40] = np.ma.masked
        tm = tr.times()
        self.assertTrue(np.alltrue(tr.data.mask == tm.mask))

    def test_modulo_operation(self):
        """
        Method for testing the modulo operation. Mainly tests part not covered
        by the doctests.
        """
        tr = Trace(data=np.arange(25))
        # Wrong type raises.
        self.assertRaises(TypeError, tr.__mod__, 5.0)
        self.assertRaises(TypeError, tr.__mod__, "123")
        # Needs to be a positive integer.
        self.assertRaises(ValueError, tr.__mod__, 0)
        self.assertRaises(ValueError, tr.__mod__, -11)
        # If num is more then the number of samples, a copy will be returned.
        st = tr % 500
        self.assertEqual(tr, st[0])
        self.assertEqual(len(st), 1)
        self.assertFalse(tr.data is st[0].data)

    def test_plot(self):
        """
        Tests plot method if matplotlib is installed
        """
        tr = Trace(data=np.arange(25))
        tr.plot(show=False)

    def test_spectrogram(self):
        """
        Tests spectrogram method if matplotlib is installed
        """
        tr = Trace(data=np.arange(25))
        tr.stats.sampling_rate = 20
        tr.spectrogram(show=False)

    def test_raiseMasked(self):
        """
        Tests that detrend() raises in case of a masked array. (see #498)
        """
        x = np.arange(10)
        x = np.ma.masked_inside(x, 3, 4)
        tr = Trace(x)
        self.assertRaises(NotImplementedError, tr.detrend)

    def test_split(self):
        """
        Tests split method of the Trace class.
        """
        # set up
        tr1 = Trace(data=np.arange(1000))
        tr1.stats.sampling_rate = 200
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr2 = Trace(data=np.arange(0, 1000)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 10
        # add will create new trace with masked array
        trace = tr1 + tr2
        self.assertTrue(isinstance(trace.data, np.ma.masked_array))
        # split
        self.assertTrue(isinstance(trace, Trace))
        st = trace.split()
        self.assertTrue(isinstance(st, Stream))
        self.assertEqual(len(st[0]), 1000)
        self.assertEqual(len(st[1]), 1000)
        # check if have no masked arrays
        self.assertFalse(isinstance(st[0].data, np.ma.masked_array))
        self.assertFalse(isinstance(st[1].data, np.ma.masked_array))

    def test_simulate_evalresp(self):
        """
        Tests that trace.simulate calls evalresp with the correct network,
        station, location and channel information.
        """
        tr = read()[0]

        # Wrap in try/except as it of course will fail because the mocked
        # function returns None.
        try:
            with mock.patch("obspy.signal.invsim.evalresp") as patch:
                tr.simulate(seedresp={"filename": "RESP.dummy",
                                      "units": "VEL",
                                      "date": tr.stats.starttime})
        except:
            pass

        self.assertEqual(patch.call_count, 1)
        _, kwargs = patch.call_args

        # Make sure that every item of the trace is passed to the evalresp
        # function.
        for key in ["network", "station", "location", "channel"]:
            self.assertEqual(
                kwargs[key if key != "location" else "locid"], tr.stats[key],
                msg="'%s' did not get passed on to evalresp" % key)

    def test_issue540(self):
        """
        Trim with pad=True and given fill value should not return a masked
        NumPy array.
        """
        # fill_value = None
        tr = read()[0]
        self.assertEqual(len(tr), 3000)
        tr.trim(starttime=tr.stats.starttime - 0.01,
                endtime=tr.stats.endtime + 0.01, pad=True, fill_value=None)
        self.assertEqual(len(tr), 3002)
        self.assertTrue(isinstance(tr.data, np.ma.masked_array))
        self.assertIs(tr.data[0], np.ma.masked)
        self.assertTrue(tr.data[1] is not np.ma.masked)
        self.assertTrue(tr.data[-2] is not np.ma.masked)
        self.assertIs(tr.data[-1], np.ma.masked)
        # fill_value = 999
        tr = read()[0]
        self.assertEqual(len(tr), 3000)
        tr.trim(starttime=tr.stats.starttime - 0.01,
                endtime=tr.stats.endtime + 0.01, pad=True, fill_value=999)
        self.assertEqual(len(tr), 3002)
        self.assertFalse(isinstance(tr.data, np.ma.masked_array))
        self.assertEqual(tr.data[0], 999)
        self.assertEqual(tr.data[-1], 999)
        # given fill_value but actually no padding at all
        tr = read()[0]
        self.assertEqual(len(tr), 3000)
        tr.trim(starttime=tr.stats.starttime,
                endtime=tr.stats.endtime, pad=True, fill_value=-999)
        self.assertEqual(len(tr), 3000)
        self.assertFalse(isinstance(tr.data, np.ma.masked_array))

    def test_resample(self):
        """
        Tests the resampling of traces.
        """
        tr = read()[0]

        self.assertEqual(tr.stats.sampling_rate, 100.0)
        self.assertEqual(tr.stats.npts, 3000)

        tr_2 = tr.copy().resample(sampling_rate=50.0)
        self.assertEqual(tr_2.stats.endtime, tr.stats.endtime - 1.0 / 100.0)
        self.assertEqual(tr_2.stats.sampling_rate, 50.0)
        self.assertEqual(tr_2.stats.starttime, tr.stats.starttime)

        tr_3 = tr.copy().resample(sampling_rate=10.0)
        self.assertEqual(tr_3.stats.endtime, tr.stats.endtime - 9.0 / 100.0)
        self.assertEqual(tr_3.stats.sampling_rate, 10.0)
        self.assertEqual(tr_3.stats.starttime, tr.stats.starttime)

        tr_4 = tr.copy()
        tr_4.data = np.require(tr_4.data,
                               dtype=tr_4.data.dtype.newbyteorder('>'))
        tr_4 = tr_4.resample(sampling_rate=10.0)
        self.assertEqual(tr_4.stats.endtime, tr.stats.endtime - 9.0 / 100.0)
        self.assertEqual(tr_4.stats.sampling_rate, 10.0)
        self.assertEqual(tr_4.stats.starttime, tr.stats.starttime)

    def test_method_chaining(self):
        """
        Tests that method chaining works for all methods on the Trace object
        where it is sensible.
        """
        # This essentially just checks that the methods are chainable. The
        # methods are tested elsewhere and a full test would be a lot of work
        # with questionable return.
        tr = read()[0]
        temp_tr = tr.trim(tr.stats.starttime + 1)\
            .verify()\
            .filter("lowpass", freq=2.0)\
            .simulate(paz_remove={'poles': [-0.037004 + 0.037016j,
                                            -0.037004 - 0.037016j,
                                            -251.33 + 0j],
                                  'zeros': [0j, 0j],
                                  'gain': 60077000.0,
                                  'sensitivity': 2516778400.0})\
            .trigger(type="zdetect", nsta=20)\
            .decimate(factor=2, no_filter=True)\
            .resample(tr.stats.sampling_rate / 2.0)\
            .differentiate()\
            .integrate()\
            .detrend()\
            .taper(max_percentage=0.05, type='cosine')\
            .normalize()
        self.assertIs(temp_tr, tr)
        self.assertTrue(isinstance(tr, Trace))
        self.assertGreater(tr.stats.npts, 0)

        # Use the processing chain to check the results. The trim() methods
        # does not have an entry in the processing chain.
        pr = tr.stats.processing
        self.assertIn("trim", pr[0])
        self.assertTrue("filter" in pr[1] and "lowpass" in pr[1])
        self.assertIn("simulate", pr[2])
        self.assertIn("trigger", pr[3])
        self.assertIn("decimate", pr[4])
        self.assertIn("resample", pr[5])
        self.assertIn("differentiate", pr[6])
        self.assertIn("integrate", pr[7])
        self.assertIn("detrend", pr[8])
        self.assertIn("taper", pr[9])
        self.assertIn("normalize", pr[10])

    def test_skip_empty_trace(self):
        tr = read()[0]
        t = tr.stats.endtime + 10
        tr.trim(t, t + 10)
        tr.detrend()
        tr.resample(400)
        tr.differentiate()
        tr.integrate()
        tr.taper()

    def test_issue_695(self):
        x = np.zeros(12)
        data = [x.reshape((12, 1)),
                x.reshape((1, 12)),
                x.reshape((2, 6)),
                x.reshape((6, 2)),
                x.reshape((2, 2, 3)),
                x.reshape((1, 2, 2, 3)),
                x[0][()],  # 0-dim array
                ]
        for d in data:
            self.assertRaises(ValueError, Trace, data=d)

    def test_remove_response(self):
        """
        Test remove_response() method against simulate() with equivalent
        parameters to check response removal from Response object read from
        StationXML against pure evalresp providing an external RESP file.
        """
        tr1 = read()[0]
        tr2 = tr1.copy()
        # deconvolve from dataless with simulate() via Parser from
        # dataless/RESP
        parser = Parser("/path/to/dataless.seed.BW_RJOB")
        tr1.simulate(seedresp={"filename": parser, "units": "VEL"},
                     water_level=60, pre_filt=(0.1, 0.5, 30, 50), sacsim=True,
                     pitsasim=False)
        # deconvolve from StationXML with remove_response()
        tr2.remove_response(pre_filt=(0.1, 0.5, 30, 50))
        np.testing.assert_array_almost_equal(tr1.data, tr2.data)

    def test_remove_polynomial_response(self):
        """
        """
        from obspy import read_inventory
        path = os.path.dirname(__file__)

        # blockette 62, stage 0
        tr = read()[0]
        tr.stats.network = 'IU'
        tr.stats.station = 'ANTO'
        tr.stats.location = '30'
        tr.stats.channel = 'LDO'
        tr.stats.starttime = UTCDateTime("2010-07-23T00:00:00")
        # remove response
        del tr.stats.response
        filename = os.path.join(path, 'data', 'stationxml_IU.ANTO.30.LDO.xml')
        inv = read_inventory(filename, format='StationXML')
        tr.attach_response(inv)
        tr.remove_response()

        # blockette 62, stage 1 + blockette 58, stage 2
        tr = read()[0]
        tr.stats.network = 'BK'
        tr.stats.station = 'CMB'
        tr.stats.location = ''
        tr.stats.channel = 'LKS'
        tr.stats.starttime = UTCDateTime("2004-06-16T00:00:00")
        # remove response
        del tr.stats.response
        filename = os.path.join(path, 'data', 'stationxml_BK.CMB.__.LKS.xml')
        inv = read_inventory(filename, format='StationXML')
        tr.attach_response(inv)
        tr.remove_response()

    def test_processing_information(self):
        """
        Test case for the automatic processing information.
        """
        tr = read()[0]
        trimming_starttime = tr.stats.starttime + 1
        tr.trim(trimming_starttime)
        tr.filter("lowpass", freq=2.0)
        tr.simulate(paz_remove={
            'poles': [-0.037004 + 0.037016j, -0.037004 - 0.037016j,
                      -251.33 + 0j],
            'zeros': [0j, 0j],
            'gain': 60077000.0,
            'sensitivity': 2516778400.0})
        tr.trigger(type="zdetect", nsta=20)
        tr.decimate(factor=2, no_filter=True)
        tr.resample(tr.stats.sampling_rate / 2.0)
        tr.differentiate()
        tr.integrate()
        tr.detrend()
        tr.taper(max_percentage=0.05, type='cosine')
        tr.normalize()

        pr = tr.stats.processing

        self.assertIn("trim", pr[0])
        self.assertEqual(
            "ObsPy %s: trim(endtime=None::fill_value=None::"
            "nearest_sample=True::pad=False::starttime=%s)" % (
                __version__, str(trimming_starttime)),
            pr[0])
        self.assertIn("filter", pr[1])
        self.assertIn("simulate", pr[2])
        self.assertIn("trigger", pr[3])
        self.assertIn("decimate", pr[4])
        self.assertIn("resample", pr[5])
        self.assertIn("differentiate", pr[6])
        self.assertIn("integrate", pr[7])
        self.assertIn("detrend", pr[8])
        self.assertIn("taper", pr[9])
        self.assertIn("normalize", pr[10])

    def test_no_processing_info_for_failed_operations(self):
        """
        If an operation fails, no processing information should be attached
        to the Trace object.
        """
        # create test Trace
        tr = Trace(data=np.arange(20))
        self.assertFalse("processing" in tr.stats)
        # This decimation by a factor of 7 in this case would change the
        # end time of the time series. Therefore it fails.
        self.assertRaises(ValueError, tr.decimate, 7, strict_length=True)
        # No processing should be applied yet.
        self.assertFalse("processing" in tr.stats)

        # Test the same but this time with an already existing processing
        # information.
        tr = Trace(data=np.arange(20))
        tr.detrend()
        self.assertEqual(len(tr.stats.processing), 1)
        info = tr.stats.processing[0]

        self.assertRaises(ValueError, tr.decimate, 7, strict_length=True)
        self.assertEqual(tr.stats.processing, [info])

    def test_meta(self):
        """
        Tests Trace.meta an alternative to Trace.stats
        """
        tr = Trace()
        tr.meta = Stats({'network': 'NW'})
        self.assertEqual(tr.stats.network, 'NW')
        tr.stats = Stats({'network': 'BW'})
        self.assertEqual(tr.meta.network, 'BW')

    def test_interpolate(self):
        """
        Tests the interpolate function.

        This also tests the interpolation in obspy.signal. No need to repeat
        the same test twice I guess.
        """
        # Load the prepared data. The data has been created using SAC.
        file_ = "interpolation_test_random_waveform_delta_0.01_npts_50.sac"
        org_tr = read("/path/to/%s" % file_)[0]
        file_ = "interpolation_test_interpolated_delta_0.003.sac"
        interp_delta_0_003 = read("/path/to/%s" % file_)[0]
        file_ = "interpolation_test_interpolated_delta_0.077.sac"
        interp_delta_0_077 = read("/path/to/%s" % file_)[0]

        # Perform the same interpolation as in Python with ObsPy.
        int_tr = org_tr.copy().interpolate(sampling_rate=1.0 / 0.003,
                                           method="weighted_average_slopes")
        # Assert that the sampling rate has been set correctly.
        self.assertEqual(int_tr.stats.delta, 0.003)
        # Assert that the new end time is smaller than the old one. SAC at
        # times performs some extrapolation which we do not want to do here.
        self.assertLessEqual(int_tr.stats.endtime, org_tr.stats.endtime)
        # SAC extrapolates a bit which we don't want here. The deviations
        # to SAC are likely due to the fact that we use double precision
        # math while SAC uses single precision math.
        self.assertTrue(np.allclose(
            int_tr.data,
            interp_delta_0_003.data[:int_tr.stats.npts],
            rtol=1E-3))

        int_tr = org_tr.copy().interpolate(sampling_rate=1.0 / 0.077,
                                           method="weighted_average_slopes")
        # Assert that the sampling rate has been set correctly.
        self.assertEqual(int_tr.stats.delta, 0.077)
        # Assert that the new end time is smaller than the old one. SAC
        # calculates one sample less in this case.
        self.assertLessEqual(int_tr.stats.endtime, org_tr.stats.endtime)
        self.assertTrue(np.allclose(
            int_tr.data[:interp_delta_0_077.stats.npts],
            interp_delta_0_077.data,
            rtol=1E-5))

        # Also test the other interpolation methods mainly by assuring the
        # correct SciPy function is called and everything stays internally
        # consistent. SciPy's functions are tested enough to be sure that
        # they work.
        for inter_type in ["linear", "nearest", "zero"]:
            with mock.patch("scipy.interpolate.interp1d") as patch:
                patch.return_value = lambda x: x
                org_tr.copy().interpolate(sampling_rate=0.5, method=inter_type)
            self.assertEqual(patch.call_count, 1)
            self.assertEqual(patch.call_args[1]["kind"], inter_type)

            int_tr = org_tr.copy().interpolate(sampling_rate=0.5,
                                               method=inter_type)
            self.assertEqual(int_tr.stats.delta, 2.0)
            self.assertLessEqual(int_tr.stats.endtime, org_tr.stats.endtime)

        for inter_type in ["slinear", "quadratic", "cubic", 1, 2, 3]:
            with mock.patch("scipy.interpolate.InterpolatedUnivariateSpline") \
                    as patch:
                patch.return_value = lambda x: x
                org_tr.copy().interpolate(sampling_rate=0.5, method=inter_type)
            s_map = {
                "slinear": 1,
                "quadratic": 2,
                "cubic": 3
            }
            if inter_type in s_map:
                inter_type = s_map[inter_type]
            self.assertEqual(patch.call_count, 1)
            self.assertEqual(patch.call_args[1]["k"], inter_type)

            int_tr = org_tr.copy().interpolate(sampling_rate=0.5,
                                               method=inter_type)
            self.assertEqual(int_tr.stats.delta, 2.0)
            self.assertLessEqual(int_tr.stats.endtime, org_tr.stats.endtime)

    def test_interpolation_time_shift(self):
        """
        Tests the time shift of the interpolation.
        """
        tr = read()[0]
        tr.stats.sampling_rate = 1.0
        tr.data = tr.data[:500]
        tr.interpolate(method="lanczos", sampling_rate=10.0, a=20)
        tr.stats.sampling_rate = 1.0
        tr.data = tr.data[:500]
        tr.stats.starttime = UTCDateTime(0)

        org_tr = tr.copy()

        # Now this does not do much for now but actually just shifts the
        # samples.
        tr.interpolate(method="lanczos", sampling_rate=1.0, a=1,
                       time_shift=0.2)
        self.assertEqual(tr.stats.starttime, org_tr.stats.starttime + 0.2)
        self.assertEqual(tr.stats.endtime, org_tr.stats.endtime + 0.2)
        np.testing.assert_allclose(tr.data, org_tr.data, atol=1E-9)

        tr.interpolate(method="lanczos", sampling_rate=1.0, a=1,
                       time_shift=0.4)
        self.assertEqual(tr.stats.starttime, org_tr.stats.starttime + 0.6)
        self.assertEqual(tr.stats.endtime, org_tr.stats.endtime + 0.6)
        np.testing.assert_allclose(tr.data, org_tr.data, atol=1E-9)

        tr.interpolate(method="lanczos", sampling_rate=1.0, a=1,
                       time_shift=-0.6)
        self.assertEqual(tr.stats.starttime, org_tr.stats.starttime)
        self.assertEqual(tr.stats.endtime, org_tr.stats.endtime)
        np.testing.assert_allclose(tr.data, org_tr.data, atol=1E-9)

        # This becomes more interesting when also fixing the sample
        # positions. Then one can shift by subsample accuracy while leaving
        # the sample positions intact. Note that there naturally are some
        # boundary effects and as the interpolation method does not deal
        # with any kind of extrapolation you will lose the first or last
        # samples.
        # This is a fairly extreme example but of course there are errors
        # when doing an interpolation - a shift using an FFT is more accurate.
        tr.interpolate(method="lanczos", sampling_rate=1.0, a=50,
                       starttime=tr.stats.starttime + tr.stats.delta,
                       time_shift=0.2)
        # The sample point did not change but we lost the first sample,
        # as we shifted towards the future.
        self.assertEqual(tr.stats.starttime, org_tr.stats.starttime + 1.0)
        self.assertEqual(tr.stats.endtime, org_tr.stats.endtime)
        # The data naturally also changed.
        self.assertRaises(AssertionError, np.testing.assert_allclose,
                          tr.data, org_tr.data[1:], atol=1E-9)
        # Shift back. This time we will lose the last sample.
        tr.interpolate(method="lanczos", sampling_rate=1.0, a=50,
                       starttime=tr.stats.starttime,
                       time_shift=-0.2)
        self.assertEqual(tr.stats.starttime, org_tr.stats.starttime + 1.0)
        self.assertEqual(tr.stats.endtime, org_tr.stats.endtime - 1.0)
        # But the data (aside from edge effects - we are going forward and
        # backwards again so they go twice as far!) should now again be the
        # same as we started out with.
        np.testing.assert_allclose(
            tr.data[100:-100], org_tr.data[101:-101], atol=1E-9, rtol=1E-4)

    def test_interpolation_arguments(self):
        """
        Test case for the interpolation arguments.
        """
        tr = read()[0]
        tr.stats.sampling_rate = 1.0
        tr.data = tr.data[:50]

        for inter_type in ["linear", "nearest", "zero", "slinear",
                           "quadratic", "cubic", 1, 2, 3,
                           "weighted_average_slopes"]:
            # If only the sampling rate is specified, the end time will be very
            # close to the original end time but never bigger.
            interp_tr = tr.copy().interpolate(sampling_rate=0.3,
                                              method=inter_type)
            self.assertEqual(tr.stats.starttime, interp_tr.stats.starttime)
            self.assertTrue(tr.stats.endtime >= interp_tr.stats.endtime >=
                            tr.stats.endtime - (1.0 / 0.3))

            # If the starttime is modified the new starttime will be used but
            # the end time will again be modified as little as possible.
            interp_tr = tr.copy().interpolate(sampling_rate=0.3,
                                              method=inter_type,
                                              starttime=tr.stats.starttime +
                                              5.0)
            self.assertEqual(tr.stats.starttime + 5.0,
                             interp_tr.stats.starttime)
            self.assertTrue(tr.stats.endtime >= interp_tr.stats.endtime >=
                            tr.stats.endtime - (1.0 / 0.3))

            # If npts is given it will be used to modify the end time.
            interp_tr = tr.copy().interpolate(sampling_rate=0.3,
                                              method=inter_type, npts=10)
            self.assertEqual(tr.stats.starttime,
                             interp_tr.stats.starttime)
            self.assertEqual(interp_tr.stats.npts, 10)

            # If npts and starttime are given, both will be modified.
            interp_tr = tr.copy().interpolate(sampling_rate=0.3,
                                              method=inter_type,
                                              starttime=tr.stats.starttime +
                                              5.0, npts=10)
            self.assertEqual(tr.stats.starttime + 5.0,
                             interp_tr.stats.starttime)
            self.assertEqual(interp_tr.stats.npts, 10)

            # An earlier starttime will raise an exception. No extrapolation
            # is supported
            self.assertRaises(ValueError, tr.copy().interpolate,
                              sampling_rate=1.0,
                              starttime=tr.stats.starttime - 10.0)
            # As will too many samples that would overstep the end time bound.
            self.assertRaises(ValueError, tr.copy().interpolate,
                              sampling_rate=1.0,
                              npts=tr.stats.npts * 1E6)

            # A negative or zero desired sampling rate should raise.
            self.assertRaises(ValueError, tr.copy().interpolate,
                              sampling_rate=0.0)
            self.assertRaises(ValueError, tr.copy().interpolate,
                              sampling_rate=-1.0)

    def test_resample_new(self):
        """
        Tests if Trace.resample works as expected and test that issue #857 is
        resolved.
        """
        starttime = UTCDateTime("1970-01-01T00:00:00.000000Z")
        tr0 = Trace(np.sin(np.linspace(0, 2*np.pi, 10)),
                    {'sampling_rate': 1.0,
                     'starttime': starttime})
        # downsample
        tr = tr0.copy()
        tr.resample(0.5, window='hanning', no_filter=True)
        self.assertEqual(len(tr.data), 5)
        expected = np.array([0.19478735, 0.83618307, 0.32200221,
                             -0.7794053, -0.57356732])
        self.assertTrue(np.all(np.abs(tr.data - expected) < 1e-7))
        self.assertEqual(tr.stats.sampling_rate, 0.5)
        self.assertEqual(tr.stats.delta, 2.0)
        self.assertEqual(tr.stats.npts, 5)
        self.assertEqual(tr.stats.starttime, starttime)
        self.assertEqual(tr.stats.endtime,
                         starttime + tr.stats.delta * (tr.stats.npts-1))

        # upsample
        tr = tr0.copy()
        tr.resample(2.0, window='hanning', no_filter=True)
        self.assertEqual(len(tr.data), 20)
        self.assertEqual(tr.stats.sampling_rate, 2.0)
        self.assertEqual(tr.stats.delta, 0.5)
        self.assertEqual(tr.stats.npts, 20)
        self.assertEqual(tr.stats.starttime, starttime)
        self.assertEqual(tr.stats.endtime,
                         starttime + tr.stats.delta * (tr.stats.npts-1))

        # downsample with non integer ratio
        tr = tr0.copy()
        tr.resample(0.75, window='hanning', no_filter=True)
        self.assertEqual(len(tr.data), int(10*.75))
        expected = np.array([0.15425413, 0.66991128, 0.74610418, 0.11960477,
                             -0.60644662, -0.77403839, -0.30938935])
        self.assertTrue(np.all(np.abs(tr.data - expected) < 1e-7))
        self.assertEqual(tr.stats.sampling_rate, 0.75)
        self.assertEqual(tr.stats.delta, 1/0.75)
        self.assertEqual(tr.stats.npts, int(10*.75))
        self.assertEqual(tr.stats.starttime, starttime)
        self.assertEqual(tr.stats.endtime,
                         starttime + tr.stats.delta * (tr.stats.npts-1))

        # downsample without window
        tr = tr0.copy()
        tr.resample(0.5, window=None, no_filter=True)
        self.assertEqual(len(tr.data), 5)
        self.assertEqual(tr.stats.sampling_rate, 0.5)
        self.assertEqual(tr.stats.delta, 2.0)
        self.assertEqual(tr.stats.npts, 5)
        self.assertEqual(tr.stats.starttime, starttime)
        self.assertEqual(tr.stats.endtime,
                         starttime + tr.stats.delta * (tr.stats.npts-1))

        # downsample with window and automatic filtering
        tr = tr0.copy()
        tr.resample(0.5, window='hanning', no_filter=False)
        self.assertEqual(len(tr.data), 5)
        self.assertEqual(tr.stats.sampling_rate, 0.5)
        self.assertEqual(tr.stats.delta, 2.0)
        self.assertEqual(tr.stats.npts, 5)
        self.assertEqual(tr.stats.starttime, starttime)
        self.assertEqual(tr.stats.endtime,
                         starttime + tr.stats.delta * (tr.stats.npts-1))

        # downsample with custom window
        tr = tr0.copy()
        window = np.ones((tr.stats.npts))
        tr.resample(0.5, window=window, no_filter=True)

        # downsample with bad window
        tr = tr0.copy()
        window = np.array([0, 1, 2, 3])
        self.assertRaises(ValueError, tr.resample,
                          sampling_rate=0.5, window=window, no_filter=True)

    def test_slide(self):
        """
        Tests for sliding a window across a trace object.
        """
        tr = Trace(data=np.linspace(0, 100, 101))
        tr.stats.starttime = UTCDateTime(0.0)
        tr.stats.sampling_rate = 5.0

        # First slice it in 4 pieces. Window length is in seconds.
        slices = []
        for window_tr in tr.slide(window_length=5.0, step=5.0):
            slices.append(window_tr)

        self.assertEqual(len(slices), 4)
        self.assertEqual(slices[0],
                         tr.slice(UTCDateTime(0), UTCDateTime(5)))
        self.assertEqual(slices[1],
                         tr.slice(UTCDateTime(5), UTCDateTime(10)))
        self.assertEqual(slices[2],
                         tr.slice(UTCDateTime(10), UTCDateTime(15)))
        self.assertEqual(slices[3],
                         tr.slice(UTCDateTime(15), UTCDateTime(20)))

        # Different step which is the distance between two windows measured
        # from the start of the first window in seconds.
        slices = []
        for window_tr in tr.slide(window_length=5.0, step=10.0):
            slices.append(window_tr)

        self.assertEqual(len(slices), 2)
        self.assertEqual(slices[0],
                         tr.slice(UTCDateTime(0), UTCDateTime(5)))
        self.assertEqual(slices[1],
                         tr.slice(UTCDateTime(10), UTCDateTime(15)))

        # Offset determines the initial starting point. It defaults to zero.
        slices = []
        for window_tr in tr.slide(window_length=5.0, step=6.5, offset=8.5):
            slices.append(window_tr)

        self.assertEqual(len(slices), 2)
        self.assertEqual(slices[0],
                         tr.slice(UTCDateTime(8.5), UTCDateTime(13.5)))
        self.assertEqual(slices[1],
                         tr.slice(UTCDateTime(15.0), UTCDateTime(20.0)))

        # By default only full length windows will be returned so any
        # remainder that can no longer make up a full window will not be
        # returned.
        slices = []
        for window_tr in tr.slide(window_length=15.0, step=15.0):
            slices.append(window_tr)

        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0],
                         tr.slice(UTCDateTime(0.0), UTCDateTime(15.0)))

        # But it can optionally be returned.
        slices = []
        for window_tr in tr.slide(window_length=15.0, step=15.0,
                                  include_partial_windows=True):
            slices.append(window_tr)

        self.assertEqual(len(slices), 2)
        self.assertEqual(slices[0],
                         tr.slice(UTCDateTime(0.0), UTCDateTime(15.0)))
        self.assertEqual(slices[1],
                         tr.slice(UTCDateTime(15.0), UTCDateTime(20.0)))

        # Negative step lengths work together with an offset.
        slices = []
        for window_tr in tr.slide(window_length=5.0, step=-5.0, offset=20.0):
            slices.append(window_tr)

        self.assertEqual(len(slices), 4)
        self.assertEqual(slices[0],
                         tr.slice(UTCDateTime(15), UTCDateTime(20)))
        self.assertEqual(slices[1],
                         tr.slice(UTCDateTime(10), UTCDateTime(15)))
        self.assertEqual(slices[2],
                         tr.slice(UTCDateTime(5), UTCDateTime(10)))
        self.assertEqual(slices[3],
                         tr.slice(UTCDateTime(0), UTCDateTime(5)))

    def test_slide_nearest_sample(self):
        """
        Tests that the nearest_sample argument is correctly passed to the
        slice function calls.
        """
        tr = Trace(data=np.linspace(0, 100, 101))
        tr.stats.starttime = UTCDateTime(0.0)
        tr.stats.sampling_rate = 5.0

        # It defaults to True.
        with mock.patch("obspy.core.trace.Trace.slice") as patch:
            patch.return_value = tr
            list(tr.slide(5, 5))

        self.assertEqual(patch.call_count, 4)
        for arg in patch.call_args_list:
            self.assertTrue(arg[1]["nearest_sample"])

        # Force True.
        with mock.patch("obspy.core.trace.Trace.slice") as patch:
            patch.return_value = tr
            list(tr.slide(5, 5, nearest_sample=True))

        self.assertEqual(patch.call_count, 4)
        for arg in patch.call_args_list:
            self.assertTrue(arg[1]["nearest_sample"])

        # Set to False.
        with mock.patch("obspy.core.trace.Trace.slice") as patch:
            patch.return_value = tr
            list(tr.slide(5, 5, nearest_sample=False))

        self.assertEqual(patch.call_count, 4)
        for arg in patch.call_args_list:
            self.assertFalse(arg[1]["nearest_sample"])

    def test_remove_response_plot(self):
        """
        Tests the plotting option of remove_response().
        """
        tr = read("/path/to/IU_ULN_00_LH1_2015-07-18T02.mseed")[0]
        inv = read_inventory("/path/to/IU_ULN_00_LH1.xml")
        tr.attach_response(inv)

        pre_filt = [0.001, 0.005, 10, 20]

        image_dir = os.path.join(os.path.dirname(__file__), 'images')
        with ImageComparison(image_dir, "trace_remove_response.png") as ic:
            tr.remove_response(pre_filt=pre_filt, output="DISP",
                               water_level=60, end_stage=None, plot=ic.name)


def suite():
    return unittest.makeSuite(TraceTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
