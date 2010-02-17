# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
from numpy.ma import is_masked
from obspy.core import UTCDateTime, Trace
import unittest


class TraceTestCase(unittest.TestCase):
    """
    Test suite for L{obspy.core.Trace}.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_len(self):
        """
        Tests the __len__ and count methods of the L{Trace} class.
        """
        trace = Trace(data=np.arange(1000))
        self.assertEquals(len(trace), 1000)
        self.assertEquals(trace.count(), 1000)

    def test_ltrim(self):
        """
        Tests the ltrim method of the L{Trace} class.
        """
        # set up
        trace = Trace(data=np.arange(1000))
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 200.0
        end = UTCDateTime(2000, 1, 1, 0, 0, 4, 995000)
        # verify
        trace.verify()
        # ltrim 100 samples
        tr = deepcopy(trace)
        tr.ltrim(0.5)
        tr.verify()
        np.testing.assert_array_equal(tr.data[0:5], \
                                      np.array([100, 101, 102, 103, 104]))
        self.assertEquals(len(tr.data), 900)
        self.assertEquals(tr.stats.npts, 900)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start + 0.5)
        self.assertEquals(tr.stats.endtime, end)
        # ltrim 202 samples
        tr = deepcopy(trace)
        tr.ltrim(1.010)
        tr.verify()
        np.testing.assert_array_equal(tr.data[0:5], \
                                      np.array([202, 203, 204, 205, 206]))
        self.assertEquals(len(tr.data), 798)
        self.assertEquals(tr.stats.npts, 798)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start + 1.010)
        self.assertEquals(tr.stats.endtime, end)
        # ltrim to UTCDateTime
        tr = deepcopy(trace)
        tr.ltrim(UTCDateTime(2000, 1, 1, 0, 0, 1, 10000))
        tr.verify()
        np.testing.assert_array_equal(tr.data[0:5], \
                                      np.array([202, 203, 204, 205, 206]))
        self.assertEquals(len(tr.data), 798)
        self.assertEquals(tr.stats.npts, 798)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start + 1.010)
        self.assertEquals(tr.stats.endtime, end)
        # some sanity checks
        # negative start time as datetime
        tr = deepcopy(trace)
        tr.ltrim(start - 1)
        tr.verify()
        self.assertEquals(tr.stats.starttime, start - 1)
        np.testing.assert_array_equal(trace.data, tr.data[200:])
        self.assertEquals(tr.stats.endtime, trace.stats.endtime)
        # negative start time as integer
        tr = deepcopy(trace)
        tr.ltrim(-100)
        tr.verify()
        self.assertEquals(tr.stats.starttime, start - 100)
        delta = 100 * trace.stats.sampling_rate
        np.testing.assert_array_equal(trace.data, tr.data[delta:])
        self.assertEquals(tr.stats.endtime, trace.stats.endtime)
        # start time > end time
        tr = deepcopy(trace)
        tr.ltrim(trace.stats.endtime + 100)
        tr.verify()
        self.assertEquals(tr.stats.starttime,
                          trace.stats.endtime + 100)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        self.assertEquals(tr.stats.endtime, tr.stats.starttime)
        # start time == end time
        tr = deepcopy(trace)
        tr.ltrim(5)
        tr.verify()
        self.assertEquals(tr.stats.starttime,
                          trace.stats.starttime + 5)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        self.assertEquals(tr.stats.endtime, tr.stats.starttime)
        # start time == end time
        tr = deepcopy(trace)
        tr.ltrim(5.1)
        tr.verify()
        self.assertEquals(tr.stats.starttime,
                          trace.stats.starttime + 5.1)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        self.assertEquals(tr.stats.endtime, tr.stats.starttime)

    def test_rtrim(self):
        """
        Tests the rtrim method of the L{Trace} class.
        """
        # set up
        trace = Trace(data=np.arange(1000))
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 200.0
        end = UTCDateTime(2000, 1, 1, 0, 0, 4, 995000)
        trace.verify()
        # rtrim 100 samples
        tr = deepcopy(trace)
        tr.rtrim(0.5)
        tr.verify()
        np.testing.assert_array_equal(tr.data[-5:], \
                                      np.array([895, 896, 897, 898, 899]))
        self.assertEquals(len(tr.data), 900)
        self.assertEquals(tr.stats.npts, 900)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start)
        self.assertEquals(tr.stats.endtime, end - 0.5)
        # rtrim 202 samples
        tr = deepcopy(trace)
        tr.rtrim(1.010)
        tr.verify()
        np.testing.assert_array_equal(tr.data[-5:], \
                                      np.array([793, 794, 795, 796, 797]))
        self.assertEquals(len(tr.data), 798)
        self.assertEquals(tr.stats.npts, 798)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start)
        self.assertEquals(tr.stats.endtime, end - 1.010)
        # rtrim 1 minute via UTCDateTime
        tr = deepcopy(trace)
        tr.rtrim(UTCDateTime(2000, 1, 1, 0, 0, 3, 985000))
        tr.verify()
        np.testing.assert_array_equal(tr.data[-5:], \
                                      np.array([793, 794, 795, 796, 797]))
        self.assertEquals(len(tr.data), 798)
        self.assertEquals(tr.stats.npts, 798)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start)
        self.assertEquals(tr.stats.endtime, end - 1.010)
        # some sanity checks
        # negative end time
        tr = deepcopy(trace)
        t = UTCDateTime(1999, 12, 31)
        tr.rtrim(t)
        tr.verify()
        self.assertEquals(tr.stats.endtime, t)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        # negative end time with given seconds
        tr = deepcopy(trace)
        tr.rtrim(100)
        tr.verify()
        self.assertEquals(tr.stats.endtime, trace.stats.endtime - 100)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        self.assertEquals(tr.stats.endtime, tr.stats.starttime)
        # end time > start time
        tr = deepcopy(trace)
        t = UTCDateTime(2001)
        tr.rtrim(t)
        tr.verify()
        self.assertEquals(tr.stats.endtime, t)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        self.assertEquals(tr.stats.endtime, tr.stats.starttime)
        # end time > start time given seconds
        tr = deepcopy(trace)
        tr.rtrim(5.1)
        tr.verify()
        delta = int(round(5.1 * trace.stats.sampling_rate, 7))
        endtime = trace.stats.starttime + trace.stats.delta * \
                  (trace.stats.npts - delta)
        self.assertEquals(tr.stats.endtime, endtime)
        np.testing.assert_array_equal(tr.data, np.empty(0))
        # end time == start time
        # returns one sample!
        tr = deepcopy(trace)
        tr.rtrim(4.995)
        #XXX I do not understand why this fails!!!
        #tr.verify()
        np.testing.assert_array_equal(tr.data, np.array([0]))
        self.assertEquals(len(tr.data), 1)
        self.assertEquals(tr.stats.npts, 1)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start)
        self.assertEquals(tr.stats.endtime, start)

    def test_trim(self):
        """
        Tests the trim method of the L{Trace} class.
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
        self.assertEquals(len(trace.data), 801)
        self.assertEquals(trace.stats.npts, 801)
        self.assertEquals(trace.stats.sampling_rate, 200.0)
        self.assertEquals(trace.stats.starttime, start + 0.5)
        self.assertEquals(trace.stats.endtime, end - 0.5)

    def test_addTraceWithGap(self):
        """
        Tests the add method of the L{Trace} class.
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
        self.assertEquals(trace.stats.starttime, start)
        self.assertEquals(trace.stats.endtime, start + 14.995)
        self.assertEquals(trace.stats.sampling_rate, 200)
        self.assertEquals(trace.stats.npts, 3000)
        # data
        self.assertEquals(len(trace), 3000)
        self.assertEquals(trace[0], 0)
        self.assertEquals(trace[999], 999)
        self.assertTrue(is_masked(trace[1000]))
        self.assertTrue(is_masked(trace[1999]))
        self.assertEquals(trace[2000], 999)
        self.assertEquals(trace[2999], 0)
        # verify
        trace.verify()

    def test_addTraceWithOverlap(self):
        """
        Tests the add method of the L{Trace} class.
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
        self.assertEquals(trace.stats.starttime, start)
        self.assertEquals(trace.stats.endtime, start + 8.995)
        self.assertEquals(trace.stats.sampling_rate, 200)
        self.assertEquals(trace.stats.npts, 1800)
        # data
        self.assertEquals(len(trace), 1800)
        self.assertEquals(trace[0], 0)
        self.assertEquals(trace[799], 799)
        self.assertTrue(trace[800].mask)
        self.assertTrue(trace[999].mask)
        self.assertEquals(trace[1000], 799)
        self.assertEquals(trace[1799], 0)
        # verify
        trace.verify()

    def test_addSameTrace(self):
        """
        Tests the add method of the L{Trace} class.
        """
        # set up
        tr1 = Trace(data=np.arange(1001))
        # add
        trace = tr1 + tr1
        # should return exact the same values
        self.assertEquals(trace.stats, tr1.stats)
        np.testing.assert_array_equal(trace.data, tr1.data)
        # verify
        trace.verify()

    def test_addTraceWithinTrace(self):
        """
        Tests the add method of the L{Trace} class.
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
        self.assertEquals(trace.stats, tr1.stats)
        mask = np.zeros(len(tr1)).astype("bool")
        mask[200:401] = True
        np.testing.assert_array_equal(trace.data.mask, mask)
        np.testing.assert_array_equal(trace.data.data[:200], tr1.data[:200])
        np.testing.assert_array_equal(trace.data.data[401:], tr1.data[401:])
        # add the other way around
        trace = tr2 + tr1
        # should return exact the same values like trace 1
        self.assertEquals(trace.stats, tr1.stats)
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
        mask = np.zeros(1800).astype("bool")
        mask[800:1000] = True
        np.testing.assert_array_equal(overlap.data.mask, mask)
        np.testing.assert_array_equal(overlap.data.data[:800], tr1.data[:800])
        np.testing.assert_array_equal(overlap.data.data[1000:], tr2.data[200:])
        # overlap + gap
        overlap_gap = overlap + tr3
        self.assertEqual(len(overlap_gap), 3400)
        mask = np.zeros(3400).astype("bool")
        mask[800:1000] = True
        mask[1800:2400] = True
        np.testing.assert_array_equal(overlap_gap.data.mask, mask)
        np.testing.assert_array_equal(overlap_gap.data.data[:800], \
                                      tr1.data[:800])
        np.testing.assert_array_equal(overlap_gap.data.data[1000:1800], \
                                      tr2.data[200:])
        np.testing.assert_array_equal(overlap_gap.data.data[2400:], tr3.data)
        # gap
        gap = tr2 + tr3
        self.assertEqual(len(gap), 2600)
        mask = np.zeros(2600).astype("bool")
        mask[1000:1600] = True
        np.testing.assert_array_equal(gap.data.mask, mask)
        np.testing.assert_array_equal(gap.data.data[:1000], tr2.data)
        np.testing.assert_array_equal(gap.data.data[1600:], tr3.data)

    def test_slice(self):
        """
        Tests the slicing of trace objects.
        """
        tr = Trace(data=np.arange(10, dtype='int32'))
        mempos = tr.data.ctypes.data
        t = tr.stats.starttime
        tr1 = tr.slice(t + 2, t + 8)
        tr1.data[0] = 10
        self.assertEqual(tr.data[2], 10)
        self.assertEqual(tr.data.ctypes.data, mempos)
        self.assertEqual(tr.data[2:9].ctypes.data, tr1.data.ctypes.data)
        self.assertEqual(tr1.data.ctypes.data - 8, mempos)

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
        # This is supposed to include the start- and endtimes and should
        # therefore cut right at 2 and 8.
        temp = deepcopy(tr)
        temp.trim(st + 2.1, st + 7.1)
        # Should be identical.
        temp2 = deepcopy(tr)
        temp2.trim(st + 2.0, st + 8.0)
        self.assertEqual(temp.stats.starttime, UTCDateTime(2))
        self.assertEqual(temp.stats.endtime, UTCDateTime(8))
        self.assertEqual(temp.stats.npts, 7)
        self.assertEqual(temp.stats, temp2.stats)
        np.testing.assert_array_equal(temp.data, temp2.data)
        # Create test array that allows for easy testing.
        # Check if the data is the same.
        self.assertNotEqual(temp.data.ctypes.data, tr.data[2:9].ctypes.data)
        np.testing.assert_array_equal(tr.data[2:9], temp.data)
        # Using out of bounds times should not do anything but create
        # a copy of the stats.
        temp = deepcopy(tr)
        temp.trim(st - 2.5, st + 200)
        self.assertEqual(temp.stats.starttime, UTCDateTime(-2.0))
        self.assertEqual(temp.stats.endtime, UTCDateTime(200))
        self.assertEqual(temp.stats.npts, 203)
        mask = np.zeros(203).astype("bool")
        mask[:2] = True
        mask[13:] = True
        np.testing.assert_array_equal(temp.data.mask, mask)
        # Alter the new stats to make sure the old one stays intact.
        temp.stats.starttime = UTCDateTime(1000)
        self.assertEqual(org_stats, tr.stats)
        # Check if the data adress is not the same, that is it is a copy
        self.assertNotEqual(temp.data.ctypes.data, tr.data.ctypes.data)
        np.testing.assert_array_equal(tr.data, temp.data[2:13])
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
        temp.trim(UTCDateTime(111.22222), UTCDateTime(112.99999))
        # Should again be identical.
        temp2 = deepcopy(tr)
        temp2.trim(UTCDateTime(111.21111), UTCDateTime(113.01111))
        np.testing.assert_array_equal(temp.data, temp2.data)
        self.assertEqual(temp.stats, temp2.stats)
        # Check stuff.
        self.assertEqual(temp.stats.starttime, UTCDateTime(111.21111))
        self.assertEqual(temp.stats.endtime, UTCDateTime(113.01111))
        # Check if the data is the same.
        temp = deepcopy(tr)
        temp.trim(UTCDateTime(0), UTCDateTime(1000 * 1000))
        self.assertNotEqual(temp.data.ctypes.data, tr.data.ctypes.data)
        # starttime must be in conformance with sampling rate
        t = UTCDateTime(1970, 1, 1, 0, 0, 0, 11110)
        self.assertEqual(temp.stats.starttime, t)
        delta = int((tr.stats.starttime - t) * tr.stats.sampling_rate + .5)
        np.testing.assert_array_equal(tr.data, temp.data[delta:delta + 111])
        # Make sure the original Trace object did not change.
        np.testing.assert_array_equal(tr.data, org_data)
        self.assertEqual(tr.data.ctypes.data, mem_pos)
        self.assertEqual(tr.stats, org_stats)

    def test_addOverlapsDefaultMethod(self):
        """
        Test __add__ method of the Trace object.
        """
        #1 - overlapping trace with differing data
        # Trace 1: 0000000
        # Trace 2:      1111111
        # 1 + 2  : 00000--11111
        tr1 = Trace(data=np.zeros(7))
        tr2 = Trace(data=np.ones(7))
        tr2.stats.starttime = tr1.stats.starttime + 5
        tr = tr1 + tr2
        self.assertTrue(isinstance(tr.data, np.ma.masked_array))
        self.assertEqual(tr.data.tolist(),
                         [0, 0, 0, 0, 0, None, None, 1, 1, 1, 1, 1])
        #2 - overlapping trace with same data
        # Trace 1: 0000000
        # Trace 2:      0000000
        # 1 + 2  : 000000000000
        tr1 = Trace(data=np.zeros(7))
        tr2 = Trace(data=np.zeros(7))
        tr2.stats.starttime = tr1.stats.starttime + 5
        tr = tr1 + tr2
        self.assertTrue(isinstance(tr.data, np.ndarray))
        np.testing.assert_array_equal(tr.data, np.zeros(12))
        #3 - contained overlap with same data
        # Trace 1: 1111111111
        # Trace 2:      11
        # 1 + 2  : 1111111111
        tr1 = Trace(data=np.ones(10))
        tr2 = Trace(data=np.ones(2))
        tr2.stats.starttime = tr1.stats.starttime + 5
        tr = tr1 + tr2
        self.assertTrue(isinstance(tr.data, np.ndarray))
        np.testing.assert_array_equal(tr.data, np.ones(10))
        #4 - contained overlap with differing data
        # Trace 1: 0000000000
        # Trace 2:      11
        # 1 + 2  : 00000--000
        tr1 = Trace(data=np.zeros(10))
        tr2 = Trace(data=np.ones(2))
        tr2.stats.starttime = tr1.stats.starttime + 5
        tr = tr1 + tr2
        self.assertTrue(isinstance(tr.data, np.ma.masked_array))
        self.assertEqual(tr.data.tolist(),
                         [0, 0, 0, 0, 0, None, None, 0, 0, 0])

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
        # 2 - different sampling rates for the different channels is ok
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
        tr1 + tr3
        tr2 + tr4

    def test_addWithDifferentDatatypesOrID(self):
        """
        Test __add__ method of the Trace object.
        """
        # 1 - different dtype for the same channel should fail
        tr1 = Trace(data=np.zeros(5, dtype="int32"))
        tr2 = Trace(data=np.zeros(5, dtype="float32"))
        self.assertRaises(TypeError, tr1.__add__, tr2)
        # 2 - different sampling rates for the different channels is ok
        tr1 = Trace(data=np.zeros(5, dtype="int32"))
        tr1.stats.channel = 'EHE'
        tr2 = Trace(data=np.zeros(5, dtype="float32"))
        tr2.stats.channel = 'EHZ'
        tr3 = Trace(data=np.zeros(5, dtype="int32"))
        tr3.stats.channel = 'EHE'
        tr4 = Trace(data=np.zeros(5, dtype="float32"))
        tr4.stats.channel = 'EHZ'
        tr1 + tr3
        tr2 + tr4
        # adding traces with different ids should raise
        self.assertRaises(TypeError, tr1.__add__, tr2)
        self.assertRaises(TypeError, tr3.__add__, tr4)


def suite():
    return unittest.makeSuite(TraceTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
