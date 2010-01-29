# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
from numpy import array
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
        trace = Trace(data=array(range(0, 1000)))
        self.assertEquals(len(trace), 1000)
        self.assertEquals(trace.count(), 1000)

    def test_ltrim(self):
        """
        Tests the ltrim method of the L{Trace} class.
        """
        # set up
        trace = Trace(data=range(0, 1000))
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
        self.assertEquals(tr.data[0:5], [100, 101, 102, 103, 104])
        self.assertEquals(len(tr.data), 900)
        self.assertEquals(tr.stats.npts, 900)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start + 0.5)
        self.assertEquals(tr.stats.endtime, end)
        # ltrim 202 samples
        tr = deepcopy(trace)
        tr.ltrim(1.010)
        tr.verify()
        self.assertEquals(tr.data[0:5], [202, 203, 204, 205, 206])
        self.assertEquals(len(tr.data), 798)
        self.assertEquals(tr.stats.npts, 798)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start + 1.010)
        self.assertEquals(tr.stats.endtime, end)
        # ltrim to UTCDateTime
        tr = deepcopy(trace)
        tr.ltrim(UTCDateTime(2000, 1, 1, 0, 0, 1, 10000))
        tr.verify()
        self.assertEquals(tr.data[0:5], [202, 203, 204, 205, 206])
        self.assertEquals(len(tr.data), 798)
        self.assertEquals(tr.stats.npts, 798)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start + 1.010)
        self.assertEquals(tr.stats.endtime, end)
        # some sanity checks
        # negative start time
        tr = deepcopy(trace)
        tr.ltrim(UTCDateTime(1999))
        tr.verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        tr.ltrim(-100)
        tr.verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        # start time > end time
        tr.ltrim(UTCDateTime(2001))
        tr.verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        tr.ltrim(5.1)
        tr.verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        # start time == end time
        tr.ltrim(5)
        tr.verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)

    def test_rtrim(self):
        """
        Tests the rtrim method of the L{Trace} class.
        """
        # set up
        trace = Trace(data=range(0, 1000))
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 200.0
        end = UTCDateTime(2000, 1, 1, 0, 0, 4, 995000)
        trace.verify()
        # rtrim 100 samples
        tr = deepcopy(trace)
        tr.rtrim(0.5)
        tr.verify()
        self.assertEquals(tr.data[-5:], [895, 896, 897, 898, 899])
        self.assertEquals(len(tr.data), 900)
        self.assertEquals(tr.stats.npts, 900)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start)
        self.assertEquals(tr.stats.endtime, end - 0.5)
        # rtrim 202 samples
        tr = deepcopy(trace)
        tr.rtrim(1.010)
        tr.verify()
        self.assertEquals(tr.data[-5:], [793, 794, 795, 796, 797])
        self.assertEquals(len(tr.data), 798)
        self.assertEquals(tr.stats.npts, 798)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start)
        self.assertEquals(tr.stats.endtime, end - 1.010)
        # rtrim 1 minute via UTCDateTime
        tr = deepcopy(trace)
        tr.rtrim(UTCDateTime(2000, 1, 1, 0, 0, 3, 985000))
        tr.verify()
        self.assertEquals(tr.data[-5:], [793, 794, 795, 796, 797])
        self.assertEquals(len(tr.data), 798)
        self.assertEquals(tr.stats.npts, 798)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start)
        self.assertEquals(tr.stats.endtime, end - 1.010)
        # some sanity checks
        # negative end time
        tr = deepcopy(trace)
        tr.rtrim(UTCDateTime(1999))
        tr.verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        tr.rtrim(-100)
        tr.verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        # end time > start time
        tr.rtrim(UTCDateTime(2001))
        tr.verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        tr.rtrim(5.1)
        tr.verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        # end time == start time
        # returns one sample!
        tr.rtrim(4.995)
        tr.verify()
        self.assertEquals(tr.data, [0])
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
        trace = Trace(data=range(0, 1001))
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 200.0
        end = UTCDateTime(2000, 1, 1, 0, 0, 5, 0)
        trace.verify()
        # rtrim 100 samples
        trace.trim(0.5, 0.5)
        trace.verify()
        self.assertEquals(trace.data[-5:], [896, 897, 898, 899, 900])
        self.assertEquals(trace.data[0:5], [100, 101, 102, 103, 104])
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
        tr1 = Trace(data=range(0, 1000))
        tr1.stats.sampling_rate = 200
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr2 = Trace(data=range(0, 1000)[::-1])
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
        tr1 = Trace(data=range(0, 1000))
        tr1.stats.sampling_rate = 200
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr2 = Trace(data=range(0, 1000)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 4
        # verify
        tr1.verify()
        tr2.verify()
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
        self.assertEquals(trace[800], (800 + 999) // 2)
        self.assertEquals(trace[999], (999 + 800) // 2)
        self.assertEquals(trace[1000], 799)
        self.assertEquals(trace[1799], 0)
        # verify
        trace.verify()

    def test_addSameTrace(self):
        """
        Tests the add method of the L{Trace} class.
        """
        # set up
        tr1 = Trace(data=range(0, 1001))
        tr1.stats.sampling_rate = 200
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        # verify
        tr1.verify()
        # add
        trace = tr1 + tr1
        # should return exact the same values
        self.assertEquals(trace.stats, tr1.stats)
        self.assertEquals(trace.data, tr1.data)
        # verify
        trace.verify()

    def test_addTraceWithinTrace(self):
        """
        Tests the add method of the L{Trace} class.
        """
        # set up
        tr1 = Trace(data=range(0, 1001))
        tr1.stats.sampling_rate = 200
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr2 = Trace(data=range(0, 201))
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 1
        # verify
        tr1.verify()
        tr2.verify()
        # add
        trace = tr1 + tr2
        # should return exact the same values like trace 1
        self.assertEquals(trace.stats, tr1.stats)
        self.assertEquals(trace.data, tr1.data)
        # add the other way around
        trace = tr2 + tr1
        # should return exact the same values like trace 1
        self.assertEquals(trace.stats, tr1.stats)
        self.assertEquals(trace.data, tr1.data)
        # verify
        trace.verify()

    def test_mergeGapAndOverlap(self):
        """
        Test order of merging traces.
        """
        # set up
        tr1 = Trace(data=range(0, 1000))
        tr1.stats.sampling_rate = 200
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr2 = Trace(data=range(0, 1000)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 4
        tr3 = Trace(data=range(0, 1000)[::-1])
        tr3.stats.sampling_rate = 200
        tr3.stats.starttime = start + 12
        # verify
        tr1.verify()
        tr2.verify()
        tr3.verify()
        # check types
        overlap = tr1 + tr2
        self.assertFalse(is_masked(overlap.data))
        masked_gap = overlap + tr3
        self.assertTrue(is_masked(masked_gap.data))
        # check types
        masked_gap = tr2 + tr3
        self.assertTrue(is_masked(masked_gap.data))
        overlap = tr1 + masked_gap
        self.assertTrue(is_masked(overlap.data))

    def test_slice(self):
        """
        Tests the slicing of trace objects.
        """
        # Create test array that allows for easy testing.
        tr = Trace(data = np.arange(11))
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
        temp = tr.slice(st + 2.9, st + 7.1)
        # Should be identical.
        temp2 = tr.slice(st + 2.0, st + 8.0)
        self.assertEqual(temp.stats.starttime, UTCDateTime(2))
        self.assertEqual(temp.stats.endtime, UTCDateTime(8))
        self.assertEqual(temp.stats.npts, 7)
        self.assertEqual(temp.stats, temp2.stats)
        np.testing.assert_array_equal(temp.data, temp2.data)
        # Create test array that allows for easy testing.
        # Check if the data is the same.
        self.assertEqual(temp.data.ctypes.data, tr.data[2:9].ctypes.data)
        np.testing.assert_array_equal(tr.data[2:9], temp.data)

        # Using out of bounds times should not do anything but create
        # a copy of the stats.
        temp = tr.slice(st - 2.5, st + 200)
        self.assertEqual(temp.stats.starttime, UTCDateTime(0))
        self.assertEqual(temp.stats.endtime, UTCDateTime(10))
        self.assertEqual(temp.stats.npts, 11)
        self.assertEqual(temp.stats, org_stats)
        # Alter the new stats to make sure the old one stays intact.
        temp.stats.starttime = UTCDateTime(1000)
        self.assertEqual(org_stats, tr.stats)

        # Check if the data is the same.
        self.assertEqual(temp.data.ctypes.data, tr.data.ctypes.data)
        np.testing.assert_array_equal(tr.data, temp.data)
        # Make sure the original Trace object did not change.
        np.testing.assert_array_equal(tr.data, org_data)
        self.assertEqual(tr.data.ctypes.data, mem_pos)
        self.assertEqual(tr.stats, org_stats)

        # Use more complicated times and sampling rate.
        tr = Trace(data = np.arange(111))
        tr.stats.starttime = UTCDateTime(111.11111)
        tr.stats.sampling_rate = 50.0
        org_stats = deepcopy(tr.stats)
        org_data = deepcopy(tr.data)
        # Save memory position of array.
        mem_pos = tr.data.ctypes.data
        # Create temp trace object used for testing.
        temp = tr.slice(UTCDateTime(111.22222), UTCDateTime(112.99999))
        # Should again be identical.
        temp2 = tr.slice(UTCDateTime(111.21111), UTCDateTime(113.01111))
        np.testing.assert_array_equal(temp.data, temp2.data)
        self.assertEqual(temp.stats, temp2.stats)
        # Check stuff.
        self.assertEqual(temp.stats.starttime, UTCDateTime(111.21111))
        self.assertEqual(temp.stats.endtime, UTCDateTime(113.01111))

        # Check if the data is the same.
        temp = tr.slice(UTCDateTime(0), UTCDateTime(1000*1000))
        self.assertEqual(temp.data.ctypes.data, tr.data.ctypes.data)
        np.testing.assert_array_equal(tr.data, temp.data)
        # Make sure the original Trace object did not change.
        np.testing.assert_array_equal(tr.data, org_data)
        self.assertEqual(tr.data.ctypes.data, mem_pos)
        self.assertEqual(tr.stats, org_stats)


def suite():
    return unittest.makeSuite(TraceTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
