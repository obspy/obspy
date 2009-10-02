# -*- coding: utf-8 -*-

from copy import deepcopy
from numpy import isnan, array
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
        trace.stats.endtime = end
        # verify
        trace._verify()
        # ltrim 100 samples
        tr = deepcopy(trace)
        tr.ltrim(0.5)
        tr._verify()
        self.assertEquals(tr.data[0:5], [100, 101, 102, 103, 104])
        self.assertEquals(len(tr.data), 900)
        self.assertEquals(tr.stats.npts, 900)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start + 0.5)
        self.assertEquals(tr.stats.endtime, end)
        # ltrim 202 samples
        tr = deepcopy(trace)
        tr.ltrim(1.010)
        tr._verify()
        self.assertEquals(tr.data[0:5], [202, 203, 204, 205, 206])
        self.assertEquals(len(tr.data), 798)
        self.assertEquals(tr.stats.npts, 798)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start + 1.010)
        self.assertEquals(tr.stats.endtime, end)
        # ltrim to UTCDateTime
        tr = deepcopy(trace)
        tr.ltrim(UTCDateTime(2000, 1, 1, 0, 0, 1, 10000))
        tr._verify()
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
        tr._verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        tr.ltrim(-100)
        tr._verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        # start time > end time
        tr.ltrim(UTCDateTime(2001))
        tr._verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        tr.ltrim(5.1)
        tr._verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        # start time == end time
        tr.ltrim(5)
        tr._verify()
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
        trace.stats.endtime = end
        # verify
        trace._verify()
        # rtrim 100 samples
        tr = deepcopy(trace)
        tr.rtrim(0.5)
        tr._verify()
        self.assertEquals(tr.data[-5:], [895, 896, 897, 898, 899])
        self.assertEquals(len(tr.data), 900)
        self.assertEquals(tr.stats.npts, 900)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start)
        self.assertEquals(tr.stats.endtime, end - 0.5)
        # rtrim 202 samples
        tr = deepcopy(trace)
        tr.rtrim(1.010)
        tr._verify()
        self.assertEquals(tr.data[-5:], [793, 794, 795, 796, 797])
        self.assertEquals(len(tr.data), 798)
        self.assertEquals(tr.stats.npts, 798)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start)
        self.assertEquals(tr.stats.endtime, end - 1.010)
        # rtrim 1 minute via UTCDateTime
        tr = deepcopy(trace)
        tr.rtrim(UTCDateTime(2000, 1, 1, 0, 0, 3, 985000))
        tr._verify()
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
        tr._verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        tr.rtrim(-100)
        tr._verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        # end time > start time
        tr.rtrim(UTCDateTime(2001))
        tr._verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        tr.rtrim(5.1)
        tr._verify()
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        # end time == start time
        # returns one sample!
        tr.rtrim(4.995)
        tr._verify()
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
        trace.stats.endtime = end
        # verify
        trace._verify()
        # rtrim 100 samples
        trace.trim(0.5, 0.5)
        trace._verify()
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
        tr1.stats.endtime = start + 4.995
        tr2 = Trace(data=range(0, 1000)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 10
        tr2.stats.endtime = start + 14.995
        # verify
        tr1._verify()
        tr2._verify()
        # add
        trace = tr1 + tr2
        # stats
        self.assertEquals(trace.stats.starttime, start)
        self.assertEquals(trace.stats.endtime, start + 14.995)
        self.assertEquals(trace.stats.sampling_rate, 200)
        self.assertEquals(trace.stats.npts, 2000)
        # data
        self.assertEquals(len(trace), 2000)
        self.assertEquals(trace[0], 0)
        self.assertEquals(trace[999], 999)
        self.assertTrue(is_masked(trace[1000]))
        self.assertTrue(is_masked(trace[1999]))
        self.assertEquals(trace[2000], 999)
        self.assertEquals(trace[2999], 0)
        # verify
        trace._verify()

    def test_addTraceWithOverlap(self):
        """
        Tests the add method of the L{Trace} class.
        """
        # set up
        tr1 = Trace(data=range(0, 1000))
        tr1.stats.sampling_rate = 200
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr1.stats.endtime = start + 4.995
        tr2 = Trace(data=range(0, 1000)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 4
        tr2.stats.endtime = start + 8.995
        # verify
        tr1._verify()
        tr2._verify()
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
        self.assertEquals(trace[800], (800 + 999) / 2)
        self.assertEquals(trace[999], (999 + 800) / 2)
        self.assertEquals(trace[1000], 799)
        self.assertEquals(trace[1799], 0)
        # verify
        trace._verify()

    def test_addSameTrace(self):
        """
        Tests the add method of the L{Trace} class.
        """
        # set up
        tr1 = Trace(data=range(0, 1001))
        tr1.stats.sampling_rate = 200
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr1.stats.endtime = start + 5
        # verify
        tr1._verify()
        # add
        trace = tr1 + tr1
        # should return exact the same values
        self.assertEquals(trace.stats, tr1.stats)
        self.assertEquals(trace.data, tr1.data)
        # verify
        trace._verify()

    def test_addTraceWithinTrace(self):
        """
        Tests the add method of the L{Trace} class.
        """
        # set up
        tr1 = Trace(data=range(0, 1001))
        tr1.stats.sampling_rate = 200
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr1.stats.endtime = start + 5
        tr2 = Trace(data=range(0, 201))
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 1
        tr2.stats.endtime = start + 2
        # verify
        tr1._verify()
        tr2._verify()
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
        trace._verify()

def suite():
    return unittest.makeSuite(TraceTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
