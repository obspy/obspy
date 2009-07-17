# -*- coding: utf-8 -*-

from copy import deepcopy
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
        Tests the len method of the L{Trace} class.
        """
        # set up
        trace = Trace(data=range(0, 1000))
        self.assertEquals(len(trace), 1000)

    def test_ltrim(self):
        """
        Tests the ltrim method of the L{Trace} class.
        """
        # set up
        trace = Trace(data=range(0, 1000))
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 200.0
        end = UTCDateTime(2000, 1, 1, 0, 0, 5, 0)
        trace.stats.endtime = end
        # ltrim 100 samples
        tr = deepcopy(trace)
        tr.ltrim(0.5)
        self.assertEquals(tr.data[0:5], [100, 101, 102, 103, 104])
        self.assertEquals(len(tr.data), 900)
        self.assertEquals(tr.stats.npts, 900)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start + 0.5)
        self.assertEquals(tr.stats.endtime, end)
        # ltrim 202 samples
        tr = deepcopy(trace)
        tr.ltrim(1.010)
        self.assertEquals(tr.data[0:5], [202, 203, 204, 205, 206])
        self.assertEquals(len(tr.data), 798)
        self.assertEquals(tr.stats.npts, 798)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start + 1.010)
        self.assertEquals(tr.stats.endtime, end)
        # ltrim to UTCDateTime
        tr = deepcopy(trace)
        tr.ltrim(UTCDateTime(2000, 1, 1, 0, 0, 1, 10000))
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
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        tr.ltrim(-100)
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        # start time > end time
        tr.ltrim(UTCDateTime(2001))
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        tr.ltrim(5.1)
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        # start time == end time
        tr.ltrim(5)
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
        end = UTCDateTime(2000, 1, 1, 0, 0, 5, 0)
        trace.stats.endtime = end
        # rtrim 100 samples
        tr = deepcopy(trace)
        tr.rtrim(0.5)
        self.assertEquals(tr.data[-5:], [895, 896, 897, 898, 899])
        self.assertEquals(len(tr.data), 900)
        self.assertEquals(tr.stats.npts, 900)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start)
        self.assertEquals(tr.stats.endtime, end - 0.5)
        # rtrim 202 samples
        tr = deepcopy(trace)
        tr.rtrim(1.010)
        self.assertEquals(tr.data[-5:], [793, 794, 795, 796, 797])
        self.assertEquals(len(tr.data), 798)
        self.assertEquals(tr.stats.npts, 798)
        self.assertEquals(tr.stats.sampling_rate, 200.0)
        self.assertEquals(tr.stats.starttime, start)
        self.assertEquals(tr.stats.endtime, end - 1.010)
        # rtrim to UTCDateTime
        tr = deepcopy(trace)
        tr.rtrim(UTCDateTime(2000, 1, 1, 0, 0, 3, 990000))
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
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        tr.rtrim(-100)
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        # end time > start time
        tr.rtrim(UTCDateTime(2001))
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        tr.rtrim(5.1)
        self.assertEquals(trace.stats, tr.stats)
        self.assertEquals(trace.data, tr.data)
        # end time == start time
        # returns one sample!
        tr.rtrim(5)
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
        trace = Trace(data=range(0, 1000))
        start = UTCDateTime(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 200.0
        end = UTCDateTime(2000, 1, 1, 0, 0, 5, 0)
        trace.stats.endtime = end
        # rtrim 100 samples
        trace.trim(0.5, 0.5)
        self.assertEquals(trace.data[-5:], [895, 896, 897, 898, 899])
        self.assertEquals(trace.data[0:5], [100, 101, 102, 103, 104])
        self.assertEquals(len(trace.data), 800)
        self.assertEquals(trace.stats.npts, 800)
        self.assertEquals(trace.stats.sampling_rate, 200.0)
        self.assertEquals(trace.stats.starttime, start + 0.5)
        self.assertEquals(trace.stats.endtime, end - 0.5)


def suite():
    return unittest.makeSuite(TraceTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
