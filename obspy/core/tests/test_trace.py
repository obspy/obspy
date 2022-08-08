# -*- coding: utf-8 -*-
import math
import os
import pickle
from copy import deepcopy
import warnings
from unittest import mock

from packaging.version import parse as parse_version
import numpy as np
import numpy.ma as ma

from obspy import Stream, Trace, __version__, read, read_inventory
from obspy import UTCDateTime as UTC
from obspy.core import Stats
from obspy.io.xseed import Parser
import pytest


class TestTrace:
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
        assert len(tr) == 4
        # NumPy masked array
        data = np.ma.array([0, 1, 2, 3], mask=[True, True, False, False])
        tr = Trace(data=data)
        assert len(tr) == 4
        # other data types will raise
        with pytest.raises(ValueError):
            Trace(data=[0, 1, 2, 3])
        with pytest.raises(ValueError):
            Trace(data=(0, 1, 2, 3))
        with pytest.raises(ValueError):
            Trace(data='1234')

    def test_setattr(self):
        """
        Tests the __setattr__ method of the Trace class.
        """
        # NumPy ndarray
        tr = Trace()
        tr.data = np.arange(4)
        assert len(tr) == 4
        # NumPy masked array
        tr = Trace()
        tr.data = np.ma.array([0, 1, 2, 3], mask=[True, True, False, False])
        assert len(tr) == 4
        # other data types will raise
        tr = Trace()
        with pytest.raises(ValueError):
            tr.__setattr__('data', [0, 1, 2, 3])
        with pytest.raises(ValueError):
            tr.__setattr__('data', (0, 1, 2, 3))
        with pytest.raises(ValueError):
            tr.__setattr__('data', '1234')

    def test_len(self):
        """
        Tests the __len__ and count methods of the Trace class.
        """
        trace = Trace(data=np.arange(1000))
        assert len(trace) == 1000
        assert trace.count() == 1000

    def test_mul(self):
        """
        Tests the __mul__ method of the Trace class.
        """
        tr = Trace(data=np.arange(10))
        st = tr * 5
        assert len(st) == 5
        # you may only multiply using an integer
        with pytest.raises(TypeError):
            tr.__mul__(2.5)
        with pytest.raises(TypeError):
            tr.__mul__('1234')

    def test_div(self):
        """
        Tests the __div__ method of the Trace class.
        """
        tr = Trace(data=np.arange(1000))
        st = tr / 5
        assert len(st) == 5
        assert len(st[0]) == 200
        # you may only multiply using an integer
        with pytest.raises(TypeError):
            tr.__div__(2.5)
        with pytest.raises(TypeError):
            tr.__div__('1234')

    def test_ltrim(self):
        """
        Tests the ltrim method of the Trace class.
        """
        # set up
        trace = Trace(data=np.arange(1000))
        start = UTC(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 200.0
        end = UTC(2000, 1, 1, 0, 0, 4, 995000)
        # verify
        trace.verify()
        # UTCDateTime/int/float required
        with pytest.raises(TypeError):
            trace._ltrim('1234')
        with pytest.raises(TypeError):
            trace._ltrim([1, 2, 3, 4])
        # ltrim 100 samples
        tr = deepcopy(trace)
        tr._ltrim(0.5)
        tr.verify()
        np.testing.assert_array_equal(tr.data[0:5],
                                      np.array([100, 101, 102, 103, 104]))
        assert len(tr.data) == 900
        assert tr.stats.npts == 900
        assert tr.stats.sampling_rate == 200.0
        assert tr.stats.starttime == start + 0.5
        assert tr.stats.endtime == end
        # ltrim 202 samples
        tr = deepcopy(trace)
        tr._ltrim(1.010)
        tr.verify()
        np.testing.assert_array_equal(tr.data[0:5],
                                      np.array([202, 203, 204, 205, 206]))
        assert len(tr.data) == 798
        assert tr.stats.npts == 798
        assert tr.stats.sampling_rate == 200.0
        assert tr.stats.starttime == start + 1.010
        assert tr.stats.endtime == end
        # ltrim to UTCDateTime
        tr = deepcopy(trace)
        tr._ltrim(UTC(2000, 1, 1, 0, 0, 1, 10000))
        tr.verify()
        np.testing.assert_array_equal(tr.data[0:5],
                                      np.array([202, 203, 204, 205, 206]))
        assert len(tr.data) == 798
        assert tr.stats.npts == 798
        assert tr.stats.sampling_rate == 200.0
        assert tr.stats.starttime == start + 1.010
        assert tr.stats.endtime == end
        # some sanity checks
        # negative start time as datetime
        tr = deepcopy(trace)
        tr._ltrim(start - 1, pad=True)
        tr.verify()
        assert tr.stats.starttime == start - 1
        np.testing.assert_array_equal(trace.data, tr.data[200:])
        assert tr.stats.endtime == trace.stats.endtime
        # negative start time as integer
        tr = deepcopy(trace)
        tr._ltrim(-100, pad=True)
        tr.verify()
        assert tr.stats.starttime == start - 100
        delta = 100 * trace.stats.sampling_rate
        np.testing.assert_array_equal(trace.data, tr.data[int(delta):])
        assert tr.stats.endtime == trace.stats.endtime
        # start time > end time
        tr = deepcopy(trace)
        tr._ltrim(trace.stats.endtime + 100)
        tr.verify()
        assert tr.stats.starttime == trace.stats.endtime + 100
        np.testing.assert_array_equal(tr.data, np.empty(0))
        assert tr.stats.endtime == tr.stats.starttime
        # start time == end time
        tr = deepcopy(trace)
        tr._ltrim(5)
        tr.verify()
        assert tr.stats.starttime == trace.stats.starttime + 5
        np.testing.assert_array_equal(tr.data, np.empty(0))
        assert tr.stats.endtime == tr.stats.starttime
        # start time == end time
        tr = deepcopy(trace)
        tr._ltrim(5.1)
        tr.verify()
        assert tr.stats.starttime == trace.stats.starttime + 5.1
        np.testing.assert_array_equal(tr.data, np.empty(0))
        assert tr.stats.endtime == tr.stats.starttime

    def test_rtrim(self):
        """
        Tests the rtrim method of the Trace class.
        """
        # set up
        trace = Trace(data=np.arange(1000))
        start = UTC(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 200.0
        end = UTC(2000, 1, 1, 0, 0, 4, 995000)
        trace.verify()
        # UTCDateTime/int/float required
        with pytest.raises(TypeError):
            trace._rtrim('1234')
        with pytest.raises(TypeError):
            trace._rtrim([1, 2, 3, 4])
        # rtrim 100 samples
        tr = deepcopy(trace)
        tr._rtrim(0.5)
        tr.verify()
        np.testing.assert_array_equal(tr.data[-5:],
                                      np.array([895, 896, 897, 898, 899]))
        assert len(tr.data) == 900
        assert tr.stats.npts == 900
        assert tr.stats.sampling_rate == 200.0
        assert tr.stats.starttime == start
        assert tr.stats.endtime == end - 0.5
        # rtrim 202 samples
        tr = deepcopy(trace)
        tr._rtrim(1.010)
        tr.verify()
        np.testing.assert_array_equal(tr.data[-5:],
                                      np.array([793, 794, 795, 796, 797]))
        assert len(tr.data) == 798
        assert tr.stats.npts == 798
        assert tr.stats.sampling_rate == 200.0
        assert tr.stats.starttime == start
        assert tr.stats.endtime == end - 1.010
        # rtrim 1 minute via UTCDateTime
        tr = deepcopy(trace)
        tr._rtrim(UTC(2000, 1, 1, 0, 0, 3, 985000))
        tr.verify()
        np.testing.assert_array_equal(tr.data[-5:],
                                      np.array([793, 794, 795, 796, 797]))
        assert len(tr.data) == 798
        assert tr.stats.npts == 798
        assert tr.stats.sampling_rate == 200.0
        assert tr.stats.starttime == start
        assert tr.stats.endtime == end - 1.010
        # some sanity checks
        # negative end time
        tr = deepcopy(trace)
        t = UTC(1999, 12, 31)
        tr._rtrim(t)
        tr.verify()
        assert tr.stats.endtime == t
        np.testing.assert_array_equal(tr.data, np.empty(0))
        # negative end time with given seconds
        tr = deepcopy(trace)
        tr._rtrim(100)
        tr.verify()
        assert tr.stats.endtime == trace.stats.endtime - 100
        np.testing.assert_array_equal(tr.data, np.empty(0))
        assert tr.stats.endtime == tr.stats.starttime
        # end time > start time
        tr = deepcopy(trace)
        t = UTC(2001)
        tr._rtrim(t)
        tr.verify()
        assert tr.stats.endtime == t
        np.testing.assert_array_equal(tr.data, np.empty(0))
        assert tr.stats.endtime == tr.stats.starttime
        # end time > start time given seconds
        tr = deepcopy(trace)
        tr._rtrim(5.1)
        tr.verify()
        delta = int(math.floor(round(5.1 * trace.stats.sampling_rate, 7)))
        endtime = trace.stats.starttime + trace.stats.delta * \
            (trace.stats.npts - delta - 1)
        assert tr.stats.endtime == endtime
        np.testing.assert_array_equal(tr.data, np.empty(0))
        # end time == start time
        # returns one sample!
        tr = deepcopy(trace)
        tr._rtrim(4.995)
        tr.verify()
        np.testing.assert_array_equal(tr.data, np.array([0]))
        assert len(tr.data) == 1
        assert tr.stats.npts == 1
        assert tr.stats.sampling_rate == 200.0
        assert tr.stats.starttime == start
        assert tr.stats.endtime == start

    def test_rtrim_with_padding(self):
        """
        Tests the _rtrim() method of the Trace class with padding. It has
        already been tested in the two sided trimming tests. This is just to
        have an explicit test. Also tests issue #429.
        """
        # set up
        trace = Trace(data=np.arange(10))
        start = UTC(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 1.0
        trace.verify()

        # Pad with no fill_value will mask the additional values.
        tr = trace.copy()
        end = tr.stats.endtime
        tr._rtrim(end + 10, pad=True)
        assert tr.stats.endtime == trace.stats.endtime + 10
        np.testing.assert_array_equal(tr.data[0:10], np.arange(10))
        # Check that the first couple of entries are not masked.
        assert not tr.data[0:10].mask.any()
        # All the other entries should be masked.
        assert tr.data[10:].mask.all()

        # Pad with fill_value.
        tr = trace.copy()
        end = tr.stats.endtime
        tr._rtrim(end + 10, pad=True, fill_value=-33)
        assert tr.stats.endtime == trace.stats.endtime + 10
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
        start = UTC(2000, 1, 1, 0, 0, 0, 0)
        trace.stats.starttime = start
        trace.stats.sampling_rate = 200.0
        end = UTC(2000, 1, 1, 0, 0, 5, 0)
        trace.verify()
        # rtrim 100 samples
        trace.trim(0.5, 0.5)
        trace.verify()
        np.testing.assert_array_equal(trace.data[-5:],
                                      np.array([896, 897, 898, 899, 900]))
        np.testing.assert_array_equal(trace.data[:5],
                                      np.array([100, 101, 102, 103, 104]))
        assert len(trace.data) == 801
        assert trace.stats.npts == 801
        assert trace.stats.sampling_rate == 200.0
        assert trace.stats.starttime == start + 0.5
        assert trace.stats.endtime == end - 0.5
        # start time should be before end time
        with pytest.raises(ValueError):
            trace.trim(end, start)

    def test_trim_all_does_not_change_dtype(self):
        """
        If a Trace is completely trimmed, e.g. no data samples are remaining,
        the dtype should remain unchanged.

        A trace with no data samples is not really senseful but the dtype
        should not be changed anyways.
        """
        # Choose non native dtype.
        tr = Trace(np.arange(100, dtype=np.int16))
        tr.trim(UTC(10000), UTC(20000))
        # Assert the result.
        assert len(tr.data) == 0
        assert tr.data.dtype == np.int16

    def test_add_trace_with_gap(self):
        """
        Tests __add__ method of the Trace class.
        """
        # set up
        tr1 = Trace(data=np.arange(1000))
        tr1.stats.sampling_rate = 200
        start = UTC(2000, 1, 1, 0, 0, 0, 0)
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
        assert trace.stats.starttime == start
        assert trace.stats.endtime == start + 14.995
        assert trace.stats.sampling_rate == 200
        assert trace.stats.npts == 3000
        # data
        assert len(trace) == 3000
        assert trace[0] == 0
        assert trace[999] == 999
        assert ma.is_masked(trace[1000])
        assert ma.is_masked(trace[1999])
        assert trace[2000] == 999
        assert trace[2999] == 0
        # verify
        trace.verify()

    def test_add_trace_with_overlap(self):
        """
        Tests __add__ method of the Trace class.
        """
        # set up
        tr1 = Trace(data=np.arange(1000))
        tr1.stats.sampling_rate = 200
        start = UTC(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr2 = Trace(data=np.arange(0, 1000)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 4
        # add
        trace = tr1 + tr2
        # stats
        assert trace.stats.starttime == start
        assert trace.stats.endtime == start + 8.995
        assert trace.stats.sampling_rate == 200
        assert trace.stats.npts == 1800
        # data
        assert len(trace) == 1800
        assert trace[0] == 0
        assert trace[799] == 799
        assert trace[800].mask
        assert trace[999].mask
        assert trace[1000] == 799
        assert trace[1799] == 0
        # verify
        trace.verify()

    def test_add_same_trace(self):
        """
        Tests __add__ method of the Trace class.
        """
        # set up
        tr1 = Trace(data=np.arange(1001))
        # add
        trace = tr1 + tr1
        # should return exact the same values
        assert trace.stats == tr1.stats
        np.testing.assert_array_equal(trace.data, tr1.data)
        # verify
        trace.verify()

    def test_add_trace_within_trace(self):
        """
        Tests __add__ method of the Trace class.
        """
        # set up
        tr1 = Trace(data=np.arange(1001))
        tr1.stats.sampling_rate = 200
        start = UTC(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr2 = Trace(data=np.arange(201))
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 1
        # add
        trace = tr1 + tr2
        # should return exact the same values like trace 1
        assert trace.stats == tr1.stats
        mask = np.zeros(len(tr1)).astype(np.bool_)
        mask[200:401] = True
        np.testing.assert_array_equal(trace.data.mask, mask)
        np.testing.assert_array_equal(trace.data.data[:200], tr1.data[:200])
        np.testing.assert_array_equal(trace.data.data[401:], tr1.data[401:])
        # add the other way around
        trace = tr2 + tr1
        # should return exact the same values like trace 1
        assert trace.stats == tr1.stats
        np.testing.assert_array_equal(trace.data.mask, mask)
        np.testing.assert_array_equal(trace.data.data[:200], tr1.data[:200])
        np.testing.assert_array_equal(trace.data.data[401:], tr1.data[401:])
        # verify
        trace.verify()

    def test_add_gap_and_overlap(self):
        """
        Test order of merging traces.
        """
        # set up
        tr1 = Trace(data=np.arange(1000))
        tr1.stats.sampling_rate = 200
        start = UTC(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr2 = Trace(data=np.arange(1000)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 4
        tr3 = Trace(data=np.arange(1000)[::-1])
        tr3.stats.sampling_rate = 200
        tr3.stats.starttime = start + 12
        # overlap
        overlap = tr1 + tr2
        assert len(overlap) == 1800
        mask = np.zeros(1800).astype(np.bool_)
        mask[800:1000] = True
        np.testing.assert_array_equal(overlap.data.mask, mask)
        np.testing.assert_array_equal(overlap.data.data[:800], tr1.data[:800])
        np.testing.assert_array_equal(overlap.data.data[1000:], tr2.data[200:])
        # overlap + gap
        overlap_gap = overlap + tr3
        assert len(overlap_gap) == 3400
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
        assert len(gap) == 2600
        mask = np.zeros(2600).astype(np.bool_)
        mask[1000:1600] = True
        np.testing.assert_array_equal(gap.data.mask, mask)
        np.testing.assert_array_equal(gap.data.data[:1000], tr2.data)
        np.testing.assert_array_equal(gap.data.data[1600:], tr3.data)

    def test_add_into_gap(self):
        """
        Test __add__ method of the Trace class
        Adding a trace that fits perfectly into gap in a trace
        """
        my_array = np.arange(6, dtype=np.int32)

        stats = Stats()
        stats.network = 'VI'
        stats['starttime'] = UTC(2009, 8, 5, 0, 0, 0)
        stats['npts'] = 0
        stats['station'] = 'IKJA'
        stats['channel'] = 'EHZ'
        stats['sampling_rate'] = 1

        bigtrace = Trace(data=np.array([], dtype=np.int32), header=stats)
        bigtrace_sort = bigtrace.copy()
        stats['npts'] = len(my_array)
        my_trace = Trace(data=my_array, header=stats)

        stats['npts'] = 2
        trace1 = Trace(data=my_array[0:2].copy(), header=stats)
        stats['starttime'] = UTC(2009, 8, 5, 0, 0, 2)
        trace2 = Trace(data=my_array[2:4].copy(), header=stats)
        stats['starttime'] = UTC(2009, 8, 5, 0, 0, 4)
        trace3 = Trace(data=my_array[4:6].copy(), header=stats)

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
                assert isinstance(tr, Trace)
                assert not isinstance(tr.data, np.ma.masked_array)

            assert (bigtrace_sort.data == my_array).all()

            fail_pattern = "\n\tExpected %s\n\tbut got  %s"
            failinfo = fail_pattern % (my_trace, bigtrace_sort)
            failinfo += fail_pattern % (my_trace.data, bigtrace_sort.data)
            assert bigtrace_sort == my_trace, failinfo

            failinfo = fail_pattern % (my_array, bigtrace.data)
            assert (bigtrace.data == my_array).all(), failinfo

            failinfo = fail_pattern % (my_trace, bigtrace)
            failinfo += fail_pattern % (my_trace.data, bigtrace.data)
            assert bigtrace == my_trace, failinfo

            for array_ in (bigtrace.data, bigtrace_sort.data):
                failinfo = fail_pattern % (my_array.dtype, array_.dtype)
                assert my_array.dtype == array_.dtype, failinfo

    def test_slice(self):
        """
        Tests the slicing of trace objects.
        """
        tr = Trace(data=np.arange(10, dtype=np.int32))
        mempos = tr.data.ctypes.data
        t = tr.stats.starttime
        tr1 = tr.slice(t + 2, t + 8)
        tr1.data[0] = 10
        assert tr.data[2] == 10
        assert tr.data.ctypes.data == mempos
        assert tr.data[2:9].ctypes.data == tr1.data.ctypes.data
        assert tr1.data.ctypes.data - 8 == mempos

        # Test the processing information for the slicing. The sliced trace
        # should have a processing information showing that it has been
        # trimmed. The original trace should have nothing.
        tr = Trace(data=np.arange(10, dtype=np.int32))
        tr2 = tr.slice(tr.stats.starttime)
        assert "processing" not in tr.stats
        assert "processing" in tr2.stats
        assert "trim" in tr2.stats.processing[0]

    def test_slice_no_starttime_or_endtime(self):
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
        assert tr_trim == tr.slice(starttime=t2)
        tr2 = tr.slice(starttime=t2, endtime=t4)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        assert tr_trim == tr2

        # test 2: only removing data at right side
        tr_trim = tr_orig.copy()
        tr_trim.trim(endtime=t3)
        assert tr_trim == tr.slice(endtime=t3)
        tr2 = tr.slice(starttime=t1, endtime=t3)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        assert tr_trim == tr2

        # test 3: not removing data at all
        tr_trim = tr_orig.copy()
        tr_trim.trim(starttime=t1, endtime=t4)
        tr2 = tr.slice()
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        assert tr_trim == tr2

        tr2 = tr.slice(starttime=t1)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        assert tr_trim == tr2

        tr2 = tr.slice(endtime=t4)
        self.__remove_processing(tr2)
        assert tr_trim == tr2

        tr2 = tr.slice(starttime=t1, endtime=t4)
        self.__remove_processing(tr2)
        assert tr_trim == tr2

        tr_trim.trim()
        tr2 = tr.slice()
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        assert tr_trim == tr2

        tr2 = tr.slice(starttime=t1)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        assert tr_trim == tr2

        tr2 = tr.slice(endtime=t4)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        assert tr_trim == tr2

        tr2 = tr.slice(starttime=t1, endtime=t4)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        assert tr_trim == tr2

        # test 4: removing data at left and right side
        tr_trim = tr_orig.copy()
        tr_trim.trim(starttime=t2, endtime=t3)
        assert tr_trim == tr.slice(t2, t3)
        assert tr_trim == tr.slice(starttime=t2, endtime=t3)

        # test 5: no data left after operation
        tr_trim = tr_orig.copy()
        tr_trim.trim(starttime=t4)

        tr2 = tr.slice(starttime=t4)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        assert tr_trim == tr2

        tr2 = tr.slice(starttime=t4, endtime=t4 + 1)
        self.__remove_processing(tr_trim)
        self.__remove_processing(tr2)
        assert tr_trim == tr2

    def test_slice_nearest_sample(self):
        """
        Tests slicing with the nearest sample flag set to on or off.
        """
        tr = Trace(data=np.arange(6))
        # Samples at:
        # 0       10       20       30       40       50
        tr.stats.sampling_rate = 0.1

        # Nearest sample flag defaults to true.
        tr2 = tr.slice(UTC(4), UTC(44))
        assert tr2.stats.starttime == UTC(0)
        assert tr2.stats.endtime == UTC(40)

        tr2 = tr.slice(UTC(8), UTC(48))
        assert tr2.stats.starttime == UTC(10)
        assert tr2.stats.endtime == UTC(50)

        # Setting it to False changes the returned values.
        tr2 = tr.slice(UTC(4), UTC(44), nearest_sample=False)
        assert tr2.stats.starttime == UTC(10)
        assert tr2.stats.endtime == UTC(40)

        tr2 = tr.slice(UTC(8), UTC(48), nearest_sample=False)
        assert tr2.stats.starttime == UTC(10)
        assert tr2.stats.endtime == UTC(40)

    def test_trim_floating_point(self):
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
        assert tr.stats.starttime == UTC(0)
        assert tr.stats.endtime == UTC(10)
        # Create temp trace object used for testing.
        st = tr.stats.starttime
        # This is supposed to include the start and end times and should
        # therefore cut right at 2 and 8.
        temp = deepcopy(tr)
        temp.trim(st + 2.1, st + 7.1)
        # Should be identical.
        temp2 = deepcopy(tr)
        temp2.trim(st + 2.0, st + 8.0)
        assert temp.stats.starttime == UTC(2)
        assert temp.stats.endtime == UTC(7)
        assert temp.stats.npts == 6
        assert temp2.stats.npts == 7
        # self.assertEqual(temp.stats, temp2.stats)
        np.testing.assert_array_equal(temp.data, temp2.data[:-1])
        # Create test array that allows for easy testing.
        # Check if the data is the same.
        assert temp.data.ctypes.data != tr.data[2:9].ctypes.data
        np.testing.assert_array_equal(tr.data[2:8], temp.data)
        # Using out of bounds times should not do anything but create
        # a copy of the stats.
        temp = deepcopy(tr)
        temp.trim(st - 2.5, st + 200)
        # The start and end times should not change.
        assert temp.stats.starttime == UTC(0)
        assert temp.stats.endtime == UTC(10)
        assert temp.stats.npts == 11
        # Alter the new stats to make sure the old one stays intact.
        temp.stats.starttime = UTC(1000)
        assert org_stats == tr.stats
        # Check if the data address is not the same, that is it is a copy
        assert temp.data.ctypes.data != tr.data.ctypes.data
        np.testing.assert_array_equal(tr.data, temp.data)
        # Make sure the original Trace object did not change.
        np.testing.assert_array_equal(tr.data, org_data)
        assert tr.data.ctypes.data == mem_pos
        assert tr.stats == org_stats
        # Use more complicated times and sampling rate.
        tr = Trace(data=np.arange(111))
        tr.stats.starttime = UTC(111.11111)
        tr.stats.sampling_rate = 50.0
        org_stats = deepcopy(tr.stats)
        org_data = deepcopy(tr.data)
        # Save memory position of array.
        mem_pos = tr.data.ctypes.data
        # Create temp trace object used for testing.
        temp = deepcopy(tr)
        temp.trim(UTC(111.22222), UTC(112.99999),
                  nearest_sample=False)
        # Should again be identical. XXX NOT!
        temp2 = deepcopy(tr)
        temp2.trim(UTC(111.21111), UTC(113.01111),
                   nearest_sample=False)
        np.testing.assert_array_equal(temp.data, temp2.data[1:-1])
        # Check stuff.
        assert temp.stats.starttime == UTC(111.23111)
        assert temp.stats.endtime == UTC(112.991110)
        # Check if the data is the same.
        temp = deepcopy(tr)
        temp.trim(UTC(0), UTC(1000 * 1000))
        assert temp.data.ctypes.data != tr.data.ctypes.data
        # starttime must be in conformance with sampling rate
        t = UTC(111.11111)
        assert temp.stats.starttime == t
        delta = int((tr.stats.starttime - t) * tr.stats.sampling_rate + .5)
        np.testing.assert_array_equal(tr.data, temp.data[delta:delta + 111])
        # Make sure the original Trace object did not change.
        np.testing.assert_array_equal(tr.data, org_data)
        assert tr.data.ctypes.data == mem_pos
        assert tr.stats == org_stats

    def test_trim_floating_point_with_padding_1(self):
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
        assert tr.stats.starttime == UTC(0)
        assert tr.stats.endtime == UTC(10)
        # Create temp trace object used for testing.
        st = tr.stats.starttime
        # Using out of bounds times should not do anything but create
        # a copy of the stats.
        temp = deepcopy(tr)
        temp.trim(st - 2.5, st + 200, pad=True)
        assert temp.stats.starttime.timestamp == -2.0
        assert temp.stats.endtime.timestamp == 200
        assert temp.stats.npts == 203
        mask = np.zeros(203).astype(np.bool_)
        mask[:2] = True
        mask[13:] = True
        np.testing.assert_array_equal(temp.data.mask, mask)
        # Alter the new stats to make sure the old one stays intact.
        temp.stats.starttime = UTC(1000)
        assert org_stats == tr.stats
        # Check if the data address is not the same, that is it is a copy
        assert temp.data.ctypes.data != tr.data.ctypes.data
        np.testing.assert_array_equal(tr.data, temp.data[2:13])
        # Make sure the original Trace object did not change.
        np.testing.assert_array_equal(tr.data, org_data)
        assert tr.data.ctypes.data == mem_pos
        assert tr.stats == org_stats

    def test_trim_floating_point_with_padding_2(self):
        """
        Use more complicated times and sampling rate.
        """
        tr = Trace(data=np.arange(111))
        tr.stats.starttime = UTC(111.11111)
        tr.stats.sampling_rate = 50.0
        org_stats = deepcopy(tr.stats)
        org_data = deepcopy(tr.data)
        # Save memory position of array.
        mem_pos = tr.data.ctypes.data
        # Create temp trace object used for testing.
        temp = deepcopy(tr)
        temp.trim(UTC(111.22222), UTC(112.99999),
                  nearest_sample=False)
        # Should again be identical.#XXX not
        temp2 = deepcopy(tr)
        temp2.trim(UTC(111.21111), UTC(113.01111),
                   nearest_sample=False)
        np.testing.assert_array_equal(temp.data, temp2.data[1:-1])
        # Check stuff.
        assert temp.stats.starttime == UTC(111.23111)
        assert temp.stats.endtime == UTC(112.991110)
        # Check if the data is the same.
        temp = deepcopy(tr)
        temp.trim(UTC(0), UTC(1000 * 1000), pad=True)
        assert temp.data.ctypes.data != tr.data.ctypes.data
        # starttime must be in conformance with sampling rate
        t = UTC(1969, 12, 31, 23, 59, 59, 991110)
        assert temp.stats.starttime == t
        delta = int((tr.stats.starttime - t) * tr.stats.sampling_rate + .5)
        np.testing.assert_array_equal(tr.data, temp.data[delta:delta + 111])
        # Make sure the original Trace object did not change.
        np.testing.assert_array_equal(tr.data, org_data)
        assert tr.data.ctypes.data == mem_pos
        assert tr.stats == org_stats

    def test_add_sanity(self):
        """
        Test sanity checks in __add__ method of the Trace object.
        """
        tr = Trace(data=np.arange(10))
        # you may only add a Trace object
        with pytest.raises(TypeError):
            tr.__add__(1234)
        with pytest.raises(TypeError):
            tr.__add__('1234')
        with pytest.raises(TypeError):
            tr.__add__([1, 2, 3, 4])
        # trace id
        tr2 = Trace()
        tr2.stats.station = 'TEST'
        with pytest.raises(TypeError):
            tr.__add__(tr2)
        # sample rate
        tr2 = Trace()
        tr2.stats.sampling_rate = 20
        with pytest.raises(TypeError):
            tr.__add__(tr2)
        # calibration factor
        tr2 = Trace()
        tr2.stats.calib = 20
        with pytest.raises(TypeError):
            tr.__add__(tr2)
        # data type
        tr2 = Trace()
        tr2.data = np.arange(10, dtype=np.float32)
        with pytest.raises(TypeError):
            tr.__add__(tr2)

    def test_add_overlaps_default_method(self):
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
        assert isinstance(tr.data, np.ma.masked_array)
        assert tr.data.tolist() == [0, 0, 0, 0, 0, None, None, 1, 1, 1, 1, 1]
        # 2 + 1  : 00000--11111
        tr = tr2 + tr1
        assert isinstance(tr.data, np.ma.masked_array)
        assert tr.data.tolist() == [0, 0, 0, 0, 0, None, None, 1, 1, 1, 1, 1]
        # 2
        # overlapping trace with same data
        # Trace 1: 0000000
        # Trace 2:      0000000
        tr1 = Trace(data=np.zeros(7))
        tr2 = Trace(data=np.zeros(7))
        tr2.stats.starttime = tr1.stats.starttime + 5
        # 1 + 2  : 000000000000
        tr = tr1 + tr2
        assert isinstance(tr.data, np.ndarray)
        np.testing.assert_array_equal(tr.data, np.zeros(12))
        # 2 + 1  : 000000000000
        tr = tr2 + tr1
        assert isinstance(tr.data, np.ndarray)
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
        assert isinstance(tr.data, np.ndarray)
        np.testing.assert_array_equal(tr.data, np.ones(10))
        # 2 + 1  : 1111111111
        tr = tr2 + tr1
        assert isinstance(tr.data, np.ndarray)
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
        assert isinstance(tr.data, np.ma.masked_array)
        assert tr.data.tolist() == [0, 0, 0, 0, 0, None, None, 0, 0, 0]
        # 2 + 1  : 00000--000
        tr = tr2 + tr1
        assert isinstance(tr.data, np.ma.masked_array)
        assert tr.data.tolist() == [0, 0, 0, 0, 0, None, None, 0, 0, 0]
        # 5
        # completely contained trace with same data until end
        # Trace 1: 1111111111
        # Trace 2: 1111111111
        tr1 = Trace(data=np.ones(10))
        tr2 = Trace(data=np.ones(10))
        # 1 + 2  : 1111111111
        tr = tr1 + tr2
        assert isinstance(tr.data, np.ndarray)
        np.testing.assert_array_equal(tr.data, np.ones(10))
        # 6
        # completely contained trace with differing data
        # Trace 1: 0000000000
        # Trace 2: 1111111111
        tr1 = Trace(data=np.zeros(10))
        tr2 = Trace(data=np.ones(10))
        # 1 + 2  : ----------
        tr = tr1 + tr2
        assert isinstance(tr.data, np.ma.masked_array)
        assert tr.data.tolist() == [None] * 10

    def test_add_with_different_sampling_rates(self):
        """
        Test __add__ method of the Trace object.
        """
        # 1 - different sampling rates for the same channel should fail
        tr1 = Trace(data=np.zeros(5))
        tr1.stats.sampling_rate = 200
        tr2 = Trace(data=np.zeros(5))
        tr2.stats.sampling_rate = 50
        with pytest.raises(TypeError):
            tr1.__add__(tr2)
        with pytest.raises(TypeError):
            tr2.__add__(tr1)
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

    def test_add_with_different_datatypes_or_id(self):
        """
        Test __add__ method of the Trace object.
        """
        # 1 - different data types for the same channel should fail
        tr1 = Trace(data=np.zeros(5, dtype=np.int32))
        tr2 = Trace(data=np.zeros(5, dtype=np.float32))
        with pytest.raises(TypeError):
            tr1.__add__(tr2)
        with pytest.raises(TypeError):
            tr2.__add__(tr1)
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
        with pytest.raises(TypeError):
            tr1.__add__(tr2)
        with pytest.raises(TypeError):
            tr3.__add__(tr4)
        with pytest.raises(TypeError):
            tr2.__add__(tr1)
        with pytest.raises(TypeError):
            tr4.__add__(tr3)

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
        with pytest.raises(NotImplementedError):
            tr1.__lt__(tr1)
        with pytest.raises(NotImplementedError):
            tr1.__le__(tr1)
        with pytest.raises(NotImplementedError):
            tr1.__gt__(tr1)
        with pytest.raises(NotImplementedError):
            tr1.__ge__(tr1)
        with pytest.raises(NotImplementedError):
            tr1.__lt__(tr2)
        with pytest.raises(NotImplementedError):
            tr1.__le__(tr2)
        with pytest.raises(NotImplementedError):
            tr1.__gt__(tr2)
        with pytest.raises(NotImplementedError):
            tr1.__ge__(tr2)
        # normal tests
        assert tr0 == tr0
        assert tr0 == tr1
        assert tr0 != tr2
        assert tr0 != tr3
        assert tr0 != tr4
        assert tr0 != tr5
        assert tr0 != tr6
        assert tr0 != tr7
        assert tr5 != tr0
        assert tr5 != tr1
        assert tr5 != tr2
        assert tr5 != tr3
        assert tr5 != tr4
        assert tr5 == tr5
        assert tr5 != tr6
        assert tr3 != tr6
        assert tr0 == tr0
        assert tr0 == tr1
        assert tr0 != tr2
        assert tr0 != tr3
        assert tr0 != tr4
        assert tr0 != tr5
        assert tr0 != tr6
        assert tr0 != tr7
        assert tr5 != tr0
        assert tr5 != tr1
        assert tr5 != tr2
        assert tr5 != tr3
        assert tr5 != tr4
        assert tr5 == tr5
        assert tr5 != tr6
        assert tr3 != tr6
        # some weirder tests against non-Trace objects
        for object in [0, 1, 0.0, 1.0, "", "test", True, False, [], [tr0],
                       set(), set(tr0), {}, {"test": "test"}, [], None, ]:
            assert tr0 != object

    def test_nearest_sample(self):
        """
        This test case shows that the libmseed is actually flooring the
        starttime to the next sample value, regardless if it is the nearest
        sample. The flag nearest_sample=True tries to avoids this and
        rounds it to the next actual possible sample point.
        """
        # set up
        trace = Trace(data=np.empty(10000))
        trace.stats.starttime = UTC("2010-06-20T20:19:40.000000Z")
        trace.stats.sampling_rate = 200.0
        # ltrim
        tr = deepcopy(trace)
        t = UTC("2010-06-20T20:19:51.494999Z")
        tr._ltrim(t - 3, nearest_sample=True)
        # see that it is actually rounded to the next sample point
        assert tr.stats.starttime == UTC("2010-06-20T20:19:48.495000Z")
        # Lots of tests follow that thoroughly check the cutting behavior
        # using nearest_sample=True/False
        # rtrim
        tr = deepcopy(trace)
        t = UTC("2010-06-20T20:19:51.494999Z")
        tr._rtrim(t + 7, nearest_sample=True)
        # see that it is actually rounded to the next sample point
        assert tr.stats.endtime == UTC("2010-06-20T20:19:58.495000Z")
        tr = deepcopy(trace)
        t = UTC("2010-06-20T20:19:51.495000Z")
        tr._rtrim(t + 7, nearest_sample=True)
        # see that it is actually rounded to the next sample point
        assert tr.stats.endtime == UTC("2010-06-20T20:19:58.495000Z")
        tr = deepcopy(trace)
        t = UTC("2010-06-20T20:19:51.495111Z")
        tr._rtrim(t + 7, nearest_sample=True)
        # see that it is actually rounded to the next sample point
        assert tr.stats.endtime == UTC("2010-06-20T20:19:58.495000Z")
        tr = deepcopy(trace)
        t = UTC("2010-06-20T20:19:51.497501Z")
        tr._rtrim(t + 7, nearest_sample=True)
        # see that it is actually rounded to the next sample point
        assert tr.stats.endtime == UTC("2010-06-20T20:19:58.500000Z")
        # rtrim
        tr = deepcopy(trace)
        t = UTC("2010-06-20T20:19:51.494999Z")
        tr._rtrim(t + 7, nearest_sample=False)
        # see that it is actually rounded to the next sample point
        assert tr.stats.endtime == UTC("2010-06-20T20:19:58.490000Z")
        tr = deepcopy(trace)
        t = UTC("2010-06-20T20:19:51.495000Z")
        tr._rtrim(t + 7, nearest_sample=False)
        # see that it is actually rounded to the next sample point
        assert tr.stats.endtime == UTC("2010-06-20T20:19:58.495000Z")
        tr = deepcopy(trace)
        t = UTC("2010-06-20T20:19:51.495111Z")
        tr._rtrim(t + 7, nearest_sample=False)
        # see that it is actually rounded to the next sample point
        assert tr.stats.endtime == UTC("2010-06-20T20:19:58.495000Z")
        tr = deepcopy(trace)
        t = UTC("2010-06-20T20:19:51.497500Z")
        tr._rtrim(t + 7, nearest_sample=False)
        # see that it is actually rounded to the next sample point
        assert tr.stats.endtime == UTC("2010-06-20T20:19:58.495000Z")

    def test_masked_array_to_string(self):
        """
        Masked arrays should be marked using __str__.
        """
        st = read()
        overlaptrace = st[0].copy()
        overlaptrace.stats.starttime += 1
        st.append(overlaptrace)
        st.merge()
        out = st[0].__str__()
        assert out.endswith('(masked)')

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
        assert tr.stats.starttime == UTC(0)
        assert tr.stats.npts == 101
        np.testing.assert_array_almost_equal(
            tr.data, np.concatenate([[0.0], np.cumsum(data)[:-1] * 0.1]))

    def test_issue_317(self):
        """
        Tests times after breaking a stream into parts and merging it again.
        """
        # create a sample trace
        org_trace = Trace(data=np.arange(22487))
        org_trace.stats.starttime = UTC()
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
            assert traces[i].stats.npts == npts
            assert sum_trace.stats.npts == (i + 1) * npts
            # check data
            np.testing.assert_array_equal(traces[i].data,
                                          np.arange(i * npts, (i + 1) * npts))
            np.testing.assert_array_equal(sum_trace.data,
                                          np.arange(0, (i + 1) * npts))
            # check delta
            assert traces[i].stats.delta == org_trace.stats.delta
            assert sum_trace.stats.delta == org_trace.stats.delta
            # check sampling rates
            diff = traces[i].stats.sampling_rate-org_trace.stats.sampling_rate
            assert round(abs(diff), 7) == 0
            diff = sum_trace.stats.sampling_rate-org_trace.stats.sampling_rate
            assert round(abs(diff), 7) == 0
            # check end times
            assert traces[i].stats.endtime == sum_trace.stats.endtime

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
        assert tr.__str__().startswith(".%t3u.. | 1970")

    def test_taper(self):
        """
        Test taper method of trace
        """
        data = np.ones(10)
        tr = Trace(data=data)
        tr.taper(max_percentage=0.05, type='cosine')
        for i in range(len(data)):
            assert tr.data[i] <= 1.
            assert tr.data[i] >= 0.

    def test_taper_onesided(self):
        """
        Test onesided taper method of trace
        """
        data = np.ones(11)
        tr = Trace(data=data)

        # overlong taper - raises UserWarning - ignoring
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", UserWarning)
            tr.taper(max_percentage=None, side="left")
        assert len(w) == 1
        assert w[0].category == UserWarning

        assert tr.data[:5].sum() < 5.
        assert tr.data[6:].sum() == 5.

        data = np.ones(11)
        tr = Trace(data=data)

        # overlong taper - raises UserWarning - ignoring
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", UserWarning)
            tr.taper(max_percentage=None, side="right")
        assert len(w) == 1
        assert w[0].category == UserWarning

        assert tr.data[:5].sum() == 5.
        assert tr.data[6:].sum() < 5.

    def test_taper_length(self):
        npts = 11
        type_ = "hann"

        data = np.ones(npts)
        tr = Trace(data=data, header={'sampling': 1.})

        # test an overlong taper request, still works but raises UserWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", UserWarning)
            tr.taper(max_percentage=0.7, max_length=int(npts / 2) + 1)
        assert len(w) == 1
        assert w[0].category == UserWarning

        data = np.ones(npts)
        tr = Trace(data=data, header={'sampling': 1.})
        # first 3 samples get tapered
        tr.taper(max_percentage=None, type=type_, side="left", max_length=3)
        # last 5 samples get tapered
        tr.taper(max_percentage=0.5, type=type_, side="right", max_length=None)
        assert np.all(tr.data[:3] < 1.)
        assert np.all(tr.data[3:6] == 1.)
        assert np.all(tr.data[6:] < 1.)

        data = np.ones(npts)
        tr = Trace(data=data, header={'sampling': 1.})
        # first 3 samples get tapered
        tr.taper(max_percentage=0.5, type=type_, side="left", max_length=3)
        # last 3 samples get tapered
        tr.taper(max_percentage=0.3, type=type_, side="right", max_length=5)
        assert np.all(tr.data[:3] < 1.)
        assert np.all(tr.data[3:8] == 1.)
        assert np.all(tr.data[8:] < 1.)

    def test_times(self):
        """
        Test if the correct times array is returned for normal traces and
        traces with gaps.
        """
        from matplotlib import __version__
        tr = Trace(data=np.ones(100))
        tr.stats.sampling_rate = 20
        delta = tr.stats.delta
        start = UTC(2000, 1, 1, 0, 0, 0, 0)
        tr.stats.starttime = start
        tm = tr.times()
        diff = tr.stats.endtime - tr.stats.starttime
        assert np.isclose(tm[-1], diff)
        tr.data = np.ma.ones(100)
        tr.data[30:40] = np.ma.masked
        tm = tr.times()
        assert np.alltrue(tr.data.mask == tm.mask)
        # test relative with reftime
        tr.data = np.ones(100)
        shift = 9.5
        reftime = start - shift
        got = tr.times(reftime=reftime)
        assert len(got) == tr.stats.npts
        expected = np.arange(shift, shift + 4.5 * delta, delta)
        np.testing.assert_allclose(got[:5], expected, rtol=1e-8)
        # test other options
        got = tr.times("UTCDateTime")
        expected = np.array([
            UTC(2000, 1, 1, 0, 0),
            UTC(2000, 1, 1, 0, 0, 0, 50000),
            UTC(2000, 1, 1, 0, 0, 0, 100000),
            UTC(2000, 1, 1, 0, 0, 0, 150000),
            UTC(2000, 1, 1, 0, 0, 0, 200000)], dtype=UTC)
        assert isinstance(got[0], UTC)
        np.testing.assert_allclose(
            [t_.timestamp for t_ in got[:5]],
            [t_.timestamp for t_ in expected], rtol=1e-17)
        got = tr.times("timestamp")
        expected = np.arange(0, 4.5 * delta, delta) + 946684800.0
        np.testing.assert_allclose(got[:5], expected, rtol=1e-17)
        got = tr.times("matplotlib")
        expected = np.array([
                10957.000000000000, 10957.000000578704, 10957.000001157407,
                10957.000001736111, 10957.000002314815])
        if parse_version(__version__) < parse_version('3.3'):
            expected = np.array([
                730120.00000000000000000000, 730120.00000057870056480169,
                730120.00000115740112960339, 730120.00000173610169440508,
                730120.00000231480225920677])
        np.testing.assert_allclose(got[:5], expected, rtol=1e-17)

    def test_modulo_operation(self):
        """
        Method for testing the modulo operation. Mainly tests part not covered
        by the doctests.
        """
        tr = Trace(data=np.arange(25))
        # Wrong type raises.
        with pytest.raises(TypeError):
            tr.__mod__(5.0)
        with pytest.raises(TypeError):
            tr.__mod__("123")
        # Needs to be a positive integer.
        with pytest.raises(ValueError):
            tr.__mod__(0)
        with pytest.raises(ValueError):
            tr.__mod__(-11)
        # If num is more then the number of samples, a copy will be returned.
        st = tr % 500
        assert tr == st[0]
        assert len(st) == 1
        assert not (tr.data is st[0].data)

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
        tr = Trace(data=np.arange(250))
        tr.stats.sampling_rate = 20
        tr.spectrogram(show=False)

    def test_raise_masked(self):
        """
        Tests that detrend() raises in case of a masked array. (see #498)
        """
        x = np.arange(10)
        x = np.ma.masked_inside(x, 3, 4)
        tr = Trace(x)
        with pytest.raises(NotImplementedError):
            tr.detrend()

    def test_split(self):
        """
        Tests split method of the Trace class.
        """
        # set up
        tr1 = Trace(data=np.arange(1000))
        tr1.stats.sampling_rate = 200
        start = UTC(2000, 1, 1, 0, 0, 0, 0)
        tr1.stats.starttime = start
        tr2 = Trace(data=np.arange(0, 1000)[::-1])
        tr2.stats.sampling_rate = 200
        tr2.stats.starttime = start + 10
        # add will create new trace with masked array
        trace = tr1 + tr2
        assert isinstance(trace.data, np.ma.masked_array)
        # split
        assert isinstance(trace, Trace)
        st = trace.split()
        assert isinstance(st, Stream)
        assert len(st[0]) == 1000
        assert len(st[1]) == 1000
        # check if have no masked arrays
        assert not isinstance(st[0].data, np.ma.masked_array)
        assert not isinstance(st[1].data, np.ma.masked_array)

    def test_split_empty_masked_array(self):
        """
        Test split method with a masked array without any data.
        """
        tr = Trace(data=np.ma.masked_all(100))

        assert isinstance(tr.data, np.ma.masked_array)
        assert isinstance(tr, Trace)

        st = tr.split()

        assert isinstance(st, Stream)
        assert len(st) == 0

    def test_split_masked_array_without_actually_masked_values(self):
        """
        Tests splitting a masked array without actually masked data.
        """
        # First non masked.
        tr = Trace(data=np.arange(100))
        st = tr.copy().split()
        assert len(st) == 1
        assert tr == st[0]
        assert not isinstance(st[0].data, np.ma.masked_array)

        # Now the same thing but with an initially masked array but no
        # masked values.
        tr = Trace(data=np.ma.arange(100))
        assert not tr.data.mask
        st = tr.copy().split()
        assert len(st) == 1
        assert tr == st[0]
        assert not isinstance(st[0].data, np.ma.masked_array)

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
        except Exception:
            pass

        assert patch.call_count == 1
        _, kwargs = patch.call_args

        # Make sure that every item of the trace is passed to the evalresp
        # function.
        for key in ["network", "station", "location", "channel"]:
            msg = "'%s' did not get passed on to evalresp" % key
            somekey = kwargs[key if key != "location" else "locid"]
            assert somekey == tr.stats[key], msg

    def test_issue_540(self):
        """
        Trim with pad=True and given fill value should not return a masked
        NumPy array.
        """
        # fill_value = None
        tr = read()[0]
        assert len(tr) == 3000
        tr.trim(starttime=tr.stats.starttime - 0.01,
                endtime=tr.stats.endtime + 0.01, pad=True, fill_value=None)
        assert len(tr) == 3002
        assert isinstance(tr.data, np.ma.masked_array)
        assert tr.data[0] is np.ma.masked
        assert tr.data[1] is not np.ma.masked
        assert tr.data[-2] is not np.ma.masked
        assert tr.data[-1] is np.ma.masked
        # fill_value = 999
        tr = read()[0]
        assert len(tr) == 3000
        tr.trim(starttime=tr.stats.starttime - 0.01,
                endtime=tr.stats.endtime + 0.01, pad=True, fill_value=999)
        assert len(tr) == 3002
        assert not isinstance(tr.data, np.ma.masked_array)
        assert tr.data[0] == 999
        assert tr.data[-1] == 999
        # given fill_value but actually no padding at all
        tr = read()[0]
        assert len(tr) == 3000
        tr.trim(starttime=tr.stats.starttime,
                endtime=tr.stats.endtime, pad=True, fill_value=-999)
        assert len(tr) == 3000
        assert not isinstance(tr.data, np.ma.masked_array)

    def test_resample(self):
        """
        Tests the resampling of traces.
        """
        tr = read()[0]

        assert tr.stats.sampling_rate == 100.0
        assert tr.stats.npts == 3000

        tr_2 = tr.copy().resample(sampling_rate=50.0)
        assert tr_2.stats.endtime == tr.stats.endtime - 1.0 / 100.0
        assert tr_2.stats.sampling_rate == 50.0
        assert tr_2.stats.starttime == tr.stats.starttime

        tr_3 = tr.copy().resample(sampling_rate=10.0)
        assert tr_3.stats.endtime == tr.stats.endtime - 9.0 / 100.0
        assert tr_3.stats.sampling_rate == 10.0
        assert tr_3.stats.starttime == tr.stats.starttime

        tr_4 = tr.copy()
        tr_4.data = np.require(tr_4.data,
                               dtype=tr_4.data.dtype.newbyteorder('>'))
        tr_4 = tr_4.resample(sampling_rate=10.0)
        assert tr_4.stats.endtime == tr.stats.endtime - 9.0 / 100.0
        assert tr_4.stats.sampling_rate == 10.0
        assert tr_4.stats.starttime == tr.stats.starttime

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
        assert temp_tr is tr
        assert isinstance(tr, Trace)
        assert tr.stats.npts > 0

        # Use the processing chain to check the results. The trim() methods
        # does not have an entry in the processing chain.
        pr = tr.stats.processing
        assert "trim" in pr[0]
        assert "filter" in pr[1] and "lowpass" in pr[1]
        assert "simulate" in pr[2]
        assert "trigger" in pr[3]
        assert "decimate" in pr[4]
        assert "resample" in pr[5]
        assert "differentiate" in pr[6]
        assert "integrate" in pr[7]
        assert "detrend" in pr[8]
        assert "taper" in pr[9]
        assert "normalize" in pr[10]

    def test_skip_empty_trace(self):
        tr = read()[0]
        t = tr.stats.endtime + 10
        tr.trim(t, t + 10)
        tr.detrend()
        tr.resample(400)
        tr.differentiate()
        tr.integrate()
        tr.taper(max_percentage=0.1)

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
            with pytest.raises(ValueError):
                Trace(data=d)

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
        tr.stats.starttime = UTC("2010-07-23T00:00:00")
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
        tr.stats.starttime = UTC("2004-06-16T00:00:00")
        # remove response
        del tr.stats.response
        filename = os.path.join(path, 'data', 'stationxml_BK.CMB.__.LKS.xml')
        inv = read_inventory(filename, format='StationXML')
        tr.attach_response(inv)

        # raises UserWarning: Stage gain not defined - ignoring
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", UserWarning)
            tr.remove_response()
        assert len(w) == 1
        assert w[0].category == UserWarning

    def test_processing_info_remove_response_and_sensitivity(self):
        """
        Tests adding processing info for remove_response() and
        remove_sensitivity().

        See #1247.
        """
        # remove_sensitivity() with response object attached to the trace.
        tr = read()[0]
        assert "processing" not in tr.stats
        tr.remove_sensitivity()
        assert "processing" in tr.stats
        assert len(tr.stats.processing) == 1
        assert tr.stats.processing[0].endswith(
            "remove_sensitivity(inventory=None)")

        # With passed inventory object.
        tr = read()[0]
        assert "processing" not in tr.stats
        tr.remove_sensitivity(inventory=read_inventory())
        assert "processing" in tr.stats
        assert len(tr.stats.processing) == 1
        assert "remove_sensitivity(inventory=<obspy.core.inventory." \
               "inventory.Inventory object " in tr.stats.processing[0]

        # remove_response()
        tr = read()[0]
        assert "processing" not in tr.stats
        tr.remove_response()
        assert "processing" in tr.stats
        assert len(tr.stats.processing) == 1
        assert "remove_response(" in tr.stats.processing[0]
        assert "inventory=None" in tr.stats.processing[0]

        # With passed inventory object.
        tr = read()[0]
        assert "processing" not in tr.stats
        tr.remove_response(inventory=read_inventory())
        assert "processing" in tr.stats
        assert len(tr.stats.processing) == 1
        assert "remove_response(" in tr.stats.processing[0]
        assert "inventory=<obspy.core.inventory.inventory.Inventory " \
               "object" in tr.stats.processing[0]

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

        assert "trim" in pr[0]
        assert "ObsPy %s: trim(endtime=None::fill_value=None::" \
            "nearest_sample=True::pad=False::starttime=%s)" % (
                __version__, repr(trimming_starttime)) == \
            pr[0]
        assert "filter" in pr[1]
        assert "simulate" in pr[2]
        assert "trigger" in pr[3]
        assert "decimate" in pr[4]
        assert "resample" in pr[5]
        assert "differentiate" in pr[6]
        assert "integrate" in pr[7]
        assert "detrend" in pr[8]
        assert "taper" in pr[9]
        assert "normalize" in pr[10]

    def test_no_processing_info_for_failed_operations(self):
        """
        If an operation fails, no processing information should be attached
        to the Trace object.
        """
        # create test Trace
        tr = Trace(data=np.arange(20))
        assert not ("processing" in tr.stats)
        # This decimation by a factor of 7 in this case would change the
        # end time of the time series. Therefore it fails.
        with pytest.raises(ValueError):
            tr.decimate(7, strict_length=True)
        # No processing should be applied yet.
        assert not ("processing" in tr.stats)

        # Test the same but this time with an already existing processing
        # information.
        tr = Trace(data=np.arange(20))
        tr.detrend()
        assert len(tr.stats.processing) == 1
        info = tr.stats.processing[0]

        with pytest.raises(ValueError):
            tr.decimate(7, strict_length=True)
        assert tr.stats.processing == [info]

    def test_meta(self):
        """
        Tests Trace.meta an alternative to Trace.stats
        """
        tr = Trace()
        tr.meta = Stats({'network': 'NW'})
        assert tr.stats.network == 'NW'
        tr.stats = Stats({'network': 'BW'})
        assert tr.meta.network == 'BW'

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
        assert int_tr.stats.delta == 0.003
        # Assert that the new end time is smaller than the old one. SAC at
        # times performs some extrapolation which we do not want to do here.
        assert int_tr.stats.endtime <= org_tr.stats.endtime
        # SAC extrapolates a bit which we don't want here. The deviations
        # to SAC are likely due to the fact that we use double precision
        # math while SAC uses single precision math.
        assert np.allclose(
            int_tr.data,
            interp_delta_0_003.data[:int_tr.stats.npts],
            rtol=1E-3)

        int_tr = org_tr.copy().interpolate(sampling_rate=1.0 / 0.077,
                                           method="weighted_average_slopes")
        # Assert that the sampling rate has been set correctly.
        assert int_tr.stats.delta == 0.077
        # Assert that the new end time is smaller than the old one. SAC
        # calculates one sample less in this case.
        assert int_tr.stats.endtime <= org_tr.stats.endtime
        assert np.allclose(
            int_tr.data[:interp_delta_0_077.stats.npts],
            interp_delta_0_077.data,
            rtol=1E-5)

        # Also test the other interpolation methods mainly by assuring the
        # correct SciPy function is called and everything stays internally
        # consistent. SciPy's functions are tested enough to be sure that
        # they work.
        for inter_type in ["linear", "nearest", "zero"]:
            with mock.patch("scipy.interpolate.interp1d") as patch:
                patch.return_value = lambda x: x
                org_tr.copy().interpolate(sampling_rate=0.5, method=inter_type)
            assert patch.call_count == 1
            assert patch.call_args[1]["kind"] == inter_type

            int_tr = org_tr.copy().interpolate(sampling_rate=0.5,
                                               method=inter_type)
            assert int_tr.stats.delta == 2.0
            assert int_tr.stats.endtime <= org_tr.stats.endtime

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
            assert patch.call_count == 1
            assert patch.call_args[1]["k"] == inter_type

            int_tr = org_tr.copy().interpolate(sampling_rate=0.5,
                                               method=inter_type)
            assert int_tr.stats.delta == 2.0
            assert int_tr.stats.endtime <= org_tr.stats.endtime

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
        tr.stats.starttime = UTC(0)

        org_tr = tr.copy()

        # Now this does not do much for now but actually just shifts the
        # samples.
        tr.interpolate(method="lanczos", sampling_rate=1.0, a=1,
                       time_shift=0.2)
        assert tr.stats.starttime == org_tr.stats.starttime + 0.2
        assert tr.stats.endtime == org_tr.stats.endtime + 0.2
        np.testing.assert_allclose(tr.data, org_tr.data, atol=1E-9)

        tr.interpolate(method="lanczos", sampling_rate=1.0, a=1,
                       time_shift=0.4)
        assert tr.stats.starttime == org_tr.stats.starttime + 0.6
        assert tr.stats.endtime == org_tr.stats.endtime + 0.6
        np.testing.assert_allclose(tr.data, org_tr.data, atol=1E-9)

        tr.interpolate(method="lanczos", sampling_rate=1.0, a=1,
                       time_shift=-0.6)
        assert tr.stats.starttime == org_tr.stats.starttime
        assert tr.stats.endtime == org_tr.stats.endtime
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
        assert tr.stats.starttime == org_tr.stats.starttime + 1.0
        assert tr.stats.endtime == org_tr.stats.endtime
        # The data naturally also changed.
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(tr.data, org_tr.data[1:], atol=1E-9)
        # Shift back. This time we will lose the last sample.
        tr.interpolate(method="lanczos", sampling_rate=1.0, a=50,
                       starttime=tr.stats.starttime,
                       time_shift=-0.2)
        assert tr.stats.starttime == org_tr.stats.starttime + 1.0
        assert tr.stats.endtime == org_tr.stats.endtime - 1.0
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
            assert tr.stats.starttime == interp_tr.stats.starttime
            assert tr.stats.endtime >= interp_tr.stats.endtime >= \
                   tr.stats.endtime - (1.0 / 0.3)

            # If the starttime is modified the new starttime will be used but
            # the end time will again be modified as little as possible.
            interp_tr = tr.copy().interpolate(sampling_rate=0.3,
                                              method=inter_type,
                                              starttime=tr.stats.starttime +
                                              5.0)
            assert tr.stats.starttime + 5.0 == interp_tr.stats.starttime
            assert tr.stats.endtime >= interp_tr.stats.endtime >= \
                   tr.stats.endtime - (1.0 / 0.3)

            # If npts is given it will be used to modify the end time.
            interp_tr = tr.copy().interpolate(sampling_rate=0.3,
                                              method=inter_type, npts=10)
            assert tr.stats.starttime == interp_tr.stats.starttime
            assert interp_tr.stats.npts == 10

            # If npts and starttime are given, both will be modified.
            interp_tr = tr.copy().interpolate(sampling_rate=0.3,
                                              method=inter_type,
                                              starttime=tr.stats.starttime +
                                              5.0, npts=10)
            assert tr.stats.starttime + 5.0 == interp_tr.stats.starttime
            assert interp_tr.stats.npts == 10

            # An earlier starttime will raise an exception. No extrapolation
            # is supported
            with pytest.raises(ValueError):
                tr.copy().interpolate(sampling_rate=1.0,
                                      starttime=tr.stats.starttime - 10.0)
            # As will too many samples that would overstep the end time bound.
            with pytest.raises(ValueError):
                tr.copy().interpolate(sampling_rate=1.0,
                                      npts=tr.stats.npts * 1E6)

            # A negative or zero desired sampling rate should raise.
            with pytest.raises(ValueError):
                tr.copy().interpolate(sampling_rate=0.0)
            with pytest.raises(ValueError):
                tr.copy().interpolate(sampling_rate=-1.0)

    def test_resample_new(self):
        """
        Tests if Trace.resample works as expected and test that issue #857 is
        resolved.
        """
        starttime = UTC("1970-01-01T00:00:00.000000Z")
        tr0 = Trace(np.sin(np.linspace(0, 2 * np.pi, 10)),
                    {'sampling_rate': 1.0,
                     'starttime': starttime})
        # downsample
        tr = tr0.copy()
        tr.resample(0.5, window='hann', no_filter=True)
        assert len(tr.data) == 5
        expected = np.array([0.19478735, 0.83618307, 0.32200221,
                             -0.7794053, -0.57356732])
        assert np.all(np.abs(tr.data - expected) < 1e-7)
        assert tr.stats.sampling_rate == 0.5
        assert tr.stats.delta == 2.0
        assert tr.stats.npts == 5
        assert tr.stats.starttime == starttime
        assert tr.stats.endtime == \
               starttime + tr.stats.delta * (tr.stats.npts - 1)

        # upsample
        tr = tr0.copy()
        tr.resample(2.0, window='hann', no_filter=True)
        assert len(tr.data) == 20
        assert tr.stats.sampling_rate == 2.0
        assert tr.stats.delta == 0.5
        assert tr.stats.npts == 20
        assert tr.stats.starttime == starttime
        assert tr.stats.endtime == \
               starttime + tr.stats.delta * (tr.stats.npts - 1)

        # downsample with non integer ratio
        tr = tr0.copy()
        tr.resample(0.75, window='hann', no_filter=True)
        assert len(tr.data) == int(10 * .75)
        expected = np.array([0.15425413, 0.66991128, 0.74610418, 0.11960477,
                             -0.60644662, -0.77403839, -0.30938935])
        assert np.all(np.abs(tr.data - expected) < 1e-7)
        assert tr.stats.sampling_rate == 0.75
        assert tr.stats.delta == 1 / 0.75
        assert tr.stats.npts == int(10 * .75)
        assert tr.stats.starttime == starttime
        assert tr.stats.endtime == \
               starttime + tr.stats.delta * (tr.stats.npts - 1)

        # downsample without window
        tr = tr0.copy()
        tr.resample(0.5, window=None, no_filter=True)
        assert len(tr.data) == 5
        assert tr.stats.sampling_rate == 0.5
        assert tr.stats.delta == 2.0
        assert tr.stats.npts == 5
        assert tr.stats.starttime == starttime
        assert tr.stats.endtime == \
               starttime + tr.stats.delta * (tr.stats.npts - 1)

        # downsample with window and automatic filtering
        tr = tr0.copy()
        tr.resample(0.5, window='hann', no_filter=False)
        assert len(tr.data) == 5
        assert tr.stats.sampling_rate == 0.5
        assert tr.stats.delta == 2.0
        assert tr.stats.npts == 5
        assert tr.stats.starttime == starttime
        assert tr.stats.endtime == \
               starttime + tr.stats.delta * (tr.stats.npts - 1)

        # downsample with custom window
        tr = tr0.copy()
        window = np.ones((tr.stats.npts))
        tr.resample(0.5, window=window, no_filter=True)

        # downsample with bad window
        tr = tr0.copy()
        window = np.array([0, 1, 2, 3])
        with pytest.raises(ValueError):
            tr.resample(sampling_rate=0.5, window=window, no_filter=True)

    def test_slide(self):
        """
        Tests for sliding a window across a trace object.
        """
        tr = Trace(data=np.linspace(0, 100, 101))
        tr.stats.starttime = UTC(0.0)
        tr.stats.sampling_rate = 5.0

        # First slice it in 4 pieces. Window length is in seconds.
        slices = []
        for window_tr in tr.slide(window_length=5.0, step=5.0):
            slices.append(window_tr)

        assert len(slices) == 4
        assert slices[0] == tr.slice(UTC(0), UTC(5))
        assert slices[1] == tr.slice(UTC(5), UTC(10))
        assert slices[2] == tr.slice(UTC(10), UTC(15))
        assert slices[3] == tr.slice(UTC(15), UTC(20))

        # Different step which is the distance between two windows measured
        # from the start of the first window in seconds.
        slices = []
        for window_tr in tr.slide(window_length=5.0, step=10.0):
            slices.append(window_tr)

        assert len(slices) == 2
        assert slices[0] == tr.slice(UTC(0), UTC(5))
        assert slices[1] == tr.slice(UTC(10), UTC(15))

        # Offset determines the initial starting point. It defaults to zero.
        slices = []
        for window_tr in tr.slide(window_length=5.0, step=6.5, offset=8.5):
            slices.append(window_tr)

        assert len(slices) == 2
        assert slices[0] == tr.slice(UTC(8.5), UTC(13.5))
        assert slices[1] == tr.slice(UTC(15.0), UTC(20.0))

        # By default only full length windows will be returned so any
        # remainder that can no longer make up a full window will not be
        # returned.
        slices = []
        for window_tr in tr.slide(window_length=15.0, step=15.0):
            slices.append(window_tr)

        assert len(slices) == 1
        assert slices[0] == tr.slice(UTC(0.0), UTC(15.0))

        # But it can optionally be returned.
        slices = []
        for window_tr in tr.slide(window_length=15.0, step=15.0,
                                  include_partial_windows=True):
            slices.append(window_tr)

        assert len(slices) == 2
        assert slices[0] == tr.slice(UTC(0.0), UTC(15.0))
        assert slices[1] == tr.slice(UTC(15.0), UTC(20.0))

        # Negative step lengths work together with an offset.
        slices = []
        for window_tr in tr.slide(window_length=5.0, step=-5.0, offset=20.0):
            slices.append(window_tr)

        assert len(slices) == 4
        assert slices[0] == tr.slice(UTC(15), UTC(20))
        assert slices[1] == tr.slice(UTC(10), UTC(15))
        assert slices[2] == tr.slice(UTC(5), UTC(10))
        assert slices[3] == tr.slice(UTC(0), UTC(5))

    def test_slide_nearest_sample(self):
        """
        Tests that the nearest_sample argument is correctly passed to the
        slice function calls.
        """
        tr = Trace(data=np.linspace(0, 100, 101))
        tr.stats.starttime = UTC(0.0)
        tr.stats.sampling_rate = 5.0

        # It defaults to True.
        with mock.patch("obspy.core.trace.Trace.slice") as patch:
            patch.return_value = tr
            list(tr.slide(5, 5))

        assert patch.call_count == 4
        for arg in patch.call_args_list:
            assert arg[1]["nearest_sample"]

        # Force True.
        with mock.patch("obspy.core.trace.Trace.slice") as patch:
            patch.return_value = tr
            list(tr.slide(5, 5, nearest_sample=True))

        assert patch.call_count == 4
        for arg in patch.call_args_list:
            assert arg[1]["nearest_sample"]

        # Set to False.
        with mock.patch("obspy.core.trace.Trace.slice") as patch:
            patch.return_value = tr
            list(tr.slide(5, 5, nearest_sample=False))

        assert patch.call_count == 4
        for arg in patch.call_args_list:
            assert not arg[1]["nearest_sample"]

    def test_remove_response_plot(self, image_path):
        """
        Tests the plotting option of remove_response().
        """
        tr = read("/path/to/IU_ULN_00_LH1_2015-07-18T02.mseed")[0]
        inv = read_inventory("/path/to/IU_ULN_00_LH1.xml")
        tr.attach_response(inv)
        pre_filt = [0.001, 0.005, 10, 20]
        tr.remove_response(pre_filt=pre_filt, output="DISP",
                           water_level=60, end_stage=None, plot=image_path)

    def test_remove_response_default_units(self):
        """
        Tests remove_response() with default units for a hydrophone.
        """
        tr = read("/path/to/1T_MONN_00_EDH.mseed")[0]
        inv = read_inventory("/path/to/1T_MONN_00_EDH.xml")
        tr.attach_response(inv)
        tr.remove_response(output='DEF')
        np.testing.assert_almost_equal(tr.max(), 54.833, decimal=3)

    def test_normalize(self):
        """
        Tests the normalize() method on normal and edge cases.
        """
        # Nothing should happen with ones.
        tr = Trace(data=np.ones(5))
        tr.normalize()
        np.testing.assert_allclose(tr.data, np.ones(5))

        # 10s should be normalized to all ones.
        tr = Trace(data=10 * np.ones(5))
        tr.normalize()
        np.testing.assert_allclose(tr.data, np.ones(5))

        # Negative 10s should be normalized to negative ones.
        tr = Trace(data=-10 * np.ones(5))
        tr.normalize()
        np.testing.assert_allclose(tr.data, -np.ones(5))

        # 10s and a couple of 5s should be normalized to 1s and a couple of
        # 0.5s.
        tr = Trace(data=np.array([10.0, 10.0, 5.0, 5.0]))
        tr.normalize()
        np.testing.assert_allclose(tr.data, np.array([1.0, 1.0, 0.5, 0.5]))

        # Same but negative values.
        tr = Trace(data=np.array([-10.0, -10.0, -5.0, -5.0]))
        tr.normalize()
        np.testing.assert_allclose(tr.data, np.array([-1.0, -1.0, -0.5, -0.5]))

        # Mixed values.
        tr = Trace(data=np.array([-10.0, -10.0, 5.0, 5.0]))
        tr.normalize()
        np.testing.assert_allclose(tr.data, np.array([-1.0, -1.0, 0.5, 0.5]))

        # Mixed values.
        tr = Trace(data=np.array([-10.0, 10.0, -5.0, 5.0]))
        tr.normalize()
        np.testing.assert_allclose(tr.data, np.array([-1.0, 1.0, -0.5, 0.5]))

        # Mixed values.
        tr = Trace(data=np.array([-10.0, -10.0, 0.0, 0.0]))
        tr.normalize()
        np.testing.assert_allclose(tr.data, np.array([-1.0, -1.0, 0.0, 0.0]))

        # Mixed values.
        tr = Trace(data=np.array([10.0, 10.0, 0.0, 0.0]))
        tr.normalize()
        np.testing.assert_allclose(tr.data, np.array([1.0, 1.0, 0.0, 0.0]))

        # Small values get larger.
        tr = Trace(data=np.array([-0.5, 0.5, 0.1, -0.1]))
        tr.normalize()
        np.testing.assert_allclose(tr.data, np.array([-1.0, 1.0, 0.2, -0.2]))

        # All zeros. Nothing should happen but a warning will be raised.
        tr = Trace(data=np.array([-0.0, 0.0, 0.0, -0.0]))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tr.normalize()
        assert w[0].category == UserWarning
        msg = "Attempting to normalize by dividing through zero."
        assert msg in w[0].message.args[0]
        np.testing.assert_allclose(tr.data, np.array([-0.0, 0.0, 0.0, -0.0]))

        # Passing the norm specifies the division factor.
        tr = Trace(data=np.array([10.0, 10.0, 0.0, 0.0]))
        tr.normalize(norm=2)
        np.testing.assert_allclose(tr.data, np.array([5.0, 5.0, 0.0, 0.0]))

        # Passing the norm specifies the division factor. Nothing happens
        # with zero.
        tr = Trace(data=np.array([10.0, 10.0, 0.0, 0.0]))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tr.normalize(norm=0)
        assert w[0].category == UserWarning
        msg = "Attempting to normalize by dividing through zero."
        assert msg in w[0].message.args[0]
        np.testing.assert_allclose(tr.data, np.array([10.0, 10.0, 0.0, 0.0]))

        # Warning is raised for a negative norm, but the positive value is
        # used.
        tr = Trace(data=np.array([10.0, 10.0, 0.0, 0.0]))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tr.normalize(norm=-2)

        assert w[0].category == UserWarning
        msg = "Normalizing with negative values is forbidden."
        assert msg in w[0].message.args[0]

        np.testing.assert_allclose(tr.data, np.array([5.0, 5.0, 0.0, 0.0]))

    def test_dtype_is_not_unnecessarily_changed(self):
        """
        The dtype of the data should not change if not necessary. In general
        this means that a float32 array should not become a float64 array
        and vice-versa. Integer arrays will always be upcasted to float64
        arrays when integer output makes no sense. Not all int32 numbers can be
        accurately represented by float32 arrays so double precision is
        required in order to not lose accuracy.

        Exceptions are custom coded C routines where we usually opt to only
        include either a single or a double precision version.
        """
        tr = read()[0]
        tr.data = tr.data[:100]

        # One for each common input dtype.
        tr_int32 = tr.copy()
        tr_int32.data = np.require(tr_int32.data, dtype=np.int32)
        tr_int64 = tr.copy()
        tr_int64.data = np.require(tr_int64.data, dtype=np.int64)
        tr_float32 = tr.copy()
        tr_float32.data = np.require(tr_float32.data, dtype=np.float32)
        tr_float64 = tr.copy()
        tr_float64.data = np.require(tr_float64.data, dtype=np.float64)

        # Trimming.
        assert tr_int32.copy().trim(1, 2).data.dtype == np.int32
        assert tr_int64.copy().trim(1, 2).data.dtype == np.int64
        assert tr_float32.copy().trim(1, 2).data.dtype == np.float32
        assert tr_float64.copy().trim(1, 2).data.dtype == np.float64

        # Filtering. SciPy converts data to 64bit floats. Filters are
        # numerically tricky so a higher accuracy is justified here.
        lowpass = tr_int32.copy().filter("lowpass", freq=2.0)
        assert lowpass.data.dtype == np.float64
        lowpass = tr_int64.copy().filter("lowpass", freq=2.0)
        assert lowpass.data.dtype == np.float64
        lowpass = tr_float32.copy().filter("lowpass", freq=2.0)
        assert lowpass.data.dtype == np.float64
        lowpass = tr_float64.copy().filter("lowpass", freq=2.0)
        assert lowpass.data.dtype == np.float64

        # Decimation should not change the dtype.
        decimate = tr_int32.copy().decimate(factor=2, no_filter=True)
        assert decimate.data.dtype == np.int32
        decimate = tr_int64.copy().decimate(factor=2, no_filter=True)
        assert decimate.data.dtype == np.int64
        decimate = tr_float32.copy().decimate(factor=2, no_filter=True)
        assert decimate.data.dtype == np.float32
        decimate = tr_float64.copy().decimate(factor=2, no_filter=True)
        assert decimate.data.dtype == np.float64

        # Detrending will upcast integers but should not touch floats.
        assert tr_int32.copy().detrend("simple").data.dtype == np.float64
        assert tr_int64.copy().detrend("simple").data.dtype == np.float64
        assert tr_float32.copy().detrend("simple").data.dtype == np.float32
        assert tr_float64.copy().detrend("simple").data.dtype == np.float64
        assert tr_int32.copy().detrend("linear").data.dtype == np.float64
        assert tr_int64.copy().detrend("linear").data.dtype == np.float64
        assert tr_float32.copy().detrend("linear").data.dtype == np.float32
        assert tr_float64.copy().detrend("linear").data.dtype == np.float64
        assert tr_int32.copy().detrend("constant").data.dtype == np.float64
        assert tr_int64.copy().detrend("constant").data.dtype == np.float64
        assert tr_float32.copy().detrend("constant").data.dtype == np.float32
        assert tr_float64.copy().detrend("constant").data.dtype == np.float64
        detrend = tr_int32.copy().detrend("polynomial", order=3)
        assert detrend.data.dtype == np.float64
        detrend = tr_int64.copy().detrend("polynomial", order=3)
        assert detrend.data.dtype == np.float64
        detrend = tr_float32.copy().detrend("polynomial", order=3)
        assert detrend.data.dtype == np.float32
        detrend = tr_float64.copy().detrend("polynomial", order=3)
        assert detrend.data.dtype == np.float64
        detrend = tr_int32.copy().detrend("spline", order=3, dspline=100)
        assert detrend.data.dtype == np.float64
        detrend = tr_int64.copy().detrend("spline", order=3, dspline=100)
        assert detrend.data.dtype == np.float64
        detrend = tr_float32.copy().detrend("spline", order=3, dspline=100)
        assert detrend.data.dtype == np.float32
        detrend = tr_float64.copy().detrend("spline", order=3, dspline=100)
        assert detrend.data.dtype == np.float64

        # Tapering. Upcast to float64 but don't change float32.
        assert tr_int32.copy().taper(0.05, "hann").data.dtype == np.float64
        assert tr_int64.copy().taper(0.05, "hann").data.dtype == np.float64
        assert tr_float32.copy().taper(0.05, "hann").data.dtype == np.float32
        assert tr_float64.copy().taper(0.05, "hann").data.dtype == np.float64

        # Normalizing. Upcast to float64 but don't change float32.
        assert tr_int32.copy().normalize().data.dtype == np.float64
        assert tr_int64.copy().normalize().data.dtype == np.float64
        assert tr_float32.copy().normalize().data.dtype == np.float32
        assert tr_float64.copy().normalize().data.dtype == np.float64

        # Differentiate. Upcast to float64 but don't change float32.
        assert tr_int32.copy().differentiate().data.dtype == np.float64
        assert tr_int64.copy().differentiate().data.dtype == np.float64
        assert tr_float32.copy().differentiate().data.dtype == np.float32
        assert tr_float64.copy().differentiate().data.dtype == np.float64

        # Integrate. Upcast to float64 but don't change float32.
        integrate = tr_int32.copy().integrate(method="cumtrapz")
        assert integrate.data.dtype == np.float64
        integrate = tr_int64.copy().integrate(method="cumtrapz")
        assert integrate.data.dtype == np.float64
        integrate = tr_float32.copy().integrate(method="cumtrapz")
        assert integrate.data.dtype == np.float32
        integrate = tr_float64.copy().integrate(method="cumtrapz")
        assert integrate.data.dtype == np.float64
        # The spline antiderivate always returns float64.
        integrate = tr_int32.copy().integrate(method="spline")
        assert integrate.data.dtype == np.float64
        integrate = tr_int64.copy().integrate(method="spline")
        assert integrate.data.dtype == np.float64
        integrate = tr_float32.copy().integrate(method="spline")
        assert integrate.data.dtype == np.float64
        integrate = tr_float64.copy().integrate(method="spline")
        assert integrate.data.dtype == np.float64

        # Simulation is an operation in the spectral domain so double
        # precision is a lot more accurate so it's fine here.
        paz_remove = {'poles': [-0.037004 + 0.037016j, -0.037004 - 0.037016j,
                                -251.33 + 0j],
                      'zeros': [0j, 0j], 'gain': 60077000.0,
                      'sensitivity': 2516778400.0}
        sim = tr_int32.copy().simulate(paz_remove=paz_remove)
        assert sim.data.dtype == np.float64
        sim = tr_int64.copy().simulate(paz_remove=paz_remove)
        assert sim.data.dtype == np.float64
        sim = tr_float32.copy().simulate(paz_remove=paz_remove)
        assert sim.data.dtype == np.float64
        sim = tr_float64.copy().simulate(paz_remove=paz_remove)
        assert sim.data.dtype == np.float64

        # Same with the fourier domain resampling.
        assert tr_int32.copy().resample(2.0).data.dtype == np.float64
        assert tr_int64.copy().resample(2.0).data.dtype == np.float64
        assert tr_float32.copy().resample(2.0).data.dtype == np.float64
        assert tr_float64.copy().resample(2.0).data.dtype == np.float64

        # Same with remove_response()
        inv = read_inventory()
        dtype = tr_int32.copy().remove_response(inventory=inv).data.dtype
        assert dtype == np.float64
        dtype = tr_int64.copy().remove_response(inventory=inv).data.dtype
        assert dtype == np.float64
        dtype = tr_float32.copy().remove_response(inventory=inv).data.dtype
        assert dtype == np.float64
        dtype = tr_float64.copy().remove_response(inventory=inv).data.dtype
        assert dtype == np.float64

        # Remove sensitivity does not have to change the dtype for float32.
        dtype = tr_int32.copy().remove_sensitivity(inventory=inv).data.dtype
        assert dtype == np.float64
        dtype = tr_int64.copy().remove_sensitivity(inventory=inv).data.dtype
        assert dtype == np.float64
        dtype = tr_float32.copy().remove_sensitivity(inventory=inv).data.dtype
        assert dtype == np.float32
        dtype = tr_float64.copy().remove_sensitivity(inventory=inv).data.dtype
        assert dtype == np.float64

        # Various interpolation routines.
        # Weighted average slopes is a custom C routine that only works with
        # double precision.
        assert tr_int32.copy().interpolate(
                1.0, method="weighted_average_slopes").data.dtype == np.float64
        assert tr_int64.copy().interpolate(
                1.0, method="weighted_average_slopes").data.dtype == np.float64
        assert tr_float32.copy().interpolate(
                1.0, method="weighted_average_slopes").data.dtype == np.float64
        assert tr_float64.copy().interpolate(
                1.0, method="weighted_average_slopes").data.dtype == np.float64
        # Scipy treats splines as double precision. No need to convert them.
        assert tr_int32.copy().interpolate(
                1.0, method="slinear").data.dtype == np.float64
        assert tr_int64.copy().interpolate(
                1.0, method="slinear").data.dtype == np.float64
        assert tr_float32.copy().interpolate(
                1.0, method="slinear").data.dtype == np.float64
        assert tr_float64.copy().interpolate(
                1.0, method="slinear").data.dtype == np.float64
        # Lanczos is a custom C routine that only works with double precision.
        assert tr_int32.copy().interpolate(
                1.0, method="lanczos", a=2).data.dtype == np.float64
        assert tr_int64.copy().interpolate(
                1.0, method="lanczos", a=2).data.dtype == np.float64
        assert tr_float32.copy().interpolate(
                1.0, method="lanczos", a=2).data.dtype == np.float64
        assert tr_float64.copy().interpolate(
                1.0, method="lanczos", a=2).data.dtype == np.float64

    def test_set_trace_id(self):
        """
        Test setter of `id` property.
        """
        tr = Trace()
        tr.stats.location = "00"
        # check setting net/sta/loc/cha with an ID
        tr.id = "GR.FUR..HHZ"
        assert tr.stats.network == "GR"
        assert tr.stats.station == "FUR"
        assert tr.stats.location == ""
        assert tr.stats.channel == "HHZ"
        # check that invalid types will raise
        invalid = (True, False, -10, 0, 1.0, [1, 4, 3, 2], np.ones(4))
        for id_ in invalid:
            with pytest.raises(TypeError):
                tr.id = id_
        # check that invalid ID strings will raise
        invalid = ("ABCD", "AB.CD", "AB.CD.00", "..EE", "....",
                   "GR.FUR..HHZ.", "GR.FUR..HHZ...")
        for id_ in invalid:
            with pytest.raises(ValueError):
                tr.id = id_

    def test_trace_contiguous(self):
        """
        Test that arbitrary operations on Trace.data will always result in
        Trace.data being C- and F-contiguous, unless explicitly opted out.
        """
        tr_default = Trace(data=np.arange(5, dtype=np.int32))
        tr_opt_out = Trace(data=np.arange(5, dtype=np.int32))
        tr_opt_out._always_contiguous = False
        # the following slicing operation only creates a view internally in
        # numpy and thus leaves the array incontiguous
        tr_default.data = tr_default.data[::2]
        tr_opt_out.data = tr_opt_out.data[::2]
        # by default it should have made contiguous, nevertheless
        assert tr_default.data.flags['C_CONTIGUOUS']
        assert tr_default.data.flags['F_CONTIGUOUS']
        # if opted out explicitly, it should be incontiguous due to the slicing
        # operation
        assert not tr_opt_out.data.flags['C_CONTIGUOUS']
        assert not tr_opt_out.data.flags['F_CONTIGUOUS']

    def test_header_dict_copied(self):
        """
        Regression test for #1934 (collisions when using  the same header
        dictionary for multiple Trace inits)
        """
        header = {'station': 'MS', 'starttime': 1}
        original_header = deepcopy(header)

        # Init two traces and make sure the original header did not change.
        tr1 = Trace(data=np.ones(2), header=header)
        assert header == original_header
        tr2 = Trace(data=np.zeros(5), header=header)
        assert header == original_header

        assert len(tr1) == 2
        assert len(tr2) == 5
        assert tr1.stats.npts == 2
        assert tr2.stats.npts == 5

    def test_pickle(self):
        """
        Test that  Trace can be pickled #1989
        """
        tr_orig = Trace()
        tr_pickled = pickle.loads(pickle.dumps(tr_orig, protocol=0))
        assert tr_orig == tr_pickled
        tr_pickled = pickle.loads(pickle.dumps(tr_orig, protocol=1))
        assert tr_orig == tr_pickled
        tr_pickled = pickle.loads(pickle.dumps(tr_orig, protocol=2))
        assert tr_orig == tr_pickled

    def test_pickle_soh(self):
        """
        Test that trace can be pickled with samplerate = 0 #1989
        """
        tr_orig = Trace()
        tr_orig.stats.sampling_rate = 0
        tr_pickled = pickle.loads(pickle.dumps(tr_orig, protocol=0))
        assert tr_orig == tr_pickled
        tr_pickled = pickle.loads(pickle.dumps(tr_orig, protocol=1))
        assert tr_orig == tr_pickled
        tr_pickled = pickle.loads(pickle.dumps(tr_orig, protocol=2))
        assert tr_orig == tr_pickled

    def test_deepcopy_issue2600(self):
        """
        Tests correct deepcopy of Trace and Stats objects, issue 2600
        """
        tr = Trace(data=np.ones(2), header={'sampling_rate': 1e5})
        sr1 = tr.stats.sampling_rate
        sr2 = tr.copy().stats.sampling_rate
        assert sr1 == 1e5
        assert sr2 == sr1

    def test_pickle_issue2600(self):
        """
        Tests correct pickle of Trace and Stats objects, issue 2600
        """
        tr = Trace(data=np.ones(2), header={'sampling_rate': 1e5})
        sr1 = tr.stats.sampling_rate
        tr_pickled = pickle.loads(pickle.dumps(tr, protocol=2))
        sr2 = tr_pickled.stats.sampling_rate
        assert sr1 == 1e5
        assert sr2 == sr1

    def test_resample_short_traces(self):
        """
        Tests that resampling of short traces leaves at least one sample
        """
        tr = Trace(data=np.ones(2), header={'sampling_rate': 100})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", UserWarning)
            tr.resample(30)

        assert len(w) == 1
        assert w[0].category == UserWarning

        assert tr.stats.sampling_rate == 30
        assert tr.data.shape[0] == 1
