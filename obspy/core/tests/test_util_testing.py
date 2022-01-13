# -*- coding: utf-8 -*-
"""
Tests for obspy's testing utilities.
"""
import numpy as np

from obspy import read, Trace
from obspy.core.util.testing import streams_almost_equal, traces_almost_equal


class TestAlmostEqual:
    """
    Tests for fuzzy equality comparisons for traces.
    """

    def test_identical_traces(self):
        """
        Should return True on identical streams but false if a value is
        greatly changed.
        """
        tr1, tr2 = read()[0], read()[0]
        assert traces_almost_equal(tr1, tr2)
        # when one number is changed significantly it should return False
        tr1.data[0] = (tr1.data[0] + 1) * 1000
        assert not traces_almost_equal(tr1, tr2)

    def test_slightly_modified_data(self):
        """
        Traces that are "close" should be considered almost equal.
        """
        tr1, tr2 = read()[0], read()[0]
        # alter one trace's data slightly
        tr1.data *= (1. + 1e-6)
        assert traces_almost_equal(tr1, tr2)

    def test_empty_traces(self):
        """
        Empty traces should be considered almost equal.
        """
        tr1, tr2 = Trace(), Trace()
        assert traces_almost_equal(tr1, tr2)

    def test_different_stats_no_processing(self):
        """
        If only the stats are different traces should not be considered almost
        equal.
        """
        tr1 = Trace(header=dict(network='UU', station='TMU', channel='HHZ'))
        tr2 = Trace(header=dict(network='UU', station='TMU', channel='HHN'))
        assert not traces_almost_equal(tr1, tr2)
        assert not traces_almost_equal(tr2, tr1)

    def test_processing(self):
        """
        Differences in processing attr of stats should only count if
        processing is True.
        """
        tr1, tr2 = read()[0], read()[0]
        # Detrend each traces once, then second trace twice for two entries
        # in processing.
        tr1.detrend()
        tr2.detrend()
        tr1.detrend()
        assert traces_almost_equal(tr1, tr2, default_stats=True)
        assert not traces_almost_equal(tr1, tr2, default_stats=False)

    def test_nan(self):
        """
        Ensure NaNs eval equal if equal_nan is used, else they do not.
        """
        tr1, tr2 = read()[0], read()[0]
        tr1.data[0], tr2.data[0] = np.NaN, np.NaN
        assert traces_almost_equal(tr1, tr2, equal_nan=True)
        assert not traces_almost_equal(tr1, tr2, equal_nan=False)

    def test_unequal_trace_lengths(self):
        """
        Ensure traces with different lengths are not almost equal.
        """
        tr1, tr2 = read()[0], read()[0]
        tr2.data = tr2.data[:-1]
        assert not traces_almost_equal(tr1, tr2)

    def test_not_a_trace(self):
        """
        Ensure comparing to someething that is not a trace returns False.
        """
        tr1 = read()[0]
        assert not traces_almost_equal(tr1, 1)
        assert not traces_almost_equal(tr1, None)
        assert not traces_almost_equal(tr1, 'not a trace')

    def test_stream_almost_equal(self):
        """
        Basic tests for almost equal. More rigorous testing is done on the
        Trace's almost_equal method, which gets called by this one.
        """
        # identical streams should be almost equal
        st1, st2 = read(), read()
        assert streams_almost_equal(st1, st2)
        # passing something other than a stream should not be almost equal
        assert not streams_almost_equal(st1, None)
        assert not streams_almost_equal(st1, 1.1)
        # passing streams of different lengths should not be almost equal
        st2 = st2[1:]
        assert not streams_almost_equal(st1, st2)
