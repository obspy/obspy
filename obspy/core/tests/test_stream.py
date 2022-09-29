# -*- coding: utf-8 -*-
import inspect
import io
import os
import pickle
import platform
import warnings
from copy import deepcopy
from os.path import dirname, join, abspath
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from obspy import Stream, Trace, UTCDateTime, read, read_inventory
from obspy.core.inventory import Channel, Inventory, Network, Station
from obspy.core.stream import _is_pickle, _read_pickle, _write_pickle
from obspy.core.util.attribdict import AttribDict
from obspy.core.util.base import NamedTemporaryFile, _get_entry_points
from obspy.core.util.obspy_types import ObsPyException
from obspy.core.util.testing import streams_almost_equal
from obspy.io.xseed import Parser


class TestStream:
    """
    Test suite for obspy.core.stream.Stream.
    """
    _current_file = inspect.getfile(inspect.currentframe())
    data_path = join(dirname(abspath(_current_file)), "data")

    @pytest.fixture()
    def mseed_stream(self):
        """Create a stream for testing."""
        rand = np.random.RandomState(815)
        header = {'network': 'BW', 'station': 'BGLD',
                  'starttime': UTCDateTime(2007, 12, 31, 23, 59, 59, 915000),
                  'npts': 412, 'sampling_rate': 200.0,
                  'channel': 'EHE'}
        trace1 = Trace(data=rand.randint(0, 1000, 412).astype(np.float64),
                       header=deepcopy(header))
        header['starttime'] = UTCDateTime(2008, 1, 1, 0, 0, 4, 35000)
        header['npts'] = 824
        trace2 = Trace(data=rand.randint(0, 1000, 824).astype(np.float64),
                       header=deepcopy(header))
        header['starttime'] = UTCDateTime(2008, 1, 1, 0, 0, 10, 215000)
        trace3 = Trace(data=rand.randint(0, 1000, 824).astype(np.float64),
                       header=deepcopy(header))
        header['starttime'] = UTCDateTime(2008, 1, 1, 0, 0, 18, 455000)
        header['npts'] = 50668
        trace4 = Trace(
            data=rand.randint(0, 1000, 50668).astype(np.float64),
            header=deepcopy(header))
        return Stream(traces=[trace1, trace2, trace3, trace4])

    @pytest.fixture()
    def gse2_stream(self):
        """Another example stream."""
        rand = np.random.RandomState(815)
        header = {'network': '', 'station': 'RNON ', 'location': '',
                  'starttime': UTCDateTime(2004, 6, 9, 20, 5, 59, 849998),
                  'sampling_rate': 200.0, 'npts': 12000,
                  'channel': '  Z'}
        trace = Trace(
            data=rand.randint(0, 1000, 12000).astype(np.float64),
            header=header)
        return Stream(traces=[trace])

    @staticmethod
    def __remove_processing(st):
        """
        Helper method removing the processing information from all traces
        within a Stream object.

        Useful for testing.
        """
        for tr in st:
            if "processing" not in tr.stats:
                continue
            del tr.stats.processing

    def test_init(self):
        """
        Tests the __init__ method of the Stream object.
        """
        # empty
        st = Stream()
        assert len(st) == 0
        # single trace
        st = Stream(Trace())
        assert len(st) == 1
        # array of traces
        st = Stream([Trace(), Trace()])
        assert len(st) == 2

    def test_setitem(self, mseed_stream):
        """
        Tests the __setitem__ method of the Stream object.
        """
        stream = mseed_stream
        stream[0] = stream[3]
        assert stream[0] == stream[3]
        st = deepcopy(stream)
        stream[0].data[0:10] = 999
        assert st[0].data[0] != 999
        st[0] = stream[0]
        np.testing.assert_array_equal(stream[0].data[:10],
                                      np.ones(10, dtype=np.int_) * 999)

    def test_getitem(self):
        """
        Tests the __getitem__ method of the Stream object.
        """
        stream = read()
        assert stream[0] == stream.traces[0]
        assert stream[-1] == stream.traces[-1]
        assert stream[2] == stream.traces[2]
        # out of index should fail
        with pytest.raises(IndexError):
            stream.__getitem__(3)
        with pytest.raises(IndexError):
            stream.__getitem__(-99)

    def test_add(self, mseed_stream, gse2_stream):
        """
        Tests the adding of two stream objects.
        """
        stream = mseed_stream
        assert 4 == len(stream)
        # Add the same stream object to itself.
        stream = stream + stream
        assert 8 == len(stream)
        # This will not create copies of Traces and thus the objects should
        # be identical (and the Traces attributes should be identical).
        for _i in range(4):
            assert (stream[_i] == stream[_i + 4])
            assert (stream[_i] is stream[_i + 4])
        # Now add another stream to it.
        other_stream = gse2_stream
        assert 1 == len(other_stream)
        new_stream = stream + other_stream
        assert 9 == len(new_stream)
        # The traces of all streams are copied.
        for _i in range(8):
            assert new_stream[_i] == stream[_i]
            assert new_stream[_i] is stream[_i]
        # Also test for the newly added stream.
        assert new_stream[8] == other_stream[0]
        assert new_stream[8].stats == other_stream[0].stats
        np.testing.assert_array_equal(new_stream[8].data, other_stream[0].data)
        # adding something else than stream or trace results into TypeError
        with pytest.raises(TypeError):
            stream.__add__(1)
        with pytest.raises(TypeError):
            stream.__add__('test')

    def test_iadd(self, mseed_stream, gse2_stream):
        """
        Tests the __iadd__ method of the Stream objects.
        """
        stream = mseed_stream
        assert 4 == len(stream)
        other_stream = gse2_stream
        assert 1 == len(other_stream)
        # Add the other stream to the stream.
        stream += other_stream
        # This will leave the Traces of the new stream and create a deepcopy of
        # the other Stream's Traces
        assert 5 == len(stream)
        assert other_stream[0] == stream[-1]
        assert other_stream[0].stats == stream[-1].stats
        np.testing.assert_array_equal(other_stream[0].data, stream[-1].data)
        # adding something else than stream or trace results into TypeError
        with pytest.raises(TypeError):
            stream.__iadd__(1)
        with pytest.raises(TypeError):
            stream.__iadd__('test')

    def test_mul(self):
        """
        Tests the __mul__ method of the Stream objects.
        """
        st = Stream(Trace())
        assert len(st) == 1
        st = st * 4
        assert len(st) == 4
        # multiplying by something else than an integer results into TypeError
        with pytest.raises(TypeError):
            st.__mul__(1.2345)
        with pytest.raises(TypeError):
            st.__mul__('test')

    def test_add_trace_to_stream(self):
        """
        Tests using a Trace on __add__ and __iadd__ methods of the Stream.
        """
        st0 = read()
        st1 = st0[0:2]
        tr = st0[2]
        # __add__
        assert st1.__add__(tr) == st0
        assert st1 + tr == st0
        # __iadd__
        st1 += tr
        assert st1 == st0

    def test_append(self, mseed_stream):
        """
        Tests the append method of the Stream object.
        """
        stream = mseed_stream
        # Check current count of traces
        assert len(stream) == 4
        # Append first traces to the Stream object.
        stream.append(stream[0])
        assert len(stream) == 5
        # This is supposed to make a deepcopy of the Trace and thus the two
        # Traces are not identical.
        assert stream[0] == stream[-1]
        # But the attributes and data values should be identical.
        assert stream[0].stats == stream[-1].stats
        np.testing.assert_array_equal(stream[0].data, stream[-1].data)
        # Append the same again
        stream.append(stream[0])
        assert len(stream) == 6
        # Now the two objects should be identical.
        assert stream[0] == stream[-1]
        # Using append with a list of Traces, or int, or ... should fail.
        with pytest.raises(TypeError):
            stream.append(stream[:])
        with pytest.raises(TypeError):
            stream.append(1)
        with pytest.raises(TypeError):
            stream.append(stream[0].data)

    def test_count_and_len(self):
        """
        Tests the count and __len__ methods of the Stream object.
        """
        # empty stream without traces
        stream = Stream()
        assert len(stream) == 0
        assert stream.count() == 0
        # stream with traces
        stream = read()
        assert len(stream) == 3
        assert stream.count() == 3

    def test_extend(self, mseed_stream):
        """
        Tests the extend method of the Stream object.
        """
        stream = mseed_stream
        # Check current count of traces
        assert len(stream) == 4
        # Extend the Stream object with the first two traces.
        stream.extend(stream[0:2])
        assert len(stream) == 6
        # This is NOT supposed to make a deepcopy of the Trace and thus the two
        # Traces compare equal and are identical.
        assert stream[0] == stream[-2]
        assert stream[1] == stream[-1]
        assert stream[0] is stream[-2]
        assert stream[1] is stream[-1]
        # Using extend with a single Traces, or a wrong list, or ...
        # should fail.
        with pytest.raises(TypeError):
            stream.extend(stream[0])
        with pytest.raises(TypeError):
            stream.extend(1)
        with pytest.raises(TypeError):
            stream.extend([stream[0], 1])

    def test_insert(self, mseed_stream):
        """
        Tests the insert Method of the Stream object.
        """
        stream = mseed_stream
        assert 4 == len(stream)
        # Insert the last Trace before the second trace.
        stream.insert(1, stream[-1])
        assert len(stream) == 5
        # This is supposed to make a deepcopy of the Trace and thus the two
        # Traces are not identical.
        # self.assertNotEqual(stream[1], stream[-1])
        assert stream[1] == stream[-1]
        # But the attributes and data values should be identical.
        assert stream[1].stats == stream[-1].stats
        np.testing.assert_array_equal(stream[1].data, stream[-1].data)
        # Do the same again
        stream.insert(1, stream[-1])
        assert len(stream) == 6
        # Now the two Traces should be identical
        assert stream[1] == stream[-1]
        # Do the same with a list of traces this time.
        # Insert the last two Trace before the second trace.
        stream.insert(1, stream[-2:])
        assert len(stream) == 8
        # This is supposed to make a deepcopy of the Trace and thus the two
        # Traces are not identical.
        assert stream[1] == stream[-2]
        assert stream[2] == stream[-1]
        # But the attributes and data values should be identical.
        assert stream[1].stats == stream[-2].stats
        np.testing.assert_array_equal(stream[1].data, stream[-2].data)
        assert stream[2].stats == stream[-1].stats
        np.testing.assert_array_equal(stream[2].data, stream[-1].data)
        # Do the same again
        stream.insert(1, stream[-2:])
        assert len(stream) == 10
        # Now the two Traces should be identical
        assert stream[1] == stream[-2]
        assert stream[2] == stream[-1]
        # Using insert without a single Traces or a list of Traces should fail.
        with pytest.raises(TypeError):
            stream.insert(1, 1)
        with pytest.raises(TypeError):
            stream.insert(stream[0], stream[0])
        with pytest.raises(TypeError):
            stream.insert(1, [stream[0], 1])

    def test_get_gaps(self, mseed_stream):
        """
        Tests the get_gaps method of the Stream objects.

        It is compared directly to the obspy.io.mseed method getGapsList which
        is assumed to be correct.
        """
        stream = mseed_stream
        gap_list = stream.get_gaps()
        # Gaps list created with obspy.io.mseed
        mseed_gap_list = [
            ('BW', 'BGLD', '', 'EHE',
             UTCDateTime(2008, 1, 1, 0, 0, 1, 970000),
             UTCDateTime(2008, 1, 1, 0, 0, 4, 35000),
             2.0599999999999999, 412.0),
            ('BW', 'BGLD', '', 'EHE',
             UTCDateTime(2008, 1, 1, 0, 0, 8, 150000),
             UTCDateTime(2008, 1, 1, 0, 0, 10, 215000),
             2.0599999999999999, 412.0),
            ('BW', 'BGLD', '', 'EHE',
             UTCDateTime(2008, 1, 1, 0, 0, 14, 330000),
             UTCDateTime(2008, 1, 1, 0, 0, 18, 455000),
             4.120, 824.0)]
        # Assert the number of gaps.
        assert len(mseed_gap_list) == len(gap_list)
        for _i in range(len(mseed_gap_list)):
            # Compare the string values directly.
            for _j in range(6):
                assert gap_list[_i][_j] == mseed_gap_list[_i][_j]
            # The small differences are probably due to rounding errors.
            gap_6 = mseed_gap_list[_i][6]
            gap_7 = mseed_gap_list[_i][7]
            assert round(abs(float(gap_6)-float(gap_6)), 3) == 0
            assert round(abs(float(gap_7)-float(gap_7)), 3) == 0

    def test_get_gaps_multiplexed_streams(self):
        """
        Tests the get_gaps method of the Stream objects.
        """
        data = np.random.randint(0, 1000, 412)
        # different channels
        st = Stream()
        for channel in ['EHZ', 'EHN', 'EHE']:
            st.append(Trace(data=data, header={'channel': channel}))
        assert len(st.get_gaps()) == 0
        # different locations
        st = Stream()
        for location in ['', '00', '01']:
            st.append(Trace(data=data, header={'location': location}))
        assert len(st.get_gaps()) == 0
        # different stations
        st = Stream()
        for station in ['MANZ', 'ROTZ', 'BLAS']:
            st.append(Trace(data=data, header={'station': station}))
        assert len(st.get_gaps()) == 0
        # different networks
        st = Stream()
        for network in ['BW', 'GE', 'GR']:
            st.append(Trace(data=data, header={'network': network}))
        assert len(st.get_gaps()) == 0

    def test_get_gaps_masked(self):
        """
        Test get_gaps method of the Stream objects (Issue #2299)
        """
        # Create a Stream with a masked array (analogous to a merged Stream)
        st = Stream()
        data = np.ma.array(np.arange(0, 100, step=1))
        data.mask = np.zeros(data.shape)
        data.mask[50:60] = 1
        st.append(Trace(data=data))
        # Expected gap
        gap = ["", "", "", "",
               UTCDateTime(1970, 1, 1, 0, 0, 49),
               UTCDateTime(1970, 1, 1, 0, 1, 0),
               10., 10]
        # Get the gaps
        gaps = st.get_gaps()
        # Assert the number of gaps
        assert len(gaps) == 1
        # Verify the resulting gap list matches what is expected
        for _i in range(6):
            assert gaps[0][_i] == gap[_i]
        assert round(abs(float(gaps[0][6])-float(gap[6])), 3) == 0
        assert round(abs(float(gaps[0][7])-float(gap[7])), 7) == 0
        # Double-check that the initial Stream is unmodified
        assert len(st) == 1

    def test_pop(self, mseed_stream):
        """
        Test the pop method of the Stream object.
        """
        stream = mseed_stream
        # Make a copy of the Traces.
        traces = deepcopy(stream[:])
        # Remove and return the last Trace.
        temp_trace = stream.pop()
        assert 3 == len(stream)
        # Assert attributes. The objects itself are not identical.
        assert temp_trace.stats == traces[-1].stats
        np.testing.assert_array_equal(temp_trace.data, traces[-1].data)
        # Remove the last copied Trace.
        traces.pop()
        # Remove and return the second Trace.
        temp_trace = stream.pop(1)
        # Assert attributes. The objects itself are not identical.
        assert temp_trace.stats == traces[1].stats
        np.testing.assert_array_equal(temp_trace.data, traces[1].data)
        # Remove the second copied Trace.
        traces.pop(1)
        # Compare all remaining Traces.
        assert 2 == len(stream)
        assert 2 == len(traces)
        for _i in range(len(traces)):
            assert traces[_i].stats == stream[_i].stats
            np.testing.assert_array_equal(traces[_i].data, stream[_i].data)

    def test_slicing(self):
        """
        Tests the __getslice__ method of the Stream object.
        """
        stream = read()
        assert stream[0:] == stream[0:]
        assert stream[:2] == stream[:2]
        assert stream[:] == stream[:]
        assert len(stream) == 3
        new_stream = stream[1:3]
        assert isinstance(new_stream, Stream)
        assert len(new_stream) == 2
        assert new_stream[0].stats == stream[1].stats
        assert new_stream[1].stats == stream[2].stats

    def test_slicing_with_steps(self):
        """
        Tests the __getslice__ method of the Stream object with step.
        """
        tr1 = Trace()
        tr2 = Trace()
        tr3 = Trace()
        tr4 = Trace()
        tr5 = Trace()
        st = Stream([tr1, tr2, tr3, tr4, tr5])
        assert st[0:6].traces == [tr1, tr2, tr3, tr4, tr5]
        assert st[0:6:1].traces == [tr1, tr2, tr3, tr4, tr5]
        assert st[0:6:2].traces == [tr1, tr3, tr5]
        assert st[1:6:2].traces == [tr2, tr4]
        assert st[1:6:6].traces == [tr2]

    def test_slice(self):
        """
        Slice method should not loose attributes set on stream object itself.
        """
        st = read()
        st.test = 1
        st.muh = "Muh"
        st2 = st.slice(st[0].stats.starttime, st[0].stats.endtime)
        assert st2.test == 1
        assert st2.muh == "Muh"

    def test_slice_nearest_sample(self):
        """
        Tests that the nearest_sample argument is correctly passed to the
        trace function calls.
        """
        # It defaults to True.
        st = read()
        utc1 = st[0].stats.starttime + 0.1
        utc2 = st[0].stats.endtime - 0.1
        with mock.patch("obspy.core.trace.Trace.slice") as patch:
            patch.return_value = st[0]
            st.slice(utc1, utc2)
        assert patch.call_count == 3
        for arg in patch.call_args_list:
            assert arg[1]["nearest_sample"]

        # Force True.
        with mock.patch("obspy.core.trace.Trace.slice") as patch:
            patch.return_value = st[0]
            st.slice(utc1, utc2, nearest_sample=True)
        assert patch.call_count == 3
        for arg in patch.call_args_list:
            assert arg[1]["nearest_sample"]

        # Set to False.
        with mock.patch("obspy.core.trace.Trace.slice") as patch:
            patch.return_value = st[0]
            st.slice(1, 2, nearest_sample=False)
        assert patch.call_count == 3
        for arg in patch.call_args_list:
            assert not arg[1]["nearest_sample"]

    def test_cutout(self):
        """
        Test cutout method of the Stream object. Compare against equivalent
        trimming operations.
        """
        t1 = UTCDateTime("2009-06-24")
        t2 = UTCDateTime("2009-08-24T00:20:06.007Z")
        t3 = UTCDateTime("2009-08-24T00:20:16.008Z")
        t4 = UTCDateTime("2011-09-11")
        st = read()
        st_cut = read()
        # 1
        st_cut.cutout(t4, t4 + 10)
        self.__remove_processing(st_cut)
        assert st == st_cut
        # 2
        st_cut.cutout(t1 - 10, t1)
        self.__remove_processing(st_cut)
        assert st == st_cut
        # 3
        st_cut.cutout(t1, t2)
        st.trim(starttime=t2, nearest_sample=True)
        self.__remove_processing(st_cut)
        self.__remove_processing(st)
        assert st == st_cut
        # 4
        st = read()
        st_cut = read()
        st_cut.cutout(t3, t4)
        st.trim(endtime=t3, nearest_sample=True)
        self.__remove_processing(st_cut)
        self.__remove_processing(st)
        assert st == st_cut
        # 5
        st = read()
        st.trim(endtime=t2, nearest_sample=True)
        tmp = read()
        tmp.trim(starttime=t3, nearest_sample=True)
        st += tmp
        st_cut = read()
        st_cut.cutout(t2, t3)
        self.__remove_processing(st_cut)
        self.__remove_processing(st)
        assert st == st_cut

    def test_pop2(self):
        """
        Test the pop method of the Stream object.
        """
        trace = Trace(data=np.arange(0, 1000))
        st = Stream([trace])
        st = st + st + st + st
        assert len(st) == 4
        st.pop()
        assert len(st) == 3
        st[1].stats.station = 'MUH'
        st.pop(0)
        assert len(st) == 2
        assert st[0].stats.station == 'MUH'

    def test_remove(self, mseed_stream):
        """
        Tests the remove method of the Stream object.
        """
        stream = mseed_stream
        # Make a copy of the Traces.
        stream2 = deepcopy(stream)
        # Use the remove method of the Stream object and of the list of Traces.
        stream.remove(stream[1])
        del stream2[1]
        stream.remove(stream[-1])
        del stream2[-1]
        # Compare remaining Streams.
        assert stream == stream2

    def test_reverse(self, mseed_stream):
        """
        Tests the reverse method of the Stream object.
        """
        stream = mseed_stream
        # Make a copy of the Traces.
        traces = deepcopy(stream[:])
        # Use reversing of the Stream object and of the list.
        stream.reverse()
        traces.reverse()
        # Compare all Traces.
        assert 4 == len(stream)
        assert 4 == len(traces)
        for _i in range(len(traces)):
            assert traces[_i].stats == stream[_i].stats
            np.testing.assert_array_equal(traces[_i].data, stream[_i].data)

    def test_select(self):
        """
        Tests the select method of the Stream object.
        """
        # Create a list of header dictionaries.
        headers = [
            {'starttime': UTCDateTime(1990, 1, 1), 'network': 'AA',
             'station': 'ZZZZ', 'channel': 'EHZ', 'sampling_rate': 200.0,
             'npts': 100},
            {'starttime': UTCDateTime(1990, 1, 1), 'network': 'BB',
             'station': 'YYYY', 'channel': 'EHN', 'sampling_rate': 200.0,
             'npts': 100},
            {'starttime': UTCDateTime(2000, 1, 1), 'network': 'AA',
             'station': 'ZZZZ', 'channel': 'BHZ', 'sampling_rate': 20.0,
             'npts': 100},
            {'starttime': UTCDateTime(1989, 1, 1), 'network': 'BB',
             'station': 'XXXX', 'channel': 'BHN', 'sampling_rate': 20.0,
             'npts': 100},
            {'starttime': UTCDateTime(2010, 1, 1), 'network': 'AA',
             'station': 'XXXX', 'channel': 'EHZ', 'sampling_rate': 200.0,
             'npts': 100, 'location': '00'}]
        # Make stream object for test case
        traces = []
        for header in headers:
            traces.append(Trace(data=np.random.randint(0, 1000, 100),
                                header=header))
        stream = Stream(traces=traces)
        # Test cases:
        stream2 = stream.select()
        assert stream == stream2
        with pytest.raises(Exception):
            stream.select(channel="EHZ",
                          component="N")
        stream2 = stream.select(channel='EHE')
        assert len(stream2) == 0
        stream2 = stream.select(channel='EHZ')
        assert len(stream2) == 2
        assert stream[0] in stream2
        assert stream[4] in stream2
        stream2 = stream.select(component='Z')
        assert len(stream2) == 3
        assert stream[0] in stream2
        assert stream[2] in stream2
        assert stream[4] in stream2
        stream2 = stream.select(component='n')
        assert len(stream2) == 2
        assert stream[1] in stream2
        assert stream[3] in stream2
        stream2 = stream.select(channel='BHZ', npts=100, sampling_rate='20.0',
                                network='AA', component='Z', station='ZZZZ')
        assert len(stream2) == 1
        assert stream[2] in stream2
        stream2 = stream.select(channel='EHZ', station="XXXX")
        assert len(stream2) == 1
        assert stream[4] in stream2
        stream2 = stream.select(network='AA')
        assert len(stream2) == 3
        assert stream[0] in stream2
        assert stream[2] in stream2
        assert stream[4] in stream2
        stream2 = stream.select(id='AA.ZZZZ..EHZ')
        assert len(stream2) == 1
        assert stream[0] in stream2
        stream2 = stream.select(sampling_rate=20.0)
        assert len(stream2) == 2
        assert stream[2] in stream2
        assert stream[3] in stream2
        # tests for wildcarded channel:
        stream2 = stream.select(channel='B*')
        assert len(stream2) == 2
        assert stream[2] in stream2
        assert stream[3] in stream2
        stream2 = stream.select(channel='EH*')
        assert len(stream2) == 3
        assert stream[0] in stream2
        assert stream[1] in stream2
        assert stream[4] in stream2
        stream2 = stream.select(channel='*Z')
        assert len(stream2) == 3
        assert stream[0] in stream2
        assert stream[2] in stream2
        assert stream[4] in stream2
        # tests for other wildcard operations:
        stream2 = stream.select(station='[XY]*')
        assert len(stream2) == 3
        assert stream[1] in stream2
        assert stream[3] in stream2
        assert stream[4] in stream2
        stream2 = stream.select(station='[A-Y]*')
        assert len(stream2) == 3
        assert stream[1] in stream2
        assert stream[3] in stream2
        assert stream[4] in stream2
        stream2 = stream.select(station='[A-Y]??*', network='A?')
        assert len(stream2) == 1
        assert stream[4] in stream2
        # test case insensitivity
        stream2 = stream.select(channel='BhZ', npts=100, sampling_rate='20.0',
                                network='aA', station='ZzZz',)
        assert len(stream2) == 1
        assert stream[2] in stream2
        stream2 = stream.select(channel='e?z', network='aa', station='x?X*',
                                location='00', component='z')
        assert len(stream2) == 1
        assert stream[4] in stream2

    def test_select_on_single_letter_channels(self):
        st = read()
        st[0].stats.channel = "Z"
        st[1].stats.channel = "N"
        st[2].stats.channel = "E"

        assert [tr.stats.channel for tr in st] == ["Z", "N", "E"]

        assert st.select(component="Z")[0] == st[0]
        assert st.select(component="N")[0] == st[1]
        assert st.select(component="E")[0] == st[2]

        assert len(st.select(component="Z")) == 1
        assert len(st.select(component="N")) == 1
        assert len(st.select(component="E")) == 1

    def test_select_from_inventory(self):
        # Create a test stream
        headers = [
            {'network': 'AA', 'station': 'ST01', 'channel': 'EHZ',
             'location': '00', 'sampling_rate': 200.0, 'npts': 100,
             'starttime': UTCDateTime(1990, 1, 1)},
            {'network': 'AA', 'station': 'ST01', 'channel': 'EHZ',
             'location': '00', 'sampling_rate': 200.0, 'npts': 100,
             'starttime': UTCDateTime(1998, 1, 1)},
            {'network': 'AA', 'station': 'ST01', 'channel': 'EHZ',
             'location': '00', 'sampling_rate': 200.0, 'npts': 100,
             'starttime': UTCDateTime(2000, 1, 1)},
            {'network': 'AA', 'station': 'ST01', 'channel': 'EHN',
             'location': '00', 'sampling_rate': 200.0, 'npts': 100,
             'starttime': UTCDateTime(2000, 1, 1)},
            {'network': 'AA', 'station': 'ST01', 'channel': 'EHE',
             'location': '00', 'sampling_rate': 200.0, 'npts': 100,
             'starttime': UTCDateTime(2000, 1, 1)},
            {'network': 'AA', 'station': 'ST02', 'channel': 'EHZ',
             'location': '00', 'sampling_rate': 200.0, 'npts': 100,
             'starttime': UTCDateTime(2000, 1, 1)},
            {'network': 'AB', 'station': 'ST01', 'channel': 'EHZ',
             'location': '00', 'sampling_rate': 200.0, 'npts': 100,
             'starttime': UTCDateTime(2000, 1, 1)}
        ]
        traces = []
        for header in headers:
            traces.append(Trace(data=np.random.randint(0, 1000, 100),
                                header=header))
        st = Stream(traces=traces)
        # Create a test inventory
        channels = [
            Channel(code='EHZ', location_code='00',
                    latitude=0.0, longitude=0.0, elevation=0.0, depth=0.0,
                    start_date=UTCDateTime(1990, 1, 1),
                    end_date=UTCDateTime(1998, 1, 1)),
            Channel(code='EHZ', location_code='00',
                    latitude=0.0, longitude=0.0, elevation=0.0, depth=0.0,
                    start_date=UTCDateTime(1999, 1, 1))
        ]
        stations = [Station(code='ST01',
                            latitude=0.0,
                            longitude=0.0,
                            elevation=0.0,
                            channels=channels)]
        networks = [Network('AA', stations=stations)]
        inv = Inventory(networks=networks, source='TEST')
        # tests
        assert len(st.select(inventory=inv)) == 2
        st_sel = st.select(network='AB')
        assert len(st_sel.select(inventory=inv)) == 0
        inv_sel = inv.select(starttime=UTCDateTime(2000, 1, 1))
        assert len(st.select(inventory=inv_sel)) == 1

    def test_sort(self):
        """
        Tests the sort method of the Stream object.
        """
        # Create new Stream
        stream = Stream()
        # Create a list of header dictionaries. The sampling rate serves as a
        # unique identifier for each Trace.
        headers = [
            {'starttime': UTCDateTime(1990, 1, 1), 'network': 'AAA',
             'station': 'ZZZ', 'channel': 'XXX', 'sampling_rate': 100.0},
            {'starttime': UTCDateTime(1990, 1, 1), 'network': 'AAA',
             'station': 'YYY', 'channel': 'CCC', 'sampling_rate': 200.0},
            {'starttime': UTCDateTime(2000, 1, 1), 'network': 'AAA',
             'station': 'EEE', 'channel': 'GGG', 'sampling_rate': 300.0},
            {'starttime': UTCDateTime(1989, 1, 1), 'network': 'AAA',
             'station': 'XXX', 'channel': 'GGG', 'sampling_rate': 400.0},
            {'starttime': UTCDateTime(2010, 1, 1), 'network': 'AAA',
             'station': 'XXX', 'channel': 'FFF', 'sampling_rate': 500.0}]
        # Create a Trace object of it and append it to the Stream object.
        for _i in headers:
            new_trace = Trace(header=_i)
            stream.append(new_trace)
        # Use normal sorting.
        stream.sort()
        assert [i.stats.sampling_rate for i in stream.traces] == \
               [300.0, 500.0, 400.0, 200.0, 100.0]
        # Sort after sampling_rate.
        stream.sort(keys=['sampling_rate'])
        assert [i.stats.sampling_rate for i in stream.traces] == \
               [100.0, 200.0, 300.0, 400.0, 500.0]
        # Sort after channel and sampling rate.
        stream.sort(keys=['channel', 'sampling_rate'])
        assert [i.stats.sampling_rate for i in stream.traces] == \
               [200.0, 500.0, 300.0, 400.0, 100.0]
        # Sort after npts and sampling_rate and endtime.
        stream.sort(keys=['npts', 'sampling_rate', 'endtime'])
        assert [i.stats.sampling_rate for i in stream.traces] == \
               [100.0, 200.0, 300.0, 400.0, 500.0]
        # The same with reverted sorting
        # Use normal sorting.
        stream.sort(reverse=True)
        assert [i.stats.sampling_rate for i in stream.traces] == \
               [100.0, 200.0, 400.0, 500.0, 300.0]
        # Sort after sampling_rate.
        stream.sort(keys=['sampling_rate'], reverse=True)
        assert [i.stats.sampling_rate for i in stream.traces] == \
               [500.0, 400.0, 300.0, 200.0, 100.0]
        # Sort after channel and sampling rate.
        stream.sort(keys=['channel', 'sampling_rate'], reverse=True)
        assert [i.stats.sampling_rate for i in stream.traces] == \
               [100.0, 400.0, 300.0, 500.0, 200.0]
        # Sort after npts and sampling_rate and endtime.
        stream.sort(keys=['npts', 'sampling_rate', 'endtime'], reverse=True)
        assert [i.stats.sampling_rate for i in stream.traces] == \
               [500.0, 400.0, 300.0, 200.0, 100.0]
        # Sorting without a list or a wrong item string should fail.
        with pytest.raises(TypeError):
            stream.sort(keys=1)
        with pytest.raises(TypeError):
            stream.sort(keys='sampling_rate')
        with pytest.raises(KeyError):
            stream.sort(keys=['npts', 'wrong_value'])

    def test_sorting_twice(self):
        """
        Sorting twice should not change order.
        """
        stream = Stream()
        headers = [
            {'starttime': UTCDateTime(1990, 1, 1),
             'endtime': UTCDateTime(1990, 1, 2), 'network': 'AAA',
             'station': 'ZZZ', 'channel': 'XXX', 'npts': 10000,
             'sampling_rate': 100.0},
            {'starttime': UTCDateTime(1990, 1, 1),
             'endtime': UTCDateTime(1990, 1, 3), 'network': 'AAA',
             'station': 'YYY', 'channel': 'CCC', 'npts': 10000,
             'sampling_rate': 200.0},
            {'starttime': UTCDateTime(2000, 1, 1),
             'endtime': UTCDateTime(2001, 1, 2), 'network': 'AAA',
             'station': 'EEE', 'channel': 'GGG', 'npts': 1000,
             'sampling_rate': 300.0},
            {'starttime': UTCDateTime(1989, 1, 1),
             'endtime': UTCDateTime(2010, 1, 2), 'network': 'AAA',
             'station': 'XXX', 'channel': 'GGG', 'npts': 10000,
             'sampling_rate': 400.0},
            {'starttime': UTCDateTime(2010, 1, 1),
             'endtime': UTCDateTime(2011, 1, 2), 'network': 'AAA',
             'station': 'XXX', 'channel': 'FFF', 'npts': 1000,
             'sampling_rate': 500.0}]
        # Create a Trace object of it and append it to the Stream object.
        for _i in headers:
            new_trace = Trace(header=_i)
            stream.append(new_trace)
        stream.sort()
        a = [i.stats.sampling_rate for i in stream.traces]
        stream.sort()
        b = [i.stats.sampling_rate for i in stream.traces]
        # should be equal
        assert a == b

    def test_merge_with_different_calibration_factors(self):
        """
        Test the merge method of the Stream object.
        """
        # 1 - different calibration factors for the same channel should fail
        tr1 = Trace(data=np.zeros(5))
        tr1.stats.calib = 1.0
        tr2 = Trace(data=np.zeros(5))
        tr2.stats.calib = 2.0
        st = Stream([tr1, tr2])
        # this also emits an UserWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with pytest.raises(Exception):
                st.merge()
        # 2 - different calibration factors for the different channels is ok
        tr1 = Trace(data=np.zeros(5))
        tr1.stats.calib = 2.00
        tr1.stats.channel = 'EHE'
        tr2 = Trace(data=np.zeros(5))
        tr2.stats.calib = 5.0
        tr2.stats.channel = 'EHZ'
        tr3 = Trace(data=np.zeros(5))
        tr3.stats.calib = 2.00
        tr3.stats.channel = 'EHE'
        tr4 = Trace(data=np.zeros(5))
        tr4.stats.calib = 5.0
        tr4.stats.channel = 'EHZ'
        st = Stream([tr1, tr2, tr3, tr4])
        st.merge()

    def test_merge_with_different_sampling_rates(self):
        """
        Test the merge method of the Stream object.
        """
        # 1 - different sampling rates for the same channel should fail
        tr1 = Trace(data=np.zeros(5))
        tr1.stats.sampling_rate = 200
        tr2 = Trace(data=np.zeros(5))
        tr2.stats.sampling_rate = 50
        st = Stream([tr1, tr2])
        # this also emits an UserWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with pytest.raises(Exception):
                st.merge()
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
        st = Stream([tr1, tr2, tr3, tr4])
        st.merge()

    def test_merge_with_different_data_types(self):
        """
        Test the merge method of the Stream object.
        """
        # 1 - different dtype for the same channel should fail
        tr1 = Trace(data=np.zeros(5, dtype=np.int32))
        tr2 = Trace(data=np.zeros(5, dtype=np.float32))
        st = Stream([tr1, tr2])
        # this also emits an UserWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with pytest.raises(Exception):
                st.merge()
        # 2 - different sampling rates for the different channels is ok
        tr1 = Trace(data=np.zeros(5, dtype=np.int32))
        tr1.stats.channel = 'EHE'
        tr2 = Trace(data=np.zeros(5, dtype=np.float32))
        tr2.stats.channel = 'EHZ'
        tr3 = Trace(data=np.zeros(5, dtype=np.int32))
        tr3.stats.channel = 'EHE'
        tr4 = Trace(data=np.zeros(5, dtype=np.float32))
        tr4.stats.channel = 'EHZ'
        st = Stream([tr1, tr2, tr3, tr4])
        st.merge()

    def test_merge_gaps(self, mseed_stream):
        """
        Test the merge method of the Stream object.
        """
        stream = mseed_stream
        start = UTCDateTime("2007-12-31T23:59:59.915000")
        end = UTCDateTime("2008-01-01T00:04:31.790000")
        assert len(stream) == 4
        assert len(stream[0]) == 412
        assert len(stream[1]) == 824
        assert len(stream[2]) == 824
        assert len(stream[3]) == 50668
        assert stream[0].stats.starttime == start
        assert stream[3].stats.endtime == end
        for i in range(4):
            assert stream[i].stats.sampling_rate == 200
            assert stream[i].get_id() == 'BW.BGLD..EHE'
        stream.verify()
        # merge it
        stream.merge()
        stream.verify()
        assert len(stream) == 1
        assert len(stream[0]) == stream[0].data.size
        assert stream[0].stats.starttime == start
        assert stream[0].stats.endtime == end
        assert stream[0].stats.sampling_rate == 200
        assert stream[0].get_id() == 'BW.BGLD..EHE'

    def test_merge_gaps_2(self):
        """
        Test the merge method of the Stream object on two traces with a gap in
        between.
        """
        tr1 = Trace(data=np.ones(4, dtype=np.int32) * 1)
        tr2 = Trace(data=np.ones(3, dtype=np.int32) * 5)
        tr2.stats.starttime = tr1.stats.starttime + 9
        stream = Stream([tr1, tr2])
        # 1 - masked array
        # Trace 1: 1111
        # Trace 2:          555
        # 1 + 2  : 1111-----555
        st = stream.copy()
        st.merge()
        assert len(st) == 1
        assert isinstance(st[0].data, np.ma.masked_array)
        assert st[0].data.tolist() == \
               [1, 1, 1, 1, None, None, None, None, None, 5, 5, 5]
        # 2 - fill in zeros
        # Trace 1: 1111
        # Trace 2:          555
        # 1 + 2  : 111100000555
        st = stream.copy()
        st.merge(fill_value=0)
        assert len(st) == 1
        assert isinstance(st[0].data, np.ndarray)
        assert st[0].data.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 5, 5]
        # 2b - fill in some other user-defined value
        # Trace 1: 1111
        # Trace 2:          555
        # 1 + 2  : 111199999555
        st = stream.copy()
        st.merge(fill_value=9)
        assert len(st) == 1
        assert isinstance(st[0].data, np.ndarray)
        assert st[0].data.tolist() == [1, 1, 1, 1, 9, 9, 9, 9, 9, 5, 5, 5]
        # 3 - use last value of first trace
        # Trace 1: 1111
        # Trace 2:          555
        # 1 + 2  : 111111111555
        st = stream.copy()
        st.merge(fill_value='latest')
        assert len(st) == 1
        assert isinstance(st[0].data, np.ndarray)
        assert st[0].data.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5]
        # 4 - interpolate
        # Trace 1: 1111
        # Trace 2:          555
        # 1 + 2  : 111112334555
        st = stream.copy()
        st.merge(fill_value='interpolate')
        assert len(st) == 1
        assert isinstance(st[0].data, np.ndarray)
        assert st[0].data.tolist() == [1, 1, 1, 1, 1, 2, 3, 3, 4, 5, 5, 5]

    def test_split(self, mseed_stream):
        """
        Testing splitting of streams containing masked arrays.
        """
        # 1 - create a Stream with gaps
        tr1 = Trace(data=np.ones(4, dtype=np.int32) * 1)
        tr2 = Trace(data=np.ones(3, dtype=np.int32) * 5)
        tr2.stats.starttime = tr1.stats.starttime + 9
        st = Stream([tr1, tr2])
        st.merge()
        assert isinstance(st[0].data, np.ma.masked_array)
        # now we split again
        st2 = st.split()
        assert len(st2) == 2
        assert isinstance(st2[0].data, np.ndarray)
        assert isinstance(st2[1].data, np.ndarray)
        assert st2[0].data.tolist() == [1, 1, 1, 1]
        assert st2[1].data.tolist() == [5, 5, 5]
        # 2 - use default example
        st = mseed_stream
        st.merge()
        assert isinstance(st[0].data, np.ma.masked_array)
        # now we split again
        st2 = st.split()
        assert len(st2) == 4
        assert len(st2[0]) == 412
        assert len(st2[1]) == 824
        assert len(st2[2]) == 824
        assert len(st2[3]) == 50668
        assert st2[0].stats.starttime == \
               UTCDateTime("2007-12-31T23:59:59.915000")
        assert st2[3].stats.endtime == \
               UTCDateTime("2008-01-01T00:04:31.790000")
        for i in range(4):
            assert st2[i].stats.sampling_rate == 200
            assert st2[i].get_id() == 'BW.BGLD..EHE'

    def test_merge_overlaps_default_method(self):
        """
        Test the merge method of the Stream object.
        """
        # 1 - overlapping trace with differing data
        # Trace 1: 0000000
        # Trace 2:      1111111
        # 1 + 2  : 00000--11111
        tr1 = Trace(data=np.zeros(7))
        tr2 = Trace(data=np.ones(7))
        tr2.stats.starttime = tr1.stats.starttime + 5
        st = Stream([tr1, tr2])
        st.merge()
        assert len(st) == 1
        assert isinstance(st[0].data, np.ma.masked_array)
        data_list = st[0].data.tolist()
        assert data_list == [0, 0, 0, 0, 0, None, None, 1, 1, 1, 1, 1]
        # 2 - overlapping trace with same data
        # Trace 1: 0123456
        # Trace 2:      56789
        # 1 + 2  : 0123456789
        tr1 = Trace(data=np.arange(7))
        tr2 = Trace(data=np.arange(5, 10))
        tr2.stats.starttime = tr1.stats.starttime + 5
        st = Stream([tr1, tr2])
        st.merge()
        assert len(st) == 1
        assert isinstance(st[0].data, np.ndarray)
        np.testing.assert_array_equal(st[0].data, np.arange(10))
        #
        # 3 - contained overlap with same data
        # Trace 1: 0123456789
        # Trace 2:      56
        # 1 + 2  : 0123456789
        tr1 = Trace(data=np.arange(10))
        tr2 = Trace(data=np.arange(5, 7))
        tr2.stats.starttime = tr1.stats.starttime + 5
        st = Stream([tr1, tr2])
        st.merge()
        assert len(st) == 1
        assert isinstance(st[0].data, np.ndarray)
        np.testing.assert_array_equal(st[0].data, np.arange(10))
        #
        # 4 - contained overlap with differing data
        # Trace 1: 0000000000
        # Trace 2:      11
        # 1 + 2  : 00000--000
        tr1 = Trace(data=np.zeros(10))
        tr2 = Trace(data=np.ones(2))
        tr2.stats.starttime = tr1.stats.starttime + 5
        st = Stream([tr1, tr2])
        st.merge()
        assert len(st) == 1
        assert isinstance(st[0].data, np.ma.masked_array)
        assert st[0].data.tolist() == [0, 0, 0, 0, 0, None, None, 0, 0, 0]

    def test_tab_completion_trace(self):
        """
        Test tab completion of Trace object.
        """
        tr = Trace()
        assert 'sampling_rate' in dir(tr.stats)
        assert 'npts' in dir(tr.stats)
        assert 'station' in dir(tr.stats)
        assert 'starttime' in dir(tr.stats)
        assert 'endtime' in dir(tr.stats)
        assert 'calib' in dir(tr.stats)
        assert 'delta' in dir(tr.stats)

    def test_bugfix_merge_drop_trace_if_already_contained(self):
        """
        Trace data already existing in another trace and ending on the same
        end time was not correctly merged until now.
        """
        trace1 = Trace(data=np.empty(10))
        trace2 = Trace(data=np.empty(2))
        trace2.stats.starttime = trace1.stats.endtime - trace1.stats.delta
        st = Stream([trace1, trace2])
        st.merge()

    def test_bugfix_merge_multiple_traces(self):
        """
        Bugfix for merging multiple traces in a row.
        """
        # create a stream with multiple traces overlapping
        trace1 = Trace(data=np.empty(10))
        traces = [trace1]
        for _ in range(10):
            trace = Trace(data=np.empty(10))
            trace.stats.starttime = \
                traces[-1].stats.endtime - trace1.stats.delta
            traces.append(trace)
        st = Stream(traces)
        st.merge()

    def test_bugfix_merge_multiple_traces_2(self):
        """
        Bugfix for merging multiple traces in a row.
        """
        trace1 = Trace(data=np.empty(4190864))
        trace1.stats.sampling_rate = 200
        trace1.stats.starttime = UTCDateTime("2010-01-21T00:00:00.015000Z")
        trace2 = Trace(data=np.empty(603992))
        trace2.stats.sampling_rate = 200
        trace2.stats.starttime = UTCDateTime("2010-01-21T05:49:14.330000Z")
        trace3 = Trace(data=np.empty(222892))
        trace3.stats.sampling_rate = 200
        trace3.stats.starttime = UTCDateTime("2010-01-21T06:39:33.280000Z")
        st = Stream([trace1, trace2, trace3])
        st.merge()

    def test_merge_with_small_sampling_rate(self):
        """
        Bugfix for merging multiple traces with very small sampling rate.
        """
        # create traces
        np.random.seed(815)
        trace1 = Trace(data=np.random.randn(1441))
        trace1.stats.delta = 60.0
        trace1.stats.starttime = UTCDateTime("2009-02-01T00:00:02.995000Z")
        trace2 = Trace(data=np.random.randn(1441))
        trace2.stats.delta = 60.0
        trace2.stats.starttime = UTCDateTime("2009-02-02T00:00:12.095000Z")
        trace3 = Trace(data=np.random.randn(1440))
        trace3.stats.delta = 60.0
        trace3.stats.starttime = UTCDateTime("2009-02-03T00:00:16.395000Z")
        trace4 = Trace(data=np.random.randn(1440))
        trace4.stats.delta = 60.0
        trace4.stats.starttime = UTCDateTime("2009-02-04T00:00:11.095000Z")
        # create stream
        st = Stream([trace1, trace2, trace3, trace4])
        # merge
        st.merge()
        # compare results
        assert len(st) == 1
        assert st[0].stats.delta == 60.0
        assert st[0].stats.starttime == trace1.stats.starttime
        # end time of last trace
        endtime = trace1.stats.starttime + \
            (4 * 1440 - 1) * trace1.stats.delta
        assert st[0].stats.endtime == endtime

    def test_merge_overlaps_method_1(self):
        """
        Test merging with method = 1.
        """
        # Test merging three traces.
        trace1 = Trace(data=np.ones(10))
        trace2 = Trace(data=10 * np.ones(11))
        trace3 = Trace(data=2 * np.ones(20))
        st = Stream([trace1, trace2, trace3])
        st.merge(method=1)
        np.testing.assert_array_equal(st[0].data, 2 * np.ones(20))
        # Any contained traces with different data will be discarded::
        #
        #    Trace 1: 111111111111 (contained trace)
        #    Trace 2:     55
        #    1 + 2  : 111111111111
        trace1 = Trace(data=np.ones(12))
        trace2 = Trace(data=5 * np.ones(2))
        trace2.stats.starttime += 4
        st = Stream([trace1, trace2])
        st.merge(method=1)
        np.testing.assert_array_equal(st[0].data, np.ones(12))
        # No interpolation (``interpolation_samples=0``)::
        #
        #    Trace 1: 11111111
        #    Trace 2:     55555555
        #    1 + 2  : 111155555555
        trace1 = Trace(data=np.ones(8))
        trace2 = Trace(data=5 * np.ones(8))
        trace2.stats.starttime += 4
        st = Stream([trace1, trace2])
        st.merge(method=1)
        np.testing.assert_array_equal(st[0].data, np.array([1] * 4 + [5] * 8))
        # Interpolate first two samples (``interpolation_samples=2``)::
        #
        #     Trace 1: 00000000
        #     Trace 2:     66666666
        #     1 + 2  : 000024666666 (interpolation_samples=2)
        trace1 = Trace(data=np.zeros(8, dtype=np.int32))
        trace2 = Trace(data=6 * np.ones(8, dtype=np.int32))
        trace2.stats.starttime += 4
        st = Stream([trace1, trace2])
        st.merge(method=1, interpolation_samples=2)
        np.testing.assert_array_equal(st[0].data,
                                      np.array([0] * 4 + [2] + [4] + [6] * 6))
        # Interpolate all samples (``interpolation_samples=-1``)::
        #
        #     Trace 1: 00000000
        #     Trace 2:     55555555
        #     1 + 2  : 000012345555
        trace1 = Trace(data=np.zeros(8, dtype=np.int32))
        trace2 = Trace(data=5 * np.ones(8, dtype=np.int32))
        trace2.stats.starttime += 4
        st = Stream([trace1, trace2])
        st.merge(method=1, interpolation_samples=(-1))
        np.testing.assert_array_equal(
            st[0].data, np.array([0] * 4 + [1] + [2] + [3] + [4] + [5] * 4))
        # Interpolate all samples (``interpolation_samples=5``)::
        # Given number of samples is bigger than the actual overlap - should
        # interpolate all samples
        #
        #     Trace 1: 00000000
        #     Trace 2:     55555555
        #     1 + 2  : 000012345555
        trace1 = Trace(data=np.zeros(8, dtype=np.int32))
        trace2 = Trace(data=5 * np.ones(8, dtype=np.int32))
        trace2.stats.starttime += 4
        st = Stream([trace1, trace2])
        st.merge(method=1, interpolation_samples=5)
        np.testing.assert_array_equal(
            st[0].data, np.array([0] * 4 + [1] + [2] + [3] + [4] + [5] * 4))

    def test_trim_removing_empty_traces(self):
        """
        A stream containing several empty traces after trimming should throw
        away the empty traces.
        """
        # create Stream.
        trace1 = Trace(data=np.zeros(10))
        trace1.stats.delta = 1.0
        trace2 = Trace(data=np.ones(10))
        trace2.stats.delta = 1.0
        trace2.stats.starttime = UTCDateTime(1000)
        trace3 = Trace(data=np.arange(10))
        trace3.stats.delta = 1.0
        trace3.stats.starttime = UTCDateTime(2000)
        stream = Stream([trace1, trace2, trace3])
        stream.trim(UTCDateTime(900), UTCDateTime(1100))
        # Check if only trace2 is still in the Stream object.
        assert len(stream) == 1
        np.testing.assert_array_equal(np.ones(10), stream[0].data)
        assert stream[0].stats.starttime == UTCDateTime(1000)
        assert stream[0].stats.npts == 10

    def test_trim_with_small_sampling_rate(self):
        """
        Bugfix for cutting multiple traces with very small sampling rate.
        """
        # create traces
        trace1 = Trace(data=np.empty(1441))
        trace1.stats.delta = 60.0
        trace1.stats.starttime = UTCDateTime("2009-02-01T00:00:02.995000Z")
        trace2 = Trace(data=np.empty(1441))
        trace2.stats.delta = 60.0
        trace2.stats.starttime = UTCDateTime("2009-02-02T00:00:12.095000Z")
        trace3 = Trace(data=np.empty(1440))
        trace3.stats.delta = 60.0
        trace3.stats.starttime = UTCDateTime("2009-02-03T00:00:16.395000Z")
        trace4 = Trace(data=np.empty(1440))
        trace4.stats.delta = 60.0
        trace4.stats.starttime = UTCDateTime("2009-02-04T00:00:11.095000Z")
        # create stream
        st = Stream([trace1, trace2, trace3, trace4])
        # trim
        st.trim(trace1.stats.starttime, trace4.stats.endtime)
        # compare results
        assert len(st) == 4
        assert st[0].stats.delta == 60.0
        assert st[0].stats.starttime == trace1.stats.starttime
        assert st[3].stats.endtime == trace4.stats.endtime

    def test_writing_masked_array(self):
        """
        Writing a masked array should raise an exception.
        """
        # np.ma.masked_array with masked values
        tr = Trace(data=np.ma.masked_all(10))
        st = Stream([tr])
        with pytest.raises(NotImplementedError):
            st.write('filename', 'MSEED')
        # np.ma.masked_array without masked values
        tr = Trace(data=np.ma.ones(10))
        st = Stream([tr])
        with pytest.raises(NotImplementedError):
            st.write('filename', 'MSEED')

    def test_pickle(self):
        """
        Testing pickling of Stream objects.
        """
        tr = Trace(data=np.random.randn(1441))
        st = Stream([tr])
        st.verify()
        # protocol 0 (ASCII)
        temp = pickle.dumps(st, protocol=0)
        st2 = pickle.loads(temp)
        np.testing.assert_array_equal(st[0].data, st2[0].data)
        assert st[0].stats == st2[0].stats
        # protocol 1 (old binary)
        temp = pickle.dumps(st, protocol=1)
        st2 = pickle.loads(temp)
        np.testing.assert_array_equal(st[0].data, st2[0].data)
        assert st[0].stats == st2[0].stats
        # protocol 2 (new binary)
        temp = pickle.dumps(st, protocol=2)
        st2 = pickle.loads(temp)
        np.testing.assert_array_equal(st[0].data, st2[0].data)
        assert st[0].stats == st2[0].stats

    def test_cpickle(self):
        """
        Testing pickling of Stream objects.
        """
        tr = Trace(data=np.random.randn(1441))
        st = Stream([tr])
        st.verify()
        # protocol 0 (ASCII)
        temp = pickle.dumps(st, protocol=0)
        st2 = pickle.loads(temp)
        np.testing.assert_array_equal(st[0].data, st2[0].data)
        assert st[0].stats == st2[0].stats
        # protocol 1 (old binary)
        temp = pickle.dumps(st, protocol=1)
        st2 = pickle.loads(temp)
        np.testing.assert_array_equal(st[0].data, st2[0].data)
        assert st[0].stats == st2[0].stats
        # protocol 2 (new binary)
        temp = pickle.dumps(st, protocol=2)
        st2 = pickle.loads(temp)
        np.testing.assert_array_equal(st[0].data, st2[0].data)
        assert st[0].stats == st2[0].stats

    def test_is_pickle(self):
        """
        Testing _is_pickle function.
        """
        # existing file
        st = read()
        with NamedTemporaryFile() as tf:
            st.write(tf.name, format='PICKLE')
            # check using file name
            assert _is_pickle(tf.name)
            # check using file handler
            assert _is_pickle(tf)
        # not existing files
        assert not _is_pickle('/path/to/pickle.file')
        assert not _is_pickle(12345)

    def test_read_write_pickle(self):
        """
        Testing _read_pickle and _write_pickle functions.
        """
        st = read()
        # write
        with NamedTemporaryFile() as tf:
            # write using file name
            _write_pickle(st, tf.name)
            assert _is_pickle(tf.name)
            # write using file handler
            _write_pickle(st, tf)
            tf.seek(0)
            assert _is_pickle(tf)
            # write using stream write method
            st.write(tf.name, format='PICKLE')
            # check and read directly
            st2 = _read_pickle(tf.name)
            assert len(st2) == 3
            np.testing.assert_array_equal(st2[0].data, st[0].data)
            # use read() with given format
            st2 = read(tf.name, format='PICKLE')
            assert len(st2) == 3
            np.testing.assert_array_equal(st2[0].data, st[0].data)
            # use read() and automatically detect format
            st2 = read(tf.name)
            assert len(st2) == 3
            np.testing.assert_array_equal(st2[0].data, st[0].data)

    def test_read_pickle(self):
        """
        Testing _read_pickle function.

        Pickle support is only provided for pickles created with the same
        versions of ObsPy/python.
        """
        def _remove_format(st):
            """Remove '_format' from stats so streams """
            for tr in st:
                tr.stats.pop("_format", None)
            return st

        # Save stream to bio.
        bio = io.BytesIO()
        st = read()
        st.write(bio, "pickle")
        # Read pickle without specifying format.
        bio.seek(0)
        st_no_format = read(bio)
        # Read pickle with specifying format.
        bio.seek(0)
        st_with_format = read(bio, format="pickle")
        # Test equalities.
        assert st == _remove_format(st_no_format)
        assert st_no_format == _remove_format(st_with_format)

    def test_get_gaps_2(self):
        """
        Test case for issue #73.
        """
        tr1 = Trace(data=np.empty(720000))
        tr1.stats.starttime = UTCDateTime("2010-02-09T00:19:19.850000Z")
        tr1.stats.sampling_rate = 200.0
        tr1.verify()
        tr2 = Trace(data=np.empty(720000))
        tr2.stats.starttime = UTCDateTime("2010-02-09T01:19:19.850000Z")
        tr2.stats.sampling_rate = 200.0
        tr2.verify()
        tr3 = Trace(data=np.empty(720000))
        tr3.stats.starttime = UTCDateTime("2010-02-09T02:19:19.850000Z")
        tr3.stats.sampling_rate = 200.0
        tr3.verify()
        st = Stream([tr1, tr2, tr3])
        st.verify()
        # same sampling rate should have no gaps
        gaps = st.get_gaps()
        assert len(gaps) == 0
        # different sampling rate should result in a gap
        tr3.stats.sampling_rate = 50.0
        gaps = st.get_gaps()
        assert len(gaps) == 1
        # but different ids will be skipped (if only one trace)
        tr3.stats.station = 'MANZ'
        gaps = st.get_gaps()
        assert len(gaps) == 0
        # multiple traces with same id will be handled again
        tr2.stats.station = 'MANZ'
        gaps = st.get_gaps()
        assert len(gaps) == 1

    def test_get_gaps_whole_overlap(self):
        """
        Test get_gaps method with a trace completely overlapping another trace.
        """
        tr1 = Trace(data=np.empty(3600))
        tr1.stats.starttime = UTCDateTime("2018-09-25T00:00:00.000000Z")
        tr1.stats.sampling_rate = 1.
        tr2 = Trace(data=np.empty(60))
        tr2.stats.starttime = UTCDateTime("2018-09-25T00:01:00.000000Z")
        tr2.stats.sampling_rate = 1.
        st = Stream([tr1, tr2])
        gaps = st.get_gaps()
        assert len(gaps) == 1
        gap = gaps[0]
        starttime = gap[4]
        assert starttime == UTCDateTime("2018-09-25T00:01:59.000000Z")
        endtime = gap[5]
        assert endtime == tr2.stats.starttime

    def test_get_gaps_overlap(self):
        """
         Tests the get_gaps method of the Stream objects.

         Test for Issue #1403. Tests if wrong overlaps are returned.
        """
        data = [
            ("2016-01-07T00:00:50.388393Z", 6158),
            ("2016-01-07T00:00:57.248393Z", 1370),
            ("2016-01-07T00:01:31.458393Z", 4107)]

        x = np.arange(20000)
        tr = Trace(x)
        tr.stats.starttime = UTCDateTime("2016-01-07T00:00:50.388393Z")
        tr.stats.sampling_rate = 100

        st = Stream()
        for i, (start, numsamp) in enumerate(data):
            tr_ = tr.slice(starttime=UTCDateTime(start))
            tr_.data = tr_.data[:numsamp]
            st.append(tr_)

        # min_gap=1 is used to only show the gaps
        assert len(st.get_gaps(min_gap=1)) == 0

    def test_comparisons(self):
        """
        Tests all rich comparison operators (==, !=, <, <=, >, >=)
        The latter four are not implemented due to ambiguous meaning and bounce
        an error.
        """
        # create test streams
        tr0 = Trace(np.arange(3))
        tr1 = Trace(np.arange(3))
        tr2 = Trace(np.arange(3), {'station': 'X'})
        tr3 = Trace(np.arange(3),
                    {'processing': ["filter:lowpass:{'freq': 10}"]})
        tr4 = Trace(np.arange(5))
        tr5 = Trace(np.arange(5), {'station': 'X'})
        tr6 = Trace(np.arange(5),
                    {'processing': ["filter:lowpass:{'freq': 10}"]})
        tr7 = Trace(np.arange(5),
                    {'processing': ["filter:lowpass:{'freq': 10}"]})
        st0 = Stream([tr0])
        st1 = Stream([tr1])
        st2 = Stream([tr0, tr1])
        st3 = Stream([tr2, tr3])
        st4 = Stream([tr1, tr2, tr3])
        st5 = Stream([tr4, tr5, tr6])
        st6 = Stream([tr0, tr6])
        st7 = Stream([tr1, tr7])
        st8 = Stream([tr7, tr1])
        st9 = Stream()
        st_a = Stream()
        # tests that should raise a NotImplementedError (i.e. <=, <, >=, >)
        with pytest.raises(NotImplementedError):
            st1.__lt__(st1)
        with pytest.raises(NotImplementedError):
            st1.__le__(st1)
        with pytest.raises(NotImplementedError):
            st1.__gt__(st1)
        with pytest.raises(NotImplementedError):
            st1.__ge__(st1)
        with pytest.raises(NotImplementedError):
            st1.__lt__(st2)
        with pytest.raises(NotImplementedError):
            st1.__le__(st2)
        with pytest.raises(NotImplementedError):
            st1.__gt__(st2)
        with pytest.raises(NotImplementedError):
            st1.__ge__(st2)
        # normal tests
        for st in [st1]:
            assert st0 == st
        for st in [st2, st3, st4, st5, st6, st7, st8, st9, st_a]:
            assert st0 != st
        for st in [st0]:
            assert st1 == st
        for st in [st2, st3, st4, st5, st6, st7, st8, st9, st_a]:
            assert st1 != st
        for st in [st0, st1, st3, st4, st5, st6, st7, st8, st9, st_a]:
            assert st2 != st
        for st in [st0, st1, st2, st4, st5, st6, st7, st8, st9, st_a]:
            assert st3 != st
        for st in [st0, st1, st2, st3, st5, st6, st7, st8, st9, st_a]:
            assert st4 != st
        for st in [st0, st1, st2, st3, st4, st6, st7, st8, st9, st_a]:
            assert st5 != st
        for st in [st7, st8]:
            assert st6 == st
        for st in [st0, st1, st2, st3, st4, st5, st9, st_a]:
            assert st6 != st
        for st in [st6, st8]:
            assert st7 == st
        for st in [st0, st1, st2, st3, st4, st5, st9, st_a]:
            assert st7 != st
        for st in [st6, st7]:
            assert st8 == st
        for st in [st0, st1, st2, st3, st4, st5, st9, st_a]:
            assert st8 != st
        for st in [st_a]:
            assert st9 == st
        for st in [st0, st1, st2, st3, st4, st5, st6, st7, st8]:
            assert st9 != st
        for st in [st9]:
            assert st_a == st
        for st in [st0, st1, st2, st3, st4, st5, st6, st7, st8]:
            assert st_a != st
        # some weird tests against non-Stream objects
        for object in [0, 1, 0.0, 1.0, "", "test", True, False, [], [tr0],
                       set(), set(tr0), {}, {"test": "test"}, Trace(), None]:
            assert st0 != object

    def test_trim_nearest_sample(self):
        """
        Tests to trim at nearest sample
        """
        head = {'sampling_rate': 1.0, 'starttime': UTCDateTime(0.0)}
        tr1 = Trace(data=np.random.randint(0, 1000, 120), header=head)
        tr2 = Trace(data=np.random.randint(0, 1000, 120), header=head)
        tr2.stats.starttime += 0.4
        st = Stream(traces=[tr1, tr2])
        # STARTTIME
        # check that trimming first selects the next best sample, and only
        # then selects the following ones
        #    |  S |    |    |
        #      |    |    |    |
        st.trim(UTCDateTime(0.6), endtime=None)
        assert st[0].stats.starttime.timestamp == 1.0
        assert st[1].stats.starttime.timestamp == 1.4
        # ENDTIME
        # check that trimming first selects the next best sample, and only
        # then selects the following ones
        #    |    |    |  E |
        #      |    |    |    |
        st.trim(starttime=None, endtime=UTCDateTime(2.6))
        assert st[0].stats.endtime.timestamp == 3.0
        assert st[1].stats.endtime.timestamp == 3.4

    def test_trim_consistent_start_end_time_nearest_sample(self):
        """
        Test case for #127. It ensures that the sample sizes stay
        consistent after trimming. That is that _ltrim and _rtrim
        round in the same direction.
        """
        data = np.zeros(10)
        t = UTCDateTime(0)
        traces = []
        for delta in (0, 0.25, 0.5, 0.75, 1):
            traces.append(Trace(data.copy()))
            traces[-1].stats.starttime = t + delta
        st = Stream(traces)
        st.trim(t + 3.5, t + 6.5)
        start = [4.0, 4.25, 4.5, 3.75, 4.0]
        end = [6.0, 6.25, 6.50, 5.75, 6.0]
        for i in range(len(st)):
            assert 3 == st[i].stats.npts
            assert st[i].stats.starttime.timestamp == start[i]
            assert st[i].stats.endtime.timestamp == end[i]

    def test_trim_consistent_start_end_time_nearest_sample_padded(self):
        """
        Test case for #127. It ensures that the sample sizes stay
        consistent after trimming. That is that _ltrim and _rtrim
        round in the same direction. Padded version.
        """
        data = np.zeros(10)
        t = UTCDateTime(0)
        traces = []
        for delta in (0, 0.25, 0.5, 0.75, 1):
            traces.append(Trace(data.copy()))
            traces[-1].stats.starttime = t + delta
        st = Stream(traces)
        st.trim(t - 3.5, t + 16.5, pad=True)
        start = [-4.0, -3.75, -3.5, -4.25, -4.0]
        end = [17.0, 17.25, 17.50, 16.75, 17.0]
        for i in range(len(st)):
            assert 22 == st[i].stats.npts
            assert st[i].stats.starttime.timestamp == start[i]
            assert st[i].stats.endtime.timestamp == end[i]

    def test_trim_consistent_start_end_time(self):
        """
        Test case for #127. It ensures that the sample start and end times
        stay consistent after trimming.
        """
        data = np.zeros(10)
        t = UTCDateTime(0)
        traces = []
        for delta in (0, 0.25, 0.5, 0.75, 1):
            traces.append(Trace(data.copy()))
            traces[-1].stats.starttime = t + delta
        st = Stream(traces)
        st.trim(t + 3.5, t + 6.5, nearest_sample=False)
        start = [4.00, 4.25, 3.50, 3.75, 4.00]
        end = [6.00, 6.25, 6.50, 5.75, 6.00]
        npts = [3, 3, 4, 3, 3]
        for i in range(len(st)):
            assert st[i].stats.npts == npts[i]
            assert st[i].stats.starttime.timestamp == start[i]
            assert st[i].stats.endtime.timestamp == end[i]

    def test_trim_consistent_start_and_time_pad(self):
        """
        Test case for #127. It ensures that the sample start and end times
        stay consistent after trimming. Padded version.
        """
        data = np.zeros(10)
        t = UTCDateTime(0)
        traces = []
        for delta in (0, 0.25, 0.5, 0.75, 1):
            traces.append(Trace(data.copy()))
            traces[-1].stats.starttime = t + delta
        st = Stream(traces)
        st.trim(t - 3.5, t + 16.5, nearest_sample=False, pad=True)
        start = [-3.00, -2.75, -3.50, -3.25, -3.00]
        end = [16.00, 16.25, 16.50, 15.75, 16.00]
        npts = [20, 20, 21, 20, 20]
        for i in range(len(st)):
            assert st[i].stats.npts == npts[i]
            assert st[i].stats.starttime.timestamp == start[i]
            assert st[i].stats.endtime.timestamp == end[i]

    def test_str(self):
        """
        Test case for issue #162 - print streams in a more consistent way.
        """
        tr1 = Trace()
        tr1.stats.station = "1"
        tr2 = Trace()
        tr2.stats.station = "12345"
        st = Stream([tr1, tr2])
        result = st.__str__()
        expected = "2 Trace(s) in Stream:\n" + \
                   ".1..     | 1970-01-01T00:00:00.000000Z - 1970-01-01" + \
                   "T00:00:00.000000Z | 1.0 Hz, 0 samples\n" + \
                   ".12345.. | 1970-01-01T00:00:00.000000Z - 1970-01-01" + \
                   "T00:00:00.000000Z | 1.0 Hz, 0 samples"
        assert result == expected
        # streams containing more than 20 lines will be compressed
        st2 = Stream([tr1]) * 40
        result = st2.__str__()
        assert '40 Trace(s) in Stream:' in result
        assert 'other traces' in result

    def test_cleanup(self, mseed_stream):
        """
        Test case for merging traces in the stream with method=-1. This only
        should merge traces that are exactly the same or contained and exactly
        the same or directly adjacent.
        """
        tr1 = mseed_stream[0]
        start = tr1.stats.starttime
        end = tr1.stats.endtime
        dt = end - start
        delta = tr1.stats.delta
        # test traces that should be merged:
        # contained traces with compatible data
        tr2 = tr1.slice(start, start + dt / 3)
        tr3 = tr1.copy()
        tr4 = tr1.slice(start + dt / 4, end - dt / 4)
        # adjacent traces
        tr5 = tr1.copy()
        tr5.stats.starttime = end + delta
        tr6 = tr1.copy()
        tr6.stats.starttime = start - dt - delta
        # create overlapping traces with compatible data
        tr_01 = tr1.copy()
        tr_01.trim(starttime=start + 2 * delta)
        tr_01.data = np.concatenate([tr_01.data, np.arange(5)])
        tr_02 = tr1.copy()
        tr_02.trim(endtime=end - 2 * delta)
        tr_02.data = np.concatenate([np.arange(5), tr_02.data])
        tr_02.stats.starttime -= 5 * delta

        for _i in [tr1, tr2, tr3, tr4, tr5, tr6, tr_01, tr_02]:
            if "processing" in _i.stats:
                del _i.stats.processing
        # test mergeable traces (contained ones)
        for tr_b in [tr2, tr3, tr4]:
            tr_a = tr1.copy()
            st = Stream([tr_a, tr_b])
            st._cleanup()
            assert st == Stream([tr1])
            assert type(st[0].data) == np.ndarray
        # test mergeable traces (adjacent ones)
        for tr_b in [tr5, tr6]:
            tr_a = tr1.copy()
            st = Stream([tr_a, tr_b])
            st._cleanup()
            assert len(st) == 1
            assert type(st[0].data) == np.ndarray
            st_result = Stream([tr1, tr_b])
            st_result.merge()
            assert st == st_result
        # test mergeable traces (overlapping ones)
        for tr_b in [tr_01, tr_02]:
            tr_a = tr1.copy()
            st = Stream([tr_a, tr_b])
            st._cleanup()
            assert len(st) == 1
            assert type(st[0].data) == np.ndarray
            st_result = Stream([tr1, tr_b])
            st_result.merge()
            assert st == st_result

        # test traces that should not be merged
        tr7 = tr1.copy()
        tr7.stats.sampling_rate *= 2
        tr8 = tr1.copy()
        tr8.stats.station = "AA"
        tr9 = tr1.copy()
        tr9.stats.starttime = end + 10 * delta
        # test some weird gaps near to one sample:
        tr10 = tr1.copy()
        tr10.stats.starttime = end + 0.5 * delta
        tr11 = tr1.copy()
        tr11.stats.starttime = end + 0.1 * delta
        tr12 = tr1.copy()
        tr12.stats.starttime = end + 0.8 * delta
        tr13 = tr1.copy()
        tr13.stats.starttime = end + 1.2 * delta
        # test non-mergeable traces
        for tr_b in [tr7, tr8, tr9, tr10, tr11, tr12, tr13]:
            tr_a = tr1.copy()
            st = Stream([tr_a, tr_b])
            # ignore UserWarnings
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('ignore', UserWarning)
                st._cleanup()
            assert st == Stream([tr_a, tr_b])

    def test_integrate_and_differentiate(self):
        """
        Test integration and differentiation methods of stream
        """
        st1 = read()
        st2 = read()

        st1.filter('lowpass', freq=1.0)
        st2.filter('lowpass', freq=1.0)

        st1.differentiate()
        st1.integrate()
        st2.integrate()
        st2.differentiate()

        np.testing.assert_array_almost_equal(
            st1[0].data[:-1], st2[0].data[:-1], decimal=5)

    def test_read(self):
        """
        Testing read function.
        """
        # 1 - default example
        # dtype
        tr = read(dtype=np.int64)[0]
        assert tr.data.dtype == np.int64
        # dtype is string
        tr2 = read(dtype='i8')[0]
        assert tr2.data.dtype == np.int64
        assert tr == tr2
        # start/end time
        tr2 = read(starttime=tr.stats.starttime + 1,
                   endtime=tr.stats.endtime - 2)[0]
        assert tr2.stats.starttime == tr.stats.starttime + 1
        assert tr2.stats.endtime == tr.stats.endtime - 2
        # headonly
        tr = read(headonly=True)[0]
        assert len(tr.data) == 0

        # 2 - via http
        # now in separate test case "test_read_url_via_network"

        # 3 - some example within obspy
        # dtype
        tr = read('/path/to/slist_float.ascii', dtype=np.int32)[0]
        assert tr.data.dtype == np.int32
        # start/end time
        tr2 = read('/path/to/slist_float.ascii',
                   starttime=tr.stats.starttime + 0.025,
                   endtime=tr.stats.endtime - 0.05)[0]
        assert tr2.stats.starttime == tr.stats.starttime + 0.025
        assert tr2.stats.endtime == tr.stats.endtime - 0.05
        # headonly
        tr = read('/path/to/slist_float.ascii', headonly=True)[0]
        assert len(tr.data) == 0
        # not existing
        with pytest.raises(OSError):
            read('/path/to/UNKNOWN')

        # 4 - file patterns
        path = os.path.dirname(__file__)
        ascii_path = os.path.join(path, "..", "..", "io", "ascii", "tests",
                                  "data")
        filename = os.path.join(ascii_path, 'slist.*')
        st = read(filename)
        assert len(st) == 2
        # exception if no file matches file pattern
        filename = path + os.sep + 'data' + os.sep + 'NOTEXISTING.*'
        with pytest.raises(Exception):
            read(filename)

        # argument headonly should not be used with start or end time or dtype
        with warnings.catch_warnings(record=True):
            # will usually warn only but here we force to raise an exception
            warnings.simplefilter('error', UserWarning)
            with pytest.raises(UserWarning):
                read('/path/to/slist_float.ascii',
                     headonly=True, starttime=0, endtime=1)

    def test_read_url_via_network(self):
        """
        Testing read function with an URL fetching data via network connection
        """
        # 2 - via http
        # dtype
        tr = read('https://examples.obspy.org/test.sac', dtype=np.int32)[0]
        assert tr.data.dtype == np.int32
        # start/end time
        tr2 = read('https://examples.obspy.org/test.sac',
                   starttime=tr.stats.starttime + 1,
                   endtime=tr.stats.endtime - 2)[0]
        assert tr2.stats.starttime == tr.stats.starttime + 1
        assert tr2.stats.endtime == tr.stats.endtime - 2
        # headonly
        tr = read('https://examples.obspy.org/test.sac', headonly=True)[0]
        assert tr.data.size == 0

    def test_read_path(self):
        """
        Test for reading a pathlib object.
        """
        base_path = Path(__file__).parent / 'data'
        data_path = base_path / 'IU_ULN_00_LH1_2015-07-18T02.mseed'
        assert data_path.exists()
        st = read(data_path)
        assert isinstance(st, Stream)

    def test_copy(self):
        """
        Testing the copy method of the Stream object.
        """
        st = read()
        st2 = st.copy()
        assert st == st2
        assert st2 == st
        assert not (st is st2)
        assert not (st2 is st)
        assert st.traces[0] == st2.traces[0]
        assert not (st.traces[0] is st2.traces[0])

    def test_merge_with_empty_trace(self):
        """
        Merging a stream containing a empty trace with a differing sampling
        rate should not fail.
        """
        # preparing a dataset
        tr = read()[0]
        st = tr / 3
        # empty and change sampling rate of second trace
        st[1].stats.sampling_rate = 0
        st[1].data = np.array([])
        # merge
        st.merge(fill_value='interpolate')
        assert len(st) == 1

    def test_rotate(self):
        """
        Testing the rotate method.
        """
        st = read()
        st += st.copy()
        st[3:].normalize()
        st2 = st.copy()
        # rotate to RT and back with 6 traces
        st.rotate(method='NE->RT', back_azimuth=30)
        assert (st[0].stats.channel[-1] + st[1].stats.channel[-1] +
                st[2].stats.channel[-1]) == 'ZRT'
        assert (st[3].stats.channel[-1] + st[4].stats.channel[-1] +
                st[5].stats.channel[-1]) == 'ZRT'
        st.rotate(method='RT->NE', back_azimuth=30)
        assert (st[0].stats.channel[-1] + st[1].stats.channel[-1] +
                st[2].stats.channel[-1]) == 'ZNE'
        assert (st[3].stats.channel[-1] + st[4].stats.channel[-1] +
                st[5].stats.channel[-1]) == 'ZNE'
        assert np.allclose(st[0].data, st2[0].data)
        assert np.allclose(st[1].data, st2[1].data)
        assert np.allclose(st[2].data, st2[2].data)
        assert np.allclose(st[3].data, st2[3].data)
        assert np.allclose(st[4].data, st2[4].data)
        assert np.allclose(st[5].data, st2[5].data)
        # again, with angles given in stats and just 2 components
        st = st2.copy()
        st = st[1:3] + st[4:]
        st[0].stats.back_azimuth = 190
        st[2].stats.back_azimuth = 200
        st.rotate(method='NE->RT')
        st.rotate(method='RT->NE')
        assert np.allclose(st[0].data, st2[1].data)
        assert np.allclose(st[1].data, st2[2].data)
        # rotate to LQT and back with 6 traces
        st = st2.copy()
        st.rotate(method='ZNE->LQT', back_azimuth=100, inclination=30)
        assert (st[0].stats.channel[-1] + st[1].stats.channel[-1] +
                st[2].stats.channel[-1]) == 'LQT'
        st.rotate(method='LQT->ZNE', back_azimuth=100, inclination=30)
        assert st[0].stats.channel[-1] + st[1].stats.channel[-1] + \
               st[2].stats.channel[-1] == 'ZNE'
        assert np.allclose(st[0].data, st2[0].data)
        assert np.allclose(st[1].data, st2[1].data)
        assert np.allclose(st[2].data, st2[2].data)
        assert np.allclose(st[3].data, st2[3].data)
        assert np.allclose(st[4].data, st2[4].data)
        assert np.allclose(st[5].data, st2[5].data)

        # unknown rotate method will raise ValueError
        with pytest.raises(ValueError):
            st.rotate(method='UNKNOWN')
        # rotating without back_azimuth raises TypeError
        st = Stream()
        with pytest.raises(TypeError):
            st.rotate(method='RT->NE')
        # rotating without inclination raises TypeError for LQT-> or ZNE->
        with pytest.raises(TypeError):
            st.rotate(method='LQT->ZNE', back_azimuth=30)
        # having traces with different timespans or sampling rates will fail
        st = read()
        st[1].stats.sampling_rate = 2.0
        with pytest.raises(ValueError):
            st.rotate(method='NE->RT')
        st = read()
        st[1].stats.starttime += 1
        with pytest.raises(ValueError):
            st.rotate(method='NE->RT')
        st = read()
        st[1].stats.sampling_rate = 2.0
        with pytest.raises(ValueError):
            st.rotate(method='ZNE->LQT')
        st = read()
        st[1].stats.starttime += 1
        with pytest.raises(ValueError):
            st.rotate(method='ZNE->LQT')

    def test_plot(self, mseed_stream):
        """
        Tests plot method if matplotlib is installed
        """
        mseed_stream.plot(show=False)

    def test_spectrogram(self, mseed_stream):
        """
        Tests spectrogram method if matplotlib is installed
        """
        mseed_stream.spectrogram(show=False)

    def test_deepcopy(self):
        """
        Tests __deepcopy__ method.

        http://lists.obspy.org/pipermail/obspy-users/2013-April/000451.html
        """
        # example stream
        st = read()
        # set a common header
        st[0].stats.network = 'AA'
        # set format specific header
        st[0].stats.mseed = AttribDict(dataquality='A')
        ct = deepcopy(st)
        # common header
        st[0].stats.network = 'XX'
        assert st[0].stats.network == 'XX'
        assert ct[0].stats.network == 'AA'
        # format specific headers
        st[0].stats.mseed.dataquality = 'X'
        assert st[0].stats.mseed.dataquality == 'X'
        assert ct[0].stats.mseed.dataquality == 'A'

    def test_write(self):
        # writing in unknown format raises ValueError
        st = read()
        with pytest.raises(ValueError):
            st.write('file.ext', format="UNKNOWN")

    def test_detrend(self):
        """
        Test detrend method of stream
        """
        t = np.arange(10)
        data = 0.1 * t + 1.

        tr = Trace(data=data.copy())
        st = Stream([tr, tr])
        st.detrend(type='simple')
        np.testing.assert_array_almost_equal(st[0].data, np.zeros(10))
        np.testing.assert_array_almost_equal(st[1].data, np.zeros(10))

        tr = Trace(data=data.copy())
        st = Stream([tr, tr])
        st.detrend(type='linear')
        np.testing.assert_array_almost_equal(st[0].data, np.zeros(10))
        np.testing.assert_array_almost_equal(st[1].data, np.zeros(10))

        data = np.zeros(10)
        data[3:7] = 1.

        tr = Trace(data=data.copy())
        st = Stream([tr, tr])
        st.detrend(type='simple')
        np.testing.assert_almost_equal(st[0].data[0], 0.)
        np.testing.assert_almost_equal(st[0].data[-1], 0.)
        np.testing.assert_almost_equal(st[1].data[0], 0.)
        np.testing.assert_almost_equal(st[1].data[-1], 0.)

        tr = Trace(data=data.copy())
        st = Stream([tr, tr])
        st.detrend(type='linear')
        np.testing.assert_almost_equal(st[0].data[0], -0.4)
        np.testing.assert_almost_equal(st[0].data[-1], -0.4)
        np.testing.assert_almost_equal(st[1].data[0], -0.4)
        np.testing.assert_almost_equal(st[1].data[-1], -0.4)

    def test_taper(self):
        """
        Test taper method of stream
        """
        data = np.ones(10)
        tr = Trace(data=data.copy())
        st = Stream([tr, tr])
        st.taper(max_percentage=0.05, type='cosine')
        for i in range(len(data)):
            assert st[0].data[i] <= 1.
            assert st[0].data[i] >= 0.
            assert st[1].data[i] <= 1.
            assert st[1].data[i] >= 0.

    def test_issue_540(self):
        """
        Trim with pad=True and given fill value should not return a masked
        NumPy array.
        """
        # fill_value = None
        st = read()
        assert len(st[0]) == 3000
        st.trim(starttime=st[0].stats.starttime - 0.01,
                endtime=st[0].stats.endtime + 0.01, pad=True, fill_value=None)
        assert len(st[0]) == 3002
        assert isinstance(st[0].data, np.ma.masked_array)
        assert st[0].data[0] is np.ma.masked
        assert st[0].data[1] is not np.ma.masked
        assert st[0].data[-2] is not np.ma.masked
        assert st[0].data[-1] is np.ma.masked
        # fill_value = 999
        st = read()
        assert len(st[1]) == 3000
        st.trim(starttime=st[1].stats.starttime - 0.01,
                endtime=st[1].stats.endtime + 0.01, pad=True, fill_value=999)
        assert len(st[1]) == 3002
        assert not isinstance(st[1].data, np.ma.masked_array)
        assert st[1].data[0] == 999
        assert st[1].data[-1] == 999
        # given fill_value but actually no padding at all
        st = read()
        assert len(st[2]) == 3000
        st.trim(starttime=st[2].stats.starttime, endtime=st[2].stats.endtime,
                pad=True, fill_value=-999)
        assert len(st[2]) == 3000
        assert not isinstance(st[2].data, np.ma.masked_array)

    def test_method_chaining(self):
        """
        Tests that method chaining works for all methods on the Stream object
        where it is sensible.
        """
        st1 = read()[0:1]
        st2 = read()

        assert len(st1) == 1
        assert len(st2) == 3

        # Test some list like methods.
        temp_st = st1.append(st1[0].copy())\
            .extend(st2)\
            .insert(0, st1[0].copy())\
            .remove(st1[0])
        assert temp_st is st1
        assert len(st1) == 5
        assert st1[0] == st1[1]
        assert st1[2] == st2[0]
        assert st1[3] == st2[1]
        assert st1[4] == st2[2]

        # Sort and reverse methods.
        st = st2.copy()
        st[0].stats.channel = "B"
        st[1].stats.channel = "C"
        st[2].stats.channel = "A"
        temp_st = st.sort(keys=["channel"]).reverse()
        assert temp_st is st
        assert [tr.stats.channel for tr in st] == ["C", "B", "A"]

        # The others are pretty hard to properly test and probably not worth
        # the effort. A simple demonstrating that they can be chained should be
        # enough.
        temp = st.trim(st[0].stats.starttime + 1, st[0].stats.starttime + 10)\
            .decimate(factor=2, no_filter=True)\
            .resample(st[0].stats.sampling_rate / 2)\
            .simulate(paz_remove={'poles': [-0.037004 + 0.037016j,
                                            -0.037004 - 0.037016j,
                                            -251.33 + 0j],
                                  'zeros': [0j, 0j],
                                  'gain': 60077000.0,
                                  'sensitivity': 2516778400.0})\
            .filter("lowpass", freq=2.0)\
            .differentiate()\
            .integrate()\
            .merge()\
            .cutout(st[0].stats.starttime + 2, st[0].stats.starttime + 2)\
            .detrend()\
            .taper(max_percentage=0.05, type="cosine")\
            .normalize()\
            .verify()\
            .trigger(type="zdetect", nsta=20)\
            .rotate(method="NE->RT", back_azimuth=40)

        # Use the processing chain to check the results. The trim(), merge(),
        # cutout(), verify(), and rotate() methods do not have an entry in the
        # processing chain.
        pr = st[0].stats.processing

        assert "decimate" in pr[1]
        assert "resample" in pr[2]
        assert "simulate" in pr[3]
        assert "filter" in pr[4] and "lowpass" in pr[4]
        assert "differentiate" in pr[5]
        assert "integrate" in pr[6]
        assert "trim" in pr[7]
        assert "detrend" in pr[8]
        assert "taper" in pr[9]
        assert "normalize" in pr[10]
        assert "trigger" in pr[11]

        assert temp is st
        # Cutout duplicates the number of traces.
        assert len(st), 6
        # Clearing also works for method chaining.
        assert len(st.clear()) == 0

    def test_simulate_seedresp_parser(self):
        """
        Test simulate() with giving a Parser object to use for RESP information
        in evalresp.
        Also tests usage without specifying a date for response lookup
        explicitely.
        """
        st = read()
        p = Parser("/path/to/dataless.seed.BW_RJOB")
        kwargs = dict(seedresp={'filename': p, 'units': "DIS"},
                      pre_filt=(1, 2, 50, 60), water_level=60)
        st.simulate(**kwargs)
        for tr in st:
            tr.stats.processing.pop()

        for resp_string, stringio in p.get_resp():
            stringio.seek(0, 0)
            component = resp_string[-1]
            with NamedTemporaryFile() as tf:
                with open(tf.name, "wb") as fh:
                    fh.write(stringio.read())
                tr1 = read().select(component=component)[0]
                tr1.simulate(**kwargs)
                tr1.stats.processing.pop()
            tr2 = st.select(component=component)[0]
            # There is some strange issue on Win32bit (see #2188) and Win64bit
            # (see #2330). Thus we just use assert_allclose() here instead of
            # testing for full equality.
            if platform.system() == "Windows":  # pragma: no cover
                assert tr1.stats == tr2.stats
                # as of 2020-01-21 we see some new Win fails with one sample
                # failing with:
                #    Mismatched elements: 1 / 3000 (0.0333%)
                #    Max absolute difference: 6.617444900424222e-24
                #    Max relative difference: 2.0
                # Maximum amplitudes in the trace are around 1e-8, so we can
                # live with an atol of 1e-22
                np.testing.assert_allclose(
                    tr1.data, tr2.data, rtol=1e-6, atol=1e-22)
            else:
                assert tr1 == tr2

    def test_select_empty_strings(self, mseed_stream):
        """
        Test that select works with values that evaluate True when testing with
        if (e.g. "", 0).
        """
        st = mseed_stream
        st[0].stats.location = "00"
        for tr in st[1:]:
            tr.stats.network = ""
            tr.stats.station = ""
            tr.stats.channel = ""
            tr.data = tr.data[0:0]
        st2 = Stream(st[1:])
        assert st.select(network="") == st2
        assert st.select(station="") == st2
        assert st.select(channel="") == st2
        assert st.select(npts=0) == st2

    def test_remove_response(self):
        """
        Tests that the remove_response method is called for all traces of a
        Stream object
        """
        st1 = read()
        st2 = read()
        for tr in st1:
            tr.remove_response(pre_filt=(0.1, 0.5, 30, 50))
        st2.remove_response(pre_filt=(0.1, 0.5, 30, 50))
        # There is some strange issue on Appveyor. Thus we just use
        # assert_allclose() here instead of testing for full equality.
        # https://ci.appveyor.com/project/obspy/obspy/
        #                                 builds/27495567/job/r4m7ely1nkjht20x
        if platform.system() == "Windows":  # pragma: no cover
            assert streams_almost_equal(st1, st2, atol=0, rtol=1e-6)
        else:
            assert st1 == st2

    def test_remove_sensitivity(self):
        """
        Tests that the remove_sensitivity method is called for all traces of a
        Stream object
        """
        st1 = read()
        st2 = read()
        for tr in st1:
            tr.remove_sensitivity()
        st2.remove_sensitivity()
        # Some Windows Appveyor CI runs have very minor differences..
        # There is some strange issue on Win32bit (see #2188) and Win64bit
        # (see #2330). Thus we just use assert_allclose() here instead of
        # testing for full equality.
        if platform.system() == "Windows":  # pragma: no cover
            for tr1, tr2 in zip(st1, st2):
                assert tr1.stats == tr2.stats
                np.testing.assert_allclose(tr1.data, tr2.data, rtol=1e-6)
        else:
            assert st1 == st2

    def test_interpolate(self):
        """
        Tests that the interpolate command is called for all traces of a
        Stream object.
        """
        st = read()
        with mock.patch("obspy.core.trace.Trace.interpolate") as patch:
            st.interpolate(sampling_rate=1.0, method="weighted_average_slopes")

        assert len(st) == patch.call_count
        expected = {"sampling_rate": 1.0, "method": "weighted_average_slopes"}
        assert expected == patch.call_args[1]

    def test_integrate(self):
        """
        Tests that the integrate command is called for all traces of a Stream
        object.
        """
        st1 = read()
        st2 = read()

        for tr in st1:
            tr.integrate()
        st2.integrate()
        assert st1 == st2

    def test_integrate_args(self):
        """
        Tests that the integrate command is called for all traces of a Stream
        object and options are passed along correctly.
        """
        st1 = read()
        st2 = read()

        for tr in st1:
            tr.integrate(method='cumtrapz')
        st2.integrate(method='cumtrapz')
        assert st1 == st2

    def test_misaligned_traces(self):
        """
        Tests the option to correct misaligned traces in `Stream._cleanup()`,
        which is the first thing done in any `Stream.merge()` operation.

        We create a simple trace and then prepare other traces that are shifted
        on sub-sample scale. With these shifted traces create different traces
        that are a) contained, b) overlapping, c) directly adjacent and d)
        separated by a "real" gap. In two loops we test a wide variety of
        combinations of actual misalignment ratio (percentage of sampling
        interval) and to-be-fixed misalignment threshold in cleanup().
        """
        samp_rate = 1.23
        delta = 1.0 / samp_rate
        tr = Trace(data=np.arange(10), header=dict(sampling_rate=samp_rate))
        t1 = tr.stats.starttime
        t2 = tr.stats.endtime

        def _gets_merged(tr_, ratio):
            st = Stream([tr.copy(), tr_.copy()])
            st._cleanup(misalignment_threshold=ratio)
            if len(st) == 1:
                return True
            elif len(st) == 2:
                return False
            raise Exception()

        for actual_misalign_percentage in np.linspace(0.01, 0.5, 6):
            # prepare two identical traces, misaligned
            tr_early = tr.copy()
            tr_early.stats.starttime -= actual_misalign_percentage * delta
            tr_late = tr.copy()
            tr_late.stats.starttime += actual_misalign_percentage * delta
            # prepare contained traces
            tr_contained1 = tr_early.copy()
            tr_contained1.trim(t1 + 2 * delta, t2 - delta)
            tr_contained2 = tr_late.copy()
            tr_contained2.trim(t1 + delta, t2 - 3 * delta)
            traces_contained = [tr_contained1, tr_contained2]
            # prepare overlapping traces
            tr_overlap1 = tr_early.copy()
            tr_overlap1.trim(t1 + 4 * delta, t2 + 4 * delta,
                             pad=True, fill_value=99)
            tr_overlap2 = tr_late.copy()
            tr_overlap2.trim(t1 - 5 * delta, t2 - 5 * delta,
                             pad=True, fill_value=99)
            traces_overlap = [tr_overlap1, tr_overlap2]
            # prepare directly adjacent traces
            tr_adjacent1 = tr_early.copy()
            tr_adjacent1.stats.starttime -= len(tr_adjacent1) * delta
            tr_adjacent2 = tr_late.copy()
            tr_adjacent2.stats.starttime -= len(tr_adjacent2) * delta
            tr_adjacent3 = tr_early.copy()
            tr_adjacent3.stats.starttime += len(tr) * delta
            tr_adjacent4 = tr_late.copy()
            tr_adjacent4.stats.starttime += len(tr) * delta
            traces_adjacent = [tr_adjacent1, tr_adjacent2, tr_adjacent3,
                               tr_adjacent4]
            # prepare traces with normal gap
            tr_gap1 = tr_adjacent1.copy()
            tr_gap1.stats.starttime -= delta
            tr_gap2 = tr_adjacent2.copy()
            tr_gap2.stats.starttime -= delta
            tr_gap3 = tr_adjacent3.copy()
            tr_gap3.stats.starttime += delta
            tr_gap4 = tr_adjacent4.copy()
            tr_gap4.stats.starttime += delta
            traces_gap = [tr_gap1, tr_gap2, tr_gap3, tr_gap4]
            for to_be_fixed_misalignmnt_ratio in np.linspace(0.011, 0.49, 5):
                # first test traces with normal gap, these should never change
                for trx in traces_gap:
                    assert not _gets_merged(
                        trx, to_be_fixed_misalignmnt_ratio)
                # now lets check the cases for which depending on
                # the combination of actual/to-be-fixed misalignment ratios
                # there should be a change/merge or not
                should_change = (to_be_fixed_misalignmnt_ratio >=
                                 actual_misalign_percentage)
                for trx in traces_contained + traces_overlap + traces_adjacent:
                    assert should_change == _gets_merged(
                        trx, to_be_fixed_misalignmnt_ratio)

    def test_slide(self):
        """
        Tests for sliding a window across a stream object.
        """
        # 0 - 20 seconds
        tr1 = Trace(data=np.linspace(0, 100, 101))
        tr1.stats.starttime = UTCDateTime(0.0)
        tr1.stats.sampling_rate = 5.0

        # 5 - 10 seconds
        tr2 = Trace(data=np.linspace(25, 75, 51))
        tr2.stats.starttime = UTCDateTime(5.0)
        tr2.stats.sampling_rate = 5.0

        # 15 - 20 seconds
        tr3 = Trace(data=np.linspace(75, 100, 26))
        tr3.stats.starttime = UTCDateTime(0.0)
        tr3.stats.sampling_rate = 15.0

        st = Stream(traces=[tr1, tr2, tr3])

        # First slice it in 4 pieces. Window length is in seconds.
        slices = []
        for window_st in st.slide(window_length=5.0, step=5.0):
            slices.append(window_st)

        assert len(slices) == 4
        assert slices[0] == st.slice(UTCDateTime(0), UTCDateTime(5))
        assert slices[1] == st.slice(UTCDateTime(5), UTCDateTime(10))
        assert slices[2] == st.slice(UTCDateTime(10), UTCDateTime(15))
        assert slices[3] == st.slice(UTCDateTime(15), UTCDateTime(20))

        # Different step which is the distance between two windows measured
        # from the start of the first window in seconds.
        slices = []
        for window_tr in st.slide(window_length=5.0, step=10.0):
            slices.append(window_tr)

        assert len(slices) == 2
        assert slices[0] == st.slice(UTCDateTime(0), UTCDateTime(5))
        assert slices[1] == st.slice(UTCDateTime(10), UTCDateTime(15))

        # Offset determines the initial starting point. It defaults to zero.
        slices = []
        for window_tr in st.slide(window_length=5.0, step=6.5, offset=8.5):
            slices.append(window_tr)

        assert len(slices) == 2
        assert slices[0] == st.slice(UTCDateTime(8.5), UTCDateTime(13.5))
        assert slices[1] == st.slice(UTCDateTime(15.0), UTCDateTime(20.0))

        # By default only full length windows will be returned so any
        # remainder that can no longer make up a full window will not be
        # returned.
        slices = []
        for window_tr in st.slide(window_length=15.0, step=15.0):
            slices.append(window_tr)

        assert len(slices) == 1
        assert slices[0] == st.slice(UTCDateTime(0.0), UTCDateTime(15.0))

        # But it can optionally be returned.
        slices = []
        for window_tr in st.slide(window_length=15.0, step=15.0,
                                  include_partial_windows=True):
            slices.append(window_tr)

        assert len(slices) == 2
        assert slices[0] == st.slice(UTCDateTime(0.0), UTCDateTime(15.0))
        assert slices[1] == st.slice(UTCDateTime(15.0), UTCDateTime(20.0))

        # Negative step lengths work together with an offset.
        slices = []
        for window_tr in st.slide(window_length=5.0, step=-5.0, offset=20.0):
            slices.append(window_tr)

        assert len(slices) == 4
        assert slices[0] == st.slice(UTCDateTime(15), UTCDateTime(20))
        assert slices[1] == st.slice(UTCDateTime(10), UTCDateTime(15))
        assert slices[2] == st.slice(UTCDateTime(5), UTCDateTime(10))
        assert slices[3] == st.slice(UTCDateTime(0), UTCDateTime(5))

    def test_slide_nearest_sample(self):
        """
        Tests that the nearest_sample argument is correctly passed to the
        slice function calls.
        """
        tr = Trace(data=np.linspace(0, 100, 101))
        tr.stats.starttime = UTCDateTime(0.0)
        tr.stats.sampling_rate = 5.0
        st = Stream(traces=[tr, tr.copy()])

        # It defaults to True.
        with mock.patch("obspy.core.trace.Trace.slice") as patch:
            patch.return_value = tr
            list(st.slide(5, 5))

        # Twice per window as two traces.
        assert patch.call_count == 8
        for arg in patch.call_args_list:
            assert arg[1]["nearest_sample"]

        # Force True.
        with mock.patch("obspy.core.trace.Trace.slice") as patch:
            patch.return_value = tr
            list(st.slide(5, 5, nearest_sample=True))

        # Twice per window as two traces.
        assert patch.call_count == 8
        for arg in patch.call_args_list:
            assert arg[1]["nearest_sample"]

        # Set to False.
        with mock.patch("obspy.core.trace.Trace.slice") as patch:
            patch.return_value = tr
            list(st.slide(5, 5, nearest_sample=False))

        # Twice per window as two traces.
        assert patch.call_count == 8
        for arg in patch.call_args_list:
            assert not arg[1]["nearest_sample"]

    def test_passing_kwargs_to_trace_detrend(self):
        """
        Simple regression test making sure kwargs are passed to the Trace's
        detrend method.
        """
        st = read()

        with mock.patch("obspy.core.trace.Trace.detrend") as patch:
            st.detrend("polynomial", order=2, plot=True)

        # 3 Traces.
        assert patch.call_count == 3

        for arg in patch.call_args_list:
            assert arg[1]["order"] == 2
            assert arg[1]["plot"]

    def test_read_check_compression(self):
        """
        Test to ensure calling read with check_compression=False does not
        call expensive tar or zip functions.
        """
        with mock.patch("tarfile.is_tarfile") as tar_p:
            with mock.patch("zipfile.is_zipfile") as zip_p:
                read('/path/to/slist.ascii', format='SLIST',
                     check_compression=False)

        # assert neither compression check function was called.
        assert tar_p.call_count == 0
        assert zip_p.call_count == 0

        # ensure compression checks get called when check_compression is True
        with mock.patch("tarfile.is_tarfile", return_value=0) as tar_p:
            with mock.patch("zipfile.is_zipfile", return_value=0) as zip_p:
                read('/path/to/slist.ascii', format='SLIST',
                     check_compression=True)
        assert tar_p.call_count == 1
        assert zip_p.call_count >= 1

    def test_rotate_to_zne(self):
        """
        Tests rotating all traces in stream to ZNE given an inventory object.
        """
        inv = read_inventory("/path/to/ffbx.stationxml", format="STATIONXML")
        parser = Parser("/path/to/ffbx.dataless")
        st_expected = read('/path/to/ffbx_rotated.slist', format='SLIST')
        st_unrotated = read("/path/to/ffbx_unrotated_gaps.mseed",
                            format="MSEED")
        for tr in st_expected:
            # ignore format specific keys and processing which also holds
            # version number
            tr.stats.pop('ascii')
            tr.stats.pop('_format')
        # check rotation using both Inventory and Parser as metadata input
        for metadata in (inv, parser):
            st = st_unrotated.copy()
            st.rotate("->ZNE", inventory=metadata)
            # do some checks on results
            assert len(st) == 30
            # compare data
            for tr_got, tr_expected in zip(st, st_expected):
                np.testing.assert_allclose(tr_got.data, tr_expected.data,
                                           rtol=1e-7)
            # compare stats
            for tr_expected, tr_got in zip(st_expected, st):
                # ignore format specific keys and processing which also holds
                # version number
                tr_got.stats.pop('mseed')
                tr_got.stats.pop('_format')
                tr_got.stats.pop('processing')
                assert tr_got.stats == tr_expected.stats

        # check that using something like `components="Z12"` also works,
        st = st_unrotated.copy()
        result = st.rotate("->ZNE", inventory=inv,
                           components='Z12')
        # check that rotation to ZNE worked..
        assert set(tr.stats.channel[-1] for tr in result) == set('ZNE')

    def test_write_empty_stream(self):
        """
        Tests error message when trying to write an empty stream
        """
        st = Stream()
        bio = io.BytesIO()
        msg = 'Can not write empty stream to file.'
        for format_ in _get_entry_points('obspy.plugin.waveform',
                                         'writeFormat').keys():
            with pytest.raises(ObsPyException, match=msg):
                st.write(bio, format=format_)

    def test_stack(self):
        """
        Tests stack method
        """
        # check number of traces and headers
        # stack with default options
        st = read()
        st2 = st.copy().stack()
        assert len(st2) == 1
        assert 'stack' in st2[0].stats
        assert st2[0].stats.stack.group == 'all'
        assert st2[0].stats.stack.count == 3
        assert st2[0].stats.stack.type == 'linear'
        # stack by SEED id
        st += st.copy()
        st2 = st.copy().stack('id')
        assert len(st2) == 3
        assert {tr.stats.stack.group for tr in st2} == {tr.id for tr in st}
        assert {tr.id for tr in st2} == {tr.id for tr in st}
        assert st2[0].stats.stack.count == 2
        # stack by other metadata
        for tr in st[3:]:
            tr.stats.station = 'OTH'
        st2 = st.copy().stack('{network}.{station}')
        assert len(st2) == 2
        assert {tr.stats.stack.group for tr in st2} == \
            {'.'.join((tr.stats.network, tr.stats.station)) for tr in st}
        assert st2[0].stats.stack.count == 3

        # check npts_tol option and correct npts of stack
        st = read()
        st[2].data = st[2].data[:-1]
        npts = len(st[0])
        # self.assertRaisesRegex(ValueError, 'number of points', st.stack)
        with pytest.raises(ValueError):
            st.stack()
        st2 = st.copy().stack(npts_tol=1)
        assert len(st2) == 1
        assert len(st2[0]) == npts-1
        assert len(st[0]) == npts

        # check correct setting of metadata
        st = read()
        st[0].stats.back_azimuth -= 10
        st[0].stats.array1 = np.array([1, 2])
        st[1].stats.array1 = np.array([1, 2, 3])
        st[2].stats.array1 = 'no array here'
        for tr in st:
            tr.stats.array2 = np.array([3, 4])
        st[0].stats.sub1 = {'a': 1}
        st[1].stats.sub1 = {'b': np.array([1, 2])}
        st[2].stats.sub1 = 5
        sub2_dict = {'a': 'b', 'b': 'c', 'nd': np.array([1, 2])}
        for tr in st:
            tr.stats.sub2 = sub2_dict
        st2 = st.copy().stack()
        assert st2[0].stats.station == st[0].stats.station
        assert st2[0].stats.inclination == st[0].stats.inclination
        assert st2[0].stats.starttime == st[0].stats.starttime
        assert st2[0].stats.channel == ''
        assert 'back_azimuth' not in st2[0].stats
        assert 'array1' not in st2[0].stats
        assert 'array2' in st2[0].stats
        assert 'sub1' not in st2[0].stats
        assert 'sub2' in st2[0].stats
        assert st2[0].stats.sub2.keys() == sub2_dict.keys()
        st[1].stats.starttime += 10
        st2 = st.copy().stack()
        assert st2[0].stats.starttime == UTCDateTime(0)
        st2 = st.copy().stack(time_tol=11)
        assert st2[0].stats.starttime == st[0].stats.starttime
        st[0].stats.sampling_rate /= 10
        # self.assertRaisesRegex(ValueError, 'Sampling rate', st.stack)
        with pytest.raises(ValueError):
            st.stack()

        # Check pw and root stacking types, these must result in the linear
        # stack for order 0, resp. 1.
        # For larger order pw stack is always of smaller magnitude than linear
        # stack. For a root stack this is not always the case, but its
        # magnitude is definitely smaller if all stacked samples have the
        # same sign.
        st = read()
        st2 = st.copy().stack()
        st3 = st.copy().stack(stack_type=('pw', 0))
        st4 = st.copy().stack(stack_type=('root', 1))
        assert len(st2) == 1
        assert len(st3) == 1
        assert len(st4) == 1
        np.testing.assert_allclose(st3[0].data, st2[0].data)
        np.testing.assert_allclose(st4[0].data, st2[0].data)
        st3 = st.copy().stack(stack_type=('pw', 2))
        st4 = st.copy().stack(stack_type=('root', 2))
        assert np.sum(np.abs(st3[0].data) <= np.abs(st2[0].data)) == npts
        all_data = np.array([tr.data for tr in st])
        same_sign = np.logical_or(np.all(all_data < 0, axis=0),
                                  np.all(all_data > 0, axis=0))
        npts = np.sum(same_sign)
        assert np.sum(np.abs(st4[0].data[same_sign]) <=
                      np.abs(st2[0].data[same_sign])) == npts

    def test_stream_trim_slice_same_length(self):
        """
        See issue #2608

        End time 00:20:04.045 is in the middle of two samples
        00:20:04.04 versus 00:20:04.05
        """
        st = read()
        utc = st[0].stats.starttime + 1.045
        n1 = len(st.slice(None, utc)[0])
        n2 = len(st.trim(None, utc)[0])
        assert n1 == n2
