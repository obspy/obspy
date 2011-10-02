# -*- coding: utf-8 -*-

from copy import deepcopy
from obspy.core import UTCDateTime, Stream, Trace, read
import numpy as np
import pickle
import unittest


class StreamTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.stream.Stream.
    """

    def setUp(self):
        # set specific seed value such that random numbers are reproducible
        np.random.seed(815)
        header = {'network': 'BW', 'station': 'BGLD',
                  'starttime': UTCDateTime(2007, 12, 31, 23, 59, 59, 915000),
                  'npts': 412, 'sampling_rate': 200.0,
                  'channel': 'EHE'}
        trace1 = Trace(data=np.random.randint(0, 1000, 412),
                       header=deepcopy(header))
        header['starttime'] = UTCDateTime(2008, 1, 1, 0, 0, 4, 35000)
        header['npts'] = 824
        trace2 = Trace(data=np.random.randint(0, 1000, 824),
                       header=deepcopy(header))
        header['starttime'] = UTCDateTime(2008, 1, 1, 0, 0, 10, 215000)
        trace3 = Trace(data=np.random.randint(0, 1000, 824),
                       header=deepcopy(header))
        header['starttime'] = UTCDateTime(2008, 1, 1, 0, 0, 18, 455000)
        header['npts'] = 50668
        trace4 = Trace(data=np.random.randint(0, 1000, 50668),
                       header=deepcopy(header))
        self.mseed_stream = Stream(traces=[trace1, trace2, trace3, trace4])
        header = {'network': '', 'station': 'RNON ', 'location': '',
                  'starttime': UTCDateTime(2004, 6, 9, 20, 5, 59, 849998),
                  'sampling_rate': 200.0, 'npts': 12000,
                  'channel': '  Z'}
        trace = Trace(data=np.random.randint(0, 1000, 12000), header=header)
        self.gse2_stream = Stream(traces=[trace])

    def test_setitem(self):
        """
        Tests the __setitem__ method of the Stream object.
        """
        stream = self.mseed_stream
        stream[0] = stream[3]
        self.assertEqual(stream[0], stream[3])
        st = deepcopy(stream)
        stream[0].data[0:10] = 999
        self.assertNotEqual(st[0].data[0], 999)
        st[0] = stream[0]
        np.testing.assert_array_equal(stream[0].data[:10],
                                      np.ones(10, dtype='int') * 999)

    def test_getitem(self):
        """
        Tests the __getitem__ method of the Stream object.
        """
        stream = self.mseed_stream
        self.assertEqual(stream[0], stream.traces[0])
        self.assertEqual(stream[-1], stream.traces[-1])
        self.assertEqual(stream[3], stream.traces[3])

    def test_add(self):
        """
        Tests the adding of two stream objects.
        """
        stream = self.mseed_stream
        self.assertEqual(4, len(stream))
        # Add the same stream object to itself.
        stream = stream + stream
        self.assertEqual(8, len(stream))
        # This will not create copies of Traces and thus the objects should
        # be identical (and the Traces attributes should be identical).
        for _i in xrange(4):
            self.assertEqual(stream[_i], stream[_i + 4])
            self.assertEqual(stream[_i] == stream[_i + 4], True)
            self.assertEqual(stream[_i] != stream[_i + 4], False)
            self.assertEqual(stream[_i] is stream[_i + 4], True)
            self.assertEqual(stream[_i] is not stream[_i + 4], False)
        # Now add another stream to it.
        other_stream = self.gse2_stream
        self.assertEqual(1, len(other_stream))
        new_stream = stream + other_stream
        self.assertEqual(9, len(new_stream))
        # The traces of all streams are copied.
        for _i in xrange(8):
            self.assertEqual(new_stream[_i], stream[_i])
            self.assertEqual(new_stream[_i] is stream[_i], True)
        # Also test for the newly added stream.
        self.assertEqual(new_stream[8], other_stream[0])
        self.assertEqual(new_stream[8].stats, other_stream[0].stats)
        np.testing.assert_array_equal(new_stream[8].data, other_stream[0].data)

    def test_iadd(self):
        """
        Tests the __iadd__ method of the Stream objects.
        """
        stream = self.mseed_stream
        self.assertEqual(4, len(stream))
        other_stream = self.gse2_stream
        self.assertEqual(1, len(other_stream))
        # Add the other stream to the stream.
        stream += other_stream
        # This will leave the Traces of the new stream and create a deepcopy of
        # the other Stream's Traces
        self.assertEqual(5, len(stream))
        self.assertEqual(other_stream[0], stream[-1])
        self.assertEqual(other_stream[0].stats, stream[-1].stats)
        np.testing.assert_array_equal(other_stream[0].data, stream[-1].data)

    def test_addTraceToStream(self):
        """
        Tests using a Trace on __add__ and __iadd__ methods of the Stream.
        """
        st0 = read()
        st1 = st0[0:2]
        tr = st0[2]
        # __add__
        self.assertEqual(st1.__add__(tr), st0)
        self.assertEqual(st1 + tr, st0)
        # __iadd__
        st1 += tr
        self.assertEqual(st1, st0)

    def test_append(self):
        """
        Tests the append method of the Stream object.
        """
        stream = self.mseed_stream
        # Check current count of traces
        self.assertEqual(len(stream), 4)
        # Append first traces to the Stream object.
        stream.append(stream[0])
        self.assertEqual(len(stream), 5)
        # This is supposed to make a deepcopy of the Trace and thus the two
        # Traces are not identical.
        self.assertEqual(stream[0], stream[-1])
        # But the attributes and data values should be identical.
        self.assertEqual(stream[0].stats, stream[-1].stats)
        np.testing.assert_array_equal(stream[0].data, stream[-1].data)
        # Append the same again
        stream.append(stream[0])
        self.assertEqual(len(stream), 6)
        # Now the two objects should be identical.
        self.assertEqual(stream[0], stream[-1])
        # Using append with a list of Traces, or int, or ... should fail.
        self.assertRaises(TypeError, stream.append, stream[:])
        self.assertRaises(TypeError, stream.append, 1)
        self.assertRaises(TypeError, stream.append, stream[0].data)

    def test_countAndLen(self):
        """
        Tests the count method and __len__ attribute of the Stream object.
        """
        # empty stream without traces
        stream = Stream()
        self.assertEqual(len(stream), 0)
        self.assertEqual(stream.count(), 0)
        # stream with traces
        stream = self.mseed_stream
        self.assertEqual(len(stream), 4)
        self.assertEqual(stream.count(), 4)

    def test_extend(self):
        """
        Tests the extend method of the Stream object.
        """
        stream = self.mseed_stream
        # Check current count of traces
        self.assertEqual(len(stream), 4)
        # Extend the Stream object with the first two traces.
        stream.extend(stream[0:2])
        self.assertEqual(len(stream), 6)
        # This is NOT supposed to make a deepcopy of the Trace and thus the two
        # Traces compare equal and are identical.
        self.assertEqual(stream[0], stream[-2])
        self.assertEqual(stream[1], stream[-1])
        self.assertTrue(stream[0] is stream[-2])
        self.assertTrue(stream[1] is stream[-1])
        # Using extend with a single Traces, or a wrong list, or ...
        # should fail.
        self.assertRaises(TypeError, stream.extend, stream[0])
        self.assertRaises(TypeError, stream.extend, 1)
        self.assertRaises(TypeError, stream.extend, [stream[0], 1])

    def test_insert(self):
        """
        Tests the insert Method of the Stream object.
        """
        stream = self.mseed_stream
        self.assertEqual(4, len(stream))
        # Insert the last Trace before the second trace.
        stream.insert(1, stream[-1])
        self.assertEqual(len(stream), 5)
        # This is supposed to make a deepcopy of the Trace and thus the two
        # Traces are not identical.
        #self.assertNotEqual(stream[1], stream[-1])
        self.assertEqual(stream[1], stream[-1])
        # But the attributes and data values should be identical.
        self.assertEqual(stream[1].stats, stream[-1].stats)
        np.testing.assert_array_equal(stream[1].data, stream[-1].data)
        # Do the same again
        stream.insert(1, stream[-1])
        self.assertEqual(len(stream), 6)
        # Now the two Traces should be identical
        self.assertEqual(stream[1], stream[-1])
        # Do the same with a list of traces this time.
        # Insert the last two Trace before the second trace.
        stream.insert(1, stream[-2:])
        self.assertEqual(len(stream), 8)
        # This is supposed to make a deepcopy of the Trace and thus the two
        # Traces are not identical.
        self.assertEqual(stream[1], stream[-2])
        self.assertEqual(stream[2], stream[-1])
        # But the attributes and data values should be identical.
        self.assertEqual(stream[1].stats, stream[-2].stats)
        np.testing.assert_array_equal(stream[1].data, stream[-2].data)
        self.assertEqual(stream[2].stats, stream[-1].stats)
        np.testing.assert_array_equal(stream[2].data, stream[-1].data)
        # Do the same again
        stream.insert(1, stream[-2:])
        self.assertEqual(len(stream), 10)
        # Now the two Traces should be identical
        self.assertEqual(stream[1], stream[-2])
        self.assertEqual(stream[2], stream[-1])
        # Using insert without a single Traces or a list of Traces should fail.
        self.assertRaises(TypeError, stream.insert, 1, 1)
        self.assertRaises(TypeError, stream.insert, stream[0], stream[0])
        self.assertRaises(TypeError, stream.insert, 1, [stream[0], 1])

    def test_getGaps(self):
        """
        Tests the getGaps method of the Stream objects.

        It is compared directly to the obspy.mseed method getGapsList which is
        assumed to be correct.
        """
        stream = self.mseed_stream
        gap_list = stream.getGaps()
        # Gaps list created with obspy.mseed
        mseed_gap_list = [
            ('BW', 'BGLD', '', 'EHE',
             UTCDateTime(2008, 1, 1, 0, 0, 1, 970000),
             UTCDateTime(2008, 1, 1, 0, 0, 4, 35000),
             2.0649999999999999, 412.0),
            ('BW', 'BGLD', '', 'EHE',
             UTCDateTime(2008, 1, 1, 0, 0, 8, 150000),
             UTCDateTime(2008, 1, 1, 0, 0, 10, 215000),
             2.0649999999999999, 412.0),
            ('BW', 'BGLD', '', 'EHE',
             UTCDateTime(2008, 1, 1, 0, 0, 14, 330000),
             UTCDateTime(2008, 1, 1, 0, 0, 18, 455000),
             4.125, 824.0)]
        # Assert the number of gaps.
        self.assertEqual(len(mseed_gap_list), len(gap_list))
        for _i in xrange(len(mseed_gap_list)):
            # Compare the string values directly.
            for _j in xrange(6):
                self.assertEqual(gap_list[_i][_j], mseed_gap_list[_i][_j])
            # The small differences are probably due to rounding errors.
            self.assertAlmostEqual(mseed_gap_list[_i][6], gap_list[_i][6],
                                   places=3)
            self.assertAlmostEqual(mseed_gap_list[_i][7], gap_list[_i][7],
                                   places=3)

    def test_getGapsMultiplexedStreams(self):
        """
        Tests the getGaps method of the Stream objects.
        """
        data = np.random.randint(0, 1000, 412)
        # different channels
        st = Stream()
        for channel in ['EHZ', 'EHN', 'EHE']:
            st.append(Trace(data=data, header={'channel': channel}))
        self.assertEqual(len(st.getGaps()), 0)
        # different locations
        st = Stream()
        for location in ['', '00', '01']:
            st.append(Trace(data=data, header={'location': location}))
        self.assertEqual(len(st.getGaps()), 0)
        # different stations
        st = Stream()
        for station in ['MANZ', 'ROTZ', 'BLAS']:
            st.append(Trace(data=data, header={'station': station}))
        self.assertEqual(len(st.getGaps()), 0)
        # different networks
        st = Stream()
        for network in ['BW', 'GE', 'GR']:
            st.append(Trace(data=data, header={'network': network}))
        self.assertEqual(len(st.getGaps()), 0)

    def test_pop(self):
        """
        Test the pop method of the Stream object.
        """
        stream = self.mseed_stream
        # Make a copy of the Traces.
        traces = deepcopy(stream[:])
        # Remove and return the last Trace.
        temp_trace = stream.pop()
        self.assertEqual(3, len(stream))
        # Assert attributes. The objects itself are not identical.
        self.assertEqual(temp_trace.stats, traces[-1].stats)
        np.testing.assert_array_equal(temp_trace.data, traces[-1].data)
        # Remove the last copied Trace.
        traces.pop()
        # Remove and return the second Trace.
        temp_trace = stream.pop(1)
        # Assert attributes. The objects itself are not identical.
        self.assertEqual(temp_trace.stats, traces[1].stats)
        np.testing.assert_array_equal(temp_trace.data, traces[1].data)
        # Remove the second copied Trace.
        traces.pop(1)
        # Compare all remaining Traces.
        self.assertEqual(2, len(stream))
        self.assertEqual(2, len(traces))
        for _i in xrange(len(traces)):
            self.assertEqual(traces[_i].stats, stream[_i].stats)
            np.testing.assert_array_equal(traces[_i].data, stream[_i].data)

    def test_slice(self):
        """
        Tests the slice of Stream object.
        This is not the test for the Stream objects slice method which is
        passed through to Trace object.
        """
        stream = self.mseed_stream
        self.assertEqual(stream[0:], stream[0:])
        self.assertEqual(stream[:2], stream[:2])
        self.assertEqual(stream[:], stream[:])
        self.assertEqual(len(stream), 4)
        new_stream = stream[1:3]
        self.assertTrue(isinstance(new_stream, Stream))
        self.assertEqual(len(new_stream), 2)
        self.assertEqual(new_stream[0].stats, stream[1].stats)
        self.assertEqual(new_stream[1].stats, stream[2].stats)

    def test_slice2(self):
        """
        Slicing using a step should return Stream objects.
        """
        tr1 = Trace()
        tr2 = Trace()
        tr3 = Trace()
        tr4 = Trace()
        tr5 = Trace()
        st = Stream([tr1, tr2, tr3, tr4, tr5])
        self.assertEqual(st[0:6].traces, [tr1, tr2, tr3, tr4, tr5])
        self.assertEqual(st[0:6:1].traces, [tr1, tr2, tr3, tr4, tr5])
        self.assertEqual(st[0:6:2].traces, [tr1, tr3, tr5])
        self.assertEqual(st[1:6:2].traces, [tr2, tr4])
        self.assertEqual(st[1:6:6].traces, [tr2])

    def test_pop2(self):
        """
        Test the pop method of the Stream object.
        """
        trace = Trace(data=np.arange(0, 1000))
        st = Stream([trace])
        st = st + st + st + st
        self.assertEqual(len(st), 4)
        st.pop()
        self.assertEqual(len(st), 3)
        st[1].stats.station = 'MUH'
        st.pop(0)
        self.assertEqual(len(st), 2)
        self.assertEqual(st[0].stats.station, 'MUH')

    def test_remove(self):
        """
        Tests the remove method of the Stream object.
        """
        stream = self.mseed_stream
        # Make a copy of the Traces.
        stream2 = deepcopy(stream)
        # Use the remove method of the Stream object and of the list of Traces.
        stream.remove(stream[1])
        del(stream2[1])
        stream.remove(stream[-1])
        del(stream2[-1])
        # Compare remaining Streams.
        self.assertTrue(stream == stream2)

    def test_reverse(self):
        """
        Tests the reverse method of the Stream object.
        """
        stream = self.mseed_stream
        # Make a copy of the Traces.
        traces = deepcopy(stream[:])
        # Use reversing of the Stream object and of the list.
        stream.reverse()
        traces.reverse()
        # Compare all Traces.
        self.assertEqual(4, len(stream))
        self.assertEqual(4, len(traces))
        for _i in xrange(len(traces)):
            self.assertEqual(traces[_i].stats, stream[_i].stats)
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
             'npts': 100}]
        # Make stream object for test case
        traces = []
        for header in headers:
            traces.append(Trace(data=np.random.randint(0, 1000, 100),
                                header=header))
        stream = Stream(traces=traces)
        # Test cases:
        stream2 = stream.select()
        self.assertEquals(stream, stream2)
        self.assertRaises(Exception, stream.select, channel="EHZ",
                          component="N")
        stream2 = stream.select(channel='EHE')
        self.assertEquals(len(stream2), 0)
        stream2 = stream.select(channel='EHZ')
        self.assertEquals(len(stream2), 2)
        self.assertTrue(stream[0] in stream2)
        self.assertTrue(stream[4] in stream2)
        stream2 = stream.select(component='Z')
        self.assertEquals(len(stream2), 3)
        self.assertTrue(stream[0] in stream2)
        self.assertTrue(stream[2] in stream2)
        self.assertTrue(stream[4] in stream2)
        stream2 = stream.select(component='n')
        self.assertEquals(len(stream2), 2)
        self.assertTrue(stream[1] in stream2)
        self.assertTrue(stream[3] in stream2)
        stream2 = stream.select(channel='BHZ', component='Z',
                sampling_rate='20.0', network='AA', station='ZZZZ', npts=100)
        self.assertEquals(len(stream2), 1)
        self.assertTrue(stream[2] in stream2)
        stream2 = stream.select(channel='EHZ', station="XXXX")
        self.assertEquals(len(stream2), 1)
        self.assertTrue(stream[4] in stream2)
        stream2 = stream.select(network='AA')
        self.assertEquals(len(stream2), 3)
        self.assertTrue(stream[0] in stream2)
        self.assertTrue(stream[2] in stream2)
        self.assertTrue(stream[4] in stream2)
        stream2 = stream.select(sampling_rate=20.0)
        self.assertEquals(len(stream2), 2)
        self.assertTrue(stream[2] in stream2)
        self.assertTrue(stream[3] in stream2)
        # tests for wildcarded channel:
        stream2 = stream.select(channel='B*')
        self.assertEquals(len(stream2), 2)
        self.assertTrue(stream[2] in stream2)
        self.assertTrue(stream[3] in stream2)
        stream2 = stream.select(channel='EH*')
        self.assertEquals(len(stream2), 3)
        self.assertTrue(stream[0] in stream2)
        self.assertTrue(stream[1] in stream2)
        self.assertTrue(stream[4] in stream2)
        stream2 = stream.select(channel='*Z')
        self.assertEquals(len(stream2), 3)
        self.assertTrue(stream[0] in stream2)
        self.assertTrue(stream[2] in stream2)
        self.assertTrue(stream[4] in stream2)
        # tests for other wildcard operations:
        stream2 = stream.select(station='[XY]*')
        self.assertEquals(len(stream2), 3)
        self.assertTrue(stream[1] in stream2)
        self.assertTrue(stream[3] in stream2)
        self.assertTrue(stream[4] in stream2)
        stream2 = stream.select(station='[A-Y]*')
        self.assertEquals(len(stream2), 3)
        self.assertTrue(stream[1] in stream2)
        self.assertTrue(stream[3] in stream2)
        self.assertTrue(stream[4] in stream2)
        stream2 = stream.select(station='[A-Y]??*', network='A?')
        self.assertEquals(len(stream2), 1)
        self.assertTrue(stream[4] in stream2)

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
        self.assertEqual([i.stats.sampling_rate for i in stream.traces],
                         [300.0, 500.0, 400.0, 200.0, 100.0])
        # Sort after sampling_rate.
        stream.sort(keys=['sampling_rate'])
        self.assertEqual([i.stats.sampling_rate for i in stream.traces],
                         [100.0, 200.0, 300.0, 400.0, 500.0])
        # Sort after channel and sampling rate.
        stream.sort(keys=['channel', 'sampling_rate'])
        self.assertEqual([i.stats.sampling_rate for i in stream.traces],
                         [200.0, 500.0, 300.0, 400.0, 100.0])
        # Sort after npts and sampling_rate and endtime.
        stream.sort(keys=['npts', 'sampling_rate', 'endtime'])
        self.assertEqual([i.stats.sampling_rate for i in stream.traces],
                         [100.0, 200.0, 300.0, 400.0, 500.0])
        # Sorting without a list or a wrong item string should fail.
        self.assertRaises(TypeError, stream.sort, keys=1)
        self.assertRaises(TypeError, stream.sort, keys='sampling_rate')
        self.assertRaises(TypeError, stream.sort, keys=['npts', 'starttime',
                                                        'wrong_value'])

    def test_sortingTwice(self):
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
        self.assertEqual(a, b)

    def test_mergeWithDifferentCalibrationFactors(self):
        """
        Test the merge method of the Stream object.
        """
        # 1 - different calibration factors for the same channel should fail
        tr1 = Trace(data=np.zeros(5))
        tr1.stats.calib = 1.0
        tr2 = Trace(data=np.zeros(5))
        tr2.stats.calib = 2.0
        st = Stream([tr1, tr2])
        self.assertRaises(Exception, st.merge)
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

    def test_mergeWithDifferentSamplingRates(self):
        """
        Test the merge method of the Stream object.
        """
        # 1 - different sampling rates for the same channel should fail
        tr1 = Trace(data=np.zeros(5))
        tr1.stats.sampling_rate = 200
        tr2 = Trace(data=np.zeros(5))
        tr2.stats.sampling_rate = 50
        st = Stream([tr1, tr2])
        self.assertRaises(Exception, st.merge)
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

    def test_mergeWithDifferentDatatypes(self):
        """
        Test the merge method of the Stream object.
        """
        # 1 - different dtype for the same channel should fail
        tr1 = Trace(data=np.zeros(5, dtype="int32"))
        tr2 = Trace(data=np.zeros(5, dtype="float32"))
        st = Stream([tr1, tr2])
        self.assertRaises(Exception, st.merge)
        # 2 - different sampling rates for the different channels is ok
        tr1 = Trace(data=np.zeros(5, dtype="int32"))
        tr1.stats.channel = 'EHE'
        tr2 = Trace(data=np.zeros(5, dtype="float32"))
        tr2.stats.channel = 'EHZ'
        tr3 = Trace(data=np.zeros(5, dtype="int32"))
        tr3.stats.channel = 'EHE'
        tr4 = Trace(data=np.zeros(5, dtype="float32"))
        tr4.stats.channel = 'EHZ'
        st = Stream([tr1, tr2, tr3, tr4])
        st.merge()

    def test_mergeGaps(self):
        """
        Test the merge method of the Stream object.
        """
        stream = self.mseed_stream
        start = UTCDateTime("2007-12-31T23:59:59.915000")
        end = UTCDateTime("2008-01-01T00:04:31.790000")
        self.assertEquals(len(stream), 4)
        self.assertEquals(len(stream[0]), 412)
        self.assertEquals(len(stream[1]), 824)
        self.assertEquals(len(stream[2]), 824)
        self.assertEquals(len(stream[3]), 50668)
        self.assertEquals(stream[0].stats.starttime, start)
        self.assertEquals(stream[3].stats.endtime, end)
        for i in xrange(4):
            self.assertEquals(stream[i].stats.sampling_rate, 200)
            self.assertEquals(stream[i].getId(), 'BW.BGLD..EHE')
        stream.verify()
        # merge it
        stream.merge()
        stream.verify()
        self.assertEquals(len(stream), 1)
        self.assertEquals(len(stream[0]), stream[0].data.size)
        self.assertEquals(stream[0].stats.starttime, start)
        self.assertEquals(stream[0].stats.endtime, end)
        self.assertEquals(stream[0].stats.sampling_rate, 200)
        self.assertEquals(stream[0].getId(), 'BW.BGLD..EHE')

    def test_mergeGaps2(self):
        """
        Test the merge method of the Stream object on two traces with a gap in
        between.
        """
        tr1 = Trace(data=np.ones(4, dtype=np.int32) * 1)
        tr2 = Trace(data=np.ones(3, dtype=np.int32) * 5)
        tr2.stats.starttime = tr1.stats.starttime + 9
        stream = Stream([tr1, tr2])
        #1 - masked array
        # Trace 1: 1111
        # Trace 2:          555
        # 1 + 2  : 1111-----555
        st = stream.copy()
        st.merge()
        self.assertEqual(len(st), 1)
        self.assertTrue(isinstance(st[0].data, np.ma.masked_array))
        self.assertEqual(st[0].data.tolist(),
                         [1, 1, 1, 1, None, None, None, None, None, 5, 5, 5])
        #2 - fill in zeros
        # Trace 1: 1111
        # Trace 2:          555
        # 1 + 2  : 111100000555
        st = stream.copy()
        st.merge(fill_value=0)
        self.assertEqual(len(st), 1)
        self.assertTrue(isinstance(st[0].data, np.ndarray))
        self.assertEqual(st[0].data.tolist(),
                         [1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 5, 5])
        #2b - fill in some other user-defined value
        # Trace 1: 1111
        # Trace 2:          555
        # 1 + 2  : 111199999555
        st = stream.copy()
        st.merge(fill_value=9)
        self.assertEqual(len(st), 1)
        self.assertTrue(isinstance(st[0].data, np.ndarray))
        self.assertEqual(st[0].data.tolist(),
                         [1, 1, 1, 1, 9, 9, 9, 9, 9, 5, 5, 5])
        #3 - use last value of first trace
        # Trace 1: 1111
        # Trace 2:          555
        # 1 + 2  : 111111111555
        st = stream.copy()
        st.merge(fill_value='latest')
        self.assertEqual(len(st), 1)
        self.assertTrue(isinstance(st[0].data, np.ndarray))
        self.assertEqual(st[0].data.tolist(),
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5])
        #4 - interpolate
        # Trace 1: 1111
        # Trace 2:          555
        # 1 + 2  : 111112334555
        st = stream.copy()
        st.merge(fill_value='interpolate')
        self.assertEqual(len(st), 1)
        self.assertTrue(isinstance(st[0].data, np.ndarray))
        self.assertEqual(st[0].data.tolist(),
                         [1, 1, 1, 1, 1, 2, 3, 3, 4, 5, 5, 5])

    def test_mergeOverlapsDefaultMethod(self):
        """
        Test the merge method of the Stream object.
        """
        #1 - overlapping trace with differing data
        # Trace 1: 0000000
        # Trace 2:      1111111
        # 1 + 2  : 00000--11111
        tr1 = Trace(data=np.zeros(7))
        tr2 = Trace(data=np.ones(7))
        tr2.stats.starttime = tr1.stats.starttime + 5
        st = Stream([tr1, tr2])
        st.merge()
        self.assertEqual(len(st), 1)
        self.assertTrue(isinstance(st[0].data, np.ma.masked_array))
        self.assertEqual(st[0].data.tolist(),
                         [0, 0, 0, 0, 0, None, None, 1, 1, 1, 1, 1])
        #2 - overlapping trace with same data
        # Trace 1: 0123456
        # Trace 2:      56789
        # 1 + 2  : 0123456789
        tr1 = Trace(data=np.arange(7))
        tr2 = Trace(data=np.arange(5, 10))
        tr2.stats.starttime = tr1.stats.starttime + 5
        st = Stream([tr1, tr2])
        st.merge()
        self.assertEqual(len(st), 1)
        self.assertTrue(isinstance(st[0].data, np.ndarray))
        np.testing.assert_array_equal(st[0].data, np.arange(10))
        #
        #3 - contained overlap with same data
        # Trace 1: 0123456789
        # Trace 2:      56
        # 1 + 2  : 0123456789
        tr1 = Trace(data=np.arange(10))
        tr2 = Trace(data=np.arange(5, 7))
        tr2.stats.starttime = tr1.stats.starttime + 5
        st = Stream([tr1, tr2])
        st.merge()
        self.assertEqual(len(st), 1)
        self.assertTrue(isinstance(st[0].data, np.ndarray))
        np.testing.assert_array_equal(st[0].data, np.arange(10))
        #
        #4 - contained overlap with differing data
        # Trace 1: 0000000000
        # Trace 2:      11
        # 1 + 2  : 00000--000
        tr1 = Trace(data=np.zeros(10))
        tr2 = Trace(data=np.ones(2))
        tr2.stats.starttime = tr1.stats.starttime + 5
        st = Stream([tr1, tr2])
        st.merge()
        self.assertEqual(len(st), 1)
        self.assertTrue(isinstance(st[0].data, np.ma.masked_array))
        self.assertEqual(st[0].data.tolist(),
                         [0, 0, 0, 0, 0, None, None, 0, 0, 0])

    def test_tabCompletionTrace(self):
        """
        Test tab completion of Trace object.
        """
        tr = Trace()
        self.assertTrue('sampling_rate' in dir(tr.stats))
        self.assertTrue('npts' in dir(tr.stats))
        self.assertTrue('station' in dir(tr.stats))
        self.assertTrue('starttime' in dir(tr.stats))
        self.assertTrue('endtime' in dir(tr.stats))
        self.assertTrue('calib' in dir(tr.stats))
        self.assertTrue('delta' in dir(tr.stats))

    def test_bugfixMergeDropTraceIfAlreadyContained(self):
        """
        Trace data already existing in another trace and ending on the same
        endtime was not correctly merged until now.
        """
        trace1 = Trace(data=np.empty(10))
        trace2 = Trace(data=np.empty(2))
        trace2.stats.starttime = trace1.stats.endtime - trace1.stats.delta
        st = Stream([trace1, trace2])
        st.merge()

    def test_bugfixMergeMultipleTraces1(self):
        """
        Bugfix for merging multiple traces in a row.
        """
        # create a stream with multiple traces overlapping
        trace1 = Trace(data=np.empty(10))
        traces = [trace1]
        for _ in xrange(10):
            trace = Trace(data=np.empty(10))
            trace.stats.starttime = \
                traces[-1].stats.endtime - trace1.stats.delta
            traces.append(trace)
        st = Stream(traces)
        st.merge()

    def test_bugfixMergeMultipleTraces2(self):
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

    def test_mergeWithSmallSamplingRate(self):
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
        self.assertEquals(len(st), 1)
        self.assertEquals(st[0].stats.delta, 60.0)
        self.assertEquals(st[0].stats.starttime, trace1.stats.starttime)
        # endtime of last trace
        endtime = trace1.stats.starttime + \
                  (4 * 1440 - 1) * trace1.stats.delta
        self.assertEquals(st[0].stats.endtime, endtime)

    def test_mergeOverlapsMethod1(self):
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
        trace1 = Trace(data=np.zeros(8, dtype='int32'))
        trace2 = Trace(data=6 * np.ones(8, dtype='int32'))
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
        trace1 = Trace(data=np.zeros(8, dtype='int32'))
        trace2 = Trace(data=5 * np.ones(8, dtype='int32'))
        trace2.stats.starttime += 4
        st = Stream([trace1, trace2])
        st.merge(method=1, interpolation_samples=(-1))
        np.testing.assert_array_equal(st[0].data,
                          np.array([0] * 4 + [1] + [2] + [3] + [4] + [5] * 4))
        # Interpolate all samples (``interpolation_samples=5``)::
        # Given number of samples is bigger than the actual overlap - should
        # interpolate all samples
        #
        #     Trace 1: 00000000
        #     Trace 2:     55555555
        #     1 + 2  : 000012345555
        trace1 = Trace(data=np.zeros(8, dtype='int32'))
        trace2 = Trace(data=5 * np.ones(8, dtype='int32'))
        trace2.stats.starttime += 4
        st = Stream([trace1, trace2])
        st.merge(method=1, interpolation_samples=5)
        np.testing.assert_array_equal(st[0].data,
                          np.array([0] * 4 + [1] + [2] + [3] + [4] + [5] * 4))

    def test_trimRemovingEmptyTraces(self):
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
        self.assertEqual(len(stream), 1)
        np.testing.assert_array_equal(np.ones(10), stream[0].data)
        self.assertEqual(stream[0].stats.starttime, UTCDateTime(1000))
        self.assertEqual(stream[0].stats.npts, 10)

    def test_trimWithSmallSamplingRate(self):
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
        self.assertEquals(len(st), 4)
        self.assertEquals(st[0].stats.delta, 60.0)
        self.assertEquals(st[0].stats.starttime, trace1.stats.starttime)
        self.assertEquals(st[3].stats.endtime, trace4.stats.endtime)

    def test_writingMaskedArrays(self):
        """
        Writing a masked array should raise an exception.
        """
        # np.ma.masked_array with masked values
        tr = Trace(data=np.ma.masked_all(10))
        st = Stream([tr])
        self.assertRaises(NotImplementedError, st.write, 'filename', 'MSEED')
        # np.ma.masked_array without masked values
        tr = Trace(data=np.ma.ones(10))
        st = Stream([tr])
        self.assertRaises(NotImplementedError, st.write, 'filename', 'MSEED')

    def test_pickle(self):
        """
        Testing pickling of Stream objects..
        """
        tr = Trace(data=np.random.randn(1441))
        st = Stream([tr])
        st.verify()
        # protocol 0 (ASCII)
        temp = pickle.dumps(st, protocol=0)
        st2 = pickle.loads(temp)
        np.testing.assert_array_equal(st[0].data, st2[0].data)
        self.assertEquals(st[0].stats, st2[0].stats)
        # protocol 1 (old binary)
        temp = pickle.dumps(st, protocol=1)
        st2 = pickle.loads(temp)
        np.testing.assert_array_equal(st[0].data, st2[0].data)
        self.assertEquals(st[0].stats, st2[0].stats)
        # protocol 2 (new binary)
        temp = pickle.dumps(st, protocol=2)
        st2 = pickle.loads(temp)
        np.testing.assert_array_equal(st[0].data, st2[0].data)
        self.assertEquals(st[0].stats, st2[0].stats)

    def test_getGaps2(self):
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
        gaps = st.getGaps()
        self.assertEquals(len(gaps), 0)
        # different sampling rate should result in a gap
        tr3.stats.sampling_rate = 50.0
        gaps = st.getGaps()
        self.assertEquals(len(gaps), 1)
        # but different ids will be skipped (if only one trace)
        tr3.stats.station = 'MANZ'
        gaps = st.getGaps()
        self.assertEquals(len(gaps), 0)
        # multiple traces with same id will be handled again
        tr2.stats.station = 'MANZ'
        gaps = st.getGaps()
        self.assertEquals(len(gaps), 1)

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
        stA = Stream()
        # tests that should raise a NotImplementedError (i.e. <=, <, >=, >)
        self.assertRaises(NotImplementedError, st1.__lt__, st1)
        self.assertRaises(NotImplementedError, st1.__le__, st1)
        self.assertRaises(NotImplementedError, st1.__gt__, st1)
        self.assertRaises(NotImplementedError, st1.__ge__, st1)
        self.assertRaises(NotImplementedError, st1.__lt__, st2)
        self.assertRaises(NotImplementedError, st1.__le__, st2)
        self.assertRaises(NotImplementedError, st1.__gt__, st2)
        self.assertRaises(NotImplementedError, st1.__ge__, st2)
        # normal tests
        for st in [st1]:
            self.assertEqual(st0 == st, True)
            self.assertEqual(st0 != st, False)
        for st in [st2, st3, st4, st5, st6, st7, st8, st9, stA]:
            self.assertEqual(st0 == st, False)
            self.assertEqual(st0 != st, True)
        for st in [st0]:
            self.assertEqual(st1 == st, True)
            self.assertEqual(st1 != st, False)
        for st in [st2, st3, st4, st5, st6, st7, st8, st9, stA]:
            self.assertEqual(st1 == st, False)
            self.assertEqual(st1 != st, True)
        for st in [st0, st1, st3, st4, st5, st6, st7, st8, st9, stA]:
            self.assertEqual(st2 == st, False)
            self.assertEqual(st2 != st, True)
        for st in [st0, st1, st2, st4, st5, st6, st7, st8, st9, stA]:
            self.assertEqual(st3 == st, False)
            self.assertEqual(st3 != st, True)
        for st in [st0, st1, st2, st3, st5, st6, st7, st8, st9, stA]:
            self.assertEqual(st4 == st, False)
            self.assertEqual(st4 != st, True)
        for st in [st0, st1, st2, st3, st4, st6, st7, st8, st9, stA]:
            self.assertEqual(st5 == st, False)
            self.assertEqual(st5 != st, True)
        for st in [st7, st8]:
            self.assertEqual(st6 == st, True)
            self.assertEqual(st6 != st, False)
        for st in [st0, st1, st2, st3, st4, st5, st9, stA]:
            self.assertEqual(st6 == st, False)
            self.assertEqual(st6 != st, True)
        for st in [st6, st8]:
            self.assertEqual(st7 == st, True)
            self.assertEqual(st7 != st, False)
        for st in [st0, st1, st2, st3, st4, st5, st9, stA]:
            self.assertEqual(st7 == st, False)
            self.assertEqual(st7 != st, True)
        for st in [st6, st7]:
            self.assertEqual(st8 == st, True)
            self.assertEqual(st8 != st, False)
        for st in [st0, st1, st2, st3, st4, st5, st9, stA]:
            self.assertEqual(st8 == st, False)
            self.assertEqual(st8 != st, True)
        for st in [stA]:
            self.assertEqual(st9 == st, True)
            self.assertEqual(st9 != st, False)
        for st in [st0, st1, st2, st3, st4, st5, st6, st7, st8]:
            self.assertEqual(st9 == st, False)
            self.assertEqual(st9 != st, True)
        for st in [st9]:
            self.assertEqual(stA == st, True)
            self.assertEqual(stA != st, False)
        for st in [st0, st1, st2, st3, st4, st5, st6, st7, st8]:
            self.assertEqual(stA == st, False)
            self.assertEqual(stA != st, True)
        # some weirder tests against non-Stream objects
        for object in [0, 1, 0.0, 1.0, "", "test", True, False, [], [tr0],
                       set(), set(tr0), {}, {"test": "test"}, Trace(), None]:
            self.assertEqual(st0 == object, False)
            self.assertEqual(st0 != object, True)

    def test_trimNearestSample(self):
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
        self.assertEqual(st[0].stats.starttime.timestamp, 1.0)
        self.assertEqual(st[1].stats.starttime.timestamp, 1.4)
        # ENDTIME
        # check that trimming first selects the next best sample, and only
        # then selects the following ones
        #    |    |    |  E |
        #      |    |    |    |
        st.trim(starttime=None, endtime=UTCDateTime(2.6))
        self.assertEqual(st[0].stats.endtime.timestamp, 3.0)
        self.assertEqual(st[1].stats.endtime.timestamp, 3.4)

    def test_trimConsistentStartEndtimeNearestSample(self):
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
        for i in xrange(len(st)):
            self.assertEquals(3, st[i].stats.npts)
            self.assertEquals(st[i].stats.starttime.timestamp, start[i])
            self.assertEquals(st[i].stats.endtime.timestamp, end[i])

    def test_trimConsistentStartEndtimeNearestSamplePadded(self):
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
        for i in xrange(len(st)):
            self.assertEquals(22, st[i].stats.npts)
            self.assertEquals(st[i].stats.starttime.timestamp, start[i])
            self.assertEquals(st[i].stats.endtime.timestamp, end[i])

    def test_trimConsistentStartEndtime(self):
        """
        Test case for #127. It ensures that the sample start and entimes
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
        for i in xrange(len(st)):
            self.assertEquals(st[i].stats.npts, npts[i])
            self.assertEquals(st[i].stats.starttime.timestamp, start[i])
            self.assertEquals(st[i].stats.endtime.timestamp, end[i])

    def test_trimConsistentStartEndtimePad(self):
        """
        Test case for #127. It ensures that the sample start and entimes
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
        for i in xrange(len(st)):
            self.assertEquals(st[i].stats.npts, npts[i])
            self.assertEquals(st[i].stats.starttime.timestamp, start[i])
            self.assertEquals(st[i].stats.endtime.timestamp, end[i])

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
        self.assertEqual(result, expected)

    def test_cleanup(self):
        """
        Test case for merging traces in the stream with method=-1. This only
        should merge traces that are exactly the same or contained and exactly
        the same or directly adjacent.
        """
        tr1 = self.mseed_stream[0]
        start = tr1.stats.starttime
        end = tr1.stats.endtime
        dt = end - start
        delta = tr1.stats.delta

        # test traces that should be merged:
        ### contained traces with compatible data
        tr2 = tr1.slice(start, start + dt / 3)
        tr3 = tr1.copy()
        tr4 = tr1.slice(start + dt / 4, end - dt / 4)
        ### adjacent traces
        tr5 = tr1.copy()
        tr5.stats.starttime = end + delta
        tr6 = tr1.copy()
        tr6.stats.starttime = start - dt - delta
        ### create overlapping traces with compatible data
        trO1 = tr1.copy()
        trO1.trim(starttime=start + 2 * delta)
        trO1.data = np.concatenate([trO1.data, np.arange(5)])
        trO2 = tr1.copy()
        trO2.trim(endtime=end - 2 * delta)
        trO2.data = np.concatenate([np.arange(5), trO2.data])
        trO2.stats.starttime -= 5 * delta
        # test mergeable traces (contained ones)
        for trB in [tr2, tr3, tr4]:
            trA = tr1.copy()
            st = Stream([trA, trB])
            st._cleanup()
            self.assertTrue(st == Stream([tr1]))
            self.assertTrue(type(st[0].data) == np.ndarray)
        # test mergeable traces (adjacent ones)
        for trB in [tr5, tr6]:
            trA = tr1.copy()
            st = Stream([trA, trB])
            st._cleanup()
            self.assertTrue(len(st) == 1)
            self.assertTrue(type(st[0].data) == np.ndarray)
            st_result = Stream([tr1, trB])
            st_result.merge()
            self.assertTrue(st == st_result)
        # test mergeable traces (overlapping ones)
        for trB in [trO1, trO2]:
            trA = tr1.copy()
            st = Stream([trA, trB])
            st._cleanup()
            self.assertTrue(len(st) == 1)
            self.assertTrue(type(st[0].data) == np.ndarray)
            st_result = Stream([tr1, trB])
            st_result.merge()
            self.assertTrue(st == st_result)

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
        for trB in [tr7, tr8, tr9, tr10, tr11, tr12, tr13]:
            trA = tr1.copy()
            st = Stream([trA, trB])
            st._cleanup()
            self.assertTrue(st == Stream([trA, trB]))


def suite():
    return unittest.makeSuite(StreamTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
