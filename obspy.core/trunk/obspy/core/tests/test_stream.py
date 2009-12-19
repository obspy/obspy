# -*- coding: utf-8 -*-

from copy import deepcopy
from obspy.core import UTCDateTime, Stream, Trace
import numpy as np
import unittest


class StreamTestCase(unittest.TestCase):
    """
    Test suite for L{obspy.core.Trace}.
    
    Please be aware that the tests will only work with the modules obspy.mseed 
    and obspy.gse2.
    """

    def setUp(self):
        # Create dynamic test_files to avoid dependecies of obspy.core
        # on obspy.mseed and obspy.gse2.
        # set specific seed value such that random numbers are reproducable
        np.random.seed(815)
        header = {'network': 'BW', 'station': 'BGLD',
            'starttime': UTCDateTime(2007, 12, 31, 23, 59, 59, 915000),
            'npts': 412, 'sampling_rate': 200.0,
            'endtime': UTCDateTime(2008, 1, 1, 0, 0, 1, 970000),
            'channel': 'EHE'}
        trace1 = Trace(data=np.random.randint(0, 1000, 412),
                                                header=deepcopy(header))
        header['starttime'] = UTCDateTime(2008, 1, 1, 0, 0, 4, 35000)
        header['endtime'] = UTCDateTime(2008, 1, 1, 0, 0, 8, 150000)
        header['npts'] = 824
        trace2 = Trace(data=np.random.randint(0, 1000, 824),
                                                header=deepcopy(header))
        header['starttime'] = UTCDateTime(2008, 1, 1, 0, 0, 10, 215000)
        header['endtime'] = UTCDateTime(2008, 1, 1, 0, 0, 14, 330000)
        trace3 = Trace(data=np.random.randint(0, 1000, 824),
                                                header=deepcopy(header))
        header['starttime'] = UTCDateTime(2008, 1, 1, 0, 0, 18, 455000)
        header['endtime'] = UTCDateTime(2008, 1, 1, 0, 4, 31, 790000)
        header['npts'] = 50668
        trace4 = Trace(data=np.random.randint(0, 1000, 50668),
                                                header=deepcopy(header))
        self.mseed_stream = Stream(traces=[trace1, trace2, trace3, trace4])
        header = {'network': '', 'station': 'RNON ', 'location': '',
                  'starttime': UTCDateTime(2004, 6, 9, 20, 5, 59, 849998),
                  'sampling_rate': 200.0, 'npts': 12000,
                  'endtime': UTCDateTime(2004, 6, 9, 20, 6, 59, 844998),
                  'channel': '  Z'}
        trace = Trace(data=np.random.randint(0, 1000, 12000),
                                                    header=header)
        self.gse2_stream = Stream(traces=[trace])

    def tearDown(self):
        pass

    def test_getitem(self):
        """
        Tests the getting of items of the Stream objects.
        """
        stream = self.mseed_stream
        self.assertEqual(stream[0], stream.traces[0])
        self.assertEqual(stream[-1], stream.traces[-1])
        self.assertEqual(stream[3], stream.traces[3])

    def test_adding(self):
        """
        Tests the adding of two stream objects.
        """
        stream = self.mseed_stream
        self.assertEqual(4, len(stream))
        # Add the same stream object to itself.
        stream = stream + stream
        self.assertEqual(8, len(stream))
        # This will create a copy of all Traces and thus the objects should not
        # be identical but the Traces attributes should be identical.
        for _i in range(4):
            self.assertNotEqual(stream[_i], stream[_i + 4])
            self.assertEqual(stream[_i].stats, stream[_i + 4].stats)
            np.testing.assert_array_equal(stream[_i].data, stream[_i + 4].data)
        # Now add another stream to it.
        other_stream = self.gse2_stream
        self.assertEqual(1, len(other_stream))
        new_stream = stream + other_stream
        self.assertEqual(9, len(new_stream))
        # The traces of all streams are copied.
        for _i in range(8):
            self.assertNotEqual(new_stream[_i], stream[_i])
            self.assertEqual(new_stream[_i].stats, stream[_i].stats)
            np.testing.assert_array_equal(new_stream[_i].data, stream[_i].data)
        # Also test for the newly added stream.
        self.assertNotEqual(new_stream[8], other_stream[0])
        self.assertEqual(new_stream[8].stats, other_stream[0].stats)
        np.testing.assert_array_equal(new_stream[8].data, other_stream[0].data)

    def test_iadding(self):
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
        self.assertNotEqual(other_stream[0], stream[-1])
        self.assertEqual(other_stream[0].stats, stream[-1].stats)
        np.testing.assert_array_equal(other_stream[0].data, stream[-1].data)

    def test_append(self):
        """
        Tests the appending method of the Stream objects.
        """
        stream = self.mseed_stream
        # Check current count of traces
        self.assertEqual(len(stream), 4)
        # Append first traces to the Stream object.
        stream.append(stream[0])
        self.assertEqual(len(stream), 5)
        # This is supposed to make a deepcopy of the Trace and thus the two
        # Traces are not identical.
        self.assertNotEqual(stream[0], stream[-1])
        # But the attributes and data values should be identical.
        self.assertEqual(stream[0].stats, stream[-1].stats)
        np.testing.assert_array_equal(stream[0].data, stream[-1].data)
        # Append the same again but pass by reference.
        stream.append(stream[0], reference=True)
        self.assertEqual(len(stream), 6)
        # Now the two objects should be identical.
        self.assertEqual(stream[0], stream[-1])
        # Using append with a list of Traces, or int, or ... should fail.
        self.assertRaises(TypeError, stream.append, stream[:])
        self.assertRaises(TypeError, stream.append, 1)
        self.assertRaises(TypeError, stream.append, stream[0].data)

    def test_countAndLen(self):
        """
        Tests the count method and __len__ attribute of the Stream objects.
        """
        stream = self.mseed_stream
        self.assertEqual(4, len(stream))
        self.assertEqual(4, stream.count())
        self.assertEqual(stream.count(), len(stream))

    def test_extend(self):
        """
        Tests the extending method of the Stream objects.
        """
        stream = self.mseed_stream
        # Check current count of traces
        self.assertEqual(len(stream), 4)
        # Extend the Stream object with the first two traces.
        stream.extend(stream[0:2])
        self.assertEqual(len(stream), 6)
        # This is supposed to make a deepcopy of the Trace and thus the two
        # Traces are not identical.
        self.assertNotEqual(stream[0], stream[-2])
        self.assertNotEqual(stream[1], stream[-1])
        # But the attributes and data values should be identical.
        self.assertEqual(stream[0].stats, stream[-2].stats)
        np.testing.assert_array_equal(stream[0].data, stream[-2].data)
        self.assertEqual(stream[1].stats, stream[-1].stats)
        np.testing.assert_array_equal(stream[1].data, stream[-1].data)
        # Extend with the same again but pass by reference.
        stream.extend(stream[0:2], reference=True)
        self.assertEqual(len(stream), 8)
        # Now the two objects should be identical.
        self.assertEqual(stream[0], stream[-2])
        self.assertEqual(stream[1], stream[-1])
        # Using extend with a single Traces, or a wrong list, or ...
        # should fail.
        self.assertRaises(TypeError, stream.extend, stream[0])
        self.assertRaises(TypeError, stream.extend, 1)
        self.assertRaises(TypeError, stream.extend, [stream[0], 1])

    def test_insert(self):
        """
        Tests the insert Method of the Stream objects.
        """
        stream = self.mseed_stream
        self.assertEqual(4, len(stream))
        # Insert the last Trace before the second trace.
        stream.insert(1, stream[-1])
        self.assertEqual(len(stream), 5)
        # This is supposed to make a deepcopy of the Trace and thus the two
        # Traces are not identical.
        self.assertNotEqual(stream[1], stream[-1])
        # But the attributes and data values should be identical.
        self.assertEqual(stream[1].stats, stream[-1].stats)
        np.testing.assert_array_equal(stream[1].data, stream[-1].data)
        # Do the same again but pass by reference.
        stream.insert(1, stream[-1], reference=True)
        self.assertEqual(len(stream), 6)
        # Now the two Traces should be identical
        self.assertEqual(stream[1], stream[-1])
        # Do the same with a list of traces this time.
        # Insert the last two Trace before the second trace.
        stream.insert(1, stream[-2:])
        self.assertEqual(len(stream), 8)
        # This is supposed to make a deepcopy of the Trace and thus the two
        # Traces are not identical.
        self.assertNotEqual(stream[1], stream[-2])
        self.assertNotEqual(stream[2], stream[-1])
        # But the attributes and data values should be identical.
        self.assertEqual(stream[1].stats, stream[-2].stats)
        np.testing.assert_array_equal(stream[1].data, stream[-2].data)
        self.assertEqual(stream[2].stats, stream[-1].stats)
        np.testing.assert_array_equal(stream[2].data, stream[-1].data)
        # Do the same again but pass by reference.
        stream.insert(1, stream[-2:], reference=True)
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
        Tests the getGaps method of the Stream objects. It is compared directly
        to the obspy.mseed method getGapsList which is assumed to be correct.
        """
        stream = self.mseed_stream
        gap_list = stream.getGaps()
        # Gaps list created with obspy.mseed
        mseed_gap_list = [('BW', 'BGLD', '', 'EHE', UTCDateTime(2008, 1, 1, 0, \
                          0, 1, 970000), UTCDateTime(2008, 1, 1, 0, 0, 4, \
                          35000), 2.0649999999999999, 412.0), ('BW', 'BGLD',
                          '', 'EHE', UTCDateTime(2008, 1, 1, 0, 0, 8, 150000),
                          UTCDateTime(2008, 1, 1, 0, 0, 10, 215000),
                          2.0649999999999999, 412.0), ('BW', 'BGLD', '', 'EHE',
                          UTCDateTime(2008, 1, 1, 0, 0, 14, 330000),
                          UTCDateTime(2008, 1, 1, 0, 0, 18, 455000), 4.125,
                          824.0)]
        # Assert the number of gaps.
        self.assertEqual(len(mseed_gap_list), len(gap_list))
        for _i in range(len(mseed_gap_list)):
            # Compare the string values directly.
            for _j in range(6):
                self.assertEqual(gap_list[_i][_j], mseed_gap_list[_i][_j])
            # The small differences are probably due to rounding errors.
            self.assertAlmostEqual(mseed_gap_list[_i][6], gap_list[_i][6],
                                   places=3)
            self.assertAlmostEqual(mseed_gap_list[_i][7], gap_list[_i][7],
                                   places=3)

    def test_pop(self):
        """
        Test the pop method of the Stream objects.
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
        for _i in range(len(traces)):
            self.assertEqual(traces[_i].stats, stream[_i].stats)
            np.testing.assert_array_equal(traces[_i].data, stream[_i].data)

    def test_getslice(self):
        """
        Tests the slicing of Stream objects.
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

    def test_pop2(self):
        """
        Test the pop method of the Stream objects.
        """
        trace = Trace(data=range(0, 1000))
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
        Tests the remove method of the Stream objects.
        """
        stream = self.mseed_stream
        # Make a copy of the Traces.
        stream2 = deepcopy(stream[:])
        # Use the remove method of the Stream object and of the list of Traces.
        stream.remove(1)
        del(stream2.traces[1])
        stream.remove(-1)
        del(stream2.traces[-1])
        # Compare all remaining Traces.
        self.assertEqual(2, len(stream))
        self.assertEqual(2, len(stream2))
        for _i in range(len(stream2)):
            self.assertEqual(stream2[_i].stats, stream[_i].stats)
            np.testing.assert_array_equal(stream2[_i].data, stream[_i].data)

    def test_reverse(self):
        """
        Tests the reverse method of the Stream objects.
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
        for _i in range(len(traces)):
            self.assertEqual(traces[_i].stats, stream[_i].stats)
            np.testing.assert_array_equal(traces[_i].data, stream[_i].data)

    def test_sort(self):
        """
        Tests the sorting of the Stream objects.
        """
        # Create new Stream
        stream = Stream()
        # Create a list of header dictionaries. The sampling rate serves as a
        # unique identifier for each Trace.
        headers = [{'starttime' : UTCDateTime(1990, 1, 1), 'endtime' : \
                UTCDateTime(1990, 1, 2), 'network' : 'AAA', 'station' : 'ZZZ',
                'channel' : 'XXX', 'sampling_rate' : 100.0},
                {'starttime' : UTCDateTime(1990, 1, 1), 'endtime' : \
                UTCDateTime(1990, 1, 3), 'network' : 'AAA', 'station' : 'YYY',
                'channel' : 'CCC', 'sampling_rate' : 200.0},
                {'starttime' : UTCDateTime(2000, 1, 1), 'endtime' : \
                UTCDateTime(2001, 1, 2), 'network' : 'AAA', 'station' : 'EEE',
                'channel' : 'GGG', 'sampling_rate' : 300.0},
                {'starttime' : UTCDateTime(1989, 1, 1), 'endtime' : \
                UTCDateTime(2010, 1, 2), 'network' : 'AAA', 'station' : 'XXX',
                'channel' : 'GGG', 'sampling_rate' : 400.0},
                {'starttime' : UTCDateTime(2010, 1, 1), 'endtime' : \
                UTCDateTime(2011, 1, 2), 'network' : 'AAA', 'station' : 'XXX',
                'channel' : 'FFF', 'sampling_rate' : 500.0}]
        # Create a Trace object of it and append it to the Stream object.
        for _i in headers:
            new_trace = Trace(header=_i)
            stream.append(new_trace, reference=True)
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
        # Sort after npts and channel and sampling_rate.
        stream.sort(keys=['endtime', 'channel', 'sampling_rate'])
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
        headers = [{'starttime' : UTCDateTime(1990, 1, 1), 'endtime' : \
                UTCDateTime(1990, 1, 2), 'network' : 'AAA', 'station' : 'ZZZ',
                'channel' : 'XXX', 'npts' : 10000, 'sampling_rate' : 100.0},
                {'starttime' : UTCDateTime(1990, 1, 1), 'endtime' : \
                UTCDateTime(1990, 1, 3), 'network' : 'AAA', 'station' : 'YYY',
                'channel' : 'CCC', 'npts' : 10000, 'sampling_rate' : 200.0},
                {'starttime' : UTCDateTime(2000, 1, 1), 'endtime' : \
                UTCDateTime(2001, 1, 2), 'network' : 'AAA', 'station' : 'EEE',
                'channel' : 'GGG', 'npts' : 1000, 'sampling_rate' : 300.0},
                {'starttime' : UTCDateTime(1989, 1, 1), 'endtime' : \
                UTCDateTime(2010, 1, 2), 'network' : 'AAA', 'station' : 'XXX',
                'channel' : 'GGG', 'npts' : 10000, 'sampling_rate' : 400.0},
                {'starttime' : UTCDateTime(2010, 1, 1), 'endtime' : \
                UTCDateTime(2011, 1, 2), 'network' : 'AAA', 'station' : 'XXX',
                'channel' : 'FFF', 'npts' : 1000, 'sampling_rate' : 500.0}]
        # Create a Trace object of it and append it to the Stream object.
        for _i in headers:
            new_trace = Trace(header=_i)
            stream.append(new_trace, reference=True)
        stream.sort()
        a = [i.stats.sampling_rate for i in stream.traces]
        stream.sort()
        b = [i.stats.sampling_rate for i in stream.traces]
        # should be equal
        self.assertEqual(a, b)

    def test_merge_gaps(self):
        """
        Test the merge method of the Stream objects.
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
        for i in range(0, 4):
            self.assertEquals(stream[i].stats.sampling_rate, 200)
            self.assertEquals(stream[i].getId(), 'BW.BGLD..EHE')
        stream._verify()
        # merge it
        stream.merge()
        stream._verify()
        self.assertEquals(len(stream), 1)
        self.assertEquals(len(stream[0]), stream[0].data.size)
        self.assertEquals(stream[0].stats.starttime, start)
        self.assertEquals(stream[0].stats.endtime, end)
        self.assertEquals(stream[0].stats.sampling_rate, 200)
        self.assertEquals(stream[0].getId(), 'BW.BGLD..EHE')

    def test_merge_overlaps(self):
        """
        Test the merge method of the Stream objects.
        """
        ts = []
        npts = 1000
        k = 0
        for _i in xrange(2):
            ts.append(Trace())
            ts[-1].data = np.zeros(npts) + k
            ts[-1].stats.sampling_rate = 1
            ts[-1].stats.starttime = UTCDateTime("20080808") + k
            ts[-1].stats.npts = npts
            ts[-1].stats.endtime = ts[-1].stats.starttime + \
                    float(ts[-1].stats.npts - 1) / ts[-1].stats.sampling_rate
            k = 20
        stream = Stream(traces=ts)
        stream._verify()
        # merge it
        stream.merge()
        # build reference array
        ref = np.zeros(npts + k) + 10.0
        ref[:k] = 0.0
        ref[-k:] = 20.0
        np.testing.assert_array_equal(stream[0].data, ref)


    def test_mergeOverlapAndDiscardRedundantTrace(self):
        """
        If a second trace is completely contained within a first trace it will
        be discarded.
        """
        trace1 = Trace(data=np.zeros(1000))
        trace1.stats.sampling_rate = 1.0
        trace1.stats.starttime = UTCDateTime(0)
        trace1.stats.endtime = trace1.stats.starttime + 1000
        trace2 = Trace(data=np.zeros(10))
        trace2.stats.sampling_rate = 1.0
        trace2.data[:] = 1
        trace2.stats.starttime = trace1.stats.starttime + 10
        trace2.stats.endtime = trace2.stats.starttime + 10
        st = Stream(traces=[trace1, trace2])
        self.assertEqual(len(st.traces), 2)
        st.merge()
        self.assertEqual(len(st.traces), 1)
        np.testing.assert_array_equal(st[0].data, np.zeros(1000))


    def test_tabCompleteStats(self):
        """
        Test stats for tab completion
        """
        tr = self.mseed_stream[0]
        self.assertTrue('sampling_rate' in dir(tr.stats))
        self.assertTrue('npts' in dir(tr.stats))
        self.assertTrue('station' in dir(tr.stats))
        self.assertTrue('starttime' in dir(tr.stats))

def suite():
    return unittest.makeSuite(StreamTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
