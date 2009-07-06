# -*- coding: utf-8 -*-

from copy import deepcopy
from obspy.core.util import UTCDateTime
import inspect
import numpy as N
import obspy
import os
import unittest

class CoreTestCase(unittest.TestCase):
    """
    Tests the obspy.core.core functions and classes. Please be aware that the
    tests will only work with installed obspy.mseed and obspy.gse2 modules.
    """

    def setUp(self):
        # Directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        # Test files, this makes also shure that obspy.mseed and obspy.gse2
        # modules are installed.
        try:
            import obspy.mseed.tests
            path2 = os.path.dirname(inspect.getsourcefile(obspy.mseed.tests))
            self.mseed_file = os.path.join(path2, 'data', 'gaps.mseed')
            import obspy.gse2.tests
            path2 = os.path.dirname(inspect.getsourcefile(obspy.gse2.tests))
            self.gse2_file = os.path.join(path2, 'data', 'loc_RNON20040609200559.z')
        except ImportError:
            msg = 'obspy.mseed and obspy.gse2 modules are necessary to ' +\
                  'test the obspy.core.core methods and functions'
            raise ImportError(msg)

    def tearDown(self):
        pass
    
    def test_getitem(self):
        """
        Tests the getting of items of the Stream objects.
        """
        stream = obspy.read(self.mseed_file)
        self.assertEqual(stream[0], stream.traces[0])
        self.assertEqual(stream[-1], stream.traces[-1])
        self.assertEqual(stream[3], stream.traces[3])
        self.assertEqual(stream[0:], stream.traces[0:])
        self.assertEqual(stream[:2], stream.traces[:2])
        self.assertEqual(stream[:], stream.traces[:])
        
    def test_adding(self):
        """
        Tests the adding of two stream objects.
        """
        stream = obspy.read(self.mseed_file)
        self.assertEqual(4, len(stream))
        # Add the same stream object to itsself.
        stream = stream + stream
        self.assertEqual(8, len(stream))
        # This will create a copy of all Traces and thus the objects should not
        # be identical but the Traces attributes should be identical.
        for _i in range(4):
            self.assertNotEqual(stream[_i], stream[_i+4])
            self.assertEqual(stream[_i].stats, stream[_i+4].stats)
            N.testing.assert_array_equal(stream[_i].data, stream[_i+4].data)
        # Now add another stream to it.
        other_stream = obspy.read(self.gse2_file)
        self.assertEqual(1, len(other_stream))
        new_stream = stream + other_stream
        self.assertEqual(9, len(new_stream))
        # The traces of all streams are copied.
        for _i in range(8):
            self.assertNotEqual(new_stream[_i], stream[_i])
            self.assertEqual(new_stream[_i].stats, stream[_i].stats)
            N.testing.assert_array_equal(new_stream[_i].data, stream[_i].data)
        # Also test for the newly added stream.
        self.assertNotEqual(new_stream[8], other_stream[0])
        self.assertEqual(new_stream[8].stats, other_stream[0].stats)
        N.testing.assert_array_equal(new_stream[8].data, other_stream[0].data)
        
    def test_iadding(self):
        """
        Tests the __iadd__ method of the Stream objects.
        """
        stream = obspy.read(self.mseed_file)
        self.assertEqual(4, len(stream))
        other_stream = obspy.read(self.gse2_file)
        self.assertEqual(1, len(other_stream))
        # Add the other stream to the stream.
        stream += other_stream
        # This will leave the Traces of the new stream and create a deepcopy of
        # the other Stream's Traces
        self.assertEqual(5, len(stream))
        self.assertNotEqual(other_stream[0], stream[-1])
        self.assertEqual(other_stream[0].stats, stream[-1].stats)
        N.testing.assert_array_equal(other_stream[0].data, stream[-1].data)

    def test_append(self):
        """
        Tests the appending method of the Stream objects.
        """
        stream = obspy.read(self.mseed_file)
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
        N.testing.assert_array_equal(stream[0].data, stream[-1].data)
        # Append the same again but pass by reference.
        stream.append(stream[0], reference = True)
        self.assertEqual(len(stream), 6)
        # Now the two objects should be identical.
        self.assertEqual(stream[0], stream[-1])
        # Using append with a list of Traces, or int, or ... should fail.
        self.assertRaises(TypeError, stream.append, stream[:])
        self.assertRaises(TypeError, stream.append, 1)
        self.assertRaises(TypeError, stream.append, stream[0].data)
        
    def test_countAndLen(self):
        """
        Tests the count method and __len__ attribut of the Stream objects.
        """
        stream = obspy.read(self.mseed_file)
        self.assertEqual(4, len(stream))
        self.assertEqual(4, stream.count())
        self.assertEqual(stream.count(), len(stream))
        
    def test_extend(self):
        """
        Tests the extending method of the Stream objects.
        """
        stream = obspy.read(self.mseed_file)
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
        N.testing.assert_array_equal(stream[0].data, stream[-2].data)
        self.assertEqual(stream[1].stats, stream[-1].stats)
        N.testing.assert_array_equal(stream[1].data, stream[-1].data)
        # Extend with the same again but pass by reference.
        stream.extend(stream[0:2], reference = True)
        self.assertEqual(len(stream), 8)
        # Now the two objects should be identical.
        self.assertEqual(stream[0], stream[-2])
        self.assertEqual(stream[1], stream[-1])
        # Using extend with a single Traces, or a wrong list, or ...
        # should fail.
        self.assertRaises(TypeError, stream.extend, stream[0])
        self.assertRaises(TypeError, stream.extend, 1)
        self.assertRaises(TypeError, stream.extend, stream[0:2].append(1))
    
    def test_insert(self):
        """
        Tests the insert Method of the Stream objects.
        """
        stream = obspy.read(self.mseed_file)
        self.assertEqual(4, len(stream))
        # Insert the last Trace before the second trace.
        stream.insert(1, stream[-1])
        self.assertEqual(len(stream), 5)
        # This is supposed to make a deepcopy of the Trace and thus the two
        # Traces are not identical.
        self.assertNotEqual(stream[1], stream[-1])
        # But the attributes and data values should be identical.
        self.assertEqual(stream[1].stats, stream[-1].stats)
        N.testing.assert_array_equal(stream[1].data, stream[-1].data)
        # Do the same again but pass by refernce.
        stream.insert(1, stream[-1], reference = True)
        self.assertEqual(len(stream), 6)
        # Now the two Traces should ne identical
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
        N.testing.assert_array_equal(stream[1].data, stream[-2].data)
        self.assertEqual(stream[2].stats, stream[-1].stats)
        N.testing.assert_array_equal(stream[2].data, stream[-1].data)
        # Do the same again but pass by refernce.
        stream.insert(1, stream[-2:], reference = True)
        self.assertEqual(len(stream), 10)
        # Now the two Traces should ne identical
        self.assertEqual(stream[1], stream[-2])
        self.assertEqual(stream[2], stream[-1])
        # Using insert without a single Traces or a list of Traces should fail.
        self.assertRaises(TypeError, stream.insert, 1, 1)
        self.assertRaises(TypeError, stream.insert, stream[0], stream[0])
        self.assertRaises(TypeError, stream.insert, 1, stream[0:2].append(1))
        
    def test_getGaps(self):
        """
        Tests the getGaps method of the Stream objects. It is compared directly
        to the obspy.mseed method getGapsList which is assumed to be correct.
        """
        stream = obspy.read(self.mseed_file)
        gap_list = stream.getGaps()
        # Gapslist created with obspy.mseed
        mseed_gap_list = [('BW', 'BGLD', '', 'EHE', UTCDateTime(2008, 1, 1, 0,\
                          0, 1, 970000), UTCDateTime(2008, 1, 1, 0, 0, 4,\
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
                                   places = 3)
            self.assertAlmostEqual(mseed_gap_list[_i][7], gap_list[_i][7],
                                   places = 3)
            
    def test_pop(self):
        """
        Test the pop method of the Stream objects.
        """
        stream = obspy.read(self.mseed_file)
        # Make a copy of the Traces.
        traces = deepcopy(stream[:])
        # Remove and return the last Trace.
        temp_trace = stream.pop()
        self.assertEqual(3, len(stream))
        # Assert attributes. The objects itsself are not identical.
        self.assertEqual(temp_trace.stats, traces[-1].stats)
        N.testing.assert_array_equal(temp_trace.data, traces[-1].data)
        # Remove the last copied Trace.
        traces.pop()
        # Remove and return the second Trace.
        temp_trace = stream.pop(1)
        # Assert attributes. The objects itsself are not identical.
        self.assertEqual(temp_trace.stats, traces[1].stats)
        N.testing.assert_array_equal(temp_trace.data, traces[1].data)
        # Remove the second copied Trace.
        traces.pop(1)
        # Compare all remaining Traces.
        self.assertEqual(2, len(stream))
        self.assertEqual(2, len(traces))
        for _i in range(len(traces)):
            self.assertEqual(traces[_i].stats, stream[_i].stats)
            N.testing.assert_array_equal(traces[_i].data, stream[_i].data)
            
    def test_remove(self):
        """
        Tests the remove method of the Stream objects.
        """
        stream = obspy.read(self.mseed_file)
        # Make a copy of the Traces.
        traces = deepcopy(stream[:])
        # Use the remove method of the Stream object and of the list of Traces.
        stream.remove(1)
        del(traces)[1]
        stream.remove(-1)
        del(traces)[-1]
        # Compare all remaining Traces.
        self.assertEqual(2, len(stream))
        self.assertEqual(2, len(traces))
        for _i in range(len(traces)):
            self.assertEqual(traces[_i].stats, stream[_i].stats)
            N.testing.assert_array_equal(traces[_i].data, stream[_i].data)
        
    def test_reverse(self):
        """
        Tests the reverse method of the Stream objects.
        """
        stream = obspy.read(self.mseed_file)
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
            N.testing.assert_array_equal(traces[_i].data, stream[_i].data)
            
    def test_sort(self):
        """
        Tests the sorting of the Stream objects.
        """
        # Create new Stream
        stream = obspy.Stream()
        # Create a list of header dictionaries. The sampling rate serves as a
        # unique identifier for each Trace.
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
            new_trace = obspy.Trace(header = _i)
            stream.append(new_trace, reference = True)
        # Use normal sorting.
        stream.sort()
        self.assertEqual([i.stats.sampling_rate for i in stream.traces], 
                         [300.0, 500.0, 400.0, 200.0, 100.0])
        # Sort after sampling_rate.
        stream.sort(keys = ['sampling_rate'])
        self.assertEqual([i.stats.sampling_rate for i in stream.traces], 
                         [100.0, 200.0, 300.0, 400.0, 500.0])
        # Sort after channel and sampling rate.
        stream.sort(keys = ['channel', 'sampling_rate'])
        self.assertEqual([i.stats.sampling_rate for i in stream.traces], 
                         [200.0, 500.0, 300.0, 400.0, 100.0])
        # Sort after npts and channel and sampling_rate.
        stream.sort(keys = ['npts', 'channel', 'sampling_rate'])
        self.assertEqual([i.stats.sampling_rate for i in stream.traces], 
                         [500.0, 300.0, 200.0, 400.0, 100.0])
        # Sorting without a list or a wrong item string should fail.
        self.assertRaises(TypeError, stream.sort, keys = 1)
        self.assertRaises(TypeError, stream.sort, keys = 'samping_rate')
        self.assertRaises(TypeError, stream.sort, keys = ['npts', 'starttime',
                                                          'wrong_value'])
        
def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
