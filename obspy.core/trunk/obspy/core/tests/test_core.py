# -*- coding: utf-8 -*-

from copy import deepcopy
from obspy.mseed import libmseed
import os
import obspy
import inspect
import unittest


class CoreTestCase(unittest.TestCase):
    """
    """

    def setUp(self):
        #Directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.file = os.path.join(path, 'data', 'gaps.mseed')

    def tearDown(self):
        pass
    
    def test_getGapList(self):
        """
        Tests the getGaps method of the Stream objects. It is compared directly
        to the obspy.mseed method getGapsList which is assumed to be correct.
        """
        stream = obspy.read(self.file)
        gap_list = stream.getGaps()
        # Get Mini-SEED Gap List
        mseed = libmseed()
        mseed_gap_list = mseed.getGapList(self.file)
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

    def test_listLikeOperations(self):
        """
        This test tests the list like operations of the Stream object by doing
        the same operations by hand on a copy of the Traces and comparing the
        result.
        
        It also tests some error handling.
        """
        # Read file.
        stream = obspy.read(self.file)
        # Test the count method.
        self.assertEqual(stream.count(), 4)
        # Save all traces and their order for future comparision.
        traces = deepcopy(stream.traces)
        # Now append the last trace to the to the stream.
        stream.append(stream[-1])
        traces.append(deepcopy(stream[-1]))
        # Assert the result.
        self.assertEqual(stream.count(), 5)
        for _i in range(stream.count()):
            self.assertEqual(stream.traces[_i].stats, traces[_i].stats)
        # Delete the Trace with index 1 from the Stream.
        stream.remove(1)
        del(traces)[1]
        # Assert the result.
        self.assertEqual(stream.count(), 4)
        for _i in range(stream.count()):
            self.assertEqual(stream.traces[_i].stats, traces[_i].stats)
        # Pop the last item.
        self.assertEqual(stream.pop().stats, traces.pop().stats)
        # Assert the result.
        self.assertEqual(stream.count(), 3)
        for _i in range(stream.count()):
            self.assertEqual(stream.traces[_i].stats, traces[_i].stats)
        # Reversing.
        stream.reverse()
        traces.reverse()
        # Assert the result.
        self.assertEqual(stream.count(), 3)
        for _i in range(stream.count()):
            self.assertEqual(stream.traces[_i].stats, traces[_i].stats)
        # Sorting.
        stream.sort()
        # Sort the other list by just reversing.
        traces.reverse()
        # Assert the result.
        self.assertEqual(stream.count(), 3)
        for _i in range(stream.count()):
            self.assertEqual(stream.traces[_i].stats, traces[_i].stats)
        # Extending.
        stream.extend(traces)
        traces.extend(traces)
        # Assert the result.
        self.assertEqual(stream.count(), 6)
        for _i in range(stream.count()):
            self.assertEqual(stream.traces[_i].stats, traces[_i].stats)
        # Test exceptions.
        # Using append with a list should fail.
        self.assertRaises(TypeError, stream.append, traces)
        # Using extend without a list should fail.
        self.assertRaises(TypeError, stream.extend, traces[0])
        # Using extend with a wrong list should fail.
        self.assertRaises(TypeError, stream.extend, [traces[0],2])
        # Using insert with a wrong list should fail.
        self.assertRaises(TypeError, stream.insert, [traces[0],2])

def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
