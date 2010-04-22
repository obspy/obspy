# -*- coding: utf-8 -*-

from obspy.core.preview import createPreview, mergePreviews
from obspy.core import Stream, Trace, UTCDateTime
import numpy as np
import unittest


class UtilTestCase(unittest.TestCase):
    """
    Test suite for obspy.db.util.
    """

    def test_createPreview(self):
        """
        Test for creating preview.
        """
        # Wrong delta should raise.
        self.assertRaises(TypeError, createPreview,
                          Trace(data = np.arange(10)), 60.0)
        self.assertRaises(TypeError, createPreview,
                          Trace(data = np.arange(10)), 0)
        #1
        trace = Trace(data=np.array([0] * 28 + [0, 1] * 30 + [-1, 1] * 29))
        trace.stats.starttime = UTCDateTime(32)
        preview = createPreview(trace, delta=60)
        self.assertEqual(preview.stats.starttime, UTCDateTime(60))
        self.assertEqual(preview.stats.endtime, UTCDateTime(120))
        self.assertEqual(preview.stats.delta, 60)
        np.testing.assert_array_equal(preview.data, np.array([1, 2]))
        #2
        trace = Trace(data=np.arange(0, 30))
        preview = createPreview(trace, delta=60)
        self.assertEqual(preview.stats.starttime, UTCDateTime(0))
        self.assertEqual(preview.stats.endtime, UTCDateTime(0))
        self.assertEqual(preview.stats.delta, 60)
        np.testing.assert_array_equal(preview.data, np.array([29]))
        #2
        trace = Trace(data=np.arange(0, 60))
        preview = createPreview(trace, delta=60)
        self.assertEqual(preview.stats.starttime, UTCDateTime(0))
        self.assertEqual(preview.stats.endtime, UTCDateTime(0))
        self.assertEqual(preview.stats.delta, 60)
        np.testing.assert_array_equal(preview.data, np.array([59]))
        #3
        trace = Trace(data=np.arange(0, 90))
        preview = createPreview(trace, delta=60)
        self.assertEqual(preview.stats.starttime, UTCDateTime(0))
        self.assertEqual(preview.stats.endtime, UTCDateTime(60))
        self.assertEqual(preview.stats.delta, 60)
        np.testing.assert_array_equal(preview.data, np.array([59, 29]))

    def test_mergePreviews(self):
        """
        Tests the merging of Previews.
        """
        # Merging non-preview traces in one Stream object should raise.
        st = Stream(traces = [Trace(data = np.empty(2)),
                              Trace(data = np.empty(2))])
        self.assertRaises(Exception, mergePreviews, st)
        # Merging totally Empty traces should return an new empty Stream
        # object.
        st = Stream()
        stream_id = id(st)
        st2 = mergePreviews(st)
        self.assertNotEqual(stream_id, id(st2))
        self.assertEqual(len(st.traces), 0)
        # Different sampling rates in one Stream object causes problems.
        tr1 = Trace(data = np.empty(10))
        tr1.stats.preview = True
        tr1.stats.sampling_rate = 100
        tr2 = Trace(data = np.empty(10))
        tr2.stats.preview = True
        st = Stream(traces = [tr1, tr2])
        self.assertRaises(Exception, mergePreviews, st)
        # Differnt dtypes should raise.
        tr1 = Trace(data = np.empty(10, dtype = 'int32'))
        tr1.stats.preview = True
        tr2 = Trace(data = np.empty(10, dtype = 'float64'))
        tr2.stats.preview = True
        st = Stream(traces = [tr1, tr2])
        self.assertRaises(Exception, mergePreviews, st)
        # Now some real tests.
        # 1
        tr1 = Trace(data = np.array([1,2] * 100))
        tr1.stats.preview = True
        tr2 = Trace(data = np.array([3,1] * 100))
        tr2.stats.preview = True
        st = Stream(traces = [tr1, tr2])
        st2 = mergePreviews(st)
        self.assertEqual(len(st2.traces), 1)
        np.testing.assert_array_equal(st2[0].data, np.array([3,2] * 100))
        # 2
        tr1 = Trace(data = np.array([1] * 10))
        tr1.stats.preview = True
        tr2 = Trace(data = np.array([2] * 10))
        tr2.stats.starttime = tr2.stats.starttime + 20
        tr2.stats.preview = True
        st = Stream(traces = [tr1, tr2])
        st2 = mergePreviews(st)
        self.assertEqual(len(st2.traces), 1)
        self.assertEqual(st2[0].stats.starttime, tr1.stats.starttime)
        np.testing.assert_array_equal(st2[0].data,
                np.array([1] * 10 + [-1] * 10 + [2] * 10))



def suite():
    return unittest.makeSuite(UtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
