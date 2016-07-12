# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core.preview import create_preview, merge_previews, resample_preview


class UtilTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.preview.
    """

    def test_create_preview(self):
        """
        Test for creating preview.
        """
        # Wrong delta should raise.
        self.assertRaises(TypeError, create_preview,
                          Trace(data=np.arange(10)), 60.0)
        self.assertRaises(TypeError, create_preview,
                          Trace(data=np.arange(10)), 0)
        # 1
        trace = Trace(data=np.array([0] * 28 + [0, 1] * 30 + [-1, 1] * 29))
        trace.stats.starttime = UTCDateTime(32)
        preview = create_preview(trace, delta=60)
        self.assertEqual(preview.stats.starttime, UTCDateTime(60))
        self.assertEqual(preview.stats.endtime, UTCDateTime(120))
        self.assertEqual(preview.stats.delta, 60)
        np.testing.assert_array_equal(preview.data, np.array([1, 2]))
        # 2
        trace = Trace(data=np.arange(0, 30))
        preview = create_preview(trace, delta=60)
        self.assertEqual(preview.stats.starttime, UTCDateTime(0))
        self.assertEqual(preview.stats.endtime, UTCDateTime(0))
        self.assertEqual(preview.stats.delta, 60)
        np.testing.assert_array_equal(preview.data, np.array([29]))
        # 3
        trace = Trace(data=np.arange(0, 60))
        preview = create_preview(trace, delta=60)
        self.assertEqual(preview.stats.starttime, UTCDateTime(0))
        self.assertEqual(preview.stats.endtime, UTCDateTime(0))
        self.assertEqual(preview.stats.delta, 60)
        np.testing.assert_array_equal(preview.data, np.array([59]))
        # 4
        trace = Trace(data=np.arange(0, 90))
        preview = create_preview(trace, delta=60)
        self.assertEqual(preview.stats.starttime, UTCDateTime(0))
        self.assertEqual(preview.stats.endtime, UTCDateTime(60))
        self.assertEqual(preview.stats.delta, 60)
        np.testing.assert_array_equal(preview.data, np.array([59, 29]))

    def test_create_preview_with_masked_arrays(self):
        """
        Test for creating preview using masked arrays.
        """
        # 1 - masked arrays without masked values
        trace = Trace(data=np.ma.ones(600))
        preview = create_preview(trace, delta=60)
        # only masked values get replaced with an -1
        np.testing.assert_array_equal(preview.data, np.array(10 * [0]))
        # 2 - masked arrays with masked values
        trace = Trace(data=np.ma.ones(600))
        trace.data.mask = [False] * 600
        trace.data.mask[200:400] = True
        preview = create_preview(trace, delta=60)
        # masked values get replaced with an -1
        np.testing.assert_array_equal(preview.data,
                                      np.array(4 * [0] + 2 * [-1] + 4 * [0]))

    def test_merge_previews(self):
        """
        Tests the merging of Previews.
        """
        # Merging non-preview traces in one Stream object should raise.
        st = Stream(traces=[Trace(data=np.empty(2)),
                            Trace(data=np.empty(2))])
        self.assertRaises(Exception, merge_previews, st)
        # Merging empty traces should return an new empty Stream object.
        st = Stream()
        stream_id = id(st)
        st2 = merge_previews(st)
        self.assertNotEqual(stream_id, id(st2))
        self.assertEqual(len(st.traces), 0)
        # Different sampling rates in one Stream object causes problems.
        tr1 = Trace(data=np.empty(10))
        tr1.stats.preview = True
        tr1.stats.sampling_rate = 100
        tr2 = Trace(data=np.empty(10))
        tr2.stats.preview = True
        st = Stream(traces=[tr1, tr2])
        self.assertRaises(Exception, merge_previews, st)
        # Different data types should raise.
        tr1 = Trace(data=np.empty(10, dtype=np.int32))
        tr1.stats.preview = True
        tr2 = Trace(data=np.empty(10, dtype=np.float64))
        tr2.stats.preview = True
        st = Stream(traces=[tr1, tr2])
        self.assertRaises(Exception, merge_previews, st)
        # Now some real tests.
        # 1
        tr1 = Trace(data=np.array([1, 2] * 100))
        tr1.stats.preview = True
        tr1.stats.starttime = UTCDateTime(500)
        tr2 = Trace(data=np.array([3, 1] * 100))
        tr2.stats.preview = True
        tr2.stats.starttime = UTCDateTime(500)
        st = Stream(traces=[tr1, tr2])
        st2 = merge_previews(st)
        self.assertEqual(len(st2.traces), 1)
        self.assertEqual(st2[0].stats.starttime, UTCDateTime(500))
        np.testing.assert_array_equal(st2[0].data, np.array([3, 2] * 100))
        # 2
        tr1 = Trace(data=np.array([1] * 10))
        tr1.stats.preview = True
        tr2 = Trace(data=np.array([2] * 9))
        tr2.stats.starttime = tr2.stats.starttime + 20
        tr2.stats.preview = True
        st = Stream(traces=[tr1, tr2])
        st2 = merge_previews(st)
        self.assertEqual(len(st2.traces), 1)
        self.assertEqual(st2[0].stats.starttime, tr1.stats.starttime)
        np.testing.assert_array_equal(st2[0].data,
                                      np.array([1] * 10 + [-1] * 10 + [2] * 9))

    def test_resample_previews(self):
        """
        Test for resampling preview.
        """
        # Trying to resample non-preview Traces should raise.
        tr = Trace(data=np.empty(100))
        self.assertRaises(Exception, resample_preview, tr, 5)
        # Currently only downsampling is supported.
        tr = Trace(data=np.empty(20))
        tr.stats.preview = True
        self.assertRaises(NotImplementedError, resample_preview, tr, 100)
        # Fast method.
        tr = Trace(data=np.array([1, 2, 3, 4] * 53 + [-1, 0, 1, 2] * 53))
        endtime = tr.stats.endtime
        tr.stats.preview = True
        omitted_samples = resample_preview(tr, 100, method='fast')
        # Assert things for this easy case.
        self.assertEqual(tr.stats.endtime, endtime)
        self.assertEqual(tr.stats.npts, 100)
        self.assertEqual(omitted_samples, 24)
        # This shows the inaccuracy of the fast method.
        np.testing.assert_array_equal(tr.data, np.array([4] * 53 + [2] * 47))
        # Slow but accurate method.
        tr = Trace(data=np.array([1, 2, 3, 4] * 53 + [-1, 0, 1, 2] * 53))
        endtime = tr.stats.endtime
        tr.stats.preview = True
        omitted_samples = resample_preview(tr, 100, method='accurate')
        # Assert things for this easy case.
        self.assertEqual(tr.stats.endtime, endtime)
        self.assertEqual(tr.stats.npts, 100)
        self.assertEqual(omitted_samples, 0)
        # This method is much more accurate.
        np.testing.assert_array_equal(tr.data, np.array([4] * 50 + [2] * 50))

    def test_merge_previews_2(self):
        """
        Test case for issue #84.
        """
        # Note: explicitly creating np.ones instead of np.empty in order to
        # prevent NumPy warnings related to max function
        tr1 = Trace(data=np.ones(2880))
        tr1.stats.starttime = UTCDateTime("2010-01-01T00:00:00.670000Z")
        tr1.stats.delta = 30.0
        tr1.stats.preview = True
        tr1.verify()
        tr2 = Trace(data=np.ones(2881))
        tr2.stats.starttime = UTCDateTime("2010-01-01T23:59:30.670000Z")
        tr2.stats.delta = 30.0
        tr2.stats.preview = True
        tr2.verify()
        st1 = Stream([tr1, tr2])
        st1.verify()
        # merge
        st2 = merge_previews(st1)
        st2.verify()
        # check
        self.assertTrue(st2[0].stats.preview)
        self.assertEqual(st2[0].stats.starttime, tr1.stats.starttime)
        self.assertEqual(st2[0].stats.endtime, tr2.stats.endtime)
        self.assertEqual(st2[0].stats.npts, 5760)
        self.assertEqual(len(st2[0]), 5760)

    def test_create_preview_with_unrounded_sample_rate(self):
        """
        Test for creating preview.
        """
        tr = Trace(data=np.arange(4000))
        tr.stats.sampling_rate = 124.999992371
        tr.stats.starttime = UTCDateTime("1989-10-06T14:31:14.000000Z")
        create_preview(tr, delta=30)

    def test_create_preview_with_very_small_sample_rate(self):
        """
        Test for creating previews with samples per slice less than 1.
        """
        tr = Trace(data=np.arange(4000))
        # 1 - should raise
        tr.stats.sampling_rate = 0.1
        self.assertRaises(ValueError, create_preview, tr)
        # 2 - should work
        tr.stats.sampling_rate = 1
        create_preview(tr)


def suite():
    return unittest.makeSuite(UtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
