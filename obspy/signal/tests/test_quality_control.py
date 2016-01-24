# -*- coding: utf-8 -*-
"""
The Quality Control test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import numpy as np

import obspy
from obspy.core.util.base import NamedTemporaryFile
from obspy.signal.quality_control import MSEEDMetadata


class QualityControlTestCase(unittest.TestCase):
    """
    Test cases for Quality Control.
    """
    def test_no_files_given(self):
        """
        Tests the raised exception if no file is given.
        """
        mseed_metadata = MSEEDMetadata()

        with self.assertRaises(ValueError) as e:
            mseed_metadata.populate_metadata(files=[])

        self.assertEqual(
            e.exception.args[0],
            "Nothing added - no data within the given temporal constraints "
            "found.")

    def test_gap_count(self):
        """
        Tests that gap counting works as expected.
        """
        # Create a file with 3 gaps.
        tr_1 = obspy.Trace(data=np.arange(10, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(0)})
        tr_2 = obspy.Trace(data=np.arange(10, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(100)})
        tr_3 = obspy.Trace(data=np.arange(10, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(200)})
        tr_4 = obspy.Trace(data=np.arange(10, dtype=np.int32),
                           header={"starttime": obspy.UTCDateTime(300)})
        st = obspy.Stream(traces=[tr_1, tr_2, tr_3, tr_4])

        with NamedTemporaryFile() as tf:
            st.write(tf.name, format="mseed")

            mseed_metadata = MSEEDMetadata()
            mseed_metadata.populate_metadata(files=[tf.name])
            self.assertEqual(mseed_metadata._ms_meta['num_gaps'], 3)

    def test_gaps_between_multiple_files(self):
        """
        Test gap counting between multiple files.
        """
        with NamedTemporaryFile() as tf1, NamedTemporaryFile() as tf2:
            # Two files, same ids but a gap in-between.
            obspy.Trace(data=np.arange(10, dtype=np.int32),
                        header={"starttime": obspy.UTCDateTime(0)}).write(
                tf1.name, format="mseed")
            obspy.Trace(data=np.arange(10, dtype=np.int32),
                        header={"starttime": obspy.UTCDateTime(100)}).write(
                    tf2.name, format="mseed")
            # Don't calculate statistics on the single segments.
            mseed_metadata = MSEEDMetadata()
            mseed_metadata.populate_metadata([tf1.name, tf2.name], c_seg=False)
            self.assertEqual(mseed_metadata._ms_meta['num_gaps'], 1)
            self.assertNotIn("c_segments", mseed_metadata._ms_meta)

    def test_file_with_no_timing_quality(self):
        """
        Tests timing quality extraction in files with no timing quality.
        """
        with NamedTemporaryFile() as tf1:
            obspy.Trace(data=np.arange(10, dtype=np.int32),
                        header={"starttime": obspy.UTCDateTime(0)}).write(
                    tf1.name, format="mseed")
            mseed_metadata = MSEEDMetadata()
            mseed_metadata.populate_metadata([tf1.name])
            self.assertEqual(mseed_metadata._ms_meta['timing_quality_max'],
                             None)
            self.assertEqual(mseed_metadata._ms_meta['timing_quality_min'],
                             None)
            self.assertEqual(mseed_metadata._ms_meta['timing_quality_mean'],
                             None)


def suite():
    return unittest.makeSuite(QualityControlTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
