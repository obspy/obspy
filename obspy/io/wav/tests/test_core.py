#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The audio wav.core test suite.
"""

import os
import unittest

import numpy as np

from obspy import Stream, Trace, read
from obspy.core.util import NamedTemporaryFile
from obspy.io.wav.core import WIDTH2DTYPE


class CoreTestCase(unittest.TestCase):
    """
    Test cases for audio WAV support
    """

    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        self.file = os.path.join(self.path, '3cssan.near.8.1.RNON.wav')

    def test_read_via_obspy(self):
        """
        Read files via obspy.core.Trace
        """
        testdata = np.array([64, 78, 99, 119, 123, 107,
                             72, 31, 2, 0, 30, 84, 141])
        tr = read(self.file)[0]
        self.assertEqual(tr.stats.npts, 2599)
        self.assertEqual(tr.stats['sampling_rate'], 7000)
        np.testing.assert_array_equal(tr.data[:13], testdata)
        tr2 = read(self.file, format='WAV')[0]
        self.assertEqual(tr2.stats.npts, 2599)
        self.assertEqual(tr2.stats['sampling_rate'], 7000)
        np.testing.assert_array_equal(tr.data[:13], testdata)

    def test_read_head_via_obspy(self):
        """
        Read files via obspy.core.Trace
        """
        tr = read(self.file, headonly=True)[0]
        self.assertEqual(tr.stats.npts, 2599)
        self.assertEqual(tr.stats['sampling_rate'], 7000)
        self.assertEqual(str(tr.data), '[]')

    def test_read_and_write_via_obspy(self):
        """
        Read and Write files via obspy.core.Trace
        """
        testdata = np.array([111, 111, 111, 111, 111, 109, 106, 103, 103,
                             110, 121, 132, 139])
        with NamedTemporaryFile() as fh:
            testfile = fh.name
            self.file = os.path.join(self.path, '3cssan.reg.8.1.RNON.wav')
            tr = read(self.file, format='WAV')[0]
            self.assertEqual(tr.stats.npts, 10599)
            self.assertEqual(tr.stats['sampling_rate'], 7000)
            np.testing.assert_array_equal(tr.data[:13], testdata)
            # write
            st2 = Stream()
            st2.traces.append(Trace())
            st2[0].data = tr.data.copy()  # copy the data
            st2.write(testfile, format='WAV', framerate=7000)
            # read without giving the WAV format option
            tr3 = read(testfile)[0]
            self.assertEqual(tr3.stats, tr.stats)
            np.testing.assert_array_equal(tr3.data[:13], testdata)

    def test_rescale_on_write(self):
        """
        Read and Write files via obspy.core.Trace
        """
        with NamedTemporaryFile() as fh:
            testfile = fh.name
            self.file = os.path.join(self.path, '3cssan.reg.8.1.RNON.wav')
            tr = read(self.file, format='WAV')[0]
            for width in (1, 2, 4, None):
                tr.write(testfile, format='WAV', framerate=7000, width=width,
                         rescale=True)
                if width is not None:
                    tr2 = read(testfile, format='WAV')[0]
                    maxint = 2 ** (8 * width - 1) - 1
                    dtype = WIDTH2DTYPE[width]
                    self.assertEqual(maxint, abs(tr2.data).max())
                    expected = (tr.data / abs(tr.data).max() *
                                maxint).astype(dtype)
                    np.testing.assert_array_almost_equal(tr2.data, expected)

    def test_write_stream_via_obspy(self):
        """
        Write streams, i.e. multiple files via obspy.core.Trace
        """
        testdata = np.array([111, 111, 111, 111, 111, 109, 106, 103, 103,
                             110, 121, 132, 139])
        with NamedTemporaryFile() as fh:
            testfile = fh.name
            self.file = os.path.join(self.path, '3cssan.reg.8.1.RNON.wav')
            tr = read(self.file, format='WAV')[0]
            np.testing.assert_array_equal(tr.data[:13], testdata)
            # write
            st2 = Stream([Trace(), Trace()])
            st2[0].data = tr.data.copy()       # copy the data
            st2[1].data = tr.data.copy() // 2  # be sure data are different
            st2.write(testfile, format='WAV', framerate=7000)
            # read without giving the WAV format option
            base, ext = os.path.splitext(testfile)
            testfile0 = "%s%03d%s" % (base, 0, ext)
            testfile1 = "%s%03d%s" % (base, 1, ext)
            tr30 = read(testfile0)[0]
            tr31 = read(testfile1)[0]
            self.assertEqual(tr30.stats, tr.stats)
            self.assertEqual(tr31.stats, tr.stats)
            np.testing.assert_array_equal(tr30.data[:13], testdata)
            np.testing.assert_array_equal(tr31.data[:13], testdata // 2)
            os.remove(testfile0)
            os.remove(testfile1)
