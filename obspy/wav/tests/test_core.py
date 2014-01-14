#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The audio wav.core test suite.
"""

from __future__ import division
from __future__ import unicode_literals
from future.builtins import str
from obspy import read, Stream, Trace
from obspy.core.util import NamedTemporaryFile
from obspy.core.util.decorator import skipIf
from obspy.wav.core import WIDTH2DTYPE
import numpy as np
import os
import unittest


numpy_version = float(".".join(np.version.version.split('.')[:2]))
if numpy_version <= 1.3:
    OLD_NUMPY = True
else:
    OLD_NUMPY = False


class CoreTestCase(unittest.TestCase):
    """
    Test cases for audio WAV support
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        self.file = os.path.join(self.path, '3cssan.near.8.1.RNON.wav')

    def test_readViaObsPy(self):
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

    def test_readHeadViaObsPy(self):
        """
        Read files via obspy.core.Trace
        """
        tr = read(self.file, headonly=True)[0]
        self.assertEqual(tr.stats.npts, 2599)
        self.assertEqual(tr.stats['sampling_rate'], 7000)
        self.assertEqual(str(tr.data), '[]')

    def test_readAndWriteViaObsPy(self):
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

    @skipIf(OLD_NUMPY, 'needs a recent NumPy version')
    def test_rescaleOnWrite(self):
        """
        Read and Write files via obspy.core.Trace
        """
        with NamedTemporaryFile() as fh:
            testfile = fh.name
            self.file = os.path.join(self.path, '3cssan.reg.8.1.RNON.wav')
            tr = read(self.file, format='WAV')[0]
            for width in (1, 2, 4):
                tr.write(testfile, format='WAV', framerate=7000, width=width,
                         rescale=True)
                tr2 = read(testfile, format='WAV')[0]
                maxint = 2 ** (8 * width - 1) - 1
                dtype = WIDTH2DTYPE[width]
                self.assertEqual(maxint, abs(tr2.data).max())
                expected = (tr.data / abs(tr.data).max() *
                            maxint).astype(dtype)
                np.testing.assert_array_almost_equal(tr2.data, expected)

    def test_writeStreamViaObsPy(self):
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


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
