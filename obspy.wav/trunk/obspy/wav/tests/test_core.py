#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The audio wav.core test suite.
"""

from obspy.core import read, Stream, Trace
from obspy.core.util import NamedTemporaryFile
import inspect
import os
import unittest
import numpy as np


class CoreTestCase(unittest.TestCase):
    """
    Test cases for audio WAV support
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.file = os.path.join(self.path, 'data', '3cssan.near.8.1.RNON.wav')

    def tearDown(self):
        pass

    def test_readViaObspy(self):
        """
        Read files via L{obspy.Trace}
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

    def test_readHeadViaObspy(self):
        """
        Read files via L{obspy.Trace}
        """
        tr = read(self.file, headonly=True)[0]
        self.assertEqual(tr.stats.npts, 2599)
        self.assertEqual(tr.stats['sampling_rate'], 7000)
        self.assertEqual(str(tr.data), '[]')

    def test_readAndWriteViaObspy(self):
        """
        Read and Write files via L{obspy.Trace}
        """
        testdata = np.array([111, 111, 111, 111, 111, 109, 106, 103, 103,
                             110, 121, 132, 139])
        testfile = NamedTemporaryFile().name
        self.file = os.path.join(self.path, 'data', '3cssan.reg.8.1.RNON.wav')
        tr = read(self.file, format='WAV')[0]
        self.assertEqual(tr.stats.npts, 10599)
        self.assertEqual(tr.stats['sampling_rate'], 7000)
        np.testing.assert_array_equal(tr.data[:13], testdata)
        # write 
        st2 = Stream()
        st2.traces.append(Trace())
        st2[0].data = tr.data.copy() #copy the data
        st2.write(testfile, format='WAV', framerate=7000)
        # read without giving the WAV format option
        tr3 = read(testfile)[0]
        self.assertEqual(tr3.stats, tr.stats)
        np.testing.assert_array_equal(tr3.data[:13], testdata)
        os.remove(testfile)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
