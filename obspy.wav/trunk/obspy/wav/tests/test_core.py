#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The audio wav.core test suite.
"""

from obspy.core import read, Stream, Trace
import inspect, os, unittest, filecmp
from obspy.core.util import NamedTemporaryFile


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
        testdata = [64, 78, 99, 119, 123, 107, 72, 31, 2, 0, 30, 84, 141]
        tr = read(self.file)[0]
        self.assertEqual(tr.stats.npts, 2599)
        self.assertEqual(tr.stats['sampling_rate'], 7000)
        for _i in xrange(13):
            self.assertEqual(tr.data[_i], testdata[_i])
        tr2 = read(self.file, format='WAV')[0]
        self.assertEqual(tr2.stats.npts, 2599)
        self.assertEqual(tr2.stats['sampling_rate'], 7000)
        for _i in xrange(13):
            self.assertEqual(tr.data[_i], testdata[_i])

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
        testdata = [111, 111, 111, 111, 111, 109, 106, 103, 103, 110, 121,
                    132, 139]
        testfile = NamedTemporaryFile().name
        self.file = os.path.join(self.path, 'data', '3cssan.reg.8.1.RNON.wav')
        tr = read(self.file, format='WAV')[0]
        self.assertEqual(tr.stats.npts, 10599)
        self.assertEqual(tr.stats['sampling_rate'], 7000)
        for _i in xrange(13):
            self.assertEqual(tr.data[_i], testdata[_i])
        # write 
        st2 = Stream()
        st2.traces.append(Trace())
        st2[0].data = tr.data[:] #copy the data
        st2.write(testfile, format='WAV', framerate=7000)
        del st2
        # and read again
        tr3 = read(testfile)[0]
        self.assertEqual(tr3.stats, tr.stats)
        for _i in xrange(13):
            self.assertEqual(tr3.data[_i], testdata[_i])
        self.assertEqual(filecmp.cmp(self.file, testfile), True)
        os.remove(testfile)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
