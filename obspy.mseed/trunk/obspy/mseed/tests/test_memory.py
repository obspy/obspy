# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime, Stream, Trace, read
from obspy.mseed.core import readMSEED
import inspect
import numpy as N
import os
import unittest


class CoreTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(inspect.getsourcefile(self.__class__))

    def tearDown(self):
        pass

    def test_readMemory(self):
        """
        Read file test via L{obspy.core.Stream}.
        """
        testfile = os.path.join(self.path, 'data', 'BW.BGLD..EHE.D.2008.001')
        for i in xrange(100):
            stream = read(testfile)
            stream.verify()
            for tr in stream:
                self.assertEqual(tr.stats.network, 'BW')
                self.assertEqual(tr.stats['station'], 'BGLD')
                self.assertEqual(tr.stats.get('npts'), 17280322)
            print '.',

def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
