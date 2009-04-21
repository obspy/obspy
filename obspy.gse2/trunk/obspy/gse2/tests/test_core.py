#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The gse2.core test suite.
"""

from obspy.gse2 import libgse2
from obspy import Trace
import inspect, os, random, unittest


class CoreTestCase(unittest.TestCase):
    """
    Test cases for libgse2.
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.file = os.path.join(path, 'data','loc_RJOB20050831023349.z')
    
    def tearDown(self):
        pass
    
    def test_readViaObspy(self):
        """
        Read files via L{obspy.Trace}
        """
        tr = Trace()
        tr.read(self.file,format='GSE2')
        self.assertEqual(tr.stats['station'], 'RJOB ')
        self.assertEqual(tr.stats.npts, 12000)
        self.assertEqual(tr.stats['sampling_rate'], 200)
        self.assertEqual(tr.stats.get('channel'), '  Z')
        self.assertEqual(tr.stats.get('starttime'), 1125455629.8499985)
        self.assertEqual(tr.data[0:13], [12, -10, 16, 33, 9, 26, 16, 7, 17, 6, 1, 3, -2])
    
    def test_writeViaObspy(self):
        """
        Write files via L{obspy.Trace}
        """
        pass


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
