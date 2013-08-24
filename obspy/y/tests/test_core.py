# -*- coding: utf-8 -*-

from obspy.y.core import isY, readY
import os
import unittest


class CoreTestCase(unittest.TestCase):
    """
    Nanometrics Y file test suite.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_isYFile(self):
        """
        Testing Y file format.
        """
        testfile = os.path.join(self.path, 'data', 'YAYT_BHZ_20021223.124800')
        self.assertEqual(isY(testfile), True)

    def test_readYFile(self):
        """
        Testing reading Y file format.
        """
        testfile = os.path.join(self.path, 'data', 'YAYT_BHZ_20021223.124800')
        st = readY(testfile)
        self.assertEquals(len(st), 1)
        tr = st[0]
        self.assertEquals(len(tr), 18000)
        self.assertEquals(tr.stats.sampling_rate, 100.0)
        self.assertEquals(tr.stats.station, 'AYT')
        self.assertEquals(tr.stats.channel, 'BHZ')
        self.assertEquals(tr.stats.location, '')
        self.assertEquals(tr.stats.network, '')


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
