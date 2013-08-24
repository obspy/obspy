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
        stream = readY(testfile)
        print stream


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
