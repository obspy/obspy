# -*- coding: utf-8 -*-
"""
The obspy.trigger test suite.
"""

from obspy.trigger import recStalta, recStaltaPy
from ctypes import ArgumentError
import numpy as N
import unittest


class TriggerTestCase(unittest.TestCase):
    """
    Test cases for obspy.trigger
    """
    def setUp(self):
        N.random.seed(815)
        self.data = N.random.randn(int(1e6))
        pass

    def tearDown(self):
        pass

    def test_trigger(self):
        """
        Test case for ctypes version of recStalta
        """
        nsta, nlta = 5, 10
        c1 = recStalta(self.data, nsta, nlta)
        self.assertAlmostEquals(c1[99], 0.80810165)
        self.assertAlmostEquals(c1[100], 0.75939449)
        self.assertAlmostEquals(c1[101], 0.91763978)
        self.assertAlmostEquals(c1[102], 0.97465004)

    def test_trigger2(self):
        """
        Test case for python version of recStalta
        """
        nsta, nlta = 5, 10
        c2 = recStaltaPy(self.data, nsta, nlta)
        self.assertAlmostEquals(c2[99], 0.80810165)
        self.assertAlmostEquals(c2[100], 0.75939449)
        self.assertAlmostEquals(c2[101], 0.91763978)
        self.assertAlmostEquals(c2[102], 0.97465004)

    def test_trigger3(self):
        """
        Type checking recStalta
        """
        self.assertRaises(ArgumentError, recStalta, [1], 5, 10)
        self.assertRaises(ArgumentError, recStalta,
                          N.array([1], dtype='int32'), 5, 10)


def suite():
    return unittest.makeSuite(TriggerTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
