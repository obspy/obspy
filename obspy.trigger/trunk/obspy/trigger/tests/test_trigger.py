# -*- coding: utf-8 -*-
"""
The obspy.trigger test suite.
"""

from obspy.trigger import recStalta, recStaltaPy
import numpy as N
import unittest


class TriggerTestCase(unittest.TestCase):
    """
    Test cases for obspy.trigger
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_trigger(self):
        """
        """
        N.random.seed(815)
        a = N.random.randn(1000000)
        nsta, nlta = 5, 10
        c1 = recStalta(a, nsta, nlta)
        self.assertAlmostEquals(c1[99], 0.80810165)
        self.assertAlmostEquals(c1[100], 0.75939449)
        self.assertAlmostEquals(c1[101], 0.91763978)
        self.assertAlmostEquals(c1[102], 0.97465004)

    def test_trigger2(self):
        """
        """
        N.random.seed(815)
        a = N.random.randn(1000000).tolist()
        nsta, nlta = 5, 10
        c2 = recStaltaPy(a, nsta, nlta)
        self.assertAlmostEquals(c2[99], 0.80810165)
        self.assertAlmostEquals(c2[100], 0.75939449)
        self.assertAlmostEquals(c2[101], 0.91763978)
        self.assertAlmostEquals(c2[102], 0.97465004)

    def test_trigger3(self):
        """
        """
        self.assertRaises(AssertionError, recStalta, [1], 5, 10)
        self.assertRaises(AssertionError, recStalta,
                          N.array([1], dtype='int32'), 5, 10)


def suite():
    return unittest.makeSuite(TriggerTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
