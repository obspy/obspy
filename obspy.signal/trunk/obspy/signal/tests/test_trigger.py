# -*- coding: utf-8 -*-
"""
The obspy.signal.trigger test suite.
"""

from obspy.signal import recStalta, recStaltaPy, triggerOnset
from ctypes import ArgumentError
import numpy as N
import unittest


class TriggerTestCase(unittest.TestCase):
    """
    Test cases for obspy.trigger
    """
    def setUp(self):
        N.random.seed(815)
        self.data = N.random.randn(int(1e5))
        pass

    def tearDown(self):
        pass

    def test_recStaltaC(self):
        """
        Test case for ctypes version of recStalta
        """
        nsta, nlta = 5, 10
        c1 = recStalta(self.data, nsta, nlta)
        self.assertAlmostEquals(c1[99], 0.80810165)
        self.assertAlmostEquals(c1[100], 0.75939449)
        self.assertAlmostEquals(c1[101], 0.91763978)
        self.assertAlmostEquals(c1[102], 0.97465004)

    def test_recStaltaPy(self):
        """
        Test case for python version of recStalta
        """
        nsta, nlta = 5, 10
        c2 = recStaltaPy(self.data, nsta, nlta)
        self.assertAlmostEquals(c2[99], 0.80810165)
        self.assertAlmostEquals(c2[100], 0.75939449)
        self.assertAlmostEquals(c2[101], 0.91763978)
        self.assertAlmostEquals(c2[102], 0.97465004)

    def test_recStaltaRaise(self):
        """
        Type checking recStalta
        """
        self.assertRaises(ArgumentError, recStalta, [1], 5, 10)
        self.assertRaises(ArgumentError, recStalta,
                          N.array([1], dtype='int32'), 5, 10)

    def test_triggerOnset(self):
        """
        Test trigger onset function
        """
        on_of = [[10, 26], [73, 89], [135, 151], [198, 214], [261, 277]]
        cft = N.sin(N.arange(0,10*N.pi,0.1))
        picks = triggerOnset(cft,0.8,0.5)
        self.assertEquals(picks,on_of)

def suite():
    return unittest.makeSuite(TriggerTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
