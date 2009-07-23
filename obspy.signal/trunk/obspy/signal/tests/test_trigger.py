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
        on_of = [[6, 31], [69, 94], [131, 181], [215, 265], [278, 315]]
        cft = N.concatenate((N.sin(N.arange(0,5*N.pi,0.1))+1,
                             N.sin(N.arange(0,5*N.pi,0.1))+2.1,
                             N.sin(N.arange(0,5*N.pi,0.1))+0.4))
        picks = triggerOnset(cft,1.5,1.0,max_len=50)
        self.assertEquals(picks,on_of)
        #
        if False: # set True for visual understanding the test
            import pylab as P
            P.plot(cft)
            P.hlines([1.5,1.0],0,len(cft))
            on_of = N.array(on_of)
            P.vlines(on_of[:,0],1.0,2.0,color='g',linewidth=2)
            P.vlines(on_of[:,1],0.5,1.5,color='r',linewidth=2)
            P.show()

def suite():
    return unittest.makeSuite(TriggerTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
