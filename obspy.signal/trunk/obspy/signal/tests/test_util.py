#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Filter test suite.
"""

from obspy.signal.util import xcorr
import unittest
import numpy as np


class UtilTestCase(unittest.TestCase):
    """
    Test cases for L{obspy.signal.util}.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_xcorr(self):
        """
        """
        # example 1 - all samples are equal
        tr1 = np.random.randn(10000).astype('float32')
        tr2 = tr1.copy()
        shift, corr = xcorr(tr1, tr2, 100)
        self.assertEquals(shift, 0)
        self.assertAlmostEquals(corr, 1, 2)
        # example 2 - all samples are different
        tr1 = np.ones(10000, dtype='float32')
        tr2 = np.zeros(10000, dtype='float32')
        shift, corr = xcorr(tr1, tr2, 100)
        self.assertEquals(shift, 0)
        self.assertAlmostEquals(corr, 0, 2)
        # example 3 - shift of 10 samples
        tr1 = np.random.randn(10000).astype('float32')
        tr2 = np.concatenate((np.zeros(10), tr1[0:-10]))
        shift, corr = xcorr(tr1, tr2, 100)
        self.assertEquals(shift, -10)
        self.assertAlmostEquals(corr, 1, 2)
        shift, corr = xcorr(tr2, tr1, 100)
        self.assertEquals(shift, 10)
        self.assertAlmostEquals(corr, 1, 2)
        # example 4 - shift of 10 samples + small sine disturbance
        tr1 = (np.random.randn(10000) * 100).astype('float32')
        var = np.sin(np.arange(10000, dtype='float32') * 0.1)
        tr2 = np.concatenate((np.zeros(10), tr1[0:-10])) * 0.9
        tr2 += var
        shift, corr = xcorr(tr1, tr2, 100)
        self.assertEquals(shift, -10)
        self.assertAlmostEquals(corr, 1, 2)
        shift, corr = xcorr(tr2, tr1, 100)
        self.assertEquals(shift, 10)
        self.assertAlmostEquals(corr, 1, 2)


def suite():
    return unittest.makeSuite(UtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
