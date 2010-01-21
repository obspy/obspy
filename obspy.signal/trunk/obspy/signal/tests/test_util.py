#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Filter test suite.
"""

from obspy.signal.util import xcorr, lib_name
import numpy as np
import ctypes as C
import inspect
import unittest
import os


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
        np.random.seed(815) # make test reproducable
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

    def test_SRL(self):
        """
        Tests if example in ObsPy paper submitted to the Electronic
        Seismologist section of SRL is still working. The test shouldn't be
        changed because the reference gets wrong.
        """
        np.random.seed(815)
        data1 = np.random.randn(1000).astype('float32')
        data2 = data1.copy()
        window_len = 100

        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        name = os.path.join(path, os.pardir, 'lib', lib_name)
        lib = C.CDLL(name)
        #
        shift = C.c_int()
        coe_p = C.c_double()
        lib.X_corr(data1.ctypes.data, data2.ctypes.data,
                   window_len, len(data1), len(data2),
                   C.byref(shift), C.byref(coe_p))

        self.assertAlmostEquals(0.0, shift.value)
        self.assertAlmostEquals(1.0, coe_p.value)

def suite():
    return unittest.makeSuite(UtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
