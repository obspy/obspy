# -*- coding: utf-8 -*-
"""
The obspy.realtime.signal.util test suite.
"""
from obspy.core import Trace
from obspy.realtime.signal.util import scale, integrate, differentiate, boxcar

import numpy as np
import unittest


class RealTimeSignalUtilTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_scale(self):
        """
        Tests the scaling function which simply scales the array of a Trace
        object.
        """
        tr = Trace(data=np.arange(10))
        data = scale(tr)
        np.testing.assert_array_equal(np.arange(10), data)
        self.assertTrue(data is tr.data)
        data = scale(tr, factor=2.0)
        np.testing.assert_array_equal(np.arange(10) * 2.0, data)
        self.assertTrue(data is tr.data)
        data = scale(tr, factor=0.0)
        np.testing.assert_array_equal(np.zeros(10), data)
        self.assertTrue(data is tr.data)

    def test_integrate(self):
        """
        Tests simple integration.
        """
        tr = Trace(data=np.arange(10, dtype='float32'))
        tr.stats.delta = 0.5
        data = integrate(tr)
        np.testing.assert_array_equal(\
            np.array([0.0, 0.5, 1.5, 3.0, 5.0, 7.5, 10.5, 14.0, 18.0, 22.5],
            dtype=np.float32), data)
        self.assertTrue(data is tr.data)

    def test_differentation(self):
        """
        Tests simple differentiation.
        """
        tr = Trace(data=np.arange(10, dtype='float32'))
        tr.stats.delta = 0.5
        data = differentiate(tr)
        np.testing.assert_array_equal(\
            np.array([0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            dtype=np.float32), data)
        self.assertTrue(data is tr.data)

    def test_boxcar_smoothing(self):
        """
        Tests simple differentiation.
        """
        tr = Trace(data=np.arange(10, dtype='float32'))
        data = boxcar(tr, width=2)

        np.testing.assert_almost_equal(\
            np.array([0.0, 0.33333334, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                     dtype=np.float32), data)
        self.assertTrue(data is tr.data)


def suite():
    return unittest.makeSuite(RealTimeSignalUtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
