# -*- coding: utf-8 -*-
"""
The obspy.seg2 test suite.
"""

import numpy as np
from obspy import read
import os
import unittest


class SEG2TestCase(unittest.TestCase):
    """
    Test cases for SEG2 reading.
    """
    def setUp(self):
        # directory where the test files are located
        self.dir = os.path.dirname(__file__)
        self.path = os.path.join(self.dir, 'data')

    def test_readDataFormat2(self):
        """
        Test reading a SEG2 data format code 2 file (int32).
        """
        basename = os.path.join(self.path,
                                '20130107_103041000.CET.3c.cont.0')
        # read SEG2 data (in counts, int32)
        st = read(basename + ".seg2.gz")
        # read reference ASCII data (in micrometer/s)
        results = np.loadtxt(basename + ".DAT.gz").T
        # test all three components
        for tr, result in zip(st, results):
            # convert raw data to micrometer/s (descaling goes to mm/s)
            scaled_data = tr.data * float(tr.stats.seg2.DESCALING_FACTOR) * 1e3
            self.assertTrue(np.allclose(scaled_data, result, rtol=1e-7,
                                        atol=1e-7))


def suite():
    return unittest.makeSuite(SEG2TestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
