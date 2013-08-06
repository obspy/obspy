# -*- coding: utf-8 -*-
"""
The obspy.segy core test suite.
"""

from obspy.segy.benchmark import plotBenchmark
import glob
import os
import unittest


class BenchmarkTestCase(unittest.TestCase):
    """
    Test cases for benchmark plots.
    """
    def test_plotBenchmark(self):
        """
        Test benchmark plot.
        """
        path = os.path.join(os.path.dirname(__file__), 'data',
                            'seismic01_*_vz.su')
        sufiles = glob.glob(path)
        # plot into binary string
        result = plotBenchmark(sufiles, format='PNG')
        self.assertEquals(result[1:4], 'PNG')


def suite():
    return unittest.makeSuite(BenchmarkTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
