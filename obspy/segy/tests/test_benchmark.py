# -*- coding: utf-8 -*-
"""
The obspy.segy benchmark test suite.
"""

from obspy.core.util.base import ImageComparison, HAS_COMPARE_IMAGE
from obspy.core.util.decorator import skipIf
from obspy.segy.benchmark import plotBenchmark
import glob
import os
import unittest


class BenchmarkTestCase(unittest.TestCase):
    """
    Test cases for benchmark plots.
    """
    @skipIf(not HAS_COMPARE_IMAGE, 'nose not installed or matplotlib too old')
    def test_plotBenchmark(self):
        """
        Test benchmark plot.
        """
        path = os.path.join(os.path.dirname(__file__), 'data')
        path_images = os.path.join(os.path.dirname(__file__), "images")
        sufiles = sorted(glob.glob(os.path.join(path, 'seismic01_*_vz.su')))
        # new temporary file with PNG extension
        with ImageComparison(path_images, 'test_plotBenchmark.png') as ic:
            # generate plot
            plotBenchmark(sufiles, outfile=ic.name, format='PNG')


def suite():
    return unittest.makeSuite(BenchmarkTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
