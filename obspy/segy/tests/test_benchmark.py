# -*- coding: utf-8 -*-
"""
The obspy.segy benchmark test suite.
"""

from obspy.core.util.base import NamedTemporaryFile
from obspy.core.util.decorator import skipIf
from obspy.segy.benchmark import plotBenchmark
import glob
import os
import unittest

# checking for newer matplotlib version and if nose is installed
try:
    from matplotlib.testing.compare import compare_images
    HAS_COMPARE_IMAGE = True
except ImportError:
    HAS_COMPARE_IMAGE = False


class BenchmarkTestCase(unittest.TestCase):
    """
    Test cases for benchmark plots.
    """
    @skipIf(not HAS_COMPARE_IMAGE, 'nose not installed or matplotlib to old')
    def test_plotBenchmark(self):
        """
        Test benchmark plot.
        """
        path = os.path.join(os.path.dirname(__file__), 'data',
                            'seismic01_*_vz.su')
        sufiles = glob.glob(path)
        # new temporary file with PNG extension
        with NamedTemporaryFile(suffix='.png') as tf:
            # generate plot
            plotBenchmark(sufiles, outfile=tf.name, format='PNG')
            # compare images
            expected_image = os.path.join(os.path.dirname(__file__), 'images',
                                          'test_plotBenchmark.png')
            compare_images(tf.name, expected_image, 0.001)


def suite():
    return unittest.makeSuite(BenchmarkTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
