# -*- coding: utf-8 -*-
"""
The obspy.segy benchmark test suite.
"""

from matplotlib.testing.compare import compare_images
from obspy.core.util.base import NamedTemporaryFile
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
