# -*- coding: utf-8 -*-
"""
The obspy.segy benchmark test suite.
"""

from obspy.core.util.testing import ImageComparison, HAS_COMPARE_IMAGE
from obspy.core.util.decorator import skipIf
from obspy.core.util.base import getMatplotlibVersion
from obspy.segy.benchmark import plotBenchmark
import glob
import os
import unittest
import warnings
import numpy as np


MATPLOTLIB_VERSION = getMatplotlibVersion()


class BenchmarkTestCase(unittest.TestCase):
    """
    Test cases for benchmark plots.
    """
    @skipIf(not HAS_COMPARE_IMAGE, 'nose not installed or matplotlib too old')
    def test_plotBenchmark(self):
        """
        Test benchmark plot.
        """
        reltol = 1
        if MATPLOTLIB_VERSION < [1, 2, 0]:
            reltol = 2
        path = os.path.join(os.path.dirname(__file__), 'data')
        path_images = os.path.join(os.path.dirname(__file__), "images")
        sufiles = sorted(glob.glob(os.path.join(path, 'seismic01_*_vz.su')))
        # new temporary file with PNG extension
        with ImageComparison(path_images, 'test_plotBenchmark.png',
                             reltol=reltol) as ic:
            # generate plot
            with warnings.catch_warnings(record=True) as w:
                warnings.resetwarnings()
                np_err = np.seterr(all="warn")
                plotBenchmark(sufiles, outfile=ic.name, format='PNG')
                np.seterr(**np_err)
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, RuntimeWarning)
            self.assertEqual(str(w[0].message),
                             'underflow encountered in divide')


def suite():
    return unittest.makeSuite(BenchmarkTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
