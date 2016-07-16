# -*- coding: utf-8 -*-
"""
The obspy.imaging.spectrum test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest
import warnings

import numpy as np
from obspy.core.util.testing import ImageComparison
from obspy.imaging import spectrum


class SpectrumTestCase(unittest.TestCase):
    """
    Test cases for spectrum computation/plotting.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'images')

    def test_spectrogram(self):
        """
        Create spectrogram plotting examples in tests/output directory.
        """

        srate = 10
        d = np.zeros(1024)
        d[512] = srate
        fig, ax, frq, X = spectrum.plot_spectrum(d, srate, show=False)

        self.assertEqual(x, np.ones((513)))


def suite():
    return unittest.makeSuite(SpectrumTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
