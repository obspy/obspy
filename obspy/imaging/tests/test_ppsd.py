# -*- coding: utf-8 -*-
"""
Image test(s) for obspy.signal.spectral_exstimation.PPSD.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import matplotlib.pyplot as plt

from obspy.core.util.testing import ImageComparison
from obspy.signal.tests.test_spectral_estimation import _get_ppsd


class PPSDTestCase(unittest.TestCase):
    """
    Test cases for PPSD plotting.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'images')
        self.ppsd = _get_ppsd()

    def test_ppsd_plot(self):
        """
        Test plot of ppsd example data, normal (non-cumulative) style.
        """
        with ImageComparison(self.path, 'ppsd.png') as ic:
            self.ppsd.plot(
                show=False, show_coverage=True, show_histogram=True,
                show_percentiles=True, percentiles=[75, 90],
                show_noise_models=True, grid=True, max_percentage=50,
                period_lim=(0.02, 100), show_mode=True, show_mean=True)
            fig = plt.gcf()
            ax = fig.axes[0]
            ax.set_ylim(-160, -130)
            plt.draw()
            fig.savefig(ic.name)

    def test_ppsd_plot_cumulative(self):
        """
        Test plot of ppsd example data, cumulative style.
        """
        with ImageComparison(self.path, 'ppsd_cumulative.png') as ic:
            self.ppsd.plot(
                show=False, show_coverage=True, show_histogram=True,
                show_noise_models=True, grid=True, period_lim=(0.02, 100),
                cumulative=True)
            fig = plt.gcf()
            ax = fig.axes[0]
            ax.set_ylim(-160, -130)
            plt.draw()
            fig.savefig(ic.name)


def suite():
    return unittest.makeSuite(PPSDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
