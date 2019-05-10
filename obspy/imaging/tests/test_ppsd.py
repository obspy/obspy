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
import numpy as np

from obspy.core.util.testing import ImageComparison, MATPLOTLIB_VERSION
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
        # Catch underflow warnings due to plotting on log-scale.
        with np.errstate(all='ignore'):
            with ImageComparison(self.path, 'ppsd.png', reltol=1.5) as ic:
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

    def test_ppsd_plot_frequency(self):
        """
        Test plot of ppsd example data, normal (non-cumulative) style.
        """
        # mpl < 2.2 has slightly offset ticks/ticklabels, so needs a higher
        # tolerance (see e.g. http://tests.obspy.org/102260)
        reltol = 2
        if MATPLOTLIB_VERSION < [2, 2]:
            reltol = 4
        # Catch underflow warnings due to plotting on log-scale.
        with np.errstate(all='ignore'):
            with ImageComparison(self.path, 'ppsd_freq.png',
                                 reltol=reltol) as ic:
                self.ppsd.plot(
                    show=False, show_coverage=False, show_histogram=True,
                    show_percentiles=True, percentiles=[20, 40],
                    show_noise_models=True, grid=False, max_percentage=50,
                    period_lim=(0.2, 50), show_mode=True, show_mean=True,
                    xaxis_frequency=True)
                fig = plt.gcf()
                ax = fig.axes[0]
                ax.set_ylim(-160, -130)
                plt.draw()
                fig.savefig(ic.name, dpi=50)

    def test_ppsd_plot_cumulative(self):
        """
        Test plot of ppsd example data, cumulative style.
        """
        # Catch underflow warnings due to plotting on log-scale.
        with np.errstate(all='ignore'):
            with ImageComparison(self.path, 'ppsd_cumulative.png',
                                 reltol=1.5) as ic:
                self.ppsd.plot(
                    show=False, show_coverage=True, show_histogram=True,
                    show_noise_models=True, grid=True, period_lim=(0.02, 100),
                    cumulative=True,
                    # This does not do anything but silences a warning that
                    # the `cumulative` and `max_percentage` arguments cannot
                    #  be used at the same time.
                    max_percentage=None)
                fig = plt.gcf()
                ax = fig.axes[0]
                ax.set_ylim(-160, -130)
                plt.draw()
                fig.savefig(ic.name)


def suite():
    return unittest.makeSuite(PPSDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
