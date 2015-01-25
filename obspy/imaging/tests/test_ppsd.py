# -*- coding: utf-8 -*-
"""
Image test(s) for obspy.signal.spectral_exstimation.PPSD.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.core.util.testing import ImageComparison
from obspy.signal.tests.test_spectral_estimation import _get_ppsd
from copy import deepcopy
import os
import unittest


ppsd = _get_ppsd()


class PPSDTestCase(unittest.TestCase):
    """
    Test cases for PPSD plotting.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'images')
        self.ppsd = deepcopy(ppsd)

    def test_ppsd_plot(self):
        """
        """
        with ImageComparison(self.path, 'ppsd.png') as ic:
            self.ppsd.plot(
                show=False, show_coverage=True, show_histogram=True,
                show_percentiles=True, percentiles=[75, 90],
                show_noise_models=True, grid=True, max_percentage=50,
                period_lim=(0.02, 100), show_mode=True, show_mean=True)
            from matplotlib.pyplot import gcf, draw
            fig = gcf()
            ax = fig.axes[0]
            ax.set_ylim(-160, -130)
            draw()
            fig.savefig(ic.name)


def suite():
    return unittest.makeSuite(PPSDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
