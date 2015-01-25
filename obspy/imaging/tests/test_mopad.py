# -*- coding: utf-8 -*-
"""
The obspy.imaging.mopad test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.core.util.testing import ImageComparison
from obspy.imaging.mopad_wrapper import Beach
import matplotlib.pyplot as plt
import os
import unittest


class MopadTestCase(unittest.TestCase):
    """
    Test cases for mopad.
    """

    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'images')

    def test_collection(self):
        """
        Tests to plot mopad beachballs as collection into an existing axis
        object. The moment tensor values are taken form the
        test_Beachball unit test. See that test for more information about
        the parameters.
        """
        mt = [[0.91, -0.89, -0.02, 1.78, -1.55, 0.47],
              [274, 13, 55],
              [130, 79, 98],
              [264.98, 45.00, -159.99],
              [160.55, 76.00, -46.78],
              [1.45, -6.60, 5.14, -2.67, -3.16, 1.36],
              [235, 80, 35],
              [138, 56, 168],
              [1, 1, 1, 0, 0, 0],
              [-1, -1, -1, 0, 0, 0],
              [1, -2, 1, 0, 0, 0],
              [1, -1, 0, 0, 0, 0],
              [1, -1, 0, 0, 0, -1],
              [179, 55, -78],
              [10, 42.5, 90],
              [10, 42.5, 92],
              [150, 87, 1],
              [0.99, -2.00, 1.01, 0.92, 0.48, 0.15],
              [5.24, -6.77, 1.53, 0.81, 1.49, -0.05],
              [16.578, -7.987, -8.592, -5.515, -29.732, 7.517],
              [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94],
              [150, 87, 1]]

        with ImageComparison(self.path, 'mopad_collection.png') as ic:
            # Initialize figure
            fig = plt.figure(figsize=(6, 6), dpi=300)
            ax = fig.add_subplot(111, aspect='equal')

            # Plot the stations or borders
            ax.plot([-100, -100, 100, 100], [-100, 100, -100, 100], 'rv')

            x = -100
            y = -100
            for i, t in enumerate(mt):
                # add the beachball (a collection of two patches) to the axis
                ax.add_collection(Beach(t, width=30, xy=(x, y), linewidth=.6))
                x += 50
                if (i + 1) % 5 == 0:
                    x = -100
                    y += 50

            # set the x and y limits
            ax.axis([-120, 120, -120, 120])

            # create and compare image
            fig.savefig(ic.name)


def suite():
    return unittest.makeSuite(MopadTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
