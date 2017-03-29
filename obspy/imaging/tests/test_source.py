# -*- coding: utf-8 -*-
"""
The obspy.imaging.source test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

from obspy.imaging.source import plot_radiation_pattern
from obspy.core.util.base import MATPLOTLIB_VERSION


class RadPatternTestCase(unittest.TestCase):
    """
    Test cases for radiation_pattern.
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.join(os.path.dirname(__file__), 'data')
        self.path = path
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')

    @unittest.skipIf(MATPLOTLIB_VERSION < [1, 4],
                     'matplotlib >= 1.4 needed for 3D quiver plot.')
    def test_farfield_with_quiver(self):
        """
        Tests to plot P/S wave farfield radiation pattern
        """
        # Peru 2001/6/23 20:34:23:
        mt = [2.245, -0.547, -1.698, 1.339, -3.728, 1.444]
        plot_radiation_pattern(
            mt, kind=['beachball', 's_quiver', 'p_quiver'], show=False)


def suite():
    return unittest.makeSuite(RadPatternTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
