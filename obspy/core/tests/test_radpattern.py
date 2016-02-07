# -*- coding: utf-8 -*-
"""
The obspy.imaging.radiation_pattern test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

from obspy import read_events
from obspy.core.event.radpattern import plot_3drpattern
from obspy.core.util.testing import ImageComparison
from obspy.core.util.base import get_matplotlib_version


MATPLOTLIB_VERSION = get_matplotlib_version()


class RadPatternTestCase(unittest.TestCase):
    """
    Test cases for radiation_pattern.
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.join(os.path.dirname(__file__), 'data')
        self.path = path
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')

    def test_farfield_without_quiver(self):
        """
        Tests to plot P/S wave farfield radiation pattern
        """
        ev = read_events("/path/to/CMTSOLUTION", format="CMTSOLUTION")[0]
        with ImageComparison(self.image_dir, 'event_radiation_pattern.png') \
                as ic:
            ev.plot(kind=['s_sphere', 'p_sphere', 'beachball'],
                    outfile=ic.name, show=False)

    @unittest.skipIf(MATPLOTLIB_VERSION < [1, 4],
                     'matplotlib >= 1.4 needed for 3D quiver plot.')
    def test_farfield_with_quiver(self):
        """
        Tests to plot P/S wave farfield radiation pattern
        """
        # Peru 2001/6/23 20:34:23:
        mt = [2.245, -0.547, -1.698, 1.339, -3.728, 1.444]
        plot_3drpattern(mt, kind=['beachball', 's_quiver', 'p_quiver'])


def suite():
    return unittest.makeSuite(RadPatternTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
