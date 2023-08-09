# -*- coding: utf-8 -*-
"""
The obspy.imaging.source test suite.
"""
import os

from obspy.imaging.source import plot_radiation_pattern


class TestRadPattern:
    """
    Test cases for radiation_pattern.
    """
    path = os.path.join(os.path.dirname(__file__), 'data')
    image_dir = os.path.join(os.path.dirname(__file__), 'images')

    def test_farfield_with_quiver(self):
        """
        Tests to plot P/S wave farfield radiation pattern
        """
        # Peru 2001/6/23 20:34:23:
        mt = [2.245, -0.547, -1.698, 1.339, -3.728, 1.444]
        plot_radiation_pattern(
            mt, kind=['beachball', 's_quiver', 'p_quiver'], show=False)
