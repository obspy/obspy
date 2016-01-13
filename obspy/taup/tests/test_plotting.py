#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the high level obspy.taup.tau interface.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import matplotlib.pyplot as plt

from obspy.core.util.testing import ImageComparison
from obspy.taup import TauPyModel


class TauPyPlottingTestCase(unittest.TestCase):
    """
    TauPy plotting tests.
    """
    def setUp(self):
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.model = TauPyModel(model="iasp91")

    def test_spherical_many_phases(self):
        with ImageComparison(self.image_dir,
                             "spherical_many_phases.png") as ic:
            self.model.get_ray_paths(500, 140).plot(plot_type="spherical",
                                                    plot_all=True, show=False)
            plt.savefig(ic.name)

    def test_spherical_many_phases_buried_station(self):
        """
        Same as test_spherical_many_phases but this time the receiver is
        buried.
        """
        with ImageComparison(self.image_dir,
                             "spherical_many_phases_buried_station.png") as ic:
            arrivals = self.model.get_ray_paths(500, 140,
                                                receiver_depth_in_km=200)
            arrivals.plot(plot_type="spherical", plot_all=True, show=False)
            plt.savefig(ic.name)

    def test_spherical_many_phases_no_other_way(self):
        """
        Same as test_spherical_many_phases but this time no phases
        travelling the other way are plotted.
        """
        with ImageComparison(self.image_dir,
                             "spherical_many_phases_single_way.png") as ic:
            self.model.get_ray_paths(500, 140).plot(plot_type="spherical",
                                                    plot_all=False, show=False)
            plt.savefig(ic.name)

    def test_spherical_more_then_360_degrees(self):
        """
        Test a plot where rays travel more than 360.0 degrees.
        """
        with ImageComparison(self.image_dir,
                             "spherical_more_then_360.png") as ic:
            self.model.get_ray_paths(0, 10, phase_list=["PPPPPP"]).plot(
                plot_type="spherical", plot_all=True, show=False)
            plt.savefig(ic.name)

    def test_spherical_diff_phases(self):
        with ImageComparison(self.image_dir,
                             "spherical_diff_phases.png") as ic:
            self.model.get_ray_paths(
                700, 140, phase_list=["Pdiff", "Sdiff", "pSdiff", "sSdiff",
                                      "pPdiff", "sPdiff"]).plot(
                plot_type="spherical", plot_all=True, show=False)
            plt.savefig(ic.name)

    def test_cartesian_many_phases(self):
        with ImageComparison(self.image_dir,
                             "cartesian_many_phases.png") as ic:
            self.model.get_ray_paths(500, 140).plot(plot_type="cartesian",
                                                    plot_all=True, show=False)
            plt.savefig(ic.name)

    def test_cartesian_many_phases_buried_station(self):
        """
        Same as test_cartesian_many_phases but this time the receiver is
        buried.
        """
        with ImageComparison(self.image_dir,
                             "cartesian_many_phases_buried_station.png") as ic:
            arrivals = self.model.get_ray_paths(500, 140,
                                                receiver_depth_in_km=200)
            arrivals.plot(plot_type="cartesian", plot_all=True, show=False)
            plt.savefig(ic.name)

    def test_cartesian_many_phases_no_other_way(self):
        """
        Same as test_cartesian_many_phases but this time no phases
        travelling the other way are plotted.
        """
        with ImageComparison(self.image_dir,
                             "cartesian_many_phases_single_way.png") as ic:
            self.model.get_ray_paths(500, 140).plot(plot_type="cartesian",
                                                    plot_all=False, show=False)
            plt.savefig(ic.name)

    def test_cartesian_multiple_station_markers(self):
        """
        Cartesian plot with two station markers.
        """
        with ImageComparison(self.image_dir,
                             "cartesian_multiple_stations.png") as ic:
            self.model.get_ray_paths(350, 100).plot(plot_type="cartesian",
                                                    plot_all=True, show=False)
            plt.savefig(ic.name)


def suite():
    return unittest.makeSuite(TauPyPlottingTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
