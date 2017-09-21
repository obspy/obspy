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
        self.image_dir = os.path.join(os.path.dirname(__file__),
                                      'images')
        self.model = TauPyModel(model="iasp91")

    def test_spherical_many_phases(self):
        """
        Spherical plot of the ray paths for many phases for a single
        epicentral distance, but both ways around the globe.
        """
        with ImageComparison(self.image_dir,
                             "spherical_many_phases.png") as ic:
            self.model.get_ray_paths(500,
                                     140).plot_rays(plot_type="spherical",
                                                    plot_all=True,
                                                    show=False)
            plt.savefig(ic.name)

    def test_spherical_many_phases_buried_station(self):
        """
        Same as test_spherical_many_phases, but this time the receiver is
        buried.
        """
        with ImageComparison(self.image_dir,
                             "spherical_many_phases_buried_station.png") as ic:
            arrivals = self.model.get_ray_paths(500, 140,
                                                receiver_depth_in_km=1000)
            arrivals.plot_rays(plot_type="spherical",
                               plot_all=True, show=False)
            plt.savefig(ic.name)

    def test_spherical_many_phases_one_way(self):
        """
        Same as test_spherical_many_phases, but this time no phases
        travelling the other way around the globe are plotted.
        """
        with ImageComparison(self.image_dir,
                             "spherical_many_phases_one_way.png") as ic:
            self.model.get_ray_paths(500,
                                     140).plot_rays(plot_type="spherical",
                                                    plot_all=False,
                                                    show=False)
            plt.savefig(ic.name)

    def test_spherical_more_then_360_degrees(self):
        """
        Spherical plot with rays traveling more than 360.0 degrees.
        """
        with ImageComparison(self.image_dir,
                             "spherical_more_then_360.png") as ic:
            self.model.get_ray_paths(0, 10, phase_list=["PPPPPP"]).plot_rays(
                plot_type="spherical", plot_all=True, show=False,
                phase_list=['PPPPPP'])
            plt.savefig(ic.name)

    def test_spherical_diff_phases(self):
        """
        Spherical plot of ``diff`` phases.
        """
        with ImageComparison(self.image_dir,
                             "spherical_diff_phases.png") as ic:
            self.model.get_ray_paths(
                700, 140, phase_list=["Pdiff", "Sdiff", "pSdiff", "sSdiff",
                                      "pPdiff", "sPdiff"]).plot_rays(
                                          plot_type="spherical", legend=True,
                                          plot_all=True, show=False)
            plt.savefig(ic.name)

    def test_cartesian_many_phases(self):
        """
         Cartesian plot of the ray paths for many phases for a single
         epicentral distance, but both ways around the globe.
         """
        with ImageComparison(self.image_dir,
                             "cartesian_many_phases.png") as ic:
            self.model.get_ray_paths(500, 140).plot_rays(plot_type="cartesian",
                                                         plot_all=True,
                                                         show=False)
            plt.savefig(ic.name)

    def test_cartesian_many_phases_buried_station(self):
        """
        Same as test_cartesian_many_phases but this time the receiver is
        buried.
        """
        with ImageComparison(self.image_dir,
                             "cartesian_many_phases_buried_station.png") as ic:
            arrivals = self.model.get_ray_paths(500, 140,
                                                receiver_depth_in_km=1000)
            arrivals.plot_rays(plot_type="cartesian", plot_all=True,
                               show=False)
            plt.savefig(ic.name)

    def test_cartesian_many_phases_one_way(self):
        """
        Same as test_cartesian_many_phases but this time no phases
        travelling the other way around the globe are plotted.
        """
        with ImageComparison(self.image_dir,
                             "cartesian_many_phases_one_way.png") as ic:
            self.model.get_ray_paths(500, 140).plot_rays(plot_type="cartesian",
                                                         plot_all=False,
                                                         show=False)
            plt.savefig(ic.name)

    def test_plot_travel_times(self):
        """
        Travel time plot for many phases at a single epicentral distance.
        """
        with ImageComparison(self.image_dir,
                             "traveltimes_many_phases.png") as ic:
            self.model.get_ray_paths(10, 100,
                                     phase_list=("ttbasic",)).plot_times(
                                         show=False, phase_list=("ttbasic",),
                                         legend=True)
            plt.savefig(ic.name)


def suite():
    return unittest.makeSuite(TauPyPlottingTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
