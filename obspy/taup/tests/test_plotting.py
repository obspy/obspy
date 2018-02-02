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
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt

from obspy.core.util.testing import ImageComparison
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from obspy.taup import TauPyModel, plot_travel_times


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
                                          phase_list=["Pdiff", "Sdiff",
                                                      "pSdiff", "sSdiff",
                                                      "pPdiff", "sPdiff"],
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
        arrivals = self.model.get_ray_paths(500, 140)
        with ImageComparison(self.image_dir,
                             "cartesian_many_phases_one_way.png") as ic:
            arrivals.plot_rays(plot_type="cartesian", plot_all=False,
                               show=False)
            plt.savefig(ic.name)
        # check warning message on deprecated routine
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with ImageComparison(self.image_dir,
                                 "cartesian_many_phases_one_way.png") as ic:
                # default for legend was True in legacy method
                arrivals.plot(plot_type="cartesian", plot_all=False,
                              show=False, legend=False)
                plt.savefig(ic.name)
            self.assertGreaterEqual(len(w), 1)
            for w_ in w:
                try:
                    self.assertEqual(
                        str(w_.message), 'The plot() function is '
                        'deprecated. Please use arrivals.plot_rays()')
                    self.assertEqual(w_.category, ObsPyDeprecationWarning)
                except AssertionError:
                    continue
                break
            else:
                raise

    def test_plot_travel_times(self):
        """
        Travel time plot for many phases at a single epicentral distance.
        """
        with ImageComparison(self.image_dir,
                             "traveltimes_many_phases.png") as ic:
            self.model.get_ray_paths(10, 100,
                                     phase_list=("ttbasic",)).plot_times(
                                         show=False, legend=True)
            plt.savefig(ic.name)

    def test_plot_travel_times_convenience(self):
        """
        Travel time plot for many phases at multiple epicentral distances,
        convenience function
        """
        with ImageComparison(
                self.image_dir,
                "traveltimes_many_phases_multiple_degrees.png") as ic:
            # base line image looks awkward with points at the left/right end
            # of the plot but that's only due to the "classic" style sheet used
            # in tests. default mpl2 style makes room around plotted artists
            # larger point size to ensure plot can fail properly if points
            # should move
            mpl.rcParams['lines.markersize'] = 20
            plot_travel_times(10, phase_list=("P", "S", "SKS", "PP"),
                              min_degrees=40, max_degrees=60, show=False,
                              legend=True, npoints=4)
            plt.savefig(ic.name)

        def _test_plot_all(plot_all):
            mpl.rcParams['lines.markersize'] = 200
            plot_travel_times(10, phase_list=("SSS",), min_degrees=150,
                              max_degrees=200, npoints=4, legend=False,
                              plot_all=plot_all)

        # now test option plot_all=False
        with ImageComparison(
                self.image_dir, "traveltimes_plot_all_False.png") as ic:
            _test_plot_all(plot_all=False)
            plt.savefig(ic.name)

        # Now with option plot_all=True
        with ImageComparison(
                self.image_dir, "traveltimes_plot_all_True.png") as ic:
            _test_plot_all(plot_all=True)
            plt.savefig(ic.name)

    def test_ray_plot_mismatching_axes_type_warnings(self):
        """
        Test warnings when attempting ray path plots in spherical/cartesian
        with bad axes type (polar/not polar).
        """
        arrivals = self.model.get_ray_paths(500, 20, phase_list=['P'])
        # polar pot attempted in cartesian axes
        fig, ax = plt.subplots()
        expected_message = ("Axes instance provided for plotting with "
                            "`plot_type='spherical'` but it seems the axes is "
                            "not a polar axes.")
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # this raises an exception as well:
                # "AttributeError: 'AxesSubplot' object has no attribute "
                # "'set_theta_zero_location'"
                with self.assertRaises(AttributeError):
                    arrivals.plot_rays(plot_type="spherical", ax=ax,
                                       show=False)
            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[0].message), expected_message)
            self.assertEqual(w[0].category, UserWarning)
        finally:
            plt.close(fig)
        # cartesian pot attempted in polar axes
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        expected_message = ("Axes instance provided for plotting with "
                            "`plot_type='cartesian'` but it seems the axes is "
                            "a polar axes.")
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                arrivals.plot_rays(plot_type="cartesian", ax=ax, show=False)
            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[0].message), expected_message)
            self.assertEqual(w[0].category, UserWarning)
        finally:
            plt.close(fig)

    def test_invalid_plot_option(self):
        """
        Test error message when attempting ray path plots with invalid plot
        type
        """
        arrivals = self.model.get_ray_paths(500, 20, phase_list=['P'])
        # polar plot attempted in cartesian axes
        expected_msg = "Plot type 'spam' is not a valid option."
        with self.assertRaises(ValueError, msg=expected_msg):
            arrivals.plot_rays(plot_type="spam")


def suite():
    return unittest.makeSuite(TauPyPlottingTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
