#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the high level obspy.taup.tau interface.
"""
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from obspy.core.util.testing import WarningsCapture
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from obspy.taup import TauPyModel, plot_travel_times


class TestTauPyPlotting:
    """
    TauPy plotting tests.
    """
    image_dir = os.path.join(os.path.dirname(__file__), 'images')

    @pytest.fixture(scope='class')
    def model(self):
        """return a model for testing."""
        return TauPyModel(model="iasp91")

    @staticmethod
    def _test_plot_all(plot_all):
        """Helper function for plotting all phases."""
        mpl.rcParams['lines.markersize'] = 200
        plot_travel_times(10, phase_list=("SSS",), min_degrees=150,
                          max_degrees=200, npoints=4, legend=False,
                          plot_all=plot_all)

    def test_spherical_many_phases(self, model, image_path):
        """
        Spherical plot of the ray paths for many phases for a single
        epicentral distance, but both ways around the globe.
        """
        rays = model.get_ray_paths(500, 140)
        rays.plot_rays(plot_type="spherical", plot_all=True, show=False)
        plt.savefig(image_path)

    def test_spherical_many_phases_buried_station(self, model, image_path):
        """
        Same as test_spherical_many_phases, but this time the receiver is
        buried.
        """
        arrs = model.get_ray_paths(500, 140, receiver_depth_in_km=1000)
        arrs.plot_rays(plot_type="spherical", plot_all=True, show=False)
        plt.savefig(image_path)

    def test_spherical_many_phases_one_way(self, model, image_path):
        """
        Same as test_spherical_many_phases, but this time no phases
        travelling the other way around the globe are plotted.
        """
        rays = model.get_ray_paths(500, 140)
        rays.plot_rays(plot_type="spherical", plot_all=False, show=False)
        plt.savefig(image_path)

    def test_spherical_more_then_360_degrees(self, model, image_path):
        """
        Spherical plot with rays traveling more than 360.0 degrees.
        """
        model.get_ray_paths(0, 10, phase_list=["PPPPPP"]).plot_rays(
            plot_type="spherical", plot_all=True, show=False,
            phase_list=['PPPPPP'])
        plt.savefig(image_path)

    def test_spherical_diff_phases(self, model, image_path):
        """
        Spherical plot of ``diff`` phases.
        """
        model.get_ray_paths(
            700, 140, phase_list=["Pdiff", "Sdiff", "pSdiff", "sSdiff",
                                  "pPdiff", "sPdiff"]).plot_rays(
                                      phase_list=["Pdiff", "Sdiff",
                                                  "pSdiff", "sSdiff",
                                                  "pPdiff", "sPdiff"],
                                      plot_type="spherical", legend=True,
                                      plot_all=True, show=False)
        plt.savefig(image_path)

    def test_cartesian_many_phases(self, model, image_path):
        """
         Cartesian plot of the ray paths for many phases for a single
         epicentral distance, but both ways around the globe.
         """
        model.get_ray_paths(500, 140).plot_rays(plot_type="cartesian",
                                                plot_all=True,
                                                show=False)
        plt.savefig(image_path)

    def test_cartesian_many_phases_buried_station(self, model, image_path):
        """
        Same as test_cartesian_many_phases but this time the receiver is
        buried.
        """
        arrivals = model.get_ray_paths(500, 140, receiver_depth_in_km=1000)
        arrivals.plot_rays(plot_type="cartesian", plot_all=True,
                           show=False)
        plt.savefig(image_path)

    def test_cartesian_many_phases_one_way(self, model, image_path):
        """
        Same as test_cartesian_many_phases but this time no phases
        travelling the other way around the globe are plotted.
        """
        arrivals = model.get_ray_paths(500, 140)
        arrivals.plot_rays(plot_type="cartesian", plot_all=False,
                           show=False)
        plt.savefig(image_path)

    def test_cartesian_many_phases_one_way_depr(self, model, image_path):
        """
        Same as above but checking deprecation warning.
        """
        arrivals = model.get_ray_paths(500, 140)
        # check warning message on deprecated routine
        with WarningsCapture() as w:
            arrivals.plot(plot_type="cartesian", plot_all=False,
                          show=False, legend=False)
            plt.savefig(image_path)
            assert len(w) >= 1
            for w_ in w:
                try:
                    assert str(w_.message) == 'The plot() function is ' \
                        'deprecated. Please use arrivals.plot_rays()'
                    assert w_.category == ObsPyDeprecationWarning
                except AssertionError:
                    continue
                break
            else:
                raise

    def test_plot_travel_times(self, model, image_path):
        """
        Travel time plot for many phases at a single epicentral distance.
        """
        ray_paths = model.get_ray_paths(10, 100, phase_list=['ttbasic', ])
        ray_paths.plot_times(show=False, legend=True)
        plt.savefig(image_path)

    def test_plot_travel_times_convenience_1(self, model, image_path):
        """
        Travel time plot for many phases at multiple epicentral distances,
        convenience function
        """
        plot_travel_times(10, phase_list=("P", "S", "SKS", "PP"),
                          min_degrees=40, max_degrees=60, show=False,
                          legend=True, npoints=4)
        plt.savefig(image_path)

    def test_plot_travel_times_convenience_2(self, model, image_path):
        """
        Same as above but with plot_all == False
        """
        self._test_plot_all(plot_all=False)
        plt.savefig(image_path)

    def test_plot_travel_times_convenience_3(self, model, image_path):
        """
        Now with plot_all == True
        """
        self._test_plot_all(plot_all=True)
        plt.savefig(image_path)

    def test_ray_plot_mismatching_axes_type_warnings(self, model):
        """
        Test warnings when attempting ray path plots in spherical/cartesian
        with bad axes type (polar/not polar).
        """
        arrivals = model.get_ray_paths(500, 20, phase_list=['P'])
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
                with pytest.raises(AttributeError):
                    arrivals.plot_rays(plot_type="spherical", ax=ax,
                                       show=False)
            assert len(w) == 1
            assert str(w[0].message) == expected_message
            assert w[0].category == UserWarning
        finally:
            plt.close(fig)
        # cartesian pot attempted in polar axes
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        expected_message = ("Axes instance provided for plotting with "
                            "`plot_type='cartesian'` but it seems the axes is "
                            "a polar axes.")
        try:
            with WarningsCapture() as w:
                arrivals.plot_rays(plot_type="cartesian", ax=ax, show=False)
            assert len(w) == 1
            assert str(w[0].message) == expected_message
            assert issubclass(w[0].category, Warning)
        finally:
            plt.close(fig)

    def test_invalid_plot_option(self, model):
        """
        Test error message when attempting ray path plots with invalid plot
        type
        """
        arrivals = model.get_ray_paths(500, 20, phase_list=['P'])
        # polar plot attempted in cartesian axes
        with pytest.raises(ValueError):
            arrivals.plot_rays(plot_type="spam")
