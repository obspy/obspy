#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the SeismicPhase class.
"""
import numpy as np
import pytest

from obspy.taup.tau import TauPyModel
from obspy.taup.taup_geo import calc_dist, calc_dist_azi
import obspy.geodetics.base as geodetics


class TestTaupGeo:
    """
    Test suite for the SeismicPhase class.
    """
    @pytest.fixture(scope='class')
    def model(self):
        return TauPyModel('iasp91')

    @pytest.mark.skipif(
        not geodetics.GEOGRAPHICLIB_VERSION_AT_LEAST_1_34,
        reason='Module geographiclib is not installed or too old.')
    def test_path_geo(self, model):
        evlat, evlon = 0., 20.
        evdepth = 10.
        stlat, stlon = 0., -80.
        arrivals = model.get_ray_paths_geo(evdepth, evlat, evlon, stlat,
                                           stlon)
        for arr in arrivals:
            stlat_path = arr.path['lat'][-1]
            stlon_path = arr.path['lon'][-1]
            assert abs(stlat-stlat_path) < 0.1
            assert abs(stlon-stlon_path) < 0.1

    @pytest.mark.skipif(
        not geodetics.GEOGRAPHICLIB_VERSION_AT_LEAST_1_34,
        reason='Module geographiclib is not installed or too old.')
    def test_path_resampling(self, model):
        """
        Test resampling of paths in regions that are pretty much straight in
        geographical coordinates and therefore only coarsely sampled by taup
        """
        kwargs = dict(source_depth_in_km=50, source_latitude_in_deg=0.,
                      source_longitude_in_deg=0., receiver_latitude_in_deg=0.,
                      receiver_longitude_in_deg=150, phase_list=('Pdiff', ))
        default = model.get_ray_paths_geo(resample=False, **kwargs)[0].path
        resampled = model.get_ray_paths_geo(resample=True, **kwargs)[0].path
        assert len(default) == 397
        assert len(resampled) == 416
        # start and end of path are unchanged
        expected = [
            (254.3313, 0., 0.00000000e+00, 50., 0., 0.),
            (254.3313, 0.122393, 5.03916e-05, 50.931307, 0., 0.00288723),
            (254.3313, 0.820524, 3.38136e-04, 56.243394, 0., 0.01937378),
            (254.3313, 2.217176, 9.15375e-04, 66.870045, 0., 0.05244716),
            (254.3313, 3.614348, 1.49496e-03, 77.5, 0., 0.08565492)]
        np.testing.assert_allclose(default[:5].tolist(), expected, rtol=1e-5)
        np.testing.assert_allclose(resampled[:5].tolist(), expected, rtol=1e-5)
        expected = [(254.33137581441554, 1040.4579085173295, 2.617806788150977,
                     5.004051595465171, 0.0, 149.98928054174857),
                    (254.33137581441554, 1040.9008781834862,
                     2.6179002698779987, 2.5047268613761844, 0.0,
                     149.99463665016856),
                    (254.33137581441554, 1041.1223593199215,
                     2.6179469824519237, 1.2550644943312363, 0.0,
                     149.99731308350465),
                    (254.33137581441554, 1041.3438379939714, 2.617993676181489,
                     0.005402127286288305, 0.0, 149.99998843713843),
                    (254.33137581441554, 1041.3447954120252, 2.617993877991496,
                     0.0, 0.0, 150.0000000000001)]
        np.testing.assert_allclose(default[-5:].tolist(), expected, rtol=1e-5)
        np.testing.assert_allclose(resampled[-5:].tolist(), expected,
                                   rtol=1e-5)
        # in the middle the paths differ
        expected_resampled = [
            (254.33137581441554, 602.8221186084121, 94.33432576321066, 2889.0,
             0.0, 94.33432576321066),
            (254.33137581441554, 617.1726077606227, 97.56720436396148, 2889.0,
             0.0, 97.56720436396148),
            (254.33137581441554, 631.5230969128334, 1.759293334017677, 2889.0,
             0.0, 100.80008296471232),
            (254.33137581441554, 643.4514604207692, 103.4785176520484,
             2880.9095668072578, 0.0, 103.4785176520484),
            (254.33137581441554, 655.379823928705, 1.8527883422049563,
             2872.8191336145155, 0.0, 106.15695233938449),
            (254.33137581441554, 664.4057532502989, 108.1520313009955,
             2856.0745668072577, 0.0, 108.1520313009955),
            (254.33137581441554, 673.4316825718927, 1.9224297356397198,
             2839.33, 0.0, 110.14711026260652),
            (254.33137581441554, 688.0227541830799, 1.9774741001400498,
             2799.2895131996584, 0.0, 113.30092003445516),
            (254.33137581441554, 691.0197687282376, 1.9886081210291868,
             2789.67, 0.0, 113.93885244041321),
            (254.33137581441554, 692.3359202082614, 1.9934773547795543,
             2785.297917906318, 0.0, 114.21783898377193)]
        expected_default = [
            (254.33137581441554, 740.555911561264, 2.160089140534382,
             2560.645140025988, 0.0, 123.76399112466144),
            (254.33137581441554, 743.6578548789879, 2.1698423363329122, 2542.0,
             0.0, 124.32280808068195),
            (254.33137581441554, 751.5598436136655, 2.194109696034333, 2492.5,
             0.0, 125.71322537149922),
            (254.33137581441554, 752.3706892487971, 2.196552592897761,
             2487.2632826174126, 0.0, 125.8531930515594),
            (254.33137581441554, 753.7391549563343, 2.200655453715524,
             2478.3607118178234, 0.0, 126.08826966034681),
            (254.33137581441554, 759.0587464034165, 2.2163649120816844, 2443.0,
             0.0, 126.98835532316427),
            (254.33137581441554, 763.4244896464312, 2.2289727423907655,
             2413.1126874221954, 0.0, 127.71073078869175),
            (254.33137581441554, 766.2313256113371, 2.2369430801261303, 2393.5,
             0.0, 128.167397502222),
            (254.33137581441554, 773.1340678380197, 2.256093920089646, 2344.0,
             0.0, 129.26465980626193),
            (254.33137581441554, 773.9286888315412, 2.258257579541221,
             2338.1899362389727, 0.0, 129.38862836114077)]
        np.testing.assert_allclose(default[205:215].tolist(), expected_default,
                                   rtol=1e-5)
        np.testing.assert_allclose(resampled[205:215].tolist(),
                                   expected_resampled, rtol=1e-5)


class TestTaupGeoDist:
    """
    Test suite for calc_dist and calc_dist_azi in taup_geo.
    """
    def assert_angle_almost_equal(self, first, second, places=7, msg=None,
                                  delta=1e-08):
        """
        Compare two angles (in degrees) for equality

        This method considers numbers close to 359.9999999 to be similar
        to 0.00000001 and supports the same arguments as assertAlmostEqual
        """
        if first > second:
            difference = (second - first) % 360.0
        else:
            difference = (first - second) % 360.0
        # Pre-pytest line was:
        # self.assertAlmostEqual(difference, 0.0, places=places, msg=msg,
        #                        delta=delta)
        # this doesn't translate exactly but should be good enough
        assert np.isclose(difference, 0.0, atol=delta)

    def test_taup_geo_calc_dist_1(self):
        """Test for calc_dist"""
        dist = calc_dist(source_latitude_in_deg=20.0,
                         source_longitude_in_deg=33.0,
                         receiver_latitude_in_deg=55.0,
                         receiver_longitude_in_deg=33.0,
                         radius_of_planet_in_km=6371.0,
                         flattening_of_planet=0.0)
        assert round(abs(dist) - 35.0, 5) == 0.0

    def test_taup_geo_calc_dist_2(self):
        """Test for calc_dist"""
        dist = calc_dist(source_latitude_in_deg=55.0,
                         source_longitude_in_deg=33.0,
                         receiver_latitude_in_deg=20.0,
                         receiver_longitude_in_deg=33.0,
                         radius_of_planet_in_km=6371.0,
                         flattening_of_planet=0.0)
        assert round(abs(dist - 35.0), 5) == 0.0

    def test_taup_geo_calc_dist_3(self):
        """Test for calc_dist"""
        dist = calc_dist(source_latitude_in_deg=-20.0,
                         source_longitude_in_deg=33.0,
                         receiver_latitude_in_deg=-55.0,
                         receiver_longitude_in_deg=33.0,
                         radius_of_planet_in_km=6371.0,
                         flattening_of_planet=0.0)
        assert round(abs(dist - 35.0), 5) == 0

    def test_taup_geo_calc_dist_4(self):
        """Test for calc_dist"""
        dist = calc_dist(source_latitude_in_deg=-20.0,
                         source_longitude_in_deg=33.0,
                         receiver_latitude_in_deg=-55.0,
                         receiver_longitude_in_deg=33.0,
                         radius_of_planet_in_km=6.371,
                         flattening_of_planet=0.0)
        assert round(abs(dist-35.0), 5) == 0

    def test_taup_geo_calc_dist_azi(self):
        """Test for calc_dist"""
        dist, azi, backazi = calc_dist_azi(
            source_latitude_in_deg=20.0,
            source_longitude_in_deg=33.0,
            receiver_latitude_in_deg=55.0,
            receiver_longitude_in_deg=33.0,
            radius_of_planet_in_km=6371.0,
            flattening_of_planet=0.0)
        assert round(abs(dist-35.0), 5) == 0
        self.assert_angle_almost_equal(azi, 0.0, 5)
        self.assert_angle_almost_equal(backazi, 180.0, 5)
        dist, azi, backazi = calc_dist_azi(
            source_latitude_in_deg=55.0,
            source_longitude_in_deg=33.0,
            receiver_latitude_in_deg=20.0,
            receiver_longitude_in_deg=33.0,
            radius_of_planet_in_km=6371.0,
            flattening_of_planet=0.0)
        assert round(abs(dist-35.0), 5) == 0
        self.assert_angle_almost_equal(azi, 180.0, 5)
        self.assert_angle_almost_equal(backazi, 0.0, 5)
        dist, azi, backazi = calc_dist_azi(
            source_latitude_in_deg=-20.0,
            source_longitude_in_deg=33.0,
            receiver_latitude_in_deg=-55.0,
            receiver_longitude_in_deg=33.0,
            radius_of_planet_in_km=6371.0,
            flattening_of_planet=0.0)
        assert round(abs(dist-35.0), 5) == 0
        self.assert_angle_almost_equal(azi, 180.0, 5)
        self.assert_angle_almost_equal(backazi, 0.0, 5)
        dist, azi, backazi = calc_dist_azi(
            source_latitude_in_deg=-20.0,
            source_longitude_in_deg=33.0,
            receiver_latitude_in_deg=-55.0,
            receiver_longitude_in_deg=33.0,
            radius_of_planet_in_km=6.371,
            flattening_of_planet=0.0)
        assert round(abs(dist-35.0), 5) == 0
        self.assert_angle_almost_equal(azi, 180.0, 5)
        self.assert_angle_almost_equal(backazi, 0.0, 5)
