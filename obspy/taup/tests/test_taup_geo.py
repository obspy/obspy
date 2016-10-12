#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the SeismicPhase class.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy.taup.tau import TauPyModel
from obspy.taup.taup_geo import calc_dist, calc_dist_azi
import obspy.geodetics.base as geodetics


class TaupGeoTestCase(unittest.TestCase):
    """
    Test suite for the SeismicPhase class.
    """
    def setUp(self):
        self.model = TauPyModel('iasp91')

    @unittest.skipIf(not geodetics.GEOGRAPHICLIB_VERSION_AT_LEAST_1_34,
                     'Module geographiclib is not installed or too old.')
    def test_path_geo(self):
        evlat, evlon = 0., 20.
        evdepth = 10.
        stlat, stlon = 0., -80.
        arrivals = self.model.get_ray_paths_geo(evdepth, evlat, evlon, stlat,
                                                stlon)
        for arr in arrivals:
            stlat_path = arr.path['lat'][-1]
            stlon_path = arr.path['lon'][-1]
            self.assertAlmostEqual(stlat, stlat_path, delta=0.1)
            self.assertAlmostEqual(stlon, stlon_path, delta=0.1)


class TaupGeoDistTestCase(unittest.TestCase):
    """
    Test suite for calc_dist and calc_dist_azi in taup_geo.
    """
    def assert_angle_almost_equal(self, first, second, places=7, msg=None,
                                  delta=None):
        """
        Compare two angles (in degrees) for equality

        This method considers numbers close to 359.9999999 to be similar
        to 0.00000001 and supports the same arguments as assertAlmostEqual
        """
        if first > second:
            difference = (second - first) % 360.0
        else:
            difference = (first - second) % 360.0
        self.assertAlmostEqual(difference, 0.0, places=places, msg=msg,
                               delta=delta)

    def test_taup_geo_calc_dist(self):
        """Test for calc_dist"""
        self.assertAlmostEqual(calc_dist(source_latitude_in_deg=20.0,
                                         source_longitude_in_deg=33.0,
                                         receiver_latitude_in_deg=55.0,
                                         receiver_longitude_in_deg=33.0,
                                         radius_of_planet_in_km=6371.0,
                                         flattening_of_planet=0.0), 35.0, 5)
        self.assertAlmostEqual(calc_dist(source_latitude_in_deg=55.0,
                                         source_longitude_in_deg=33.0,
                                         receiver_latitude_in_deg=20.0,
                                         receiver_longitude_in_deg=33.0,
                                         radius_of_planet_in_km=6371.0,
                                         flattening_of_planet=0.0), 35.0, 5)
        self.assertAlmostEqual(calc_dist(source_latitude_in_deg=-20.0,
                                         source_longitude_in_deg=33.0,
                                         receiver_latitude_in_deg=-55.0,
                                         receiver_longitude_in_deg=33.0,
                                         radius_of_planet_in_km=6371.0,
                                         flattening_of_planet=0.0), 35.0, 5)
        self.assertAlmostEqual(calc_dist(source_latitude_in_deg=-20.0,
                                         source_longitude_in_deg=33.0,
                                         receiver_latitude_in_deg=-55.0,
                                         receiver_longitude_in_deg=33.0,
                                         radius_of_planet_in_km=6.371,
                                         flattening_of_planet=0.0), 35.0, 5)

    def test_taup_geo_calc_dist_azi(self):
        """Test for calc_dist"""
        dist, azi, backazi = calc_dist_azi(source_latitude_in_deg=20.0,
                                           source_longitude_in_deg=33.0,
                                           receiver_latitude_in_deg=55.0,
                                           receiver_longitude_in_deg=33.0,
                                           radius_of_planet_in_km=6371.0,
                                           flattening_of_planet=0.0)
        self.assertAlmostEqual(dist, 35.0, 5)
        self.assert_angle_almost_equal(azi, 0.0, 5)
        self.assert_angle_almost_equal(backazi, 180.0, 5)
        dist, azi, backazi = calc_dist_azi(source_latitude_in_deg=55.0,
                                           source_longitude_in_deg=33.0,
                                           receiver_latitude_in_deg=20.0,
                                           receiver_longitude_in_deg=33.0,
                                           radius_of_planet_in_km=6371.0,
                                           flattening_of_planet=0.0)
        self.assertAlmostEqual(dist, 35.0, 5)
        self.assert_angle_almost_equal(azi, 180.0, 5)
        self.assert_angle_almost_equal(backazi, 0.0, 5)
        dist, azi, backazi = calc_dist_azi(source_latitude_in_deg=-20.0,
                                           source_longitude_in_deg=33.0,
                                           receiver_latitude_in_deg=-55.0,
                                           receiver_longitude_in_deg=33.0,
                                           radius_of_planet_in_km=6371.0,
                                           flattening_of_planet=0.0)
        self.assertAlmostEqual(dist, 35.0, 5)
        self.assert_angle_almost_equal(azi, 180.0, 5)
        self.assert_angle_almost_equal(backazi, 0.0, 5)
        dist, azi, backazi = calc_dist_azi(source_latitude_in_deg=-20.0,
                                           source_longitude_in_deg=33.0,
                                           receiver_latitude_in_deg=-55.0,
                                           receiver_longitude_in_deg=33.0,
                                           radius_of_planet_in_km=6.371,
                                           flattening_of_planet=0.0)
        self.assertAlmostEqual(dist, 35.0, 5)
        self.assert_angle_almost_equal(azi, 180.0, 5)
        self.assert_angle_almost_equal(backazi, 0.0, 5)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TaupGeoTestCase, 'test'))
    suite.addTest(unittest.makeSuite(TaupGeoDistTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
