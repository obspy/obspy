# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import math
import unittest
import warnings
import numpy as np

from obspy.geodetics import (calc_vincenty_inverse, degrees2kilometers,
                             gps2dist_azimuth, kilometer2degrees,
                             locations2degrees)
from obspy.geodetics.base import HAS_GEOGRAPHICLIB


def dms2dec(degs, mins, secs):
    """Converts angle given in degrees, mins and secs to decimal degrees"""
    return (degs + mins / 60.0 + secs / 3600.0)


class UtilGeodeticsTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.geodetics
    """
    def test_calc_vincenty_inverse(self):
        """
        Tests for the Vincenty's Inverse formulae.
        """
        # the following will raise StopIteration exceptions because of two
        # nearly antipodal points
        self.assertRaises(StopIteration, calc_vincenty_inverse,
                          15.26804251, 2.93007342, -14.80522806, -177.2299081)
        self.assertRaises(StopIteration, calc_vincenty_inverse,
                          27.3562106, 72.2382356, -27.55995499, -107.78571981)
        self.assertRaises(StopIteration, calc_vincenty_inverse,
                          27.4675551, 17.28133229, -27.65771704, -162.65420626)
        self.assertRaises(StopIteration, calc_vincenty_inverse,
                          27.4675551, 17.28133229, -27.65771704, -162.65420626)
        self.assertRaises(StopIteration, calc_vincenty_inverse, 0, 0, 0, 13)
        # working examples
        res = calc_vincenty_inverse(0, 0.2, 0, 20)
        self.assertAlmostEqual(res[0], 2204125.9174282863)
        self.assertAlmostEqual(res[1], 90.0)
        self.assertAlmostEqual(res[2], 270.0)
        res = calc_vincenty_inverse(0, 0, 0, 10)
        self.assertAlmostEqual(res[0], 1113194.9077920639)
        self.assertAlmostEqual(res[1], 90.0)
        self.assertAlmostEqual(res[2], 270.0)
        res = calc_vincenty_inverse(0, 0, 0, 17)
        self.assertAlmostEqual(res[0], 1892431.3432465086)
        self.assertAlmostEqual(res[1], 90.0)
        self.assertAlmostEqual(res[2], 270.0)
        # out of bounds
        self.assertRaises(ValueError, calc_vincenty_inverse, 91, 0, 0, 0)
        self.assertRaises(ValueError, calc_vincenty_inverse, -91, 0, 0, 0)
        self.assertRaises(ValueError, calc_vincenty_inverse, 0, 0, 91, 0)
        self.assertRaises(ValueError, calc_vincenty_inverse, 0, 0, -91, 0)

    @unittest.skipIf(not HAS_GEOGRAPHICLIB, 'Module geographiclib is not '
                                            'installed')
    def test_gps_2_dist_azimuth_with_geographiclib(self):
        """
        Testing gps2dist_azimuth function using the module geographiclib.
        """
        # nearly antipodal points
        result = gps2dist_azimuth(15.26804251, 2.93007342, -14.80522806,
                                  -177.2299081)
        self.assertAlmostEqual(result[0], 19951425.048688546)
        self.assertAlmostEqual(result[1], 8.65553241932755)
        self.assertAlmostEqual(result[2], 351.36325485132306)
        # out of bounds
        self.assertRaises(ValueError, gps2dist_azimuth, 91, 0, 0, 0)
        self.assertRaises(ValueError, gps2dist_azimuth, -91, 0, 0, 0)
        self.assertRaises(ValueError, gps2dist_azimuth, 0, 0, 91, 0)
        self.assertRaises(ValueError, gps2dist_azimuth, 0, 0, -91, 0)

    def test_calc_vincenty_inverse_2(self):
        """
        Test calc_vincenty_inverse() method with test data from Geocentric
        Datum of Australia. (see http://www.icsm.gov.au/gda/gdatm/gdav2.3.pdf)
        """
        # test data:
        # Point 1: Flinders Peak, Point 2: Buninyong
        lat1 = -(37 + (57 / 60.) + (3.72030 / 3600.))
        lon1 = 144 + (25 / 60.) + (29.52440 / 3600.)
        lat2 = -(37 + (39 / 60.) + (10.15610 / 3600.))
        lon2 = 143 + (55 / 60.) + (35.38390 / 3600.)
        dist = 54972.271
        alpha12 = 306 + (52 / 60.) + (5.37 / 3600.)
        alpha21 = 127 + (10 / 60.) + (25.07 / 3600.)

        # calculate result
        calc_dist, calc_alpha12, calc_alpha21 = calc_vincenty_inverse(
            lat1, lon1, lat2, lon2)

        # calculate deviations from test data
        dist_err_rel = abs(dist - calc_dist) / dist
        alpha12_err = abs(alpha12 - calc_alpha12)
        alpha21_err = abs(alpha21 - calc_alpha21)

        self.assertEqual(dist_err_rel < 1.0e-5, True)
        self.assertEqual(alpha12_err < 1.0e-5, True)
        self.assertEqual(alpha21_err < 1.0e-5, True)

        # calculate result with +- 360 for lon values
        dist, alpha12, alpha21 = calc_vincenty_inverse(
            lat1, lon1 + 360, lat2, lon2 - 720)
        self.assertAlmostEqual(dist, calc_dist)
        self.assertAlmostEqual(alpha12, calc_alpha12)
        self.assertAlmostEqual(alpha21, calc_alpha21)

    def test_calc_vincenty_inverse_tabulated(self):
        """ Tabulated results for Vincenty Inverse

        Table II of Vincenty's paper (T. Vincenty 1975, "Direct and inverse
        solutions of geodesics on the ellipsoid with application of nested
        equations" Survey Review XXII pp.88-93) has five test examples for
        the forward and inverse problem (with results rounded to 0.00001
        seconds of arc and 1 mm). The inverse versions of these are implemented
        here. Note the non-standard (old) ellipsoid usage. Here we test that
        we match these examples for the inverse problem. """
        # Row "A"
        # NB: for this case there seems to be a typo in
        #     the tabulated data. Tabulated data is commented
        #     out and values from geographiclib are used in their place
        # dist = 14110526.170
        dist = 14039003.954192352
        # azi1 = dms2dec(96.0, 36.0, 8.79960)
        azi1 = 95.88145755849257
        # azi2 = dms2dec(137.0, 52.0, 22.01454)
        azi2 = 138.30481836546775
        bazi = azi2 + 180.0
        lat1 = dms2dec(55.0, 45.0, 0.0)
        lat2 = dms2dec(-33.0, 26.0, 0.0)
        lon2 = dms2dec(108.0, 13.0, 0.0)
        a = 6377397.155
        f = 1.0 / 299.1528128
        calc_dist, calc_azi1, calc_bazi = calc_vincenty_inverse(
            lat1, 0.0, lat2, lon2, a, f)
        self.assertAlmostEqual(dist, calc_dist, 2)
        self.assertAlmostEqual(azi1, calc_azi1, 5)
        self.assertAlmostEqual(bazi, calc_bazi, 5)

        # Row "B"
        dist = 4085966.703
        azi1 = dms2dec(95.0, 27.0, 59.63089)
        azi2 = dms2dec(118, 5.0, 58.96161)
        bazi = azi2 + 180.0
        lat1 = dms2dec(37.0, 19.0, 54.95367)
        lat2 = dms2dec(26.0, 7.0, 42.83946)
        lon2 = dms2dec(41.0, 28.0, 35.50729)
        a = 6378388.000
        f = 1.0 / 297.0
        calc_dist, calc_azi1, calc_bazi = calc_vincenty_inverse(
            lat1, 0.0, lat2, lon2, a, f)
        self.assertAlmostEqual(dist, calc_dist, 2)
        self.assertAlmostEqual(azi1, calc_azi1, 5)
        self.assertAlmostEqual(bazi, calc_bazi, 5)

        # Row "C"
        dist = 8084823.839
        azi1 = dms2dec(15.0, 44.0, 23.74850)
        azi2 = dms2dec(144.0, 55.0, 39.92147)
        bazi = azi2 + 180.0
        lat1 = dms2dec(35.0, 16.0, 11.24862)
        lat2 = dms2dec(67.0, 22.0, 14.77638)
        lon2 = dms2dec(137.0, 47.0, 28.31435)
        a = 6378388.000
        f = 1.0 / 297.0
        calc_dist, calc_azi1, calc_bazi = calc_vincenty_inverse(
            lat1, 0.0, lat2, lon2, a, f)
        self.assertAlmostEqual(dist, calc_dist, 2)
        self.assertAlmostEqual(azi1, calc_azi1, 5)
        self.assertAlmostEqual(bazi, calc_bazi, 5)

    @unittest.skipIf(HAS_GEOGRAPHICLIB, 'Module geographiclib is installed, '
                                        'not using calc_vincenty_inverse')
    def test_gps_2_dist_azimuth_bug150(self):
        """
        Test case for #150: UserWarning will be only raised if geographiclib is
        not installed.
        """
        # this raises UserWarning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            self.assertRaises(UserWarning, gps2dist_azimuth, 0, 0, 0, 180)

    def test_kilometer2degrees(self):
        """
        Simple test of the convenience function.
        """
        # Test if it works.
        self.assertEqual(kilometer2degrees(111.19492664455873, radius=6371),
                         1.0)
        # Test if setting the radius actually does something. Round to avoid
        # some precision problems on different machines.
        self.assertEqual(round(kilometer2degrees(111.19492664455873,
                         radius=6381), 5), round(0.99843284751606332, 5))

    def test_degrees2kilometers(self):
        """
        """
        # Test if it works.
        self.assertEqual(degrees2kilometers(1.0, radius=6371),
                         111.19492664455873)
        # Test if setting the radius actually does something. Round to avoid
        # some precision problems on different machines.
        self.assertEqual(round(degrees2kilometers(1.0, radius=6381), 5),
                         round(111.36945956975816, 5))

    def test_locations2degrees(self):
        """
        Test the location 2 degree conversion.
        """
        # Inline method to avoid messy code.
        def assert_loc(lat1, long1, lat2, long2, approx_distance):
            self.assertTrue(abs(math.radians(locations2degrees(
                lat1, long1, lat2, long2)) * 6371 - approx_distance) <= 20)

        # Approximate values from the Great Circle Calculator:
        #   http://williams.best.vwh.net/gccalc.htm

        # Random location.
        assert_loc(36.12, -86.67, 33.94, -118.40, 2893)
        # Test several combinations of quadrants.
        assert_loc(11.11, 22.22, 33.33, 44.44, 3346)
        assert_loc(-11.11, -22.22, -33.33, -44.44, 3346)
        assert_loc(11.11, 22.22, -33.33, -44.44, 8596)
        assert_loc(-11.11, -22.22, 33.33, 44.44, 8596)
        assert_loc(11.11, -22.22, 33.33, -44.44, 3346)
        assert_loc(-11.11, 22.22, 33.33, 44.44, 5454)
        assert_loc(11.11, -22.22, 33.33, 44.44, 7177)
        assert_loc(11.11, 22.22, -33.33, 44.44, 5454)
        assert_loc(11.11, 22.22, 33.33, -44.44, 7177)
        # Test some extreme values.
        assert_loc(90, 0, 0, 0, 10018)
        assert_loc(180, 0, 0, 0, 20004)
        assert_loc(0, 90, 0, 0, 10018)
        assert_loc(0, 180, 0, 0, 20004)
        assert_loc(0, 0, 90, 0, 10018)
        assert_loc(0, 0, 180, 0, 20004)
        assert_loc(0, 0, 0, 90, 10018)
        assert_loc(0, 0, 0, 180, 20004)
        assert_loc(11, 55, 11, 55, 0)

        # test numpy inputs:
        # Inline method to avoid messy code.
        def assert_loc_np(lat1, long1, lat2, long2,
                          approx_distance, expected_output_len):
            loc2deg = locations2degrees(np.array(lat1),
                                        np.array(long1),
                                        np.array(lat2),
                                        np.array(long2))
            self.assertTrue((np.abs(np.radians(loc2deg) * 6371 -
                                    approx_distance) <= 20).all())
            self.assertTrue(np.isscalar(loc2deg)
                            if expected_output_len == 0 else
                            len(loc2deg) == expected_output_len)

        # Test just with random location (combining scalars and arrays).
        assert_loc_np(36.12, -86.67, 33.94, -118.40, 2893, 0)
        assert_loc_np([36.12, 36.12], -86.67, 33.94, -118.40,
                      2893, 2)
        assert_loc_np(36.12, [-86.67, -86.67], 33.94, -118.40,
                      2893, 2)
        assert_loc_np(36.12, -86.67, [33.94, 33.94], -118.40,
                      2893, 2)
        assert_loc_np(36.12, -86.67, 33.94, [-118.40, -118.40],
                      2893, 2)
        assert_loc_np([36.12, 36.12], [-86.67, -86.67], 33.94, -118.40,
                      2893, 2)
        assert_loc_np([36.12, 36.12], -86.67, [33.94, 33.94], -118.40,
                      2893, 2)
        assert_loc_np([36.12, 36.12], -86.67, 33.94, [-118.40, -118.40],
                      2893, 2)
        assert_loc_np([36.12, 36.12], [-86.67, -86.67], [33.94, 33.94],
                      -118.40, 2893, 2)
        assert_loc_np([36.12, 36.12], -86.67, [33.94, 33.94],
                      [-118.40, -118.40], 2893, 2)
        assert_loc_np(36.12, [-86.67, -86.67], [33.94, 33.94],
                      [-118.40, -118.40], 2893, 2)
        assert_loc_np([36.12, 36.12], [-86.67, -86.67], [33.94, 33.94],
                      [-118.40, -118.40], 2893, 2)

        # test numpy broadcasting (bad shapes)
        with self.assertRaises(ValueError):
            locations2degrees(1, 2, [3, 4], [5, 6, 7])

    @unittest.skipIf(not HAS_GEOGRAPHICLIB, 'Module geographiclib is not '
                                            'installed')
    def test_issue_375(self):
        """
        Test for #375.
        """
        _, azim, bazim = gps2dist_azimuth(50, 10, 50 + 1, 10 + 1)
        self.assertEqual(round(azim, 0), 32)
        self.assertEqual(round(bazim, 0), 213)
        _, azim, bazim = gps2dist_azimuth(50, 10, 50 + 1, 10 - 1)
        self.assertEqual(round(azim, 0), 328)
        self.assertEqual(round(bazim, 0), 147)
        _, azim, bazim = gps2dist_azimuth(50, 10, 50 - 1, 10 + 1)
        self.assertEqual(round(azim, 0), 147)
        self.assertEqual(round(bazim, 0), 327)
        _, azim, bazim = gps2dist_azimuth(50, 10, 50 - 1, 10 - 1)
        self.assertEqual(round(azim, 0), 213)
        self.assertEqual(round(bazim, 0), 33)


def suite():
    return unittest.makeSuite(UtilGeodeticsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
