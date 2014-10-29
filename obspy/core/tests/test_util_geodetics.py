# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.core.util.decorator import skipIf
from obspy.core.util.geodetics import kilometer2degrees, locations2degrees, \
    calcVincentyInverse, gps2DistAzimuth, degrees2kilometers
import math
import unittest
import warnings

# checking for geographiclib
try:
    import geographiclib  # @UnusedImport # NOQA
    HAS_GEOGRAPHICLIB = True
except ImportError:
    HAS_GEOGRAPHICLIB = False


class UtilGeodeticsTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.geodetics
    """
    def test_calcVincentyInverse(self):
        """
        Tests for the Vincenty's Inverse formulae.
        """
        # the following will raise StopIteration exceptions because of two
        # nearly antipodal points
        self.assertRaises(StopIteration, calcVincentyInverse,
                          15.26804251, 2.93007342, -14.80522806, -177.2299081)
        self.assertRaises(StopIteration, calcVincentyInverse,
                          27.3562106, 72.2382356, -27.55995499, -107.78571981)
        self.assertRaises(StopIteration, calcVincentyInverse,
                          27.4675551, 17.28133229, -27.65771704, -162.65420626)
        self.assertRaises(StopIteration, calcVincentyInverse,
                          27.4675551, 17.28133229, -27.65771704, -162.65420626)
        self.assertRaises(StopIteration, calcVincentyInverse, 0, 0, 0, 13)
        # working examples
        res = calcVincentyInverse(0, 0.2, 0, 20)
        self.assertAlmostEqual(res[0], 2204125.9174282863)
        self.assertAlmostEqual(res[1], 90.0)
        self.assertAlmostEqual(res[2], 270.0)
        res = calcVincentyInverse(0, 0, 0, 10)
        self.assertAlmostEqual(res[0], 1113194.9077920639)
        self.assertAlmostEqual(res[1], 90.0)
        self.assertAlmostEqual(res[2], 270.0)
        res = calcVincentyInverse(0, 0, 0, 17)
        self.assertAlmostEqual(res[0], 1892431.3432465086)
        self.assertAlmostEqual(res[1], 90.0)
        self.assertAlmostEqual(res[2], 270.0)
        # out of bounds
        self.assertRaises(ValueError, calcVincentyInverse, 91, 0, 0, 0)
        self.assertRaises(ValueError, calcVincentyInverse, -91, 0, 0, 0)
        self.assertRaises(ValueError, calcVincentyInverse, 0, 0, 91, 0)
        self.assertRaises(ValueError, calcVincentyInverse, 0, 0, -91, 0)

    @skipIf(not HAS_GEOGRAPHICLIB, 'Module geographiclib is not installed')
    def test_gps2DistAzimuthWithGeographiclib(self):
        """
        Testing gps2DistAzimuth function using the module geographiclib.
        """
        # nearly antipodal points
        result = gps2DistAzimuth(15.26804251, 2.93007342, -14.80522806,
                                 -177.2299081)
        self.assertAlmostEqual(result[0], 19951425.048688546)
        self.assertAlmostEqual(result[1], 8.65553241932755)
        self.assertAlmostEqual(result[2], 351.36325485132306)
        # out of bounds
        self.assertRaises(ValueError, gps2DistAzimuth, 91, 0, 0, 0)
        self.assertRaises(ValueError, gps2DistAzimuth, -91, 0, 0, 0)
        self.assertRaises(ValueError, gps2DistAzimuth, 0, 0, 91, 0)
        self.assertRaises(ValueError, gps2DistAzimuth, 0, 0, -91, 0)

    def test_calcVincentyInverse2(self):
        """
        Test calcVincentyInverse() method with test data from Geocentric Datum
        of Australia. (see http://www.icsm.gov.au/gda/gdatm/gdav2.3.pdf)
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
        calc_dist, calc_alpha12, calc_alpha21 = calcVincentyInverse(lat1, lon1,
                                                                    lat2, lon2)

        # calculate deviations from test data
        dist_err_rel = abs(dist - calc_dist) / dist
        alpha12_err = abs(alpha12 - calc_alpha12)
        alpha21_err = abs(alpha21 - calc_alpha21)

        self.assertEqual(dist_err_rel < 1.0e-5, True)
        self.assertEqual(alpha12_err < 1.0e-5, True)
        self.assertEqual(alpha21_err < 1.0e-5, True)

        # calculate result with +- 360 for lon values
        dist, alpha12, alpha21 = calcVincentyInverse(lat1, lon1 + 360,
                                                     lat2, lon2 - 720)
        self.assertAlmostEqual(dist, calc_dist)
        self.assertAlmostEqual(alpha12, calc_alpha12)
        self.assertAlmostEqual(alpha21, calc_alpha21)

    @skipIf(HAS_GEOGRAPHICLIB,
            'Module geographiclib is installed, not using calcVincentyInverse')
    def test_gps2DistAzimuthBUG150(self):
        """
        Test case for #150: UserWarning will be only raised if geographiclib is
        not installed.
        """
        # this raises UserWarning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            self.assertRaises(UserWarning, gps2DistAzimuth, 0, 0, 0, 180)

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
        def assertLoc(lat1, long1, lat2, long2, approx_distance):
            self.assertTrue(abs(math.radians(locations2degrees(
                lat1, long1, lat2, long2)) * 6371 - approx_distance) <= 20)

        # Approximate values from the Great Circle Calculator:
        #   http://williams.best.vwh.net/gccalc.htm

        # Random location.
        assertLoc(36.12, -86.67, 33.94, -118.40, 2893)
        # Test several combinations of quadrants.
        assertLoc(11.11, 22.22, 33.33, 44.44, 3346)
        assertLoc(-11.11, -22.22, -33.33, -44.44, 3346)
        assertLoc(11.11, 22.22, -33.33, -44.44, 8596)
        assertLoc(-11.11, -22.22, 33.33, 44.44, 8596)
        assertLoc(11.11, -22.22, 33.33, -44.44, 3346)
        assertLoc(-11.11, 22.22, 33.33, 44.44, 5454)
        assertLoc(11.11, -22.22, 33.33, 44.44, 7177)
        assertLoc(11.11, 22.22, -33.33, 44.44, 5454)
        assertLoc(11.11, 22.22, 33.33, -44.44, 7177)
        # Test some extreme values.
        assertLoc(90, 0, 0, 0, 10018)
        assertLoc(180, 0, 0, 0, 20004)
        assertLoc(0, 90, 0, 0, 10018)
        assertLoc(0, 180, 0, 0, 20004)
        assertLoc(0, 0, 90, 0, 10018)
        assertLoc(0, 0, 180, 0, 20004)
        assertLoc(0, 0, 0, 90, 10018)
        assertLoc(0, 0, 0, 180, 20004)
        assertLoc(11, 55, 11, 55, 0)

    @skipIf(not HAS_GEOGRAPHICLIB, 'Module geographiclib is not installed')
    def test_issue_375(self):
        """
        Test for #375.
        """
        _, azim, bazim = gps2DistAzimuth(50, 10, 50 + 1, 10 + 1)
        self.assertEqual(round(azim, 0), 32)
        self.assertEqual(round(bazim, 0), 213)
        _, azim, bazim = gps2DistAzimuth(50, 10, 50 + 1, 10 - 1)
        self.assertEqual(round(azim, 0), 328)
        self.assertEqual(round(bazim, 0), 147)
        _, azim, bazim = gps2DistAzimuth(50, 10, 50 - 1, 10 + 1)
        self.assertEqual(round(azim, 0), 147)
        self.assertEqual(round(bazim, 0), 327)
        _, azim, bazim = gps2DistAzimuth(50, 10, 50 - 1, 10 - 1)
        self.assertEqual(round(azim, 0), 213)
        self.assertEqual(round(bazim, 0), 33)


def suite():
    return unittest.makeSuite(UtilGeodeticsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
