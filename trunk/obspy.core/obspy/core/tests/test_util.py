# -*- coding: utf-8 -*-

from obspy.core.util import calcVincentyInverse, gps2DistAzimuth, skipIf
import sys
import unittest
import warnings

# checking for geographiclib
try:
    import geographiclib  # @UnusedImport
    HAS_GEOGRAPHICLIB = True
except ImportError:
    HAS_GEOGRAPHICLIB = False


class UtilTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util
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
        self.assertAlmostEquals(res[0], 2204125.9174282863)
        self.assertAlmostEquals(res[1], 90.0)
        self.assertAlmostEquals(res[2], 270.0)
        res = calcVincentyInverse(0, 0, 0, 10)
        self.assertAlmostEquals(res[0], 1113194.9077920639)
        self.assertAlmostEquals(res[1], 90.0)
        self.assertAlmostEquals(res[2], 270.0)
        res = calcVincentyInverse(0, 0, 0, 17)
        self.assertAlmostEquals(res[0], 1892431.3432465086)
        self.assertAlmostEquals(res[1], 90.0)
        self.assertAlmostEquals(res[2], 270.0)
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
        self.assertAlmostEquals(gps2DistAzimuth(15.26804251, 2.93007342,
                                                -14.80522806, -177.2299081),
                                (19951425.048688546, 8.65553241932755,
                                 351.36325485132306))
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
    @skipIf(sys.hexversion < 0x02060000, "Python 2.5.x not supported")
    def test_gps2DistAzimuthBUG150(self):
        """
        Test case for #150: UserWarning will be only raised if geographiclib is
        not installed.
        """
        # this raises UserWarning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            self.assertRaises(UserWarning, gps2DistAzimuth, 0, 0, 0, 180)


def suite():
    return unittest.makeSuite(UtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
