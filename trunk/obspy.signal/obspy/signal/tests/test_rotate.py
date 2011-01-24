#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Rotate test suite.
"""

from obspy.signal import rotate_NE_RT, gps2DistAzimuth, rotate_ZNE_LQT, \
        rotate_LQT_ZNE
import os
import unittest
import gzip
import numpy as np


class RotateTestCase(unittest.TestCase):
    """
    Test cases for Rotate.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_rotate_NE_RTVsPitsa(self):
        """
        Test horizontal component rotation against PITSA.
        """
        # load test files
        file = os.path.join(self.path, 'rjob_20051006_n.gz')
        f = gzip.open(file)
        data_n = np.loadtxt(f)
        f.close()
        file = os.path.join(self.path, 'rjob_20051006_e.gz')
        f = gzip.open(file)
        data_e = np.loadtxt(f)
        f.close()
        #test different angles, one from each sector
        for angle in [30, 115, 185, 305]:
            # rotate traces
            datcorr_r, datcorr_t = rotate_NE_RT(data_n, data_e, angle)
            # load pitsa files
            file = os.path.join(self.path, 'rjob_20051006_r_%sdeg.gz' % angle)
            f = gzip.open(file)
            data_pitsa_r = np.loadtxt(f)
            f.close()
            file = os.path.join(self.path, 'rjob_20051006_t_%sdeg.gz' % angle)
            f = gzip.open(file)
            data_pitsa_t = np.loadtxt(f)
            f.close()
            # calculate normalized rms
            rms = np.sqrt(np.sum((datcorr_r - data_pitsa_r) ** 2) /
                          np.sum(data_pitsa_r ** 2))
            rms += np.sqrt(np.sum((datcorr_t - data_pitsa_t) ** 2) /
                           np.sum(data_pitsa_t ** 2))
            rms /= 2.0
            #from pylab import figure,plot,legend,show
            #figure()
            #plot(datcorr_r,label="R ObsPy")
            #plot(data_pitsa_r,label="R PITSA")
            #plot(datcorr_t,label="T ObsPy")
            #plot(data_pitsa_t,label="T PITSA")
            #legend()
            #show()
            self.assertEqual(rms < 1.0e-5, True)

    def test_rotate_ZNE_LQT(self):
        """
        Test 3-component rotation with some simple examples.
        """
        c = 0.5 * 3 ** 0.5
        z, n, e = np.array([1., 2.4]), np.array([0.3, -0.7]), np.array([-0.2, 0.])
        ba_inc_l_q_t = ((180, 0, z, n, -e),
                        (0, 0, z, -n, e),
                        (180, 90, n, -z, -e),
                        (270, 0, z, e, n),
                        (30, 30, z * c - n * c / 2 - e / 4,
                         - z / 2 - n * c ** 2 - e * c / 2, -n / 2 + e * c),
                        (180, 180, -z, -n, -e),
                        (270, 270, -e, z, n))
        for ba, inc, l, q, t in ba_inc_l_q_t:
            l2, q2, t2 = rotate_ZNE_LQT(z, n, e, ba, inc)
            z2, n2, e2 = rotate_LQT_ZNE(l, q, t, ba, inc)
            # calculate normalized rms
            rms = np.sqrt(np.sum((l2 - l) ** 2) / np.sum(l ** 2))
            rms += np.sqrt(np.sum((q2 - q) ** 2) / np.sum(q ** 2))
            rms += np.sqrt(np.sum((t2 - t) ** 2) / np.sum(t ** 2))
            rms /= 3.
            rms2 = np.sqrt(np.sum((z2 - z) ** 2) / np.sum(z ** 2))
            rms2 += np.sqrt(np.sum((n2 - n) ** 2) / np.sum(n ** 2))
            rms2 += np.sqrt(np.sum((e2 - e) ** 2) / np.sum(e ** 2))
            rms2 /= 3.
            self.assertEqual(rms < 1.0e-5, True)
            self.assertEqual(rms2 < 1.0e-5, True)


    def test_gps2DistAzimuth(self):
        """
        Test gps2DistAzimuth() method with test data from Geocentric Datum of 
        Australia. (see http://www.icsm.gov.au/gda/gdatm/gdav2.3.pdf)
        """
        # test data:
        #Point 1: Flinders Peak, Point 2: Buninyong
        lat1 = -(37 + (57 / 60.) + (3.72030 / 3600.))
        lon1 = 144 + (25 / 60.) + (29.52440 / 3600.)
        lat2 = -(37 + (39 / 60.) + (10.15610 / 3600.))
        lon2 = 143 + (55 / 60.) + (35.38390 / 3600.)
        dist = 54972.271
        alpha12 = 306 + (52 / 60.) + (5.37 / 3600.)
        alpha21 = 127 + (10 / 60.) + (25.07 / 3600.)

        #calculate result
        calc_dist, calc_alpha12, calc_alpha21 = gps2DistAzimuth(lat1, lon1,
                                                                lat2, lon2)

        #calculate deviations from test data
        dist_err_rel = abs(dist - calc_dist) / dist
        alpha12_err = abs(alpha12 - calc_alpha12)
        alpha21_err = abs(alpha21 - calc_alpha21)

        self.assertEqual(dist_err_rel < 1.0e-5, True)
        self.assertEqual(alpha12_err < 1.0e-5, True)
        self.assertEqual(alpha21_err < 1.0e-5, True)

        #calculate result with +- 360 for lon values
        dist, alpha12, alpha21 = gps2DistAzimuth(lat1, lon1 + 360,
                                                 lat2, lon2 - 720)
        self.assertAlmostEqual(dist, calc_dist)
        self.assertAlmostEqual(alpha12, calc_alpha12)
        self.assertAlmostEqual(alpha21, calc_alpha21)

    def test_gps2DistAzimuthBUG150(self):
        """
        Test case for #150
        """
        res = gps2DistAzimuth(0, 0, 0, 180)
        self.assertEqual(res, (20004314.5, 0.0, 0.0))

def suite():
    return unittest.makeSuite(RotateTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
