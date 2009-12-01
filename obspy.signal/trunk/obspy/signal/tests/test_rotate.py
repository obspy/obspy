#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Rotate test suite.
"""

from obspy.signal import rotate_NE_RT, gps2DistAzimuth
import inspect, os, unittest, gzip
import numpy as N


class RotateTestCase(unittest.TestCase):
    """
    Test cases for Rotate.
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'data')

    def tearDown(self):
        pass

    def test_rotate_NE_RTVsPitsa(self):
        """
        Test horizontal component rotation against PITSA.
        """
        # load test files
        file = os.path.join(self.path, 'rjob_20051006_n.gz')
        f = gzip.open(file)
        data_n = N.loadtxt(f)
        f.close()
        file = os.path.join(self.path, 'rjob_20051006_e.gz')
        f = gzip.open(file)
        data_e = N.loadtxt(f)
        f.close()
        #test different angles, one from each sector
        for angle in [30, 115, 185, 305]:
            # rotate traces
            datcorr_r, datcorr_t = rotate_NE_RT(data_n, data_e, angle)
            # load pitsa files
            file = os.path.join(self.path, 'rjob_20051006_r_%sdeg.gz' % angle)
            f = gzip.open(file)
            data_pitsa_r = N.loadtxt(f)
            f.close()
            file = os.path.join(self.path, 'rjob_20051006_t_%sdeg.gz' % angle)
            f = gzip.open(file)
            data_pitsa_t = N.loadtxt(f)
            f.close()
            # calculate normalized rms
            rms = N.sqrt(N.sum((datcorr_r - data_pitsa_r) ** 2) / N.sum(data_pitsa_r ** 2))
            rms += N.sqrt(N.sum((datcorr_t - data_pitsa_t) ** 2) / N.sum(data_pitsa_t ** 2))
            rms /= 2.0
            #from pylab import figure,plot,legend,show
            #figure()
            #plot(datcorr_r,label="R ObsPy")
            #plot(data_pitsa_r,label="R PITSA")
            #plot(datcorr_t,label="T ObsPy")
            #plot(data_pitsa_t,label="T PITSA")
            #legend()
            #show()
            #print "RMS misfit:",rms
            self.assertEqual(rms < 1.e-5, True)
            
    def test_gps2DistAzimuth(self):
        """
        Test gps2DistAzimuth() method with test data from Geocentric Datum of Australia.
        (see http://www.icsm.gov.au/gda/gdatm/gdav2.3.pdf)
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
        calc_dist, calc_alpha12, calc_alpha21 = gps2DistAzimuth(lat1, lon1, lat2, lon2)

        #calculate deviations from test data
        dist_err_rel = abs(dist - calc_dist) / dist
        alpha12_err = abs(alpha12 - calc_alpha12)
        alpha21_err = abs(alpha21 - calc_alpha21)

        self.assertEqual(dist_err_rel < 1.e-5, True)
        self.assertEqual(alpha12_err < 1.e-5, True)
        self.assertEqual(alpha21_err < 1.e-5, True)

def suite():
    return unittest.makeSuite(RotateTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
