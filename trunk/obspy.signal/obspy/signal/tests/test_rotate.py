#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Rotate test suite.
"""

from obspy.signal import rotate_NE_RT, rotate_ZNE_LQT, rotate_LQT_ZNE
import gzip
import numpy as np
import os
import unittest


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

    def test_rotate_ZNE_LQTVsPitsa(self):
        """
        Test LQT component rotation against PITSA. Test back-rotation.
        """
        # load test files
        file = os.path.join(self.path, 'rjob_20051006.gz')
        data_z = np.loadtxt(gzip.open(file))
        file = os.path.join(self.path, 'rjob_20051006_n.gz')
        data_n = np.loadtxt(gzip.open(file))
        file = os.path.join(self.path, 'rjob_20051006_e.gz')
        data_e = np.loadtxt(gzip.open(file))
        # test different backazimuth/incidence combinations
        for ba, inci in ((60, 130), (210, 60)):
            # rotate traces
            data_l, data_q, data_t = rotate_ZNE_LQT(data_z, data_n, data_e,
                                                    ba, inci)
            # rotate traces back to ZNE
            data_back_z, data_back_n, data_back_e = \
                rotate_LQT_ZNE(data_l, data_q, data_t, ba, inci)
            # load pitsa files
            file = os.path.join(self.path, 'rjob_20051006_l_%sba_%sinc.gz'
                                % (ba, inci))
            data_pitsa_l = np.loadtxt(gzip.open(file))
            file = os.path.join(self.path, 'rjob_20051006_q_%sba_%sinc.gz'
                                % (ba, inci))
            data_pitsa_q = np.loadtxt(gzip.open(file))
            file = os.path.join(self.path, 'rjob_20051006_t_%sba_%sinc.gz'
                                % (ba, inci))
            data_pitsa_t = np.loadtxt(gzip.open(file))
            # calculate normalized rms
            rms = np.sqrt(np.sum((data_l - data_pitsa_l) ** 2) /
                          np.sum(data_pitsa_l ** 2))
            rms += np.sqrt(np.sum((data_q - data_pitsa_q) ** 2) /
                          np.sum(data_pitsa_q ** 2))
            rms += np.sqrt(np.sum((data_t - data_pitsa_t) ** 2) /
                          np.sum(data_pitsa_t ** 2))
            rms /= 3.0
            rms2 = np.sqrt(np.sum((data_z - data_back_z) ** 2) /
                          np.sum(data_z ** 2))
            rms2 += np.sqrt(np.sum((data_n - data_back_n) ** 2) /
                          np.sum(data_n ** 2))
            rms2 += np.sqrt(np.sum((data_e - data_back_e) ** 2) /
                          np.sum(data_e ** 2))
            rms2 /= 3.0
            #from matplotlib.pyplot import figure,plot,legend,show, subplot
            #figure()
            #subplot(311)
            #plot(data_l,label="L ObsPy")
            #plot(data_pitsa_l,label="L PITSA")
            #legend()
            #subplot(312)
            #plot(data_q,label="Q ObsPy")
            #plot(data_pitsa_q,label="Q PITSA")
            #legend()
            #subplot(313)
            #plot(data_t,label="T ObsPy")
            #plot(data_pitsa_t,label="T PITSA")
            #legend()
            #show()
            self.assertTrue(rms < 1.0e-5)
            self.assertTrue(rms2 < 1.0e-5)


def suite():
    return unittest.makeSuite(RotateTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
