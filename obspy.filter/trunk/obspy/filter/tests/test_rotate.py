#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The Rotate test suite.
"""

from obspy.filter import rotate_NE_RT
import inspect, os, random, unittest, filecmp
import numpy as N
import math as M
from pylab import load
import gzip


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
        data_n = load(f)
        f.close()
        file = os.path.join(self.path, 'rjob_20051006_e.gz')
        f = gzip.open(file)
        data_e = load(f)
        f.close()
        #test different angles, one from each sector
        for angle in [30,115,185,305]:
            # rotate traces
            datcorr_r,datcorr_t = rotate_NE_RT(data_n,data_e,angle)
            # load pitsa files
            file = os.path.join(self.path, 'rjob_20051006_r_%sdeg.gz'%angle)
            f = gzip.open(file)
            data_pitsa_r = load(f)
            f.close()
            file = os.path.join(self.path, 'rjob_20051006_t_%sdeg.gz'%angle)
            f = gzip.open(file)
            data_pitsa_t = load(f)
            f.close()
            # calculate normalized rms
            rms = N.sqrt(N.sum((datcorr_r-data_pitsa_r)**2)/N.sum(data_pitsa_r**2))
            rms = rms + N.sqrt(N.sum((datcorr_t-data_pitsa_t)**2)/N.sum(data_pitsa_t**2))
            rms = rms/2.
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

def suite():
    return unittest.makeSuite(RotateTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
