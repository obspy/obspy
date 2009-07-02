#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The Filter test suite.
"""

from obspy.filter import bandpass, bandpassZPHSH, bandstop, bandstopZPHSH, lowpass, lowpassZPHSH, highpass, highpassZPHSH, envelope
import inspect, os, random, unittest, filecmp
import numpy as N
import math as M
from pylab import load
import gzip


class FilterTestCase(unittest.TestCase):
    """
    Test cases for Filter.
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'data')
    
    def tearDown(self):
        pass

    def test_bandpassVsPitsa(self):
        """
        Test Butterworth bandpass filter against Butterworth bandpass filter of pitsa.
        """
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.gz')
        f = gzip.open(file)
        data = load(f)
        f.close()

        # parameters for the test
        samp_rate = 200.0
        freq_min = 5
        freq_max = 10
        corners = 4
        
        # filter trace
        datcorr = bandpass(data,freqmin,freqmax,df=samp_rate,corners=corners)
        # load pitsa file
        file = os.path.join(self.path, 'rjob_20051006_bandpass.gz')
        f = gzip.open(file)
        data_pitsa = load(f)
        f.close()
        # calculate normalized rms
        rms = N.sqrt(N.sum((datcorr-data_pitsa)**2)/N.sum(data_pitsa**2))
        #print "RMS misfit %15s:"%instrument,rms
        self.assertEqual(rms < 1.e-5, True)

def suite():
    return unittest.makeSuite(FilterTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
