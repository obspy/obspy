#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The InvSim test suite.
"""

from obspy.filter import specInv, pazToFreqResp, cosTaper
import inspect, os, random, unittest, filecmp
import numpy as N
from pylab import load
import gzip


class InvSimTestCase(unittest.TestCase):
    """
    Test cases for InvSim.
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'data')
    
    def tearDown(self):
        pass
    
    def test_InvSimVsPitsa(self):
        """
        Compares instrument deconvolution of invsim versus pitsa.
        """
        nfft = 4096
        t_samp = 1/200.0
        poles = [-4.21000 +4.66000j, -4.21000 -4.66000j, -2.105000+0.00000j]
        zeroes = [0.0 +0.0j, 0.0 +0.0j, 0.0 +0.0j, 0.0 +0.0j]
        scale_fac = 0.4

        #
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.asc.gz')
        f = gzip.open(file)
        data = load(f)
        f.close()
        #
        ndat = len(data)
        data = N.concatenate((data,N.zeros(nfft-ndat)))
        # In the following line there is index mistake in Pitsa (start at
        # 0!), need to reimplement it to be able to compare
        data[1:ndat+1] = data[1:ndat+1] * cosTaper(ndat,0.05)
        freq_response = pazToFreqResp(poles,zeroes,scale_fac,t_samp,nfft)
        found = specInv(freq_response,600,nfft)
        spec = N.fft.rfft(data)
        spec2 = N.conj(freq_response) * spec
        data2 = N.fft.irfft(spec2)[0:ndat]
        #
        # linear detrend, 
        x1=data2[0]
        x2=data2[-1]
        # Again mistake in pitsa, it should be ndat-1. Need to
        # reimplement it to be able to compare
        data2 -= ( x1 + N.arange(ndat)*(x2-x1) / float(ndat) )
        #
        # load pitsa file
        file = os.path.join(self.path, 'rjob_20051006_corr.asc.gz')
        f = gzip.open(file)
        data_pitsa = load(f)
        f.close()
        #
        rms = N.sqrt(N.sum((data2-data_pitsa)**2)/ndat)
        self.assertEqual(rms < 0.02, True)
        N.testing.assert_array_almost_equal(data2,data_pitsa,decimal=1)
        #print "RMS misfit:",rms
        #print "RMS orig:   0.0179758634012"
    

def suite():
    return unittest.makeSuite(InvSimTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
