#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The InvSim test suite.
"""

from obspy.filter import specInv, cosTaper, seisSim, pazToFreqResp
from obspy.filter.seismometer import wood_anderson, wwssn_sp, wwssn_lp, kirnos
import inspect, os, unittest, gzip
import numpy as N
import math as M


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

    def cosTaperPitsa(self,i,n1,n2,n3,n4):
        """
        Cosinus Taper definition of Pitsa
        """
        PI = M.pi
        if (i <= n1) or (i >= n4):
            #check for zero taper
            if (i == n2) or (i == n3):
                return 1.0
            else:
                return 0.0
        elif (i > n1) and (i <= n2):
            temp =  PI * (i-n1)/float(n2-n1+1)
            fact = 0.5 - 0.5*M.cos(temp)
            return abs(fact)
        elif (i >= n3) and (i < n4):
            temp = PI * (n4-i)/float(n4-n3+1)
            fact = 0.5 - 0.5*M.cos(temp)
            return abs(fact)
        else:
            return 1.0
    
    def test_seisSimVsPitsa1(self):
        """
        Test seisSim seismometer simulation against seismometer simulation
        of pitsa --- LE3D seismometer.
        """
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.gz')
        f = gzip.open(file)
        data = N.loadtxt(f)
        f.close()

        # paz of test file
        samp_rate = 200.0
        le3d = {
            'poles' :  [-4.21000 +4.66000j, -4.21000 -4.66000j, -2.105000+0.00000j],
            'zeroes' : [0.0 +0.0j, 0.0 +0.0j, 0.0 +0.0j, 0.0 +0.0j],
            'gain' : 0.4
        }
        
        ##import pylab as pl
        ##pl.figure()
        ##ii = 1
        for instrument in ['None','wood_anderson', 'wwssn_sp', 'wwssn_lp', 'kirnos']:
            # simulate instrument
            exec "datcorr = seisSim(data,samp_rate,le3d,inst_sim=%s,water_level=600.0)" % instrument
            # load pitsa file
            file = os.path.join(self.path, 'rjob_20051006_%s.gz'%instrument)
            f = gzip.open(file)
            data_pitsa = N.loadtxt(f)
            f.close()
            # calculate normalized rms
            rms = N.sqrt(N.sum((datcorr-data_pitsa)**2)/N.sum(data_pitsa**2))
            #print "RMS misfit %15s:"%instrument,rms
            self.assertEqual(rms < 1.e-5, True)
            ##pl.subplot(3,2,ii)
            ##pl.plot(datcorr,'r',data_pitsa,'b--')
            ##pl.title(instrument)
            ##ii += 1
        ##pl.subplot(3,2,6)
        ##pl.plot(data,'k')
        ##pl.title('original data')
        ##pl.savefig("instrument.ps")

    def test_seisSimVsPitsa2(self):
        """
        Test seisSim seismometer simulation against seismometer simulation of pitsa
        --- STS-2 seismometer.
        """
        # load test file
        file = os.path.join(self.path, 'rotz_20081028.gz')
        f = gzip.open(file)
        data = N.loadtxt(f)
        f.close()

        # paz of test file
        samp_rate = 200.0
        sts2 = {
            'poles' :  [-0.03736 -0.03617j, -0.03736 +0.03617j],
            'zeroes' : [0.0 +0.0j, 0.0 +0.0j, 0.0 +0.0j],
            'gain' : 1.5
        }
        
        ##import pylab as pl
        ##pl.figure()
        ##ii = 1
        for instrument in ['None','wood_anderson', 'wwssn_sp', 'wwssn_lp', 'kirnos']:
            # simulate instrument
            exec "datcorr = seisSim(data,samp_rate,sts2,inst_sim=%s,water_level=600.0)" % instrument
            # load pitsa file
            file = os.path.join(self.path, 'rotz_20081028_%s.gz'%instrument)
            f = gzip.open(file)
            data_pitsa = N.loadtxt(f)
            f.close()
            # calculate normalized rms
            rms = N.sqrt(N.sum((datcorr-data_pitsa)**2)/N.sum(data_pitsa**2))
            #print "RMS misfit %15s:"%instrument,rms
            self.assertEqual(rms < 1.e-4, True)
            ##pl.subplot(3,2,ii)
            ##pl.plot(datcorr,'r',data_pitsa,'b--')
            ##pl.title(instrument)
            ##ii += 1
        ##pl.subplot(3,2,6)
        ##pl.plot(data,'k')
        ##pl.title('original data')
        ##pl.savefig("instrument.ps")

    #def test_seisSimVsPitsa2(self):
    #    from obspy.mseed import test as tests_
    #    path = os.path.dirname(inspect.getsourcefile(tests_))
    #    file = os.path.join(path, 'data', 'BW.BGLD..EHE.D.2008.001')
    #    g = Trace()
    #    g.read(file,format='MSEED')
    #    # paz of test file
    #    samp_rate = 200.0

def suite():
    return unittest.makeSuite(InvSimTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
