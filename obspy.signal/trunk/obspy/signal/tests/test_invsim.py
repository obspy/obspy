#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The InvSim test suite.
"""

from obspy.signal import seisSim
from obspy.signal.seismometer import PAZ_KIRNOS, PAZ_WOOD_ANDERSON, \
    PAZ_WWSSN_LP, PAZ_WWSSN_SP
import gzip
import inspect
import numpy as N
import os
import unittest


INSTRUMENTS = {
    'None': None,
    'kirnos': PAZ_KIRNOS,
    'wood_anderson': PAZ_WOOD_ANDERSON,
    'wwssn_lp': PAZ_WWSSN_LP,
    'wwssn_sp': PAZ_WWSSN_SP
}


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

    def test_seisSimVsPitsa1(self):
        """
        Test seisSim seismometer simulation against seismometer simulation
        of Pitsa - LE3D seismometer.
        """
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.gz')
        f = gzip.open(file)
        data = N.loadtxt(f)
        f.close()

        # paz of test file
        samp_rate = 200.0
        PAZ_LE3D = {
            'poles': [-4.21000 + 4.66000j,
                      - 4.21000 - 4.66000j,
                      - 2.105000 + 0.00000j],
            'zeros': [0.0 + 0.0j] * 3,
            'gain' : 0.4
        }

        for id, paz in INSTRUMENTS.iteritems():
            # simulate instrument
            datcorr = seisSim(data, samp_rate, PAZ_LE3D, inst_sim=paz,
                              water_level=600.0)
            # load pitsa file
            file = os.path.join(self.path, 'rjob_20051006_%s.gz' % id)
            f = gzip.open(file)
            data_pitsa = N.loadtxt(f)
            f.close()
            # calculate normalized rms
            rms = N.sqrt(N.sum((datcorr - data_pitsa) ** 2) / \
                         N.sum(data_pitsa ** 2))
            #print "RMS misfit %15s:" % id, rms
            self.assertTrue(rms < 1.1e-05)

    def test_seisSimVsPitsa2(self):
        """
        Test seisSim seismometer simulation against seismometer simulation of 
        Pitsa - STS-2 seismometer.
        """
        # load test file
        file = os.path.join(self.path, 'rotz_20081028.gz')
        f = gzip.open(file)
        data = N.loadtxt(f)
        f.close()

        # paz of test file
        samp_rate = 200.0
        PAZ_STS2 = {
            'poles': [-0.03736 - 0.03617j,
                      - 0.03736 + 0.03617j],
            'zeros': [0.0 + 0.0j] * 2,
            'gain' : 1.5
        }

        for id, paz in INSTRUMENTS.iteritems():
            # simulate instrument
            datcorr = seisSim(data, samp_rate, PAZ_STS2, inst_sim=paz,
                              water_level=600.0)
            # load pitsa file
            file = os.path.join(self.path, 'rotz_20081028_%s.gz' % id)
            f = gzip.open(file)
            data_pitsa = N.loadtxt(f)
            f.close()
            # calculate normalized rms
            rms = N.sqrt(N.sum((datcorr - data_pitsa) ** 2) / \
                         N.sum(data_pitsa ** 2))
            #print "RMS misfit %15s:" % id, rms
            self.assertTrue(rms < 1e-04)

    #XXX: Test for really big signal is missing, where the water level is
    # actually acting
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
