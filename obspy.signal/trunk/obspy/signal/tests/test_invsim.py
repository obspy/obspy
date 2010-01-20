#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The InvSim test suite.
"""

from obspy.core import Stream, Trace, UTCDateTime
from obspy.signal import seisSim, cornFreq2Paz, lowpass
from obspy.signal.seismometer import PAZ_KIRNOS, PAZ_WOOD_ANDERSON, \
    PAZ_WWSSN_LP, PAZ_WWSSN_SP
import gzip
import inspect
import numpy as np
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
        data = np.loadtxt(f)
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
            data_pitsa = np.loadtxt(f)
            f.close()
            # calculate normalized rms
            rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) / \
                         np.sum(data_pitsa ** 2))
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
        data = np.loadtxt(f)
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
            data_pitsa = np.loadtxt(f)
            f.close()
            # calculate normalized rms
            rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) / \
                         np.sum(data_pitsa ** 2))
            #print "RMS misfit %15s:" % id, rms
            self.assertTrue(rms < 1e-04)


    def test_SRL(self):
        """
        Tests if example in ObsPy paper submitted to the Electronic
        Seismologist section of SRL is still working. The test shouldn't be
        changed because the reference gets wrong. Please do not change the
        import for these test either.
        """
        paz = {'gain': 60077000.0,
               'poles': [(-0.037004000000000002+0.037016j),
                         (-0.037004000000000002-0.037016j),
                         (-251.33000000000001+0j),
                         (-131.03999999999999-467.29000000000002j),
                         (-131.03999999999999+467.29000000000002j)],
               'sensitivity': 2516778400.0,
               'zeros': [0j, 0j]}
        
        stats = {'network': 'BW', 'station': 'RJOB', 'sampling_rate': 200.0,
                 'starttime': UTCDateTime(2009, 8, 24, 0, 20, 3), 'npts': 6001,
                 'channel': 'EHZ'}
        f = gzip.open(os.path.join(self.path, 'srl.data.gz'))
        srl_data = np.loadtxt(f)
        f.close()
        f = gzip.open(os.path.join(self.path, 'srl.res.gz'))
        srl_res = np.loadtxt(f)
        f.close()
        # Generate the stream from data verify it
        st = Stream([Trace(header=stats, data=srl_data)])
        st.verify()        
        one_hertz = cornFreq2Paz(1.0) # 1Hz instrument
        #2 Correct for frequency response of the instrument
        res = seisSim(st[0].data.astype("float32"),
                      st[0].stats.sampling_rate,
                      paz, inst_sim=one_hertz)
        # correct for overall sensitivity, nm/s
        res *= 1e9 / paz["sensitivity"]
        #3 Apply lowpass at 10Hz
        res = lowpass(res, 10, df=st[0].stats.sampling_rate,
        corners=4)
        # test versus saved result
        np.testing.assert_array_almost_equal(res, srl_res)


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
