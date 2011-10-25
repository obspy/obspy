#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The InvSim test suite.
"""

from obspy.core import Stream, Trace, UTCDateTime
from obspy.signal import seisSim, cornFreq2Paz, lowpass, estimateMagnitude
from obspy.sac import attach_paz
import gzip
import numpy as np
import os
import unittest


# Seismometers defined as in Pitsa with one zero less. The corrected
# signals are in velocity, thus must be integrated to offset and take one
# zero less than pitsa (remove 1/w in frequency domain)
PAZ_WOOD_ANDERSON = {'poles': [-6.2832 - 4.7124j,
                               - 6.2832 + 4.7124j],
                     'zeros': [0.0 + 0.0j] * 1,
                     'gain': 1. / 2.25}

PAZ_WWSSN_SP = {'poles': [-4.0093 - 4.0093j,
                          - 4.0093 + 4.0093j,
                          - 4.6077 - 6.9967j,
                          - 4.6077 + 6.9967j],
                'zeros': [0.0 + 0.0j] * 2,
                'gain': 1. / 1.0413}

PAZ_WWSSN_LP = {'poles': [-0.4189 + 0.0j,
                          - 0.4189 + 0.0j,
                          - 0.0628 + 0.0j,
                          - 0.0628 + 0.0j],
                'zeros': [0.0 + 0.0j] * 2,
                'gain': 1. / 0.0271}

PAZ_KIRNOS = {'poles': [-0.1257 - 0.2177j,
                        - 0.1257 + 0.2177j,
                        - 83.4473 + 0.0j,
                        - 0.3285 + 0.0j],
              'zeros': [0.0 + 0.0j] * 2,
              'gain': 1. / 1.61}

INSTRUMENTS = {'None': None,
               'kirnos': PAZ_KIRNOS,
               'wood_anderson': PAZ_WOOD_ANDERSON,
               'wwssn_lp': PAZ_WWSSN_LP,
               'wwssn_sp': PAZ_WWSSN_SP}


class InvSimTestCase(unittest.TestCase):
    """
    Test cases for InvSim.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

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
        PAZ_LE3D = {'poles': [-4.21 + 4.66j,
                              - 4.21 - 4.66j,
                              - 2.105 + 0.0j],
                    'zeros': [0.0 + 0.0j] * 3,
                    'gain': 0.4}

        for id, paz in INSTRUMENTS.iteritems():
            # simulate instrument
            datcorr = seisSim(data, samp_rate, paz_remove=PAZ_LE3D,
                              paz_simulate=paz, water_level=600.0,
                              zero_mean=False)
            # load pitsa file
            file = os.path.join(self.path, 'rjob_20051006_%s.gz' % id)
            f = gzip.open(file)
            data_pitsa = np.loadtxt(f)
            f.close()
            # calculate normalized rms
            rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) / \
                          np.sum(data_pitsa ** 2))
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
        PAZ_STS2 = {'poles': [-0.03736 - 0.03617j,
                              - 0.03736 + 0.03617j],
                    'zeros': [0.0 + 0.0j] * 2,
                    'gain': 1.5}

        for id, paz in INSTRUMENTS.iteritems():
            # simulate instrument
            datcorr = seisSim(data, samp_rate, paz_remove=PAZ_STS2,
                              paz_simulate=paz, water_level=600.0,
                              zero_mean=False)
            # load pitsa file
            file = os.path.join(self.path, 'rotz_20081028_%s.gz' % id)
            f = gzip.open(file)
            data_pitsa = np.loadtxt(f)
            f.close()
            # calculate normalized rms
            rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) / \
                         np.sum(data_pitsa ** 2))
            self.assertTrue(rms < 1e-04)

    def test_SRLSeisSim(self):
        """
        Tests if example in ObsPy paper submitted to the Electronic
        Seismologist section of SRL is still working. The test shouldn't be
        changed because the reference gets wrong. Please do not change the
        import for these test either.
        """
        paz = {'gain': 60077000.0,
               'poles': [(-0.037004000000000002 + 0.037016j),
                         (-0.037004000000000002 - 0.037016j),
                         (-251.33000000000001 + 0j),
                         (-131.03999999999999 - 467.29000000000002j),
                         (-131.03999999999999 + 467.29000000000002j)],
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
        one_hertz = cornFreq2Paz(1.0)  # 1Hz instrument
        # 2 Correct for frequency response of the instrument
        res = seisSim(st[0].data.astype("float32"),
                      st[0].stats.sampling_rate,
                      paz_remove=paz, paz_simulate=one_hertz, zero_mean=False)
        # correct for overall sensitivity, nm/s
        res *= 1e9 / paz["sensitivity"]
        # 3 Apply lowpass at 10Hz
        res = lowpass(res, 10, df=st[0].stats.sampling_rate,
        corners=4)
        # test versus saved result
        np.testing.assert_array_almost_equal(res, srl_res)

    def test_estimateMagnitude(self):
        """
        Tests against PITSA. Note that PITSA displays microvolt, that is
        the amplitude values must be computed back into counts (for this
        stations .596microvolt/count was used). Pitsa internally calculates
        with the sensitivity 2800 of the WA. Using this we get for the
        following for event 2009-07-19 23:03::

            RTSH PITSA 2.263 ObsPy 2.294
            RTBE PITSA 1.325 ObsPy 1.363
            RMOA PITSA 1.629 ObsPy 1.675
        """
        paz = {'poles': [-4.444 + 4.444j, -4.444 - 4.444j, -1.083 + 0j], \
               'zeros': [0 + 0j, 0 + 0j, 0 + 0j], \
               'gain': 1.0, \
               'sensitivity': 671140000.0}
        mag_RTSH = estimateMagnitude(paz, 3.34e6, 0.065, 0.255)
        self.assertAlmostEqual(mag_RTSH, 2.1653454839257327)
        mag_RTBE = estimateMagnitude(paz, 3.61e4, 0.08, 2.197)
        self.assertAlmostEqual(mag_RTBE, 1.2334841683429503)
        mag_RNON = estimateMagnitude(paz, 6.78e4, 0.125, 1.538)
        self.assertAlmostEqual(mag_RNON, 1.5455526399683184)

    #XXX: Test for really big signal is missing, where the water level is
    # actually acting
    #def test_seisSimVsPitsa2(self):
    #    from obspy.mseed import test as tests_
    #    path = os.path.dirname(__file__)
    #    file = os.path.join(path, 'data', 'BW.BGLD..EHE.D.2008.001')
    #    g = Trace()
    #    g.read(file,format='MSEED')
    #    # paz of test file
    #    samp_rate = 200.0

    def test_SacInstCorrection(self):
        # SAC recommends to taper the transfer function if a pure
        # deconvolution is done instead of simulating a different
        # instrument. This test checks the difference between the
        # result from removing the instrument response using SAC or
        # ObsPy. Visual inspection shows that the traces are pretty
        # much identical but differences remain (rms ~ 0.042). Haven't
        # found the cause for those, yet. One possible reason is the
        # floating point arithmetic of SAC vs. the double precision
        # arithmetic of Python. However differences still seem to be
        # too big for that.
        pzf = os.path.join(self.path, 'SAC_PZs_KARC_BHZ')
        sacf = os.path.join(self.path, 'KARC.LHZ.SAC.asc.gz')
        testsacf = os.path.join(self.path, 'KARC_corrected.sac.asc.gz')
        plow = 160.
        phigh = 4.
        fl1 = 1.0 / (plow + 0.0625 * plow)
        fl2 = 1.0 / plow
        fl3 = 1.0 / phigh
        fl4 = 1.0 / (phigh - 0.25 * phigh)
        #Uncomment the following to run the sac-commands
        #that created the testing file
        #if 1:
        #    import subprocess as sp
        #    p = sp.Popen('sac',shell=True,stdin=sp.PIPE)
        #    cd1 = p.stdin
        #    print >>cd1, "r %s"%sacf
        #    print >>cd1, "rmean"
        #    print >>cd1, "rtrend"
        #    print >>cd1, "taper type cosine width 0.03"
        #    print >>cd1, "transfer from polezero subtype %s to none \
        #    freqlimits %f %f %f %f" % (pzf, fl1, fl2, fl3, fl4)
        #    print >>cd1, "w over ./data/KARC_corrected.sac"
        #    print >>cd1, "quit"
        #    cd1.close()
        #    p.wait()

        data = np.loadtxt(sacf)
        stats = {'network': 'KA', 'delta': 0.99999988079072466,
                 'station': 'KARC', 'location': 'S1',
                 'starttime': UTCDateTime(2001, 2, 13, 0, 0, 0, 993700),
                 'npts': 86399, 'calib': 1.00868e+09,
                 'sampling_rate': 1.0000001192092896, 'channel': 'BHZ'}
        tr = Trace(data, stats)

        attach_paz(tr, pzf, tovel=False)
        tr.data = seisSim(tr.data, tr.stats.sampling_rate,
                          paz_remove=tr.stats.paz, remove_sensitivity=False,
                          pre_filt=(fl1, fl2, fl3, fl4))
        data = np.loadtxt(testsacf)
        stats = {'network': 'KA', 'delta': 0.99999988079072466,
                 'station': 'KARC', 'location': 'S1',
                 'starttime': UTCDateTime(2001, 2, 13, 0, 0, 0, 993700),
                 'npts': 86399, 'calib': 1.0,
                 'sampling_rate': 1.0000001192092896, 'channel': 'BHZ'}
        tr2 = Trace(data, stats)

        #import pylab as plt
        #plt.plot(tr.data)
        #plt.plot(tr2.data)
        #plt.show()
        rms = np.sqrt(np.sum((tr.data - tr2.data) ** 2) / \
                      np.sum(tr2.data ** 2))
        self.assertTrue(rms < 0.0421)

    def test_evalrespsac_vs_obspy(self):
        """
        Compare results from removing instrument response using
        evalresp in SAC and ObsPy. Visual inspection shows that the traces are
        pretty much identical but differences remain (rms ~ 0.042). Haven't
        found the cause for those, yet.
        """
        evalrespf = os.path.join(self.path, 'CRLZ_.HHZ.10.NZ.SAC_resp.asc.gz')
        rawf = os.path.join(self.path, 'CRLZ_.HHZ.10.NZ.SAC.asc.gz')
        respf = os.path.join(self.path, 'RESP.NZ.CRLZ.10.HHZ')

        data = np.loadtxt(rawf)
        test_data = np.loadtxt(evalrespf)
        stats = {'network': 'NZ', 'delta': 0.01,
                 'station': 'CRLZ', 'location': '10',
                 'starttime': UTCDateTime(2010, 9, 4, 4, 52, 58, 997000),
                 'npts': 49000, 'calib': 1.0,
                 'sampling_rate': 100.0, 'channel': 'HHZ'}
        tr = Trace(data, stats)
        trtest = Trace(test_data, stats)
        fl1 = 0.00588
        fl2 = 0.00625
        fl3 = 30.
        fl4 = 35.
        date = UTCDateTime(2003, 11, 1, 0, 0, 0)
        seedresp = {'filename': respf, 'date': date, 'units': 'VEL'}
        tr.data = seisSim(tr.data, tr.stats.sampling_rate, paz_remove=None,
                          remove_sensitivity=False,
                          pre_filt=(fl1, fl2, fl3, fl4), seedresp=seedresp)
        tr.data *= 1e9
        rms = np.sqrt(np.sum((tr.data - trtest.data) ** 2) / \
                      np.sum(trtest.data ** 2))
        self.assertTrue(rms < 0.0041)


def suite():
    return unittest.makeSuite(InvSimTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
