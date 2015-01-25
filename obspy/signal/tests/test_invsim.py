#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The InvSim test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.utils import native_str
from future.builtins import *  # NOQA

from obspy import Trace, UTCDateTime, read
from obspy.core.util.base import NamedTemporaryFile
from obspy.core.util.misc import CatchOutput
from obspy.sac import attach_paz
from obspy.signal.invsim import seisSim, estimateMagnitude, evalresp
from obspy.signal.invsim import cosTaper
from obspy.signal.headers import clibevresp

import io
import gzip
import numpy as np
import os
import unittest
import ctypes as C

# Seismometers defined as in Pitsa with one zero less. The corrected
# signals are in velocity, thus must be integrated to offset and take one
# zero less than pitsa (remove 1/w in frequency domain)
PAZ_WOOD_ANDERSON = {'poles': [-6.2832 - 4.7124j,
                               -6.2832 + 4.7124j],
                     'zeros': [0.0 + 0.0j] * 1,
                     'sensitivity': 1.0,
                     'gain': 1. / 2.25}

PAZ_WWSSN_SP = {'poles': [-4.0093 - 4.0093j,
                          -4.0093 + 4.0093j,
                          -4.6077 - 6.9967j,
                          -4.6077 + 6.9967j],
                'zeros': [0.0 + 0.0j] * 2,
                'sensitivity': 1.0,
                'gain': 1. / 1.0413}

PAZ_WWSSN_LP = {'poles': [-0.4189 + 0.0j,
                          -0.4189 + 0.0j,
                          -0.0628 + 0.0j,
                          -0.0628 + 0.0j],
                'zeros': [0.0 + 0.0j] * 2,
                'sensitivity': 1.0,
                'gain': 1. / 0.0271}

PAZ_KIRNOS = {'poles': [-0.1257 - 0.2177j,
                        -0.1257 + 0.2177j,
                        -83.4473 + 0.0j,
                        -0.3285 + 0.0j],
              'zeros': [0.0 + 0.0j] * 2,
              'sensitivity': 1.0,
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
        # no with due to py 2.6
        f = gzip.open(file)
        data = np.loadtxt(f)
        f.close()

        # paz of test file
        samp_rate = 200.0
        PAZ_LE3D = {'poles': [-4.21 + 4.66j,
                              -4.21 - 4.66j,
                              -2.105 + 0.0j],
                    'zeros': [0.0 + 0.0j] * 3,
                    'sensitivity': 1.0,
                    'gain': 0.4}

        for id, paz in INSTRUMENTS.items():
            # simulate instrument
            datcorr = seisSim(data, samp_rate, paz_remove=PAZ_LE3D,
                              paz_simulate=paz, water_level=600.0,
                              zero_mean=False, nfft_pow2=True)
            # load pitsa file
            file = os.path.join(self.path, 'rjob_20051006_%s.gz' % id)
            # no with due to py 2.6
            f = gzip.open(file)
            data_pitsa = np.loadtxt(f)
            f.close()
            # calculate normalized rms
            rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) /
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
                              -0.03736 + 0.03617j],
                    'zeros': [0.0 + 0.0j] * 2,
                    'sensitivity': 1.0,
                    'gain': 1.5}

        for id, paz in INSTRUMENTS.items():
            # simulate instrument
            datcorr = seisSim(data, samp_rate, paz_remove=PAZ_STS2,
                              paz_simulate=paz, water_level=600.0,
                              zero_mean=False, nfft_pow2=True)
            # load pitsa file
            file = os.path.join(self.path, 'rotz_20081028_%s.gz' % id)
            # no with due to py 2.6
            f = gzip.open(file)
            data_pitsa = np.loadtxt(f)
            f.close()
            # calculate normalized rms
            rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) /
                          np.sum(data_pitsa ** 2))
            self.assertTrue(rms < 1e-04)

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
        paz = {'poles': [-4.444 + 4.444j, -4.444 - 4.444j, -1.083 + 0j],
               'zeros': [0 + 0j, 0 + 0j, 0 + 0j],
               'gain': 1.0,
               'sensitivity': 671140000.0}
        mag_RTSH = estimateMagnitude(paz, 3.34e6, 0.065, 0.255)
        self.assertAlmostEqual(mag_RTSH, 2.1328727151723488)
        mag_RTBE = estimateMagnitude(paz, 3.61e4, 0.08, 2.197)
        self.assertAlmostEqual(mag_RTBE, 1.1962687721890191)
        mag_RNON = estimateMagnitude(paz, 6.78e4, 0.125, 1.538)
        self.assertAlmostEqual(mag_RNON, 1.4995311686507182)

    # XXX: Test for really big signal is missing, where the water level is
    # actually acting
    # def test_seisSimVsPitsa2(self):
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
        # Uncomment the following to run the sac-commands
        # that created the testing file
        # if 1:
        #    import subprocess as sp
        #    p = sp.Popen('sac',shell=True,stdin=sp.PIPE)
        #    cd1 = p.stdin
        #    print("r %s"%sacf, file=cd1)
        #    print("rmean", file=cd1)
        #    print("rtrend", file=cd1)
        #    print("taper type cosine width 0.03", file=cd1)
        #    print("transfer from polezero subtype %s to none \
        #    freqlimits %f %f %f %f" % (pzf, fl1, fl2, fl3, fl4), file=cd1)
        #    print("w over ./data/KARC_corrected.sac", file=cd1)
        #    print("quit", file=cd1)
        #    cd1.close()
        #    p.wait()

        stats = {'network': 'KA', 'delta': 0.99999988079072466,
                 'station': 'KARC', 'location': 'S1',
                 'starttime': UTCDateTime(2001, 2, 13, 0, 0, 0, 993700),
                 'calib': 1.00868e+09, 'channel': 'BHZ'}
        f = gzip.open(sacf)
        tr = Trace(np.loadtxt(f), stats)
        f.close()

        attach_paz(tr, pzf, tovel=False)
        tr.data = seisSim(tr.data, tr.stats.sampling_rate,
                          paz_remove=tr.stats.paz, remove_sensitivity=False,
                          pre_filt=(fl1, fl2, fl3, fl4))

        f = gzip.open(testsacf)
        data = np.loadtxt(f)
        f.close()

        # import matplotlib.pyplot as plt
        # plt.plot(tr.data)
        # plt.plot(data)
        # plt.show()
        rms = np.sqrt(np.sum((tr.data - data) ** 2) /
                      np.sum(tr.data ** 2))
        self.assertTrue(rms < 0.0421)

    def test_evalrespVsObsPy(self):
        """
        Compare results from removing instrument response using
        evalresp in SAC and ObsPy. Visual inspection shows that the traces are
        pretty much identical but differences remain (rms ~ 0.042). Haven't
        found the cause for those, yet.
        """
        evalrespf = os.path.join(self.path, 'CRLZ.HHZ.10.NZ.SAC_resp')
        rawf = os.path.join(self.path, 'CRLZ.HHZ.10.NZ.SAC')
        respf = os.path.join(self.path, 'RESP.NZ.CRLZ.10.HHZ')
        fl1 = 0.00588
        fl2 = 0.00625
        fl3 = 30.
        fl4 = 35.

#        #Set the following if-clause to True to run
#        #the sac-commands that created the testing file
#        if False:
#            import subprocess as sp
#            p = sp.Popen('sac', stdin=sp.PIPE)
#            cd1 = p.stdin
#            print("r %s" % rawf, file=cd1)
#            print("rmean", file=cd1)
#            print("taper type cosine width 0.05", file=cd1)
#            print("transfer from evalresp fname %s to vel freqlimits\
#            %f %f %f %f" % (respf, fl1, fl2, fl3, fl4), file=cd1)
#            print("w over %s" % evalrespf, file=cd1)
#            print("quit", file=cd1)
#            cd1.close()
#            p.wait()

        tr = read(rawf)[0]
        trtest = read(evalrespf)[0]
        date = UTCDateTime(2003, 11, 1, 0, 0, 0)
        seedresp = {'filename': respf, 'date': date, 'units': 'VEL',
                    'network': 'NZ', 'station': 'CRLZ', 'location': '10',
                    'channel': 'HHZ'}
        tr.data = seisSim(tr.data, tr.stats.sampling_rate, paz_remove=None,
                          pre_filt=(fl1, fl2, fl3, fl4),
                          seedresp=seedresp, taper_fraction=0.1,
                          pitsasim=False, sacsim=True)
        tr.data *= 1e9
        rms = np.sqrt(np.sum((tr.data - trtest.data) ** 2) /
                      np.sum(trtest.data ** 2))
        self.assertTrue(rms < 0.0094)
        # import matplotlib.pyplot as plt #plt.plot(tr.data-trtest.data,'b')
        # plt.plot(trtest.data,'g')
        # plt.figure()
        # plt.psd(tr.data,Fs=100.,NFFT=32768)
        # plt.psd(trtest.data,Fs=100.,NFFT=32768)
        # plt.figure()
        # plt.psd(tr.data - trtest.data, Fs=100., NFFT=32768)
        # plt.show()

    def test_cosineTaper(self):
        # SAC trace was generated with:
        # taper type cosine width 0.05
        for i in [99, 100]:
            sac_taper = os.path.join(self.path,
                                     'ones_trace_%d_tapered.sac' % i)
            tr = read(sac_taper)[0]
            tap = cosTaper(i, p=0.1, halfcosine=False, sactaper=True)
            np.testing.assert_array_almost_equal(tap, tr.data, decimal=6)

        # The following lines compare the cosTaper result with
        # the result of the algorithm used by SAC in its taper routine
        # (taper.c)
        # freqs = np.fft.fftfreq(2**15,0.01)
        # fl1 = 0.00588
        # fl2 = 0.00625
        # fl3 = 30.0
        # fl4 = 35.0
        # npts = freqs.size
        # tap = cosTaper(freqs.size, freqs=freqs, flimit=(fl1, fl2, fl3, fl4))
        # tap2 = c_sac_taper(freqs, flimit=(fl1, fl2, fl3, fl4))
        # import matplotlib.pyplot as plt
        # plt.plot(tap,'b')
        # plt.plot(tap2,'g--')
        # plt.show()

    def test_evalrespUsingDifferentLineSeparator(self):
        """
        The evalresp needs a file with correct line separator, so '\n' for
        POSIX, '\r' for Mac OS, or '\r\n' for Windows. Here we check that
        evalresp reads all three formats.

        This test only checks the parsing capabilities of evalresp,
        the number of fft points used (nfft) can therefore be chosen
        small.
        """
        dt = UTCDateTime(2003, 11, 1, 0, 0, 0)
        nfft = 8
        # Linux
        respf = os.path.join(self.path, 'RESP.NZ.CRLZ.10.HHZ')
        evalresp(0.01, nfft, respf, dt)
        # Mac
        respf = os.path.join(self.path, 'RESP.NZ.CRLZ.10.HHZ.mac')
        evalresp(0.01, nfft, respf, dt)
        # Windows
        respf = os.path.join(self.path, 'RESP.NZ.CRLZ.10.HHZ.windows')
        evalresp(0.01, nfft, respf, dt)

    def test_evalrespBug395(self):
        """
        Was a bug due to inconstistent numerical range
        """
        resp = os.path.join(self.path, 'RESP.CH._.HHZ.gz')
        with NamedTemporaryFile() as fh:
            tmpfile = fh.name
            # no with due to py 2.6
            f = gzip.open(resp)
            fh.write(f.read())
            f.close()
            samprate = 120.0
            nfft = 56328
            args = [1.0 / samprate, nfft, tmpfile,
                    UTCDateTime(2012, 9, 4, 5, 12, 15, 863300)]
            kwargs = {'units': 'VEL', 'freq': True}
            _h, f = evalresp(*args, **kwargs)
            self.assertEqual(len(f), nfft // 2 + 1)

    def test_evalresp_file_like_object(self):
        """
        Test evalresp with file like object
        """
        rawf = os.path.join(self.path, 'CRLZ.HHZ.10.NZ.SAC')
        respf = os.path.join(self.path, 'RESP.NZ.CRLZ.10.HHZ')

        tr1 = read(rawf)[0]
        tr2 = read(rawf)[0]

        date = UTCDateTime(2003, 11, 1, 0, 0, 0)
        seedresp = {'filename': respf, 'date': date, 'units': 'VEL',
                    'network': 'NZ', 'station': 'CRLZ', 'location': '10',
                    'channel': 'HHZ'}
        tr1.data = seisSim(tr1.data, tr1.stats.sampling_rate,
                           seedresp=seedresp)

        with open(respf, 'rb') as fh:
            stringio = io.BytesIO(fh.read())
        seedresp['filename'] = stringio
        tr2.data = seisSim(tr2.data, tr2.stats.sampling_rate,
                           seedresp=seedresp)

        self.assertEqual(tr1, tr2)

    def test_segfaulting_RESP_file(self):
        """
        Test case for a file that segfaults when compiled with clang and
        active optimization.

        As long as the test does not segfault it is ok.
        """
        filename = os.path.join(self.path, "segfaulting_RESPs",
                                "RESP.IE.LLRI..EHZ")
        date = UTCDateTime(2003, 11, 1, 0, 0, 0)
        with CatchOutput():
            self.assertRaises(ValueError, evalresp, t_samp=10.0, nfft=256,
                              filename=filename, date=date, station="LLRI",
                              channel="EHZ", network="IE", locid="*",
                              units="VEL")

    def test_evalresp_seed_identifiers_work(self):
        """
        Asserts that the network, station, location and channel identifiers can
        be used to select difference responses.
        """
        kwargs = {"filename": os.path.join(self.path, "RESP.OB.AAA._.BH_"),
                  "t_samp": 0.1, "nfft": 1024, "units": "VEL",
                  "date": UTCDateTime(2013, 1, 1), "network": "OP",
                  "station": "AAA", "locid": "", "freq": False, "debug": False}

        # Get the response for the first channel
        kwargs["channel"] = "BHE"
        response_1 = evalresp(**kwargs)

        # Get the second one. Should be different.
        kwargs["channel"] = "BHN"
        response_2 = evalresp(**kwargs)

        # The only thing that changed was the channel code. This should change
        # the response.
        rel_diff = np.abs(response_2 - response_1).ptp() / \
            max(np.abs(response_1).ptp(), np.abs(response_2).ptp())
        self.assertTrue(rel_diff > 1E-3)

        # The RESP file only contains two channels.
        kwargs["channel"] = "BHZ"
        with CatchOutput() as out:
            self.assertRaises(ValueError, evalresp, **kwargs)
        self.assertTrue(b"no response found for" in out.stderr.lower())

    def test_evalresp_spline(self):
        """
        evr_spline was based on GPL plotutils, now replaced by LGPL spline
        library. Unittest for this function.
        """
        # char *evr_spline(int num_points, double *t, double *y,
        #                  double tension, double k,
        #                  double *xvals_arr, int num_xvals,
        #                  double **p_retvals_arr, int *p_num_retvals)
        clibevresp.evr_spline.argtypes = [
            C.c_int,  # num_points
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags=native_str('C_CONTIGUOUS')),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags=native_str('C_CONTIGUOUS')),
            C.c_double,  # tension
            C.c_double,  # k
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags=native_str('C_CONTIGUOUS')),
            C.c_int,  # num_xvals
            C.POINTER(C.POINTER(C.c_double)),
            C.POINTER(C.c_int)
        ]
        clibevresp.evr_spline.restype = C.c_char_p

        x = np.arange(1.2, 2.0, .1)
        N = len(x)
        y = np.sin(x)

        xi = x[:-1] + .05
        Ni = len(xi)

        p_num_retvals = C.c_int(0)
        p_retvals_arr = C.POINTER(C.c_double)()
        res = clibevresp.evr_spline(N, x, y, 0.0, 1.0, xi, Ni,
                                    C.byref(p_retvals_arr),
                                    C.byref(p_num_retvals))
        self.assertEqual(res, None)
        self.assertEqual(Ni, p_num_retvals.value)
        yi = np.array([p_retvals_arr[i] for i in range(Ni)])

        if False:  # visually verify
            import matplotlib.pyplot as plt
            plt.plot(x, y, 'bo-', 'Orig values')
            plt.plot(xi, yi, 'ro-', 'Cubic Spline interpolated values')
            plt.legend()
            plt.show()

        yi_ref = [0.94899576, 0.97572004, 0.9927136, 0.99978309, 0.99686554,
                  0.98398301, 0.96128491]
        self.assertTrue(np.allclose(yi, yi_ref, rtol=1e-7, atol=0))


def suite():
    return unittest.makeSuite(InvSimTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
