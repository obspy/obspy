#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The polarization.core test suite.
"""
import unittest
from os.path import dirname, join

import numpy as np
from scipy import signal

import obspy
from obspy.signal import polarization, util


def _create_test_data():
    """
    Test data used for some polarization tests.
    :return:
    """
    x = np.arange(0, 2048 / 20.0, 1.0 / 20.0)
    x *= 2. * np.pi
    y = np.cos(x)
    tr_z = obspy.Trace(data=y)
    tr_z.stats.sampling_rate = 20.
    tr_z.stats.starttime = obspy.UTCDateTime('2014-03-01T00:00')
    tr_z.stats.station = 'POLT'
    tr_z.stats.channel = 'HHZ'
    tr_z.stats.network = 'XX'

    tr_n = tr_z.copy()
    tr_n.data *= 2.
    tr_n.stats.channel = 'HHN'
    tr_e = tr_z.copy()
    tr_e.stats.channel = 'HHE'

    sz = obspy.Stream()
    sz.append(tr_z)
    sz.append(tr_n)
    sz.append(tr_e)
    sz.sort(reverse=True)

    return sz


class PolarizationTestCase(unittest.TestCase):
    """
    Test cases for polarization analysis
    """
    def setUp(self):
        path = join(dirname(__file__), 'data')
        # setting up sliding window data
        data_z = np.loadtxt(join(path, 'MBGA_Z.ASC'))
        data_e = np.loadtxt(join(path, 'MBGA_E.ASC'))
        data_n = np.loadtxt(join(path, 'MBGA_N.ASC'))
        n = 256
        fs = 75
        inc = int(0.05 * fs)
        self.data_win_z, self.nwin, self.no_win = \
            util.enframe(data_z, signal.hamming(n), inc)
        self.data_win_e, self.nwin, self.no_win = \
            util.enframe(data_e, signal.hamming(n), inc)
        self.data_win_n, self.nwin, self.no_win = \
            util.enframe(data_n, signal.hamming(n), inc)
        # global test input
        self.fk = [2, 1, 0, -1, -2]
        self.norm = pow(np.max(data_z), 2)
        self.res = np.loadtxt(join(path, '3cssan.hy.1.MBGA_Z'))

    def tearDown(self):
        pass

    def test_polarization(self):
        """
        windowed data
        """
        pol = polarization.eigval(self.data_win_e, self.data_win_n,
                                  self.data_win_z, self.fk, self.norm)
        rms = np.sqrt(np.sum((pol[0] - self.res[:, 34]) ** 2) /
                      np.sum(self.res[:, 34] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[1] - self.res[:, 35]) ** 2) /
                      np.sum(self.res[:, 35] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[2] - self.res[:, 36]) ** 2) /
                      np.sum(self.res[:, 36] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[3] - self.res[:, 40]) ** 2) /
                      np.sum(self.res[:, 40] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[4] - self.res[:, 42]) ** 2) /
                      np.sum(self.res[:, 42] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[5][:, 0] - self.res[:, 37]) ** 2) /
                      np.sum(self.res[:, 37] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[5][:, 1] - self.res[:, 38]) ** 2) /
                      np.sum(self.res[:, 38] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[5][:, 2] - self.res[:, 39]) ** 2) /
                      np.sum(self.res[:, 39] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[6] - self.res[:, 41]) ** 2) /
                      np.sum(self.res[:, 41] ** 2))
        self.assertEqual(rms < 1.0e-5, True)
        rms = np.sqrt(np.sum((pol[7] - self.res[:, 43]) ** 2) /
                      np.sum(self.res[:, 43] ** 2))
        self.assertEqual(rms < 1.0e-5, True)

    def test_polarization_1d(self):
        """
        1 dimenstional input --- regression test case for bug #919
        """
        pol = polarization.eigval(self.data_win_e[100, :],
                                  self.data_win_n[100, :],
                                  self.data_win_z[100, :],
                                  self.fk, self.norm)
        pol_5_ref = [2.81387533e-04, 3.18409580e-04, 6.74030846e-04,
                     5.55067015e-01, 4.32938188e-01]
        self.assertTrue(np.allclose(np.concatenate(pol[:5]), pol_5_ref))

    def test_polarization_pm(self):
        st = _create_test_data()
        t = st[0].stats.starttime
        e = st[0].stats.endtime
        wlen = 10.0
        wfrac = 0.1

        out = polarization.polarization_analysis(
            st, win_len=wlen, win_frac=wfrac, frqlow=1.0, frqhigh=5.0,
            verbose=False, stime=t, etime=e, method="pm",
            var_noise=0.0)

        # all values should be equal for the test data, so check first value
        # and make sure all values are almost equal
        self.assertEqual(out["timestamp"][0], 1393632005.0)
        self.assertEqual(out["timestamp"][0], t + wlen / 2.0)
        self.assertAlmostEqual(out["azimuth"][0], 26.56505117707799)
        self.assertAlmostEqual(out["incidence"][0], 65.905157447889309)
        self.assertAlmostEqual(out["azimuth_error"][0], 0.000000)
        self.assertAlmostEqual(out["incidence_error"][0], 0.000000)
        for key in ["azimuth", "incidence"]:
            got = out[key]
            self.assertTrue(np.allclose(got / got[0], np.ones_like(got),
                                        rtol=1e-4))
        for key in ["azimuth_error", "incidence_error"]:
            got = out[key]
            expected = np.empty_like(got)
            expected.fill(got[0])
            self.assertTrue(np.allclose(got, expected, rtol=1e-4, atol=1e-16))
        self.assertTrue(np.allclose(out["timestamp"] - out["timestamp"][0],
                                    np.arange(0, 92, 1)))

    def test_polarization_flinn(self):
        st = _create_test_data()
        t = st[0].stats.starttime
        e = st[0].stats.endtime
        wlen = 10.0
        wfrac = 0.1

        out = polarization.polarization_analysis(
            st, win_len=wlen, win_frac=wfrac, frqlow=1.0, frqhigh=5.0,
            verbose=False, stime=t, etime=e,
            method="flinn", var_noise=0.0)

        # all values should be equal for the test data, so check first value
        # and make sure all values are almost equal
        self.assertEqual(out["timestamp"][0], 1393632005.0)
        self.assertEqual(out["timestamp"][0], t + wlen / 2.0)
        self.assertAlmostEqual(out["azimuth"][0], 26.56505117707799)
        self.assertAlmostEqual(out["incidence"][0], 65.905157447889309)
        self.assertAlmostEqual(out["rectilinearity"][0], 1.000000)
        self.assertAlmostEqual(out["planarity"][0], 1.000000)
        for key in ["azimuth", "incidence", "rectilinearity", "planarity"]:
            got = out[key]
            self.assertTrue(np.allclose(got / got[0], np.ones_like(got),
                                        rtol=1e-4))
        self.assertTrue(np.allclose(out["timestamp"] - out["timestamp"][0],
                                    np.arange(0, 92, 1)))

    def test_polarization_vidale(self):
        st = _create_test_data()
        t = st[0].stats.starttime
        e = st[0].stats.endtime

        out = polarization.polarization_analysis(
            st, win_len=10.0, win_frac=0.1, frqlow=1.0, frqhigh=5.0,
            verbose=False, stime=t, etime=e,
            method="vidale", var_noise=0.0)

        # all values should be equal for the test data, so check first value
        # and make sure all values are almost equal
        self.assertEqual(out["timestamp"][0], 1393632003.0)
        self.assertAlmostEqual(out["azimuth"][0], 26.56505117707799)
        self.assertAlmostEqual(out["incidence"][0], 65.905157447889309)
        self.assertAlmostEqual(out["rectilinearity"][0], 1.000000)
        self.assertAlmostEqual(out["planarity"][0], 1.000000)
        self.assertAlmostEqual(out["ellipticity"][0], 3.8195545129768958e-06)
        for key in ["azimuth", "incidence", "rectilinearity", "planarity",
                    "ellipticity"]:
            got = out[key]
            self.assertTrue(np.allclose(got / got[0], np.ones_like(got),
                                        rtol=1e-4))
        self.assertTrue(np.allclose(out["timestamp"] - out["timestamp"][0],
                                    np.arange(0, 97.85, 0.05), rtol=1e-5))
