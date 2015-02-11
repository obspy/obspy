#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Filter test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import gzip
import os
import unittest

import numpy as np
import scipy.signal as sg

from obspy.signal import bandpass, highpass, lowpass
from obspy.signal.filter import envelope, lowpassCheby2


class FilterTestCase(unittest.TestCase):
    """
    Test cases for Filter.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_bandpassVsPitsa(self):
        """
        Test Butterworth bandpass filter against Butterworth bandpass filter
        of PITSA. Note that the corners value is twice the value of the filter
        sections in PITSA. The rms of the difference between ObsPy and PITSA
        tends to get bigger with higher order filtering.
        """
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.gz')
        f = gzip.open(file)
        data = np.loadtxt(f)
        f.close()
        # parameters for the test
        samp_rate = 200.0
        freq1 = 5
        freq2 = 10
        corners = 4
        # filter trace
        datcorr = bandpass(data, freq1, freq2, df=samp_rate, corners=corners)
        # load pitsa file
        file = os.path.join(self.path, 'rjob_20051006_bandpass.gz')
        f = gzip.open(file)
        data_pitsa = np.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) /
                      np.sum(data_pitsa ** 2))
        self.assertEqual(rms < 1.0e-05, True)

    def test_bandpassZPHSHVsPitsa(self):
        """
        Test Butterworth zero-phase bandpass filter against Butterworth
        zero-phase bandpass filter of PITSA. Note that the corners value is
        twice the value of the filter sections in PITSA. The rms of the
        difference between ObsPy and PITSA tends to get bigger with higher
        order filtering.
        Note: The Zero-Phase filters deviate from PITSA's zero-phase filters
        at the end of the trace! The rms for the test is calculated omitting
        the last 200 samples, as this part of the trace is assumed to
        generally be of low interest/importance.
        """
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.gz')
        f = gzip.open(file)
        data = np.loadtxt(f)
        f.close()
        # parameters for the test
        samp_rate = 200.0
        freq1 = 5
        freq2 = 10
        corners = 2
        # filter trace
        datcorr = bandpass(data, freq1, freq2, df=samp_rate,
                           corners=corners, zerophase=True)
        # load pitsa file
        file = os.path.join(self.path, 'rjob_20051006_bandpassZPHSH.gz')
        f = gzip.open(file)
        data_pitsa = np.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr[:-200] - data_pitsa[:-200]) ** 2) /
                      np.sum(data_pitsa[:-200] ** 2))
        self.assertEqual(rms < 1.0e-05, True)

    def test_lowpassVsPitsa(self):
        """
        Test Butterworth lowpass filter against Butterworth lowpass filter of
        PITSA. Note that the corners value is twice the value of the filter
        sections in PITSA. The rms of the difference between ObsPy and PITSA
        tends to get bigger with higher order filtering.
        """
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.gz')
        f = gzip.open(file)
        data = np.loadtxt(f)
        f.close()
        # parameters for the test
        samp_rate = 200.0
        freq = 5
        corners = 4
        # filter trace
        datcorr = lowpass(data, freq, df=samp_rate, corners=corners)
        # load pitsa file
        file = os.path.join(self.path, 'rjob_20051006_lowpass.gz')
        f = gzip.open(file)
        data_pitsa = np.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) /
                      np.sum(data_pitsa ** 2))
        self.assertEqual(rms < 1.0e-05, True)

    def test_lowpassZPHSHVsPitsa(self):
        """
        Test Butterworth zero-phase lowpass filter against Butterworth
        zero-phase lowpass filter of PITSA. Note that the corners value is
        twice the value of the filter sections in PITSA. The rms of the
        difference between ObsPy and PITSA tends to get bigger with higher
        order filtering.
        Note: The Zero-Phase filters deviate from PITSA's zero-phase filters
        at the end of the trace! The rms for the test is calculated omitting
        the last 200 samples, as this part of the trace is assumed to
        generally be of low interest/importance.
        """
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.gz')
        f = gzip.open(file)
        data = np.loadtxt(f)
        f.close()
        # parameters for the test
        samp_rate = 200.0
        freq = 5
        corners = 2
        # filter trace
        datcorr = lowpass(data, freq, df=samp_rate, corners=corners,
                          zerophase=True)
        # load pitsa file
        file = os.path.join(self.path, 'rjob_20051006_lowpassZPHSH.gz')
        f = gzip.open(file)
        data_pitsa = np.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr[:-200] - data_pitsa[:-200]) ** 2) /
                      np.sum(data_pitsa[:-200] ** 2))
        self.assertEqual(rms < 1.0e-05, True)

    def test_highpassVsPitsa(self):
        """
        Test Butterworth highpass filter against Butterworth highpass filter
        of PITSA. Note that the corners value is twice the value of the filter
        sections in PITSA. The rms of the difference between ObsPy and PITSA
        tends to get bigger with higher order filtering.
        """
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.gz')
        f = gzip.open(file)
        data = np.loadtxt(f)
        f.close()
        # parameters for the test
        samp_rate = 200.0
        freq = 10
        corners = 4
        # filter trace
        datcorr = highpass(data, freq, df=samp_rate, corners=corners)
        # load pitsa file
        file = os.path.join(self.path, 'rjob_20051006_highpass.gz')
        f = gzip.open(file)
        data_pitsa = np.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) /
                      np.sum(data_pitsa ** 2))
        self.assertEqual(rms < 1.0e-05, True)

    def test_highpassZPHSHVsPitsa(self):
        """
        Test Butterworth zero-phase highpass filter against Butterworth
        zero-phase highpass filter of PITSA. Note that the corners value is
        twice the value of the filter sections in PITSA. The rms of the
        difference between ObsPy and PITSA tends to get bigger with higher
        order filtering.
        Note: The Zero-Phase filters deviate from PITSA's zero-phase filters
        at the end of the trace! The rms for the test is calculated omitting
        the last 200 samples, as this part of the trace is assumed to
        generally be of low interest/importance.
        """
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.gz')
        f = gzip.open(file)
        data = np.loadtxt(f)
        f.close()
        # parameters for the test
        samp_rate = 200.0
        freq = 10
        corners = 2
        # filter trace
        datcorr = highpass(data, freq, df=samp_rate, corners=corners,
                           zerophase=True)
        # load pitsa file
        file = os.path.join(self.path, 'rjob_20051006_highpassZPHSH.gz')
        f = gzip.open(file)
        data_pitsa = np.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr[:-200] - data_pitsa[:-200]) ** 2) /
                      np.sum(data_pitsa[:-200] ** 2))
        self.assertEqual(rms < 1.0e-05, True)

    def test_envelopeVsPitsa(self):
        """
        Test Envelope filter against PITSA.
        The rms is not so good, but the fit is still good in most parts.
        """
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.gz')
        f = gzip.open(file)
        data = np.loadtxt(f)
        f.close()
        # filter trace
        datcorr = envelope(data)
        # load pitsa file
        file = os.path.join(self.path, 'rjob_20051006_envelope.gz')
        f = gzip.open(file)
        data_pitsa = np.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) /
                      np.sum(data_pitsa ** 2))
        self.assertEqual(rms < 1.0e-02, True)

    def test_lowpassCheby2(self):
        """
        Check magnitudes of basic lowpass cheby2
        """
        df = 200  # Hz
        b, a = lowpassCheby2(data=None, freq=50,
                             df=df, maxorder=12, ba=True)
        nyquist = 100
        # calculate frequency response
        w, h = sg.freqz(b, a, nyquist)
        freq = w / np.pi * nyquist
        h_db = 20 * np.log10(abs(h))
        # be smaller than -96dB above lowpass frequency
        self.assertTrue(h_db[freq > 50].max() < -96)
        # be 0 (1dB ripple) before filter ramp
        self.assertTrue(h_db[freq < 25].min() > -1)


def suite():
    return unittest.makeSuite(FilterTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
