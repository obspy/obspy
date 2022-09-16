#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Filter test suite.
"""
import gzip
import os
import unittest
import warnings

import numpy as np
import scipy.signal as sg

from obspy import read
from obspy.signal.filter import (bandpass, highpass, lowpass, envelope,
                                 lowpass_cheby_2)


class FilterTestCase(unittest.TestCase):
    """
    Test cases for Filter.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_bandpass_vs_pitsa(self):
        """
        Test Butterworth bandpass filter against Butterworth bandpass filter
        of PITSA. Note that the corners value is twice the value of the filter
        sections in PITSA. The rms of the difference between ObsPy and PITSA
        tends to get bigger with higher order filtering.
        """
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.gz')
        with gzip.open(file) as f:
            data = np.loadtxt(f)
        # parameters for the test
        samp_rate = 200.0
        freq1 = 5
        freq2 = 10
        corners = 4
        # filter trace
        datcorr = bandpass(data, freq1, freq2, df=samp_rate, corners=corners)
        # load pitsa file
        filename = os.path.join(self.path, 'rjob_20051006_bandpass.gz')
        with gzip.open(filename) as f:
            data_pitsa = np.loadtxt(f)
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) /
                      np.sum(data_pitsa ** 2))
        self.assertEqual(rms < 1.0e-05, True)

    def test_bandpass_zphsh_vs_pitsa(self):
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
        filename = os.path.join(self.path, 'rjob_20051006.gz')
        with gzip.open(filename) as f:
            data = np.loadtxt(f)
        # parameters for the test
        samp_rate = 200.0
        freq1 = 5
        freq2 = 10
        corners = 2
        # filter trace
        datcorr = bandpass(data, freq1, freq2, df=samp_rate,
                           corners=corners, zerophase=True)
        # load pitsa file
        filename = os.path.join(self.path, 'rjob_20051006_bandpassZPHSH.gz')
        with gzip.open(filename) as f:
            data_pitsa = np.loadtxt(f)
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr[:-200] - data_pitsa[:-200]) ** 2) /
                      np.sum(data_pitsa[:-200] ** 2))
        self.assertEqual(rms < 1.0e-05, True)

    def test_lowpass_vs_pitsa(self):
        """
        Test Butterworth lowpass filter against Butterworth lowpass filter of
        PITSA. Note that the corners value is twice the value of the filter
        sections in PITSA. The rms of the difference between ObsPy and PITSA
        tends to get bigger with higher order filtering.
        """
        # load test file
        filename = os.path.join(self.path, 'rjob_20051006.gz')
        with gzip.open(filename) as f:
            data = np.loadtxt(f)
        # parameters for the test
        samp_rate = 200.0
        freq = 5
        corners = 4
        # filter trace
        datcorr = lowpass(data, freq, df=samp_rate, corners=corners)
        # load pitsa file
        filename = os.path.join(self.path, 'rjob_20051006_lowpass.gz')
        with gzip.open(filename) as f:
            data_pitsa = np.loadtxt(f)
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) /
                      np.sum(data_pitsa ** 2))
        self.assertEqual(rms < 1.0e-05, True)

    def test_lowpass_zphsh_vs_pitsa(self):
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
        filename = os.path.join(self.path, 'rjob_20051006.gz')
        with gzip.open(filename) as f:
            data = np.loadtxt(f)
        # parameters for the test
        samp_rate = 200.0
        freq = 5
        corners = 2
        # filter trace
        datcorr = lowpass(data, freq, df=samp_rate, corners=corners,
                          zerophase=True)
        # load pitsa file
        filename = os.path.join(self.path, 'rjob_20051006_lowpassZPHSH.gz')
        with gzip.open(filename) as f:
            data_pitsa = np.loadtxt(f)
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr[:-200] - data_pitsa[:-200]) ** 2) /
                      np.sum(data_pitsa[:-200] ** 2))
        self.assertEqual(rms < 1.0e-05, True)

    def test_highpass_vs_pitsa(self):
        """
        Test Butterworth highpass filter against Butterworth highpass filter
        of PITSA. Note that the corners value is twice the value of the filter
        sections in PITSA. The rms of the difference between ObsPy and PITSA
        tends to get bigger with higher order filtering.
        """
        # load test file
        filename = os.path.join(self.path, 'rjob_20051006.gz')
        with gzip.open(filename) as f:
            data = np.loadtxt(f)
        # parameters for the test
        samp_rate = 200.0
        freq = 10
        corners = 4
        # filter trace
        datcorr = highpass(data, freq, df=samp_rate, corners=corners)
        # load pitsa file
        filename = os.path.join(self.path, 'rjob_20051006_highpass.gz')
        with gzip.open(filename) as f:
            data_pitsa = np.loadtxt(f)
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) /
                      np.sum(data_pitsa ** 2))
        self.assertEqual(rms < 1.0e-05, True)

    def test_highpass_zphsh_vs_pitsa(self):
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
        filename = os.path.join(self.path, 'rjob_20051006.gz')
        with gzip.open(filename) as f:
            data = np.loadtxt(f)
        # parameters for the test
        samp_rate = 200.0
        freq = 10
        corners = 2
        # filter trace
        datcorr = highpass(data, freq, df=samp_rate, corners=corners,
                           zerophase=True)
        # load pitsa file
        filename = os.path.join(self.path, 'rjob_20051006_highpassZPHSH.gz')
        with gzip.open(filename) as f:
            data_pitsa = np.loadtxt(f)
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr[:-200] - data_pitsa[:-200]) ** 2) /
                      np.sum(data_pitsa[:-200] ** 2))
        self.assertEqual(rms < 1.0e-05, True)

    def test_envelope_vs_pitsa(self):
        """
        Test Envelope filter against PITSA.
        The rms is not so good, but the fit is still good in most parts.
        """
        # load test file
        filename = os.path.join(self.path, 'rjob_20051006.gz')
        with gzip.open(filename) as f:
            data = np.loadtxt(f)
        # filter trace
        datcorr = envelope(data)
        # load pitsa file
        filename = os.path.join(self.path, 'rjob_20051006_envelope.gz')
        with gzip.open(filename) as f:
            data_pitsa = np.loadtxt(f)
        # calculate normalized rms
        rms = np.sqrt(np.sum((datcorr - data_pitsa) ** 2) /
                      np.sum(data_pitsa ** 2))
        self.assertEqual(rms < 1.0e-02, True)

    def test_lowpass_cheby_2(self):
        """
        Check magnitudes of basic lowpass cheby2
        """
        df = 200  # Hz
        b, a = lowpass_cheby_2(data=None, freq=50,
                               df=df, maxorder=12, ba=True)
        nyquist = 100
        # calculate frequency response
        w, h = sg.freqz(b, a, nyquist)
        freq = w / np.pi * nyquist
        h_db = 20 * np.log10(abs(h))
        # be smaller than -96dB above lowpass frequency
        self.assertGreater(-96, h_db[freq > 50].max())
        # be 0 (1dB ripple) before filter ramp
        self.assertGreater(h_db[freq < 25].min(), -1)

    def test_bandpass_high_corner_at_nyquist(self):
        """
        Check that using exactly Nyquist for high corner gives correct results.
        See #1451.
        """
        tr = read()[0]
        data = tr.data[:1000]

        df = tr.stats.sampling_rate
        nyquist = df / 2.0

        for low_corner in (6.0, 8.55, 8.59):
            for corners in (3, 4, 5, 6):
                # this is filtering with high corner slightly below what we
                # catch and change into highpass
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    expected = bandpass(
                        data, low_corner, nyquist * (1 - 1.1e-6), df=df,
                        corners=corners)
                    self.assertEqual(len(w), 0)
                # all of these should be changed into a highpass
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    got1 = bandpass(data, low_corner, nyquist * (1 - 0.9e-6),
                                    df=df, corners=corners)
                    got2 = bandpass(data, low_corner, nyquist,
                                    df=df, corners=corners)
                    got3 = bandpass(data, low_corner, nyquist + 1.78,
                                    df=df, corners=corners)
                    self.assertEqual(len(w), 3)
                    for w_ in w:
                        self.assertTrue('Selected high corner frequency ' in
                                        str(w[0].message))
                        self.assertTrue('Applying a high-pass instead.' in
                                        str(w[0].message))
                for got in (got1, got2, got3):
                    np.testing.assert_allclose(got, expected, rtol=1e-3,
                                               atol=0.9)
