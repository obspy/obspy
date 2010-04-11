#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Filter test suite.
"""

from obspy.signal import bandpass, lowpass, highpass
from obspy.signal.filter import envelope
import inspect
import os
import unittest
import gzip
import numpy as N


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
        Test Butterworth bandpass filter against Butterworth bandpass filter 
        of PITSA. Note that the corners value is twice the value of the filter 
        sections in PITSA. The rms of the difference between ObsPy and PITSA 
        tends to get bigger with higher order filtering.
        """
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.gz')
        f = gzip.open(file)
        data = N.loadtxt(f)
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
        data_pitsa = N.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = N.sqrt(N.sum((datcorr - data_pitsa) ** 2) / \
                     N.sum(data_pitsa ** 2))
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
        data = N.loadtxt(f)
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
        data_pitsa = N.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = N.sqrt(N.sum((datcorr[:-200] - data_pitsa[:-200]) ** 2) / \
                     N.sum(data_pitsa[:-200] ** 2))
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
        data = N.loadtxt(f)
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
        data_pitsa = N.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = N.sqrt(N.sum((datcorr - data_pitsa) ** 2) /
                     N.sum(data_pitsa ** 2))
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
        data = N.loadtxt(f)
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
        data_pitsa = N.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = N.sqrt(N.sum((datcorr[:-200] - data_pitsa[:-200]) ** 2) / \
                     N.sum(data_pitsa[:-200] ** 2))
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
        data = N.loadtxt(f)
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
        data_pitsa = N.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = N.sqrt(N.sum((datcorr - data_pitsa) ** 2) / \
                     N.sum(data_pitsa ** 2))
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
        data = N.loadtxt(f)
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
        data_pitsa = N.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = N.sqrt(N.sum((datcorr[:-200] - data_pitsa[:-200]) ** 2) / \
                     N.sum(data_pitsa[:-200] ** 2))
        self.assertEqual(rms < 1.0e-05, True)

    def test_envelopeVsPitsa(self):
        """
        Test Envelope filter against PITSA.
        The rms is not so good, but the fit is still good in most parts.
        """
        # load test file
        file = os.path.join(self.path, 'rjob_20051006.gz')
        f = gzip.open(file)
        data = N.loadtxt(f)
        f.close()
        # filter trace
        datcorr = envelope(data)
        # load pitsa file
        file = os.path.join(self.path, 'rjob_20051006_envelope.gz')
        f = gzip.open(file)
        data_pitsa = N.loadtxt(f)
        f.close()
        # calculate normalized rms
        rms = N.sqrt(N.sum((datcorr - data_pitsa) ** 2) / \
                     N.sum(data_pitsa ** 2))
        self.assertEqual(rms < 1.0e-02, True)


def suite():
    return unittest.makeSuite(FilterTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
