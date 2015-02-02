#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The psd test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy import Trace, Stream, UTCDateTime, read
from obspy.core.util.base import NamedTemporaryFile
from obspy.signal.spectral_estimation import PPSD, psd, welch_window, \
    welch_taper
import numpy as np
import os
import gzip
import unittest
import warnings


PATH = os.path.join(os.path.dirname(__file__), 'data')


def _get_sample_data():
    """
    Returns some real data (trace and poles and zeroes) for PPSD testing.

    Data was downsampled to 100Hz so the PPSD is a bit distorted which does
    not matter for the purpose of testing.
    """
    # load test file
    file_data = os.path.join(
        PATH, 'BW.KW1._.EHZ.D.2011.090_downsampled.asc.gz')
    # parameters for the test
    # no with due to py 2.6
    f = gzip.open(file_data)
    data = np.loadtxt(f)
    f.close()
    stats = {'_format': 'MSEED',
             'calib': 1.0,
             'channel': 'EHZ',
             'delta': 0.01,
             'endtime': UTCDateTime(2011, 3, 31, 2, 36, 0, 180000),
             'location': '',
             'mseed': {'dataquality': 'D', 'record_length': 512,
                       'encoding': 'STEIM2', 'byteorder': '>'},
             'network': 'BW',
             'npts': 936001,
             'sampling_rate': 100.0,
             'starttime': UTCDateTime(2011, 3, 31, 0, 0, 0, 180000),
             'station': 'KW1'}
    tr = Trace(data, stats)

    paz = {'gain': 60077000.0,
           'poles': [(-0.037004 + 0.037016j), (-0.037004 - 0.037016j),
                     (-251.33 + 0j), (-131.04 - 467.29j),
                     (-131.04 + 467.29j)],
           'sensitivity': 2516778400.0,
           'zeros': [0j, 0j]}

    return tr, paz


def _get_ppsd():
    """
    Returns ready computed ppsd for testing purposes.
    """
    tr, paz = _get_sample_data()
    st = Stream([tr])
    ppsd = PPSD(tr.stats, paz, db_bins=(-200, -50, 0.5))
    ppsd.add(st)
    return ppsd


class PsdTestCase(unittest.TestCase):
    """
    Test cases for psd.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = PATH

    def test_obspy_psd_vs_pitsa(self):
        """
        Test to compare results of PITSA's psd routine to the
        :func:`matplotlib.mlab.psd` routine wrapped in
        :func:`obspy.signal.spectral_estimation.psd`.
        The test works on 8192 samples long Gaussian noise with a standard
        deviation of 0.1 generated with PITSA, sampling rate for processing in
        PITSA was 100.0 Hz, length of nfft 512 samples. The overlap in PITSA
        cannot be controlled directly, instead only the number of overlapping
        segments can be specified.  Therefore the test works with zero overlap
        to have full control over the data segments used in the psd.
        It seems that PITSA has one frequency entry more, i.e. the psd is one
        point longer. I dont know were this can come from, for now this last
        sample in the psd is ignored.
        """
        SAMPLING_RATE = 100.0
        NFFT = 512
        NOVERLAP = 0
        file_noise = os.path.join(self.path, "pitsa_noise.npy")
        fn_psd_pitsa = "pitsa_noise_psd_samprate_100_nfft_512_noverlap_0.npy"
        file_psd_pitsa = os.path.join(self.path, fn_psd_pitsa)

        noise = np.load(file_noise)
        # in principle to mimic PITSA's results detrend should be specified as
        # some linear detrending (e.g. from matplotlib.mlab.detrend_linear)
        psd_obspy, _ = psd(noise, NFFT=NFFT, Fs=SAMPLING_RATE,
                           window=welch_taper, noverlap=NOVERLAP)
        psd_pitsa = np.load(file_psd_pitsa)

        # mlab's psd routine returns Nyquist frequency as last entry, PITSA
        # seems to omit it and returns a psd one frequency sample shorter.
        psd_obspy = psd_obspy[:-1]

        # test results. first couple of frequencies match not as exactly as all
        # the rest, test them separately with a little more allowance..
        np.testing.assert_array_almost_equal(psd_obspy[:3], psd_pitsa[:3],
                                             decimal=4)
        np.testing.assert_array_almost_equal(psd_obspy[1:5], psd_pitsa[1:5],
                                             decimal=5)
        np.testing.assert_array_almost_equal(psd_obspy[5:], psd_pitsa[5:],
                                             decimal=6)

    def test_welch_window_vs_pitsa(self):
        """
        Test that the helper function to generate the welch window delivers the
        same results as PITSA's routine.
        Testing both even and odd values for length of window.
        Not testing strange cases like length <5, though.
        """
        file_welch_even = os.path.join(self.path, "pitsa_welch_window_512.npy")
        file_welch_odd = os.path.join(self.path, "pitsa_welch_window_513.npy")

        for file, N in zip((file_welch_even, file_welch_odd), (512, 513)):
            window_pitsa = np.load(file)
            window_obspy = welch_window(N)
            np.testing.assert_array_almost_equal(window_pitsa, window_obspy)

    def test_PPSD(self):
        """
        Test PPSD routine with some real data.
        """
        # paths of the expected result data
        file_histogram = os.path.join(
            self.path,
            'BW.KW1._.EHZ.D.2011.090_downsampled__ppsd_hist_stack.npy')
        file_binning = os.path.join(
            self.path, 'BW.KW1._.EHZ.D.2011.090_downsampled__ppsd_mixed.npz')
        file_mode_mean = os.path.join(
            self.path,
            'BW.KW1._.EHZ.D.2011.090_downsampled__ppsd_mode_mean.npz')
        tr, paz = _get_sample_data()
        st = Stream([tr])
        ppsd = _get_ppsd()
        # read results and compare
        result_hist = np.load(file_histogram)
        self.assertEqual(len(ppsd.times), 4)
        self.assertEqual(ppsd.nfft, 65536)
        self.assertEqual(ppsd.nlap, 49152)
        np.testing.assert_array_equal(ppsd.hist_stack, result_hist)
        # add the same data a second time (which should do nothing at all) and
        # test again - but it will raise UserWarnings, which we omit for now
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore', UserWarning)
            ppsd.add(st)
            np.testing.assert_array_equal(ppsd.hist_stack, result_hist)
        # test the binning arrays
        binning = np.load(file_binning)
        np.testing.assert_array_equal(ppsd.spec_bins, binning['spec_bins'])
        np.testing.assert_array_equal(ppsd.period_bins, binning['period_bins'])

        # test the mode/mean getter functions
        per_mode, mode = ppsd.get_mode()
        per_mean, mean = ppsd.get_mean()
        result_mode_mean = np.load(file_mode_mean)
        np.testing.assert_array_equal(per_mode, result_mode_mean['per_mode'])
        np.testing.assert_array_equal(mode, result_mode_mean['mode'])
        np.testing.assert_array_equal(per_mean, result_mode_mean['per_mean'])
        np.testing.assert_array_equal(mean, result_mode_mean['mean'])

        # test saving and loading of the PPSD (using a temporary file)
        with NamedTemporaryFile() as tf:
            filename = tf.name
            # test saving and loading an uncompressed file
            ppsd.save(filename, compress=False)
            ppsd_loaded = PPSD.load(filename)
            self.assertEqual(len(ppsd_loaded.times), 4)
            self.assertEqual(ppsd_loaded.nfft, 65536)
            self.assertEqual(ppsd_loaded.nlap, 49152)
            np.testing.assert_array_equal(ppsd_loaded.hist_stack, result_hist)
            np.testing.assert_array_equal(ppsd_loaded.spec_bins,
                                          binning['spec_bins'])
            np.testing.assert_array_equal(ppsd_loaded.period_bins,
                                          binning['period_bins'])
            # test saving and loading a compressed file
            ppsd.save(filename, compress=True)
            ppsd_loaded = PPSD.load(filename)
            self.assertEqual(len(ppsd_loaded.times), 4)
            self.assertEqual(ppsd_loaded.nfft, 65536)
            self.assertEqual(ppsd_loaded.nlap, 49152)
            np.testing.assert_array_equal(ppsd_loaded.hist_stack, result_hist)
            np.testing.assert_array_equal(ppsd_loaded.spec_bins,
                                          binning['spec_bins'])
            np.testing.assert_array_equal(ppsd_loaded.period_bins,
                                          binning['period_bins'])

    def test_PPSD_w_IRIS(self):
        # Bands to be used this is the upper and lower frequency band pairs
        fresup = [0.2, 0.1]
        freslow = [0.1, 0.05]

        file_dataANMO = os.path.join(
            PATH, 'IUANMO.seed')
        # Read in ANMO data for one day
        st = read(file_dataANMO)

        # Use a canned ANMO response which will stay static
        paz = {'gain': 86298.5, 'zeros': [0, 0],
               'poles': [-59.4313, -22.7121 + 27.1065j, -22.7121 + 27.1065j,
                         -0.0048004, -0.073199], 'sensitivity': 3.3554*10**9}

        # Make an empty PPSD and add the data
        ppsd = PPSD(st[0].stats, paz)
        ppsd.add(st)

        # Get the 50th percentile from the PPSD
        (per, perval) = ppsd.get_percentile(percentile=50)

        # For each frequency pair we want to compare the mean of the bands
        for idxfre, curfreUp in enumerate(fresup):
            perGoodOBSPY = []
            pervalGoodOBSPY = []

            # Get the values for the bands from the PPSD
            for idx, curper in enumerate(per):
                if freslow[idxfre] < 1.0/curper < curfreUp:
                    perGoodOBSPY.append(1.0/curper)
                    pervalGoodOBSPY.append(perval[idx])

            # Read in the results obtained from a Mustang flat file
            file_dataIRIS = os.path.join(
                PATH, 'IRISpdfExample')
            fh = open(file_dataIRIS, 'r')

            # We will now read in line by line the IRIS file and save
            # a limited number of values
            # Array of data values that will be within our frequency band
            goodlines = []
            # Distinct frequency values that we will want
            freqdistinct = []
            for line in fh:
                if "#" not in line:
                    freq, power, hits = line.split(",")
                    # Now we look at values between 5 and 20 seconds period
                    if freslow[idxfre] < float(freq) < curfreUp:
                        goodlines.append(line)
                        freqdistinct.append(float(freq))
            fh.close()

            # We now have all of the frequency values of interest
            # We will get the distinct frequency values
            freqdistinct = sorted(list(set(freqdistinct)), reverse=True)

            percenlist = []
            # We will loop through the frequency values and compute a
            # 50th percentile
            for curfreq in freqdistinct:
                tempvals = []
                for curline in goodlines:
                    if curfreq == float(curline.split(",")[0]):
                        tempvals += int(curline.split(",")[2]) * \
                            [int(curline.split(",")[1])]
                percenlist.append(np.percentile(tempvals, 50))
            # Here is the actual test
            np.testing.assert_allclose(np.mean(pervalGoodOBSPY),
                                       np.mean(percenlist), rtol=0.0, atol=1.0)


def suite():
    return unittest.makeSuite(PsdTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
