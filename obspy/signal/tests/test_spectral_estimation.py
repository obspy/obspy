#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The psd test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import gzip
import os
import unittest
import warnings
from copy import deepcopy

import numpy as np

from obspy import Stream, Trace, UTCDateTime, read, read_inventory
from obspy.core import Stats
from obspy.core.util.base import NamedTemporaryFile
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from obspy.core.util.testing import (
    ImageComparison, ImageComparisonException, MATPLOTLIB_VERSION)
from obspy.io.xseed import Parser
from obspy.signal.spectral_estimation import (PPSD, psd, welch_taper,
                                              welch_window)


PATH = os.path.join(os.path.dirname(__file__), 'data')


def _internal_get_sample_data():
    """
    Returns some real data (trace and poles and zeroes) for PPSD testing.

    Data was downsampled to 100Hz so the PPSD is a bit distorted which does
    not matter for the purpose of testing.
    """
    # load test file
    file_data = os.path.join(
        PATH, 'BW.KW1._.EHZ.D.2011.090_downsampled.asc.gz')
    # parameters for the test
    with gzip.open(file_data) as f:
        data = np.loadtxt(f)
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


_sample_data = _internal_get_sample_data()


def _get_sample_data():
    tr, paz = _sample_data
    return tr.copy(), deepcopy(paz)


def _internal_get_ppsd():
    """
    Returns ready computed ppsd for testing purposes.
    """
    tr, paz = _get_sample_data()
    st = Stream([tr])
    ppsd = PPSD(tr.stats, paz, db_bins=(-200, -50, 0.5))
    ppsd.add(st)
    ppsd.calculate_histogram()
    return ppsd


_ppsd = _internal_get_ppsd()


def _get_ppsd():
    return deepcopy(_ppsd)


class PsdTestCase(unittest.TestCase):
    """
    Test cases for psd.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = PATH
        self.path_images = os.path.join(PATH, os.pardir, "images")

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
        sampling_rate = 100.0
        nfft = 512
        noverlap = 0
        file_noise = os.path.join(self.path, "pitsa_noise.npy")
        fn_psd_pitsa = "pitsa_noise_psd_samprate_100_nfft_512_noverlap_0.npy"
        file_psd_pitsa = os.path.join(self.path, fn_psd_pitsa)

        noise = np.load(file_noise)
        # in principle to mimic PITSA's results detrend should be specified as
        # some linear detrending (e.g. from matplotlib.mlab.detrend_linear)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            psd_obspy, _ = psd(noise, NFFT=nfft, Fs=sampling_rate,
                               window=welch_taper, noverlap=noverlap)
            self.assertEqual(len(w), 1)
            self.assertTrue('This wrapper is no longer necessary.' in
                            str(w[0].message))

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

    def test_ppsd(self):
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
        self.assertEqual(len(ppsd.times_processed), 4)
        self.assertEqual(ppsd.nfft, 65536)
        self.assertEqual(ppsd.nlap, 49152)
        np.testing.assert_array_equal(ppsd.current_histogram, result_hist)
        # add the same data a second time (which should do nothing at all) and
        # test again - but it will raise UserWarnings, which we omit for now
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore', UserWarning)
            ppsd.add(st)
            np.testing.assert_array_equal(ppsd.current_histogram, result_hist)
        # test the binning arrays
        binning = np.load(file_binning)
        np.testing.assert_array_equal(ppsd.db_bin_edges, binning['spec_bins'])
        np.testing.assert_array_equal(ppsd.period_bin_centers,
                                      binning['period_bins'])

        # test the mode/mean getter functions
        per_mode, mode = ppsd.get_mode()
        per_mean, mean = ppsd.get_mean()
        result_mode_mean = np.load(file_mode_mean)
        np.testing.assert_array_equal(per_mode, result_mode_mean['per_mode'])
        np.testing.assert_array_equal(mode, result_mode_mean['mode'])
        np.testing.assert_array_equal(per_mean, result_mode_mean['per_mean'])
        np.testing.assert_array_equal(mean, result_mode_mean['mean'])

        # test saving and loading of the PPSD (using a temporary file)
        with NamedTemporaryFile(suffix=".npz") as tf:
            filename = tf.name
            # test saving and loading to npz
            ppsd.save_npz(filename)
            ppsd_loaded = PPSD.load_npz(filename)
            ppsd_loaded.calculate_histogram()
            self.assertEqual(len(ppsd_loaded.times_processed), 4)
            self.assertEqual(ppsd_loaded.nfft, 65536)
            self.assertEqual(ppsd_loaded.nlap, 49152)
            np.testing.assert_array_equal(ppsd_loaded.current_histogram,
                                          result_hist)
            np.testing.assert_array_equal(ppsd_loaded.db_bin_edges,
                                          binning['spec_bins'])
            np.testing.assert_array_equal(ppsd_loaded.period_bin_centers,
                                          binning['period_bins'])

    def test_ppsd_w_iris(self):
        # Bands to be used this is the upper and lower frequency band pairs
        fres = zip([0.1, 0.05], [0.2, 0.1])

        file_data_anmo = os.path.join(self.path, 'IUANMO.seed')
        # Read in ANMO data for one day
        st = read(file_data_anmo)

        # Use a canned ANMO response which will stay static
        paz = {'gain': 86298.5, 'zeros': [0, 0],
               'poles': [-59.4313, -22.7121 + 27.1065j, -22.7121 + 27.1065j,
                         -0.0048004, -0.073199],
               'sensitivity': 3.3554 * 10 ** 9}

        # Make an empty PPSD and add the data
        # use highest frequency given by IRIS Mustang noise-pdf web service
        # (0.475683 Hz == 2.10224036 s) as center of first bin, so that we
        # end up with the same bins.
        ppsd = PPSD(st[0].stats, paz, period_limits=(2.10224036, 1400))
        ppsd.add(st)
        ppsd.calculate_histogram()

        # Get the 50th percentile from the PPSD
        (per, perval) = ppsd.get_percentile(percentile=50)
        perinv = 1 / per

        # Read in the results obtained from a Mustang flat file
        file_data_iris = os.path.join(self.path, 'IRISpdfExample')
        data = np.genfromtxt(
            file_data_iris, comments='#', delimiter=',',
            dtype=[(native_str("freq"), np.float64),
                   (native_str("power"), np.int32),
                   (native_str("hits"), np.int32)])
        freq = data["freq"]
        power = data["power"]
        hits = data["hits"]
        # cut data to same period range as in the ppsd we computed
        # (Mustang returns more long periods, probably due to some zero padding
        # or longer nfft in psd)
        num_periods = len(ppsd.period_bin_centers)
        freqdistinct = np.array(sorted(set(freq), reverse=True)[:num_periods])
        # just make sure that we compare the same periods in the following
        # (as we access both frequency arrays by indices from now on)
        np.testing.assert_allclose(freqdistinct, 1 / ppsd.period_bin_centers,
                                   rtol=1e-4)

        # For each frequency pair we want to compare the mean of the bands
        for fre in fres:
            # determine which bins we want to compare
            mask = (fre[0] < perinv) & (perinv < fre[1])

            # Get the values for the bands from the PPSD
            per_val_good_obspy = perval[mask]

            percenlist = []
            # Now we sort out all of the data from the IRIS flat file
            # We will loop through the frequency values and compute a
            # 50th percentile
            for curfreq in freqdistinct[mask]:
                mask_ = curfreq == freq
                tempvalslist = np.repeat(power[mask_], hits[mask_])
                percenlist.append(np.percentile(tempvalslist, 50))
            # Here is the actual test
            np.testing.assert_allclose(np.mean(per_val_good_obspy),
                                       np.mean(percenlist), rtol=0.0, atol=1.2)

    def test_ppsd_w_iris_against_obspy_results(self):
        """
        Test against results obtained after merging of #1108.
        """
        # Read in ANMO data for one day
        st = read(os.path.join(self.path, 'IUANMO.seed'))

        # Read in metadata in various different formats
        paz = {'gain': 86298.5, 'zeros': [0, 0],
               'poles': [-59.4313, -22.7121 + 27.1065j, -22.7121 + 27.1065j,
                         -0.0048004, -0.073199],
               'sensitivity': 3.3554 * 10 ** 9}
        resp = os.path.join(self.path, 'IUANMO.resp')
        parser = Parser(os.path.join(self.path, 'IUANMO.dataless'))
        inv = read_inventory(os.path.join(self.path, 'IUANMO.xml'))

        # load expected results, for both only PAZ and full response
        filename_paz = os.path.join(self.path, 'IUANMO_ppsd_paz.npz')
        results_paz = PPSD.load_npz(filename_paz, metadata=None)
        filename_full = os.path.join(self.path,
                                     'IUANMO_ppsd_fullresponse.npz')
        results_full = PPSD.load_npz(filename_full, metadata=None)

        # Calculate the PPSDs and test against expected results
        # first: only PAZ
        ppsd = PPSD(st[0].stats, paz)
        ppsd.add(st)
        # commented code to generate the test data:
        # ## np.savez(filename_paz,
        # ##          **dict([(k, getattr(ppsd, k))
        # ##                  for k in PPSD.NPZ_STORE_KEYS]))
        for key in PPSD.NPZ_STORE_KEYS_ARRAY_TYPES:
            np.testing.assert_allclose(
                getattr(ppsd, key), getattr(results_paz, key), rtol=1e-5)
        for key in PPSD.NPZ_STORE_KEYS_LIST_TYPES:
            for got, expected in zip(getattr(ppsd, key),
                                     getattr(results_paz, key)):
                np.testing.assert_allclose(got, expected, rtol=1e-5)
        for key in PPSD.NPZ_STORE_KEYS_SIMPLE_TYPES:
            if key in ["obspy_version", "numpy_version", "matplotlib_version"]:
                continue
            self.assertEqual(getattr(ppsd, key), getattr(results_paz, key))
        # second: various methods for full response
        # (also test various means of initialization, basically testing the
        #  decorator that maps the deprecated keywords)
        for metadata in [parser, inv, resp]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                ppsd = PPSD(st[0].stats, paz=metadata)
            self.assertEqual(len(w), 1)
            self.assertIs(w[0].category, ObsPyDeprecationWarning)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                ppsd = PPSD(st[0].stats, parser=metadata)
            self.assertEqual(len(w), 1)
            self.assertIs(w[0].category, ObsPyDeprecationWarning)

            ppsd = PPSD(st[0].stats, metadata)
            ppsd.add(st)
            # commented code to generate the test data:
            # ## np.savez(filename_full,
            # ##          **dict([(k, getattr(ppsd, k))
            # ##                  for k in PPSD.NPZ_STORE_KEYS]))
            for key in PPSD.NPZ_STORE_KEYS_ARRAY_TYPES:
                np.testing.assert_allclose(
                    getattr(ppsd, key), getattr(results_full, key), rtol=1e-5)
            for key in PPSD.NPZ_STORE_KEYS_LIST_TYPES:
                for got, expected in zip(getattr(ppsd, key),
                                         getattr(results_full, key)):
                    np.testing.assert_allclose(got, expected, rtol=1e-5)
            for key in PPSD.NPZ_STORE_KEYS_SIMPLE_TYPES:
                if key in ["obspy_version", "numpy_version",
                           "matplotlib_version"]:
                    continue
                self.assertEqual(getattr(ppsd, key),
                                 getattr(results_full, key))

    def test_ppsd_save_and_load_npz(self):
        """
        Test PPSD.load_npz() and PPSD.save_npz()
        """
        _, paz = _get_sample_data()
        ppsd = _get_ppsd()

        # save results to npz file
        with NamedTemporaryFile(suffix=".npz") as tf:
            filename = tf.name
            # test saving and loading an uncompressed file
            ppsd.save_npz(filename)
            ppsd_loaded = PPSD.load_npz(filename, metadata=paz)

        for key in PPSD.NPZ_STORE_KEYS:
            if isinstance(getattr(ppsd, key), np.ndarray) or \
                    key == '_binned_psds':
                np.testing.assert_equal(getattr(ppsd, key),
                                        getattr(ppsd_loaded, key))
            else:
                self.assertEqual(getattr(ppsd, key), getattr(ppsd_loaded, key))

    def test_ppsd_restricted_stacks(self):
        """
        Test PPSD.calculate_histogram() with restrictions to what data should
        be stacked. Also includes image tests.
        """
        # set up a bogus PPSD, with fixed random psds but with real start times
        # of psd pieces, to facilitate testing the stack selection.
        ppsd = PPSD(stats=Stats(dict(sampling_rate=150)), metadata=None,
                    db_bins=(-200, -50, 20.), period_step_octaves=1.4)
        ppsd._times_processed = np.load(
            os.path.join(self.path, "ppsd_times_processed.npy")).tolist()
        np.random.seed(1234)
        ppsd._binned_psds = [
            arr for arr in np.random.uniform(
                -200, -50,
                (len(ppsd._times_processed), len(ppsd.period_bin_centers)))]

        # Test callback function that selects a fixed random set of the
        # timestamps.  Also checks that we get passed the type we expect,
        # which is 1D numpy ndarray of float type.
        def callback(t_array):
            self.assertIsInstance(t_array, np.ndarray)
            self.assertEqual(t_array.shape, (len(ppsd._times_processed),))
            self.assertEqual(t_array.dtype, np.float64)
            np.random.seed(1234)
            res = np.random.random_integers(0, 1, len(t_array)).astype(np.bool)
            return res

        # test several different sets of stack criteria, should cover
        # everything, even with lots of combined criteria
        stack_criteria_list = [
            dict(starttime=UTCDateTime(2015, 3, 8), month=[2, 3, 5, 7, 8]),
            dict(endtime=UTCDateTime(2015, 6, 7), year=[2015],
                 time_of_weekday=[(1, 0, 24), (2, 0, 24), (-1, 0, 11)]),
            dict(year=[2013, 2014, 2016, 2017], month=[2, 3, 4]),
            dict(month=[1, 2, 5, 6, 8], year=2015),
            dict(isoweek=[4, 5, 6, 13, 22, 23, 24, 44, 45]),
            dict(time_of_weekday=[(5, 22, 24), (6, 0, 2), (6, 22, 24)]),
            dict(callback=callback, month=[1, 3, 5, 7]),
            dict(callback=callback)]
        expected_selections = np.load(
            os.path.join(self.path, "ppsd_stack_selections.npy"))

        # test every set of criteria
        for stack_criteria, expected_selection in zip(
                stack_criteria_list, expected_selections):
            selection_got = ppsd._stack_selection(**stack_criteria)
            np.testing.assert_array_equal(selection_got, expected_selection)

        # test one particular selection as an image test
        plot_kwargs = dict(max_percentage=15, xaxis_frequency=True,
                           period_lim=(0.01, 50))
        ppsd.calculate_histogram(**stack_criteria_list[1])
        with ImageComparison(self.path_images,
                             'ppsd_restricted_stack.png', reltol=1.5) as ic:
            fig = ppsd.plot(show=False, **plot_kwargs)
            # some matplotlib/Python version combinations lack the left-most
            # tick/label "Jan 2015". Try to circumvent and get the (otherwise
            # OK) test by changing the left x limit a bit further out (by two
            # days, axis is in mpl days). See e.g.
            # https://tests.obspy.org/30657/#1
            fig.axes[1].set_xlim(left=fig.axes[1].get_xlim()[0] - 2)
            _t = np.geterr()
            np.seterr(under="ignore")
            try:
                fig.savefig(ic.name)
            finally:
                np.seterr(**_t)

        # test it again, checking that updating an existing plot with different
        # stack selection works..
        #  a) we start with the stack for the expected image and test that it
        #     matches (like above):
        ppsd.calculate_histogram(**stack_criteria_list[1])
        with ImageComparison(self.path_images,
                             'ppsd_restricted_stack.png', reltol=1.5,
                             plt_close_all_exit=False) as ic:
            fig = ppsd.plot(show=False, **plot_kwargs)
            # some matplotlib/Python version combinations lack the left-most
            # tick/label "Jan 2015". Try to circumvent and get the (otherwise
            # OK) test by changing the left x limit a bit further out (by two
            # days, axis is in mpl days). See e.g.
            # https://tests.obspy.org/30657/#1
            fig.axes[1].set_xlim(left=fig.axes[1].get_xlim()[0] - 2)
            _t = np.geterr()
            np.seterr(under="ignore")
            try:
                fig.savefig(ic.name)
            finally:
                np.seterr(**_t)
        #  b) now reuse figure and set the histogram with a different stack,
        #     image test should fail:
        ppsd.calculate_histogram(**stack_criteria_list[3])
        try:
            with ImageComparison(self.path_images,
                                 'ppsd_restricted_stack.png',
                                 adjust_tolerance=False,
                                 plt_close_all_enter=False,
                                 plt_close_all_exit=False) as ic:
                # rms of the valid comparison above is ~31,
                # rms of the invalid comparison we test here is ~36
                if MATPLOTLIB_VERSION == [1, 1, 1]:
                    ic.tol = 33
                ppsd._plot_histogram(fig=fig, draw=True)
                _t = np.geterr()
                np.seterr(under="ignore")
                try:
                    fig.savefig(ic.name)
                except:
                    np.seterr(**_t)
        except ImageComparisonException:
            pass
        else:
            msg = "Expected ImageComparisonException was not raised."
            self.fail(msg)
        #  c) now reuse figure and set the original histogram stack again,
        #     image test should pass agin:
        ppsd.calculate_histogram(**stack_criteria_list[1])
        with ImageComparison(self.path_images,
                             'ppsd_restricted_stack.png', reltol=1.5,
                             plt_close_all_enter=False) as ic:
            ppsd._plot_histogram(fig=fig, draw=True)
            _t = np.geterr()
            np.seterr(under="ignore")
            try:
                fig.savefig(ic.name)
            except:
                np.seterr(**_t)

    def test_ppsd_add_npz(self):
        """
        Test PPSD.add_npz().
        """
        # set up a bogus PPSD, with fixed random psds but with real start times
        # of psd pieces, to facilitate testing the stack selection.
        ppsd = PPSD(stats=Stats(dict(sampling_rate=150)), metadata=None,
                    db_bins=(-200, -50, 20.), period_step_octaves=1.4)
        _times_processed = np.load(
            os.path.join(self.path, "ppsd_times_processed.npy")).tolist()
        np.random.seed(1234)
        _binned_psds = [
            arr for arr in np.random.uniform(
                -200, -50,
                (len(_times_processed), len(ppsd.period_bin_centers)))]

        with NamedTemporaryFile(suffix=".npz") as tf1, \
                NamedTemporaryFile(suffix=".npz") as tf2, \
                NamedTemporaryFile(suffix=".npz") as tf3:
            # save data split up over three separate temporary files
            ppsd._times_processed = _times_processed[:200]
            ppsd._binned_psds = _binned_psds[:200]
            ppsd.save_npz(tf1.name)
            ppsd._times_processed = _times_processed[200:400]
            ppsd._binned_psds = _binned_psds[200:400]
            ppsd.save_npz(tf2.name)
            ppsd._times_processed = _times_processed[400:]
            ppsd._binned_psds = _binned_psds[400:]
            ppsd.matplotlib_version = "X.X.X"
            ppsd.save_npz(tf3.name)
            # now load these saved npz files and check if all data is present
            ppsd = PPSD.load_npz(tf1.name, metadata=None)
            ppsd.add_npz(tf2.name)
            # we changed a version number so this should emit a warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                ppsd.add_npz(tf3.name)
                self.assertEqual(len(w), 1)
            np.testing.assert_array_equal(_binned_psds, ppsd._binned_psds)
            np.testing.assert_array_equal(_times_processed,
                                          ppsd._times_processed)
            # adding data already present should also emit a warning and the
            # PPSD should not be changed
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                ppsd.add_npz(tf2.name)
                self.assertEqual(len(w), 1)
            np.testing.assert_array_equal(_binned_psds, ppsd._binned_psds)
            np.testing.assert_array_equal(_times_processed,
                                          ppsd._times_processed)

    def test_issue1216(self):
        tr, paz = _get_sample_data()
        st = Stream([tr])
        ppsd = PPSD(tr.stats, paz, db_bins=(-200, -50, 0.5))
        ppsd.add(st)
        # After adding data internal representation of hist stack is None
        self.assertIsNone(ppsd._current_hist_stack)
        # Accessing the current_histogram property calculates the default stack
        self.assertIsNotNone(ppsd.current_histogram)
        self.assertIsNotNone(ppsd._current_hist_stack)
        # Adding the same data again does not invalidate the internal stack
        ppsd.add(st)
        self.assertIsNotNone(ppsd._current_hist_stack)
        # Adding new data invalidates the internal stack
        tr.stats.starttime += 3600
        st2 = Stream([tr])
        ppsd.add(st2)
        self.assertIsNone(ppsd._current_hist_stack)
        # Accessing current_histogram again calculates the stack
        self.assertIsNotNone(ppsd.current_histogram)
        self.assertIsNotNone(ppsd._current_hist_stack)

    def test_wrong_trace_id_message(self):
        """
        Test that we get the expected warning message on waveform/metadata
        mismatch.
        """
        tr, paz = _get_sample_data()
        inv = read_inventory(os.path.join(self.path, 'IUANMO.xml'))
        st = Stream([tr])
        ppsd = PPSD(tr.stats, inv)
        # metadata doesn't fit the trace ID specified via stats
        # should show a warning..
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ret = ppsd.add(st)
            # the trace is sliced into four segments, so we get the warning
            # message four times..
            self.assertEqual(len(w), 4)
            for w_ in w:
                self.assertTrue(str(w_.message).startswith(
                    "Error getting response from provided metadata"))
        # should not add the data to the ppsd
        self.assertFalse(ret)


def suite():
    return unittest.makeSuite(PsdTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
