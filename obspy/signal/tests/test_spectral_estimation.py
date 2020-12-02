#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The psd test suite.
"""
import gzip
import io
import os
import unittest
import warnings
from copy import deepcopy

import numpy as np
from obspy import Stream, Trace, UTCDateTime, read, read_inventory, Inventory
from obspy.core import Stats
from obspy.core.inventory import Response
from obspy.core.util import NUMPY_VERSION
from obspy.core.util.base import NamedTemporaryFile, MATPLOTLIB_VERSION
from obspy.core.util.obspy_types import ObsPyException
from obspy.core.util.testing import (
    ImageComparison, ImageComparisonException)
from obspy.io.xseed import Parser
from obspy.signal.spectral_estimation import (PPSD, welch_taper, welch_window)
from obspy.signal.spectral_estimation import earthquake_models
from obspy.signal.spectral_estimation import get_idc_infra_low_noise
from obspy.signal.spectral_estimation import get_idc_infra_hi_noise


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


# XXX get rid of if/else again when bumping minimal numpy to 1.10
if NUMPY_VERSION >= [1, 10]:
    allow_pickle = {'allow_pickle': True}
    allow_pickle_false = {'allow_pickle': False}
else:
    allow_pickle = {}
    allow_pickle_false = {}


class PsdTestCase(unittest.TestCase):
    """
    Test cases for psd.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = PATH
        self.path_images = os.path.join(PATH, os.pardir, "images")
        # some pre-computed ppsd used for plotting tests:
        # (ppsd._psd_periods was downcast to np.float16 to save space)
        self.example_ppsd_npz = os.path.join(PATH, "ppsd_kw1_ehz.npz")
        # ignore some "RuntimeWarning: underflow encountered in multiply"
        self.nperr = np.geterr()
        np.seterr(all='ignore')

    def tearDown(self):
        np.seterr(**self.nperr)

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
        from matplotlib.mlab import psd
        sampling_rate = 100.0
        nfft = 512
        noverlap = 0
        file_noise = os.path.join(self.path, "pitsa_noise.npy")
        fn_psd_pitsa = "pitsa_noise_psd_samprate_100_nfft_512_noverlap_0.npy"
        file_psd_pitsa = os.path.join(self.path, fn_psd_pitsa)
        noise = np.load(file_noise, **allow_pickle)
        # in principle to mimic PITSA's results detrend should be specified as
        # some linear detrending (e.g. from matplotlib.mlab.detrend_linear)
        psd_obspy, _ = psd(noise, NFFT=nfft, Fs=sampling_rate,
                           window=welch_taper, noverlap=noverlap,
                           sides="onesided", scale_by_freq=True)

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

        for file, n in zip((file_welch_even, file_welch_odd), (512, 513)):
            window_pitsa = np.load(file)
            window_obspy = welch_window(n)
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
        tr, _paz = _get_sample_data()
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

    def test_ppsd_warnings(self):
        """
        Test some warning messages shown by PPSD routine
        """
        ppsd = _get_ppsd()
        # test warning message if SEED ID is mismatched
        for key in ('network', 'station', 'location', 'channel'):
            tr, _ = _get_sample_data()
            # change starttime, data could then be added if ID and sampling
            # rate match
            tr.stats.starttime += 24 * 3600
            tr.stats[key] = 'XX'
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', UserWarning)
                self.assertEqual(ppsd.add(tr), False)
            self.assertEqual(len(w), 1)
            self.assertEqual(
                str(w[0].message),
                'No traces with matching SEED ID in provided stream object.')
        # test warning message if sampling rate is mismatched
        tr, _ = _get_sample_data()
        # change starttime, data could then be added if ID and sampling
        # rate match
        tr.stats.starttime += 24 * 3600
        tr.stats.sampling_rate = 123
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', UserWarning)
            self.assertEqual(ppsd.add(tr), False)
        self.assertEqual(len(w), 1)
        self.assertEqual(
            str(w[0].message),
            'No traces with matching sampling rate in provided stream object.')

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
            dtype=[("freq", np.float64),
                   ("power", np.int32),
                   ("hits", np.int32)])
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
        results_paz = PPSD.load_npz(filename_paz, metadata=None,
                                    allow_pickle=True)
        filename_full = os.path.join(self.path,
                                     'IUANMO_ppsd_fullresponse.npz')
        results_full = PPSD.load_npz(filename_full, metadata=None,
                                     allow_pickle=True)

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
        for metadata in [parser, inv, resp]:
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
        # change data to nowadays used nanoseconds POSIX timestamp
        ppsd._times_processed = [
            UTCDateTime(t)._ns for t in np.load(
                os.path.join(self.path, "ppsd_times_processed.npy")).tolist()]
        np.random.seed(1234)
        ppsd._binned_psds = [
            arr for arr in np.random.uniform(
                -200, -50,
                (len(ppsd._times_processed), len(ppsd.period_bin_centers)))]

        # Test callback function that selects a fixed random set of the
        # timestamps.  Also checks that we get passed the type we expect,
        # which is 1D numpy ndarray of int type.
        def callback(t_array):
            self.assertIsInstance(t_array, np.ndarray)
            self.assertEqual(t_array.shape, (len(ppsd._times_processed),))
            self.assertTrue(np.issubdtype(t_array.dtype, np.integer))
            np.random.seed(1234)
            res = np.random.randint(0, 2, len(t_array)).astype(np.bool)
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
        # mpl < 2.2 has slightly offset ticks/ticklabels, so needs a higher
        # `reltol` tolerance (see e.g. http://tests.obspy.org/102260)
        reltol = 1.5
        plot_kwargs = dict(max_percentage=15, xaxis_frequency=True,
                           period_lim=(0.01, 50))
        ppsd.calculate_histogram(**stack_criteria_list[1])
        with ImageComparison(self.path_images,
                             'ppsd_restricted_stack.png', reltol=reltol) as ic:
            fig = ppsd.plot(show=False, **plot_kwargs)
            # some matplotlib/Python version combinations lack the left-most
            # tick/label "Jan 2015". Try to circumvent and get the (otherwise
            # OK) test by changing the left x limit a bit further out (by two
            # days, axis is in mpl days). See e.g.
            # https://tests.obspy.org/30657/#1
            fig.axes[1].set_xlim(left=fig.axes[1].get_xlim()[0] - 2)
            with np.errstate(under='ignore'):
                fig.savefig(ic.name)

        # test it again, checking that updating an existing plot with different
        # stack selection works..
        #  a) we start with the stack for the expected image and test that it
        #     matches (like above):
        ppsd.calculate_histogram(**stack_criteria_list[1])
        with ImageComparison(self.path_images,
                             'ppsd_restricted_stack.png', reltol=reltol,
                             plt_close_all_exit=False) as ic:
            fig = ppsd.plot(show=False, **plot_kwargs)
            # some matplotlib/Python version combinations lack the left-most
            # tick/label "Jan 2015". Try to circumvent and get the (otherwise
            # OK) test by changing the left x limit a bit further out (by two
            # days, axis is in mpl days). See e.g.
            # https://tests.obspy.org/30657/#1
            fig.axes[1].set_xlim(left=fig.axes[1].get_xlim()[0] - 2)
            with np.errstate(under='ignore'):
                fig.savefig(ic.name)
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
                ppsd._plot_histogram(fig=fig, draw=True)
                with np.errstate(under='ignore'):
                    fig.savefig(ic.name)
        except ImageComparisonException:
            pass
        else:
            msg = "Expected ImageComparisonException was not raised."
            self.fail(msg)
        #  c) now reuse figure and set the original histogram stack again,
        #     image test should pass agin:
        ppsd.calculate_histogram(**stack_criteria_list[1])
        with ImageComparison(self.path_images,
                             'ppsd_restricted_stack.png', reltol=reltol,
                             plt_close_all_enter=False) as ic:
            ppsd._plot_histogram(fig=fig, draw=True)
            with np.errstate(under='ignore'):
                fig.savefig(ic.name)

    def test_earthquake_models(self):
        """
        Test earthquake models
        """
        ppsd = _get_ppsd()
        test_magnitudes = [3.5, 2.5, 1.5]
        distance = 10
        for magnitude in test_magnitudes:
            key = (magnitude, distance)
            fig = ppsd.plot(
                show_earthquakes=(magnitude - 0.5, magnitude + 0.5, 5, 15),
                show_noise_models=False, show=False)
            ax = fig.axes[0]
            line = ax.lines[0]
            frequencies, accelerations = earthquake_models[key]
            accelerations = np.array(accelerations)
            periods = 1 / np.array(frequencies)
            power = accelerations / (periods ** (-.5))
            power = 20 * np.log10(power / 2)
            self.assertEqual(list(line.get_ydata()), list(power))
            self.assertEqual(list(line.get_xdata()), list(periods))
            caption = 'M%.1f\n%dkm' % (magnitude, distance)
            self.assertEqual(ax.texts[0].get_text(), caption)

    def test_ppsd_infrasound(self):
        """
        Test plotting psds on infrasound data
        """
        wf = os.path.join(
            PATH, 'IM.I59H1..BDF_2020_10_31.mseed')
        md = os.path.join(
            PATH, 'IM.I59H1..BDF_2020_10_31.xml')
        st = read(wf)
        inv = read_inventory(md)
        tr = st[0]
        ppsd = PPSD(tr.stats, metadata=inv, special_handling='infrasound',
                    db_bins=(-100, 40, 1.), ppsd_length=300, overlap=0.5)
        ppsd.add(st)
        fig = ppsd.plot(xaxis_frequency=True,  period_lim=(0.01, 10),
                        show=False)
        models = (get_idc_infra_hi_noise(), get_idc_infra_low_noise())
        lines = fig.axes[0].lines
        freq1 = lines[0].get_xdata()
        per1 = 1 / freq1
        hn = lines[0].get_ydata()
        freq2 = lines[1].get_xdata()
        per2 = 1 / freq2
        ln = lines[1].get_ydata()
        per1_m, hn_m = models[0]
        per2_m, ln_m = models[1]
        np.testing.assert_array_equal(hn, hn_m)
        np.testing.assert_array_equal(ln, ln_m)
        np.testing.assert_array_equal(per1, per1_m)
        np.testing.assert_array_equal(per2, per2_m)
        # test calculated psd values
        psd = ppsd.psd_values[0]
        self.assertEqual(len(psd), 73)
        psd = psd[:20]
        expected = np.array([
            -63.424206, -64.07918, -64.47593, -64.77374, -65.09937,
            -67.17343, -66.36576, -65.75002, -65.34155, -64.58012,
            -63.72327, -62.615784, -61.612656, -61.085754, -60.09534,
            -58.949272, -57.600315, -56.43776, -55.448067, -54.242218],
            dtype=np.float32)
        np.testing.assert_array_almost_equal(psd, expected, decimal=2)

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
        # change data to nowadays used nanoseconds POSIX timestamp
        _times_processed = [UTCDateTime(t)._ns for t in _times_processed]
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

    def test_ppsd_time_checks(self):
        """
        Some tests that make sure checking if a new PSD slice to be addded to
        existing PPSD has an invalid overlap or not works as expected.
        """
        ppsd = PPSD(Stats(), Response())
        one_second = 1000000000
        t0 = 946684800000000000  # 2000-01-01T00:00:00
        time_diffs = [
            0, one_second, one_second * 2, one_second * 3,
            one_second * 8, one_second * 9, one_second * 10]
        ppsd._times_processed = [t0 + td for td in time_diffs]
        ppsd.ppsd_length = 2
        ppsd.overlap = 0.5
        # valid time stamps to insert data for (i.e. data that overlaps with
        # existing data at most "overlap" times "ppsd_length")
        ns_ok = [
            t0 - 3 * one_second,
            t0 - 1.01 * one_second,
            t0 - one_second,
            t0 + 4 * one_second,
            t0 + 4.01 * one_second,
            t0 + 6 * one_second,
            t0 + 7 * one_second,
            t0 + 6.99 * one_second,
            t0 + 11 * one_second,
            t0 + 11.01 * one_second,
            t0 + 15 * one_second,
            ]
        for ns in ns_ok:
            t = UTCDateTime(ns=int(ns))
            # getting False means time is not present yet and a PSD slice would
            # be added to the PPSD data
            self.assertFalse(ppsd._PPSD__check_time_present(t))
        # invalid time stamps to insert data for (i.e. data that overlaps with
        # existing data more than "overlap" times "ppsd_length")
        ns_bad = [
            t0 - 0.99 * one_second,
            t0 - 0.5 * one_second,
            t0,
            t0 + 1.1 * one_second,
            t0 + 3.99 * one_second,
            t0 + 7.01 * one_second,
            t0 + 7.5 * one_second,
            t0 + 8 * one_second,
            t0 + 8.8 * one_second,
            t0 + 10 * one_second,
            t0 + 10.99 * one_second,
            ]
        for ns in ns_bad:
            t = UTCDateTime(ns=int(ns))
            # getting False means time is not present yet and a PSD slice would
            # be added to the PPSD data
            self.assertTrue(ppsd._PPSD__check_time_present(t))

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
        # but raises "UserWarning: Already covered time spans detected"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', UserWarning)
            ppsd.add(st)
            self.assertEqual(len(w), 4)
            for w_ in w:
                self.assertTrue(str(w_.message).startswith(
                    "Already covered time spans detected"))
        self.assertIsNotNone(ppsd._current_hist_stack)
        # Adding new data invalidates the internal stack
        tr.stats.starttime += 3600
        st2 = Stream([tr])
        # raises "UserWarning: Already covered time spans detected"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', UserWarning)
            ppsd.add(st2)
            self.assertEqual(len(w), 2)
            for w_ in w:
                self.assertTrue(str(w_.message).startswith(
                    "Already covered time spans detected"))
        self.assertIsNone(ppsd._current_hist_stack)
        # Accessing current_histogram again calculates the stack
        self.assertIsNotNone(ppsd.current_histogram)
        self.assertIsNotNone(ppsd._current_hist_stack)

    def test_wrong_trace_id_message(self):
        """
        Test that we get the expected warning message on waveform/metadata
        mismatch.
        """
        tr, _paz = _get_sample_data()
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

    def test_ppsd_psd_values(self):
        """
        Test property psd values
        """
        ppsd = _get_ppsd()
        # just test against existing low level data access
        self.assertEqual(ppsd.psd_values, ppsd._binned_psds)
        np.testing.assert_array_equal(ppsd.psd_values, ppsd._binned_psds)
        # property can't be set
        with self.assertRaises(AttributeError):
            ppsd.psd_values = 123

    def test_ppsd_temporal_plot(self):
        """
        Test plot of several period bins over time
        """
        ppsd = PPSD.load_npz(self.example_ppsd_npz, allow_pickle=True)

        restrictions = {'starttime': UTCDateTime(2011, 2, 6, 1, 1),
                        'endtime': UTCDateTime(2011, 2, 7, 21, 12),
                        'year': [2011],
                        'time_of_weekday': [(-1, 2, 23)]}

        # add some gaps in the middle
        for i in sorted(list(range(30, 40)) + list(range(8, 18)) + [4])[::-1]:
            ppsd._times_processed.pop(i)
            ppsd._binned_psds.pop(i)

        with ImageComparison(self.path_images, 'ppsd_temporal.png',
                             reltol=1.5) as ic:
            fig = ppsd.plot_temporal([0.1, 1, 10], filename=None, show=False,
                                     **restrictions)
            fig.savefig(ic.name)
        with ImageComparison(self.path_images, 'ppsd_temporal.png',
                             reltol=1.5) as ic:
            ppsd.plot_temporal([0.1, 1, 10], filename=ic.name, show=False,
                               **restrictions)

    def test_exclude_last_sample(self):
        start = UTCDateTime("2017-01-01T00:00:00")
        header = {
            "starttime": start,
            "network": "GR",
            "station": "FUR",
            "channel": "BHZ"
        }
        # 49 segments of 30 minutes to allow 30 minutes overlap in next day
        tr = Trace(data=np.arange(30 * 60 * 4, dtype=np.int32), header=header)

        ppsd = PPSD(tr.stats, read_inventory())
        ppsd.add(tr)

        self.assertEqual(3, len(ppsd._times_processed))
        self.assertEqual(3600, ppsd.len)
        for i, time in enumerate(ppsd._times_processed):
            current = start.ns + (i * 30 * 60) * 1e9
            self.assertTrue(time == current)

    @unittest.skipIf(MATPLOTLIB_VERSION[0] >= 3,
                     'matplotlib >= 3 shifts labels')
    def test_ppsd_spectrogram_plot(self):
        """
        Test spectrogram type plot of PPSD

        Matplotlib version 3 shifts the x-axis labels but everything else looks
        the same. Skipping test for matplotlib >= 3 on 05/12/2018.
        """
        ppsd = PPSD.load_npz(self.example_ppsd_npz, allow_pickle=True)

        # add some gaps in the middle
        for i in sorted(list(range(30, 40)) + list(range(8, 18)) + [4])[::-1]:
            ppsd._times_processed.pop(i)
            ppsd._binned_psds.pop(i)

        with ImageComparison(self.path_images, 'ppsd_spectrogram.png',
                             reltol=1.5) as ic:
            fig = ppsd.plot_spectrogram(filename=None, show=False)
            fig.savefig(ic.name)
        with ImageComparison(self.path_images, 'ppsd_spectrogram.png',
                             reltol=1.5) as ic:
            ppsd.plot_spectrogram(filename=ic.name, show=False)

    def test_exception_reading_newer_npz(self):
        """
        Checks that an exception is properly raised when trying to read a npz
        that was written on a more recent ObsPy version (specifically that has
        a higher 'ppsd_version' number which is used to keep track of changes
        in PPSD and the npz file used for serialization).
        """
        msg = ("Trying to read/add a PPSD npz with 'ppsd_version=100'. This "
               "file was written on a more recent ObsPy version that very "
               "likely has incompatible changes in PPSD internal structure "
               "and npz serialization. It can not safely be read with this "
               "ObsPy version (current 'ppsd_version' is {!s}). Please "
               "consider updating your ObsPy installation.".format(
                   PPSD(stats=Stats(), metadata=None).ppsd_version))
        # 1 - loading a npz
        data = np.load(self.example_ppsd_npz, **allow_pickle)
        # we have to load, modify 'ppsd_version' and save the npz file for the
        # test..
        items = {key: data[key] for key in data.files}
        # deliberately set a higher ppsd_version number
        items['ppsd_version'] = items['ppsd_version'].copy()
        items['ppsd_version'].fill(100)
        with NamedTemporaryFile() as tf:
            filename = tf.name
            with open(filename, 'wb') as fh:
                np.savez(fh, **items)
            with self.assertRaises(ObsPyException) as e:
                PPSD.load_npz(filename)
        self.assertEqual(str(e.exception), msg)
        # 2 - adding a npz
        ppsd = PPSD.load_npz(self.example_ppsd_npz, allow_pickle=True)
        for method in (ppsd.add_npz, ppsd._add_npz):
            with NamedTemporaryFile() as tf:
                filename = tf.name
                with open(filename, 'wb') as fh:
                    np.savez(fh, **items)
                with self.assertRaises(ObsPyException) as e:
                    method(filename)
                self.assertEqual(str(e.exception), msg)

    def test_nice_ringlaser_metadata_error_msg(self):
        with self.assertRaises(TypeError) as e:
            PPSD(stats=Stats(), metadata=Inventory(networks=[], source=""),
                 special_handling='ringlaser')
        expected = ("When using `special_handling='ringlaser'`, `metadata` "
                    "must be a plain dictionary with key 'sensitivity' "
                    "stating the overall sensitivity`.")
        self.assertEqual(str(e.exception), expected)

    def test_can_read_npz_without_pickle(self):
        """
        Ensures that a default PPSD can be written and read without having to
        allow np.load the use of pickle, or that a helpful error message is
        raised if allow_pickle is required. See #2409.
        """
        # Init a test PPSD and empty byte stream.
        ppsd = PPSD.load_npz(self.example_ppsd_npz, allow_pickle=True)
        byte_me = io.BytesIO()
        # Save PPSD to byte stream and rewind to 0.
        ppsd.save_npz(byte_me)
        byte_me.seek(0)
        # Load dict, will raise an exception if pickle is needed.
        loaded_dict = dict(np.load(byte_me, **allow_pickle_false))
        self.assertIsInstance(loaded_dict, dict)
        # the rest of the test is only relevant on numpy versions that have
        # allow_pickle kwarg (starting with version 1.10.0), older versions
        # will always allow pickle and thus reading works
        if NUMPY_VERSION < [1, 10]:
            return
        # A helpful error message is issued when allow_pickle is needed.
        with self.assertRaises(ValueError) as context:
            PPSD.load_npz(self.example_ppsd_npz)
        self.assertIn('Loading PPSD results', str(context.exception))

    def test_can_add_npz_without_pickle(self):
        """
        Ensure PPSD can be added without using the pickle protocol, or
        that a helpful error message is raised if allow_pickle is required.
        See #2409.
        """

        def _save_nps_require_pickle(filename, ppsd):
            """ Save npz in such a way that requires pickle to load"""
            out = {}
            for key in PPSD.NPZ_STORE_KEYS:
                out[key] = getattr(ppsd, key)
            np.savez_compressed(filename, **out)

        ppsd = _internal_get_ppsd()
        # save PPSD in such a way to mock old versions.
        with NamedTemporaryFile(suffix='.npz') as ntemp:
            temp_path = ntemp.name
            _save_nps_require_pickle(temp_path, ppsd)
            # We should be able to load the files when allowing pickle.
            ppsd.add_npz(temp_path, allow_pickle=True)
            # the rest of the test is only relevant on numpy versions that have
            # allow_pickle kwarg (starting with version 1.10.0), older versions
            # will always allow pickle and thus reading works
            if NUMPY_VERSION < [1, 10]:
                return
            # If not allow_pickle,  a helpful error msg should be raised.
            with self.assertRaises(ValueError) as context:
                ppsd.add_npz(temp_path)
            self.assertIn('Loading PPSD results', str(context.exception))


def suite():
    return unittest.makeSuite(PsdTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
