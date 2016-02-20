#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Filename: spectral_estimation.py
#  Purpose: Various Routines Related to Spectral Estimation
#   Author: Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2011-2012 Tobias Megies
# -----------------------------------------------------------------------------
"""
Various Routines Related to Spectral Estimation

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import bisect
import bz2
import glob
import math
import os
import pickle
import warnings

import numpy as np
from matplotlib import mlab
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import date2num
from matplotlib.mlab import detrend_none, window_hanning
from matplotlib.ticker import FormatStrFormatter

from obspy import Stream, Trace, UTCDateTime, __version__
from obspy.core import Stats
from obspy.imaging.scripts.scan import compress_start_end
from obspy.core.inventory import Inventory
from obspy.core.util import get_matplotlib_version, AttribDict
from obspy.core.util.decorator import deprecated_keywords, deprecated
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from obspy.imaging.cm import obspy_sequential
from obspy.io.xseed import Parser
from obspy.signal.invsim import cosine_taper
from obspy.signal.util import prev_pow_2
from obspy.signal.invsim import paz_to_freq_resp, evalresp


MATPLOTLIB_VERSION = get_matplotlib_version()

dtiny = np.finfo(0.0).tiny

NOISE_MODEL_FILE = os.path.join(os.path.dirname(__file__),
                                "data", "noise_models.npz")


def psd(x, NFFT=256, Fs=2, detrend=detrend_none, window=window_hanning,  # noqa
        noverlap=0):
    """
    Wrapper for :func:`matplotlib.mlab.psd`.

    Always returns a onesided psd (positive frequencies only), corrects for
    this fact by scaling with a factor of 2. Also, always normalizes to 1/Hz
    by dividing with sampling rate.

    .. deprecated:: 0.11.0

        This wrapper is no longer necessary. Please use the
        :func:`matplotlib.mlab.psd` function directly, specifying
        `sides="onesided"` and `scale_by_freq=True`.

    .. note::
        For details on all arguments see :func:`matplotlib.mlab.psd`.

    .. note::
        When using `window=welch_taper`
        (:func:`obspy.signal.spectral_estimation.welch_taper`)
        and `detrend=detrend_linear` (:func:`matplotlib.mlab.detrend_linear`)
        the psd function delivers practically the same results as PITSA.
        Only DC and the first 3-4 lowest non-DC frequencies deviate very
        slightly. In contrast to PITSA, this routine also returns the psd value
        at the Nyquist frequency and therefore is one frequency sample longer.
    """
    msg = ('This wrapper is no longer necessary. Please use the '
           'matplotlib.mlab.psd function directly, specifying '
           '`sides="onesided"` and `scale_by_freq=True`.')
    warnings.warn(msg, ObsPyDeprecationWarning, stacklevel=2)

    # build up kwargs
    kwargs = {}
    kwargs['NFFT'] = NFFT
    kwargs['Fs'] = Fs
    kwargs['detrend'] = detrend
    kwargs['window'] = window
    kwargs['noverlap'] = noverlap
    # These settings make sure that the scaling is already done during the
    # following psd call for matplotlib versions newer than 0.98.4.
    kwargs['pad_to'] = None
    kwargs['sides'] = 'onesided'
    kwargs['scale_by_freq'] = True
    # do the actual call to mlab.psd
    Pxx, freqs = mlab.psd(x, **kwargs)
    return Pxx, freqs


def fft_taper(data):
    """
    Cosine taper, 10 percent at each end (like done by [McNamara2004]_).

    .. warning::
        Inplace operation, so data should be float.
    """
    data *= cosine_taper(len(data), 0.2)
    return data


def welch_taper(data):
    """
    Applies a welch window to data. See
    :func:`~obspy.signal.spectral_estimation.welch_window`.

    .. warning::
        Inplace operation, so data should be float.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to apply the taper to. Inplace operation, but also
        returns data for convenience.
    :returns: Tapered data.
    """
    data *= welch_window(len(data))
    return data


def welch_window(n):
    """
    Return a welch window for data of length n.

    Routine is checked against PITSA for both even and odd values, but not for
    strange values like n<5.

    .. note::
        See e.g.:
        http://www.cescg.org/CESCG99/TTheussl/node7.html

    :type n: int
    :param n: Length of window function.
    :rtype: :class:`~numpy.ndarray`
    :returns: Window function for tapering data.
    """
    n = math.ceil(n / 2.0)
    taper_left = np.arange(n, dtype=np.float64)
    taper_left = 1 - np.power(taper_left / n, 2)
    # first/last sample is zero by definition
    if n % 2 == 0:
        # even number of samples: two ones in the middle, perfectly symmetric
        taper_right = taper_left
    else:
        # odd number of samples: still two ones in the middle, however, not
        # perfectly symmetric anymore. right side is shorter by one sample
        nn = n - 1
        taper_right = np.arange(nn, dtype=np.float64)
        taper_right = 1 - np.power(taper_right / nn, 2)
    taper_left = taper_left[::-1]
    # first/last sample is zero by definition
    taper_left[0] = 0.0
    taper_right[-1] = 0.0
    taper = np.concatenate((taper_left, taper_right))
    return taper


class PPSD(object):
    """
    Class to compile probabilistic power spectral densities for one combination
    of network/station/location/channel/sampling_rate.

    Calculations are based on the routine used by [McNamara2004]_.
    For information on New High/Low Noise Model see [Peterson1993]_.

    .. rubric:: Basic Usage

    >>> from obspy import read
    >>> from obspy.signal import PPSD

    >>> st = read()
    >>> tr = st.select(channel="EHZ")[0]
    >>> paz = {'gain': 60077000.0,
    ...        'poles': [-0.037004+0.037016j, -0.037004-0.037016j,
    ...                  -251.33+0j, -131.04-467.29j, -131.04+467.29j],
    ...        'sensitivity': 2516778400.0,
    ...        'zeros': [0j, 0j]}

    >>> ppsd = PPSD(tr.stats, paz)
    >>> print(ppsd.id)
    BW.RJOB..EHZ
    >>> print(ppsd.times_processed)
    []

    Now we could add data to the probabilistic psd (all processing like
    demeaning, tapering and so on is done internally) and plot it like ...

    >>> ppsd.add(st) # doctest: +SKIP
    >>> print(ppsd.times) # doctest: +SKIP
    >>> ppsd.plot() # doctest: +SKIP

    ... but the example stream is too short and does not contain enough data.

    .. note::

        For a real world example see the `ObsPy Tutorial`_.

    .. rubric:: Saving and Loading

    The PPSD object supports saving to a numpy npz compressed binary file:

    >>> ppsd.save_npz("myfile.npz") # doctest: +SKIP

    The saved PPSD can then be loaded again using the static method
    :func:`~obspy.signal.spectral_estimation.PPSD.load_npz`, e.g. to add more
    data afterwards or to simply plot the results again. Metadata must be
    provided again, since it is not stored in the numpy npz file:

    >>> ppsd = PPSD.load_npz("myfile.npz", metadata=paz)  # doctest: +SKIP

    .. note::

        When using metadata from an
        :class:`~obspy.core.inventory.inventory.Inventory`,
        a :class:`~obspy.io.xseed.parser.Parser` instance or from a RESP file,
        information on metadata will be correctly picked for the respective
        starttime of the data trace. This means that instrument changes are
        correctly taken into account during response removal.
        This is obviously not the case for a static PAZ dictionary!

    .. _`ObsPy Tutorial`: https://docs.obspy.org/tutorial/
    """
    NPZ_STORE_KEYS_LIST_TYPES = [
        # things related to processed data
        '_times_data',
        '_times_gaps',
        '_times_processed',
        '_binned_psds',
        ]
    NPZ_STORE_KEYS_VERSION_NUMBERS = [
        # version numbers
        'ppsd_version',
        'obspy_version',
        'numpy_version',
        'matplotlib_version',
        ]
    NPZ_STORE_KEYS_SIMPLE_TYPES = [
        # things related to Stats passed at __init__
        'id',
        'sampling_rate',
        # kwargs passed during __init__
        'skip_on_gaps',
        'ppsd_length',
        'overlap',
        'special_handling',
        # attributes derived during __init__
        '_len',
        '_nlap',
        '_nfft',
        ]
    NPZ_STORE_KEYS_ARRAY_TYPES = [
        # attributes derived during __init__
        '_db_bin_edges',
        '_psd_periods',
        '_period_binning',
        ]
    NPZ_STORE_KEYS = (
        NPZ_STORE_KEYS_ARRAY_TYPES +
        NPZ_STORE_KEYS_LIST_TYPES +
        NPZ_STORE_KEYS_SIMPLE_TYPES +
        NPZ_STORE_KEYS_VERSION_NUMBERS)

    @deprecated_keywords({'paz': 'metadata', 'parser': 'metadata',
                          'water_level': None})
    def __init__(self, stats, metadata, skip_on_gaps=False,
                 db_bins=(-200, -50, 1.), ppsd_length=3600.0, overlap=0.5,
                 special_handling=None, period_smoothing_width_octaves=1.0,
                 period_step_octaves=0.125, period_limits=None, **kwargs):
        """
        Initialize the PPSD object setting all fixed information on the station
        that should not change afterwards to guarantee consistent spectral
        estimates.
        The necessary instrument response information can be provided in
        several ways using the `metadata` keyword argument:

        * Providing an :class:`~obspy.core.inventory.inventory.Inventory`
          object (e.g. read from a StationXML file using
          :func:`~obspy.core.inventory.inventory.read_inventory` or fetched
          from a :mod:`FDSN <obspy.clients.fdsn>` webservice).
        * Providing an
          :class:`obspy.io.xseed Parser <obspy.io.xseed.parser.Parser>`,
          (e.g. containing metadata from a Dataless SEED file).
        * Providing the filename/path to a local RESP file.
        * Providing a dictionary containing poles and zeros information. Be
          aware that this leads to wrong results if the instrument's response
          is changing over the timespans that are added to the PPSD.
          Use with caution!

        :note: When using `special_handling="ringlaser"` the applied processing
               steps are changed. Differentiation of data (converting velocity
               to acceleration data) will be omitted and a flat instrument
               response is assumed, leaving away response removal and only
               dividing by `metadata['sensitivity']` specified in the provided
               `metadata` dictionary (other keys do not have to be present
               then). For scaling factors that are usually multiplied to the
               data remember to use the inverse as `metadata['sensitivity']`.

        :type stats: :class:`~obspy.core.trace.Stats`
        :param stats: Stats of the station/instrument to process
        :type metadata: :class:`~obspy.core.inventory.inventory.Inventory` or
            :class:`~obspy.io.xseed Parser` or str or dict
        :param metadata: Response information of instrument. See above notes
            for details.
        :type skip_on_gaps: bool, optional
        :param skip_on_gaps: Determines whether time segments with gaps should
                be skipped entirely. [McNamara2004]_ merge gappy
                traces by filling with zeros. This results in a clearly
                identifiable outlier psd line in the PPSD visualization. Select
                `skip_on_gaps=True` for not filling gaps with zeros which might
                result in some data segments shorter than `ppsd_length` not
                used in the PPSD.
        :type db_bins: tuple of three ints/floats
        :param db_bins: Specify the lower and upper boundary and the width of
                the db bins. The bin width might get adjusted to fit  a number
                of equally spaced bins in between the given boundaries.
        :type ppsd_length: float, optional
        :param ppsd_length: Length of data segments passed to psd in seconds.
                In the paper by [McNamara2004]_ a value of 3600 (1 hour) was
                chosen. Longer segments increase the upper limit of analyzed
                periods but decrease the number of analyzed segments.
        :type overlap: float, optional
        :param overlap: Overlap of segments passed to psd. Overlap may take
                values between 0 and 1 and is given as fraction of the length
                of one segment, e.g. `ppsd_length=3600` and `overlap=0.5`
                result in an overlap of 1800s of the segments.
        :type special_handling: str, optional
        :param special_handling: Switches on customized handling for
            data other than seismometer recordings. Can be one of: 'ringlaser'
            (no instrument correction, just division by
            `metadata["sensitivity"]` of provided metadata dictionary),
            'hydrophone' (no differentiation after instrument correction).
        :type period_smoothing_width_octaves: float
        :param period_smoothing_width_octaves: Determines over what
            period/frequency range the psd is smoothed around every central
            period/frequency. Given in fractions of octaves (default of ``1``
            means the psd is averaged over a full octave at each central
            frequency).
        :type period_step_octaves: float
        :param period_step_octaves: Step length on frequency axis in fraction
            of octaves (default of ``0.125`` means one smoothed psd value on
            the frequency axis is measured every 1/8 of an octave).
        :type period_limits: tuple/list of two float
        :param period_limits: Set custom lower and upper end of period range
            (e.g. ``(0.01, 100)``). The specified lower end of period range
            will be set as the central period of the first bin (geometric mean
            of left/right edges of smoothing interval). At the upper end of the
            specified period range, no more additional bins will be added after
            the bin whose center frequency exceeds the given upper end for the
            first time.
        """
        # remove after release of 0.11.0
        if kwargs.pop("is_rotational_data", None) is True:
            msg = ("Keyword 'is_rotational_data' is deprecated. Please use "
                   "'special_handling=\"ringlaser\"' instead.")
            warnings.warn(msg, ObsPyDeprecationWarning)
            special_handling = "ringlaser"

        # save things related to args
        self.id = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats
        self.sampling_rate = stats.sampling_rate
        self.metadata = metadata

        # save things related to kwargs
        self.skip_on_gaps = skip_on_gaps
        self.db_bins = db_bins
        self.ppsd_length = ppsd_length
        self.overlap = overlap
        self.special_handling = special_handling and special_handling.lower()
        if self.special_handling not in (None, "ringlaser", "hydrophone"):
            msg = "Unsupported value for 'special_handling' parameter: %s"
            msg = msg % self.special_handling
            raise ValueError(msg)

        # save version numbers
        self.ppsd_version = 1
        self.obspy_version = __version__
        self.matplotlib_version = ".".join(map(str, MATPLOTLIB_VERSION))
        self.numpy_version = np.__version__

        # calculate derived attributes
        # nfft is determined mimicking the fft setup in McNamara&Buland
        # paper:
        # (they take 13 segments overlapping 75% and truncate to next lower
        #  power of 2)
        #  - take number of points of whole ppsd segment (default 1 hour)
        nfft = self.ppsd_length * self.sampling_rate
        #  - make 13 single segments overlapping by 75%
        #    (1 full segment length + 25% * 12 full segment lengths)
        nfft = nfft / 4.0
        #  - go to next smaller power of 2 for nfft
        self._nfft = prev_pow_2(nfft)
        #  - use 75% overlap
        #    (we end up with a little more than 13 segments..)
        self._nlap = int(0.75 * self.nfft)
        # Trace length for one psd segment.
        self._len = int(self.sampling_rate * self.ppsd_length)
        # make an initial dummy psd and to get the array of periods
        _, freq = mlab.psd(np.ones(self.len), self.nfft,
                           self.sampling_rate, noverlap=self.nlap)
        # leave out first entry (offset)
        freq = freq[1:]
        self._psd_periods = 1.0 / freq[::-1]

        if period_limits is None:
            period_limits = (self.psd_periods[0], self.psd_periods[-1])
        self._setup_period_binning(
            period_smoothing_width_octaves, period_step_octaves, period_limits)
        # setup db binning
        # Set up the binning for the db scale.
        num_bins = int((db_bins[1] - db_bins[0]) / db_bins[2])
        self._db_bin_edges = np.linspace(db_bins[0], db_bins[1],
                                         num_bins + 1, endpoint=True)

        # lists related to persistent processed data
        self._times_processed = []
        self._times_data = []
        self._times_gaps = []
        self._binned_psds = []

        # internal attributes for stacks on processed data
        self._current_hist_stack = None
        self._current_hist_stack_cumulative = None
        self._current_times_used = []
        self._current_times_all_details = []

    @property
    def network(self):
        return self.id.split(".")[0]

    @property
    def station(self):
        return self.id.split(".")[1]

    @property
    def location(self):
        return self.id.split(".")[2]

    @property
    def channel(self):
        return self.id.split(".")[3]

    @property
    def delta(self):
        return 1.0 / self.sampling_rate

    @property
    def len(self):
        """
        Trace length for one psd segment.
        """
        return self._len

    @property
    def nfft(self):
        return self._nfft

    @property
    def nlap(self):
        return self._nlap

    @property
    def merge_method(self):
        if self.skip_on_gaps:
            return -1
        else:
            return 0

    @property
    @deprecated("PPSD attribute 'spec_bins' is deprecated, please use "
                "'db_bin_edges' instead.")
    def spec_bins(self):
        return self.db_bin_edges

    @property
    def db_bin_edges(self):
        return self._db_bin_edges

    @property
    def db_bin_centers(self):
        return (self.db_bin_edges[:-1] + self.db_bin_edges[1:]) / 2.0

    @property
    def psd_frequencies(self):
        return 1.0 / self.psd_periods[::-1]

    @property
    def psd_periods(self):
        return self._psd_periods

    @property
    @deprecated("PPSD attribute 'per' is deprecated, please use "
                "'psd_periods' instead.")
    def per(self):
        return self.psd_periods

    @property
    @deprecated("PPSD attribute 'freq' is deprecated, please use "
                "'psd_frequencies' instead.")
    def freq(self):
        return self.psd_frequencies

    @property
    @deprecated("PPSD attribute 'per_octaves_left' is deprecated, please use "
                "'period_bin_left_edges' instead.")
    def per_octaves_left(self):
        return self.period_bin_left_edges

    @property
    @deprecated("PPSD attribute 'per_octaves_right' is deprecated, please use "
                "'period_bin_right_edges' instead.")
    def per_octaves_right(self):
        return self.period_bin_right_edges

    @property
    @deprecated("PPSD attribute 'per_octaves' is deprecated, please use "
                "'period_bin_centers' instead.")
    def per_octaves(self):
        return self.period_bin_centers

    @property
    @deprecated("PPSD attribute 'period_bins' is deprecated, please use "
                "'period_bin_centers' instead.")
    def period_bins(self):
        return self.period_bin_centers

    @property
    def period_bin_centers(self):
        """
        Return centers of period bins (geometric mean of left and right edge of
        period smoothing ranges).
        """
        return self._period_binning[2, :]

    @property
    def period_xedges(self):
        """
        Returns edges of period histogram bins (one element longer than number
        of bins). These are the edges of the plotted histogram/pcolormesh, but
        not the edges used for smoothing along the period axis of the psd
        (before binning).
        """
        return np.concatenate([self._period_binning[1, 0:1],
                               self._period_binning[3, :]])

    @property
    def period_bin_left_edges(self):
        """
        Returns left edges of period bins (same length as number of bins).
        These are the edges used for smoothing along the period axis of the psd
        (before binning), not the edges of the histogram/pcolormesh in the
        plot.
        """
        return self._period_binning[0, :]

    @property
    def period_bin_right_edges(self):
        """
        Returns right edges of period bins (same length as number of bins).
        These are the edges used for smoothing along the period axis of the psd
        (before binning), not the edges of the histogram/pcolormesh in the
        plot.
        """
        return self._period_binning[4, :]

    @property
    @deprecated("PPSD attribute 'times' is deprecated, please use "
                "'times_processed', 'times_data' and 'times_gaps' instead.")
    def times(self):
        return list(map(UTCDateTime, self._times_processed))

    @property
    def times_processed(self):
        return list(map(UTCDateTime, self._times_processed))

    @property
    @deprecated("PPSD attribute 'times_used' is deprecated, please use "
                "'current_times_used' or 'times_processed' instead.")
    def times_used(self):
        return self.current_times_used

    @property
    def times_data(self):
        return [(UTCDateTime(t1), UTCDateTime(t2))
                for t1, t2 in self._times_data]

    @property
    def times_gaps(self):
        return [(UTCDateTime(t1), UTCDateTime(t2))
                for t1, t2 in self._times_gaps]

    @property
    def current_histogram(self):
        self.__check_histogram()
        return self._current_hist_stack

    @property
    def current_histogram_cumulative(self):
        self.__check_histogram()
        return self._current_hist_stack_cumulative

    @property
    def current_histogram_count(self):
        self.__check_histogram()
        return len(self._current_times_used)

    @property
    def current_times_used(self):
        self.__check_histogram()
        return list(map(UTCDateTime, self._current_times_used))

    def _setup_period_binning(self, period_smoothing_width_octaves,
                              period_step_octaves, period_limits):
        """
        Set up period binning.
        """
        # we step through the period range at step width controlled by
        # period_step_octaves (default 1/8 octave)
        period_step_factor = 2 ** period_step_octaves
        # the width of frequencies we average over for every bin is controlled
        # by period_smoothing_width_octaves (default one full octave)
        period_smoothing_width_factor = \
            2 ** period_smoothing_width_octaves
        # calculate left/right edge and center of first period bin
        # set first smoothing bin's left edge such that the center frequency is
        # the lower limit specified by the user (or the lowest period in the
        # psd)
        per_left = (period_limits[0] /
                    (period_smoothing_width_factor ** 0.5))
        per_right = per_left * period_smoothing_width_factor
        per_center = math.sqrt(per_left * per_right)
        # build up lists
        per_octaves_left = [per_left]
        per_octaves_right = [per_right]
        per_octaves_center = [per_center]
        # do this for the whole period range and append the values to our lists
        while per_center < period_limits[1]:
            # move left edge of smoothing bin further
            per_left *= period_step_factor
            # determine right edge of smoothing bin
            per_right = per_left * period_smoothing_width_factor
            # determine center period of smoothing/binning
            per_center = math.sqrt(per_left * per_right)
            # append to lists
            per_octaves_left.append(per_left)
            per_octaves_right.append(per_right)
            per_octaves_center.append(per_center)
        per_octaves_left = np.array(per_octaves_left)
        per_octaves_right = np.array(per_octaves_right)
        per_octaves_center = np.array(per_octaves_center)
        valid = per_octaves_right > self.psd_periods[0]
        valid &= per_octaves_left < self.psd_periods[-1]
        per_octaves_left = per_octaves_left[valid]
        per_octaves_right = per_octaves_right[valid]
        per_octaves_center = per_octaves_center[valid]
        self._period_binning = np.vstack([
            # left edge of smoothing (for calculating the bin value from psd
            per_octaves_left,
            # left xedge of bin (for plotting)
            per_octaves_center / (period_step_factor ** 0.5),
            # bin center (for plotting)
            per_octaves_center,
            # right xedge of bin (for plotting)
            per_octaves_center * (period_step_factor ** 0.5),
            # right edge of smoothing (for calculating the bin value from psd
            per_octaves_right])

    def __sanity_check(self, trace):
        """
        Checks if trace is compatible for use in the current PPSD instance.
        Returns True if trace can be used or False if not.

        :type trace: :class:`~obspy.core.trace.Trace`
        """
        if trace.id != self.id:
            return False
        if trace.stats.sampling_rate != self.sampling_rate:
            return False
        return True

    def __insert_processed_data(self, utcdatetime, spectrum):
        """
        Inserts the given UTCDateTime and processed/octave-binned spectrum at
        the right position in the lists, keeping the order intact.

        Replaces old :meth:`PPSD.__insert_used_time()` private method and the
        addition ot the histogram stack that was performed directly in
        :meth:`PPSD.__process()`.

        :type utcdatetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :type spectrum: :class:`numpy.ndarray`
        """
        ind = bisect.bisect(self._times_processed, utcdatetime.timestamp)
        self._times_processed.insert(ind, utcdatetime.timestamp)
        self._binned_psds.insert(ind, spectrum)

    def __insert_gap_times(self, stream):
        """
        Gets gap information of stream and adds the encountered gaps to the gap
        list of the PPSD instance.

        :type stream: :class:`~obspy.core.stream.Stream`
        """
        self._times_gaps += [[gap[4].timestamp, gap[5].timestamp]
                             for gap in stream.get_gaps()]

    def __insert_data_times(self, stream):
        """
        Gets gap information of stream and adds the encountered gaps to the gap
        list of the PPSD instance.

        :type stream: :class:`~obspy.core.stream.Stream`
        """
        self._times_data += \
            [[tr.stats.starttime.timestamp, tr.stats.endtime.timestamp]
             for tr in stream]

    def __check_time_present(self, utcdatetime):
        """
        Checks if the given UTCDateTime is already part of the current PPSD
        instance. That is, checks if from utcdatetime to utcdatetime plus
        ppsd_length there is already data in the PPSD.
        Returns True if adding ppsd_length starting at the given time
        would result in an overlap of the ppsd data base, False if it is OK to
        insert this piece of data.
        """
        index1 = bisect.bisect_left(self._times_processed,
                                    utcdatetime.timestamp)
        index2 = bisect.bisect_right(self._times_processed,
                                     utcdatetime.timestamp + self.ppsd_length)
        if index1 != index2:
            return True
        else:
            return False

    def __check_histogram(self):
        # check if any data has been added yet
        if self._current_hist_stack is None:
            if self._times_processed:
                self.calculate_histogram()
            else:
                msg = 'No data accumulated'
                raise Exception(msg)

    def __invalidate_histogram(self):
        self._current_hist_stack = None
        self._current_hist_stack_cumulative = None
        self._current_times_used = []
        self._current_times_all_details = []

    def add(self, stream, verbose=False):
        """
        Process all traces with compatible information and add their spectral
        estimates to the histogram containing the probabilistic psd.
        Also ensures that no piece of data is inserted twice.

        :type stream: :class:`~obspy.core.stream.Stream` or
                :class:`~obspy.core.trace.Trace`
        :param stream: Stream or trace with data that should be added to the
                probabilistic psd histogram.
        :returns: True if appropriate data were found and the ppsd statistics
                were changed, False otherwise.
        """
        if self.metadata is None:
            msg = ("PPSD instance has no metadata attached, which are needed "
                   "for processing the data. When using 'PPSD.load_npz()' use "
                   "'metadata' kwarg to provide metadata.")
            raise Exception(msg)
        # return later if any changes were applied to the ppsd statistics
        changed = False
        # prepare the list of traces to go through
        if isinstance(stream, Trace):
            stream = Stream([stream])
        # select appropriate traces
        stream = stream.select(id=self.id,
                               sampling_rate=self.sampling_rate)
        # save information on available data and gaps
        self.__insert_data_times(stream)
        self.__insert_gap_times(stream)
        # merge depending on skip_on_gaps set during __init__
        stream.merge(self.merge_method, fill_value=0)

        for tr in stream:
            # the following check should not be necessary due to the select()..
            if not self.__sanity_check(tr):
                msg = "Skipping incompatible trace."
                warnings.warn(msg)
                continue
            t1 = tr.stats.starttime
            t2 = tr.stats.endtime
            while t1 + self.ppsd_length <= t2:
                if self.__check_time_present(t1):
                    msg = "Already covered time spans detected (e.g. %s), " + \
                          "skipping these slices."
                    msg = msg % t1
                    warnings.warn(msg)
                else:
                    # throw warnings if trace length is different
                    # than ppsd_length..!?!
                    slice = tr.slice(t1, t1 + self.ppsd_length)
                    # XXX not good, should be working in place somehow
                    # XXX how to do it with the padding, though?
                    success = self.__process(slice)
                    if success:
                        if verbose:
                            print(t1)
                        changed = True
                t1 += (1 - self.overlap) * self.ppsd_length  # advance

            # enforce time limits, pad zeros if gaps
            # tr.trim(t, t+PPSD_LENGTH, pad=True)
        if changed:
            self.__invalidate_histogram()
        return changed

    def __process(self, tr):
        """
        Processes a segment of data and save the psd information.
        Whether `Trace` is compatible (station, channel, ...) has to
        checked beforehand.

        :type tr: :class:`~obspy.core.trace.Trace`
        :param tr: Compatible Trace with data of one PPSD segment
        :returns: `True` if segment was successfully processed,
            `False` otherwise.
        """
        # XXX DIRTY HACK!!
        if len(tr) == self.len + 1:
            tr.data = tr.data[:-1]
        # one last check..
        if len(tr) != self.len:
            msg = "Got a piece of data with wrong length. Skipping"
            warnings.warn(msg)
            print(len(tr), self.len)
            return False
        # being paranoid, only necessary if in-place operations would follow
        tr.data = tr.data.astype(np.float64)
        # if trace has a masked array we fill in zeros
        try:
            tr.data[tr.data.mask] = 0.0
        # if it is no masked array, we get an AttributeError
        # and have nothing to do
        except AttributeError:
            pass

        # restitution:
        # mcnamara apply the correction at the end in freq-domain,
        # does it make a difference?
        # probably should be done earlier on bigger chunk of data?!
        # Yes, you should avoid removing the response until after you
        # have estimated the spectra to avoid elevated lp noise

        spec, _freq = mlab.psd(tr.data, self.nfft, self.sampling_rate,
                               detrend=mlab.detrend_linear, window=fft_taper,
                               noverlap=self.nlap, sides='onesided',
                               scale_by_freq=True)

        # leave out first entry (offset)
        spec = spec[1:]

        # working with the periods not frequencies later so reverse spectrum
        spec = spec[::-1]

        # Here we remove the response using the same conventions
        # since the power is squared we want to square the sensitivity
        # we can also convert to acceleration if we have non-rotational data
        if self.special_handling == "ringlaser":
            # in case of rotational data just remove sensitivity
            spec /= self.metadata['sensitivity'] ** 2
        # special_handling "hydrophone" does instrument correction same as
        # "normal" data
        else:
            # determine instrument response from metadata
            try:
                resp = self._get_response(tr)
            except Exception as e:
                msg = ("Error getting response from provided metadata:\n"
                       "%s: %s\n"
                       "Skipping time segment(s).")
                msg = msg % (e.__class__.__name__, e.message)
                warnings.warn(msg)
                return False

            resp = resp[1:]
            resp = resp[::-1]
            # Now get the amplitude response (squared)
            respamp = np.absolute(resp * np.conjugate(resp))
            # Make omega with the same conventions as spec
            w = 2.0 * math.pi * _freq[1:]
            w = w[::-1]
            # Here we do the response removal
            # Do not differentiate when `special_handling="hydrophone"`
            if self.special_handling == "hydrophone":
                spec = spec / respamp
            else:
                spec = (w ** 2) * spec / respamp
        # avoid calculating log of zero
        idx = spec < dtiny
        spec[idx] = dtiny

        # go to dB
        spec = np.log10(spec)
        spec *= 10

        smoothed_psd = []
        # do this for the whole period range and append the values to our lists
        for per_left, per_right in zip(self.period_bin_left_edges,
                                       self.period_bin_right_edges):
            specs = spec[(per_left <= self.psd_periods) &
                         (self.psd_periods <= per_right)]
            smoothed_psd.append(specs.mean())
        smoothed_psd = np.array(smoothed_psd, dtype=np.float32)
        self.__insert_processed_data(tr.stats.starttime, smoothed_psd)
        return True

    @property
    @deprecated("PPSD attribute 'hist_stack' is deprecated, please use new "
                "'calculate_histogram()' method with improved functionality "
                "instead to compute a histogram stack dynamically and then "
                "'current_histogram' attribute to access the current "
                "histogram stack.")
    def hist_stack(self):
        self.calculate_histogram()
        return self.current_histogram

    def _get_times_all_details(self):
        # check if we can reuse a previously cached array of all times as
        # day of week as int and time of day in float hours
        if len(self._current_times_all_details) == len(self._times_processed):
            return self._current_times_all_details
        # otherwise compute it and store it for subsequent stacks on the
        # same data (has to be recomputed when additional data gets added)
        else:
            dtype = np.dtype([(native_str('time_of_day'), np.float32),
                              (native_str('iso_weekday'), np.int8),
                              (native_str('iso_week'), np.int8),
                              (native_str('year'), np.int16),
                              (native_str('month'), np.int8)])
            times_all_details = np.empty(shape=len(self._times_processed),
                                         dtype=dtype)
            utc_times_all = [UTCDateTime(t) for t in self._times_processed]
            times_all_details['time_of_day'][:] = \
                [t._get_hours_after_midnight() for t in utc_times_all]
            times_all_details['iso_weekday'][:] = \
                [t.isoweekday() for t in utc_times_all]
            times_all_details['iso_week'][:] = \
                [t.isocalendar()[1] for t in utc_times_all]
            times_all_details['year'][:] = \
                [t.year for t in utc_times_all]
            times_all_details['month'][:] = [t.month for t in utc_times_all]
            self._current_times_all_details = times_all_details
            return times_all_details

    def _stack_selection(self, starttime=None, endtime=None,
                         time_of_weekday=None, year=None, month=None,
                         isoweek=None, callback=None):
        """
        For details on restrictions see :meth:`calculate_histogram`.

        :rtype: :class:`numpy.ndarray` of bool
        :returns: Boolean array of which psd pieces should be included in the
            stack.
        """
        times_all = np.array(self._times_processed)
        selected = np.ones(len(times_all), dtype=np.bool)
        if starttime is not None:
            selected &= times_all > starttime.timestamp
        if endtime is not None:
            selected &= times_all < endtime.timestamp
        if time_of_weekday is not None:
            times_all_details = self._get_times_all_details()
            # we need to do a logical OR over all different user specified time
            # windows, so we start with an array of False and set all matching
            # pieces True for the final logical AND against the previous
            # restrictions
            selected_time_of_weekday = np.zeros(len(times_all), dtype=np.bool)
            for weekday, start, end in time_of_weekday:
                if weekday == -1:
                    selected_ = np.ones(len(times_all), dtype=np.bool)
                else:
                    selected_ = (
                        times_all_details['iso_weekday'] == weekday)
                selected_ &= times_all_details['time_of_day'] > start
                selected_ &= times_all_details['time_of_day'] < end
                selected_time_of_weekday |= selected_
            selected &= selected_time_of_weekday
        if year is not None:
            try:
                year[0]
            except TypeError:
                year = [year]
            times_all_details = self._get_times_all_details()
            selected_ = times_all_details['year'] == year[0]
            for year_ in year[1:]:
                selected_ |= times_all_details['year'] == year_
            selected &= selected_
        if month is not None:
            try:
                month[0]
            except TypeError:
                month = [month]
            times_all_details = self._get_times_all_details()
            selected_ = times_all_details['month'] == month[0]
            for month_ in month[1:]:
                selected_ |= times_all_details['month'] == month_
            selected &= selected_
        if isoweek is not None:
            times_all_details = self._get_times_all_details()
            selected_ = times_all_details['iso_week'] == isoweek[0]
            for isoweek_ in isoweek[1:]:
                selected_ |= times_all_details['iso_week'] == isoweek_
            selected &= selected_
        if callback is not None:
            selected &= callback(times_all)
        return selected

    def calculate_histogram(self, starttime=None, endtime=None,
                            time_of_weekday=None, year=None, month=None,
                            isoweek=None, callback=None):
        """
        Calculate and set current 2D histogram stack, optionally with start-
        and endtime and time of day restrictions.

        .. note::
            All restrictions to the stack are evaluated as a logical AND, i.e.
            only individual psd pieces are included in the stack that match
            *all* of the specified restrictions (e.g. `isoweek=40, month=2` can
            never match any data).

        .. note::
            All time restrictions are specified in UTC, so actual time in local
            time zone might not be the same across start/end date of daylight
            saving time periods.

        .. note::
            Time restrictions only check the starttime of the individual psd
            pieces.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: If set, data before the specified time is excluded
            from the returned stack.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: If set, data after the specified time is excluded
            from the returned stack.
        :type time_of_weekday: list of (int, float, float) 3-tuples
        :param time_of_weekday: If set, restricts the data that is included
            in the stack by time of day and weekday. Monday is `1`, Sunday is
            `7`, `-1` for any day of week. For example, using
            `time_of_weekday=[(-1, 0, 2), (-1, 22, 24)]` only individual
            spectra that have a starttime in between 10pm and 2am are used in
            the stack for all days of week, using
            `time_of_weekday=[(5, 22, 24), (6, 0, 2), (6, 22, 24), (7, 0, 2)]`
            only spectra with a starttime in between Friday 10pm to Saturdays
            2am and Saturday 10pm to Sunday 2am are used.
            Note that time of day is specified in UTC (time of day might have
            to be adapted to daylight saving time). Also note that this setting
            filters only by starttime of the used psd time slice, so the length
            of individual slices (set at initialization:
            :meth:`PPSD(..., ppsd_length=XXX, ...) <PPSD.__init__>` in seconds)
            has to be taken into consideration (e.g. with a `ppsd_length` of
            one hour and a `time_of_weekday` restriction to 10pm-2am
            actually includes data from 10pm-3am).
        :type year: list of int
        :param year: If set, restricts the data that is included in the stack
            by year. For example, using `year=[2015]` only individual spectra
            from year 2015 are used in the stack, using `year=[2013, 2015]`
            only spectra from exactly year 2013 or exactly year 2015 are used.
        :type month: list of int
        :param month: If set, restricts the data that is included in the stack
            by month of year. For example, using `month=[2]` only individual
            spectra from February are used in the stack, using `month=[4, 7]`
            only spectra from exactly April or exactly July are used.
        :type isoweek: list of int
        :param isoweek: If set, restricts the data that is included in the
            stack by ISO week number of year. For example, using `isoweek=[2]`
            only individual spectra from 2nd ISO week of any year are used in
            the stack, using `isoweek=[4, 7]` only spectra from exactly 4th ISO
            week or exactly 7th ISO week are used.
        :type callback: func
        :param callback: Custom user defined callback function that can be used
            for more complex scenarios to specify whether an individual psd
            piece should be included in the stack or not. The function will be
            passed an array with the starttimes of all psd pieces (as a POSIX
            timestamp that can be used as a single argument to initialize a
            :class:`~obspy.core.utcdatetime.UTCDateTime` object) and
            should return a boolean array specifying which psd pieces should be
            included in the stack (`True`) and which should be excluded
            (`False`). Note that even when callback returns `True` the psd
            piece will be excluded if it does not match all other criteria
            (e.g. `starttime`).
        :rtype: None
        """
        # no data at all
        if not self._times_processed:
            self._current_hist_stack = None
            self._current_hist_stack_cumulative = None
            self._current_times_used = []
            return

        # determine which psd pieces should be used in the stack,
        # based on all selection criteria specified by user
        selected = self._stack_selection(
            starttime=starttime, endtime=endtime,
            time_of_weekday=time_of_weekday, year=year, month=month,
            isoweek=isoweek, callback=callback)
        used_indices = selected.nonzero()[0]
        used_count = len(used_indices)
        used_times = np.array(self._times_processed)[used_indices]

        num_period_bins = len(self.period_bin_centers)
        num_db_bins = len(self.db_bin_centers)

        # initial setup of 2D histogram
        hist_stack = np.zeros((num_period_bins, num_db_bins), dtype=np.uint64)

        # empty selection, set all histogram stacks to zeros
        if not used_count:
            self._current_hist_stack = hist_stack
            self._current_hist_stack_cumulative = np.zeros_like(
                hist_stack, dtype=np.float32)
            self._current_times_used = used_times
            return

        # concatenate all used spectra, evaluate index of amplitude bin each
        # value belongs to
        inds = np.hstack([self._binned_psds[i] for i in used_indices])
        # for "inds" now a number of ..
        #   - 0 means below lowest bin (bin index 0)
        #   - 1 means, hit lowest bin (bin index 0)
        #   - ..
        #   - len(self.db_bin_edges) means above top bin
        # we need minus one because searchsorted returns the insertion index in
        # the array of bin edges which is the index of the corresponding bin
        # plus one
        inds = self.db_bin_edges.searchsorted(inds, side="left") - 1
        # for "inds" now a number of ..
        #   - -1 means below lowest bin (bin index 0)
        #   - 0 means, hit lowest bin (bin index 0)
        #   - ..
        #   - (len(self.db_bin_edges)-1) means above top bin
        # values that are left of first bin edge have to be moved back into the
        # binning
        inds[inds == -1] = 0
        # same goes for values right of last bin edge
        inds[inds == num_db_bins] -= 1
        # reshape such that we can iterate over the array, extracting for
        # each period bin an array of all amplitude bins we have hit
        inds = inds.reshape((used_count, num_period_bins)).T
        for i, inds_ in enumerate(inds):
            # count how often each bin has been hit for this period bin,
            # set the current 2D histogram column accordingly
            hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)

        # calculate and set the cumulative version (i.e. going from 0 to 1 from
        # low to high psd values for every period column) of the current
        # histogram stack.
        # sum up the columns to cumulative entries
        hist_stack_cumul = hist_stack.cumsum(axis=1)
        # normalize every column with its overall number of entries
        # (can vary from the number of self.times because of values outside
        #  the histogram db ranges)
        norm = hist_stack_cumul[:, -1].copy().astype(np.float64)
        # avoid zero division
        norm[norm == 0] = 1
        hist_stack_cumul = (hist_stack_cumul.T / norm).T
        # set everything that was calculated
        self._current_hist_stack = hist_stack
        self._current_hist_stack_cumulative = hist_stack_cumul
        self._current_times_used = used_times

    def _get_response(self, tr):
        # check type of metadata and use the correct subroutine
        # first, to save some time, tried to do this in __init__ like:
        #   self._get_response = self._get_response_from_inventory
        # but that makes the object non-picklable
        if isinstance(self.metadata, Inventory):
            return self._get_response_from_inventory(tr)
        elif isinstance(self.metadata, Parser):
            return self._get_response_from_parser(tr)
        elif isinstance(self.metadata, dict):
            return self._get_response_from_paz_dict(tr)
        elif isinstance(self.metadata, (str, native_str)):
            return self._get_response_from_resp(tr)
        else:
            msg = "Unexpected type for `metadata`: %s" % type(metadata)
            raise TypeError(msg)

    def _get_response_from_inventory(self, tr):
        inventory = self.metadata
        response = inventory.get_response(self.id, tr.stats.starttime)
        resp, _ = response.get_evalresp_response(
            t_samp=self.delta, nfft=self.nfft, output="VEL")
        return resp

    def _get_response_from_parser(self, tr):
        parser = self.metadata
        resp_key = "RESP." + self.id
        for key, resp_file in parser.get_resp():
            if key == resp_key:
                break
        else:
            msg = "Response for %s not found in Parser" % self.id
            raise ValueError(msg)
        resp_file.seek(0, 0)
        resp = evalresp(t_samp=self.delta, nfft=self.nfft,
                        filename=resp_file, date=tr.stats.starttime,
                        station=self.station, channel=self.channel,
                        network=self.network, locid=self.location,
                        units="VEL", freq=False, debug=False)
        return resp

    def _get_response_from_paz_dict(self, tr):
        paz = self.metadata
        resp = paz_to_freq_resp(paz['poles'], paz['zeros'],
                                paz['gain'] * paz['sensitivity'],
                                self.delta, nfft=self.nfft)
        return resp

    def _get_response_from_resp(self, tr):
        resp = evalresp(t_samp=self.delta, nfft=self.nfft,
                        filename=self.metadata, date=tr.stats.starttime,
                        station=self.station, channel=self.channel,
                        network=self.network, locid=self.location,
                        units="VEL", freq=False, debug=False)
        return resp

    def get_percentile(self, percentile=50):
        """
        Returns periods and approximate psd values for given percentile value.

        :type percentile: int
        :param percentile: percentile for which to return approximate psd
                value. (e.g. a value of 50 is equal to the median.)
        :returns: (periods, percentile_values)
        """
        hist_cum = self.current_histogram_cumulative
        if hist_cum is None:
            return None
        # go to percent
        percentile = percentile / 100.0
        if percentile == 0:
            # only for this special case we have to search from the other side
            # (otherwise we always get index 0 in .searchsorted())
            side = "right"
        else:
            side = "left"
        percentile_values = [col.searchsorted(percentile, side=side)
                             for col in hist_cum]
        # map to power db values
        percentile_values = self.db_bin_edges[percentile_values]
        return (self.period_bin_centers, percentile_values)

    def get_mode(self):
        """
        Returns periods and mode psd values (i.e. for each frequency the psd
        value with the highest probability is selected).

        :returns: (periods, psd mode values)
        """
        hist = self.current_histogram
        if hist is None:
            return None
        mode = self.db_bin_centers[hist.argmax(axis=1)]
        return (self.period_bin_centers, mode)

    def get_mean(self):
        """
        Returns periods and mean psd values (i.e. for each frequency the mean
        psd value is selected).

        :returns: (periods, psd mean values)
        """
        hist = self.current_histogram
        if hist is None:
            return None
        hist_count = self.current_histogram_count
        mean = (hist * self.db_bin_centers / hist_count).sum(axis=1)
        return (self.period_bin_centers, mean)

    @deprecated("Old save/load mechanism based on pickle module is not "
                "working well across versions, so please use new "
                "'save_npz'/'load_npz' mechanism.")
    def save(self, filename, compress=False):
        """
        DEPRECATED! Use :meth:`~PPSD.save_npz` and :meth:`~PPSD.load_npz`
        instead!
        """
        if compress:
            # due to an bug in older python version we can't use with
            # https://bugs.python.org/issue8601
            file_ = bz2.BZ2File(filename, 'wb')
            pickle.dump(self, file_)
            file_.close()
        else:
            with open(filename, 'wb') as file_:
                pickle.dump(self, file_)

    @staticmethod
    @deprecated("Old save/load mechanism based on pickle module is not "
                "working well across versions, so please use new "
                "'save_npz'/'load_npz' mechanism.")
    def load(filename):
        """
        DEPRECATED! Use :meth:`~PPSD.save_npz` and :meth:`~PPSD.load_npz`
        instead!
        """
        # identify bzip2 compressed file using bzip2's magic number
        bz2_magic = b'\x42\x5a\x68'
        with open(filename, 'rb') as file_:
            file_start = file_.read(len(bz2_magic))

        if file_start == bz2_magic:
            # In theory a file containing random data could also start with the
            # bzip2 magic number. However, since save() (implicitly) uses
            # version "0" of the pickle protocol, the pickled data is
            # guaranteed to be ASCII encoded and hence cannot start with this
            # magic number.
            # cf. https://docs.python.org/3/library/pickle.html
            #
            # due to an bug in older python version we can't use with
            # https://bugs.python.org/issue8601
            file_ = bz2.BZ2File(filename, 'rb')
            ppsd = pickle.load(file_)
            file_.close()
        else:
            with open(filename, 'rb') as file_:
                ppsd = pickle.load(file_)

        # some workarounds for older PPSD pickle files
        if hasattr(ppsd, "is_rotational_data"):
            if ppsd.is_rotational_data is True:
                ppsd.special_handling = "ringlaser"
            delattr(ppsd, "is_rotational_data")
        if not hasattr(ppsd, "special_handling"):
            ppsd.special_handling = None
        # Adds ppsd_length and overlap attributes if not existing. This
        # ensures compatibility with pickled objects without these attributes.
        try:
            self.ppsd_length
            self.overlap
        except AttributeError:
            self.ppsd_length = 3600.
            self.overlap = 0.5

        return ppsd

    def save_npz(self, filename):
        """
        Saves the PPSD as a compressed numpy binary (npz format).

        The resulting file can be restored using `my_ppsd.load_npz(filename)`.

        :type filename: str
        :param filename: Name of numpy .npz output file
        """
        out = dict([(key, getattr(self, key)) for key in self.NPZ_STORE_KEYS])
        np.savez_compressed(filename, **out)

    @staticmethod
    def load_npz(filename, metadata=None):
        """
        Load previously computed PPSD results.

        Load previously computed PPSD results from a
        compressed numpy binary in npz format, written with
        :meth:`~PPSD.write_npz`.
        If more data are to be added and processed, metadata have to be
        specified again during loading because they are not
        stored in the npz format.

        :type filename: str
        :param filename: Name of numpy .npz file with stored PPSD data
        :type metadata: :class:`~obspy.core.inventory.inventory.Inventory` or
            :class:`~obspy.io.xseed Parser` or str or dict
        :param metadata: Response information of instrument. See notes in
            :meth:`PPSD.__init__` for details.
        """
        data = np.load(filename)
        # the information regarding stats is set from the npz
        ppsd = PPSD(Stats(), metadata=metadata)
        for key in ppsd.NPZ_STORE_KEYS:
            # data is stored as arrays in the npz.
            # we have to convert those back to lists (or simple types), so that
            # additionally processed data can be appended/inserted later.
            data_ = data[key]
            if key in ppsd.NPZ_STORE_KEYS_LIST_TYPES:
                if key in ['_times_data', '_times_gaps']:
                    data_ = data_.tolist()
                else:
                    data_ = [d for d in data_]
            elif key in (ppsd.NPZ_STORE_KEYS_SIMPLE_TYPES +
                         ppsd.NPZ_STORE_KEYS_VERSION_NUMBERS):
                data_ = data_.item()
            setattr(ppsd, key, data_)
        return ppsd

    def add_npz(self, filename):
        """
        Add previously computed PPSD results to current PPSD instance.

        Load previously computed PPSD results from a
        compressed numpy binary in npz format, written with
        :meth:`~PPSD.write_npz` and add the information to the current PPSD
        instance.
        Before adding the data it is checked if the data was computed with the
        same settings, then any time periods that are not yet covered are added
        to the current PPSD (a warning is emitted if any segments are omitted).

        :type filename: str
        :param filename: Name of numpy .npz file(s) with stored PPSD data.
            Wildcards are possible and will be expanded using
            :py:func:`glob.glob`.
        """
        for filename in glob.glob(filename):
            self._add_npz(filename)

    def _add_npz(self, filename):
        """
        See :meth:`PPSD.add_npz()`.
        """
        data = np.load(filename)
        # check if all metadata agree
        for key in self.NPZ_STORE_KEYS_SIMPLE_TYPES:
            if getattr(self, key) != data[key].item():
                msg = ("Mismatch in '%s' attribute.\n\tCurrent:\n\t%s\n\t"
                       "Loaded:\n\t%s")
                msg = msg % (key, getattr(self, key), data[key].item())
                raise AssertionError(msg)
        for key in self.NPZ_STORE_KEYS_ARRAY_TYPES:
            try:
                np.testing.assert_array_equal(getattr(self, key), data[key])
            except AssertionError as e:
                msg = ("Mismatch in '%s' attribute.\n") % key
                raise AssertionError(msg + str(e))
        # load new psd data
        for key in self.NPZ_STORE_KEYS_VERSION_NUMBERS:
            if getattr(self, key) != data[key].item():
                msg = ("Mismatch in version numbers (%s) between current data "
                       "(%s) and loaded data (%s).") % (
                           key, getattr(self, key), data[key].item())
                warnings.warn(msg)
        _times_data = data["_times_data"].tolist()
        _times_gaps = data["_times_gaps"].tolist()
        _times_processed = [d_ for d_ in data["_times_processed"]]
        _binned_psds = [d_ for d_ in data["_binned_psds"]]
        # add new data
        self._times_data.extend(_times_data)
        self._times_gaps.extend(_times_gaps)
        duplicates = 0
        for t, psd in zip(_times_processed, _binned_psds):
            t = UTCDateTime(t)
            if self.__check_time_present(t):
                duplicates += 1
                continue
            self.__insert_processed_data(t, psd)
        # warn if some segments were omitted
        if duplicates:
            msg = ("%d/%d segments omitted in file '%s' "
                   "(time ranges already covered).")
            msg = msg % (duplicates, len(_times_processed), filename)
            warnings.warn(msg)

    def plot(self, filename=None, show_coverage=True, show_histogram=True,
             show_percentiles=False, percentiles=[0, 25, 50, 75, 100],
             show_noise_models=True, grid=True, show=True,
             max_percentage=None, period_lim=(0.01, 179), show_mode=False,
             show_mean=False, cmap=obspy_sequential, cumulative=False,
             cumulative_number_of_colors=20, xaxis_frequency=False):
        """
        Plot the 2D histogram of the current PPSD.
        If a filename is specified the plot is saved to this file, otherwise
        a plot window is shown.

        :type filename: str, optional
        :param filename: Name of output file
        :type show_coverage: bool, optional
        :param show_coverage: Enable/disable second axes with representation of
                data coverage time intervals.
        :type show_percentiles: bool, optional
        :param show_percentiles: Enable/disable plotting of approximated
                percentiles. These are calculated from the binned histogram and
                are not the exact percentiles.
        :type show_histogram: bool, optional
        :param show_histogram: Enable/disable plotting of histogram. This
                can be set ``False`` e.g. to make a plot with only percentiles
                plotted. Defaults to ``True``.
        :type percentiles: list of ints
        :param percentiles: percentiles to show if plotting of percentiles is
                selected.
        :type show_noise_models: bool, optional
        :param show_noise_models: Enable/disable plotting of noise models.
        :type grid: bool, optional
        :param grid: Enable/disable grid in histogram plot.
        :type show: bool, optional
        :param show: Enable/disable immediately showing the plot.
        :type max_percentage: float, optional
        :param max_percentage: Maximum percentage to adjust the colormap. The
            default is 30% unless ``cumulative=True``, in which case this value
            is ignored.
        :type period_lim: tuple of 2 floats, optional
        :param period_lim: Period limits to show in histogram. When setting
            ``xaxis_frequency=True``, this is expected to be frequency range in
            Hz.
        :type show_mode: bool, optional
        :param show_mode: Enable/disable plotting of mode psd values.
        :type show_mean: bool, optional
        :param show_mean: Enable/disable plotting of mean psd values.
        :type cmap: :class:`matplotlib.colors.Colormap`
        :param cmap: Colormap to use for the plot. To use the color map like in
            PQLX, [McNamara2004]_ use :const:`obspy.imaging.cm.pqlx`.
        :type cumulative: bool
        :param cumulative: Can be set to `True` to show a cumulative
            representation of the histogram, i.e. showing color coded for each
            frequency/amplitude bin at what percentage in time the value is
            not exceeded by the data (similar to the `percentile` option but
            continuously and color coded over the whole area). `max_percentage`
            is ignored when this option is specified.
        :type cumulative_number_of_colors: int
        :param cumulative_number_of_colors: Number of discrete color shades to
            use, `None` for a continuous colormap.
        :type xaxis_frequency: bool
        :param xaxis_frequency: If set to `True`, the x axis will be frequency
            in Hertz as opposed to the default of period in seconds.
        """
        import matplotlib.pyplot as plt
        self.__check_histogram()
        fig = plt.figure()
        fig.ppsd = AttribDict()

        if show_coverage:
            ax = fig.add_axes([0.12, 0.3, 0.90, 0.6])
            ax2 = fig.add_axes([0.15, 0.17, 0.7, 0.04])
        else:
            ax = fig.add_subplot(111)

        if show_percentiles:
            # for every period look up the approximate place of the percentiles
            for percentile in percentiles:
                periods, percentile_values = \
                    self.get_percentile(percentile=percentile)
                ax.plot(periods, percentile_values, color="black", zorder=8)

        if show_mode:
            periods, mode_ = self.get_mode()
            if cmap.name == "viridis":
                color = "0.8"
            else:
                color = "black"
            ax.plot(periods, mode_, color=color, zorder=9)

        if show_mean:
            periods, mean_ = self.get_mean()
            if cmap.name == "viridis":
                color = "0.8"
            else:
                color = "black"
            ax.plot(periods, mean_, color=color, zorder=9)

        if show_noise_models:
            for periods, noise_model in (get_nhnm(), get_nlnm()):
                if xaxis_frequency:
                    periods = 1.0 / periods
                ax.plot(periods, noise_model, '0.4', linewidth=2, zorder=10)

        if show_histogram:
            label = "[%]"
            if cumulative:
                label = "non-exceedance (cumulative) [%]"
                if max_percentage is not None:
                    msg = ("Parameter 'max_percentage' is ignored when "
                           "'cumulative=True'.")
                    warnings.warn(msg)
                max_percentage = 100
                if cumulative_number_of_colors is not None:
                    cmap = LinearSegmentedColormap(
                        name=cmap.name, segmentdata=cmap._segmentdata,
                        N=cumulative_number_of_colors)
            elif max_percentage is None:
                # Set default only if cumulative is not True.
                max_percentage = 30

            fig.ppsd.cumulative = cumulative
            fig.ppsd.cmap = cmap
            fig.ppsd.label = label
            fig.ppsd.max_percentage = max_percentage
            fig.ppsd.grid = grid
            fig.ppsd.xaxis_frequency = xaxis_frequency
            if max_percentage is not None:
                color_limits = (0, max_percentage)
                fig.ppsd.color_limits = color_limits

            self._plot_histogram(fig=fig)

        ax.semilogx()
        ax.set_xlim(period_lim)
        ax.set_ylim(self.db_bin_edges[0], self.db_bin_edges[-1])
        if xaxis_frequency:
            ax.set_xlabel('Frequency [Hz]')
        else:
            ax.set_xlabel('Period [s]')
        if self.special_handling is None:
            ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
        else:
            ax.set_ylabel('Amplitude [dB]')
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.set_title(self._get_plot_title())

        if show_coverage:
            self.__plot_coverage(ax2)
            # emulating fig.autofmt_xdate():
            for label in ax2.get_xticklabels():
                label.set_ha("right")
                label.set_rotation(30)

        # Catch underflow warnings due to plotting on log-scale.
        _t = np.geterr()
        np.seterr(all="ignore")
        try:
            if filename is not None:
                plt.savefig(filename)
                plt.close()
            elif show:
                plt.draw()
                plt.show()
            else:
                plt.draw()
                return fig
        finally:
            np.seterr(**_t)

    def _plot_histogram(self, fig, draw=False, filename=None):
        """
        Reuse a previously created figure returned by :meth:`plot(show=False)`
        and plot the current histogram stack (pre-computed using
        :meth:`calculate_histogram()`) into the figure. If a filename is
        provided, the figure will be saved to a local file.
        Note that many aspects of the plot are statically set during the first
        :meth:`plot()` call, so this routine can only be used to update with
        data from a new stack.
        """
        import matplotlib.pyplot as plt
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        if "quadmesh" in fig.ppsd:
            ax.collections.remove(fig.ppsd.pop("quadmesh"))

        if fig.ppsd.cumulative:
            data = self.current_histogram_cumulative * 100.0
        else:
            # avoid divison with zero in case of empty stack
            data = (
                self.current_histogram * 100.0 /
                (self.current_histogram_count or 1))

        xedges = self.period_xedges
        if fig.ppsd.xaxis_frequency:
            xedges = 1.0 / xedges

        if "meshgrid" not in fig.ppsd:
            fig.ppsd.meshgrid = np.meshgrid(xedges, self.db_bin_edges)
        X, Y = fig.ppsd.meshgrid
        ppsd = ax.pcolormesh(X, Y, data.T, cmap=fig.ppsd.cmap, zorder=-1)
        fig.ppsd.quadmesh = ppsd

        if "colorbar" not in fig.ppsd:
            cb = plt.colorbar(ppsd, ax=ax)
            cb.set_clim(*fig.ppsd.color_limits)
            cb.set_label(fig.ppsd.label)
            fig.ppsd.colorbar = cb

        if fig.ppsd.max_percentage is not None:
            ppsd.set_clim(*fig.ppsd.color_limits)

        if fig.ppsd.grid:
            if fig.ppsd.cmap.name == "viridis":
                color = {"color": "0.7"}
            else:
                color = {}
            ax.grid(b=True, which="major", **color)
            ax.grid(b=True, which="minor", **color)

        ax.set_xlim(*xlim)

        if filename is not None:
            plt.savefig(filename)
        elif draw:
            with np.errstate(under="ignore"):
                plt.draw()
        return fig

    def _get_plot_title(self):
        title = "%s   %s -- %s  (%i/%i segments)"
        title = title % (self.id,
                         UTCDateTime(self._times_processed[0]).date,
                         UTCDateTime(self._times_processed[-1]).date,
                         self.current_histogram_count,
                         len(self._times_processed))
        return title

    def plot_coverage(self, filename=None):
        """
        Plot the data coverage of the histogram of the current PPSD.
        If a filename is specified the plot is saved to this file, otherwise
        a plot window is shown.

        :type filename: str, optional
        :param filename: Name of output file
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)

        self.__plot_coverage(ax)
        fig.autofmt_xdate()
        ax.set_title(self._get_plot_title())

        plt.draw()
        if filename is not None:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def __plot_coverage(self, ax):
        """
        Helper function to plot coverage into given axes.
        """
        ax.figure
        ax.clear()
        ax.xaxis_date()
        ax.set_yticks([])

        # plot data used in histogram stack
        used_times = [UTCDateTime(t) for t in self._times_processed
                      if t in self._current_times_used]
        unused_times = [UTCDateTime(t) for t in self._times_processed
                        if t not in self._current_times_used]
        for times, color in zip((used_times, unused_times), ("b", "0.6")):
            # skip on empty lists (i.e. all data used, or none used in stack)
            if not times:
                continue
            starts = [date2num(t.datetime) for t in times]
            ends = [date2num((t + self.ppsd_length).datetime)
                    for t in times]
            startends = np.array([starts, ends])
            startends = compress_start_end(startends.T, 20,
                                           merge_overlaps=True)
            starts, ends = startends[:, 0], startends[:, 1]
            for start, end in zip(starts, ends):
                ax.axvspan(start, end, 0, 0.6, fc=color, lw=0)
        # plot data that was fed to PPSD
        for start, end in self.times_data:
            start = date2num(start.datetime)
            end = date2num(end.datetime)
            ax.axvspan(start, end, 0.6, 1, facecolor="g", lw=0)
        # plot gaps in data fed to PPSD
        for start, end in self.times_gaps:
            start = date2num(start.datetime)
            end = date2num(end.datetime)
            ax.axvspan(start, end, 0.6, 1, facecolor="r", lw=0)

        ax.autoscale_view()


@deprecated("get_NLNM() has been renamed to get_nlnm().")
def get_NLNM():  # noqa
    return get_nlnm()


def get_nlnm():
    """
    Returns periods and psd values for the New Low Noise Model.
    For information on New High/Low Noise Model see [Peterson1993]_.
    """
    data = np.load(NOISE_MODEL_FILE)
    periods = data['model_periods']
    nlnm = data['low_noise']
    return (periods, nlnm)


@deprecated("get_NHNM() has been renamed to get_nhnm().")
def get_NHNM():  # noqa
    return get_nhnm()


def get_nhnm():
    """
    Returns periods and psd values for the New High Noise Model.
    For information on New High/Low Noise Model see [Peterson1993]_.
    """
    data = np.load(NOISE_MODEL_FILE)
    periods = data['model_periods']
    nlnm = data['high_noise']
    return (periods, nlnm)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
