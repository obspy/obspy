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
import bisect
import glob
import math
import os
import warnings

import numpy as np
from matplotlib import mlab
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patheffects import withStroke

from obspy import Stream, Trace, UTCDateTime, __version__
from obspy.core import Stats
from obspy.imaging.scripts.scan import compress_start_end
from obspy.core.inventory import Inventory
from obspy.core.util import AttribDict, NUMPY_VERSION
from obspy.core.util.base import MATPLOTLIB_VERSION
from obspy.core.util.obspy_types import ObsPyException
from obspy.imaging.cm import obspy_sequential
from obspy.imaging.util import _set_xaxis_obspy_dates
from obspy.io.xseed import Parser
from obspy.signal.invsim import cosine_taper
from obspy.signal.util import prev_pow_2
from obspy.signal.invsim import paz_to_freq_resp, evalresp


dtiny = np.finfo(0.0).tiny

NOISE_MODEL_FILE = os.path.join(os.path.dirname(__file__), "data",
                                "noise_models.npz")

# Noise models for special_handling="infrasound"
NOISE_MODEL_FILE_INF = os.path.join(os.path.dirname(__file__), "data",
                                    "idc_noise_models.npz")

earthquake_models = {
    (1.5, 10): [[7.0700000e-01, 1.4140000e+00, 2.8280000e+00, 5.6600000e+00,
                 1.1300000e+01, 2.2600000e+01],
                [1.1940722e-06, 5.9652811e-06, 4.2022150e-05, 2.1857010e-04,
                 6.4011669e-04, 1.0630361e-03]],
    (1.5, 100): [[1.4140000e+00, 2.8280000e+00, 5.6600000e+00, 1.1300000e+01,
                  2.2600000e+01],
                 [2.0636508e-07, 1.0591860e-06, 4.4487964e-06, 1.1678490e-05,
                  1.2427714e-05]],
    (2.5, 10): [[3.5350000e-01, 7.0700000e-01, 1.4140000e+00, 2.8280000e+00,
                 5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
                [3.1301161e-06, 1.9775227e-05, 1.5482427e-04, 8.3811450e-04,
                 3.7537242e-03, 7.3529155e-03, 6.5422494e-03]],
    (2.5, 100): [[7.0700000e-01, 1.4140000e+00, 2.8280000e+00, 5.6600000e+00,
                  1.1300000e+01, 2.2600000e+01],
                 [7.4170515e-07, 4.9107516e-06, 2.5050832e-05, 8.4981163e-05,
                  9.5891456e-05, 6.5125171e-05]],
    (3.5, 10): [[1.7670000e-01, 3.5350000e-01, 7.0700000e-01, 1.4140000e+00,
                 2.8280000e+00, 5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
                [1.4096634e-05, 8.5860094e-05, 5.4403050e-04, 3.4576905e-03,
                 1.3476040e-02, 3.4295149e-02, 4.7010730e-02, 2.3188069e-02]],
    (3.5, 100): [[3.5350000e-01, 7.0700000e-01, 1.4140000e+00, 2.8280000e+00,
                  5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
                 [2.0677442e-06, 1.4237587e-05, 8.5915636e-05, 3.8997357e-04,
                  1.0533844e-03, 8.7243737e-04, 2.8224063e-04]],
    (4.5, 10): [[1.7670000e-01, 3.5350000e-01, 7.0700000e-01, 1.4140000e+00,
                 2.8280000e+00, 5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
                [4.0994260e-05, 2.9194046e-04, 1.9989323e-03, 1.0610332e-02,
                 3.1188683e-02, 6.5872809e-02, 7.7301339e-02, 2.7431279e-02]],
    (4.5, 100): [[1.7670000e-01, 3.5350000e-01, 7.0700000e-01, 1.4140000e+00,
                  2.8280000e+00, 5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
                 [8.3385376e-06, 4.8130301e-05, 3.1461832e-04, 1.1054180e-03,
                  3.2015584e-03, 5.9504166e-03, 3.4106672e-03, 1.0389085e-03]],
    (5.5, 10): [[4.4187500e-02, 8.8375000e-02, 1.7670000e-01, 3.5350000e-01,
                 7.0700000e-01, 1.4140000e+00, 2.8280000e+00, 5.6600000e+00,
                 1.1300000e+01, 2.2600000e+01],
                [5.1597812e-05, 2.4350754e-04, 1.5861857e-03, 9.1333683e-03,
                 5.4321168e-02, 3.0210031e-01, 4.4505525e-01, 3.3287588e-01,
                 3.0775399e-01, 1.2501407e-01]],
    (5.5, 100): [[4.4187500e-02, 8.8375000e-02, 1.7670000e-01, 3.5350000e-01,
                  7.0700000e-01, 1.4140000e+00, 2.8280000e+00, 5.6600000e+00,
                  1.1300000e+01, 2.2600000e+01],
                 [7.6042044e-06, 5.0568931e-05, 2.6926792e-04, 9.7744507e-04,
                  3.4530700e-03, 9.2832164e-03, 1.5054122e-02, 1.3421025e-02,
                  8.0524871e-03, 2.0691071e-03]],
    (6.5, 10): [[8.8375000e-02, 1.7670000e-01, 3.5350000e-01, 7.0700000e-01,
                 1.4140000e+00, 2.8280000e+00, 5.6600000e+00, 1.1300000e+01,
                 2.2600000e+01],
                [1.7468292e-02, 6.5551361e-02, 1.8302926e-01, 3.9065640e-01,
                 7.1420714e-01, 1.0447672e+00, 1.2770160e+00, 1.0372500e+00,
                 5.0465056e-01]],
    (6.5, 100): [[2.2097000e-02, 4.4187500e-02, 8.8375000e-02, 1.7670000e-01,
                  3.5350000e-01, 7.0700000e-01, 1.4140000e+00, 2.8280000e+00,
                  5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
                 [1.4561822e-05, 1.4224456e-04, 8.7870935e-04, 2.4575511e-03,
                  8.0615599e-03, 2.3743599e-02, 4.8707533e-02, 7.0969056e-02,
                  7.6622487e-02, 2.6998756e-02, 4.6235290e-03]],
    (7, 10): [[4.4187500e-02, 8.8375000e-02, 1.7670000e-01, 3.5350000e-01,
               7.0700000e-01, 1.4140000e+00, 2.8280000e+00, 5.6600000e+00,
               1.1300000e+01, 2.2600000e+01, 4.5200000e+01, 7.0700000e+01],
              [9.1004029e-03, 4.7514014e-02, 1.5787725e-01, 3.1426661e-01,
               6.3241827e-01, 1.0667690e+00, 1.2675411e+00, 1.1326387e+00,
               8.3962478e-01, 4.1229082e-01, 1.6311586e-01, 5.6992659e-02]],
    (7, 100): [[2.2097000e-02, 4.4187500e-02, 8.8375000e-02, 1.7670000e-01,
                3.5350000e-01, 7.0700000e-01, 1.4140000e+00, 2.8280000e+00,
                5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
               [5.8736062e-04, 3.2124167e-03, 1.5314560e-02, 3.3624616e-02,
                6.4082408e-02, 1.3541913e-01, 1.8690009e-01, 1.8862548e-01,
                1.1841223e-01, 4.8641263e-02, 1.3723779e-02]],
    (6, 3000): [[5.0000000e-03, 4.5000000e-02, 3.0000000e-01, 1.3000000e+00,
                 6.0000000e+00],
                [9.8100000e-10, 7.8480000e-07, 4.9050000e-06, 4.9050000e-06,
                 7.8480000e-07]],
    (7, 3000): [[1.5000000e-03, 2.1000000e-02, 4.5000000e-02, 9.0000000e-02,
                 8.0000000e-01, 6.0000000e+00],
                [1.4715000e-09, 7.8480000e-06, 2.9430000e-05, 3.9240000e-05,
                 1.4715000e-05, 9.8100000e-07]],
    (8, 3000): [[1.5000000e-03, 1.0000000e-02, 4.5000000e-02, 9.0000000e-02,
                 3.0000000e+00, 6.0000000e+00],
                [9.8100000e-09, 3.9240000e-06, 1.1772000e-04, 1.9620000e-04,
                 4.9050000e-06, 1.9620000e-06]]}


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
    >>> print(ppsd.times_processed) # doctest: +SKIP
    [UTCDateTime(...), UTCDateTime(...), ..., UTCDateTime(...)]
    >>> ppsd.plot() # doctest: +SKIP

    ... but the example stream is too short and does not contain enough data.

    .. note::

        For a real world example see the `ObsPy Tutorial`_.

    .. rubric:: Saving and Loading

    The PPSD object supports saving to a numpy npz compressed binary file:

    >>> ppsd.save_npz("myfile.npz") # doctest: +SKIP

    The saved PPSD can then be loaded again using the static method
    :func:`~obspy.signal.spectral_estimation.PPSD.load_npz`, e.g. to plot the
    results again. If additional data is to be processed (note that another
    option is to combine multiple npz files using
    :meth:`~obspy.signal.spectral_estimation.PPSD.add_npz`), metadata must be
    provided again, since they are not stored in the numpy npz file:

    >>> ppsd = PPSD.load_npz("myfile.npz")  # doctest: +SKIP

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
        '_binned_psds']
    NPZ_STORE_KEYS_VERSION_NUMBERS = [
        # version numbers
        'ppsd_version',
        'obspy_version',
        'numpy_version',
        'matplotlib_version']
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
        '_nfft']
    NPZ_STORE_KEYS_ARRAY_TYPES = [
        # attributes derived during __init__
        '_db_bin_edges',
        '_psd_periods',
        '_period_binning']
    NPZ_STORE_KEYS = (
        NPZ_STORE_KEYS_ARRAY_TYPES +
        NPZ_STORE_KEYS_LIST_TYPES +
        NPZ_STORE_KEYS_SIMPLE_TYPES +
        NPZ_STORE_KEYS_VERSION_NUMBERS)
    # A mapping of values for storing info in the NPZ file. This is needed
    # because some types are not loadable without allowing pickle (#2409).
    NPZ_SIMPLE_TYPE_MAP = {None: ''}
    NPZ_SIMPLE_TYPE_MAP_R = {v: i for i, v in NPZ_SIMPLE_TYPE_MAP.items()}
    # Add current version as a class attribute to avoid hard coding it.
    _CURRENT_VERSION = 3

    def __init__(self, stats, metadata, skip_on_gaps=False,
                 db_bins=(-200, -50, 1.), ppsd_length=3600.0, overlap=0.5,
                 special_handling=None, period_smoothing_width_octaves=1.0,
                 period_step_octaves=0.125, period_limits=None,
                 **kwargs):  # @UnusedVariable
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
        if self.special_handling == 'ringlaser':
            if not isinstance(self.metadata, dict):
                msg = ("When using `special_handling='ringlaser'`, `metadata` "
                       "must be a plain dictionary with key 'sensitivity' "
                       "stating the overall sensitivity`.")
                raise TypeError(msg)
        elif self.special_handling == 'hydrophone':
            pass
        # Add special handling option for infrasound
        elif self.special_handling == 'infrasound':
            pass
        elif self.special_handling is not None:
            msg = "Unsupported value for 'special_handling' parameter: %s"
            msg = msg % self.special_handling
            raise ValueError(msg)

        # save version numbers
        self.ppsd_version = self._CURRENT_VERSION
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
    def step(self):
        """
        Time step between start times of adjacent psd segments in seconds
        (assuming gap-less data).
        """
        return self.ppsd_length * (1.0 - self.overlap)

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
    def psd_values(self):
        """
        Returns all individual smoothed psd arrays as a list. The corresponding
        times can be accessed as :attr:`PPSD.times_processed`, the
        corresponding central periods in seconds (central frequencies in Hertz)
        can be accessed as :attr:`PPSD.psd_periods`
        (:attr:`PPSD.psd_frequencies`).
        """
        return self._binned_psds

    @property
    def times_processed(self):
        return [UTCDateTime(ns=ns) for ns in self._times_processed]

    @property
    def times_data(self):
        return [(UTCDateTime(ns=t1), UTCDateTime(ns=t2))
                for t1, t2 in self._times_data]

    @property
    def times_gaps(self):
        return [(UTCDateTime(ns=t1), UTCDateTime(ns=t2))
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
        return [UTCDateTime(ns=ns) for ns in self._current_times_used]

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
        t = utcdatetime._ns
        ind = bisect.bisect(self._times_processed, t)
        self._times_processed.insert(ind, t)
        self._binned_psds.insert(ind, spectrum)

    def __insert_gap_times(self, stream):
        """
        Gets gap information of stream and adds the encountered gaps to the gap
        list of the PPSD instance.

        :type stream: :class:`~obspy.core.stream.Stream`
        """
        self._times_gaps += [[gap[4]._ns, gap[5]._ns]
                             for gap in stream.get_gaps()]

    def __insert_data_times(self, stream):
        """
        Gets gap information of stream and adds the encountered gaps to the gap
        list of the PPSD instance.

        :type stream: :class:`~obspy.core.stream.Stream`
        """
        self._times_data += \
            [[tr.stats.starttime._ns, tr.stats.endtime._ns]
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
        if not self._times_processed:
            return False
        # new data comes before existing data.
        if utcdatetime._ns < self._times_processed[0]:
            overlap_seconds = (
                (utcdatetime._ns + self.ppsd_length * 1e9) -
                self._times_processed[0]) / 1e9
            # the new data is welcome if any overlap that would be introduced
            # is less or equal than the overlap used by default on continuous
            # data.
            if overlap_seconds / self.ppsd_length > self.overlap:
                return True
            else:
                return False
        # new data exactly at start of first data segment
        elif utcdatetime._ns == self._times_processed[0]:
            return True
        # new data comes after existing data.
        elif utcdatetime._ns > self._times_processed[-1]:
            overlap_seconds = (
                (self._times_processed[-1] + self.ppsd_length * 1e9) -
                utcdatetime._ns) / 1e9
            # the new data is welcome if any overlap that would be introduced
            # is less or equal than the overlap used by default on continuous
            # data.
            if overlap_seconds / self.ppsd_length > self.overlap:
                return True
            else:
                return False
        # new data exactly at start of last data segment
        elif utcdatetime._ns == self._times_processed[-1]:
            return True
        # otherwise we are somewhere within the currently already present time
        # range..
        else:
            index1 = bisect.bisect_left(self._times_processed,
                                        utcdatetime._ns)
            index2 = bisect.bisect_right(self._times_processed,
                                         utcdatetime._ns)
            # if bisect left/right gives same result, we are not exactly at one
            # sampling point but in between to timestamps
            if index1 == index2:
                t1 = self._times_processed[index1 - 1]
                t2 = self._times_processed[index1]
                # check if we are overlapping on left side more than the normal
                # overlap specified during init
                overlap_seconds_left = (
                    (t1 + self.ppsd_length * 1e9) - utcdatetime._ns) / 1e9
                # check if we are overlapping on right side more than the
                # normal overlap specified during init
                overlap_seconds_right = (
                    (utcdatetime._ns + self.ppsd_length * 1e9) - t2) / 1e9
                max_overlap = max(overlap_seconds_left,
                                  overlap_seconds_right) / self.ppsd_length
                if max_overlap > self.overlap:
                    return True
                else:
                    return False
            # if bisect left/right gives different results, we are at exactly
            # one timestamp that is already present
            else:
                return True
        raise NotImplementedError('This should not happen, please report on '
                                  'github.')

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
        if not stream:
            msg = 'Empty stream object provided to PPSD.add()'
            warnings.warn(msg)
            return False
        # select appropriate traces
        stream = stream.select(id=self.id)
        if not stream:
            msg = 'No traces with matching SEED ID in provided stream object.'
            warnings.warn(msg)
            return False
        stream = stream.select(sampling_rate=self.sampling_rate)
        if not stream:
            msg = ('No traces with matching sampling rate in provided stream '
                   'object.')
            warnings.warn(msg)
            return False
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
            while t1 + self.ppsd_length - tr.stats.delta <= t2:
                if self.__check_time_present(t1):
                    msg = "Already covered time spans detected (e.g. %s), " + \
                          "skipping these slices."
                    msg = msg % t1
                    warnings.warn(msg)
                else:
                    # throw warnings if trace length is different
                    # than ppsd_length..!?!
                    slice = tr.slice(t1, t1 + self.ppsd_length -
                                     tr.stats.delta)
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
            msg = ("Got a piece of data with wrong length. Skipping:\n" +
                   str(tr))
            warnings.warn(msg)
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
                msg = msg % (e.__class__.__name__, str(e))
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
            if self.special_handling in ("hydrophone", "infrasound"):
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

    def _get_times_all_details(self):
        # check if we can reuse a previously cached array of all times as
        # day of week as int and time of day in float hours
        if len(self._current_times_all_details) == len(self._times_processed):
            return self._current_times_all_details
        # otherwise compute it and store it for subsequent stacks on the
        # same data (has to be recomputed when additional data gets added)
        else:
            dtype = np.dtype([('time_of_day', np.float32),
                              ('iso_weekday', np.int8),
                              ('iso_week', np.int8),
                              ('year', np.int16),
                              ('month', np.int8)])
            times_all_details = np.empty(shape=len(self._times_processed),
                                         dtype=dtype)
            utc_times_all = [UTCDateTime(ns=t) for t in self._times_processed]
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
            selected &= times_all > starttime._ns
        if endtime is not None:
            selected &= times_all < endtime._ns
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
        :param endtime: If set, data after the specified time is excluded
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
        # (can vary from the number of self.times_processed because of values
        #  outside the histogram db ranges)
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
        elif isinstance(self.metadata, str):
            return self._get_response_from_resp(tr)
        else:
            msg = "Unexpected type for `metadata`: %s" % type(self.metadata)
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

    def _get_response_from_paz_dict(self, tr):  # @UnusedVariable
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
        return self.period_bin_centers, mean

    def save_npz(self, filename):
        """
        Saves the PPSD as a compressed numpy binary (npz format).

        The resulting file can be restored using `my_ppsd.load_npz(filename)`.

        :type filename: str
        :param filename: Name of numpy .npz output file
        """
        out = {}
        for key in self.NPZ_STORE_KEYS:
            value = getattr(self, key)
            # Some values need to be replaced to allow non-pickle
            # serialization (#2409).
            if key in self.NPZ_STORE_KEYS_SIMPLE_TYPES:
                value = self.NPZ_SIMPLE_TYPE_MAP.get(value, value)
            out[key] = value
        np.savez_compressed(filename, **out)

    @staticmethod
    def load_npz(filename, metadata=None, allow_pickle=False):
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
        :type allow_pickle: bool
        :param allow_pickle:
            Allow the pickle protocol to be used when de-serializing saved
            PPSDs. This is only required for npz files written by ObsPy
            versions less than 1.2.0.
        """
        def _load(data):
            # the information regarding stats is set from the npz
            ppsd = PPSD(Stats(), metadata=metadata)
            # check ppsd_version version and raise if higher than current
            _check_npz_ppsd_version(ppsd, data)
            for key in ppsd.NPZ_STORE_KEYS:
                # data is stored as arrays in the npz.
                # we have to convert those back to lists (or simple types), so
                # that additionally processed data can be appended/inserted
                # later.
                try:
                    data_ = data[key]
                except ValueError:
                    msg = ("Loading PPSD results saved with ObsPy versions < "
                           "1.2 requires setting the allow_pickle parameter "
                           "of PPSD.load_npz to True")
                    raise ValueError(msg)
                if key in ppsd.NPZ_STORE_KEYS_LIST_TYPES:
                    if key in ['_times_data', '_times_gaps']:
                        data_ = data_.tolist()
                    else:
                        data_ = [d for d in data_]
                elif key in (ppsd.NPZ_STORE_KEYS_SIMPLE_TYPES +
                             ppsd.NPZ_STORE_KEYS_VERSION_NUMBERS):
                    data_ = data_.item()
                    data_ = ppsd.NPZ_SIMPLE_TYPE_MAP_R.get(data_, data_)
                # convert floating point POSIX second timestamps from older npz
                # files
                if (key in ppsd.NPZ_STORE_KEYS_LIST_TYPES and
                        data['ppsd_version'].item() == 1):
                    if key in ['_times_data', '_times_gaps']:
                        data_ = [[UTCDateTime(start)._ns, UTCDateTime(end)._ns]
                                 for start, end in data_]
                    elif key == '_times_processed':
                        data_ = [UTCDateTime(t)._ns for t in data_]
                setattr(ppsd, key, data_)
            # we converted all data, so update ppsd version
            ppsd.ppsd_version = PPSD._CURRENT_VERSION
            return ppsd

        # XXX get rid of if/else again when bumping minimal numpy to 1.7
        if NUMPY_VERSION >= [1, 7]:
            # XXX get rid of if/else again when bumping minimal numpy to 1.10
            if NUMPY_VERSION >= [1, 10]:
                kwargs = {'allow_pickle': allow_pickle}
            else:
                kwargs = {}
            with np.load(filename, **kwargs) as data:
                return _load(data)
        else:
            data = np.load(filename)
            try:
                return _load(data)
            finally:
                data.close()

    def add_npz(self, filename, allow_pickle=False):
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
        :type allow_pickle: bool
        :param allow_pickle:
            Allow the pickle protocol to be used when de-serializing saved
            PPSDs. This is only required for npz files written by ObsPy
            versions less than 1.2.0.
        """
        for filename in glob.glob(filename):
            self._add_npz(filename, allow_pickle=allow_pickle)

    def _add_npz(self, filename, allow_pickle=False):
        """
        See :meth:`PPSD.add_npz()`.
        """
        def _add(data):
            # check ppsd_version version and raise if higher than current
            _check_npz_ppsd_version(self, data)
            # check if all metadata agree
            for key in self.NPZ_STORE_KEYS_SIMPLE_TYPES:
                value_ = data[key].item()
                value = self.NPZ_SIMPLE_TYPE_MAP_R.get(value_, value_)
                if getattr(self, key) != value:
                    msg = ("Mismatch in '%s' attribute.\n\tCurrent:\n\t%s\n\t"
                           "Loaded:\n\t%s")
                    msg = msg % (key, getattr(self, key), data[key].item())
                    raise AssertionError(msg)
            for key in self.NPZ_STORE_KEYS_ARRAY_TYPES:
                try:
                    np.testing.assert_array_equal(getattr(self, key),
                                                  data[key])
                except AssertionError as e:
                    msg = ("Mismatch in '%s' attribute.\n") % key
                    raise AssertionError(msg + str(e))
            # load new psd data
            for key in self.NPZ_STORE_KEYS_VERSION_NUMBERS:
                if getattr(self, key) != data[key].item():
                    msg = ("Mismatch in version numbers (%s) between current "
                           "data (%s) and loaded data (%s).") % (
                               key, getattr(self, key), data[key].item())
                    warnings.warn(msg)
            _times_data = data["_times_data"].tolist()
            _times_gaps = data["_times_gaps"].tolist()
            _times_processed = [d_ for d_ in data["_times_processed"]]
            _binned_psds = [d_ for d_ in data["_binned_psds"]]
            # convert floating point POSIX second timestamps from older npz
            # files
            if data['ppsd_version'].item() == 1:
                _times_data = [[UTCDateTime(start)._ns, UTCDateTime(end)._ns]
                               for start, end in _times_data]
                _times_gaps = [[UTCDateTime(start)._ns, UTCDateTime(end)._ns]
                               for start, end in _times_gaps]
                _times_processed = [
                    UTCDateTime(t)._ns for t in _times_processed]
            # add new data
            self._times_data.extend(_times_data)
            self._times_gaps.extend(_times_gaps)
            duplicates = 0
            for t, psd in zip(_times_processed, _binned_psds):
                t = UTCDateTime(ns=t)
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

        # XXX get rid of if/else again when bumping minimal numpy to 1.7
        if NUMPY_VERSION >= [1, 7]:
            # XXX get rid of if/else again when bumping minimal numpy to 1.10
            if NUMPY_VERSION >= [1, 10]:
                kwargs = {'allow_pickle': allow_pickle}
            else:
                kwargs = {}
            try:
                with np.load(filename, **kwargs) as data:
                    _add(data)
            except ValueError:
                msg = ("Loading PPSD results saved with ObsPy versions < "
                       "1.2 requires setting the allow_pickle parameter "
                       "of PPSD.load_npz to True (needs numpy>=1.10).")
                raise ValueError(msg)
        else:
            data = np.load(filename)
            try:
                _add(data)
            finally:
                data.close()

    def _split_lists(self, times, psds):
        """
        """
        t_diff_gapless = self.step * 1e9
        gap_indices = np.argwhere(np.diff(times) - t_diff_gapless)
        gap_indices = (gap_indices.flatten() + 1).tolist()

        if not len(gap_indices):
            return [(times, psds)]

        gapless = []
        indices_start = [0] + gap_indices
        indices_end = gap_indices + [len(times)]
        for start, end in zip(indices_start, indices_end):
            gapless.append((times[start:end], psds[start:end]))
        return gapless

    def _get_gapless_psd(self):
        """
        Helper routine to get a list of 2-tuples with gapless portions of
        processed PPSD time ranges.
        This means that PSD time history is split whenever to adjacent PSD
        timestamps are not separated by exactly
        ``self.ppsd_length * (1 - self.overlap)``.
        """
        return self._split_lists(self.times_processed, self.psd_values)

    def plot_spectrogram(self, cmap=obspy_sequential, clim=None, grid=True,
                         filename=None, show=True):
        """
        Plot the temporal evolution of the PSD in a spectrogram-like plot.

        .. note::
            For example plots see the :ref:`Obspy Gallery <gallery>`.

        :type cmap: :class:`matplotlib.colors.Colormap`
        :param cmap: Specify a custom colormap instance. If not specified, then
            the default ObsPy sequential colormap is used.
        :type clim: list
        :param clim: Minimum/maximum dB values for lower/upper end of colormap.
            Specified as type ``float`` or ``None`` for no clipping on one end
            of the scale (e.g. ``clim=[-150, None]`` for a lower limit of
            ``-150`` dB and no clipping on upper end).
        :type grid: bool
        :param grid: Enable/disable grid in histogram plot.
        :type filename: str
        :param filename: Name of output file
        :type show: bool
        :param show: Enable/disable immediately showing the plot.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        quadmeshes = []
        yedges = self.period_xedges

        for times, psds in self._get_gapless_psd():
            xedges = [t.matplotlib_date for t in times] + \
                [(times[-1] + self.step).matplotlib_date]
            meshgrid_x, meshgrid_y = np.meshgrid(xedges, yedges)
            data = np.array(psds).T

            quadmesh = ax.pcolormesh(meshgrid_x, meshgrid_y, data, cmap=cmap,
                                     zorder=-1)
            quadmeshes.append(quadmesh)

        if clim is None:
            cmin = min(qm.get_clim()[0] for qm in quadmeshes)
            cmax = max(qm.get_clim()[1] for qm in quadmeshes)
            clim = (cmin, cmax)

        for quadmesh in quadmeshes:
            quadmesh.set_clim(*clim)

        cb = plt.colorbar(quadmesh, ax=ax)

        if grid:
            ax.grid()

        if self.special_handling is None:
            cb.ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
        elif self.special_handling == "infrasound":
            ax.set_ylabel('Amplitude [$Pa^2/Hz$] [dB]')
        else:
            cb.ax.set_ylabel('Amplitude [dB]')
        ax.set_ylabel('Period [s]')

        fig.autofmt_xdate()
        _set_xaxis_obspy_dates(ax)

        ax.set_yscale("log")
        ax.set_xlim(self.times_processed[0].matplotlib_date,
                    (self.times_processed[-1] + self.step).matplotlib_date)
        ax.set_ylim(yedges[0], yedges[-1])
        try:
            ax.set_facecolor('0.8')
        # mpl <2 has different API for setting Axes background color
        except AttributeError:
            ax.set_axis_bgcolor('0.8')

        fig.tight_layout()

        if filename is not None:
            plt.savefig(filename)
            plt.close()
        elif show:
            plt.draw()
            plt.show()
        else:
            plt.draw()
            return fig

    def extract_psd_values(self, period):
        """
        Extract PSD values for given period in seconds.

        Selects the period bin whose center period is closest to the specified
        period. Also returns the minimum, center and maximum period of the
        selected bin. The respective times of the PSD values can be accessed as
        :attr:`PPSD.times_processed`.

        :type period: float
        :param period: Period to extract PSD values for in seconds.
        :rtype: four-tuple of (list, float, float, float)
        :returns: PSD values for requested period (at times)
        """
        # evaluate which period bin to extract
        period_diff = np.abs(self.period_bin_centers - period)
        index = np.argmin(period_diff)
        period_min = self.period_bin_left_edges[index]
        period_max = self.period_bin_right_edges[index]
        period_center = self.period_bin_centers[index]
        psd_values = [psd[index] for psd in self.psd_values]
        return psd_values, period_min, period_center, period_max

    def plot_temporal(self, period, color=None, legend=True, grid=True,
                      linestyle="-", marker=None, filename=None, show=True,
                      **temporal_restrictions):
        """
        Plot the evolution of PSD value of one (or more) period bins over time.

        If a filename is specified the plot is saved to this file, otherwise
        a matplotlib figure is returned or shown.

        Additional keyword arguments are passed on to :meth:`_stack_selection`
        to restrict at which times PSD values are selected (e.g. to compare
        temporal evolution during a specific time span of each day).

        .. note::
            For example plots see the :ref:`Obspy Gallery <gallery>`.

        :type period: float (or list thereof)
        :param period: Period of PSD values to plot. The period bin with the
            central period that is closest to the specified value is selected.
            Multiple values can be specified in a list (``color`` option should
            then also be a list of color specifications, or left ``None``).
        :type color: matplotlib color specification (or list thereof)
        :param color: Color specification understood by :mod:`matplotlib` (or a
            list thereof in case of multiple periods to plot). ``None`` for
            default colors.
        :type grid: bool
        :param grid: Enable/disable grid in histogram plot.
        :type legend: bool
        :param legend: Enable/disable grid in histogram plot.
        :type linestyle: str
        :param linestyle: Linestyle for lines in the plot (see
            :func:`matplotlib.pyplot.plot`).
        :type marker: str
        :param marker: Marker for lines in the plot (see
            :func:`matplotlib.pyplot.plot`).
        :type filename: str
        :param filename: Name of output file
        :type show: bool
        :param show: Enable/disable immediately showing the plot.
        """
        import matplotlib.pyplot as plt

        try:
            len(period)
        except TypeError:
            periods = [period]
        else:
            periods = period

        if color is None:
            colors = [None] * len(periods)
        else:
            if len(periods) == 1:
                colors = [color]
            else:
                colors = color

        times = self._times_processed

        if temporal_restrictions:
            mask = ~self._stack_selection(**temporal_restrictions)
            times = [x for i, x in enumerate(times) if not mask[i]]
        else:
            mask = None

        fig, ax = plt.subplots()

        for period, color in zip(periods, colors):
            cur_color = color
            # extract psd values for given period
            psd_values, period_min, _, period_max = \
                self.extract_psd_values(period)
            if mask is not None:
                psd_values = [x for i, x in enumerate(psd_values)
                              if not mask[i]]
            # if right edge of period range is less than one second we label
            # the line in Hertz
            if period_max < 1:
                label = "{:.2g}-{:.2g} [Hz]".format(
                    1.0 / period_max, 1.0 / period_min)
            else:
                label = "{:.2g}-{:.2g} [s]".format(period_min, period_max)

            for i, (times_, psd_values) in enumerate(
                    self._split_lists(times, psd_values)):
                # only label first line plotted for each period
                if i:
                    label = None
                # older matplotlib raises when passing in `color=None`
                if color is None:
                    if cur_color is None:
                        color_kwargs = {}
                    else:
                        color_kwargs = {'color': cur_color}
                else:
                    color_kwargs = {'color': color}
                times_ = [UTCDateTime(ns=t).matplotlib_date for t in times_]
                line = ax.plot(times_, psd_values, label=label, ls=linestyle,
                               marker=marker, **color_kwargs)[0]
                # plot the next lines with the same color (we can't easily
                # determine the color beforehand if we rely on the color cycle,
                # i.e. when user doesn't specify colors explictly)
                cur_color = line.get_color()

        if legend:
            ax.legend()

        if grid:
            ax.grid()

        if self.special_handling is None:
            ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
        elif self.special_handling == "infrasound":
            ax.set_ylabel('Amplitude [$Pa^2/Hz$] [dB]')
        else:
            ax.set_ylabel('Amplitude [dB]')

        fig.autofmt_xdate()
        _set_xaxis_obspy_dates(ax)

        if filename is not None:
            plt.savefig(filename)
            plt.close()
        elif show:
            plt.draw()
            plt.show()
        else:
            plt.draw()
            return fig

    def plot(self, filename=None, show_coverage=True, show_histogram=True,
             show_percentiles=False, percentiles=[0, 25, 50, 75, 100],
             show_noise_models=True, grid=True, show=True,
             max_percentage=None, period_lim=(0.01, 179), show_mode=False,
             show_mean=False, cmap=obspy_sequential, cumulative=False,
             cumulative_number_of_colors=20, xaxis_frequency=False,
             show_earthquakes=None):
        """
        Plot the 2D histogram of the current PPSD.
        If a filename is specified the plot is saved to this file, otherwise
        a plot window is shown.

        .. note::
            For example plots see the :ref:`Obspy Gallery <gallery>`.

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
        :type show_earthquakes: bool, optional
        :param show_earthquakes: Enable/disable plotting of earthquake models
            like in [ClintonHeaton2002]_ and [CauzziClinton2013]_. Disabled by
            default (``None``). Specify ranges (minimum and maximum) for
            magnitude and distance of earthquake models given as four floats,
            e.g. ``(0, 5, 0, 99)`` for magnitude 1.5 - 4.5 at a epicentral
            distance of 10 km. Note only 10, 100 and 3000 km distances and
            magnitudes 1.5 to 7.5 are available. Alternatively, a distance can
            be specified in last float of a tuple of three, e.g. ``(0, 5, 10)``
            for 10 km distance, or magnitude and distance can be specified in
            a tuple of two floats, e.g. ``(5.5, 10)`` for magnitude 5.5 at 10
            km distance.
        :type grid: bool, optional
        :param grid: Enable/disable grid in histogram plot.
        :type show: bool, optional
        :param show: Enable/disable immediately showing the plot. If
            ``show=False``, then the matplotlib figure handle is returned.
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
                if xaxis_frequency:
                    xdata = 1.0 / periods
                else:
                    xdata = periods
                ax.plot(xdata, percentile_values, color="black", zorder=8)

        if show_mode:
            periods, mode_ = self.get_mode()
            if xaxis_frequency:
                xdata = 1.0 / periods
            else:
                xdata = periods
            if cmap.name == "viridis":
                color = "0.8"
            else:
                color = "black"
            ax.plot(xdata, mode_, color=color, zorder=9)

        if show_mean:
            periods, mean_ = self.get_mean()
            if xaxis_frequency:
                xdata = 1.0 / periods
            else:
                xdata = periods
            if cmap.name == "viridis":
                color = "0.8"
            else:
                color = "black"
            ax.plot(xdata, mean_, color=color, zorder=9)

        # Choose the correct noise model
        if self.special_handling == "infrasound":
            # Use IDC global infrasound models
            models = (get_idc_infra_hi_noise(), get_idc_infra_low_noise())
        else:
            # Use Peterson NHNM and NLNM
            models = (get_nhnm(), get_nlnm())

        if show_noise_models:
            for periods, noise_model in models:
                if xaxis_frequency:
                    xdata = 1.0 / periods
                else:
                    xdata = periods
                ax.plot(xdata, noise_model, '0.4', linewidth=2, zorder=10)

        if show_earthquakes is not None:
            if len(show_earthquakes) == 2:
                show_earthquakes = (show_earthquakes[0],
                                    show_earthquakes[0] + 0.1,
                                    show_earthquakes[1],
                                    show_earthquakes[1] + 1)
            if len(show_earthquakes) == 3:
                show_earthquakes += (show_earthquakes[-1] + 1, )
            min_mag, max_mag, min_dist, max_dist = show_earthquakes
            for key, data in earthquake_models.items():
                magnitude, distance = key
                frequencies, accelerations = data
                accelerations = np.array(accelerations)
                frequencies = np.array(frequencies)
                periods = 1.0 / frequencies
                # Eq.1 from Clinton and Cauzzi (2013) converts
                # power to density
                ydata = accelerations / (periods ** (-.5))
                ydata = 20 * np.log10(ydata / 2)
                if not (min_mag <= magnitude <= max_mag and
                        min_dist <= distance <= max_dist and
                        min(ydata) < self.db_bin_edges[-1]):
                    continue
                xdata = periods
                if xaxis_frequency:
                    xdata = frequencies
                ax.plot(xdata, ydata, '0.4', linewidth=2)
                leftpoint = np.argsort(xdata)[0]
                if not ydata[leftpoint] < self.db_bin_edges[-1]:
                    continue
                ax.text(xdata[leftpoint],
                        ydata[leftpoint],
                        'M%.1f\n%dkm' % (magnitude, distance),
                        ha='right', va='top',
                        color='w', weight='bold', fontsize='x-small',
                        path_effects=[withStroke(linewidth=3,
                                                 foreground='0.4')])

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
        if xaxis_frequency:
            ax.set_xlabel('Frequency [Hz]')
            ax.invert_xaxis()
        else:
            ax.set_xlabel('Period [s]')
        ax.set_xlim(period_lim)
        ax.set_ylim(self.db_bin_edges[0], self.db_bin_edges[-1])
        if self.special_handling is None:
            ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
        elif self.special_handling == "infrasound":
            ax.set_ylabel('Amplitude [$Pa^2/Hz$] [dB]')
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
        with np.errstate(all="ignore"):
            if filename is not None:
                plt.savefig(filename)
                plt.close()
            elif show:
                plt.draw()
                plt.show()
            else:
                plt.draw()
                return fig

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
        ppsd = ax.pcolormesh(
            fig.ppsd.meshgrid[0], fig.ppsd.meshgrid[1], data.T,
            cmap=fig.ppsd.cmap, zorder=-1)
        fig.ppsd.quadmesh = ppsd

        if "colorbar" not in fig.ppsd:
            cb = plt.colorbar(ppsd, ax=ax)
            cb.mappable.set_clim(*fig.ppsd.color_limits)
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
                         UTCDateTime(ns=self._times_processed[0]).date,
                         UTCDateTime(ns=self._times_processed[-1]).date,
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
        fig, ax = plt.subplots()

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
        used_times = [UTCDateTime(ns=t) for t in self._times_processed
                      if t in self._current_times_used]
        unused_times = [UTCDateTime(ns=t) for t in self._times_processed
                        if t not in self._current_times_used]
        for times, color in zip((used_times, unused_times), ("b", "0.6")):
            # skip on empty lists (i.e. all data used, or none used in stack)
            if not times:
                continue
            starts = [t.matplotlib_date for t in times]
            ends = [(t + self.ppsd_length).matplotlib_date for t in times]
            startends = np.array([starts, ends])
            startends = compress_start_end(startends.T, 20,
                                           merge_overlaps=True)
            starts, ends = startends[:, 0], startends[:, 1]
            for start, end in zip(starts, ends):
                ax.axvspan(start, end, 0, 0.6, fc=color, lw=0)
        # plot data that was fed to PPSD
        for start, end in self.times_data:
            start = start.matplotlib_date
            end = end.matplotlib_date
            ax.axvspan(start, end, 0.6, 1, facecolor="g", lw=0)
        # plot gaps in data fed to PPSD
        for start, end in self.times_gaps:
            start = start.matplotlib_date
            end = end.matplotlib_date
            ax.axvspan(start, end, 0.6, 1, facecolor="r", lw=0)

        ax.autoscale_view()


def get_nlnm():
    """
    Returns periods and psd values for the New Low Noise Model.
    For information on New High/Low Noise Model see [Peterson1993]_.
    """
    data = np.load(NOISE_MODEL_FILE)
    periods = data['model_periods']
    nlnm = data['low_noise']
    return (periods, nlnm)


def get_nhnm():
    """
    Returns periods and psd values for the New High Noise Model.
    For information on New High/Low Noise Model see [Peterson1993]_.
    """
    data = np.load(NOISE_MODEL_FILE)
    periods = data['model_periods']
    nlnm = data['high_noise']
    return (periods, nlnm)


def get_idc_infra_low_noise():
    """
    Returns periods and psd values for the IDC infrasound global low noise
    model. For information on the IDC noise models, see [Brown2012]_.
    """
    data = np.load(NOISE_MODEL_FILE_INF)
    periods = data['model_periods']
    nlnm = data['low_noise']
    return (periods, nlnm)


def get_idc_infra_hi_noise():
    """
    Returns periods and psd values for the IDC infrasound global high noise
    model. For information on the IDC noise models, see [Brown2012]_.
    """
    data = np.load(NOISE_MODEL_FILE_INF)
    periods = data['model_periods']
    nhnm = data['high_noise']
    return (periods, nhnm)


def _check_npz_ppsd_version(ppsd, npzfile):
    # add some future-proofing and show a warning if older ObsPy
    # versions should read a more recent ppsd npz file, since this is very
    # like problematic
    if npzfile['ppsd_version'].item() > ppsd.ppsd_version:
        msg = ("Trying to read/add a PPSD npz with 'ppsd_version={}'. This "
               "file was written on a more recent ObsPy version that very "
               "likely has incompatible changes in PPSD internal "
               "structure and npz serialization. It can not safely be "
               "read with this ObsPy version (current 'ppsd_version' is "
               "{}). Please consider updating your ObsPy "
               "installation.").format(npzfile['ppsd_version'].item(),
                                       ppsd.ppsd_version)
        raise ObsPyException(msg)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
