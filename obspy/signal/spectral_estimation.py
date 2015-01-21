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
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import warnings
import pickle
import math
import bisect
import bz2
import numpy as np
from obspy import Trace, Stream
from obspy.core.util import getMatplotlibVersion
from obspy.signal import cosTaper
from obspy.signal.util import prevpow2


MATPLOTLIB_VERSION = getMatplotlibVersion()

dtiny = np.finfo(0.0).tiny


from matplotlib import mlab
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.mlab import detrend_none, window_hanning


# build colormap as done in paper by mcnamara
CDICT = {'red': ((0.0, 1.0, 1.0),
                 (0.05, 1.0, 1.0),
                 (0.2, 0.0, 0.0),
                 (0.4, 0.0, 0.0),
                 (0.6, 0.0, 0.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.05, 0.0, 0.0),
                   (0.2, 0.0, 0.0),
                   (0.4, 1.0, 1.0),
                   (0.6, 1.0, 1.0),
                   (0.8, 1.0, 1.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.05, 1.0, 1.0),
                  (0.2, 1.0, 1.0),
                  (0.4, 1.0, 1.0),
                  (0.6, 0.0, 0.0),
                  (0.8, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
NOISE_MODEL_FILE = os.path.join(os.path.dirname(__file__),
                                "data", "noise_models.npz")


def psd(x, NFFT=256, Fs=2, detrend=detrend_none, window=window_hanning,
        noverlap=0):
    """
    Wrapper for :func:`matplotlib.mlab.psd`.

    Always returns a onesided psd (positive frequencies only), corrects for
    this fact by scaling with a factor of 2. Also, always normalizes to dB/Hz
    by dividing with sampling rate.

    This wrapper is intended to intercept changes in
    :func:`matplotlib.mlab.psd` default behavior which changes with
    matplotlib version 0.98.4:

    * http://matplotlib.org/users/whats_new.html#psd-amplitude-scaling
    * http://matplotlib.org/_static/CHANGELOG
      (entries on 2009-05-18 and 2008-11-11)
    * http://matplotlib.svn.sourceforge.net/viewvc/matplotlib\
?view=revision&revision=6518
    * http://matplotlib.org/api/api_changes.html#changes-for-0-98-x

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
    # check matplotlib version
    if MATPLOTLIB_VERSION >= [0, 98, 4]:
        new_matplotlib = True
    else:
        new_matplotlib = False
    # build up kwargs that do not change with version 0.98.4
    kwargs = {}
    kwargs['NFFT'] = NFFT
    kwargs['Fs'] = Fs
    kwargs['detrend'] = detrend
    kwargs['window'] = window
    kwargs['noverlap'] = noverlap
    # add additional kwargs to control behavior for matplotlib versions higher
    # than 0.98.4. These settings make sure that the scaling is already done
    # during the following psd call for newer matplotlib versions.
    if new_matplotlib:
        kwargs['pad_to'] = None
        kwargs['sides'] = 'onesided'
        kwargs['scale_by_freq'] = True
    # do the actual call to mlab.psd
    Pxx, freqs = mlab.psd(x, **kwargs)
    # do scaling manually for old matplotlib versions
    if not new_matplotlib:
        Pxx = Pxx / Fs
        Pxx[1:-1] = Pxx[1:-1] * 2.0
    return Pxx, freqs


def fft_taper(data):
    """
    Cosine taper, 10 percent at each end (like done by [McNamara2004]_).

    .. warning::
        Inplace operation, so data should be float.
    """
    data *= cosTaper(len(data), 0.2)
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


def welch_window(N):
    """
    Return a welch window for data of length N.

    Routine is checked against PITSA for both even and odd values, but not for
    strange values like N<5.

    .. note::
        See e.g.:
        http://www.cescg.org/CESCG99/TTheussl/node7.html

    :type N: int
    :param N: Length of window function.
    :rtype: :class:`~numpy.ndarray`
    :returns: Window function for tapering data.
    """
    n = math.ceil(N / 2.0)
    taper_left = np.arange(n, dtype=np.float64)
    taper_left = 1 - np.power(taper_left / n, 2)
    # first/last sample is zero by definition
    if N % 2 == 0:
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


class PPSD():
    """
    Class to compile probabilistic power spectral densities for one combination
    of network/station/location/channel/sampling_rate.

    Calculations are based on the routine used by [McNamara2004]_.
    For information on New High/Low Noise Model see [Peterson2003]_.

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
    >>> print(ppsd.times)
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

    The PPSD object supports saving to a pickled file with optional
    compression:

    >>> ppsd.save("myfile.pkl.bz2", compress=True) # doctest: +SKIP

    The saved PPSD can then be loaded again using the static method
    :func:`~obspy.signal.spectral_estimation.PPSD.load`, e.g. to add more data
    or plot it again:

    >>> ppsd = PPSD.load("myfile.pkl.bz2")  # doctest: +SKIP

    The :func:`~obspy.signal.spectral_estimation.PPSD.load` method detects
    compression automatically.

    .. note::

        While saving the PPSD with compression enabled takes significantly
        longer, it can reduce the resulting file size by more than 80%.

    .. note::

        It is safer (but a bit slower) to provide a
        :class:`~obspy.xseed.parser.Parser` instance with information from
        e.g. a Dataless SEED than to just provide a static PAZ dictionary.

    .. _`ObsPy Tutorial`: http://docs.obspy.org/tutorial/
    """
    def __init__(self, stats, paz=None, parser=None, skip_on_gaps=False,
                 is_rotational_data=False, db_bins=(-200, -50, 1.),
                 ppsd_length=3600., overlap=0.5, water_level=600.0):
        """
        Initialize the PPSD object setting all fixed information on the station
        that should not change afterwards to guarantee consistent spectral
        estimates.
        The necessary instrument response information can be provided in two
        ways:

        * Providing an `obspy.xseed` :class:`~obspy.xseed.parser.Parser`,
          e.g. containing metadata from a Dataless SEED file. This is the safer
          way but it might a bit slower because for every processed time
          segment the response information is extracted from the parser.
        * Providing a dictionary containing poles and zeros information. Be
          aware that this leads to wrong results if the instrument's response
          is changing with data added to the PPSD. Use with caution!

        :note: When using `is_rotational_data=True` the applied processing
               steps are changed. Differentiation of data (converting velocity
               to acceleration data) will be omitted and a flat instrument
               response is assumed, leaving away response removal and only
               dividing by `paz['sensitivity']` specified in the provided `paz`
               dictionary (other keys do not have to be present then). For
               scaling factors that are usually multiplied to the data remember
               to use the inverse as `paz['sensitivity']`.

        :type stats: :class:`~obspy.core.trace.Stats`
        :param stats: Stats of the station/instrument to process
        :type paz: dict, optional
        :param paz: Response information of instrument. If not specified the
                information is supposed to be present as stats.paz.
        :type parser: :class:`obspy.xseed.parser.Parser`, optional
        :param parser: Parser instance with response information (e.g. read
                from a Dataless SEED volume)
        :type skip_on_gaps: bool, optional
        :param skip_on_gaps: Determines whether time segments with gaps should
                be skipped entirely. [McNamara2004]_ merge gappy
                traces by filling with zeros. This results in a clearly
                identifiable outlier psd line in the PPSD visualization. Select
                `skip_on_gaps=True` for not filling gaps with zeros which might
                result in some data segments shorter than `ppsd_length` not
                used in the PPSD.
        :type is_rotational_data: bool, optional
        :param is_rotational_data: If set to True adapt processing of data to
                rotational data. See note for details.
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
        :type water_level: float, optional
        :param water_level: Water level used in instrument correction.
        """
        if paz is not None and parser is not None:
            msg = "Both paz and parser specified. Using parser object for " \
                  "metadata."
            warnings.warn(msg)

        self.id = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats
        self.network = stats.network
        self.station = stats.station
        self.location = stats.location
        self.channel = stats.channel
        self.sampling_rate = stats.sampling_rate
        self.delta = 1.0 / self.sampling_rate
        self.is_rotational_data = is_rotational_data
        self.ppsd_length = ppsd_length
        self.overlap = overlap
        self.water_level = water_level
        # trace length for one segment
        self.len = int(self.sampling_rate * ppsd_length)
        # set paz either from kwarg or try to get it from stats
        self.paz = paz
        self.parser = parser
        if skip_on_gaps:
            self.merge_method = -1
        else:
            self.merge_method = 0
        # nfft is determined mimicking the fft setup in McNamara&Buland paper:
        # (they take 13 segments overlapping 75% and truncate to next lower
        #  power of 2)
        #  - take number of points of whole ppsd segment (default 1 hour)
        self.nfft = ppsd_length * self.sampling_rate
        #  - make 13 single segments overlapping by 75%
        #    (1 full segment length + 25% * 12 full segment lengths)
        self.nfft = self.nfft / 4.0
        #  - go to next smaller power of 2 for nfft
        self.nfft = prevpow2(self.nfft)
        #  - use 75% overlap (we end up with a little more than 13 segments..)
        self.nlap = int(0.75 * self.nfft)
        self.times_used = []
        self.times = self.times_used
        self.times_data = []
        self.times_gaps = []
        self.hist_stack = None
        self.__setup_bins()
        # set up the binning for the db scale
        num_bins = int((db_bins[1] - db_bins[0]) / db_bins[2])
        self.spec_bins = np.linspace(db_bins[0], db_bins[1], num_bins + 1,
                                     endpoint=True)
        self.colormap = LinearSegmentedColormap('mcnamara', CDICT, 1024)

    def __setup_bins(self):
        """
        Makes an initial dummy psd and thus sets up the bins and all the rest.
        Should be able to do it without a dummy psd..
        """
        dummy = np.ones(self.len)
        _spec, freq = mlab.psd(dummy, self.nfft, self.sampling_rate,
                               noverlap=self.nlap)

        # leave out first entry (offset)
        freq = freq[1:]

        per = 1.0 / freq[::-1]
        self.freq = freq
        self.per = per
        # calculate left/right edge of first period bin,
        # width of bin is one octave
        per_left = per[0] / 2
        per_right = 2 * per_left
        # calculate center period of first period bin
        per_center = math.sqrt(per_left * per_right)
        # calculate mean of all spectral values in the first bin
        per_octaves_left = [per_left]
        per_octaves_right = [per_right]
        per_octaves = [per_center]
        # we move through the period range at 1/8 octave steps
        factor_eighth_octave = 2 ** 0.125
        # do this for the whole period range and append the values to our lists
        while per_right < per[-1]:
            per_left *= factor_eighth_octave
            per_right = 2 * per_left
            per_center = math.sqrt(per_left * per_right)
            per_octaves_left.append(per_left)
            per_octaves_right.append(per_right)
            per_octaves.append(per_center)
        self.per_octaves_left = np.array(per_octaves_left)
        self.per_octaves_right = np.array(per_octaves_right)
        self.per_octaves = np.array(per_octaves)

        self.period_bins = per_octaves
        # mid-points of all the period bins
        self.period_bin_centers = np.mean((self.period_bins[:-1],
                                           self.period_bins[1:]), axis=0)

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

    def __insert_used_time(self, utcdatetime):
        """
        Inserts the given UTCDateTime at the right position in the list keeping
        the order intact.

        :type utcdatetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        """
        bisect.insort(self.times_used, utcdatetime)

    def __insert_gap_times(self, stream):
        """
        Gets gap information of stream and adds the encountered gaps to the gap
        list of the PPSD instance.

        :type stream: :class:`~obspy.core.stream.Stream`
        """
        self.times_gaps += [[gap[4], gap[5]] for gap in stream.getGaps()]

    def __insert_data_times(self, stream):
        """
        Gets gap information of stream and adds the encountered gaps to the gap
        list of the PPSD instance.

        :type stream: :class:`~obspy.core.stream.Stream`
        """
        self.times_data += \
            [[tr.stats.starttime, tr.stats.endtime] for tr in stream]

    def __check_time_present(self, utcdatetime):
        """
        Checks if the given UTCDateTime is already part of the current PPSD
        instance. That is, checks if from utcdatetime to utcdatetime plus
        ppsd_length there is already data in the PPSD.
        Returns True if adding ppsd_length starting at the given time
        would result in an overlap of the ppsd data base, False if it is OK to
        insert this piece of data.
        """
        index1 = bisect.bisect_left(self.times_used, utcdatetime)
        index2 = bisect.bisect_right(self.times_used,
                                     utcdatetime + self.ppsd_length)
        if index1 != index2:
            return True
        else:
            return False

    def __check_ppsd_length(self):
        """
        Adds ppsd_length and overlap attributes if not existing.
        This ensures compatibility with pickled objects without these
        attributes.
        """
        try:
            self.ppsd_length
            self.overlap
        except AttributeError:
            self.ppsd_length = 3600.
            self.overlap = 0.5

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
        self.__check_ppsd_length()
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
                        self.__insert_used_time(t1)
                        if verbose:
                            print(t1)
                        changed = True
                t1 += (1 - self.overlap) * self.ppsd_length  # advance

            # enforce time limits, pad zeros if gaps
            # tr.trim(t, t+PPSD_LENGTH, pad=True)
        return changed

    def __process(self, tr):
        """
        Processes a segment of data and adds the information to the
        PPSD histogram. If Trace is compatible (station, channel, ...) has to
        checked beforehand.

        :type tr: :class:`~obspy.core.trace.Trace`
        :param tr: Compatible Trace with data of one PPSD segment
        :returns: True if segment was successfully added to histogram, False
                otherwise.
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

        # get instrument response preferably from parser object
        try:
            paz = self.parser.getPAZ(self.id, datetime=tr.stats.starttime)
        except Exception as e:
            if self.parser is not None:
                msg = "Error getting response from parser:\n%s: %s\n" \
                      "Skipping time segment(s)."
                msg = msg % (e.__class__.__name__, e.message)
                warnings.warn(msg)
                return False
            paz = self.paz
        if paz is None:
            msg = "Missing poles and zeros information for response " \
                  "removal. Skipping time segment(s)."
            warnings.warn(msg)
            return False
        # restitution:
        # mcnamara apply the correction at the end in freq-domain,
        # does it make a difference?
        # probably should be done earlier on bigger chunk of data?!
        if self.is_rotational_data:
            # in case of rotational data just remove sensitivity
            tr.data /= paz['sensitivity']
        else:
            tr.simulate(paz_remove=paz, remove_sensitivity=True,
                        paz_simulate=None, simulate_sensitivity=False,
                        water_level=self.water_level)

        # go to acceleration, do nothing for rotational data:
        if self.is_rotational_data:
            pass
        else:
            tr.data = np.gradient(tr.data, self.delta)

        # use our own wrapper for mlab.psd to have consistent results on all
        # matplotlib versions
        spec, _freq = psd(tr.data, self.nfft, self.sampling_rate,
                          detrend=mlab.detrend_linear, window=fft_taper,
                          noverlap=self.nlap)

        # leave out first entry (offset)
        spec = spec[1:]

        # working with the periods not frequencies later so reverse spectrum
        spec = spec[::-1]

        # avoid calculating log of zero
        idx = spec < dtiny
        spec[idx] = dtiny

        # go to dB
        spec = np.log10(spec)
        spec *= 10

        spec_octaves = []
        # do this for the whole period range and append the values to our lists
        for per_left, per_right in zip(self.per_octaves_left,
                                       self.per_octaves_right):
            specs = spec[(per_left <= self.per) & (self.per <= per_right)]
            spec_center = specs.mean()
            spec_octaves.append(spec_center)
        spec_octaves = np.array(spec_octaves)

        hist, self.xedges, self.yedges = np.histogram2d(
            self.per_octaves,
            spec_octaves, bins=(self.period_bins, self.spec_bins))

        try:
            # we have to make sure manually that the bins are always the same!
            # this is done with the various assert() statements above.
            self.hist_stack += hist
        except TypeError:
            # only during first run initialize stack with first histogram
            self.hist_stack = hist
        return True

    def get_percentile(self, percentile=50, hist_cum=None):
        """
        Returns periods and approximate psd values for given percentile value.

        :type percentile: int
        :param percentile: percentile for which to return approximate psd
                value. (e.g. a value of 50 is equal to the median.)
        :type hist_cum: :class:`numpy.ndarray`, optional
        :param hist_cum: if it was already computed beforehand, the normalized
                cumulative histogram can be provided here (to avoid computing
                it again), otherwise it is computed from the currently stored
                histogram.
        :returns: (periods, percentile_values)
        """
        if hist_cum is None:
            hist_cum = self.__get_normalized_cumulative_histogram()
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
        percentile_values = self.spec_bins[percentile_values]
        return (self.period_bin_centers, percentile_values)

    def get_mode(self):
        """
        Returns periods and mode psd values (i.e. for each frequency the psd
        value with the highest probability is selected).

        :returns: (periods, psd mode values)
        """
        db_bin_centers = (self.spec_bins[:-1] + self.spec_bins[1:]) / 2.0
        mode = db_bin_centers[self.hist_stack.argmax(axis=1)]
        return (self.period_bin_centers, mode)

    def get_mean(self):
        """
        Returns periods and mean psd values (i.e. for each frequency the mean
        psd value is selected).

        :returns: (periods, psd mean values)
        """
        db_bin_centers = (self.spec_bins[:-1] + self.spec_bins[1:]) / 2.0
        mean = (self.hist_stack * db_bin_centers /
                len(self.times_used)).sum(axis=1)
        return (self.period_bin_centers, mean)

    def __get_normalized_cumulative_histogram(self):
        """
        Returns the current histogram in a cumulative version normalized per
        period column, i.e. going from 0 to 1 from low to high psd values for
        every period column.
        """
        # sum up the columns to cumulative entries
        hist_cum = self.hist_stack.cumsum(axis=1)
        # normalize every column with its overall number of entries
        # (can vary from the number of self.times because of values outside
        #  the histogram db ranges)
        norm = hist_cum[:, -1].copy()
        # avoid zero division
        norm[norm == 0] = 1
        hist_cum = (hist_cum.T / norm).T
        return hist_cum

    def save(self, filename, compress=False):
        """
        Saves the PPSD as a pickled file with optional compression.

        The resulting file can be restored using PPSD.load(filename).

        :type filename: str
        :param filename: Name of output file with pickled PPSD object
        :type compress: bool, optional
        :param compress: Enable/disable file compression.
        """
        if compress:
            # due to an bug in older python version we can't use with
            # http://bugs.python.org/issue8601
            file_ = bz2.BZ2File(filename, 'wb')
            pickle.dump(self, file_)
            file_.close()
        else:
            with open(filename, 'wb') as file_:
                pickle.dump(self, file_)

    @staticmethod
    def load(filename):
        """
        Restores a PPSD instance from a file.

        Automatically determines whether the file was saved with compression
        enabled or disabled.

        :type filename: str
        :param filename: Name of file containing the pickled PPSD object
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
            # cf. http://docs.python.org/2/library/pickle.html
            #
            # due to an bug in older python version we can't use with
            # http://bugs.python.org/issue8601
            file_ = bz2.BZ2File(filename, 'rb')
            ppsd = pickle.load(file_)
            file_.close()
        else:
            with open(filename, 'rb') as file_:
                ppsd = pickle.load(file_)

        return ppsd

    def plot(self, filename=None, show_coverage=True, show_histogram=True,
             show_percentiles=False, percentiles=[0, 25, 50, 75, 100],
             show_noise_models=True, grid=True, show=True,
             max_percentage=30, period_lim=(0.01, 179), show_mode=False,
             show_mean=False):
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
        :param max_percentage: Maximum percentage to adjust the colormap.
        :type period_lim: tuple of 2 floats, optional
        :param period_lim: Period limits to show in histogram.
        :type show_mode: bool, optional
        :param show_mode: Enable/disable plotting of mode psd values.
        :type show_mean: bool, optional
        :param show_mean: Enable/disable plotting of mean psd values.
        """
        # check if any data has been added yet
        if self.hist_stack is None:
            msg = 'No data to plot'
            raise Exception(msg)

        X, Y = np.meshgrid(self.xedges, self.yedges)
        hist_stack = self.hist_stack * 100.0 / len(self.times_used)

        fig = plt.figure()

        if show_coverage:
            ax = fig.add_axes([0.12, 0.3, 0.90, 0.6])
            ax2 = fig.add_axes([0.15, 0.17, 0.7, 0.04])
        else:
            ax = fig.add_subplot(111)

        if show_histogram:
            ppsd = ax.pcolormesh(X, Y, hist_stack.T, cmap=self.colormap)
            cb = plt.colorbar(ppsd, ax=ax)
            cb.set_label("[%]")
            color_limits = (0, max_percentage)
            ppsd.set_clim(*color_limits)
            cb.set_clim(*color_limits)
            if grid:
                ax.grid(b=grid, which="major")
                ax.grid(b=grid, which="minor")

        if show_percentiles:
            hist_cum = self.__get_normalized_cumulative_histogram()
            # for every period look up the approximate place of the percentiles
            for percentile in percentiles:
                periods, percentile_values = \
                    self.get_percentile(percentile=percentile,
                                        hist_cum=hist_cum)
                ax.plot(periods, percentile_values, color="black")

        if show_mode:
            periods, mode_ = self.get_mode()
            ax.plot(periods, mode_, color="black")

        if show_mean:
            periods, mean_ = self.get_mean()
            ax.plot(periods, mean_, color="black")

        if show_noise_models:
            model_periods, high_noise = get_NHNM()
            ax.plot(model_periods, high_noise, '0.4', linewidth=2)
            model_periods, low_noise = get_NLNM()
            ax.plot(model_periods, low_noise, '0.4', linewidth=2)

        ax.semilogx()
        ax.set_xlim(period_lim)
        ax.set_ylim(self.spec_bins[0], self.spec_bins[-1])
        ax.set_xlabel('Period [s]')
        ax.set_ylabel('Amplitude [dB]')
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        title = "%s   %s -- %s  (%i segments)"
        title = title % (self.id, self.times_used[0].date,
                         self.times_used[-1].date, len(self.times_used))
        ax.set_title(title)

        if show_coverage:
            self.__plot_coverage(ax2)
            # emulating fig.autofmt_xdate():
            for label in ax2.get_xticklabels():
                label.set_ha("right")
                label.set_rotation(30)

        plt.draw()
        if filename is not None:
            plt.savefig(filename)
            plt.close()
        elif show:
            plt.show()

    def plot_coverage(self, filename=None):
        """
        Plot the data coverage of the histogram of the current PPSD.
        If a filename is specified the plot is saved to this file, otherwise
        a plot window is shown.

        :type filename: str, optional
        :param filename: Name of output file
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        self.__plot_coverage(ax)
        fig.autofmt_xdate()
        title = "%s   %s -- %s  (%i segments)"
        title = title % (self.id, self.times_used[0].date,
                         self.times_used[-1].date, len(self.times_used))
        ax.set_title(title)

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
        self.__check_ppsd_length()
        ax.figure
        ax.clear()
        ax.xaxis_date()
        ax.set_yticks([])

        # plot data coverage
        starts = [date2num(t.datetime) for t in self.times_used]
        ends = [date2num((t + self.ppsd_length).datetime)
                for t in self.times_used]
        for start, end in zip(starts, ends):
            ax.axvspan(start, end, 0, 0.7, alpha=0.5, lw=0)
        # plot data
        for start, end in self.times_data:
            start = date2num(start.datetime)
            end = date2num(end.datetime)
            ax.axvspan(start, end, 0.7, 1, facecolor="g", lw=0)
        # plot gaps
        for start, end in self.times_gaps:
            start = date2num(start.datetime)
            end = date2num(end.datetime)
            ax.axvspan(start, end, 0.7, 1, facecolor="r", lw=0)

        ax.autoscale_view()


def get_NLNM():
    """
    Returns periods and psd values for the New Low Noise Model.
    For information on New High/Low Noise Model see [Peterson2003]_.
    """
    data = np.load(NOISE_MODEL_FILE)
    periods = data['model_periods']
    nlnm = data['low_noise']
    return (periods, nlnm)


def get_NHNM():
    """
    Returns periods and psd values for the New High Noise Model.
    For information on New High/Low Noise Model see [Peterson2003]_.
    """
    data = np.load(NOISE_MODEL_FILE)
    periods = data['model_periods']
    nlnm = data['high_noise']
    return (periods, nlnm)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
