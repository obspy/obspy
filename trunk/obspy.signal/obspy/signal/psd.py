#!/usr/bin/env python
#------------------------------------------------------------------------------
# Filename: psd.py
#  Purpose: Various Routines Related to Spectral Estimation
#   Author: Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2011 Tobias Megies
#------------------------------------------------------------------------------
"""
Various Routines Related to Spectral Estimation

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import with_statement
import os
import warnings
import pickle
import math
import bisect
import numpy as np
from obspy.core import Trace, Stream
from obspy.signal import cosTaper
from obspy.signal.util import prevpow2

try:
    # Import matplotlib routines. These are no official dependency of
    # obspy.signal so an import error should really only be raised if any
    # routine is used which relies on matplotlib (at the moment: psd, PPSD).
    import matplotlib
    from matplotlib import mlab
    import matplotlib.pyplot as plt
    from matplotlib.dates import date2num
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.mlab import detrend_none, window_hanning
    MATPLOTLIB_VERSION = map(int, matplotlib.__version__.split("."))
except ImportError:
    # if matplotlib is not present be silent about it and only raise the
    # ImportError if matplotlib actually is used (currently in psd() and
    # PPSD())
    MATPLOTLIB_VERSION = None
    msg_matplotlib_ImportError = "Failed to import matplotlib. While this " \
            "is no dependency of obspy.signal it is however necessary for a " \
            "few routines. Please install matplotlib in order to be able " \
            "to use e.g. psd() or PPSD()."
    # set up two dummy functions. this makes it possible to make the docstring
    # of psd() look like it should with two functions as default values for
    # kwargs although matplotlib might not be present and the routines
    # therefore not usable
    def detrend_none(): pass
    def window_hanning(): pass


# build colormap as done in paper by mcnamara
CDICT = {'red': ((0.0,  1.0, 1.0),
                 (0.05,  1.0, 1.0),
                 (0.2,  0.0, 0.0),
                 (0.4,  0.0, 0.0),
                 (0.6,  0.0, 0.0),
                 (0.8,  1.0, 1.0),
                 (1.0,  1.0, 1.0)),
         'green': ((0.0,  1.0, 1.0),
                   (0.05,  0.0, 0.0),
                   (0.2,  0.0, 0.0),
                   (0.4,  1.0, 1.0),
                   (0.6,  1.0, 1.0),
                   (0.8,  1.0, 1.0),
                   (1.0,  0.0, 0.0)),
         'blue': ((0.0,  1.0, 1.0),
                  (0.05,  1.0, 1.0),
                  (0.2,  1.0, 1.0),
                  (0.4,  1.0, 1.0),
                  (0.6,  0.0, 0.0),
                  (0.8,  0.0, 0.0),
                  (1.0,  0.0, 0.0))}
NOISE_MODEL_FILE = os.path.join(os.path.dirname(__file__),
                                "data", "noise_models.npz")
# do not change these variables, otherwise results may differ from PQLX!
PPSD_LENGTH = 3600 # psds are calculated on 1h long segments
PPSD_STRIDE = 1800 # psds are calculated overlapping, moving 0.5h ahead


def psd(x, NFFT=256, Fs=2, detrend=detrend_none, window=window_hanning, noverlap=0):
    """
    Wrapper for `matplotlib.mlab.psd`.

    Always returns a onesided psd (positive frequencies only), corrects for
    this fact by scaling with a factor of 2. Also, always normalizes to dB/Hz
    by dividing with sampling rate.
    
    This wrapper is intended to intercept changes in `mlab.psd`'s default
    behavior which changes with matplotlib version 0.98.4:
    
      - http://matplotlib.sourceforge.net/
                users/whats_new.html#psd-amplitude-scaling
      - http://matplotlib.sourceforge.net/_static/CHANGELOG
                (entries on 2009-05-18 and 2008-11-11)
      - http://matplotlib.svn.sourceforge.net/
                viewvc/matplotlib?view=revision&revision=6518
      - http://matplotlib.sourceforge.net/
                api/api_changes.html#changes-for-0-98-x

    :note:
        For details on all arguments see `matplotlib.mlab.psd`_.

    .. _`matplotlib.mlab.psd`: http://matplotlib.sourceforge.net/api/mlab_api.html#matplotlib.mlab.psd
    """
    # check if matplotlib is available, no official dependency for obspy.signal
    if MATPLOTLIB_VERSION is None:
        raise ImportError(msg_matplotlib_ImportError)

    # check matplotlib version
    elif MATPLOTLIB_VERSION >= [0, 98, 4]:
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
    Cosine taper, 10 percent at each end.
    Like done in mcnamara etal.
    Caution: inplace operation, so data should be float.
    """
    data *= cosTaper(len(data), 0.2)
    return data


class PPSD():
    """
    Class to compile probabilistic power spectral densities for one combination
    of network/station/location/channel/sampling_rate.
    Calculations are based on the routine used by::

        Daniel E. McNamara and Raymond P. Buland
        Ambient noise levels in the continental United States
        Bulletin of the Seismological Society of America,
        (August 2004), 94(4):1517-1527

    For information on high/low noise models see::

        Jon Peterson
        Observations and modeling of seismic background noise
        USGS Open File Report 93-322
        http://ehp3-earthquake.wr.usgs.gov/regional/asl/pubs/files/ofr93-322.pdf

    Basic Usage
    -----------

    >>> from obspy.core import read
    >>> from obspy.signal import PPSD

    >>> st = read()
    >>> tr = st.select(channel="EHZ")[0]
    >>> paz = {'gain': 60077000.0, 
    ...        'poles': [-0.037004+0.037016j, -0.037004-0.037016j,
    ...                  -251.33+0j, -131.04-467.29j, -131.04+467.29j],
    ...        'sensitivity': 2516778400.0,
    ...        'zeros': [0j, 0j]}

    >>> ppsd = PPSD(tr.stats, paz)
    >>> print ppsd.id
    BW.RJOB..EHZ
    >>> print ppsd.times
    []

    Now we could add data to the probabilistic psd and plot it like..

    >>> ppsd.add(st) # doctest: +SKIP
    >>> print ppsd.times # doctest: +SKIP
    >>> ppsd.plot() # doctest: +SKIP

    .. but the example stream is too short and does not contain enough data.

    For a real world example see the `ObsPy Tutorial`_.

    .. _`ObsPy Tutorial`: http://www.obspy.org/wiki/ObspyTutorial
    """
    def __init__(self, stats, paz=None, merge_method=0):
        """
        Initialize the PPSD object setting all fixed information on the station
        that should not change afterwards to guarantee consistent spectral
        estimates.
        
        :type stats: :class:`~obspy.core.trace.Stats`
        :param stats: Stats of the station/instrument to process
        :type paz: dict (optional)
        :param paz: Response information of instrument. If not specified the
                information is supposed to be present as stats.paz.
        :type merge_method: int (optional)
        :param merge_method: Merging method to use. McNamara&Buland merge gappy
                traces by filling with zeros. This results in a clearly
                identifiable outlier psd line in the PPSD visualization. Select
                `merge_method=-1` for not filling gaps with zeros which might
                result in some data segments shorter than 1 hour not used in
                the PPSD.
        """
        # check if matplotlib is available, no official dependency for
        # obspy.signal
        if MATPLOTLIB_VERSION is None:
            raise ImportError(msg_matplotlib_ImportError)

        self.id = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats
        self.network = stats.network
        self.station = stats.station
        self.location = stats.location
        self.channel = stats.channel
        self.sampling_rate = stats.sampling_rate
        self.delta = 1.0 / self.sampling_rate
        # trace length for one hour piece
        self.len = int(self.sampling_rate * PPSD_LENGTH)
        # set paz either from kwarg or try to get it from stats
        self.paz = paz
        if self.paz is None:
            self.paz = stats.paz
        self.merge_method = merge_method
        #if paz is None:
        #    try:
        #        paz = tr.stats.paz
        #    except:
        #        msg = "No paz provided and no paz information found " + \
        #              "in trace.stats.paz"
        #        raise Exception(msg)
        # nfft is determined mimicing the fft setup in McNamara&Buland paper:
        # (they take 13 segments overlapping 75% and truncate to next lower
        #  power of 2)
        #  - take number of points of whole ppsd segment (currently 1 hour)
        self.nfft = PPSD_LENGTH * self.sampling_rate
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
        self.colormap = LinearSegmentedColormap('mcnamara', CDICT, 1024)

    def __setup_bins(self):
        """
        Makes an initial dummy psd and thus sets up the bins and all the rest.
        Should be able to do it without a dummy psd..
        """
        dummy = np.empty(self.len)
        spec, freq = mlab.psd(dummy, self.nfft, self.sampling_rate, noverlap=self.nlap)

        # leave out first entry (offset)
        freq = freq[1:]

        per = 1.0 / freq[::-1]
        self.freq = freq
        self.per = per
        # calculate left/rigth edge of first period bin, width of bin is one octave
        per_left = per[0] / 2
        per_right = 2 * per_left
        # calculate center period of first period bin
        per_center = math.sqrt(per_left * per_right)
        # calculate mean of all spectral values in the first bin
        per_octaves_left = [per_left]
        per_octaves_right = [per_right]
        per_octaves = [per_center]
        # we move through the period range at 1/8 octave steps
        factor_eighth_octave = 2**0.125
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
        # set up the binning for the db scale
        self.spec_bins = np.linspace(-200, -50, 301, endpoint=True)

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
        self.times_data += [[tr.stats.starttime, tr.stats.endtime] for tr in stream]

    def __check_time_present(self, utcdatetime):
        """
        Checks if the given UTCDateTime is already part of the current PPSD
        instance. That is, checks if from utcdatetime to utcdatetime plus 1
        hour there is already data in the PPSD.
        Returns True if adding an one hour piece starting at the given time
        would result in an overlap of the ppsd data base, False if it is OK to
        insert this piece of data.
        """
        index1 = bisect.bisect_left(self.times_used, utcdatetime)
        index2 = bisect.bisect_right(self.times_used, utcdatetime + PPSD_LENGTH)
        if index1 != index2:
            return True
        else:
            return False

    def add(self, stream, verbose=False):
        """
        Process all traces with compatible information and add their spectral
        estimates to the histogram containg the probabilistic psd.
        Also ensures that no piece of data is inserted twice.

        :type stream: :class:`~obspy.core.stream.Stream` or
                :class:`~obspy.core.trace.Trace`
        :param stream: Stream or trace with data that should be added to the
                probabilistic psd histogram.
        :returns: True if appropriate data were found and the ppsd statistics
                were changed, False otherwise.
        """
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
        # merge depending on merge_method set during __init__
        stream.merge(self.merge_method, fill_value=0)

        for tr in stream:
            # the following check should not be necessary due to the select()..
            if not self.__sanity_check(tr):
                msg = "Skipping incompatible trace."
                warnings.warn(msg)
                continue
            t1 = tr.stats.starttime
            t2 = tr.stats.endtime
            while t1 + PPSD_LENGTH <= t2:
                if self.__check_time_present(t1):
                    msg = "Already covered time spans detected (e.g. %s), " + \
                          "skipping these slices."
                    msg = msg % t1
                    warnings.warn(msg)
                else:
                    # throw warnings if trace length is different than one hour..!?!
                    slice = tr.slice(t1, t1 + PPSD_LENGTH)
                    # XXX not good, should be working in place somehow
                    # XXX how to do it with the padding, though?
                    self.__process(slice)
                    self.__insert_used_time(t1)
                    if verbose:
                        print t1
                    changed = True
                t1 += PPSD_STRIDE # advance half an hour

            # enforce time limits, pad zeros if gaps
            #tr.trim(t, t+PPSD_LENGTH, pad=True)
        return changed
            
    def __process(self, tr):
        """
        Processes a one-hour segment of data and adds the information to the
        PPSD histogram. If Trace is compatible (station, channel, ...) has to
        checked beforehand.

        :type tr: :class:`~obspy.core.trace.Trace`
        :param tr: Compatible Trace with
        """
        # XXX DIRTY HACK!!
        if len(tr) == self.len + 1:
            tr.data = tr.data[:-1]
        # one last check..
        if len(tr) != self.len:
            msg = "Got an non-one-hour piece of data to process. Skipping"
            warnings.warn(msg)
            print len(tr), self.len
            return
        # being paranoid, only necessary if in-place operations would follow
        tr.data = tr.data.astype("float64")
        # if trace has a masked array we fill in zeros
        try:
            tr.data[tr.data.mask] = 0.0
        # if its no masked array, we get an AttributeError and have nothing to do
        except AttributeError:
            pass

        # restitution:
        # mcnamara apply the correction at the end in freq-domain,
        # does it make a difference?
        # probably should be done earlier on bigger junk of data?!
        tr.simulate(paz_remove=self.paz, remove_sensitivity=True,
                    paz_simulate=None, simulate_sensitivity=False)

        # go to acceleration:
        tr.data = np.gradient(tr.data, self.delta)

        # use our own wrapper for mlab.psd to have consistent results on all
        # matplotlib versions
        spec, freq = psd(tr.data, self.nfft, self.sampling_rate,
                         detrend=mlab.detrend_linear, window=fft_taper,
                         noverlap=self.nlap)

        # convert to acceleration
        #try:
        #    paz['zeros'].remove(0j)
        #except ValueError:
        #    msg = "No zero at 0j to remove from PAZ in order to convert to acceleration."
        #    raise Exception(msg)

        # remove instrument response
        #import ipdb;ipdb.set_trace()
        #freq_response = pazToFreqResp(paz['poles'], paz['zeros'], paz['gain'], delta, nfft)
        #XXX  alternative version with sanity check if frequencies match:
        #XXX resp_amp, resp_freq = pazToFreqResp(paz['poles'], paz['zeros'],paz['gain'], delta, nfft * 2, freq=True)
        #XXX np.testing.assert_array_equal(freq, resp_freq)
        #water_level = 600.0
        #specInv(freq_response, water_level)
        #freq_response = (freq_response.real * freq_response.real) * (freq_response.imag * freq_response.imag)
        #freq_response /= paz['sensitivity']**2
        #spec *= freq_response
        #XXX freq_response = np.sqrt(freq_response.real**2 + freq_response.imag**2)
        #XXX spec *= freq_response
        #XXX spec *= freq_response

        # leave out first entry (offset)
        spec = spec[1:]

        # working with the periods not frequencies later so reverse spectrum
        spec = spec[::-1]

        # go to dB
        spec = np.log10(spec)
        spec *= 10

        spec_octaves = []
        # do this for the whole period range and append the values to our lists
        for per_left, per_right in zip(self.per_octaves_left, self.per_octaves_right):
            spec_center = spec[(per_left <= self.per) & (self.per <= per_right)].mean()
            spec_octaves.append(spec_center)
        spec_octaves = np.array(spec_octaves)

        hist, self.xedges, self.yedges = np.histogram2d(self.per_octaves, spec_octaves, bins=(self.period_bins, self.spec_bins))

        try:
            # we have to make sure manually that the bins are always the same!
            # this is done with the various assert() statements above.
            self.hist_stack += hist
        except TypeError:
            # only during first run initialize stack with first histogram
            self.hist_stack = hist

    def save(self, filename):
        """
        Saves PPSD instance as a pickled file that can be loaded again using
        pickle.load(filename).

        :type filename: str
        :param filename: Name of output file with pickled PPSD object
        """
        with open(filename, "w") as file:
            pickle.dump(self, file)

    def plot(self, filename=None, show_coverage=True):
        """
        Plot the 2D histogram of the current PPSD.
        If a filename is specified the plot is saved to this file, otherwise
        a plot window is shown.

        :type filename: str (optional)
        :param filename: Name of output file
        :type show_coverage: bool (optional)
        :param show_coverage: Enable/disable second axes with representation of
                data coverage time intervals.
        """
        X, Y = np.meshgrid(self.xedges, self.yedges)
        hist_stack = self.hist_stack * 100.0 / len(self.times_used)
        
        fig = plt.figure()

        if show_coverage:
            ax = fig.add_axes([0.12, 0.3, 0.90, 0.6])
            ax2 = fig.add_axes([0.15, 0.17, 0.7, 0.04])
        else:
            ax = fig.add_subplot(111)

        ppsd = ax.pcolor(X, Y, hist_stack.T, cmap=self.colormap)
        cb = plt.colorbar(ppsd, ax=ax)
        cb.set_label("[%]")

        data = np.load(NOISE_MODEL_FILE)
        model_periods = data['model_periods']
        high_noise = data['high_noise']
        low_noise = data['low_noise']
        ax.plot(model_periods, high_noise, '0.4', linewidth=2)
        ax.plot(model_periods, low_noise, '0.4', linewidth=2)

        color_limits = (0, 30)
        ppsd.set_clim(*color_limits)
        cb.set_clim(*color_limits)
        ax.semilogx()
        ax.set_xlim(0.01, 179)
        ax.set_ylim(-200, -50)
        ax.set_xlabel('Period [s]') 
        ax.set_ylabel('Amplitude [dB]')
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        title = "%s   %s -- %s  (%i segments)"
        title = title % (self.id, self.times_used[0].date, self.times_used[-1].date,
                         len(self.times_used))
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
        else:
            plt.show()

    def plot_coverage(self, filename=None):
        """
        Plot the data coverage of the histogram of the current PPSD.
        If a filename is specified the plot is saved to this file, otherwise
        a plot window is shown.

        :type filename: str (optional)
        :param filename: Name of output file
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        self.__plot_coverage(ax)
        fig.autofmt_xdate()
        title = "%s   %s -- %s  (%i segments)"
        title = title % (self.id, self.times_used[0].date, self.times_used[-1].date,
                         len(self.times_used))
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
        fig = ax.figure
        ax.clear()
        ax.xaxis_date()
        ax.set_yticks([])
        
        # plot data coverage
        starts = [date2num(t.datetime) for t in self.times_used]
        ends = [date2num((t+PPSD_LENGTH).datetime) for t in self.times_used]
        for start, end in zip(starts, ends):
            ax.axvspan(start, end, 0, 0.7, alpha=0.5)
        # plot data
        for start, end in self.times_data:
            start = date2num(start.datetime)
            end = date2num(end.datetime)
            ax.axvspan(start, end, 0.7, 1, facecolor="g")
        # plot gaps
        for start, end in self.times_gaps:
            start = date2num(start.datetime)
            end = date2num(end.datetime)
            ax.axvspan(start, end, 0.7, 1, facecolor="r")
        
        ax.autoscale_view()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
