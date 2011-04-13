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
from matplotlib import mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
from obspy.core import Trace, Stream
from obspy.signal import cosTaper #, pazToFreqResp, specInv
from obspy.signal.util import nearestPow2


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
CM_MCNAMARA = LinearSegmentedColormap('mcnamara', CDICT, 1024)
NOISE_MODEL_FILE = os.path.join(os.path.dirname(__file__),
                                "data", "noise_models.npz")
PSD_LENGTH = 3600 # psds are calculated on 1h long segments
PSD_STRIDE = 1800 # psds are calculated overlapping, moving 0.5h ahead


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

    """
    def __init__(self, stats, paz=None):
        """
        Initialize the PPSD object setting all fixed information on the station
        that should not change afterwards to guarantee consistent spectral
        estimates.
        
        :type stats: :class:`~obspy.core.trace.Stats`
        :param stats: Stats of the station/instrument to process
        :type paz: dict (optional)
        :param paz: Response information of instrument. If not specified the
                information is supposed to be present as stats.paz.
        """
        self.id = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats
        self.network = stats.network
        self.station = stats.station
        self.location = stats.location
        self.channel = stats.channel
        self.sampling_rate = stats.sampling_rate
        self.delta = 1.0 / self.sampling_rate
        # trace length for one hour piece
        self.len = int(self.sampling_rate * PSD_LENGTH)
        # set paz either from kwarg or try to get it from stats
        self.paz = paz
        if self.paz is None:
            self.paz = stats.paz
        #if paz is None:
        #    try:
        #        paz = tr.stats.paz
        #    except:
        #        msg = "No paz provided and no paz information found " + \
        #              "in trace.stats.paz"
        #        raise Exception(msg)
        # setup some other values for the spectral estimates
        #self.trace_length = len(tr)
        # mcnamara: 2**15 sec at 40Hz ~ 2**18 points
        self.nfft = 2**15 * 40 / 2
        self.nfft = int(nearestPow2(self.nfft))
        # mcnamara uses always 13 segments per hour trimming to nfft
        # we leave the psd routine alone and use as many segments as possible
        self.nlap = int(0.75 * self.nfft)
        self.times = []
        self.hist_stack = None
        self.__setup_bins()

    def __setup_bins(self):
        """
        Makes an initial dummy psd and thus sets up the bins and all the rest.
        Should be able to do it without a dummy psd..
        """
        dummy = np.empty(self.len)
        spec, freq = mlab.psd(dummy, self.nfft, self.sampling_rate, noverlap=self.nlap)

        # leave out first entry (offset)
        freq = freq[1:]

        # XXX mcnamara etal do no frequency binning but instead a octave scaled geometric mean afterwards
        # XXX period_bins = np.logspace(math.log10(1.0/freq[-1]), math.log10(1.0/freq[1]), 301, endpoint=True)
        # XXX spec_bins = np.linspace(-200, -50, 151, endpoint=True)
        # XXX hist, xedges, yedges = np.histogram2d(1.0/freq, spec, bins=(period_bins, spec_bins))

        # XXX XXX XXX
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
        """
        if trace.id != self.id:
            return False
        if trace.stats.sampling_rate != self.sampling_rate:
            return False
        return True

    def __insert_time(self, utcdatetime):
        """
        Inserts the given UTCDateTime at the right position in the list keeping
        the order intact.
        """
        bisect.insort(self.times, utcdatetime)

    def __check_time_present(self, utcdatetime):
        """
        Checks if the given UTCDateTime is already part of the current PPSD
        instance. That is, checks if from utcdatetime to utcdatetime plus 1
        hour there is already data in the PPSD.
        Returns True if adding an one hour piece starting at the given time
        would result in an overlap of the ppsd data base, False if it is OK to
        insert this piece of data.
        """
        index1 = bisect.bisect_left(self.times, utcdatetime)
        index2 = bisect.bisect_right(self.times, utcdatetime + PSD_LENGTH)
        if index1 != index2:
            return True
        else:
            return False

    def add(self, stream):
        """
        Process all traces with compatible information and add their spectral
        estimates to the histogram containg the probabilistic psd.
        Also ensures that no piece of data is inserted twice.

        :type stream: :class:`~obspy.core.stream.Stream` or
                :class:`~obspy.core.trace.Trace`
        :param stream: Stream or trace with data that should be added to the
                probabilistic psd histogram.
        """
        # prepare the list of traces to go through
        if isinstance(stream, Trace):
            stream = Stream([stream])
        stream.merge(-1)
        traces = stream.select(id=self.id,
                               sampling_rate=self.sampling_rate).traces

        for tr in traces:
            # the following check should not be necessary due to the select()..
            if not self.__sanity_check(tr):
                msg = "Skipping incompatible trace."
                warnings.warn(msg)
                continue
            t1 = tr.stats.starttime
            t2 = tr.stats.endtime
            if self.__check_time_present(t1):
                msg = "Time seems to be covered already, skipping trace."
                warnings.warn(msg)
                continue
            while t1 + PSD_LENGTH < t2:
                # throw warnings if trace length is different than one hour..!?!
                slice = tr.slice(t1, t1 + PSD_LENGTH)
                # XXX not good, should be working in place somehow
                # XXX how to do it with the padding, though?
                self.__process(slice)
                t1 += PSD_STRIDE # advance half an hour

            # enforce time limits, pad zeros if gaps
            #tr.trim(t, t+PSD_LENGTH, pad=True)
            
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

        # restitution
        # XXX mcnamara apply the correction at the end in freq-domain,
        # XXX does it make a difference?
        # XXX probably should be done earlier on bigger junk of data?!?!
        tr.simulate(paz_remove=self.paz, remove_sensitivity=True,
                    paz_simulate=None, simulate_sensitivity=False)

        # XXX try to go to acceleration:
        tr.data = np.gradient(tr.data, self.delta)

        # XXX if isinstance(tr.data, np.ma.MaskedArray):
        # XXX     print "omitting masked array"
        # XXX     continue

        spec, freq = mlab.psd(tr.data, self.nfft, self.sampling_rate,
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

        # fft rescaling?!?
        # siehe Buttkus: Spektralanalyse und... S.185 Gl. (9.94)
        spec *= 2.0 * self.delta

        #spec = spec*(freq**2)
        #specs[i, :] = np.log10(spec[1:, :]) * 10
        # XXX by roberto: what does it do? -> acceleration?
        # XXX spec *= (freq**2)
        spec = np.log10(spec)
        spec *= 10

        # XXX mcnamara etal do no frequency binning but instead a octave scaled geometric mean afterwards
        # XXX period_bins = np.logspace(math.log10(1.0/freq[-1]), math.log10(1.0/freq[1]), 301, endpoint=True)
        # XXX spec_bins = np.linspace(-200, -50, 151, endpoint=True)
        # XXX hist, xedges, yedges = np.histogram2d(1.0/freq, spec, bins=(period_bins, spec_bins))

        # XXX this should be done only one during __init__
        # XXX XXX XXX
        spec_octaves = []
        # do this for the whole period range and append the values to our lists
        for per_left, per_right in zip(self.per_octaves_left, self.per_octaves_right):
            #print per_left, per_right
            spec_center = spec[(per_left <= self.per) & (self.per <= per_right)].mean()
            spec_octaves.append(spec_center)
        spec_octaves = np.array(spec_octaves)

        hist, self.xedges, self.yedges = np.histogram2d(self.per_octaves, spec_octaves, bins=(self.period_bins, self.spec_bins))
        # XXX XXX XXX

        try:
            # we have to make sure manually that the bins are always the same!
            # this is done with the various assert() statements above.
            self.hist_stack += hist
        except TypeError:
            # only during first run initialize stack with first histogram
            self.hist_stack = hist
        self.__insert_time(tr.stats.starttime)

    def save(self, filename):
        """
        Saves PPSD instance as a pickled file that can be loaded again using
        pickle.load(filename).

        :type filename: str
        :param filename: Name of output file with pickled PPSD object
        """
        with open(filename, "w") as file:
            pickle.dump(self, file)

    def plot(self, filename=None):
        """
        Plot the 2D histogram of the current PPSD.
        If a filename is specified the plot is saved to this file, otherwise
        a plot window is shown.

        :type filename: str (optional)
        :param filename: Name of output file
        """
        X, Y = np.meshgrid(self.xedges, self.yedges)
        hist_stack = self.hist_stack * 100.0 / len(self.times)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ppsd = ax.pcolor(X, Y, hist_stack.T, cmap=CM_MCNAMARA)
        cb = plt.colorbar(ppsd)

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

        ax.set_title(self.id)
        plt.draw()
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
