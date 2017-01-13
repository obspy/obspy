#!/usr/bin/env python
"""
Seismic array class.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import copy
import math
import warnings

import numpy as np
from scipy import interpolate
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, MultipleLocator

from obspy.core import UTCDateTime
from obspy.geodetics import degrees2kilometers
from obspy.imaging import cm


def _plot_array_analysis(out, sllx, slmx, slly, slmy, sls,
                         filename_patterns, baz_plot, method,
                         st_workon, starttime, wlen, endtime):
    """
    Some plotting taken out from _array_analysis_helper. Can't do the array
    response overlay now though.
    :param baz_plot: Whether to show backazimuth-slowness map (True) or
     slowness x-y map (False).
    """
    import matplotlib.pyplot as plt
    trace = []
    t, rel_power, abs_power, baz, slow = out.T
    baz[baz < 0.0] += 360
    # now let's do the plotting
    cmap = cm.rainbow
    # we will plot everything in s/deg
    slow = degrees2kilometers(slow)
    sllx = degrees2kilometers(sllx)
    slmx = degrees2kilometers(slmx)
    slly = degrees2kilometers(slly)
    slmy = degrees2kilometers(slmy)
    sls = degrees2kilometers(sls)

    numslice = len(t)
    powmap = []

    slx = np.arange(sllx - sls, slmx, sls)
    sly = np.arange(slly - sls, slmy, sls)
    if baz_plot:
        maxslowg = np.sqrt(slmx * slmx + slmy * slmy)
        bzs = np.arctan2(sls, np.sqrt(
            slmx * slmx + slmy * slmy)) * 180 / np.pi
        xi = np.arange(0., maxslowg, sls)
        yi = np.arange(-180., 180., bzs)
        grid_x, grid_y = np.meshgrid(xi, yi)
    # reading in the rel-power maps
    for i in range(numslice):
        powmap.append(np.load(filename_patterns[0] % i))
        if method != 'FK':
            trace.append(np.load(filename_patterns[1] % i))
    # remove last item as a cludge to get plotting to work - not sure
    # it's always clever or just a kind of rounding or modulo problem
    if len(slx) == len(powmap[0][0]) + 1:
        slx = slx[:-1]
    if len(sly) == len(powmap[0][1]) + 1:
        sly = sly[:-1]

    npts = st_workon[0].stats.npts
    df = st_workon[0].stats.sampling_rate
    t = np.arange(0, npts / df, 1 / df)

    # if we choose windowlen > 0. we now move through our slices
    for i in range(numslice):
        st = UTCDateTime(t[i]) - starttime
        if wlen <= 0:
            en = endtime
        else:
            en = st + wlen
        print(UTCDateTime(t[i]))
        # add polar and colorbar axes
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_axes([0.1, 0.87, 0.7, 0.10])
        # here we plot the first trace on top of the slowness map
        # and indicate the possibiton of the lsiding window as green box
        if method == 'FK':
            ax1.plot(t, st_workon[0].data, 'k')
            if wlen > 0.:
                try:
                    ax1.axvspan(st, en, facecolor='g', alpha=0.3)
                except IndexError:
                    pass
        else:
            t = np.arange(0, len(trace[i]) / df, 1 / df)
            ax1.plot(t, trace[i], 'k')

        ax1.yaxis.set_major_locator(MaxNLocator(3))

        ax = fig.add_axes([0.10, 0.1, 0.70, 0.7])

        # if we have chosen the baz_plot option a re-griding
        # of the sx,sy slowness map is needed
        if baz_plot:
            slowgrid = []
            power = np.asarray(powmap[i])
            for ix, sx in enumerate(slx):
                for iy, sy in enumerate(sly):
                    bbaz = np.arctan2(sx, sy) * 180 / np.pi + 180.
                    if bbaz > 180.:
                        bbaz = -180. + (bbaz - 180.)
                    slowgrid.append((np.sqrt(sx * sx + sy * sy), bbaz,
                                     power[ix, iy]))

            slowgrid = np.asarray(slowgrid)
            sl = slowgrid[:, 0]
            bz = slowgrid[:, 1]
            slowg = slowgrid[:, 2]
            grid = interpolate.griddata((sl, bz), slowg,
                                        (grid_x, grid_y),
                                        method='nearest')
            ax.pcolormesh(xi, yi, grid, cmap=cmap)

            ax.set_xlabel('slowness [s/deg]')
            ax.set_ylabel('backazimuth [deg]')
            ax.set_xlim(xi[0], xi[-1])
            ax.set_ylim(yi[0], yi[-1])
        else:
            ax.set_xlabel('slowness [s/deg]')
            ax.set_ylabel('slowness [s/deg]')
            slow_x = np.cos((baz[i] + 180.) * np.pi / 180.) * slow[i]
            slow_y = np.sin((baz[i] + 180.) * np.pi / 180.) * slow[i]
            ax.pcolormesh(slx, sly, powmap[i].T)
            ax.arrow(0, 0, slow_y, slow_x, head_width=0.005,
                     head_length=0.01, fc='k', ec='k')
            ax.set_ylim(slx[0], slx[-1])
            ax.set_xlim(sly[0], sly[-1])
        new_time = t[i]

        result = "BAZ: %.2f, Slow: %.2f s/deg, Time %s" % (
            baz[i], slow[i], UTCDateTime(new_time))
        ax.set_title(result)
        plt.show()


class BeamformerResult(object):
    """
    Contains results from beamforming and attached plotting methods.

    This class is an attempt to standardise the output of the various
    beamforming algorithms in :class:`SeismicArray`, and provide a range of
    plotting options. To that end, the raw output from the beamformers is
    combined with metadata to identify what method produced it with which
    parameters, including: the frequency and slowness ranges, time frame and
    the inventory defining the array (this will only include the stations for
    which data were actually available during the considered time frame).

    .. rubric:: Data transformation

    The different beamforming algorithms produce output in different
    formats, so it needs to be standardised. The
    :meth:`.SeismicArray.slowness_whitened_power`,
    :meth:`.SeismicArray.phase_weighted_stack`,
    :meth:`.SeismicArray.delay_and_sum`,
    and :meth:`.SeismicArray.fk_analysis` routines internally perform grid
    searches of slownesses in x and y directions, producing numpy arrays of
    power for every slowness, time window and discrete frequency. Only the
    maximum relative and absolute powers of each time window are returned,
    as well as the slowness values at which they were found. The x-y
    slowness values are also converted to radial slowness and backazimuth.
    Returning the complete data arrays for these methods is not currently
    implemented.

    Working somewhat differently,
    :meth:`.SeismicArray.three_component_beamforming` returns a
    four-dimensional numpy array of relative powers for every backazimuth,
    slowness, time window and discrete frequency, but no absolute powers.
    This data allows the creation of the same plots as the previous methods,
    as the maximum powers is easily calulated from the full power array.

    .. rubric:: Concatenating results

    To allow for the creation of beamformer output plots over long
    timescales where the beamforming might performed not all at once,
    the results may be concatenated (added). This is done by simple adding
    syntax:

    >>> long_results = beamresult_1 + beamresult_2 # doctest:+SKIP

    Of course, this only makes sense if the two added result objects are
    consistent. The :meth:`__add__` method checks that they were created by
    the same beamforming method, with the same slowness and frequency
    ranges. Only then are the data combined.

    :param inventory: The inventory that was actually used in the beamforming.
    :param win_starttimes: Start times of the beamforming windows.
    :type win_starttimes: numpy array of
     :class:`obspy.core.utcdatetime.UTCDateTime`
    :param slowness_range: The slowness range used for the beamforming.
    :param max_rel_power: Maximum relative power at every timestep.
    :param max_abs_power: Maximum absolute power at every timestep.
    :param max_pow_baz: Backazimuth of the maximum power value at every
     timestep.
    :param max_pow_slow: Slowness of the maximum power value at every timestep.
    :param full_beamres: 4D numpy array holding relative power results for
     every backazimuth, slowness, window and discrete frequency (in that
     order).
    :param freqs: The discrete frequencies used for which the full_beamres
     was computed.
    :param incidence: Rayleigh wave incidence angles (only from three
     component beamforming).
    :param method: Method used for the beamforming.
    """

    def __init__(self, inventory, win_starttimes, slowness_range,
                 max_rel_power=None, max_abs_power=None, max_pow_baz=None,
                 max_pow_slow=None, full_beamres=None, freqs=None,
                 incidence=None, method=None, timestep=None):

        self.inventory = copy.deepcopy(inventory)
        self.win_starttimes = win_starttimes
        self.starttime = win_starttimes[0]
        if timestep is not None:
            self.timestep = timestep
        elif timestep is None and len(win_starttimes) == 1:
            msg = "Can't calculate a timestep. Please set manually."
            warnings.warn(msg)
            self.timestep = None
        else:
            self.timestep = win_starttimes[1] - win_starttimes[0]
        if self.timestep is not None:
            # Don't use unjustified higher precision.
            try:
                self.endtime = UTCDateTime(win_starttimes[-1] + self.timestep,
                                           precision=win_starttimes[-1].
                                           precision)
            except AttributeError:
                self.endtime = UTCDateTime(win_starttimes[-1] + self.timestep)
        self.max_rel_power = max_rel_power
        if max_rel_power is not None:
            self.max_rel_power = self.max_rel_power.astype(float)
        self.max_abs_power = max_abs_power
        if max_abs_power is not None:
            self.max_abs_power = self.max_abs_power.astype(float)
        self.max_pow_baz = max_pow_baz
        if max_pow_baz is not None:
            self.max_pow_baz = self.max_pow_baz.astype(float)
        self.max_pow_slow = max_pow_slow
        if self.max_pow_slow is not None:
            self.max_pow_slow = self.max_pow_slow.astype(float)
        if len(slowness_range) == 1:
            raise ValueError("Need at least two slowness values.")
        self.slowness_range = slowness_range.astype(float)
        self.freqs = freqs
        self.incidence = incidence
        self.method = method

        if full_beamres is not None and full_beamres.ndim != 4:
            raise ValueError("Full beamresults should be 4D array.")
        self.full_beamres = full_beamres
        # FK and other 1cbf return max relative (and absolute) powers,
        # as well as the slowness and azimuth where appropriate. 3cbf as of
        # now returns the whole results.
        if(max_rel_power is None and max_pow_baz is None and
                max_pow_slow is None and full_beamres is not None):
            self._calc_max_values()

    def __add__(self, other):
        """
        Add two sequential BeamformerResult instances. Must have been created
        with identical frequency and slowness ranges.
        """
        if not isinstance(other, BeamformerResult):
            raise TypeError('unsupported operand types')
        if self.method != other.method:
            raise ValueError('Methods must be equal.')
        if self.freqs is None or other.freqs is None:
            attrs = ['slowness_range']
        else:
            attrs = ['freqs', 'slowness_range']
        if any((self.__dict__[attr] != other.__dict__[attr])
               for attr in attrs):
            raise ValueError('Frequency and slowness range parameters must be '
                             'equal.')
        times = np.append(self.win_starttimes, other.win_starttimes)
        if self.full_beamres is not None and other.full_beamres is not None:
            full_beamres = np.append(self.full_beamres,
                                     other.full_beamres, axis=2)
            max_rel_power, max_pow_baz, \
                max_pow_slow, max_abs_power = None, None, None, None
        else:
            full_beamres = None
            max_rel_power = np.append(self.max_rel_power, other.max_rel_power)
            max_pow_baz = np.append(self.max_pow_baz, other.max_pow_baz)
            max_pow_slow = np.append(self.max_pow_slow, other.max_pow_slow)
            if self.max_abs_power is not None:
                max_abs_power = np.append(self.max_abs_power,
                                          other.max_abs_power)
            else:
                max_abs_power = None

        out = self.__class__(self.inventory, times,
                             full_beamres=full_beamres,
                             slowness_range=self.slowness_range,
                             freqs=self.freqs,
                             method=self.method,
                             max_abs_power=max_abs_power,
                             max_rel_power=max_rel_power,
                             max_pow_baz=max_pow_baz,
                             max_pow_slow=max_pow_slow)
        return out

    def _calc_max_values(self):
        """
        If the maximum power etc. values are unset, but the full results are
        available (currently only from :meth:`three_component_beamforming`),
        calculate the former from the latter.
        """
        # Average over all frequencies.
        freqavg = self.full_beamres.mean(axis=3)
        num_win = self.full_beamres.shape[2]
        self.max_rel_power = np.empty(num_win, dtype=float)
        self.max_pow_baz = np.empty_like(self.max_rel_power)
        self.max_pow_slow = np.empty_like(self.max_rel_power)
        for win in range(num_win):
            self.max_rel_power[win] = freqavg[:, :, win].max()
            ibaz = np.where(freqavg[:, :, win] == freqavg[:, :, win].max())[0]
            islow = np.where(freqavg[:, :, win] == freqavg[:, :, win].max())[1]
            # Add [0] in case of multiple matches.
            self.max_pow_baz[win] = np.arange(0, 362, 2)[ibaz[0]]
            self.max_pow_slow[win] = self.slowness_range[islow[0]]

    def _get_plotting_timestamps(self, extended=False):
        """
        Convert the times to the time reference matplotlib uses and return as
        timestamps. Returns the timestamps in days (decimals represent hours,
        minutes and seconds) since '0001-01-01T00:00:00' as needed for
        matplotlib date plotting (see e.g. matplotlibs num2date).
        """
        if extended:
            # With pcolormesh, will miss one window if only plotting window
            # start times.
            plot_times = list(self.win_starttimes)
            plot_times.append(self.endtime)
        else:
            plot_times = self.win_starttimes
        # Honestly, this is black magic to me.
        newtimes = np.array([t.timestamp / (24*3600) + 719163
                             for t in plot_times])
        return newtimes

    def plot_baz_hist(self, show=True):
        """
        Plot a backazimuth - slowness radial histogram.

        The backazimuth and slowness values of the maximum relative powers
        of each beamforming window are counted into bins defined
        by slowness and backazimuth, weighted by the power.

        :param show: Whether to call plt.show() immediately.
        """
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import Normalize
        import matplotlib.pyplot as plt
        cmap = cm.viridis
        # Can't plot negative slownesses:
        sll = abs(self.slowness_range).min()
        slm = self.slowness_range.max()

        # choose number of azimuth bins in plot
        # (desirably 360 degree/azimuth_bins is an integer!)
        azimuth_bins = 36
        # number of slowness bins
        slowness_bins = len(self.slowness_range)
        # Plot is not too readable beyond a certain number of bins.
        slowness_bins = 30 if slowness_bins > 30 else slowness_bins
        abins = np.arange(azimuth_bins + 1) * 360. / azimuth_bins
        sbins = np.linspace(sll, slm, slowness_bins + 1)

        # sum rel power in bins given by abins and sbins
        hist, baz_edges, sl_edges = \
            np.histogram2d(self.max_pow_baz, self.max_pow_slow,
                           bins=[abins, sbins], weights=self.max_rel_power)

        # transform to radian
        baz_edges = np.radians(baz_edges)

        # add polar and colorbar axes
        fig = plt.figure(figsize=(8, 8))
        cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
        ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")

        dh = abs(sl_edges[1] - sl_edges[0])
        dw = abs(baz_edges[1] - baz_edges[0])

        # circle through backazimuth
        for i, row in enumerate(hist):
            ax.bar(left=(i * dw) * np.ones(slowness_bins),
                   height=dh * np.ones(slowness_bins),
                   width=dw, bottom=dh * np.arange(slowness_bins),
                   color=cmap(row / hist.max()))

        ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        ax.set_xticklabels(['N', 'E', 'S', 'W'])

        # set slowness limits
        ax.set_ylim(sll, slm)
        ColorbarBase(cax, cmap=cmap,
                     norm=Normalize(vmin=hist.min(), vmax=hist.max()))
        plt.suptitle('{} beamforming: results from \n{} to {}'
                     .format(self.method, self.starttime,
                             self.endtime))
        if show is True:
            plt.show()

    def plot_bf_results_over_time(self, show=True):
        """
        Plot beamforming results over time, with the relative power as
        colorscale.

        :param show: Whether to call plt.show() immediately.
        """
        import matplotlib.pyplot as plt
        labels = ['Rel. Power', 'Abs. Power', 'Backazimuth', 'Slowness']
        datas = [self.max_rel_power, self.max_abs_power,
                 self.max_pow_baz, self.max_pow_slow]
        # To account for e.g. the _beamforming method not returning absolute
        # powers:
        for data, lab in zip(reversed(datas), reversed(labels)):
            if data is None:
                datas.remove(data)
                labels.remove(lab)

        xlocator = mdates.AutoDateLocator(interval_multiples=True)
        ymajorlocator = MultipleLocator(90)

        fig = plt.figure()
        for i, (data, lab) in enumerate(zip(datas, labels)):
            ax = fig.add_subplot(len(labels), 1, i + 1)
            ax.scatter(self._get_plotting_timestamps(), data,
                       c=self.max_rel_power, alpha=0.6, edgecolors='none',
                       cmap=cm.viridis)
            ax.set_ylabel(lab)
            timemargin = 0.05 * (self._get_plotting_timestamps()[-1] -
                                 self._get_plotting_timestamps()[0])
            ax.set_xlim(self._get_plotting_timestamps()[0] - timemargin,
                        self._get_plotting_timestamps()[-1] + timemargin)
            if lab == 'Backazimuth':
                ax.set_ylim(0, 360)
                ax.yaxis.set_major_locator(ymajorlocator)
            else:
                datamargin = 0.05 * (data.max() - data.min())
                ax.set_ylim(data.min() - datamargin, data.max() + datamargin)
            ax.xaxis.set_major_locator(xlocator)
            ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))

        fig.suptitle('{} beamforming: results from \n{} to {}'
                     .format(self.method, self.starttime,
                             self.endtime))
        fig.autofmt_xdate()
        fig.subplots_adjust(left=0.15, top=0.9, right=0.95, bottom=0.2,
                            hspace=0.1)
        if show is True:
            plt.show()

    def plot_power(self, plot_frequency=None, show=True):
        """
        Plot relative power as a function of backazimuth and time, like a
        Vespagram.

        Requires full 4D results, at the moment only provided by
        :meth:`three_component_beamforming`.
        :param plot_frequencies: Discrete frequencies for which windows
         should be plotted, otherwise an average of frequencies is plotted.
        :param show: Whether to call plt.show() immediately.
        """
        import matplotlib.pyplot as plt
        if self.full_beamres is None:
            raise ValueError('Insufficient data. Try other plotting options.')
        if plot_frequency is not None:
            # Prepare data.
            # works because freqs is a range
            ifreq = np.searchsorted(self.freqs, float(plot_frequency))
            freqavg = np.squeeze(self.full_beamres[:, :, :, ifreq])
        else:
            freqavg = self.full_beamres.mean(axis=3)
        num_win = self.full_beamres.shape[2]
        # This is 2D, with time windows and baz (in this order)
        # as indices.
        maxazipows = np.array([[azipows.T[t].max() for azipows in freqavg]
                               for t in range(num_win)])
        azis = np.arange(0, 362, 2)
        labels = ['baz']
        maskedazipows = np.ma.array(maxazipows, mask=np.isnan(maxazipows))
        # todo (maybe) implement plotting of slowness map corresponding to the
        # max powers
        datas = [maskedazipows]  # , maskedslow]

        xlocator = mdates.AutoDateLocator(interval_multiples=True)
        ymajorlocator = MultipleLocator(90)
        fig = plt.figure()
        for i, (data, lab) in enumerate(zip(datas, labels)):
            ax = fig.add_subplot(len(labels), 1, i + 1)

            pc = ax.pcolormesh(self._get_plotting_timestamps(extended=True),
                               azis, data.T, cmap=cm.viridis,
                               rasterized=True)
            timemargin = 0.05 * (self._get_plotting_timestamps(extended=True
                                                               )[-1] -
                                 self._get_plotting_timestamps()[0])
            ax.set_xlim(self._get_plotting_timestamps()[0] - timemargin,
                        self._get_plotting_timestamps(extended=True)[-1] +
                        timemargin)
            ax.set_ylim(0, 360)
            ax.yaxis.set_major_locator(ymajorlocator)
            cbar = fig.colorbar(pc)
            cbar.solids.set_rasterized(True)
            ax.set_ylabel('Backazimuth')
            ax.xaxis.set_major_locator(xlocator)
            ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))

        fig.suptitle('{} beamforming: results from \n{} to {}'
                     .format(self.method, self.starttime,
                             self.endtime))
        fig.autofmt_xdate()
        fig.subplots_adjust(left=0.15, top=0.9, right=0.95, bottom=0.2,
                            hspace=0.1)
        if show is True:
            plt.show()

    def plot_bf_plots(self, average_windows=True, average_freqs=True,
                      plot_frequencies=None, show=True):
        """
        Plot beamforming results as individual polar plots of relative power as
        function of backazimuth and slowness.

        Can plot results averaged over windows/frequencies, results for each
        window and every frequency individually, or for selected frequencies
        only.

        :param average_windows: Whether to plot an average of results over all
         windows.
        :param average_freqs: Whether to plot an average of results over all
         frequencies.
        :param plot_frequencies: Tuple of discrete frequencies (f1, f2) for
         which windows should be plotted, if not provided an average of
         frequencies is plotted (ignored if average_freqs is True).
        :param show: Whether to call plt.show() immediately.
        """
        import matplotlib.pyplot as plt
        if self.full_beamres is None:
            raise ValueError('Insufficient data for this plotting method.')
        if average_freqs is True and plot_frequencies is not None:
            warnings.warn("Ignoring plot_frequencies, only plotting an average"
                          " of all frequencies.")
        if(hasattr(plot_frequencies, '__getitem__') is False and
           plot_frequencies is not None):
            plot_frequencies = tuple([plot_frequencies])

        theo_backazi = np.arange(0, 362, 2) * math.pi / 180.

        def _actual_plotting(bfres, title):
            """
            Pass in a 2D bfres array of beamforming results with
            averaged or selected windows and frequencies.
            """
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='polar')
            cmap = cm.viridis
            contf = ax.contourf(theo_backazi, self.slowness_range, bfres.T,
                                100, cmap=cmap, antialiased=True,
                                linstyles='dotted')
            ax.contour(theo_backazi, self.slowness_range, bfres.T, 100,
                       cmap=cmap)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_rmax(self.slowness_range[-1])
            # Setting this to -0 means that if the slowness range doesn't
            # start at 0, the plot is shown with a 'hole' in the middle
            # rather than stitching it up. This behaviour shows the
            # resulting shapes much better.
            ax.set_rmin(-0)
            fig.colorbar(contf)
            ax.grid(True)
            ax.set_title(title)

        beamresnz = self.full_beamres
        if average_windows:
            beamresnz = beamresnz.mean(axis=2)
        if average_freqs:
            # Always an average over the last axis, whether or not windows
            # were averaged.
            beamresnz = beamresnz.mean(axis=beamresnz.ndim - 1)

        if average_windows and average_freqs:
            _actual_plotting(beamresnz,
                             '{} beamforming result, averaged over all time '
                             'windows\n ({} to {}) and frequencies.'
                             .format(self.method, self.starttime,
                                     self.endtime))

        if average_windows and not average_freqs:
            if plot_frequencies is None:
                warnings.warn('No frequency specified for plotting.')
            else:
                for plot_freq in plot_frequencies:
                    # works because freqs is a range
                    ifreq = np.searchsorted(self.freqs, plot_freq)
                    _actual_plotting(np.squeeze(beamresnz[:, :, ifreq]),
                                     '{} beamforming result, averaged over all'
                                     ' time windows\n for frequency {} Hz.'
                                     .format(self.method, self.freqs[ifreq]))

        if average_freqs and not average_windows:
            for iwin in range(len(beamresnz[0, 0, :])):
                _actual_plotting(np.squeeze(beamresnz[:, :, iwin]),
                                 '{} beamforming result, averaged over all '
                                 'frequencies,\n for window {} '
                                 '(starting {})'
                                 .format(self.method, iwin,
                                         self.win_starttimes[iwin]))

        # Plotting all windows, selected frequencies.
        if average_freqs is False and average_windows is False:
            if plot_frequencies is None:
                warnings.warn('No frequency specified for plotting.')
            else:
                for plot_freq in plot_frequencies:
                    ifreq = np.searchsorted(self.freqs, plot_freq)
                    for iwin in range(len(beamresnz[0, 0, :, 0])):
                        _actual_plotting(beamresnz[:, :, iwin, ifreq],
                                         '{} beamforming result, for frequency'
                                         ' {} Hz,\n, window {} (starting {}).'
                                         .format(self.method,
                                                 self.freqs[ifreq], iwin,
                                                 self.win_starttimes[iwin]))

        if show is True:
            plt.show()

    def plot_radial_transfer_function(self, plot_freqs=None):
        """
        Plot the radial transfer function of the array and slowness range used
        to produce this result.

        :param plot_freqs: List of discrete frequencies for which the transfer
         function should be plotted. Defaults to the minimum and maximum of the
         frequency range used in the generation of this results object.
        """
        from obspy.signal.array_analysis import SeismicArray

        if plot_freqs is None:
            plot_freqs = [self.freqs[0], self.freqs[-1]]
        if type(plot_freqs) is float or type(plot_freqs) is int:
            plot_freqs = [plot_freqs]

        # Need absolute values:
        absolute_slownesses = [abs(_) for _ in self.slowness_range]

        # Need to create an array object:
        plot_array = SeismicArray('plot_array', self.inventory)
        plot_array.plot_radial_transfer_function(min(absolute_slownesses),
                                                 max(absolute_slownesses),
                                                 self.slowness_range[1] -
                                                 self.slowness_range[0],
                                                 plot_freqs)

    def plot_transfer_function_freqslowness(self):
        """
        Plot an x-y transfer function of the array used to produce this
        result, as a function of the set slowness and frequency ranges.
        """
        from obspy.signal.array_analysis import SeismicArray

        # Need absolute values:
        absolute_slownesses = [abs(_) for _ in self.slowness_range]
        # Need to create an array object:
        arr = SeismicArray('plot_array', self.inventory)
        arr.plot_transfer_function_freqslowness(max(absolute_slownesses),
                                                self.slowness_range[1] -
                                                self.slowness_range[0],
                                                min(self.freqs),
                                                max(self.freqs),
                                                abs(self.freqs[1] -
                                                    self.freqs[0]))


if __name__ == '__main__':
    import doctest

    doctest.testmod(exclude_empty=True)
