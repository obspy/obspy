# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: waveform.py
#  Purpose: Waveform plotting for obspy.Stream objects
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2011 Lion Krischer
#---------------------------------------------------------------------
"""
Waveform plotting for obspy.Stream objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU General Public License (GPL)
    (http://www.gnu.org/licenses/gpl.txt)
"""

from copy import deepcopy, copy
from datetime import datetime
from math import ceil
from matplotlib.cm import hsv #@UnresolvedImport
from matplotlib.dates import date2num, num2date
from matplotlib.ticker import FuncFormatter
from obspy.core import UTCDateTime, Stream, Trace
from obspy.core.preview import mergePreviews
import StringIO
import matplotlib.pyplot as plt
import numpy as np
import warnings


class WaveformPlotting(object):
    """
    Class that provides several solutions for plotting large and small waveform
    data sets.

    It uses matplotlib to plot the waveforms.
    """

    def __init__(self, **kwargs):
        """
        Checks some variables and maps the kwargs to class variables.
        """
        self.stream = kwargs.get('stream')
        # Check if it is a Stream or a Trace object.
        if isinstance(self.stream, Trace):
            self.stream = Stream([self.stream])
        elif not isinstance(self.stream, Stream):
            msg = 'Plotting is only supported for Stream or Trace objects.'
            raise TypeError(msg)
        # Stream object should contain at least one Trace
        if len(self.stream) < 1:
            msg = "Empty stream object"
            raise IndexError(msg)
        # Type of the plot.
        self.type = kwargs.get('type', 'normal')
        # Start- and endtimes of the plots.
        self.starttime = kwargs.get('starttime', None)
        self.endtime = kwargs.get('endtime', None)
        self.fig_obj = kwargs.get('fig', None)
        # If no times are given take the min/max values from the stream object.
        if not self.starttime:
            self.starttime = min([trace.stats.starttime for \
                             trace in self.stream])
        if not self.endtime:
            self.endtime = max([trace.stats.endtime for \
                           trace in self.stream])
        # Map stream object and slice just in case.
        self.stream = self.stream.slice(self.starttime, self.endtime)
        # normalize times
        if self.type == 'relative':
            dt = self.starttime
            # fix plotting boundaries
            self.endtime = UTCDateTime(self.endtime - self.starttime)
            self.starttime = UTCDateTime(0)
            # fix stream times
            for tr in self.stream:
                tr.stats.starttime = UTCDateTime(tr.stats.starttime - dt)
        # Below that value the data points will be plotted normally. Above it
        # the data will be plotted using a different approach (details see
        # below).
        self.max_npts = 400000
        # If automerge is enabled. Merge traces with the same id for the plot.
        self.automerge = kwargs.get('automerge', True)
        # Set default values.
        # The default value for the size is determined dynamically because
        # there might be more than one channel to plot.
        self.size = kwargs.get('size', None)
        # Values that will be used to calculate the size of the plot.
        self.default_width = 800
        self.default_height_per_channel = 250
        if not self.size:
            self.width = 800
            # Check the kind of plot.
            if self.type == 'dayplot':
                self.height = 600
            else:
                # One plot for each trace.
                if self.automerge:
                    count = []
                    for tr in self.stream:
                        if hasattr(tr.stats, 'preview') and tr.stats.preview:
                            tr_id = tr.id + 'preview'
                        else:
                            tr_id = tr.id
                        if not tr_id in count:
                            count.append(tr_id)
                    count = len(count)
                else:
                    count = len(self.stream)
                self.height = count * 250
        else:
            self.width, self.height = self.size
        # Interval length in minutes for dayplot.
        self.interval = 60 * kwargs.get('interval', 15)
        # Scaling.
        self.vertical_scaling_range = kwargs.get('vertical_scaling_range',
                                                 None)
        # Dots per inch of the plot. Might be useful for printing plots.
        self.dpi = kwargs.get('dpi', 100)
        # Color of the graph.
        if self.type == 'dayplot':
            self.color = kwargs.get('color', ('#B2000F', '#004C12', '#847200',
                                              '#0E01FF'))
            if isinstance(self.color, basestring):
                self.color = (self.color,)
            self.number_of_ticks = kwargs.get('number_of_ticks', None)
        else:
            self.color = kwargs.get('color', 'k')
            self.number_of_ticks = kwargs.get('number_of_ticks', 5)
        # Background and face color.
        self.background_color = kwargs.get('bgcolor', 'w')
        self.face_color = kwargs.get('face_color', 'w')
        # Transparency. Overwrites background and facecolor settings.
        self.transparent = kwargs.get('transparent', False)
        if self.transparent:
            self.background_color = None
        # Ticks.
        self.tick_format = kwargs.get('tick_format', '%H:%M:%S')
        self.tick_rotation = kwargs.get('tick_rotation', 0)
        # Whether or not to save a file.
        self.outfile = kwargs.get('outfile')
        self.handle = kwargs.get('handle')
        # File format of the resulting file. Usually defaults to PNG but might
        # be dependent on your matplotlib backend.
        self.format = kwargs.get('format')

    def plotWaveform(self, *args, **kwargs):
        """
        Creates a graph of any given ObsPy Stream object. It either saves the
        image directly to the file system or returns an binary image string.

        For all color values you can use legit HTML names, HTML hex strings
        (e.g. '#eeefff') or you can pass an R , G , B tuple, where each of
        R , G , B are in the range [0, 1]. You can also use single letters for
        basic built-in colors ('b' = blue, 'g' = green, 'r' = red, 'c' = cyan,
        'm' = magenta, 'y' = yellow, 'k' = black, 'w' = white) and gray shades
        can be given as a string encoding a float in the 0-1 range.
        """
        # Setup the figure if not passed explicitly.
        if not self.fig_obj:
            self.__setupFigure()
        else:
            self.fig = self.fig_obj
        # Determine kind of plot and do the actual plotting.
        if self.type == 'dayplot':
            self.plotDay(*args, **kwargs)
        else:
            self.plot(*args, **kwargs)
        self.fig.canvas.draw()
        # The following just serves as a unified way of saving and displaying
        # the plots.
        if self.outfile:
            if not self.transparent:
                extra_args = {'dpi':self.dpi,
                              'facecolor':self.face_color,
                              'edgecolor':self.face_color}
            else:
                extra_args = {'dpi':self.dpi,
                              'transparent': self.transparent}
            # If format is set use it.
            if self.format:
                self.fig.savefig(self.outfile, format=self.format,
                                 **extra_args)
            # Otherwise use format from self.outfile or default to PNG.
            else:
                self.fig.savefig(self.outfile, **extra_args)
        else:
            # Return an binary imagestring if not self.outfile but self.format.
            if self.format:
                imgdata = StringIO.StringIO()
                self.fig.savefig(imgdata, format=self.format,
                                 **extra_args)
                imgdata.seek(0)
                return imgdata.read()
            elif self.handle:
                return self.fig
            else:
                if not self.fig_obj:
                    plt.show()

    def plot(self, *args, **kwargs):
        """
        Plot the Traces showing one graph per Trace.

        Plots the whole time series for self.max_npts points and less. For more
        points it plots minmax values.
        """
        stream_new = []
        # Just remove empty traces.
        if not self.automerge:
            for tr in self.stream:
                stream_new.append([])
                if len(tr.data):
                    stream_new[-1].append(tr)
        else:
            # Generate sorted list of traces (no copy)
            # Sort order, id, starttime, endtime
            ids = []
            for tr in self.stream:
                if hasattr(tr.stats, 'preview') and tr.stats.preview:
                    id = tr.id + 'preview'
                else:
                    id = tr.id
                if not id in ids:
                    ids.append(id)
            for id in ids:
                stream_new.append([])
                for tr in self.stream:
                    if hasattr(tr.stats, 'preview') and tr.stats.preview:
                        tr_id = tr.id + 'preview'
                    else:
                        tr_id = tr.id
                    if tr_id == id:
                        # does not copy the elements of the data array
                        tr_ref = copy(tr)
                        # Trim does nothing if times are outside
                        if self.starttime >= tr_ref.stats.endtime or \
                                self.endtime <= tr_ref.stats.starttime:
                            continue
                        if tr_ref.data.size:
                            stream_new[-1].append(tr_ref)
                # delete if empty list
                if not len(stream_new[-1]):
                    stream_new.pop()
                    continue
                stream_new[-1].sort(key=lambda x: x.stats.endtime)
                stream_new[-1].sort(key=lambda x: x.stats.starttime)
        # If everything is lost in the process raise an Exception.
        if not len(stream_new):
            raise Exception("Nothing to plot")
        # Create helper variable to track ids and min/max/mean values.
        self.stats = []
        # Loop over each Trace and call the appropriate plotting method.
        self.axis = []
        for _i, tr in enumerate(stream_new):
            # Each trace needs to have the same sampling rate.
            sampling_rates = set([_tr.stats.sampling_rate for _tr in tr])
            if len(sampling_rates) > 1:
                msg = "All traces with the same id need to have the same " + \
                      "sampling rate."
                raise Exception(msg)
            sampling_rate = sampling_rates.pop()
            if self.background_color:
                ax = self.fig.add_subplot(len(stream_new), 1, _i + 1,
                                          axisbg=self.background_color)
            else:
                ax = self.fig.add_subplot(len(stream_new), 1, _i + 1)
            self.axis.append(ax)
            # XXX: Also enable the minmax plotting for previews.
            if (self.endtime - self.starttime) * sampling_rate > self.max_npts:
                self.__plotMinMax(stream_new[_i], ax, *args, **kwargs)
            else:
                self.__plotStraight(stream_new[_i], ax, *args, **kwargs)
        # Set ticks.
        self.__plotSetXTicks()
        self.__plotSetYTicks()

    def plotDay(self, *args, **kwargs):
        """
        Extend the seismogram.
        """
        # Get minmax array.
        self.__dayplotGetMinMaxValues(self, *args, **kwargs)
        # Normalize array
        self.__dayplotNormalizeValues(self, *args, **kwargs)
        # Get timezone information. If none is  given, use local time.
        self.time_offset = kwargs.get('time_offset',
                           round((UTCDateTime(datetime.now()) - \
                           UTCDateTime()) / 3600.0 , 2))
        self.timezone = kwargs.get('timezone', 'local time')
        # Try to guess how many steps are needed to advance one full time unit.
        self.repeat = None
        if self.interval < 60 and 60 % self.interval == 0:
            self.repeat = 60 / self.interval
        elif self.interval < 1800 and 3600 % self.interval == 0:
            self.repeat = 3600 / self.interval
        # Otherwise use a maximum value of 10.
        else:
            if self.steps >= 10:
                self.repeat = 10
            else:
                self.repeat = self.steps
        # Create axis to plot on.
        if self.background_color:
            ax = self.fig.add_subplot(1, 1, 1, axisbg=self.background_color)
        else:
            ax = self.fig.add_subplot(1, 1, 1)
        # Adjust the subplots to be symmetrical. Also make some more room
        # at the top.
        self.fig.subplots_adjust(left=0.12, right=0.88, top=0.88)
        # Create x_value_array.
        aranged_array = np.arange(self.width)
        x_values = np.empty(2 * self.width)
        x_values[0::2] = aranged_array
        x_values[1::2] = aranged_array
        # Loop over each step.
        for _i in xrange(self.steps):
            # Create offset array.
            y_values = np.empty(self.width * 2)
            y_values.fill(self.steps - (_i + 1))
            # Add min and max values.
            y_values[0::2] += self.extreme_values[_i, :, 0]
            y_values[1::2] += self.extreme_values[_i, :, 1]
            # Plot the values.
            ax.plot(x_values, y_values,
                    color=self.color[_i % len(self.color)])
        # Set ranges.
        ax.set_xlim(0, self.width - 1)
        ax.set_ylim(-0.3 , self.steps + 0.3)
        self.axis = [ax]
        # Set ticks.
        self.__dayplotSetYTicks()
        self.__dayplotSetXTicks()
        # Choose to show grid but only on the x axis.
        self.fig.axes[0].grid()
        self.fig.axes[0].yaxis.grid(False)
        # Set the title of the plot.
        self.fig.suptitle(self.stream[0].id, fontsize='medium')

    def __plotStraight(self, trace, ax, *args, **kwargs):  # @UnusedVariable
        """
        Just plots the data samples in the self.stream. Useful for smaller
        datasets up to around 1000000 samples (depending on the machine its
        being run on).

        Slow and high memory consumption for large datasets.
        """
        # Copy to avoid any changes to original data.
        trace = deepcopy(trace)
        if len(trace) > 1:
            stream = Stream(traces=trace)
            # Merge with 'interpolation'. In case of overlaps this method will
            # always use the longest available trace.
            if hasattr(trace[0].stats, 'preview') and trace[0].stats.preview:
                stream = Stream(traces=stream)
                stream = mergePreviews(stream)
            else:
                stream.merge(method=1)
            trace = stream[0]
        else:
            trace = trace[0]
        # Check if it is a preview file and adjust accordingly.
        # XXX: Will look weird if the preview file is too small.
        if hasattr(trace.stats, 'preview') and trace.stats.preview:
            # Mask the gaps.
            trace.data = np.ma.masked_array(trace.data)
            trace.data[trace.data == -1] = np.ma.masked
            # Recreate the min_max scene.
            dtype = trace.data.dtype
            old_time_range = trace.stats.endtime - trace.stats.starttime
            data = np.empty(2 * trace.stats.npts, dtype=dtype)
            data[0::2] = trace.data / 2.0
            data[1::2] = -trace.data / 2.0
            trace.data = data
            # The times are not supposed to change.
            trace.stats.delta = old_time_range / float(trace.stats.npts - 1)
        # Write to self.stats.
        calib = trace.stats.calib
        max = trace.data.max()
        min = trace.data.min()
        if hasattr(trace.stats, 'preview') and trace.stats.preview:
            tr_id = trace.id + ' [preview]'
        else:
            tr_id = trace.id
        self.stats.append([tr_id, calib * trace.data.mean(),
                           calib * min, calib * max])
        # Pad the beginning and the end with masked values if necessary. Might
        # seem like overkill but it works really fast and is a clean solution
        # to gaps at the beginning/end.
        concat = [trace]
        if self.starttime != trace.stats.starttime:
            samples = (trace.stats.starttime - self.starttime) * \
                trace.stats.sampling_rate
            temp = [np.ma.masked_all(int(samples))]
            concat = temp.extend(concat)
            concat = temp
        if self.endtime != trace.stats.endtime:
            samples = (self.endtime - trace.stats.endtime) * \
                      trace.stats.sampling_rate
            concat.append(np.ma.masked_all(int(samples)))
        if len(concat) > 1:
            # Use the masked array concatenate, otherwise it will result in a
            # not masked array.
            trace.data = np.ma.concatenate(concat)
            # set starttime and calculate endtime
            trace.stats.starttime = self.starttime
        trace.data *= calib
        ax.plot(trace.data, color=self.color)
        # Set the x limit for the graph to also show the masked values at the
        # beginning/end.
        ax.set_xlim(0, len(trace.data) - 1)

    def __plotMinMax(self, trace, ax, *args, **kwargs):  # @UnusedVariable
        """
        Plots the data using a min/max approach that calculated the minimum and
        maximum values of each "pixel" and than plots only these values. Works
        much faster with large data sets.
        """
        # Some variables to help calculate the values.
        starttime = self.starttime.timestamp
        endtime = self.endtime.timestamp
        # The same trace will always have the same sampling_rate.
        sampling_rate = trace[0].stats.sampling_rate
        # The samples per resulting pixel.
        pixel_length = int((endtime - starttime) / self.width *
                           sampling_rate)
        # Loop over all the traces. Do not merge them as there are many samples
        # and therefore merging would be slow.
        for _i, _t in enumerate(trace):
            # Get the start of the next pixel in case the starttime of the
            # trace does not match the starttime of the plot.
            ts = _t.stats.starttime
            if ts > self.starttime:
                start = int(ceil(((ts - self.starttime) * \
                        sampling_rate) / pixel_length))
                # Samples before start.
                prestart = int(((self.starttime + start * pixel_length /
                           sampling_rate) - ts) * sampling_rate)
            else:
                start = 0
                prestart = 0
            # Figure out the number of pixels in the current trace.
            length = len(_t.data) - prestart
            pixel_count = int(length // pixel_length)
            rest = int(length % pixel_length)
            # Reference to new data array which does not copy data but is
            # reshapeable.
            data = _t.data[prestart: prestart + pixel_count * pixel_length]
            data = data.reshape(pixel_count, pixel_length)
            # Calculate extreme_values and put them into new array.
            extreme_values = np.ma.masked_all((self.width, 2), dtype=np.float)
            min = data.min(axis=1) * _t.stats.calib
            max = data.max(axis=1) * _t.stats.calib
            extreme_values[start: start + pixel_count, 0] = min
            extreme_values[start: start + pixel_count, 1] = max
            # First and last and last pixel need separate treatment.
            if start and prestart:
                extreme_values[start - 1, 0] = \
                    _t.data[:prestart].min() * _t.stats.calib
                extreme_values[start - 1, 1] = \
                    _t.data[:prestart].max() * _t.stats.calib
            if rest:
                if start + pixel_count == self.width:
                    index = self.width - 1
                else:
                    index = start + pixel_count
                extreme_values[index, 0] = \
                    _t.data[-rest:].min() * _t.stats.calib
                extreme_values[index, 1] = \
                    _t.data[-rest:].max() * _t.stats.calib
            # Use the first array as a reference and merge all following
            # extreme_values into it.
            if _i == 0:
                minmax = extreme_values
            else:
                # Merge minmax and extreme_values.
                min = np.ma.empty((self.width, 2))
                max = np.ma.empty((self.width, 2))
                # Fill both with the values.
                min[:, 0] = minmax[:, 0]
                min[:, 1] = extreme_values[:, 0]
                max[:, 0] = minmax[:, 1]
                max[:, 1] = extreme_values[:, 1]
                # Find the minimum and maximum values.
                min = min.min(axis=1)
                max = max.max(axis=1)
                # Write again to minmax.
                minmax[:, 0] = min
                minmax[:, 1] = max
        # Write to self.stats.
        self.stats.append([trace[0].id, minmax.mean(),
                           minmax[:, 0].min(),
                           minmax[:, 1].max()])
        # Finally plot the data.
        x_values = np.empty(2 * self.width)
        aranged = np.arange(self.width)
        x_values[0::2] = aranged
        x_values[1::2] = aranged
        # Initialize completely masked array. This version is a little bit
        # slower than first creating an empty array and then setting the mask
        # to True. But on NumPy 1.1 this results in a 0-D array which can not
        # be indexed.
        y_values = np.ma.masked_all(2 * self.width)
        y_values[0::2] = minmax[:, 0]
        y_values[1::2] = minmax[:, 1]
        ax.plot(x_values, y_values, color=self.color)
        # Set the x-limit to avoid clipping of masked values.
        ax.set_xlim(0, self.width - 1)

    def __plotSetXTicks(self, *args, **kwargs):  # @UnusedVariable
        """
        Goes through all axes in pyplot and sets time ticks on the x axis.
        """
        # Loop over all axes.
        for ax in self.axis:
            # Get the xlimits.
            start, end = ax.get_xlim()
            # Set the location of the ticks.
            ax.set_xticks(np.linspace(start, end, self.number_of_ticks))
            # Figure out times.
            interval = float(self.endtime - self.starttime) / (self.number_of_ticks
                                                         - 1)
            # Set the actual labels.
            if self.type == 'relative':
                labels = ['%.2f' % (self.starttime + _i * interval).timestamp \
                          for _i in range(self.number_of_ticks)]
            else:
                labels = [(self.starttime + _i * \
                          interval).strftime(self.tick_format) for _i in \
                          range(self.number_of_ticks)]

            ax.set_xticklabels(labels, fontsize='small', rotation=self.tick_rotation)

    def __plotSetYTicks(self, *args, **kwargs):  # @UnusedVariable
        """
        Goes through all axes in pyplot, reads self.stats and sets all ticks on
        the y axis.

        This method also adjusts the y limits so that the mean value is always
        in the middle of the graph and all graphs are equally scaled.
        """
        # Figure out the maximum distance from the mean value to either end.
        # Add 10 percent for better looking graphs.
        max_distance = max([max(trace[1] - trace[2], trace[3] - trace[1])
                            for trace in self.stats]) * 1.1
        # Loop over all axes.
        for _i, ax in enumerate(self.axis):
            mean = self.stats[_i][1]
            # Set the ylimit.
            min_range = mean - max_distance
            max_range = mean + max_distance
            # Set the location of the ticks.
            ticks = [mean - 0.75 * max_distance,
                     mean - 0.5 * max_distance,
                     mean - 0.25 * max_distance,
                     mean,
                     mean + 0.25 * max_distance,
                     mean + 0.5 * max_distance,
                     mean + 0.75 * max_distance]
            ax.set_yticks(ticks)
            # Setup format of the major ticks
            if max(ticks) - min(ticks) > 10:
                fmt = '%d'
            else:
                fmt = '%.2g'
            ax.set_yticklabels([fmt % t for t in ax.get_yticks()],
                               fontsize='small')
            # Set the title of each plot.
            ax.set_title(self.stats[_i][0], horizontalalignment='left',
                      fontsize='small', verticalalignment='center')
            ax.set_ylim(min_range, max_range)

    def __dayplotGetMinMaxValues(self, *args, **kwargs):  # @UnusedVariable
        """
        Takes a Stream object and calculates the min and max values for each
        pixel in the dayplot.

        Writes a three dimensional array. The first axis is the step, i.e
        number of trace, the second is the pixel in that step and the third
        contains the minimum and maximum value of the pixel.
        """
        # XXX: Currently only works for Streams containing just one Trace.
        if len(self.stream) != 1:
            msg = 'Currently only Stream objects with one Trace are supported.'
            raise NotImplementedError(msg)
        # Helper variables for easier access.
        trace = self.stream[0]
        trace_length = len(trace.data)
        # Samples per interval.
        spt = int(self.interval * trace.stats.sampling_rate)
        # Calculate the steps. Cut the result after three digits.
        steps = ceil(trace_length / spt)
        self.steps = int(steps)
        # How many data points in one pixel.
        pixel_width = int(spt // self.width)
        # Create array for min/max values. Use masked arrays to handle gaps.
        extreme_values = np.ma.empty((self.steps, self.width, 2))
        for _i in range(self.steps):
            p0 = _i * spt
            for _j in range(self.width):
                p = p0 + _j * pixel_width
                pe = p + pixel_width
                extreme_values[_i, _j, 0] = min(trace.data[p:pe])
                extreme_values[_i, _j, 1] = max(trace.data[p:pe])
        # Set class variable.
        self.extreme_values = extreme_values

    def __dayplotNormalizeValues(self, *args, **kwargs):  # @UnusedVariable
        """
        Normalizes all values in the 3 dimensional array, so that the minimum
        value will be 0 and the maximum value will be 1.

        It will also convert all values to floats.
        """
        # Convert to native floats.
        self.extreme_values = self.extreme_values.astype(np.float) * \
                              self.stream[0].stats.calib
        # Make sure that the mean value is at 0
        self.extreme_values -= self.extreme_values.mean()

        # Scale so that 99.5 % of the data will fit the given range.
        if self.vertical_scaling_range is None:
            percentile_delta = 0.005
            max_values = self.extreme_values[:, :, 1].ravel().copy()
            min_values = self.extreme_values[:, :, 0].ravel().copy()
            max_values.sort()
            min_values.sort()
            length = len(max_values)
            index = int((1.0 - percentile_delta) * length)
            max_val = max_values[index]
            index = int(percentile_delta * length)
            min_val = min_values[index]
        # Exact fit.
        elif float(self.vertical_scaling_range) == 0.0:
            max_val = self.extreme_values[:, :, 1].max()
            min_val = self.extreme_values[:, :, 0].min()
        # Fit with custom range.
        else:
            max_val = min_val = abs(self.vertical_scaling_range)/2.0

        # Scale from 0 to 1.
        self.extreme_values = (self.extreme_values / max(abs(max_val),
                                                 abs(min_val))) / 2 + 0.5


    def __dayplotSetXTicks(self, *args, **kwargs):  # @UnusedVariable
        """
        Sets the xticks for the dayplot.
        """
        max_value = self.width - 1
        # Check whether it are sec/mins/hours and convert to a universal unit.
        if self.interval < 240:
            time_type = 'seconds'
            time_value = self.interval
        elif self.interval < 24000:
            time_type = 'minutes'
            time_value = self.interval / 60
        else:
            time_type = 'hours'
            time_value = self.interval / 3600
        count = None
        # Hardcode some common values. The plus one is itentional. It had
        # hardly any performance impact and enhances readability.
        if self.interval == 15*60:
            count = 15 + 1
        elif self.interval == 20*60:
            count = 4 + 1
        elif self.interval == 30*60:
            count = 6 + 1
        elif self.interval == 60*60:
            count = 4 + 1
        elif self.interval == 90*60:
            count = 6 + 1
        elif self.interval == 120*60:
            count = 4 + 1
        elif self.interval == 180*60:
            count = 6 + 1
        elif self.interval == 240*60:
            count = 6 + 1
        elif self.interval == 300*60:
            count = 6 + 1
        elif self.interval == 360*60:
            count = 12 + 1
        elif self.interval == 720*60:
            count = 12 + 1
        # Otherwise run some kind of autodetection routine.
        if not count:
            # Up to 15 time units and if its a full number, show every unit.
            if time_value <= 15 and time_value % 1 == 0:
                count = time_value
            # Otherwise determine whether they are dividable for numbers up to
            # 15. If a number is not dividable just show 10 units.
            else:
                count = 10
                for _i in xrange(15, 1, -1):
                    if time_value % _i == 0:
                        count = _i
                        break
            # Show at least 5 ticks.
            if count < 5:
                count = 5
        # Everything can be overwritten by user specified number of ticks.
        if self.number_of_ticks:
            count = self.number_of_ticks
        # Calculate and set ticks.
        ticks = np.linspace(0.0, max_value, count)
        ticklabels = ['%i' % _i for _i in np.linspace(0.0,
                                    time_value, count)]
        self.axis[0].set_xticks(ticks)
        self.axis[0].set_xticklabels(ticklabels, rotation=self.tick_rotation)
        self.axis[0].set_xlabel('time in %s' % time_type)

    def __dayplotSetYTicks(self, *args, **kwargs):  # @UnusedVariable
        """
        Sets the yticks for the dayplot.
        """
        # Do not display all ticks except if it are five or less steps.
        if self.steps <= 5:
            tick_steps = range(0, self.steps)
            ticks = np.arange(self.steps, 0, -1, dtype=np.float)
            ticks -= 0.5
        else:
            tick_steps = range(0, self.steps, self.repeat)
            ticks = np.arange(self.steps, 0, -1 * self.repeat, dtype=np.float)
            ticks -= 0.5
        ticklabels = [(self.starttime + _i * self.interval).strftime('%H:%M') \
                      for _i in tick_steps]
        self.axis[0].set_yticks(ticks)
        self.axis[0].set_yticklabels(ticklabels)
        self.axis[0].set_ylabel('UTC')
        # Save range.
        yrange = self.axis[0].get_ylim()
        # Create twin axis.
        #XXX
        self.twin = self.axis[0].twinx()
        self.twin.set_ylim(yrange)
        self.twin.set_yticks(ticks)
        ticklabels = [(self.starttime + _i * self.interval + self.time_offset \
                      * 3600).strftime('%H:%M') for _i in tick_steps]
        self.twin.set_yticklabels(ticklabels)
        # Complicated way to calculate the label of the y-Axis showing the
        # second time zone.
        sign = '%+i' % self.time_offset
        sign = sign[0]
        time_label = self.timezone.strip() + ' (UTC%s%02i:%02i)' % \
                     (sign, abs(self.time_offset), (self.time_offset % 1 * 60))
        self.twin.set_ylabel(time_label)

    def __setupFigure(self):
        """
        The design and look of the whole plot to be produced.
        """
        # Setup figure and axes
        self.fig = plt.figure(num=None, dpi=self.dpi,
                              figsize=(float(self.width) / self.dpi,
                                       float(self.height) / self.dpi))
        # XXX: Figure out why this is needed sometimes.
        # Set size and dpi.
        self.fig.set_dpi(self.dpi)
        self.fig.set_figwidth(float(self.width) / self.dpi)
        self.fig.set_figheight(float(self.height) / self.dpi)
        # hide time information if set as option
        if self.type == 'relative':
            return
        if self.type == 'dayplot':
            suptitle = self.starttime.strftime('%Y-%m-%d')
            self.fig.suptitle(suptitle, y=0.94, fontsize='small')
        else:
            pattern = '%Y-%m-%dT%H:%M:%SZ'
            suptitle = '%s  -  %s' % (self.starttime.strftime(pattern),
                                      self.endtime.strftime(pattern))
            self.fig.suptitle(suptitle, x=0.02, y=0.96, fontsize='small',
                              horizontalalignment='left')


def _plot_list(streams):

    def formatXTicklabels(x, pos):  # @UnusedVariable
        """
        Make a nice formatting for x axis ticklabels
        """
        # pos is the ticklabel index, None if hovering with the mouse
        dt = UTCDateTime(num2date(x))
        return str(dt).rstrip("0Z").replace("T", " ")

    st = Stream()

    # go through all streams and collect traces
    for item in streams:
        if isinstance(item, Stream):
            for tr in item:
                st.append(tr)
        elif isinstance(item, Trace):
            st.append(item)
        else:
            msg = "Some items in list could not be recognized and were " + \
                  "ignored. Only Trace and Stream items are recognized " + \
                  "as list items at the moment."
            warnings.warn(msg)

    # go through stream and compile unique lists of ids
    comp_ids = list(set([tr.stats.channel[-1] for tr in st]))
    # sort component ids
    for comp in ["Z", "N", "E", "R", "T", "L", "Q", "T"][::-1]:
        try:
            comp_ids.remove(comp)
            comp_ids.insert(0, comp)
        except:
            pass
    sta_ids = ["%s.%s.%s" % (tr.stats.network, tr.stats.station,
               tr.stats.location) for tr in st]
    sta_ids = list(set(sta_ids))
    num_plots = len(comp_ids)
    num_lines = len(sta_ids)

    # assign one color per network-station id
    colors = {}
    for i, sta_id in enumerate(sta_ids):
        colors[sta_id] = hsv(float(i) / num_lines)
    alpha = 2. / len(sta_ids)

    # assign one subplot per component id
    # the first one we do by hand to be able to sharex with it
    fig = plt.figure()
    axs = {}
    ax0 = fig.add_subplot(num_plots, 1, 1)
    axs[comp_ids[0]] = ax0
    for i, comp_id in enumerate(comp_ids[1:]):
        axs[comp_id] = fig.add_subplot(num_plots, 1, i + 2, sharex=ax0)

    # plot every trace with respective color in respective subplot
    for tr in st:
        comp_id = tr.stats.channel[-1]
        sta_id = "%s.%s.%s" % (tr.stats.network, tr.stats.station,
                               tr.stats.location)
        start = date2num(tr.stats.starttime)
        end = date2num(tr.stats.endtime)
        time = np.linspace(start, end, tr.stats.npts)
        axs[comp_id].plot(time, tr.data, ls="-", marker="", c=colors[sta_id],
                          alpha=alpha, label=sta_id)

    # we have to group our axes for sharing x after the plotting commands,
    # plot_date has problems otherwise.
    # as we don't use plot_date anymore, we can do the sharex right at the
    # initialization of the subplots above and dont need to work on the axes
    # grouping low-level.
    ax1 = fig.axes[-1]
    #grouper = ax0._shared_x_axes
    for comp_id, ax in axs.items():
        #grouper.join(ax0, ax)
        ax.text(0.02, 0.95, comp_id, color="b", fontsize=16, ha="left",
                va="top", transform=ax.transAxes)
        ax.xaxis.set_major_formatter(FuncFormatter(formatXTicklabels))
        plt.setp(ax.get_xticklabels(), rotation=20,
                 horizontalalignment="right")
        plt.setp(ax.xaxis.get_ticklabels(), visible=False)
    plt.setp(ax1.xaxis.get_ticklabels(), visible=True)
    ax0.legend()

    fig.subplots_adjust(top=0.95, right=0.95, bottom=0.2, hspace=0)
    plt.show()


def plot_trigger(trace, cft, thrOn, thrOff, show=True):
    """
    Plot characteristic function of trigger along with waveform data and
    trigger On/Off from given thresholds.

    :type trace: :class:`~obspy.core.trace.Trace`
    :param trace: waveform data
    :type cft: :class:`numpy.ndarray`
    :param cft: characteristic function as returned by a trigger in
            :mod:`obspy.signal.trigger`
    :type thrOn: float
    :param thrOn: threshold for switching trigger on
    :type thrOff: float
    :param thrOff: threshold for switching trigger off
    :type show: bool
    :param show: Do not call `plt.show()` at end of routine. That way,
            further modifications can be done to the figure before showing it.
    """
    try:
        from obspy.signal.trigger import triggerOnset
    except ImportError:
        msg = "Error during import from obspy.signal. Please make " + \
              "sure obspy.signal is installed properly."
        raise ImportError(msg)
    df = trace.stats.sampling_rate
    npts = trace.stats.npts
    t = np.arange(npts, dtype='float32') / df
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(t, trace.data, 'k')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(t, cft, 'k')
    onOff = np.array(triggerOnset(cft, thrOn, thrOff))
    i, j = ax1.get_ylim()
    try:
        ax1.vlines(onOff[:, 0] / df, i, j, color='r', lw=2, label="Trigger On")
        ax1.vlines(onOff[:, 1] / df, i, j, color='b', lw=2,
                   label="Trigger Off")
        ax1.legend()
    except IndexError:
        pass
    ax2.axhline(thrOn, color='red', lw=1, ls='--')
    ax2.axhline(thrOff, color='blue', lw=1, ls='--')
    ax2.set_xlabel("Time after %s [s]" % trace.stats.starttime.isoformat())
    fig.suptitle(trace.id)
    fig.canvas.draw()
    if show:
        plt.show()
