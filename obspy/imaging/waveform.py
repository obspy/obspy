# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: waveform.py
#  Purpose: Waveform plotting for obspy.Stream objects
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2012 Lion Krischer
# --------------------------------------------------------------------
"""
Waveform plotting for obspy.Stream objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import native_str

from obspy import UTCDateTime, Stream, Trace
from obspy.core.preview import mergePreviews
from obspy.core.util import createEmptyDataChunk, FlinnEngdahl, \
    locations2degrees
from obspy.core.util.decorator import deprecated_keywords

from copy import copy
from datetime import datetime
import io
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import scipy.signal as signal
import warnings


class WaveformPlotting(object):
    """
    Class that provides several solutions for plotting large and small waveform
    data sets.

    .. warning::

        This class should NOT be used directly, instead use the
        :meth:`~obspy.core.stream.Stream.plot` method of the
        ObsPy :class:`~obspy.core.stream.Stream` or
        :class:`~obspy.core.trace.Trace` objects.

    It uses matplotlib to plot the waveforms.
    """

    def __init__(self, **kwargs):
        """
        Checks some variables and maps the kwargs to class variables.
        """
        self.kwargs = kwargs
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
        self.stream = self.stream.copy()
        # Type of the plot.
        self.type = kwargs.get('type', 'normal')
        # Start and end times of the plots.
        self.starttime = kwargs.get('starttime', None)
        self.endtime = kwargs.get('endtime', None)
        self.fig_obj = kwargs.get('fig', None)
        # If no times are given take the min/max values from the stream object.
        if not self.starttime:
            self.starttime = min([trace.stats.starttime for trace in
                                  self.stream])
        if not self.endtime:
            self.endtime = max([trace.stats.endtime for trace in self.stream])
        # Map stream object and slice just in case.
        self.stream.trim(self.starttime, self.endtime)
        # Assigning values for type 'section'
        self.sect_offset_min = kwargs.get('offset_min', None)
        self.sect_offset_max = kwargs.get('offset_max', None)
        self.sect_dist_degree = kwargs.get('dist_degree', False)
        # TODO Event data from class Event()
        self.ev_coord = kwargs.get('ev_coord', None)
        self.alpha = kwargs.get('alpha', 0.5)
        self.sect_plot_dx = kwargs.get('plot_dx', None)
        self.sect_timedown = kwargs.get('time_down', False)
        self.sect_recordstart = kwargs.get('recordstart', None)
        self.sect_recordlength = kwargs.get('recordlength', None)
        self.sect_norm_method = kwargs.get('norm_method', 'trace')
        self.sect_user_scale = kwargs.get('scale', 1.0)
        self.sect_vred = kwargs.get('vred', None)
        # normalize times
        if self.type == 'relative':
            dt = kwargs.get('reftime', self.starttime)
            # fix plotting boundaries
            self.endtime = UTCDateTime(self.endtime - dt)
            self.starttime = UTCDateTime(self.starttime - dt)
            # fix stream times
            for tr in self.stream:
                tr.stats.starttime = UTCDateTime(tr.stats.starttime - dt)
        elif self.type == 'section':
            self.sect_reftime = kwargs.get('reftime', None)
        # Whether to use straight plotting or the fast minmax method. If not
        # set explicitly by the user "full" method will be used by default and
        # "fast" method will be used above some threshold of data points to
        # plot.
        self.plotting_method = kwargs.get('method', None)
        # Below that value the data points will be plotted normally. Above it
        # the data will be plotted using a different approach (details see
        # below). Can be overwritten by the above self.plotting_method kwarg.
        if self.type == 'section':
            # section may consists of hundreds of seismograms
            self.max_npts = 10000
        else:
            self.max_npts = 400000
        # If automerge is enabled, merge traces with the same id for the plot.
        self.automerge = kwargs.get('automerge', True)
        # If equal_scale is enabled, all plots are equally scaled.
        self.equal_scale = kwargs.get('equal_scale', True)
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
            elif self.type == 'section':
                self.width = 1000
                self.height = 600
            else:
                # One plot for each trace.
                if self.automerge:
                    count = self.__getMergablesIds()
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
            if isinstance(self.color, (str, native_str)):
                self.color = (self.color,)
            self.number_of_ticks = kwargs.get('number_of_ticks', None)
        else:
            self.color = kwargs.get('color', 'k')
            self.number_of_ticks = kwargs.get('number_of_ticks', 4)
        # Background, face and grid color.
        self.background_color = kwargs.get('bgcolor', 'w')
        self.face_color = kwargs.get('face_color', 'w')
        self.grid_color = kwargs.get('grid_color', 'black')
        self.grid_linewidth = kwargs.get('grid_linewidth', 0.5)
        self.grid_linestyle = kwargs.get('grid_linestyle', ':')
        # Transparency. Overwrites background and facecolor settings.
        self.transparent = kwargs.get('transparent', False)
        if self.transparent:
            self.background_color = None
        # Ticks.
        if self.type == 'relative':
            self.tick_format = kwargs.get('tick_format', '%.2f')
        else:
            self.tick_format = kwargs.get('tick_format', '%H:%M:%S')
        self.tick_rotation = kwargs.get('tick_rotation', 0)
        # Whether or not to save a file.
        self.outfile = kwargs.get('outfile')
        self.handle = kwargs.get('handle')
        # File format of the resulting file. Usually defaults to PNG but might
        # be dependent on your matplotlib backend.
        self.format = kwargs.get('format')
        self.show = kwargs.get('show', True)
        self.draw = kwargs.get('draw', True)
        self.block = kwargs.get('block', True)
        # plot parameters options
        self.x_labels_size = kwargs.get('x_labels_size', 8)
        self.y_labels_size = kwargs.get('y_labels_size', 8)
        self.title_size = kwargs.get('title_size', 10)
        self.linewidth = kwargs.get('linewidth', 1)
        self.linestyle = kwargs.get('linestyle', '-')
        self.subplots_adjust_left = kwargs.get('subplots_adjust_left', 0.12)
        self.subplots_adjust_right = kwargs.get('subplots_adjust_right', 0.88)
        self.subplots_adjust_top = kwargs.get('subplots_adjust_top', 0.95)
        self.subplots_adjust_bottom = kwargs.get('subplots_adjust_bottom', 0.1)
        self.right_vertical_labels = kwargs.get('right_vertical_labels', False)
        self.one_tick_per_line = kwargs.get('one_tick_per_line', False)
        self.show_y_UTC_label = kwargs.get('show_y_UTC_label', True)
        self.title = kwargs.get('title', self.stream[0].id)

    def __del__(self):
        """
        Destructor closes the figure instance if it has been created by the
        class.
        """
        if self.kwargs.get("fig", None) is None:
            plt.close()

    def __getMergeId(self, tr):
        tr_id = tr.id
        # don't merge normal traces with previews
        try:
            if tr.stats.preview:
                tr_id += 'preview'
        except AttributeError:
            pass
        # don't merge traces with different processing steps
        try:
            if tr.stats.processing:
                tr_id += str(tr.stats.processing)
        except AttributeError:
            pass
        return tr_id

    def __getMergablesIds(self):
        ids = []
        for tr in self.stream:
            tr_id = self.__getMergeId(tr)
            if tr_id not in ids:
                ids.append(tr_id)
        return ids

    def plotWaveform(self, *args, **kwargs):
        """
        Creates a graph of any given ObsPy Stream object. It either saves the
        image directly to the file system or returns a binary image string.

        For all color values you can use legit HTML names, HTML hex strings
        (e.g. '#eeefff') or you can pass an RGB tuple, where each of R, G, and
        B are in the range [0, 1]. You can also use single letters for basic
        built-in colors ('b' = blue, 'g' = green, 'r' = red, 'c' = cyan,
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
        elif self.type == 'section':
            self.plotSection(*args, **kwargs)
        else:
            self.plot(*args, **kwargs)
        # Adjust the subplot so there is always a fixed margin on every side
        if self.type != 'dayplot':
            fract_y = 60.0 / self.height
            fract_y2 = 40.0 / self.height
            fract_x = 80.0 / self.width
            self.fig.subplots_adjust(top=1.0 - fract_y, bottom=fract_y2,
                                     left=fract_x, right=1.0 - fract_x / 2)
        if self.draw:
            self.fig.canvas.draw()
        # The following just serves as a unified way of saving and displaying
        # the plots.
        if not self.transparent:
            extra_args = {'dpi': self.dpi,
                          'facecolor': self.face_color,
                          'edgecolor': self.face_color}
        else:
            extra_args = {'dpi': self.dpi,
                          'transparent': self.transparent,
                          'facecolor': 'k'}
        if self.outfile:
            # If format is set use it.
            if self.format:
                self.fig.savefig(self.outfile, format=self.format,
                                 **extra_args)
            # Otherwise use format from self.outfile or default to PNG.
            else:
                self.fig.savefig(self.outfile, **extra_args)
        else:
            # Return a binary image string if not self.outfile but self.format.
            if self.format:
                imgdata = io.BytesIO()
                self.fig.savefig(imgdata, format=self.format,
                                 **extra_args)
                imgdata.seek(0)
                return imgdata.read()
            elif self.handle:
                return self.fig
            else:
                if not self.fig_obj and self.show:
                    try:
                        plt.show(block=self.block)
                    except:
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
            # Sort order: id, starttime, endtime
            ids = self.__getMergablesIds()
            for id in ids:
                stream_new.append([])
                for tr in self.stream:
                    tr_id = self.__getMergeId(tr)
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
            if self.plotting_method is None:
                if ((self.endtime - self.starttime) * sampling_rate >
                        self.max_npts):
                    self.__plotMinMax(stream_new[_i], ax, *args, **kwargs)
                else:
                    self.__plotStraight(stream_new[_i], ax, *args, **kwargs)
            else:
                if self.plotting_method.lower() == 'full':
                    self.__plotStraight(stream_new[_i], ax, *args, **kwargs)
                elif self.plotting_method.lower() == 'fast':
                    self.__plotMinMax(stream_new[_i], ax, *args, **kwargs)
                else:
                    msg = ("Invalid plot method: '%s'") % self.plotting_method
                    raise ValueError(msg)
        # Set ticks.
        self.__plotSetXTicks()
        self.__plotSetYTicks()

    @deprecated_keywords({'swap_time_axis': None})
    def plotDay(self, *args, **kwargs):
        """
        Extend the seismogram.
        """
        # Merge and trim to pad.
        self.stream.merge()
        if len(self.stream) != 1:
            msg = "All traces need to be of the same id for a dayplot"
            raise ValueError(msg)
        self.stream.trim(self.starttime, self.endtime, pad=True)
        # Get minmax array.
        self.__dayplotGetMinMaxValues(self, *args, **kwargs)
        # Normalize array
        self.__dayplotNormalizeValues(self, *args, **kwargs)
        # Get timezone information. If none is given, use local time.
        self.time_offset = kwargs.get(
            'time_offset',
            round((UTCDateTime(datetime.now()) - UTCDateTime()) / 3600.0, 2))
        self.timezone = kwargs.get('timezone', 'local time')
        # Try to guess how many steps are needed to advance one full time unit.
        self.repeat = None
        intervals = self.extreme_values.shape[0]
        if self.interval < 60 and 60 % self.interval == 0:
            self.repeat = 60 // self.interval
        elif self.interval < 1800 and 3600 % self.interval == 0:
            self.repeat = 3600 // self.interval
        # Otherwise use a maximum value of 10.
        else:
            if intervals >= 10:
                self.repeat = 10
            else:
                self.repeat = intervals
        # Create axis to plot on.
        if self.background_color:
            ax = self.fig.add_subplot(1, 1, 1, axisbg=self.background_color)
        else:
            ax = self.fig.add_subplot(1, 1, 1)
        # Adjust the subplots
        self.fig.subplots_adjust(left=self.subplots_adjust_left,
                                 right=self.subplots_adjust_right,
                                 top=self.subplots_adjust_top,
                                 bottom=self.subplots_adjust_bottom)
        # Create x_value_array.
        x_values = np.repeat(np.arange(self.width), 2)
        intervals = self.extreme_values.shape[0]

        for _i in range(intervals):
            # Create offset array.
            y_values = np.ma.empty(self.width * 2)
            y_values.fill(intervals - (_i + 1))
            # Add min and max values.
            y_values[0::2] += self.extreme_values[_i, :, 0]
            y_values[1::2] += self.extreme_values[_i, :, 1]
            # Plot the values.
            ax.plot(x_values, y_values,
                    color=self.color[_i % len(self.color)],
                    linewidth=self.linewidth, linestyle=self.linestyle)
        # Plot the scale, if required.
        scale_unit = kwargs.get("data_unit", None)
        if scale_unit is not None:
            self._plotDayplotScale(unit=scale_unit)
        # Set ranges.
        ax.set_xlim(0, self.width - 1)
        ax.set_ylim(-0.3, intervals + 0.3)
        self.axis = [ax]
        # Set ticks.
        self.__dayplotSetYTicks(*args, **kwargs)
        self.__dayplotSetXTicks(*args, **kwargs)
        # Choose to show grid but only on the x axis.
        self.fig.axes[0].grid(color=self.grid_color,
                              linestyle=self.grid_linestyle,
                              linewidth=self.grid_linewidth)
        self.fig.axes[0].yaxis.grid(False)
        # Set the title of the plot.
        if self.title is not None:
            self.fig.suptitle(self.title, fontsize=self.title_size)
        # Now try to plot some events.
        events = kwargs.get("events", [])
        # Potentially download some events with the help of obspy.fdsn.
        if "min_magnitude" in events:
            try:
                from obspy.fdsn import Client
                c = Client("NERIES")
                events = c.get_events(starttime=self.starttime,
                                      endtime=self.endtime,
                                      minmagnitude=events["min_magnitude"])
            except Exception as e:
                events = None
                msg = "Could not download the events because of '%s: %s'." % \
                    (e.__class__.__name__, e.message)
                warnings.warn(msg)
        if events:
            for event in events:
                self._plotEvent(event)

    def _plotEvent(self, event):
        """
        Helper function to plot an event into the dayplot.
        """
        ax = self.fig.axes[0]
        seed_id = self.stream[0].id
        if hasattr(event, "preferred_origin"):
            # Get the time from the preferred origin, alternatively the first
            # origin.
            origin = event.preferred_origin()
            if origin is None:
                if event.origins:
                    origin = event.origins[0]
                else:
                    return
            time = origin.time

            # Do the same for the magnitude.
            mag = event.preferred_magnitude()
            if mag is None:
                if event.magnitudes:
                    mag = event.magnitudes[0]
            if mag is None:
                mag = ""
            else:
                mag = "%.1f %s" % (mag.mag, mag.magnitude_type)

            region = FlinnEngdahl().get_region(origin.longitude,
                                               origin.latitude)
            text = region
            if mag:
                text += ", %s" % mag
        else:
            time = event["time"]
            text = event["text"] if "text" in event else None

        # Nothing to do if the event is not on the plot.
        if time < self.starttime or time > self.endtime:
            return
        # Now find the position of the event in plot coordinates.

        def time2xy(time):
            frac = (time - self.starttime) / (self.endtime - self.starttime)
            int_frac = (self.interval) / (self.endtime - self.starttime)
            event_frac = frac / int_frac
            y_pos = self.extreme_values.shape[0] - int(event_frac) - 0.5
            x_pos = (event_frac - int(event_frac)) * self.width
            return x_pos, y_pos
        x_pos, y_pos = time2xy(time)

        if text:
            # Some logic to get a somewhat sane positioning of the annotation
            # box and the arrow..
            text_offset_x = 0.10 * self.width
            text_offset_y = 1.00
            # Relpos determines the connection of the arrow on the box in
            # relative coordinates.
            relpos = [0.0, 0.5]
            # Arc strength is the amount of bending of the arrow.
            arc_strength = 0.25
            if x_pos < (self.width / 2.0):
                text_offset_x_sign = 1.0
                ha = "left"
                # Arc sign determines the direction of bending.
                arc_sign = "+"
            else:
                text_offset_x_sign = -1.0
                ha = "right"
                relpos[0] = 1.0
                arc_sign = "-"
            if y_pos < (self.extreme_values.shape[0] / 2.0):
                text_offset_y_sign = 1.0
                va = "bottom"
            else:
                text_offset_y_sign = -1.0
                va = "top"
                if arc_sign == "-":
                    arc_sign = "+"
                else:
                    arc_sign = "-"

            # Draw the annotation including box.
            ax.annotate(text,
                        # The position of the event.
                        xy=(x_pos, y_pos),
                        # The position of the text, offset depending on the
                        # previously calculated variables.
                        xytext=(x_pos + text_offset_x_sign * text_offset_x,
                                y_pos + text_offset_y_sign * text_offset_y),
                        # Everything in data coordinates.
                        xycoords="data", textcoords="data",
                        # Set the text alignment.
                        ha=ha, va=va,
                        # Text box style.
                        bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                        # Arrow style
                        arrowprops=dict(
                            arrowstyle="-",
                            connectionstyle="arc3, rad=%s%.1f" % (
                                arc_sign, arc_strength),
                            relpos=relpos, shrinkB=7),
                        zorder=10)
        # Draw the actual point. Use a marker with a star shape.
        ax.plot(x_pos, y_pos, "*", color="yellow",
                markersize=12, linewidth=self.linewidth)

        for pick in getattr(event, 'picks', []):
            # check that network/station/location matches
            if pick.waveform_id.getSEEDString().split(".")[:-1] != \
               seed_id.split(".")[:-1]:
                continue
            x_pos, y_pos = time2xy(pick.time)
            ax.plot(x_pos, y_pos, "|", color="red",
                    markersize=50, markeredgewidth=self.linewidth * 4)

    def _plotDayplotScale(self, unit):
        """
        Plots the dayplot scale if requested.
        """
        left = self.width
        right = left + 5
        top = 2
        bottom = top - 1

        very_right = right + (right - left)
        middle = bottom + (top - bottom) / 2.0

        verts = [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
            (right, middle),
            (very_right, middle)
        ]

        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.MOVETO,
                 Path.LINETO
                 ]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, lw=1, facecolor="none")
        patch.set_clip_on(False)
        self.fig.axes[0].add_patch(patch)
        factor = self._normalization_factor
        # Manually determine the number of digits after decimal
        if factor >= 1000:
            fmt_string = "%.0f %s"
        elif factor >= 100:
            fmt_string = "%.1f %s"
        else:
            fmt_string = "%.2f %s"
        self.fig.axes[0].text(
            very_right + 3, middle,
            fmt_string % (self._normalization_factor, unit), ha="left",
            va="center", fontsize="small")

    def __plotStraight(self, trace, ax, *args, **kwargs):  # @UnusedVariable
        """
        Just plots the data samples in the self.stream. Useful for smaller
        datasets up to around 1000000 samples (depending on the machine on
        which it's being run).

        Slow and high memory consumption for large datasets.
        """
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
        # set label
        if hasattr(trace.stats, 'preview') and trace.stats.preview:
            tr_id = trace.id + ' [preview]'
        elif hasattr(trace, 'label'):
            tr_id = trace.label
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
        trace.data = np.require(trace.data, np.float64) * calib
        ax.plot(
            trace.data, color=self.color, linewidth=self.linewidth,
            linestyle=self.linestyle)
        # Set the x limit for the graph to also show the masked values at the
        # beginning/end.
        ax.set_xlim(0, len(trace.data) - 1)

    def __plotMinMax(self, trace, ax, *args, **kwargs):  # @UnusedVariable
        """
        Plots the data using a min/max approach that calculated the minimum and
        maximum values of each "pixel" and then plots only these values. Works
        much faster with large data sets.
        """
        # Some variables to help calculate the values.
        starttime = self.starttime.timestamp
        endtime = self.endtime.timestamp
        # The same trace will always have the same sampling_rate.
        sampling_rate = trace[0].stats.sampling_rate
        # The samples per resulting pixel. The end time is defined as the time
        # of the last sample.
        pixel_length = int(
            np.ceil(((endtime - starttime) * sampling_rate + 1) / self.width))
        # Loop over all the traces. Do not merge them as there are many samples
        # and therefore merging would be slow.
        for _i, tr in enumerate(trace):
            # Get the start of the next pixel in case the starttime of the
            # trace does not match the starttime of the plot.
            if tr.stats.starttime > self.starttime:
                offset = int(
                    np.ceil(((tr.stats.starttime - self.starttime) *
                             sampling_rate) / pixel_length))
            else:
                offset = 0
            # Figure out the number of pixels in the current trace.
            trace_length = len(tr.data) - offset
            pixel_count = int(trace_length // pixel_length)
            remaining_samples = int(trace_length % pixel_length)
            # Reference to new data array which does not copy data but can be
            # reshaped.
            data = tr.data[offset: offset + pixel_count * pixel_length]
            data = data.reshape(pixel_count, pixel_length)
            # Calculate extreme_values and put them into new array.
            extreme_values = np.ma.masked_all((self.width, 2), dtype=np.float)
            min = data.min(axis=1) * tr.stats.calib
            max = data.max(axis=1) * tr.stats.calib
            extreme_values[offset: offset + pixel_count, 0] = min
            extreme_values[offset: offset + pixel_count, 1] = max
            # First and last pixel need separate treatment.
            if offset:
                extreme_values[offset - 1, 0] = \
                    tr.data[:offset].min() * tr.stats.calib
                extreme_values[offset - 1, 1] = \
                    tr.data[:offset].max() * tr.stats.calib
            if remaining_samples:
                if offset + pixel_count == self.width:
                    index = self.width - 1
                else:
                    index = offset + pixel_count
                extreme_values[index, 0] = \
                    tr.data[-remaining_samples:].min() * tr.stats.calib
                extreme_values[index, 1] = \
                    tr.data[-remaining_samples:].max() * tr.stats.calib
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
        # set label
        if hasattr(trace[0], 'label'):
            tr_id = trace[0].label
        else:
            tr_id = trace[0].id
        # Write to self.stats.
        self.stats.append([tr_id, minmax.mean(),
                           minmax[:, 0].min(),
                           minmax[:, 1].max()])
        # Finally plot the data.
        x_values = np.repeat(np.arange(self.width), 2)
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
        for ax in self.axis:
            start, end = ax.get_xlim()
            # Set the location of the ticks.
            ax.set_xticks(np.linspace(start, end, self.number_of_ticks))
            # Figure out times.
            interval = float(self.endtime - self.starttime) / \
                (self.number_of_ticks - 1)
            # Set the actual labels.
            if self.type == 'relative':
                labels = [self.tick_format % (self.starttime
                                              + _i * interval).timestamp
                          for _i in range(self.number_of_ticks)]
            else:
                labels = [(self.starttime + _i *
                          interval).strftime(self.tick_format) for _i in
                          range(self.number_of_ticks)]

            ax.set_xticklabels(labels, fontsize='small',
                               rotation=self.tick_rotation)

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

        for _i, ax in enumerate(self.axis):
            mean = self.stats[_i][1]
            if not self.equal_scale:
                trace = self.stats[_i]
                max_distance = max(trace[1] - trace[2],
                                   trace[3] - trace[1]) * 1.1
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
            if abs(max(ticks) - min(ticks)) > 10:
                # integer numbers
                fmt = '%d'
                if abs(min(ticks)) > 10e6:
                    # but switch back to exponential for huge numbers
                    fmt = '%.2g'
            else:
                fmt = '%.2g'
            ax.set_yticklabels([fmt % t for t in ax.get_yticks()],
                               fontsize='small')
            # Set the title of each plot.
            ax.set_title(self.stats[_i][0], horizontalalignment='center',
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
        # Helper variables for easier access.
        trace = self.stream[0]
        trace_length = len(trace.data)

        # Samples per interval.
        spi = int(self.interval * trace.stats.sampling_rate)
        # Check the approximate number of samples per pixel and raise
        # error as fit.
        spp = float(spi) / self.width
        if spp < 1.0:
            msg = """
            Too few samples to use dayplot with the given arguments.
            Adjust your arguments or use a different plotting method.
            """
            msg = " ".join(msg.strip().split())
            raise ValueError(msg)
        # Number of intervals plotted.
        noi = float(trace_length) / spi
        inoi = int(round(noi))
        # Plot an extra interval if at least 2 percent of the last interval
        # will actually contain data. Do it this way to lessen floating point
        # inaccuracies.
        if abs(noi - inoi) > 2E-2:
            noi = inoi + 1
        else:
            noi = inoi

        # Adjust data. Fill with masked values in case it is necessary.
        number_of_samples = noi * spi
        delta = number_of_samples - trace_length
        if delta < 0:
            trace.data = trace.data[:number_of_samples]
        elif delta > 0:
            trace.data = np.ma.concatenate(
                [trace.data, createEmptyDataChunk(delta, trace.data.dtype)])

        # Create array for min/max values. Use masked arrays to handle gaps.
        extreme_values = np.ma.empty((noi, self.width, 2))
        trace.data.shape = (noi, spi)

        ispp = int(spp)
        fspp = spp % 1.0
        if fspp == 0.0:
            delta = None
        else:
            delta = spi - ispp * self.width

        # Loop over each interval to avoid larger errors towards the end.
        for _i in range(noi):
            if delta:
                cur_interval = trace.data[_i][:-delta]
                rest = trace.data[_i][-delta:]
            else:
                cur_interval = trace.data[_i]
            cur_interval.shape = (self.width, ispp)
            extreme_values[_i, :, 0] = cur_interval.min(axis=1)
            extreme_values[_i, :, 1] = cur_interval.max(axis=1)
            # Add the rest.
            if delta:
                extreme_values[_i, -1, 0] = min(extreme_values[_i, -1, 0],
                                                rest.min())
                extreme_values[_i, -1, 1] = max(extreme_values[_i, -1, 0],
                                                rest.max())
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
            max_values = self.extreme_values[:, :, 1].compressed()
            min_values = self.extreme_values[:, :, 0].compressed()
            # Remove masked values.
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
            max_val = min_val = abs(self.vertical_scaling_range) / 2.0

        # Normalization factor.
        self._normalization_factor = max(abs(max_val), abs(min_val)) * 2

        # Scale from 0 to 1.
        self.extreme_values = self.extreme_values / self._normalization_factor
        self.extreme_values += 0.5

    def __dayplotSetXTicks(self, *args, **kwargs):  # @UnusedVariable
        """
        Sets the xticks for the dayplot.
        """
        localization_dict = kwargs.get('localization_dict', {})
        localization_dict.setdefault('seconds', 'seconds')
        localization_dict.setdefault('minutes', 'minutes')
        localization_dict.setdefault('hours', 'hours')
        localization_dict.setdefault('time in', 'time in')
        max_value = self.width - 1
        # Check whether it is sec/mins/hours and convert to a universal unit.
        if self.interval < 240:
            time_type = localization_dict['seconds']
            time_value = self.interval
        elif self.interval < 24000:
            time_type = localization_dict['minutes']
            time_value = self.interval / 60
        else:
            time_type = localization_dict['hours']
            time_value = self.interval / 3600
        count = None
        # Hardcode some common values. The plus one is intentional. It had
        # hardly any performance impact and enhances readability.
        if self.interval == 15 * 60:
            count = 15 + 1
        elif self.interval == 20 * 60:
            count = 4 + 1
        elif self.interval == 30 * 60:
            count = 6 + 1
        elif self.interval == 60 * 60:
            count = 4 + 1
        elif self.interval == 90 * 60:
            count = 6 + 1
        elif self.interval == 120 * 60:
            count = 4 + 1
        elif self.interval == 180 * 60:
            count = 6 + 1
        elif self.interval == 240 * 60:
            count = 6 + 1
        elif self.interval == 300 * 60:
            count = 6 + 1
        elif self.interval == 360 * 60:
            count = 12 + 1
        elif self.interval == 720 * 60:
            count = 12 + 1
        # Otherwise run some kind of autodetection routine.
        if not count:
            # Up to 15 time units and if it's a full number, show every unit.
            if time_value <= 15 and time_value % 1 == 0:
                count = time_value
            # Otherwise determine whether they are divisible for numbers up to
            # 15. If a number is not divisible just show 10 units.
            else:
                count = 10
                for _i in range(15, 1, -1):
                    if time_value % _i == 0:
                        count = _i
                        break
            # Show at least 5 ticks.
            if count < 5:
                count = 5
        # Everything can be overwritten by user-specified number of ticks.
        if self.number_of_ticks:
            count = self.number_of_ticks
        # Calculate and set ticks.
        ticks = np.linspace(0.0, max_value, count)
        ticklabels = ['%i' % _i for _i in np.linspace(0.0, time_value, count)]
        self.axis[0].set_xticks(ticks)
        self.axis[0].set_xticklabels(ticklabels, rotation=self.tick_rotation,
                                     size=self.x_labels_size)
        self.axis[0].set_xlabel('%s %s' % (localization_dict['time in'],
                                           time_type), size=self.x_labels_size)

    def __dayplotSetYTicks(self, *args, **kwargs):  # @UnusedVariable
        """
        Sets the yticks for the dayplot.
        """
        intervals = self.extreme_values.shape[0]
        # Only display all ticks if there are five or less steps or if option
        # is set.
        if intervals <= 5 or self.one_tick_per_line:
            tick_steps = list(range(0, intervals))
            ticks = np.arange(intervals, 0, -1, dtype=np.float)
            ticks -= 0.5
        else:
            tick_steps = list(range(0, intervals, self.repeat))
            ticks = np.arange(intervals, 0, -1 * self.repeat, dtype=np.float)
            ticks -= 0.5

        # Complicated way to calculate the label of
        # the y-axis showing the second time zone.
        sign = '%+i' % self.time_offset
        sign = sign[0]
        label = "UTC (%s = UTC %s %02i:%02i)" % (
            self.timezone.strip(), sign, abs(self.time_offset),
            (self.time_offset % 1 * 60))
        ticklabels = [(self.starttime + _i *
                       self.interval).strftime(self.tick_format)
                      for _i in tick_steps]
        self.axis[0].set_yticks(ticks)
        self.axis[0].set_yticklabels(ticklabels, size=self.y_labels_size)
        # Show time zone label if requested
        if self.show_y_UTC_label:
            self.axis[0].set_ylabel(label)
        if self.right_vertical_labels:
            yrange = self.axis[0].get_ylim()
            self.twin_x = self.axis[0].twinx()
            self.twin_x.set_ylim(yrange)
            self.twin_x.set_yticks(ticks)
            y_ticklabels_twin = [(self.starttime + (_i + 1) *
                                  self.interval).strftime(self.tick_format)
                                 for _i in tick_steps]
            self.twin_x.set_yticklabels(y_ticklabels_twin,
                                        size=self.y_labels_size)

    def plotSection(self, *args, **kwargs):  # @UnusedVariable
        """
        Plots multiple waveforms as a record section on a single plot.
        """
        # Initialise data and plot
        self.__sectInitTraces()
        ax = self.__sectInitPlot()
        # Setting up line properties
        for line in ax.lines:
            line.set_alpha(self.alpha)
            line.set_linewidth(self.linewidth)
            line.set_color(self.color)
        # Setting up plot axes
        if self.sect_offset_min is not None:
            ax.set_xlim(left=self.__sectOffsetToFraction(self._offset_min))
        if self.sect_offset_max is not None:
            ax.set_xlim(right=self.__sectOffsetToFraction(self._offset_max))
        # Set up offset ticks
        tick_min, tick_max = \
            self.__sectFractionToOffset(np.array(ax.get_xlim()))
        if tick_min != 0.0 and self.sect_plot_dx is not None:
            tick_min += self.sect_plot_dx - (tick_min % self.sect_plot_dx)
        # Define tick vector for offset axis
        if self.sect_plot_dx is None:
            ticks = np.int_(np.linspace(tick_min, tick_max, 10))
        else:
            ticks = np.arange(tick_min, tick_max, self.sect_plot_dx)
            if len(ticks) > 100:
                self.fig.clf()
                msg = 'Too many ticks! Try changing plot_dx.'
                raise ValueError(msg)
        ax.set_xticks(self.__sectOffsetToFraction(ticks))
        # Setting up tick labels
        ax.set_ylabel('Time [s]')
        if not self.sect_dist_degree:
            ax.set_xlabel('Offset [km]')
            ax.set_xticklabels(ticks / 1e3)
        else:
            ax.set_xlabel(u'Offset [°]')
            ax.set_xticklabels(ticks)
        ax.minorticks_on()
        # Limit time axis
        ax.set_ylim([self._time_min, self._time_max])
        if self.sect_recordstart is not None:
            ax.set_ylim(bottom=self.sect_recordstart)
        if self.sect_recordlength is not None:
            ax.set_ylim(top=self.sect_recordlength + ax.get_ylim()[0])
        # Invert time axis if requested
        if self.sect_timedown:
            ax.invert_yaxis()
        # Draw grid on xaxis only
        ax.grid(
            color=self.grid_color,
            linestyle=self.grid_linestyle,
            linewidth=self.grid_linewidth)
        ax.xaxis.grid(False)

    def __sectInitTraces(self):
        """
        Arrange the trace data used for plotting.

        If necessary the data is resampled before
        being collected in a continuous list.
        """
        # Extract distances from st[].stats.distance
        # or from st.[].stats.coordinates.latitude...
        self._tr_offsets = np.empty(len(self.stream))
        if not self.sect_dist_degree:
            # Define offset in km from tr.stats.distance
            try:
                for _i, tr in enumerate(self.stream):
                    self._tr_offsets[_i] = tr.stats.distance
            except:
                msg = 'Define trace.stats.distance in meters to epicenter'
                raise ValueError(msg)
        else:
            # Define offset as degree from epicenter
            try:
                for _i, tr in enumerate(self.stream):
                    self._tr_offsets[_i] = locations2degrees(
                        tr.stats.coordinates.latitude,
                        tr.stats.coordinates.longitude,
                        self.ev_coord[0], self.ev_coord[1])
            except:
                msg = 'Define latitude/longitude in trace.stats.' + \
                    'coordinates and ev_coord. See documentation.'
                raise ValueError(msg)
        # Define minimum and maximum offsets
        if self.sect_offset_min is None:
            self._offset_min = self._tr_offsets.min()
        else:
            self._offset_min = self.sect_offset_min

        if self.sect_offset_max is None:
            self._offset_max = self._tr_offsets.max()
        else:
            self._offset_max = self.sect_offset_max
        # Reduce data to indexes within offset_min/max
        mask = ((self._tr_offsets >= self._offset_min) &
                (self._tr_offsets <= self._offset_max))
        self._tr_offsets = self._tr_offsets[mask]
        stream = [tr for m, tr in zip(mask, self.stream) if m]
        # Normalized offsets for plotting
        self._tr_offsets_norm = self._tr_offsets / self._tr_offsets.max()
        # Number of traces
        self._tr_num = len(self._tr_offsets)
        # Arranging trace data in single list
        self._tr_data = []
        # Maximum counts, npts, starttime and delta of each selected trace
        self._tr_starttimes = []
        self._tr_max_count = np.empty(self._tr_num)
        self._tr_npts = np.empty(self._tr_num)
        self._tr_delta = np.empty(self._tr_num)
        # TODO dynamic DATA_MAXLENGTH according to dpi
        for _i, tr in enumerate(stream):
            if len(tr.data) >= self.max_npts:
                tmp_data = signal.resample(tr.data, self.max_npts)
            else:
                tmp_data = tr.data
            # Initialising trace stats
            self._tr_data.append(tmp_data)
            self._tr_starttimes.append(tr.stats.starttime)
            self._tr_max_count[_i] = tmp_data.max()
            self._tr_npts[_i] = tmp_data.size
            self._tr_delta[_i] = (
                tr.stats.endtime -
                tr.stats.starttime) / self._tr_npts[_i]
        # Init time vectors
        self.__sectInitTime()

    def __sectScaleTraces(self, scale=None):
        """
        The traces have to be scaled to fit between 0-1., each trace
        gets 1./num_traces space. adjustable by scale=1.0.
        """
        if scale:
            self.sect_user_scale = scale
        self._sect_scale = self._tr_num * 1.5 * (1. / self.sect_user_scale)

    def __sectInitTime(self):
        """
        Define the time vector for each trace
        """
        reftime = self.sect_reftime or min(self._tr_starttimes)
        self._tr_times = []
        for _tr in range(self._tr_num):
            self._tr_times.append(
                (np.arange(self._tr_npts[_tr]) +
                 (self._tr_starttimes[_tr] - reftime)) * self._tr_delta[_tr])
            if self.sect_vred:
                self._tr_times[-1] -= self._tr_offsets[_tr] / self.sect_vred

        self._time_min = np.concatenate(self._tr_times).min()
        self._time_max = np.concatenate(self._tr_times).max()

    def __sectOffsetToFraction(self, offset):
        """
        Helper function to return offsets from fractions
        """
        return offset / self._tr_offsets.max()

    def __sectFractionToOffset(self, fraction):
        """
        Helper function to return fractions from offsets
        """
        return fraction * self._tr_offsets.max()

    def __sectInitPlot(self):
        """
        Function initialises plot all the illustration is done by
        self.plotSection()
        """
        ax = self.fig.gca()
        # Calculate normalizing factor
        self.__sectNormalizeTraces()
        # Calculate scaling factor
        self.__sectScaleTraces()
        # ax.plot() preferred over containers
        for _tr in range(self._tr_num):
            # Scale, normalize and shift traces by offset
            # for plotting
            ax.plot(self._tr_data[_tr] / self._tr_normfac[_tr]
                    * (1. / self._sect_scale)
                    + self._tr_offsets_norm[_tr],
                    self._tr_times[_tr])
        return ax

    def __sectNormalizeTraces(self):
        """
        This helper function normalizes the traces
        """
        self._tr_normfac = np.ones(self._tr_num)
        if self.sect_norm_method == 'trace':
            # Normalize against each traces' maximum
            for tr in range(self._tr_num):
                self._tr_normfac[tr] = np.abs(self._tr_data[tr]).max()
        elif self.sect_norm_method == 'stream':
            # Normalize the whole stream
            tr_max_count_glob = np.abs(self._tr_max_count).max()
            self._tr_normfac.fill(tr_max_count_glob)
        else:
            msg = 'Define a normalisation method. Valid normalisations' + \
                'are \'trace\', \'stream\'. See documentation.'
            raise ValueError(msg)

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
        x = self.__getX(10)
        y = self.__getY(15)
        # Default timestamp pattern
        pattern = '%Y-%m-%dT%H:%M:%SZ'

        if hasattr(self.stream, 'label'):
            suptitle = self.stream.label
        elif self.type == 'relative':
            # hide time information for relative plots
            return
        elif self.type == 'dayplot':
            suptitle = '%s %s' % (self.stream[0].id,
                                  self.starttime.strftime('%Y-%m-%d'))
            x = self.fig.subplotpars.left
        elif self.type == 'section':
            suptitle = 'Network: %s [%s] - (%i traces / %s)' % \
                (self.stream[-1].stats.network, self.stream[-1].stats.channel,
                 len(self.stream),
                 self.starttime.strftime(pattern))
        else:
            suptitle = '%s  -  %s' % (self.starttime.strftime(pattern),
                                      self.endtime.strftime(pattern))
        self.fig.suptitle(suptitle, x=x, y=y, fontsize='small',
                          horizontalalignment='left')

    def __getY(self, dy):
        return (self.height - dy) * 1.0 / self.height

    def __getX(self, dx):
        return dx * 1.0 / self.width
