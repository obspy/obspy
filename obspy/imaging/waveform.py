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
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import native_str

import io
import warnings
from copy import copy
from datetime import datetime
from dateutil.rrule import MINUTELY, SECONDLY

import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as patches
from matplotlib.cm import get_cmap
from matplotlib.dates import AutoDateLocator, date2num
from matplotlib.path import Path
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import scipy.signal as signal

from obspy import Stream, Trace, UTCDateTime
from obspy.core.util import create_empty_data_chunk
from obspy.core.util.decorator import deprecated
from obspy.geodetics import FlinnEngdahl, kilometer2degrees, locations2degrees
from obspy.imaging.util import (ObsPyAutoDateFormatter, _id_key, _timestring)


MINMAX_ZOOMLEVEL_WARNING_TEXT = "Warning: Zooming into MinMax Plot!"
SECONDS_PER_DAY = 3600.0 * 24.0
DATELOCATOR_WARNING_MSG = (
    "AutoDateLocator was unable to pick an appropriate interval for this date "
    "range. It may be necessary to add an interval value to the "
    "AutoDateLocator's intervald dictionary.")


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
        self.stream.trim(self.starttime, self.endtime)
        # Assigning values for type 'section'
        self.sect_offset_min = kwargs.get('offset_min', None)
        self.sect_offset_max = kwargs.get('offset_max', None)
        self.sect_dist_degree = kwargs.get('dist_degree', False)
        # TODO Event data from class Event()
        self.ev_coord = kwargs.get('ev_coord', None)
        self.alpha = kwargs.get('alpha', 0.5)
        self.sect_plot_dx = kwargs.get('plot_dx', None)
        if self.sect_plot_dx is not None and not self.sect_dist_degree:
            self.sect_plot_dx /= 1e3
        self.sect_timedown = kwargs.get('time_down', False)
        self.sect_recordstart = kwargs.get('recordstart', None)
        self.sect_recordlength = kwargs.get('recordlength', None)
        self.sect_norm_method = kwargs.get('norm_method', 'trace')
        self.sect_user_scale = kwargs.get('scale', 1.0)
        self.sect_vred = kwargs.get('vred', None)
        if self.sect_vred and self.sect_dist_degree:
            self.sect_vred = kilometer2degrees(self.sect_vred / 1e3)
        self.sect_orientation = kwargs.get('orientation', 'vertical')
        if self.type == 'relative':
            self.reftime = kwargs.get('reftime', self.starttime)
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
                    count = self.__get_mergable_ids()
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
        import matplotlib.pyplot as plt
        if self.kwargs.get('fig', None) is None and \
                not self.kwargs.get('handle'):
            plt.close()

    def __get_merge_id(self, tr):
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

    def __get_mergable_ids(self):
        ids = set()
        for tr in self.stream:
            ids.add(self.__get_merge_id(tr))
        return sorted(ids, key=_id_key)

    @deprecated(
        "'plotWaveform' has been renamed to "  # noqa
        "'plot_waveform'. Use that instead.")
    def plotWaveform(self, *args, **kwargs):
        '''
        DEPRECATED: 'plotWaveform' has been renamed to
        'plot_waveform'. Use that instead.
        '''
        return self.plot_waveform(*args, **kwargs)

    def plot_waveform(self, *args, **kwargs):
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
        import matplotlib.pyplot as plt
        # Setup the figure if not passed explicitly.
        if not self.fig_obj:
            self.__setup_figure()
        else:
            self.fig = self.fig_obj
        # Determine kind of plot and do the actual plotting.
        if self.type == 'dayplot':
            self.plot_day(*args, **kwargs)
        elif self.type == 'section':
            self.plot_section(*args, **kwargs)
        else:
            self.plot(*args, **kwargs)
        # Adjust the subplot so there is always a fixed margin on every side
        if self.type != 'dayplot':
            fract_y = 60.0 / self.height
            fract_y2 = 40.0 / self.height
            fract_x = 80.0 / self.width
            self.fig.subplots_adjust(top=1.0 - fract_y, bottom=fract_y2,
                                     left=fract_x, right=1.0 - fract_x / 2)
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("ignore", DATELOCATOR_WARNING_MSG,
                                    UserWarning, "matplotlib.dates")
            if self.draw:
                self.fig.canvas.draw()
            # The following just serves as a unified way of saving and
            # displaying the plots.
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
                # Return a binary image string if not self.outfile but
                # self.format.
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
            ids = self.__get_mergable_ids()
            for id in ids:
                stream_new.append([])
                for tr in self.stream:
                    tr_id = self.__get_merge_id(tr)
                    if tr_id == id:
                        # does not copy the elements of the data array
                        tr_ref = copy(tr)
                        if tr_ref.data.size:
                            stream_new[-1].append(tr_ref)
                # delete if empty list
                if not len(stream_new[-1]):
                    stream_new.pop()
                    continue
        # If everything is lost in the process raise an Exception.
        if not len(stream_new):
            raise Exception("Nothing to plot")
        # Create helper variable to track ids and min/max/mean values.
        self.ids = []
        # Loop over each Trace and call the appropriate plotting method.
        self.axis = []
        for _i, tr in enumerate(stream_new):
            # Each trace needs to have the same sampling rate.
            sampling_rates = {_tr.stats.sampling_rate for _tr in tr}
            if len(sampling_rates) > 1:
                msg = "All traces with the same id need to have the same " + \
                      "sampling rate."
                raise Exception(msg)
            sampling_rate = sampling_rates.pop()
            if _i == 0:
                sharex = None
            else:
                sharex = self.axis[0]
            ax = self.fig.add_subplot(len(stream_new), 1, _i + 1,
                                      axisbg=self.background_color,
                                      sharex=sharex)
            self.axis.append(ax)
            # XXX: Also enable the minmax plotting for previews.
            method_ = self.plotting_method
            if method_ is None:
                if ((self.endtime - self.starttime) * sampling_rate >
                        self.max_npts):
                    method_ = "fast"
                else:
                    method_ = "full"
            method_ = method_.lower()
            if method_ == 'full':
                self.__plot_straight(stream_new[_i], ax, *args, **kwargs)
            elif method_ == 'fast':
                self.__plot_min_max(stream_new[_i], ax, *args, **kwargs)
            else:
                msg = "Invalid plot method: '%s'" % method_
                raise ValueError(msg)
        # Set ticks.
        self.__plot_set_x_ticks()
        self.__plot_set_y_ticks()
        xmin = self._time_to_xvalue(self.starttime)
        xmax = self._time_to_xvalue(self.endtime)
        ax.set_xlim(xmin, xmax)
        self._draw_overlap_axvspan_legend()

    @deprecated(
        "'plotDay' has been renamed to "  # noqa
        "'plot_day'. Use that instead.")
    def plotDay(self, *args, **kwargs):
        '''
        DEPRECATED: 'plotDay' has been renamed to
        'plot_day'. Use that instead.
        '''
        return self.plot_day(*args, **kwargs)

    def plot_day(self, *args, **kwargs):
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
        self.__dayplot_get_min_max_values(self, *args, **kwargs)
        # Normalize array
        self.__dayplot_normalize_values(self, *args, **kwargs)
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
            self._plot_dayplot_scale(unit=scale_unit)
        # Set ranges.
        ax.set_xlim(0, self.width - 1)
        ax.set_ylim(-0.3, intervals + 0.3)
        self.axis = [ax]
        # Set ticks.
        self.__dayplot_set_y_ticks(*args, **kwargs)
        self.__dayplot_set_x_ticks(*args, **kwargs)
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
        # Potentially download some events with the help of obspy.clients.fdsn.
        if "min_magnitude" in events:
            try:
                from obspy.clients.fdsn import Client
                c = Client("EMSC")
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
                self._plot_event(event)

    def _plot_event(self, event):
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
            if pick.waveform_id.get_seed_string().split(".")[:-1] != \
               seed_id.split(".")[:-1]:
                continue
            x_pos, y_pos = time2xy(pick.time)
            ax.plot(x_pos, y_pos, "|", color="red",
                    markersize=50, markeredgewidth=self.linewidth * 4)

    def _plot_dayplot_scale(self, unit):
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

    def __plot_straight(self, trace, ax, *args, **kwargs):  # @UnusedVariable
        """
        Just plots the data samples in the self.stream. Useful for smaller
        datasets up to around 1000000 samples (depending on the machine on
        which it's being run).

        Slow and high memory consumption for large datasets.
        """
        # trace argument seems to actually be a list of traces..
        st = Stream(trace)
        self._draw_overlap_axvspans(st, ax)
        for trace in st:
            # Check if it is a preview file and adjust accordingly.
            # XXX: Will look weird if the preview file is too small.
            if trace.stats.get('preview'):
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
                trace.stats.delta = (
                    old_time_range / float(trace.stats.npts - 1))
            trace.data = np.require(trace.data, np.float64) * trace.stats.calib
            if self.type == 'relative':
                # use seconds of relative sample times and shift by trace's
                # start time, which was set relative to `reftime`.
                x_values = (
                    trace.times() + (trace.stats.starttime - self.reftime))
            else:
                # convert seconds of relative sample times to days and add
                # start time of trace.
                x_values = ((trace.times() / SECONDS_PER_DAY) +
                            date2num(trace.stats.starttime.datetime))
            ax.plot(x_values, trace.data, color=self.color,
                    linewidth=self.linewidth, linestyle=self.linestyle)
        # Write to self.ids
        trace = st[0]
        if trace.stats.get('preview'):
            tr_id = trace.id + ' [preview]'
        elif hasattr(trace, 'label'):
            tr_id = trace.label
        else:
            tr_id = trace.id
        self.ids.append(tr_id)

    def __plot_min_max(self, trace, ax, *args, **kwargs):  # @UnusedVariable
        """
        Plots the data using a min/max approach that calculated the minimum and
        maximum values of each "pixel" and then plots only these values. Works
        much faster with large data sets.
        """
        self._draw_overlap_axvspans(Stream(trace), ax)
        # Some variables to help calculate the values.
        starttime = self._time_to_xvalue(self.starttime)
        endtime = self._time_to_xvalue(self.endtime)
        # The same trace will always have the same sampling_rate.
        sampling_rate = trace[0].stats.sampling_rate
        # width of x axis in seconds
        x_width = endtime - starttime
        # normal plots have x-axis in days, so convert x_width to seconds
        if self.type != "relative":
            x_width = x_width * SECONDS_PER_DAY
        # number of samples that get represented by one min-max pair
        pixel_length = int(
            np.ceil((x_width * sampling_rate + 1) / self.width))
        # Loop over all the traces. Do not merge them as there are many samples
        # and therefore merging would be slow.
        for _i, tr in enumerate(trace):
            trace_length = len(tr.data)
            pixel_count = int(trace_length // pixel_length)
            remaining_samples = int(trace_length % pixel_length)
            remaining_seconds = remaining_samples / sampling_rate
            if self.type != "relative":
                remaining_seconds /= SECONDS_PER_DAY
            # Reference to new data array which does not copy data but can be
            # reshaped.
            if remaining_samples:
                data = tr.data[:-remaining_samples]
            else:
                data = tr.data
            data = data.reshape(pixel_count, pixel_length)
            min_ = data.min(axis=1) * tr.stats.calib
            max_ = data.max(axis=1) * tr.stats.calib
            # Calculate extreme_values and put them into new array.
            if remaining_samples:
                extreme_values = np.empty((pixel_count + 1, 2), dtype=np.float)
                extreme_values[:-1, 0] = min_
                extreme_values[:-1, 1] = max_
                extreme_values[-1, 0] = \
                    tr.data[-remaining_samples:].min() * tr.stats.calib
                extreme_values[-1, 1] = \
                    tr.data[-remaining_samples:].max() * tr.stats.calib
            else:
                extreme_values = np.empty((pixel_count, 2), dtype=np.float)
                extreme_values[:, 0] = min_
                extreme_values[:, 1] = max_
            # Finally plot the data.
            start = self._time_to_xvalue(tr.stats.starttime)
            end = self._time_to_xvalue(tr.stats.endtime)
            if remaining_samples:
                # the last minmax pair is inconsistent regarding x-spacing
                x_values = np.linspace(start, end - remaining_seconds,
                                       num=extreme_values.shape[0] - 1)
                x_values = np.concatenate([x_values, [end]])
            else:
                x_values = np.linspace(start, end, num=extreme_values.shape[0])
            x_values = np.repeat(x_values, 2)
            y_values = extreme_values.flatten()
            ax.plot(x_values, y_values, color=self.color)
        # remember xlim state and add callback to warn when zooming in
        self._initial_xrange = (self._time_to_xvalue(self.endtime) -
                                self._time_to_xvalue(self.starttime))
        self._minmax_plot_xrange_dangerous = False
        ax.callbacks.connect("xlim_changed", self._warn_on_xaxis_zoom)
        # set label, write to self.ids
        if hasattr(trace[0], 'label'):
            tr_id = trace[0].label
        else:
            tr_id = trace[0].id
        self.ids.append(tr_id)

    def __plot_set_x_ticks(self, *args, **kwargs):  # @UnusedVariable
        """
        Goes through all axes in pyplot and sets time ticks on the x axis.
        """
        import matplotlib.pyplot as plt
        self.fig.subplots_adjust(hspace=0)
        # Loop over all but last axes.
        for ax in self.axis[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        # set bottom most axes:
        ax = self.axis[-1]
        if self.type == "relative":
            locator = MaxNLocator(5)
        else:
            ax.xaxis_date()
            locator = AutoDateLocator(minticks=3, maxticks=6)
            locator.intervald[MINUTELY] = [1, 2, 5, 10, 15, 30]
            locator.intervald[SECONDLY] = [1, 2, 5, 10, 15, 30]
            ax.xaxis.set_major_formatter(ObsPyAutoDateFormatter(locator))
        ax.xaxis.set_major_locator(locator)
        plt.setp(ax.get_xticklabels(), fontsize='small',
                 rotation=self.tick_rotation)

    def __plot_set_y_ticks(self, *args, **kwargs):  # @UnusedVariable
        """
        """
        import matplotlib.pyplot as plt
        if self.equal_scale:
            ylims = np.vstack([ax.get_ylim() for ax in self.axis])
            yranges = np.diff(ylims).flatten()
            yrange_max = yranges.max()
            yrange_paddings = -yranges + yrange_max
            ylims[:, 0] -= yrange_paddings[:] / 2
            ylims[:, 1] += yrange_paddings[:] / 2
            for ax, ylims_ in zip(self.axis, ylims):
                ax.set_ylim(*ylims_)
        for _i, ax in enumerate(self.axis):
            # Set the title of each plot.
            ax.text(0.02, 0.95, self.ids[_i], transform=ax.transAxes,
                    fontdict=dict(fontsize="small", ha='left', va='top'),
                    bbox=dict(boxstyle="round", fc="w", alpha=0.8))
            plt.setp(ax.get_yticklabels(), fontsize='small')
            ax.yaxis.set_major_locator(MaxNLocator(7, prune="both"))
            ax.yaxis.set_major_formatter(ScalarFormatter())

    def __dayplot_get_min_max_values(self, *args, **kwargs):  # @UnusedVariable
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
                [trace.data, create_empty_data_chunk(delta, trace.data.dtype)])

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

    def __dayplot_normalize_values(self, *args, **kwargs):  # @UnusedVariable
        """
        Normalizes all values in the 3 dimensional array, so that the minimum
        value will be 0 and the maximum value will be 1.

        It will also convert all values to floats.
        """
        # Convert to native floats.
        self.extreme_values = self.extreme_values.astype(np.float) * \
            self.stream[0].stats.calib
        # Make sure that the mean value is at 0
        # raises underflow warning / error for numpy 1.9
        # even though mean is 0.09
        # self.extreme_values -= self.extreme_values.mean()
        self.extreme_values -= self.extreme_values.sum() / \
            self.extreme_values.size

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
        # raises underflow warning / error for numpy 1.9
        # even though normalization_factor is 2.5
        # self.extreme_values = self.extreme_values / \
        #     self._normalization_factor
        self.extreme_values = self.extreme_values * \
            (1. / self._normalization_factor)
        self.extreme_values += 0.5

    def __dayplot_set_x_ticks(self, *args, **kwargs):  # @UnusedVariable
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

    def __dayplot_set_y_ticks(self, *args, **kwargs):  # @UnusedVariable
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

    @deprecated(
        "'plotSection' has been renamed to "  # noqa
        "'plot_section'. Use that instead.")
    def plotSection(self, *args, **kwargs):
        '''
        DEPRECATED: 'plotSection' has been renamed to
        'plot_section'. Use that instead.
        '''
        return self.plot_section(*args, **kwargs)

    def plot_section(self, *args, **kwargs):  # @UnusedVariable
        """
        Plots multiple waveforms as a record section on a single plot.
        """
        # Initialise data and plot
        self.__sect_init_traces()
        ax, lines = self.__sect_init_plot()

        # Setting up line properties
        try:
            names_colors = self.sect_color.items()
        except AttributeError:
            for line in lines:
                line.set_alpha(self.alpha)
                line.set_linewidth(self.linewidth)
                line.set_color(self.sect_color)
        else:
            for line, tr in zip(lines, self.stream):
                line.set_alpha(self.alpha)
                line.set_linewidth(self.linewidth)
                color = self.sect_color[getattr(tr.stats, self.color)]
                line.set_color(color)

            legend_lines = []
            legend_labels = []
            for name, color in sorted(names_colors):
                legend_lines.append(
                    mlines.Line2D([], [], color=color, alpha=self.alpha,
                                  linewidth=self.linewidth))
                legend_labels.append(name)
            ax.legend(legend_lines, legend_labels)

        # Setting up plot axes
        if self.sect_offset_min is not None:
            self.set_offset_lim(left=self._offset_min)
        if self.sect_offset_max is not None:
            self.set_offset_lim(right=self._offset_max)
        # Set up offset ticks
        tick_min, tick_max = np.array(self.get_offset_lim())
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
        self.set_offset_ticks(ticks)
        # Setting up tick labels
        self.set_time_label('Time [s]')
        if not self.sect_dist_degree:
            self.set_offset_label('Offset [km]')
            self.set_offset_ticklabels(ticks)
        else:
            self.set_offset_label(u'Offset [°]')
            self.set_offset_ticklabels(ticks)
        ax.minorticks_on()
        # Limit time axis
        self.set_time_lim([self._time_min, self._time_max])
        if self.sect_recordstart is not None:
            self.set_time_lim(bottom=self.sect_recordstart)
        if self.sect_recordlength is not None:
            self.set_time_lim(
                top=self.sect_recordlength + self.get_time_lim()[0])
        # Invert time axis if requested
        if self.sect_orientation == 'vertical' and self.sect_timedown:
            ax.invert_yaxis()
        # Draw grid on xaxis only
        ax.grid(
            color=self.grid_color,
            linestyle=self.grid_linestyle,
            linewidth=self.grid_linewidth)
        self.offset_axis.grid(False)

    def __sect_init_traces(self):
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
                msg = 'trace.stats.distance undefined ' +\
                      '(set before plotting [in m], ' +\
                      'or use the ev_coords argument)'
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
        self.stream = [tr for m, tr in zip(mask, self.stream) if m]
        # Use km on distance axis, if not degrees
        if not self.sect_dist_degree:
            self._tr_offsets /= 1e3
            self._offset_min /= 1e3
            self._offset_max /= 1e3
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
        for _i, tr in enumerate(self.stream):
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
        self.__sect_init_time()
        # Init trace colors
        self.__sect_init_color()

    def __sect_scale_traces(self):
        """
        The traces have to be scaled to fit between 0-1., each trace
        gets distance-range/num_traces space. adjustable by scale=1.0.
        """
        self._sect_scale = (
            (self._offset_max - self._offset_min) * self.sect_user_scale /
            (self._tr_num * 1.5))

    def __sect_init_time(self):
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

    def __sect_init_color(self):
        """
        Define the color of each trace
        """
        if self.color == 'network':
            colors = {_tr.stats.network for _tr in self.stream}
        elif self.color == 'station':
            colors = {_tr.stats.station for _tr in self.stream}
        elif self.color == 'channel':
            colors = {_tr.stats.channel for _tr in self.stream}
        else:
            self.sect_color = self.color
            return

        cmap = get_cmap('Paired', lut=len(colors))
        self.sect_color = {k: cmap(i) for i, k in enumerate(sorted(colors))}

    def __sect_fraction_to_offset(self, fraction):
        """
        Helper function to return fractions from offsets
        """
        return fraction * self._tr_offsets.max()

    def __sect_init_plot(self):
        """
        Function initialises plot all the illustration is done by
        self.plot_section()
        """
        ax = self.fig.gca()
        # Calculate normalizing factor
        self.__sect_normalize_traces()
        # Calculate scaling factor
        self.__sect_scale_traces()
        lines = []
        # ax.plot() preferred over containers
        for _tr in range(self._tr_num):
            # Scale, normalize and shift traces by offset for plotting
            data = ((self._tr_data[_tr] / self._tr_normfac[_tr] *
                     self._sect_scale) +
                    self._tr_offsets[_tr])
            time = self._tr_times[_tr]
            if self.sect_orientation == 'vertical':
                lines += ax.plot(data, time)
            elif self.sect_orientation == 'horizontal':
                lines += ax.plot(time, data)
            else:
                raise NotImplementedError("sect_orientiation '%s' is not "
                                          "valid." % self.sect_orientation)

        # Set correct axes orientation
        if self.sect_orientation == 'vertical':
            self.set_offset_lim = ax.set_xlim
            self.get_offset_lim = ax.get_xlim
            self.set_offset_ticks = ax.set_xticks
            self.set_offset_label = ax.set_xlabel
            self.set_offset_ticklabels = ax.set_xticklabels
            self.offset_axis = ax.xaxis
            self.set_time_lim = ax.set_ylim
            self.get_time_lim = ax.get_ylim
            self.set_time_label = ax.set_ylabel
        elif self.sect_orientation == 'horizontal':
            def _set_xlim_from_ylim(*args, **kwargs):
                if 'bottom' in kwargs:
                    kwargs['left'] = kwargs.pop('bottom')
                if 'top' in kwargs:
                    kwargs['right'] = kwargs.pop('top')
                ax.set_xlim(*args, **kwargs)

            self.set_offset_lim = ax.set_ylim
            self.get_offset_lim = ax.get_ylim
            self.set_offset_ticks = ax.set_yticks
            self.set_offset_label = ax.set_ylabel
            self.set_offset_ticklabels = ax.set_yticklabels
            self.offset_axis = ax.yaxis
            self.set_time_lim = _set_xlim_from_ylim
            self.get_time_lim = ax.get_xlim
            self.set_time_label = ax.set_xlabel
        else:
            raise NotImplementedError("sect_orientiation '%s' is not "
                                      "valid." % self.sect_orientation)

        return ax, lines

    def __sect_normalize_traces(self):
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

    def __setup_figure(self):
        """
        The design and look of the whole plot to be produced.
        """
        import matplotlib.pyplot as plt
        # Setup figure and axes
        self.fig = plt.figure(num=None, dpi=self.dpi,
                              figsize=(float(self.width) / self.dpi,
                                       float(self.height) / self.dpi))
        # XXX: Figure out why this is needed sometimes.
        # Set size and dpi.
        self.fig.set_dpi(self.dpi)
        self.fig.set_figwidth(float(self.width) / self.dpi)
        self.fig.set_figheight(float(self.height) / self.dpi)

        if hasattr(self.stream, 'label'):
            suptitle = self.stream.label
        elif self.type == 'relative':
            suptitle = ("Time in seconds relative to %s" %
                        _timestring(self.reftime))
        elif self.type == 'dayplot':
            suptitle = '%s %s' % (self.stream[0].id,
                                  self.starttime.strftime('%Y-%m-%d'))
        elif self.type == 'section':
            suptitle = 'Network: %s [%s] - (%i traces / %s)' % \
                (self.stream[-1].stats.network, self.stream[-1].stats.channel,
                 len(self.stream), _timestring(self.starttime))
        else:
            suptitle = '%s  -  %s' % (_timestring(self.starttime),
                                      _timestring(self.endtime))
        # add suptitle
        y = (self.height - 15.0) / self.height
        self.fig.suptitle(suptitle, y=y, fontsize='small',
                          horizontalalignment='center')

    def _warn_on_xaxis_zoom(self, ax):
        """
        Method to be used as a callback on `method=fast`, "minmax"-type plots
        to warn the user when zooming into the plot.
        """
        xlim = ax.get_xlim()
        if xlim[1] - xlim[0] < 0.9 * self._initial_xrange:
            dangerous = True
        else:
            dangerous = False
        if dangerous and not self._minmax_plot_xrange_dangerous:
            self._add_zoomlevel_warning_text()
        elif self._minmax_plot_xrange_dangerous and not dangerous:
            self._remove_zoomlevel_warning_text()
        self._minmax_plot_xrange_dangerous = dangerous

    def _add_zoomlevel_warning_text(self):
        ax = self.fig.axes[0]
        self._minmax_warning_text = ax.text(
            0.95, 0.9, MINMAX_ZOOMLEVEL_WARNING_TEXT, color="r",
            ha="right", va="top", transform=ax.transAxes)

    def _remove_zoomlevel_warning_text(self):
        ax = self.fig.axes[0]
        if self._minmax_warning_text in ax.texts:
            ax.texts.remove(self._minmax_warning_text)
        self._minmax_warning_text = None

    def _draw_overlap_axvspans(self, st, ax):
        for _, _, _, _, start, end, delta, _ in st.get_gaps():
            if delta > 0:
                continue
            start = self._time_to_xvalue(start)
            end = self._time_to_xvalue(end)
            self._overlap_axvspan = \
                ax.axvspan(start, end, color="r", zorder=-10, alpha=0.5)

    def _draw_overlap_axvspan_legend(self):
        if hasattr(self, "_overlap_axvspan"):
            self.fig.axes[-1].legend(
                [self._overlap_axvspan], ["Overlaps"],
                loc="lower right", prop=dict(size="small"))

    def _time_to_xvalue(self, t):
            if self.type == 'relative':
                return t - self.reftime
            else:
                return date2num(t.datetime)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
