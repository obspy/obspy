# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: waveform.py
#  Purpose: Waveform plotting for obspy.Stream objects
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2010 Lion Krischer
#---------------------------------------------------------------------
"""
Waveform plotting for obspy.Stream objects.

GNU General Public License (GPL)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301, USA.
"""

from copy import deepcopy
from math import ceil
from numpy import isfinite
from numpy.ma import is_masked
from obspy.core import UTCDateTime, Stream, Trace
import StringIO
import matplotlib.pyplot as plt


def plotWaveform(stream_object, outfile=None, format=None,
               size=None, starttime=False, endtime=False,
               dpi=100, color='k', bgcolor='w', face_color='w',
               transparent=False, minmaxlist=False,
               number_of_ticks=5, tick_format='%H:%M:%S', tick_rotation=0):
    """
    Creates a graph of any given ObsPy Stream object. It either saves the image
    directly to the file system or returns an binary image string.
    
    For all color values you can use legit html names, html hex strings
    (e.g. '#eeefff') or you can pass an R , G , B tuple, where each of
    R , G , B are in the range [0,1]. You can also use single letters for
    basic builtin colors ('b' = blue, 'g' = green, 'r' = red, 'c' = cyan,
    'm' = magenta, 'y' = yellow, 'k' = black, 'w' = white) and gray shades
    can be given as a string encoding a float in the 0-1 range.
    
    @param stream_object: ObsPy Stream object.
    @param outfile: Output file string. Also used to automatically
        determine the output format. Currently supported is emf, eps, pdf,
        png, ps, raw, rgba, svg and svgz output.
        Defaults to None.
    @param format: Format of the graph picture. If no format is given the
        outfile parameter will be used to try to automatically determine
        the output format. If no format is found it defaults to png output.
        If no outfile is specified but a format is than a binary
        imagestring will be returned.
        Defaults to None.
    @param size: Size tupel in pixel for the output file. This corresponds
        to the resolution of the graph for vector formats.
        Defaults to a width of 800 pixel and a height of 250 pixel for each
        seperate plot.
    @param starttime: Starttime of the graph as a datetime object. If not
        set the graph will be plotted from the beginning.
        Defaults to False.
    @param endtime: Endtime of the graph as a datetime object. If not set
        the graph will be plotted until the end.
        Defaults to False.
    @param dpi: Dots per inch of the output file. This also affects the
        size of most elements in the graph (text, linewidth, ...).
        Defaults to 100.
    @param color: Color of the graph.
        Defaults to 'k' (black).
    @param bgcolor: Background color of the graph. No gradients are supported
        for the background.
        Defaults to 'w' (white).
    @param facecolor: Background color for the all background except behind
        the graphs.
        This will affect the default  matplotlib backend only when saving to a
        file. This behaviour might change depending on what backend you are
        using.
        Defaults to 'w' (white).
    @param transparent: Make all backgrounds transparent (True/False). This
        will overwrite the bgcolor and face_color attributes.
        Defaults to False.
    @param minmaxlist: A list containing minimum, maximum and timestamp
        values. If none is supplied it will be created automatically.
        Useful for caching.
        Defaults to False.
    @param number_of_ticks: Number of the ticks on the time scale to display.
        Defaults to 5.
    @param tick_format: Format of the time ticks according to strftime methods.
        Defaults to '%H:%M:%S'.
    @param tick_rotation: Number of degrees of the rotation of the ticks an the
        time scale. Ticks with big rotations might be cut off depending on the
        tick_format.
        Defaults to 0.
    """
    # check stream
    if isinstance(stream_object, Trace):
        stream_object = Stream([stream_object])
    elif not isinstance(stream_object, Stream):
        raise TypeError
    # Stream object should contain at least one Trace
    if len(stream_object) < 1:
        msg = "Empty stream object"
        raise IndexError(msg)
    # Default width.
    if not size:
        temp_width = 800
    else:
        temp_width = size[0]
    # Calculate the start-and the endtime and convert them to UNIX timestamps.
    if not starttime:
        starttime = min([trace.stats.starttime for \
                         trace in stream_object]).timestamp
    else:
        starttime = starttime.timestamp
    if not endtime:
        endtime = max([trace.stats.endtime for \
                       trace in stream_object]).timestamp
    else:
        endtime = endtime.timestamp
    # Turn interactive mode off or otherwise only the first plot will be fast.
    plt.ioff()
    # Get the maximum possible frequency.
    max_freq = max([trace.stats.sampling_rate for trace in stream_object])
    # Now decide whether to use a min-max approach and plot vertical lines
    # or just plot the data as it comes. It currently uses the min-max approach
    # if the biggest Trace has more then 500000 samples in the timeframe to
    # plot.
    if (endtime - starttime) * max_freq > 500000:
        _minMaxApproach(stream_object , size, starttime, endtime,
               dpi, color, bgcolor, minmaxlist,
               number_of_ticks, tick_format, tick_rotation, temp_width)
    else:
        _straightPlottingApproach(stream_object , size, starttime, endtime,
               dpi, color, bgcolor,
               number_of_ticks, tick_format, tick_rotation)
    #Save file.
    if outfile:
        #If format is set use it.
        if format:
            plt.savefig(outfile, dpi=dpi, transparent=transparent,
                facecolor=face_color, edgecolor=face_color, format=format)
        #Otherwise try to get the format from outfile or default to png.
        else:
            plt.savefig(outfile, dpi=dpi, transparent=transparent,
                facecolor=face_color, edgecolor=face_color)
    #Return an binary imagestring if outfile is not set but format is.
    if not outfile:
        if format:
            imgdata = StringIO.StringIO()
            plt.savefig(imgdata, dpi=dpi, transparent=transparent,
                    facecolor=face_color, edgecolor=face_color, format=format)
            imgdata.seek(0)
            return imgdata.read()
        else:
            plt.show()

def _minMaxApproach(stream_object , size, starttime, endtime, dpi, color,
                   bgcolor, minmaxlist,
                   number_of_ticks, tick_format, tick_rotation, temp_width):
    """
    Calculates a list containing minima and maxima values for each delta
    element and draws a horizontal line for each pair.
    
    Very useful for large datasets.
    """
    # Get a list with minimum and maximum values.
    if not minmaxlist:
        minmaxlist = _getMinMaxList(stream_object=stream_object,
                                                width=temp_width,
                                                starttime=starttime,
                                                endtime=endtime)
    # Number of plots.
    minmax_items = len(minmaxlist) - 2
    # Determine the size of the picture. Each plot will get 250 px in height.
    if not size:
        size = (800, minmax_items * 250)
    minmaxlist = minmaxlist[2:]
    # Get all stats attributes any make a new list.
    stats_list = []
    for _i in minmaxlist:
        stats_list.append(_i.pop())
    # Setup plot.
    _setupPlot(size, dpi, starttime, endtime)
    # Determine range for the y axis. This will ensure that at least 98% of all
    # values are fully visible.
    # This also assures that all subplots have the same scale on the y-axis.
    minlist = []
    maxlist = []
    for _i in minmaxlist:
        minlist.append([j[0] for j in _i])
        maxlist.append([j[1] for j in _i])
    list_length = len(minlist)
    for _i in xrange(list_length):
        # Sort the lists for each plot.
        minlist[_i].sort()
        maxlist[_i].sort()
        # Check if the ranges from the 3 and 97 % percent quantiles are much
        # smaller then the total range. If so throw away everything below
        # or above the quantiles.
        if (minlist[_i][int(0.97 * list_length)] - minlist[_i][int(0.03 * \
           list_length)]) > 25 * (minlist[_i][-1] - minlist[_i][0]):
            minlist[_i] = minlist[_i][int(0.03 * list_length)]
        else:
            minlist[_i] = minlist[_i][0]
        if (maxlist[_i][int(0.97 * list_length)] - maxlist[_i][int(0.03 * \
           list_length)]) > 25 * (maxlist[_i][-1] - maxlist[_i][0]):
            maxlist[_i] = maxlist[_i][int(0.03 * list_length)]
        else:
            maxlist[_i] = maxlist[_i][-1]
    # Now we need to calculate the average values of all Traces and create a
    # list with it
    average_list = []
    for _i in minmaxlist:
        average_list.append(sum([(_j[0] + _j[1]) / 2 for _j in _i]) / len(_i))
    # The next step is to find the maximum distance from the average values in
    # the yrange and store it.
    half_max_range = 0
    for _i in xrange(list_length):
        if average_list[_i] - minlist[_i] > half_max_range:
            half_max_range = average_list[_i] - minlist[_i]
        if maxlist[_i] - average_list[_i] > half_max_range:
            half_max_range = maxlist[_i] - average_list[_i]
    # Add ten percent on each side.
    half_max_range += half_max_range * 0.2
    # Create the location of the ticks.
    tick_location = []
    x_range = endtime - starttime
    if number_of_ticks > 1:
        for _i in xrange(number_of_ticks):
            tick_location.append(starttime + _i * x_range / \
                                                        (number_of_ticks - 1))
    elif number_of_ticks == 1:
        tick_location.append(starttime + x_range / 2)
    # Create tick names
    tick_names = [UTCDateTime(_i).strftime(tick_format) for _i in \
                  tick_location]
    if number_of_ticks > 1:
        # Adjust first tick to be visible.
        tick_location[0] = tick_location[0] + 0.5 * x_range / size[0]
    # Loop over all items in minmaxlist.
    for _j in range(minmax_items):
        cur_stats = stats_list[_j]
        # Calculate the number of samples in the current plot. To do this
        # loop over all traces in the stream_object, check if they are in the
        # current trace, calculate their samplecount in the timeframe to plot
        # and add it to the number of points.
        npts = 0
        for _i in xrange(len(stream_object)):
            # Check if it is actually in the plot.
            if stream_object[_i].stats.network != cur_stats.network or\
            stream_object[_i].stats.location != cur_stats.location or\
            stream_object[_i].stats.station != cur_stats.station or\
            stream_object[_i].stats.channel != cur_stats.channel:
                continue
            # A different approach is needed whether the data values are
            # masked arrays or not.
            if is_masked(stream_object[_i].data):
                npts += _getSamplesFromMaskedTrace(stream_object[_i].data,
                        starttime, endtime,
                        stream_object[_i].stats.starttime.timestamp,
                        stream_object[_i].stats.endtime.timestamp,
                        stream_object[_i].stats.sampling_rate)
            else:
                npts += _getSamplesFromTimes(stream_object[_i].data,
                        starttime, endtime,
                        stream_object[_i].stats.starttime.timestamp,
                        stream_object[_i].stats.endtime.timestamp,
                        stream_object[_i].stats.sampling_rate)
        # Make title
        title_text = '%s.%s.%s.%s, %s Hz, %s samples' % (cur_stats.network,
            cur_stats.station, cur_stats.location, cur_stats.channel,
            cur_stats.sampling_rate, npts)
        # Set axes and disable ticks on the y axis.
        cur_min_y = average_list[_j] - half_max_range
        cur_max_y = average_list[_j] + half_max_range
        bar_color = color
        yticks_location = [int(average_list[_j] - 0.8 * half_max_range),
                           int(average_list[_j]),
                           int(average_list[_j] + 0.8 * half_max_range)]
        # Setup up the figure for the current Trace.
        _setupFigure(_j, minmax_items, bgcolor, cur_min_y, cur_max_y,
                     title_text, tick_location, tick_names,
                     tick_rotation, yticks_location)
        # Draw horizontal lines.
        for _i in range(len(minmaxlist[_j])):
            bar_color = color
            #Calculate relative values needed for drawing the lines.
            yy = (float(minmaxlist[_j][_i][0]) - cur_min_y) / \
                                                    (cur_max_y - cur_min_y)
            xx = (float(minmaxlist[_j][_i][1]) - cur_min_y) / \
                                                    (cur_max_y - cur_min_y)
            # Set the x-range.
            plt.xlim(starttime, endtime)
            #Draw actual data lines.
            plt.axvline(x=minmaxlist[_j][_i][2], ymin=yy, ymax=xx,
                        color=bar_color)

def _straightPlottingApproach(stream_object , size, starttime, endtime,
               dpi, color, bgcolor,
               number_of_ticks, tick_format, tick_rotation):
    """
    Just plots the data samples in the stream_object. Useful for smaller
    datasets up to around 1000000 samples (depending on your computer).
    
    Slow and high memory consumption for large datasets.
    """
    # Copy the stream object and merge it for easier plotting. The copy is
    # needed to not alter the original stream object.
    stream_object = deepcopy(stream_object)
    stream_object.merge()
    stream_object.trim(UTCDateTime(starttime), UTCDateTime(endtime))
    number_of_plots = len(stream_object)
    # Set default size if no size was given.
    if not size:
        size = (800, number_of_plots * 250)
    # Setup plot.
    _setupPlot(size, dpi, starttime, endtime)
    # Get a list of max, min and average values for all traces.
    minlist = [_j.data.min() for _j in stream_object]
    maxlist = [_j.data.max() for _j in stream_object]
    average_list = [_j.data.mean() for _j in stream_object]
    # The next step is to find the maximum distance from the average values in
    # the yrange and store it.
    half_max_range = 0
    for _i in xrange(len(stream_object)):
        if average_list[_i] - minlist[_i] > half_max_range:
            half_max_range = average_list[_i] - minlist[_i]
        if maxlist[_i] - average_list[_i] > half_max_range:
            half_max_range = maxlist[_i] - average_list[_i]
    # Add ten percent on each side.
    half_max_range += half_max_range * 0.2
    # Create tick names. These are the same for each plot
    time_range = endtime - starttime
    tick_names = [UTCDateTime(starttime + time_range * _i / \
                              (number_of_ticks - 1)).strftime(tick_format) \
                              for _i in range(number_of_ticks)]
    for _j in xrange(number_of_plots):
        cur_stats = stream_object[_j].stats
        #All plots are centered on the mean value of the data values.
        cur_mean = stream_object[_j].data.mean()
        cur_min_y = cur_mean - half_max_range
        cur_max_y = cur_mean + half_max_range
        # Make title
        title_text = '%s.%s.%s.%s, %s Hz, %s samples' % (cur_stats.network,
            cur_stats.station, cur_stats.location, cur_stats.channel,
            cur_stats.sampling_rate, cur_stats.npts)
        # Get the yticks and the yrange.
        yticks_location = [int(cur_mean - 0.8 * half_max_range) , int(cur_mean),
                           int(cur_mean + 0.8 * half_max_range)]
        # Determine the xrange. This is necessary because trim and merge
        # on obspy.Trace object does not add NaNs at the beginning or the
        # end.
        cur_starttime = stream_object[_j].stats.starttime.timestamp
        cur_endtime = stream_object[_j].stats.endtime.timestamp
        if cur_starttime == starttime:
            cur_x_min = 0
        else:
            cur_x_min = int(-(cur_starttime - starttime) * \
                                    stream_object[_j].stats.sampling_rate)
        if cur_endtime == endtime:
            cur_x_max = stream_object[_j].data.size - 1
        else :
            cur_x_max = int(stream_object[_j].data.size - 1 + (endtime - \
                        cur_endtime) * stream_object[_j].stats.sampling_rate)
        # Create the location of the xticks.
        tick_location = []
        x_range = cur_x_max - cur_x_min
        if number_of_ticks > 1:
            for _i in xrange(number_of_ticks - 1):
                tick_location.append(cur_x_min + _i * x_range / \
                                                        (number_of_ticks - 1))
            tick_location.append(cur_x_max)
        elif number_of_ticks == 1:
            tick_location.append(cur_x_min + x_range / 2)
        # Setup up the figure for the current Trace.
        _setupFigure(_j, number_of_plots, bgcolor, cur_min_y, cur_max_y,
                     title_text, tick_location, tick_names,
                     tick_rotation, yticks_location)
        # Set the xrange.
        plt.xlim(cur_x_min, cur_x_max)
        # Plot the data, disable automatic scaling and set the color. Also
        # enable antialiasing. Otherwise the plots are just plain awful.
        plt.plot(stream_object[_j].data, scalex=False, scaley=False,
                 color=color, aa=True)

def _setupPlot(size, dpi, starttime, endtime):
    """
    The design and look of the whole plot to be produces.
    """
    # Setup figure and axes
    plt.figure(num=None, figsize=(float(size[0]) / dpi,
                     float(size[1]) / dpi))
    pattern = '%Y-%m-%d %H:%M:%S'
    suptitle = '%s  -  %s' % (UTCDateTime(starttime).strftime(pattern),
                              UTCDateTime(endtime).strftime(pattern))
    plt.suptitle(suptitle, x=0.02, y=0.96, fontsize='small',
                 horizontalalignment='left')

def _setupFigure(_j, minmax_items, bgcolor, cur_min_y, cur_max_y, title_text,
                 tick_location, tick_names, tick_rotation,
                 yticks_location):
    """
    The setup of the plot of each individual Trace.
    """
    plt.subplot(minmax_items, 1, _j + 1, axisbg=bgcolor)
    plt.title(title_text, horizontalalignment='left', fontsize='small',
              verticalalignment='center')
    plt.ylim(cur_min_y, cur_max_y)
    plt.yticks(yticks_location, fontsize='small')
    plt.xticks(tick_location, tick_names, rotation=tick_rotation,
               fontsize='small')

def _getMinMaxList(stream_object, width, starttime=None,
                   endtime=None):
    """
    Creates a list with tuples containing a minimum value, a maximum value
    and a timestamp in microseconds.
    
    Only values between the start- and the endtime will be calculated. The
    first two items of the returned list are the actual start- and endtimes
    of the returned list. This is needed to cope with all possible Stream
    types.
    The returned timestamps are the mean times of the minmax value pair.
    
    @requires: The Mini-SEED file has to contain only one trace. It may
        contain gaps and overlaps and it may be arranged in any order but
        the first and last records must be in chronological order as they
        are used to determine the start- and endtime.
    
    @param stream_object: ObsPy Stream object.
    @param width: Number of tuples in the list. Corresponds to the width
        in pixel of the graph.
    @param starttime: Starttime of the List/Graph as a Datetime object. If
        none is supplied the starttime of the file will be used.
        Defaults to None.
    @param endtime: Endtime of the List/Graph as a Datetime object. If none
        is supplied the endtime of the file will be used.
        Defaults to None.
    """
    plot_starttime = starttime
    plot_endtime = endtime
    # Sort stream
    stream_object.sort()
    all_traces = stream_object.traces
    # Create a sorted list of traces with one item for each identical trace
    # which contains all obspy.Trace objects that belong to the same trace.
    sorted_trace_list = []
    for _i in xrange(len(all_traces)):
        if len(sorted_trace_list) == 0:
            sorted_trace_list.append([all_traces[_i]])
        else:
            cur_trace_stats = all_traces[_i].stats
            last_trace_stats = sorted_trace_list[-1][-1].stats
            if cur_trace_stats.station == last_trace_stats.station and \
               cur_trace_stats.network == last_trace_stats.network and \
               cur_trace_stats.location == last_trace_stats.location and \
               cur_trace_stats.channel == last_trace_stats.channel:
                sorted_trace_list[-1].append(all_traces[_i])
            else:
                sorted_trace_list.append([all_traces[_i]])
    # Calculate time for one pixel.
    stepsize = (plot_endtime - plot_starttime) / width
    # First two items are start- and endtime.
    full_minmaxlist = [plot_starttime, plot_endtime]
    for traces in sorted_trace_list:
        # Create a list with the true sample count (including masked elements)
        # for faster access in the following loops.
        npts_list = [_i.data.size for _i in traces]
        # Reset loop times
        starttime = plot_starttime
        endtime = plot_endtime
        minmaxlist = []
        # While loop over the plotting duration.
        while starttime < endtime:
            pixel_endtime = starttime + stepsize
            maxlist = []
            minlist = []
            # Helper index to be able to index the npts_list.
            cur_trace_count = -1
            # Inner Loop over all traces.
            for _i in traces:
                cur_trace_count += 1
                a_stime = _i.stats['starttime'].timestamp
                a_etime = _i.stats['endtime'].timestamp
                npts = npts_list[cur_trace_count]
                # If the starttime is bigger than the endtime of the current
                # trace delete the item from the list.
                if starttime > a_etime:
                    pass
                elif starttime < a_stime:
                    # If starttime and endtime of the current pixel are too
                    # small than leave the list.
                    if pixel_endtime < a_stime:
                        #Leave the loop.
                        pass
                    # Otherwise append the border to tempdatlist.
                    else:
                        end = float((pixel_endtime - a_stime)) / \
                              (a_etime - a_stime) * npts
                        if end > a_etime:
                            end = a_etime
                        maxlist.append(_i.data[0 : int(end)].max())
                        minlist.append(_i.data[0 : int(end)].min())
                # Starttime is right in the current trace.
                else:
                    # Endtime also is in the trace. Append to tempdatlist.
                    if pixel_endtime < a_etime:
                        start = float((starttime - a_stime)) / (a_etime - \
                                                    a_stime) * npts
                        end = float((pixel_endtime - a_stime)) / \
                              (a_etime - a_stime) * npts
                        maxlist.append(_i.data[int(start) : int(end)].max())
                        minlist.append(_i.data[int(start) : int(end)].min())
                    # Endtime is not in the trace. Append to tempdatlist.
                    else:
                        start = float((starttime - a_stime)) / (a_etime - \
                                                    a_stime) * npts
                        maxlist.append(_i.data[int(start) : \
                                               npts].max())
                        minlist.append(_i.data[int(start) : \
                                               npts].min())
            # Remove all NaNs from min-and maxlist.
            minlist = [_i for _i in minlist if isfinite(_i)]
            maxlist = [_i for _i in maxlist if isfinite(_i)]
            # If empty list do nothing.
            if minlist == []:
                pass
            # If not empty append min, max and timestamp values to list.
            else:
                minmaxlist.append((min(minlist), max(maxlist),
                                   starttime + 0.5 * stepsize))
            # New starttime for while loop.
            starttime = pixel_endtime
        # Appends the stats dictionary to each different trace.
        minmaxlist.append(traces[0].stats)
        full_minmaxlist.append(minmaxlist)
    return full_minmaxlist

def _getSamplesFromTimes(data, starttime, endtime, cur_starttime, cur_endtime,
                         sampling_rate):
    """
    Helper method to calculate the number of samples with sampling rate and
    samples from cur_starttime till cur_starttime that are in the timeframe
    from starttime to endtime.
    """
    # If out of bounds return 0.
    if starttime > cur_endtime or endtime < cur_starttime:
        return 0
    # Fix times.
    if starttime < cur_starttime:
        starttime = cur_starttime
    if endtime > cur_endtime:
        endtime = cur_endtime
    return data[int((starttime - cur_starttime) * sampling_rate) : \
                int(ceil((endtime - cur_starttime) * sampling_rate) + 1)].size

def _getSamplesFromMaskedTrace(data, starttime, endtime, cur_starttime,
                               cur_endtime, sampling_rate):
    """
    Does the same as _getSamplesFromTimes but works for masked arrays.
    """
    # If out of bounds return 0.
    if starttime > cur_endtime or endtime < cur_starttime:
        return 0
    # Fix times.
    if starttime < cur_starttime:
        starttime = cur_starttime
    if endtime > cur_endtime:
        endtime = cur_endtime
    # Use the count method of masked arrays to not count the masked elements.
    return data[int((starttime - cur_starttime) * sampling_rate) : \
                int(ceil((endtime - cur_starttime) * sampling_rate) + 1)].count()
