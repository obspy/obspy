# -*- coding: utf-8 -*-

import StringIO
import matplotlib.pyplot as plt


def plotWaveform(stream_object, outfile=None, format=None,
               size=(800, 200), starttime=False, endtime=False,
               dpi=100, color='red', bgcolor='white',
               transparent=False, shadows=False, minmaxlist=False):
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
        Defaults to 800x200 px.
    @param starttime: Starttime of the graph as a datetime object. If not
        set the graph will be plotted from the beginning.
        Defaults to False.
    @param endtime: Endtime of the graph as a datetime object. If not set
        the graph will be plotted until the end.
        Defaults to False.
    @param dpi: Dots per inch of the output file. This also affects the
        size of most elements in the graph (text, linewidth, ...).
        Defaults to 100.
    @param color: Color of the graph. If the supplied parameter is a
        2-tupel containing two html hex string colors a gradient between
        the two colors will be applied to the graph.
        Defaults to 'red'.
    @param bgcolor: Background color of the graph. If the supplied 
        parameter is a 2-tupel containing two html hex string colors a 
        gradient between the two colors will be applied to the background.
        Defaults to 'white'.
    @param transparent: Make all backgrounds transparent (True/False). This
        will overwrite the bgcolor param.
        Defaults to False.
    @param shadows: Adds a very basic drop shadow effect to the graph.
        Defaults to False.
    @param minmaxlist: A list containing minimum, maximum and timestamp
        values. If none is supplied it will be created automatically.
        Useful for caching.
        Defaults to False.
    """
    # Turn interactive mode off or otherwise only the first plot will be fast.
    plt.ioff()
    # Get a list with minimum and maximum values.
    if not minmaxlist:
        minmaxlist = _getMinMaxList(stream_object=stream_object,
                                                width=size[0],
                                                starttime=starttime,
                                                endtime=endtime)
    starttime = minmaxlist[0]
    endtime = minmaxlist[1]
    stepsize = (endtime - starttime) / size[0]
    minmaxlist = minmaxlist[2:]
    length = len(minmaxlist)
    # Setup figure and axes
    fig = plt.figure(num=None, figsize=(float(size[0]) / dpi,
                     float(size[1]) / dpi))
    ax = fig.add_subplot(111)
    # hide axes + ticks
    ax.axison = False
    # Make the graph fill the whole image.
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    # Determine range for the y axis. This will ensure that at least 98% of all
    # values are fully visible.
    minlist = [i[0] for i in minmaxlist]
    maxlist = [i[1] for i in minmaxlist]
    minlist.sort()
    maxlist.sort()
    miny = minlist[0]
    maxy = maxlist[-1]
    # Determines the 2 and 98 percent quantiles of min and max values.
    eighty_nine_miny = minlist[int(int(length * 0.02))]
    eighty_nine_maxy = maxlist[int(int(length * 0.98))]
    # Calculate 98%-range.
    yrange = eighty_nine_maxy - eighty_nine_miny
    # If the outer two percent are more than 10 times bigger discard them.
    if miny + 10 * yrange < maxy:
        miny = eighty_nine_miny
        maxy = eighty_nine_maxy
    else:
        yrange = maxy - miny
    miny = miny - (yrange * 0.1)
    maxy = maxy + (yrange * 0.1)
    # Set axes and disable ticks
    plt.ylim(miny, maxy)
    plt.xlim(starttime, endtime)
    plt.yticks([])
    plt.xticks([])
    # Overwrite the background gradient if transparent is set.
    if transparent:
        bgcolor = None
    # Draw gradient background if needed.
    if type(bgcolor) == type((1, 2)):
        for _i in xrange(size[0] + 1):
            #Convert hex values to integers
            r1 = int(bgcolor[0][1:3], 16)
            r2 = int(bgcolor[1][1:3], 16)
            delta_r = (float(r2) - float(r1)) / size[0]
            g1 = int(bgcolor[0][3:5], 16)
            g2 = int(bgcolor[1][3:5], 16)
            delta_g = (float(g2) - float(g1)) / size[0]
            b1 = int(bgcolor[0][5:], 16)
            b2 = int(bgcolor[1][5:], 16)
            delta_b = (float(b2) - float(b1)) / size[0]
            new_r = hex(int(r1 + delta_r * _i))[2:]
            new_g = hex(int(g1 + delta_g * _i))[2:]
            new_b = hex(int(b1 + delta_b * _i))[2:]
            if len(new_r) == 1:
                new_r = '0' + new_r
            if len(new_g) == 1:
                new_g = '0' + new_g
            if len(new_b) == 1:
                new_b = '0' + new_b
            #Create color string
            bglinecolor = '#' + new_r + new_g + new_b
            plt.axvline(x=starttime + _i * stepsize, color=bglinecolor)
        bgcolor = 'white'
    # Clone color for looping.
    loop_color = color
    # Draw horizontal lines.
    for _i in range(length):
        #Make gradient if color is a 2-tupel.
        if type(loop_color) == type((1, 2)):
            #Convert hex values to integers
            r1 = int(loop_color[0][1:3], 16)
            r2 = int(loop_color[1][1:3], 16)
            delta_r = (float(r2) - float(r1)) / length
            g1 = int(loop_color[0][3:5], 16)
            g2 = int(loop_color[1][3:5], 16)
            delta_g = (float(g2) - float(g1)) / length
            b1 = int(loop_color[0][5:], 16)
            b2 = int(loop_color[1][5:], 16)
            delta_b = (float(b2) - float(b1)) / length
            new_r = hex(int(r1 + delta_r * _i))[2:]
            new_g = hex(int(g1 + delta_g * _i))[2:]
            new_b = hex(int(b1 + delta_b * _i))[2:]
            if len(new_r) == 1:
                new_r = '0' + new_r
            if len(new_g) == 1:
                new_g = '0' + new_g
            if len(new_b) == 1:
                new_b = '0' + new_b
            #Create color string
            color = '#' + new_r + new_g + new_b
        #Calculate relative values needed for drawing the lines.
        yy = (float(minmaxlist[_i][0]) - miny) / (maxy - miny)
        xx = (float(minmaxlist[_i][1]) - miny) / (maxy - miny)
        #Draw shadows if desired.
        if shadows:
            plt.axvline(x=minmaxlist[_i][2] + stepsize, ymin=yy - 0.01,
                        ymax=xx - 0.01, color='k', alpha=0.4)
        #Draw actual data lines.
        plt.axvline(x=minmaxlist[_i][2], ymin=yy, ymax=xx,
                    color=color)
    #Save file.
    if outfile:
        #If format is set use it.
        if format:
            plt.savefig(outfile, dpi=dpi, transparent=transparent,
                facecolor=bgcolor, edgecolor=bgcolor, format=format)
        #Otherwise try to get the format from outfile or default to png.
        else:
            plt.savefig(outfile, dpi=dpi, transparent=transparent,
                facecolor=bgcolor, edgecolor=bgcolor)
    #Return an binary imagestring if outfile is not set but format is.
    if not outfile:
        if format:
            imgdata = StringIO.StringIO()
            plt.savefig(imgdata, dpi=dpi, transparent=transparent,
                    facecolor=bgcolor, edgecolor=bgcolor, format=format)
            imgdata.seek(0)
            return imgdata.read()
        else:
            plt.show()


def _getMinMaxList(stream_object, width, starttime=None,
                   endtime=None):
    """
    Creates a list with tuples containing a minimum value, a maximum value
    and a timestamp in microseconds.
    
    Only values between the start- and the endtime will be calculated. The
    first two items of the returned list are the actual start- and endtimes
    of the returned list. This is needed to cope with many different
    Mini-SEED files.
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
    # Sort traces according to starttime.
    traces = stream_object.traces
    traces.sort(key=lambda x:x.stats['starttime'])
    #Get start- and endtime and convert them to UNIX timestamp.
    if not starttime:
        starttime = traces[0].stats['starttime'].timestamp
    else:
        starttime = starttime.timestamp
    if not endtime:
        # The endtime of the last trace in the previously sorted list is
        # supposed to be the endtime of the plot.
        endtime = traces[-1].stats['endtime'].timestamp
    else:
        endtime = endtime.timestamp
    #Calculate time for one pixel.
    stepsize = (endtime - starttime) / width
    #First two items are start- and endtime.
    minmaxlist = [starttime, endtime]
    #While loop over the plotting duration.
    while starttime < endtime:
        pixel_endtime = starttime + stepsize
        maxlist = []
        minlist = []
        #Inner Loop over all traces.
        for _i in traces:
            a_stime = _i.stats['starttime'].timestamp
            a_etime = _i.stats['endtime'].timestamp
            npts = _i.stats['npts']
            #If the starttime is bigger than the endtime of the current
            #trace delete the item from the list.
            if starttime > a_etime:
                pass
            elif starttime < a_stime:
                #If starttime and endtime of the current pixel are too
                #small than leave the list.
                if pixel_endtime < a_stime:
                    #Leave the loop.
                    pass
                #Otherwise append the border to tempdatlist.
                else:
                    end = float((pixel_endtime - a_stime)) / \
                          (a_etime - a_stime) * npts
                    if end > a_etime:
                        end = a_etime
                    maxlist.append(_i.data[0 : int(end)].max())
                    minlist.append(_i.data[0 : int(end)].min())
            #Starttime is right in the current trace.
            else:
                #Endtime also is in the trace. Append to tempdatlist.
                if pixel_endtime < a_etime:
                    start = float((starttime - a_stime)) / (a_etime - a_stime) * \
                            npts
                    end = float((pixel_endtime - a_stime)) / \
                          (a_etime - a_stime) * npts
                    maxlist.append(_i.data[int(start) : int(end)].max())
                    minlist.append(_i.data[int(start) : int(end)].min())
                #Endtime is not in the trace. Append to tempdatlist.
                else:
                    start = float((starttime - a_stime)) / (a_etime - a_stime) * \
                            npts
                    maxlist.append(_i.data[int(start) : \
                                           npts].max())
                    minlist.append(_i.data[int(start) : \
                                           npts].min())
        #If empty list do nothing.
        if minlist == []:
        #if tempdatlist == array.array('l'):
            pass
        #If not empty append min, max and timestamp values to list.
        else:
            minmaxlist.append((min(minlist), max(maxlist),
                               starttime + 0.5 * stepsize))
        #New starttime for while loop.
        starttime = pixel_endtime
    return minmaxlist
