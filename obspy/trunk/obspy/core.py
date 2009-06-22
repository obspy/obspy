# -*- coding: utf-8 -*-
"""
The ObsPy core classes.
"""
from obspy.util import Stats, getFormatsAndMethods
import copy
import os


def read(filename, format = None):
    """
    This function will check a file's format and read it into a Stream object
    if possible.
    """
    if not os.path.exists(filename):
        msg = "File not found '%s'" % (filename)
        raise IOError(msg)
    # Gets the available formats and the corresponding methods.
    formats = getFormatsAndMethods()
    if len(formats) == 0:
        msg = """Your current ObsPy installation does not support any file
                 reading formats. Please update or extend your ObsPy
                 installation"""
        raise Exception(msg)
    format = ''
    # Check the format of the file.
    for _i in formats:
        if _i[1](filename):
            format = _i
            break
    if len(format) == 0:
        msg = 'Format is not supported. Supported Formats: '+ \
              ', '.join([_i[0] for _i in formats])
        raise TypeError(msg)
    temp_object = format[2](filename)
    if isinstance(temp_object, Trace):
        return Stream(traces = [temp_object])
    return temp_object 

class Trace(object):
    """
    ObsPy Trace class.
    
    This class contains information about one trace.

    @type data: Numpy ndarray
    @ivar data: Data samples
    """
    def __init__(self, header = None, data = None):
        self.stats = Stats()
        self.data = None
        if header != None:
            for _i in header.keys():
                self.stats[_i] = header[_i]
        if data != None:
            self.data = data
    def __str__(self):
        return_string = self.stats['network'] + self.stats['station'] +\
            self.stats['channel'] + ' | ' +\
            str(self.stats['starttime'].strftime('%Y-%m-%d,%H:%M:%S'))+'--' +\
            str(self.stats['endtime'].strftime('%Y-%m-%d,%H:%M:%S')) + ' | ' +\
            str(self.stats['sampling_rate']) + ' Hz, ' +\
            str(self.stats['npts']) + ' samples'
        return return_string


class Stream(object):
    """
    ObsPy Stream class to collect Traces.
    
    """
    def __init__(self, traces = None):
        self.traces = []
        if traces:
            self.traces.extend(traces)
            
    def __add__(self, other):
        """
        Method to add two streams.
        
        It will just take deepcopies of both Stream's Traces and create a new
        Stream object.
        """
        traces = copy.deepcopy(self.traces)
        traces.extend(copy.deepcopy(other.traces))
        return Stream(traces = traces)
    
    def __str__(self):
        """
        __str__ method of obspy.Stream objects.
        
        It will contain the number of Traces in the Stream and the return value
        of each Trace's __str__ method.
        """
        return_string = str(len(self.traces))+ ' Trace(s) in Stream:'
        for _i in self.traces:
            return_string = return_string + '\n' + str(_i)
        return return_string
    
    def __getitem__(self, index):
        """
        __getitem__ method of obspy.Stream objects.
        
        @return: Trace objects
        """
        return self.traces[index]
    
    def plot(self, **kwargs):
        """
        Creates a graph of ObsPy Stream object. It either saves the image
        directly to the file system or returns an binary image string.
        
        For all color values you can use legit html names, html hex strings
        (e.g. '#eeefff') or you can pass an R , G , B tuple, where each of
        R , G , B are in the range [0,1]. You can also use single letters for
        basic builtin colors ('b' = blue, 'g' = green, 'r' = red, 'c' = cyan,
        'm' = magenta, 'y' = yellow, 'k' = black, 'w' = white) and gray shades
        can be given as a string encoding a float in the 0-1 range.
        
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
        try:
            from obspy.imaging import waveform
        except:
            msg = """Please install the obspy.imaging module to be able to plot
                  ObsPy Stream objects.\n"""
            print msg
            raise
        waveform.plotWaveform(self, **kwargs)
