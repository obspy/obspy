# -*- coding: utf-8 -*-
"""
The ObsPy core classes.
"""
from obspy.core.util import Stats, getFormatsAndMethods
import copy
import math
import os


def read(filename, format=None):
    """
    This function will check a file's format and read it into a Stream object
    if possible.
    
    @param format: Format of the file to read. If it is none the format will be
        autodetected. If you specify a format no further format checking is
        done. To avoid problems please use the option only when you are sure
        which format your file has. Defaults to None.
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
    fileformat = []
    if not format:
        # Autodetect format
        for _i in formats:
            if _i[1](filename):
                fileformat = _i
                break
        if len(fileformat) == 0:
            msg = 'Format is not supported. Supported Formats: ' + \
                  ', '.join([_i[0] for _i in formats])
            raise TypeError(msg)
    else:
        try:
            format_index = [_i[0] for _i in formats].index(format.upper())
            fileformat = formats[format_index]
        except:
            msg = 'Format is not supported. Supported Formats: ' + \
                  ', '.join([_i[0] for _i in formats])
            raise TypeError(msg)
    temp_object = fileformat[2](filename)
    if isinstance(temp_object, Trace):
        return Stream(traces=[temp_object])
    return temp_object

def supportedFormats():
    """
    This function will return a list with all file formats supported by your
    ObSpy installation.
    """
    return [_i[0] for _i in getFormatsAndMethods()]

class Trace(object):
    """
    ObsPy Trace class.
    
    This class contains information about one trace.
    
    @type data: Numpy ndarray 
    @ivar data: Data samples 
    """
    def __init__(self, header=None, data=None):
        self.stats = Stats()
        self.data = None
        if header != None:
            for _i in header.keys():
                if type(header[_i]) == dict:
                    self.stats[_i] = Stats(dummy=False)
                    for _j in header[_i].keys():
                        self.stats[_i][_j] = header[_i][_j]
                else:
                    self.stats[_i] = header[_i]
        if data != None:
            self.data = data
    def __str__(self):
        out = "%(network)s.%(station)s.%(location)s.%(channel)s | " + \
              "%(starttime)s -- %(endtime)s | " + \
              "%(sampling_rate)f Hz, %(npts)d samples"
        return out % (self.stats)

class Stream(object):
    """
    ObsPy Stream class to collect Traces.
    
    """
    def __init__(self, traces=None):
        self.traces = []
        if traces:
            self.traces.extend(traces)

    def __add__(self, other):
        """
        Method to add two streams.
        
        It will make a deepcopy of both Stream's Traces and create a new
        Stream object.
        """
        traces = copy.deepcopy(self.traces)
        traces.extend(copy.deepcopy(other.traces))
        return Stream(traces=traces)

    def __iadd__(self, other):
        """
        Method to add two streams with self += other.
        
        It will make a deepcopy of the other Stream's Traces and extend the
        Stream object with them.
        """
        new_traces = copy.deepcopy(other.traces)
        self.extend(new_traces)
        return self

    def __len__(self):
        """
        Returns the number of Traces in the Stream object.
        """
        return len(self.traces)

    def __str__(self):
        """
        __str__ method of obspy.Stream objects.
        
        It will contain the number of Traces in the Stream and the return value
        of each Trace's __str__ method.
        """
        return_string = str(len(self.traces)) + ' Trace(s) in Stream:'
        for _i in self.traces:
            return_string = return_string + '\n' + str(_i)
        return return_string

    def __getitem__(self, index):
        """ 
        __getitem__ method of obspy.Stream objects. 
          
        @return: Trace objects 
        """
        return self.traces[index]

    def append(self, trace, reference=False):
        """
        This method appends a single Trace object to the Stream object.
        
        @param trace: obspy.Trace object.
        @param reference: If true it will pass the Trace by reference. If
            false it will make a deepcopy of the Trace. Defaults to False.
        """
        if isinstance(trace, Trace):
            if not reference:
                self.traces.append(copy.deepcopy(trace))
            else:
                self.traces.append(trace)
        else:
            msg = 'Append only supports a single Trace object as an argument.'
            raise TypeError(msg)

    def count(self):
        """
        Returns the number of Traces in the Stream object.
        """
        return len(self.traces)

    def extend(self, trace_list, reference=False):
        """
        This method will extend the traces attribut of the Stream object with
        a list of Trace objects.
        
        @param trace_list: list of obspy.Trace objects.
        @param reference: If true it will pass the Traces by reference. If
            false it will make a deepcopy of the Traces. Defaults to False.
        """
        if isinstance(trace_list, list):
            for _i in trace_list:
                # Make sure each item in the list is a trace.
                if not isinstance(_i, Trace):
                    msg = 'Extend only accepts a list of Trace objects.'
                    raise TypeError(msg)
            if not reference:
                self.traces.extend(copy.deepcopy(trace_list))
            else:
                self.traces.extend(trace_list)
        else:
            msg = 'Extend only supports a list of Trace objects as argument.'
            raise TypeError(msg)

    def getGaps(self, min_gap=None, max_gap=None):
        """
        Returns a list which contains information about all gaps/overlaps that
        result from the Traces in the Stream object.
        
        The returned list contains one item in the following form for each gap/
        overlap:
        [network, station, location, channel, starttime of the gap, endtime of
        the gap, duration of the gap, number of missing samples]
        
        Please be aware that no sorting and checking of stations, channels, ...
        is done. This method only compares the start- and endtimes of the
        Traces.
        
        @param min_gap: All gaps smaller than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.
        @param max_gap: All gaps larger than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.
        """
        gap_list = []
        for _i in range(len(self.traces) - 1):
            stats = self.traces[_i].stats
            stime = stats['endtime']
            etime = self.traces[_i + 1].stats['starttime']
            duration = etime.timestamp - stime.timestamp
            gap = etime.timestamp - stime.timestamp
             # Check that any overlap is not larger than the trace coverage
            if gap < 0:
                delta = 1 / float(self.traces[_i + 1].stats['sampling_rate'])
                temp = self.traces[_i + 1].stats['endtime'].timestamp - \
                       etime.timestamp
                if (gap * -1) > temp:
                    gap = -1 * temp
            # Check gap/overlap criteria
            if min_gap and gap < min_gap:
                continue
            if max_gap and gap > max_gap:
                continue
            # Number of missing samples
            nsamples = math.fabs(gap) * stats['sampling_rate']
            if gap > 0:
                nsamples -= 1
            else:
                nsamples += 1
            gap_list.append([stats['network'], stats['station'],
                            stats['location'], stats['channel'],
                            stime, etime, duration,
                            nsamples])
        return gap_list

    def insert(self, index, object, reference=False):
        """
        Inserts either a single Trace object or a list of Trace objects before
        index.
        
        @param index: The Trace will be inserted before index.
        @param object: Single Trace object or list of Trace objects.
        @param reference: If True it will pass the Traces by reference. If
            false it will make a deepcopy of the Traces. Defaults to False.
        """
        if isinstance(object, Trace):
            if not reference:
                self.traces.insert(index, copy.deepcopy(object))
            else:
                self.traces.insert(index, object)
        elif isinstance(object, list):
            # Make sure each item in the list is a trace.
            for _i in object:
                if not isinstance(_i, Trace):
                    msg = 'Only accepts a Trace object or a list of Trace ' + \
                           'objects.'
                    raise TypeError(msg)
            # Insert each item of the list.
            for _i in range(len(object)):
                if not reference:
                    self.traces.insert(index + _i, copy.deepcopy(object[_i]))
                else:
                    self.traces.insert(index + _i, object[_i])
        else:
            msg = 'Only accepts a Trace object or a list of Trace objects.'
            raise TypeError(msg)

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

    def pop(self, index= -1):
        """
        Removes the Trace object specified by index from the Stream object and
        returns it. If no index is given it will remove the last Trace.
        
        @param index: Index of the Trace object to be removed.
        """
        temp_trace = self.traces[index]
        del(self.traces)[index]
        return temp_trace

    def printGaps(self, **kwargs):
        """
        Print gap/overlap list summary information of the Stream object.
        """
        result = self.getGaps(**kwargs)
        print "%-17s %-26s %-26s %-15s %-8s" % ('Source', 'Last Sample',
                                               'Next Sample', 'Gap', 'Samples')
        for r in result:
            print "%-17s %-26s %-26s %-15s %-.8g" % ('_'.join(r[0:4]),
                                                    r[4].isoformat(),
                                                    r[5].isoformat(),
                                                    r[6], r[7])
        print "Total: %d gap(s) or overlap(s)" % len(result)

    def remove(self, index):
        """
        Removes the Trace object specified by index from the Stream object.
        
        @param index: Index of the Trace object to be removed
        """
        del(self.traces)[index]

    def reverse(self):
        """
        Reverses the Traces of the Stream object in place.
        """
        self.traces.reverse()

    def sort(self, keys=['network', 'station', 'channel', 'starttime']):
        """
        Method to sort the traces in the Stream object.
        
        The traces will be sorted according to the keys list. It will be sorted
        by the first item first, then by the second and so on. It will always
        be sorted from low to high and from A to Z.
        
        @param keys: List containing the values according to which the traces
             will be sorted. They will be sorted by the first item first and
             then by the second item and so on.
             Available items: 'network', 'station', 'channel', 'location',
             'starttime', 'endtime', 'sampling_rate', 'npts', 'dataquality' 
             Defaults to ['network', 'station', 'channel', 'starttime'].
        """
        # Check the list and all items.
        msg = "keys must be a list of item strings. Available items to sort " + \
              "after: \n'network', 'station', 'channel', 'location', " + \
              "'starttime', 'endtime', 'sampling_rate', 'npts', 'dataquality'"
        if not isinstance(keys, list):
            raise TypeError(msg)
        items = ['network', 'station', 'channel', 'location', 'starttime',
                 'endtime', 'sampling_rate', 'npts', 'dataquality']
        for _i in keys:
            try:
                items.index(_i)
            except:
                raise TypeError(msg)
        # Reverse list first.
        keys.reverse()
        # Loop over all items in keys.
        for _i in keys:
            self.traces.sort(key=lambda x:x.stats[_i])

    def write(self, filename, format, **kwargs):
        """
        """
        formats = getFormatsAndMethods()
        try:
            format_index = [_i[0] for _i in formats].index(format.upper())
        except:
            msg = 'Format is not supported. Supported Formats: ' + \
                  ', '.join([_i[0] for _i in formats])
            raise TypeError(msg)
        formats[format_index][3](self, filename, **kwargs)
