# -*- coding: utf-8 -*-

from glob import iglob
from obspy.core.trace import Trace
from pkg_resources import iter_entry_points, load_entry_point
import copy
import math
import numpy as np
import os


def read(pathname, format=None, headonly=False, **kwargs):
    """
    Reads a file into a L{obspy.core.Stream} object.

    """
    # Reads files using the given wild card into a L{obspy.core.Stream} object.
    st = Stream()
    for file in iglob(pathname):
        st.extend(_read(file, format, headonly, **kwargs).traces)
    if len(st) == 0:
        raise Exception("Cannot open file/files", pathname)
    return st


def _read(filename, format=None, headonly=False, **kwargs):
    """
    Reads a file into a L{obspy.core.Stream} object.
    
    :param format: Format of the file to read. If it is None the format will be
        automatically detected. If you specify a format no further format 
        checking is done. To avoid problems please use the option only when 
        you are sure which format your file has. Defaults to None.
    """
    if not os.path.exists(filename):
        msg = "File not found '%s'" % (filename)
        raise IOError(msg)
    # Gets the available formats and the corresponding methods as entry points.
    formats_ep = dict([(ep.name, ep) \
                      for ep in iter_entry_points('obspy.plugin.waveform')])
    if not formats_ep:
        msg = "Your current ObsPy installation does not support any file " + \
              "reading formats. Please update or extend your ObsPy " + \
              "installation."
        raise Exception(msg)
    format_ep = None
    if not format:
        # detect format
        for ep in formats_ep.values():
            try:
                # search isFormat for given entry point
                isFormat = load_entry_point(ep.dist.key,
                                            'obspy.plugin.waveform.' + ep.name,
                                            'isFormat')
            except Exception, e:
                # verbose error handling/parsing
                print "WARNING: Cannot load module %s:" % ep.dist.key, e
                continue
            if isFormat(filename):
                format_ep = ep
                break
    else:
        # format given via argument
        format = format.upper()
        if format in formats_ep:
            format_ep = formats_ep[format]
    # file format should be known by now 
    try:
        # search readFormat for given entry point
        readFormat = load_entry_point(format_ep.dist.key,
                                      'obspy.plugin.waveform.' + \
                                      format_ep.name, 'readFormat')
    except:
        msg = "Format is not supported. Supported Formats: "
        raise TypeError(msg + ', '.join([ep.name for ep in formats_ep]))
    if headonly:
        stream = readFormat(filename, headonly=True, **kwargs)
    else:
        stream = readFormat(filename, **kwargs)
    return stream


class Stream(object):
    """
    ObsPy Stream class to collect L{Trace} objects.
    """
    def __init__(self, traces=None):
        self.traces = []
        if traces:
            self.traces.extend(traces)

    def __add__(self, stream):
        """
        Method to add two streams.
        
        It will create a new Stream object.
        """
        if not isinstance(stream, Stream):
            raise TypeError
        traces = copy.deepcopy(self.traces)
        traces.extend(stream.traces)
        return Stream(traces=traces)

    def __iadd__(self, stream):
        """
        Method to add two streams with self += other.
        
        It will extend the Stream object with the other one.
        """
        if not isinstance(stream, Stream):
            raise TypeError
        self.extend(stream.traces)
        return self

    def __len__(self):
        """
        Returns the number of Traces in the Stream object.
        """
        return len(self.traces)

    count = __len__

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

    def __eq__(self, other):
        """
        """
        if not isinstance(other, Stream):
            return False
        if self.traces != other.traces:
            return False
        return True


    def __getitem__(self, index):
        """ 
        __getitem__ method of obspy.Stream objects. 
          
        :return: Trace objects 
        """
        return self.traces[index]

    def __getslice__(self, i, j):
        """
        __getslice__ method of obspy.Stream objects.
        
        :return: Stream object
        """
        return Stream(traces=self.traces[i:j])

    def append(self, trace):
        """
        This method appends a single Trace object to the Stream object.
        
        :param trace: obspy.Trace object.
        """
        if isinstance(trace, Trace):
            self.traces.append(trace)
        else:
            msg = 'Append only supports a single Trace object as an argument.'
            raise TypeError(msg)

    def extend(self, trace_list):
        """
        This method will extend the traces attribute of the Stream object with
        a list of Trace objects.
        
        :param trace_list: list of obspy.Trace objects.
        """
        if isinstance(trace_list, list):
            for _i in trace_list:
                # Make sure each item in the list is a trace.
                if not isinstance(_i, Trace):
                    msg = 'Extend only accepts a list of Trace objects.'
                    raise TypeError(msg)
            self.traces.extend(trace_list)
        elif isinstance(trace_list, Stream):
            self.extend(trace_list.traces)
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
        
        :param min_gap: All gaps smaller than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.
        :param max_gap: All gaps larger than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.
        """
        gap_list = []
        for _i in xrange(len(self.traces) - 1):
            stats = self.traces[_i].stats
            stime = stats['endtime']
            etime = self.traces[_i + 1].stats['starttime']
            duration = etime.timestamp - stime.timestamp
            gap = etime.timestamp - stime.timestamp
            # Check that any overlap is not larger than the trace coverage
            if gap < 0:
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

    def insert(self, index, object):
        """
        Inserts either a single Trace object or a list of Trace objects before
        index.
        
        :param index: The Trace will be inserted before index.
        :param object: Single Trace object or list of Trace objects.
        """
        if isinstance(object, Trace):
            self.traces.insert(index, object)
        elif isinstance(object, list):
            # Make sure each item in the list is a trace.
            for _i in object:
                if not isinstance(_i, Trace):
                    msg = 'Trace object or a list of Trace objects expected!'
                    raise TypeError(msg)
            # Insert each item of the list.
            for _i in xrange(len(object)):
                self.traces.insert(index + _i, object[_i])
        elif isinstance(object, Stream):
            self.insert(index, object.traces)
        else:
            msg = 'Only accepts a Trace object or a list of Trace objects.'
            raise TypeError(msg)

    def plot(self, **kwargs):
        """
        Creates a graph of ObsPy Stream object. It either saves the image
        directly to the file system or returns an binary image string.
        
        For all color values you can use valid HTML names, HTML hex strings
        (e.g. '#eeefff') or you can pass an R , G , B tuple, where each of
        R , G , B are in the range [0,1]. You can also use single letters for
        basic builtin colors ('b' = blue, 'g' = green, 'r' = red, 'c' = cyan,
        'm' = magenta, 'y' = yellow, 'k' = black, 'w' = white) and gray shades
        can be given as a string encoding a float in the 0-1 range.
        
        :param outfile: Output file string. Also used to automatically
            determine the output format. Currently supported is emf, eps, pdf,
            png, ps, raw, rgba, svg and svgz output.
            Defaults to None.
        :param format: Format of the graph picture. If no format is given the
            outfile parameter will be used to try to automatically determine
            the output format. If no format is found it defaults to png output.
            If no outfile is specified but a format is than a binary
            imagestring will be returned.
            Defaults to None.
        :param size: Size tupel in pixel for the output file. This corresponds
            to the resolution of the graph for vector formats.
            Defaults to 800x200 px.
        :param starttime: Starttime of the graph as a datetime object. If not
            set the graph will be plotted from the beginning.
            Defaults to False.
        :param endtime: Endtime of the graph as a datetime object. If not set
            the graph will be plotted until the end.
            Defaults to False.
        :param dpi: Dots per inch of the output file. This also affects the
            size of most elements in the graph (text, linewidth, ...).
            Defaults to 100.
        :param color: Color of the graph. If the supplied parameter is a
            2-tupel containing two html hex string colors a gradient between
            the two colors will be applied to the graph.
            Defaults to 'red'.
        :param bgcolor: Background color of the graph. If the supplied 
            parameter is a 2-tupel containing two html hex string colors a 
            gradient between the two colors will be applied to the background.
            Defaults to 'white'.
        :param transparent: Make all backgrounds transparent (True/False). This
            will overwrite the bgcolor param.
            Defaults to False.
        :param shadows: Adds a very basic drop shadow effect to the graph.
            Defaults to False.
        :param minmaxlist: A list containing minimum, maximum and timestamp
            values. If none is supplied it will be created automatically.
            Useful for caching.
            Defaults to False.
        """
        try:
            from obspy.imaging import waveform
        except:
            msg = "Please install module obspy.imaging to be able to " + \
                  "plot ObsPy Stream objects."
            print msg
            raise
        waveform.plotWaveform(self, **kwargs)

    def pop(self, index= -1):
        """
        Removes the Trace object specified by index from the Stream object and
        returns it. If no index is given it will remove the last Trace.
        
        :param index: Index of the Trace object to be returned and removed.
        """
        temp_trace = self.traces[index]
        del(self.traces)[index]
        return temp_trace

    def printGaps(self, **kwargs):
        """
        Print gap/overlap list summary information of the Stream object.
        """
        result = self.getGaps(**kwargs)
        print "%-17s %-27s %-27s %-15s %-8s" % ('Source', 'Last Sample',
                                               'Next Sample', 'Gap', 'Samples')
        gaps = 0
        for r in result:
            if r[6] > 0:
                gaps += 1
            print "%-17s %-27s %-27s %-15.6f %-8d" % ('.'.join(r[0:4]),
                                                      r[4], r[5], r[6], r[7])
        overlaps = len(result) - gaps
        print "Total: %d gap(s) and %d overlap(s)" % (gaps, overlaps)

    def remove(self, index):
        """
        Removes the Trace object specified by index from the Stream object.
        
        :param index: Index of the Trace object to be removed
        """
        del(self.traces)[index]

    def reverse(self):
        """
        Reverses the Traces of the Stream object in place.
        """
        self.traces.reverse()

    def sort(self, keys=['network', 'station', 'location', 'channel',
                         'starttime', 'endtime']):
        """
        Method to sort the traces in the Stream object.
        
        The traces will be sorted according to the keys list. It will be sorted
        by the first item first, then by the second and so on. It will always
        be sorted from low to high and from A to Z.
        
        :param keys: List containing the values according to which the traces
             will be sorted. They will be sorted by the first item first and
             then by the second item and so on.
             Available items: 'network', 'station', 'channel', 'location',
             'starttime', 'endtime', 'sampling_rate', 'npts', 'dataquality' 
             Defaults to ['network', 'station', 'location', 'channel',
             'starttime', 'endtime'].
        """
        # Check the list and all items.
        msg = "keys must be a list of item strings. Available items to " + \
              "sort after: \n'network', 'station', 'channel', 'location', " + \
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
        # Loop over all keys in reversed order.
        for _i in keys[::-1]:
            self.traces.sort(key=lambda x:x.stats[_i], reverse=False)

    def write(self, filename, format, **kwargs):
        """
        Saves stream into a file.
        """
        format = format.upper()
        # Gets all available formats and the corresponding entry points.
        formats_ep = dict([(ep.name, ep) for ep in \
                           iter_entry_points('obspy.plugin.waveform')])
        try:
            # search writeFormat for given entry point
            ep = formats_ep[format]
            writeFormat = load_entry_point(ep.dist.key,
                                           'obspy.plugin.waveform.' + \
                                           ep.name, 'writeFormat')
        except:
            msg = "Format is not supported. Supported Formats: "
            raise TypeError(msg + ', '.join([ep.name for ep in formats_ep]))
        writeFormat(self, filename, **kwargs)

    def trim(self, starttime, endtime):
        """
        Cuts all traces of this Stream object to given start and end time.
        """
        for trace in self:
            trace.trim(starttime, endtime)

    def ltrim(self, starttime):
        """
        Cuts all traces of this Stream object to given start time.
        """
        for trace in self:
            trace.ltrim(starttime)

    def rtrim(self, endtime):
        """
        Cuts all traces of this Stream object to given end time.
        """
        for trace in self:
            trace.rtrim(endtime)

    def _verify(self):
        """
        Verifies all L{Trace} objects in this L{Stream}.
        """
        for trace in self:
            trace._verify()

    def old_merge(self):
        """
        Merges L{Trace} objects with same IDs.
        
        Gaps and overlaps are usually separated in distinct traces. This method
        tries to merge them and to create distinct traces within this L{Stream}
        object.  
        """
        # order matters!
        self.sort()
        traces_dict = {}
        # using pop() and try-except saves memory
        try:
            while True:
                trace = self.traces.pop()
                id = trace.getId()
                if id not in traces_dict:
                    traces_dict[id] = trace
                else:
                    traces_dict[id] = traces_dict[id] + trace
        except IndexError:
            pass
        self.traces = []
        # same here
        try:
            while True:
                id, trace = traces_dict.popitem()
                self.traces.append(trace)
        except KeyError:
            pass
        self.sort()


    def merge(self):
        """
        Merges L{Trace} objects with same IDs.
        
        Gaps and overlaps are usually separated in distinct traces. This method
        tries to merge them and to create distinct traces within this L{Stream}
        object.  
        """
        # order matters!
        self.sort(keys=['network', 'station', 'location', 'channel',
                         'starttime', 'endtime'])
        traces_dict = {}
        # using pop() and try-except saves memory
        try:
            while True:
                trace = self.traces.pop(0)
                id = trace.getId()
                if id not in traces_dict:
                    traces_dict[id] = [trace]
                else:
                    traces_dict[id].append(trace)
        except IndexError:
            pass
        self.traces = []
        # same here
        for id in traces_dict.keys():
            stats = copy.deepcopy(traces_dict[id][0].stats)
            sampling_rate = stats.sampling_rate
            stats.starttime = traces_dict[id][0].stats.starttime
            stats.endtime = traces_dict[id][-1].stats.endtime
            old_starttime = traces_dict[id][0].stats.starttime
            old_endtime = traces_dict[id][0].stats.endtime
            # This is the data list to which we extend
            cur_trace = [traces_dict[id].pop(0).data]
            for _i in xrange(len(traces_dict[id])):
                trace = traces_dict[id].pop(0)
                delta = int(round((trace.stats.starttime - \
                        old_endtime) * sampling_rate)) - 1
                # Overlap
                if delta <= 0:
                    # Left delta is the new starttime - old starttime. This
                    # always has to be greater or equal to zero.
                    left_delta = int(round((trace.stats.starttime - \
                                           old_starttime) * sampling_rate))
                    # The Endtime difference.
                    right_delta = int(round((trace.stats.endtime - \
                                           old_endtime) * sampling_rate))
                    # If right_delta is negative or zero throw the trace away.
                    if right_delta <= 0:
                        continue
                    # Update the old trace with the interpolation.
                    cur_trace[-1][left_delta:] = \
                        (cur_trace[-1][left_delta:] + trace[:-right_delta]) / 2
                    # Append the rest of the trace.
                    cur_trace.append(trace[-right_delta:])
                # Gap
                else:
                    nans = np.ma.masked_all(delta)
                    cur_trace.extend([nans, trace.data])
                old_endtime = trace.stats.endtime
                old_starttime = trace.stats.starttime
            if True in [np.ma.is_masked(_i) for _i in cur_trace]:
                data = np.ma.concatenate(cur_trace)
                stats.npts = data.size
                self.traces.append(Trace(data=data, header=stats))
            else:
                data = np.concatenate(cur_trace)
                stats.npts = data.size
                self.traces.append(Trace(data=data, header=stats))
