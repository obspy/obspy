# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Stream objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from glob import iglob
from obspy.core.trace import Trace
from obspy.core.util import NamedTemporaryFile
from pkg_resources import iter_entry_points, load_entry_point
import copy
import math
import numpy as np
import os
import urllib2


def read(pathname_or_url, format=None, headonly=False, **kwargs):
    """
    Read waveform files into an ObsPy Stream object.

    The `read` function opens either one or multiple files given via wildcards
    or a URL of a waveform file given in the *pathname_or_url* attribute. This
    function returns a ObsPy :class:`~obspy.core.stream.Stream` object.

    The format of the waveform file will be automatically detected if not
    given. Allowed formats depend on ObsPy packages installed. See the notes
    section below.

    Basic Usage
    -----------
    Examples files may be retrieved via http://examples.obspy.org.

    >>> from obspy.core import read # doctest: +SKIP
    >>> read("loc_RJOB20050831023349.z") # doctest: +SKIP
    <obspy.core.stream.Stream object at 0x101700150>

    Parameters
    ----------
    pathname_or_url : string
        String containing a file name or a URL. Wildcards are allowed for a
        file name.
    format : string, optional
        Format of the file to read. Commonly one of "GSE2", "MSEED", "SAC",
        "SEISAN", "WAV", "Q" or "SH_ASC". If it is None the format will be
        automatically detected which results in a slightly slower reading.
        If you specify a format no further format checking is done.
    headonly : bool, optional
        If set to True, read only the data header. This is most useful for
        scanning available meta information of huge data sets.

    Notes
    -----
    Additional ObsPy modules extend the functionality of the
    :func:`~obspy.core.stream.read` function. The following table summarizes
    all known formats currently available for ObsPy.

    Please refer to the linked function call of each module for any extra
    options available at the import stage.

    =======  ===================  ====================================
    Format   Required Module      Linked Function Call
    =======  ===================  ====================================
    MSEED    :mod:`obspy.mseed`   :func:`obspy.mseed.core.readMSEED`
    GSE2     :mod:`obspy.gse2`    :func:`obspy.gse2.core.readGSE2`
    SAC      :mod:`obspy.sac`     :func:`obspy.sac.core.readSAC`
    SEISAN   :mod:`obspy.seisan`  :func:`obspy.seisan.core.readSEISAN`
    WAV      :mod:`obspy.wav`     :func:`obspy.wav.core.readWAV`
    Q        :mod:`obspy.sh`      :func:`obspy.sh.core.readQ`
    SH_ASC   :mod:`obspy.sh`      :func:`obspy.sh.core.readASC`
    =======  ===================  ====================================

    Next to the `read` function the :meth:`~Stream.write` function is a method
    of the returned :class:`~obspy.core.stream.Stream` object.

    Examples
    --------
    Examples files may be retrieved via http://examples.obspy.org.

    (1) The following code uses wildcards, in this case it matches two files.
        Both files are then read into a single
        :class:`~obspy.core.stream.Stream` object.

        >>> from obspy.core import read  # doctest: +SKIP
        >>> st = read(("loc_R*.z"))  # doctest: +SKIP
        >>> print st  # doctest: +SKIP
        2 Trace(s) in Stream:
        .RJOB ..  Z | 2005-08-31T02:33:49.849998Z - 2005-08-31T02:34:49.8449...
        .RNON ..  Z | 2004-06-09T20:05:59.849998Z - 2004-06-09T20:06:59.8449...

    (2) Using the ``format`` parameter disables the autodetection and enforces
        reading a file in a given format.

        >>> from obspy.core import read  # doctest: +SKIP
        >>> read("loc_RJOB20050831023349.z", format="GSE2") # doctest: +SKIP
        <obspy.core.stream.Stream object at 0x101700150>

    (3) Reading via HTTP protocol.

        >>> from obspy.core import read
        >>> st = read("http://examples.obspy.org/loc_RJOB20050831023349.z")
        >>> print st  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        .RJOB ..  Z | 2005-08-31T02:33:49.849998Z - 2005-08-31T02:34:49.8449...
    """
    st = Stream()
    if "//" in pathname_or_url:
        # some URL
        fh = NamedTemporaryFile()
        fh.write(urllib2.urlopen(pathname_or_url).read())
        fh.seek(0)
        st.extend(_read(fh.name, format, headonly, **kwargs).traces)
        fh.close()
        os.remove(fh.name)
    else:
        pathname = pathname_or_url
        for file in iglob(pathname):
            st.extend(_read(file, format, headonly, **kwargs).traces)
        if len(st) == 0:
            raise Exception("Cannot open file/files", pathname)
    return st


def _read(filename, format=None, headonly=False, **kwargs):
    """
    Reads a single file into a :class:`~obspy.core.stream.Stream` object.
    """
    if not os.path.exists(filename):
        msg = "File not found '%s'" % (filename)
        raise IOError(msg)
    # Gets the available formats and the corresponding methods as entry points.
    formats_ep = {}
    for ep in iter_entry_points('obspy.plugin.waveform', None):
        _l = list(iter_entry_points('obspy.plugin.waveform.' + ep.name,
                                    'readFormat'))
        if _l:
            formats_ep[ep.name] = ep
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
        raise TypeError(msg + ', '.join(formats_ep.keys()))
    if headonly:
        stream = readFormat(filename, headonly=True, **kwargs)
    else:
        stream = readFormat(filename, **kwargs)
    # set a format keyword for each trace
    for trace in stream:
        trace.stats._format = format_ep.name
    return stream


class Stream(object):
    """
    List like object of multiple ObsPy trace objects.

    Parameters
    ----------
    traces : list of :class:`~obspy.core.trace.Trace`, optional
        Initial list of ObsPy Trace objects.

    Basic Usage
    -----------
    >>> trace1 = Trace()
    >>> trace2 = Trace()
    >>> stream = Stream(traces=[trace1, trace2])
    >>> print stream    #doctest: +ELLIPSIS
    2 Trace(s) in Stream:
    ...

    Supported Operations
    --------------------
    ``stream = streamA + streamB``
        Merges all traces within the two Stream objects ``streamA`` and
        ``streamB`` into the new Stream object ``stream``.
        See also: :meth:`Stream.__add__`.
    ``stream += streamA``
        Extends the Stream object ``stream`` with all traces from ``streamA``.
        See also: :meth:`Stream.__iadd__`.
    ``len(stream)``
        Returns the number of Traces in the Stream object ``stream``.
        See also: :meth:`Stream.__len__`.
    ``str(stream)``
        Contains the number of traces in the Stream object and returns the
        value of each Trace's __str__ method.
        See also: :meth:`Stream.__str__`.
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
        Compares two ObsPy Stream objects.
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
        Appends a single Trace object to the current Stream object.

        :param trace: obspy.Trace object.
        """
        if isinstance(trace, Trace):
            self.traces.append(trace)
        else:
            msg = 'Append only supports a single Trace object as an argument.'
            raise TypeError(msg)

    def extend(self, trace_list):
        """
        Extends the current Stream object with a list of Trace objects.

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
        Returns a list of all trace gaps/overlaps of the Stream object.

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
                             stats['location'], stats['channel'], stime, etime,
                             duration, nsamples])
        return gap_list

    def insert(self, position, object):
        """
        Inserts either a single Trace or a list of Traces before index.

        :param position: The Trace will be inserted at position.
        :param object: Single Trace object or list of Trace objects.
        """
        if isinstance(object, Trace):
            self.traces.insert(position, object)
        elif isinstance(object, list):
            # Make sure each item in the list is a trace.
            for _i in object:
                if not isinstance(_i, Trace):
                    msg = 'Trace object or a list of Trace objects expected!'
                    raise TypeError(msg)
            # Insert each item of the list.
            for _i in xrange(len(object)):
                self.traces.insert(position + _i, object[_i])
        elif isinstance(object, Stream):
            self.insert(position, object.traces)
        else:
            msg = 'Only accepts a Trace object or a list of Trace objects.'
            raise TypeError(msg)

    def plot(self, *args, **kwargs):
        """
        Creates a graph of the current ObsPy Stream object.

        It either saves the image directly to the file system or returns a
        binary image string.

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
            from obspy.imaging.waveform import WaveformPlotting
        except:
            msg = "Please install module obspy.imaging to be able to " + \
                  "plot ObsPy Stream objects."
            print msg
            raise
        waveform = WaveformPlotting(stream=self, *args, **kwargs)
        waveform.plotWaveform()

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
            self.traces.sort(key=lambda x: x.stats[_i], reverse=False)

    def write(self, filename, format="", **kwargs):
        """
        Saves stream into a file.

        Basic Usage
        -----------

        >>> from obspy.core import read # doctest: +SKIP
        >>> st = read("loc_RJOB20050831023349.z") # doctest: +SKIP
        >>> st.write("loc.ms", format="MSEED") # doctest: +SKIP

        Parameters
        ----------
        filename : string
            The name of the file to write.
        format : string
            The format to write must be specified. Depending on you obspy
            installation one of "MSEED", "GSE2", "SAC", "SEIAN", "WAV",
            "Q", "SH_ASC"

        Notes
        -----
        Additional ObsPy modules extend the parameters of the
        :func:`~obspy.core.stream.Stream.write` function. The following
        table summarizes all known formats currently available for ObsPy.

        Please refer to the linked function call of each module for any extra
        options available.

        =======  ===================  ====================================
        Format   Required Module      Linked Function Call
        =======  ===================  ====================================
        MSEED    :mod:`obspy.mseed`   :func:`obspy.mseed.core.writeMSEED`
        GSE2     :mod:`obspy.gse2`    :func:`obspy.gse2.core.writeGSE2`
        SAC      :mod:`obspy.sac`     :func:`obspy.sac.core.writeSAC`
        SEISAN   :mod:`obspy.seisan`  :func:`obspy.seisan.core.writeSEISAN`
        WAV      :mod:`obspy.wav`     :func:`obspy.wav.core.writeWAV`
        Q        :mod:`obspy.sh`      :func:`obspy.sh.core.writeQ`
        SH_ASC   :mod:`obspy.sh`      :func:`obspy.sh.core.writeASC`
        =======  ===================  ====================================
        """
        format = format.upper()
        # Gets all available formats and the corresponding entry points.
        formats_ep = {}
        for ep in iter_entry_points('obspy.plugin.waveform', None):
            _l = list(iter_entry_points('obspy.plugin.waveform.' + ep.name,
                                        'writeFormat'))
            if _l:
                formats_ep[ep.name] = ep
        if not format:
            msg = "Please provide a output format. Supported Formats: "
            print msg + ', '.join(formats_ep.keys())
            return
        try:
            # search writeFormat for given entry point
            ep = formats_ep[format]
            writeFormat = load_entry_point(ep.dist.key,
                                           'obspy.plugin.waveform.' + \
                                           ep.name, 'writeFormat')
        except:
            msg = "Format is not supported. Supported Formats: "
            raise TypeError(msg + ', '.join(formats_ep.keys()))
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

    def slice(self, starttime, endtime):
        """
        Returns new Stream object cut to the given start- and endtime.

        Does not copy the data but only passes a reference.
        """
        traces = []
        for trace in self:
            traces.append(trace.slice(starttime, endtime))
        return Stream(traces=traces)

    def verify(self):
        """
        Verifies all traces of current Stream against available meta data.

        Basic Usage
        -----------
        >>> tr = Trace(data=[1,2,3,4])
        >>> tr.stats.npts = 100
        >>> st = Stream([tr])
        >>> st.verify()  #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        Exception: ntps(100) differs from data size(4)
        """
        for trace in self:
            trace.verify()

    def merge(self):
        """
        Merges ObsPy Trace objects with same IDs.

        Gaps and overlaps are usually separated in distinct traces. This method
        tries to merge them and to create distinct traces within this 
        :class:`~Stream` object.
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
                    right_delta = -1 * int(round((trace.stats.endtime - \
                                                 old_endtime) * sampling_rate))
                    # If right_delta is negative or zero throw the trace away.
                    if right_delta > 0:
                        continue
                    # Update the old trace with the interpolation.
                    cur_trace[-1][left_delta:] = \
                        (cur_trace[-1][left_delta:] + trace[:right_delta]) / 2
                    # Append the rest of the trace.
                    cur_trace.append(trace[right_delta:])
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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
