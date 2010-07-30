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
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.trace import Trace
from obspy.core.util import NamedTemporaryFile, _getPlugins
from pkg_resources import load_entry_point
from StringIO import StringIO
import copy
import math
import numpy as np
import os
import urllib2
import warnings


def read(pathname_or_url=None, format=None, headonly=False,
         nearest_sample=True, **kwargs):
    """
    Read waveform files into an ObsPy Stream object.

    The `read` function opens either one or multiple files given via wildcards
    or a URL of a waveform file given in the *pathname_or_url* attribute. This
    function returns a ObsPy :class:`~obspy.core.stream.Stream` object.

    If no file location or URL is specified, a Stream object with an example
    seismogram is returned.

    The format of the waveform file will be automatically detected if not
    given. Allowed formats depend on ObsPy packages installed. See the notes
    section below.

    Basic Usage
    -----------
    An example can be created by omitting all arguments.
    Other example files may be read directly from http://examples.obspy.org.

    >>> from obspy.core import read
    >>> st = read()
    >>> print st
    3 Trace(s) in Stream:
    BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
    BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples

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
    starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        Specify the start time to read.
    endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        Specify the end time to read.
    nearest_sample bool, optional
        Only applied if starttime or endtime is given. Select nearest
        sample or the one containing the specified time. For more info, see
        :meth:`~obspy.core.trace.Trace.trim.`

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
        >>> st = read("loc_R*.z")  # doctest: +SKIP
        >>> print st  # doctest: +SKIP
        2 Trace(s) in Stream:
        .RJOB..Z | 2005-08-31T02:33:49.849998Z - 2005-08-31T02:34:49.8449...
        .RNON..Z | 2004-06-09T20:05:59.849998Z - 2004-06-09T20:06:59.8449...

    (2) Using the ``format`` parameter disables the autodetection and enforces
        reading a file in a given format.

        >>> from obspy.core import read  # doctest: +SKIP
        >>> read("loc_RJOB20050831023349.z", format="GSE2") # doctest: +SKIP
        <obspy.core.stream.Stream object at 0x101700150>

    (3) Reading via HTTP protocol.

        >>> from obspy.core import read
        >>> st = read("http://examples.obspy.org/loc_RJOB20050831023349.z") \
            # doctest: +SKIP
        >>> print st  # doctest: +ELLIPSIS +SKIP
        1 Trace(s) in Stream:
        .RJOB..Z | 2005-08-31T02:33:49.849998Z - 2005-08-31T02:34:49.8449...
    """
    # if no pathname or URL specified, make example stream
    if not pathname_or_url:
        return _read_example()

    st = Stream()
    if "://" in pathname_or_url:
        # some URL
        fh = NamedTemporaryFile()
        fh.write(urllib2.urlopen(pathname_or_url).read())
        fh.seek(0)
        st.extend(_read(fh.name, format, headonly, **kwargs).traces)
        fh.close()
        os.remove(fh.name)
    else:
        # file name
        pathname = pathname_or_url
        for file in iglob(pathname):
            st.extend(_read(file, format, headonly, **kwargs).traces)
        if len(st) == 0:
            raise Exception("Cannot open file/files", pathname)
    # Trim if times are given.
    starttime = kwargs.get('starttime')
    endtime = kwargs.get('endtime')
    if headonly and (starttime or endtime):
        msg = "headonly cannot be combined with starttime or endtime"
        raise Exception(msg)
    if starttime:
        st.ltrim(starttime, nearest_sample=nearest_sample)
    if endtime:
        st.rtrim(endtime, nearest_sample=nearest_sample)
    return st


def _read(filename, format=None, headonly=False, **kwargs):
    """
    Reads a single file into a ObsPy Stream object.
    """
    if not os.path.exists(filename):
        msg = "File not found '%s'" % (filename)
        raise IOError(msg)
    # Gets the available formats and the corresponding methods as entry points.
    formats_ep = _getPlugins('obspy.plugin.waveform', 'readFormat')
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

def _read_example():
    """
    Robust method to create example stream.
    Data arrays are stored in numpy's npz format. The stats information we
    fill in here directly.

    PAZ of the used instrument:
    paz = {'gain': 60077000.0, 
           'poles': [(-0.037004000000000002+0.037016j), 
                     (-0.037004000000000002-0.037016j), 
                     (-251.33000000000001+0j), 
                     (-131.03999999999999-467.29000000000002j), 
                     (-131.03999999999999+467.29000000000002j)], 
           'sensitivity': 2516778400.0, 
           'zeros': [0j, 0j]}} 
    """
    #msg = "No pathname or URL specified. Ignoring other arguments and " + \
    #      "loading example data."
    #warnings.warn(msg)
    path = os.path.dirname(__file__)
    path = os.path.join(path, "data", "example.npz")
    data = np.load(path)
    header = {'network': "BW",
              'station': "RJOB",
              'location': "",
              'npts': 3000,
              'starttime': UTCDateTime(2009, 8, 24, 0, 20, 3),
              'sampling_rate': 100.0,
              'processing': ["filter:lowpass:{'freq': 40.0}",
                             "downsample:integerDecimation:2"],
              'calib': 1.0}
    st = Stream()
    for channel in ["EHZ", "EHN", "EHE"]:
        header['channel'] = channel
        st.append(Trace(data=data[channel], header=header))
    return st

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
        return self.__class__(traces=traces)

    def __iadd__(self, stream):
        """
        Method to add two streams with self += other.

        It will extend the Stream object with the other one.
        """
        if not isinstance(stream, Stream):
            raise TypeError
        self.extend(stream.traces)
        return self

    def __iter__(self):
        """
        Return a robust iterator for stream.traces.

        Doing this it is safe to remove traces from streams inside of
        for-loops using stream's remove() method. Actually this creates a new
        iterator every time a trace is removed inside the for-loop.

        >>> from obspy.core import read
        >>> st = Stream()
        >>> for component in ["1", "Z", "2", "3", "Z", "N", "E", "4", "5"]:
        ...     channel = "EH" + component
        ...     tr = Trace(header={'channel': channel})
        ...     st.append(tr)
        >>> print st
        9 Trace(s) in Stream:
        ...EH1 | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.000000Z | 1.0 Hz, 0 samples
        ...EHZ | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.000000Z | 1.0 Hz, 0 samples
        ...EH2 | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.000000Z | 1.0 Hz, 0 samples
        ...EH3 | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.000000Z | 1.0 Hz, 0 samples
        ...EHZ | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.000000Z | 1.0 Hz, 0 samples
        ...EHN | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.000000Z | 1.0 Hz, 0 samples
        ...EHE | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.000000Z | 1.0 Hz, 0 samples
        ...EH4 | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.000000Z | 1.0 Hz, 0 samples
        ...EH5 | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.000000Z | 1.0 Hz, 0 samples
        
        >>> for tr in st:
        ...     if tr.stats.channel[-1] not in ["Z", "N", "E"]:
        ...         st.remove(tr)
        >>> print st
        4 Trace(s) in Stream:
        ...EHZ | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.000000Z | 1.0 Hz, 0 samples
        ...EHZ | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.000000Z | 1.0 Hz, 0 samples
        ...EHN | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.000000Z | 1.0 Hz, 0 samples
        ...EHE | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.000000Z | 1.0 Hz, 0 samples
        
        """
        # Uncomment next line to see the default behavior and run doctest to
        # see how it can go wrong...
        # XXX return self.traces.__iter__()
        return list(self.traces).__iter__()

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
        return_string = str(len(self.traces)) + ' Trace(s) in Stream:\n'
        return return_string + "\n".join([str(tr) for tr in self])

    def __eq__(self, other):
        """
        Implements rich comparison of Stream objects for "==" operator.

        Example
        -------
        >>> from obspy.core import read
        >>> st = read()
        >>> st2 = st.copy()
        >>> st is st2
        False
        >>> st == st2
        True

        Streams are the same, if both contain the same traces, i.e. after a
        sort operation going through both streams every trace should be equal
        according to Trace's __eq__ operator.
        """
        if not isinstance(other, Stream):
            return False

        # this is maybe still not 100% satisfactory, the question here is if
        # two streams should be the same in comparison if one of the streams
        # has a duplicate trace. Using sets at the moment, two equal traces
        # in one of the Streams would lead to two non-equal Streams.
        # This is a bit more conservative and most likely the expected behavior
        # in most cases.
        self_sorted = self.select()
        self_sorted.sort()
        other_sorted = other.select()
        other_sorted.sort()
        if not self_sorted.traces == other_sorted.traces:
            return False

        return True

    def __ne__(self, other):
        """
        Implements rich comparison of Stream objects for "!=" operator.

        Example
        -------
        >>> from obspy.core import read
        >>> st = read()
        >>> st2 = st.copy()
        >>> st is st2
        False
        >>> st != st2
        False

        Calls __eq__() and returns the opposite.
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        """
        Too ambiguous, throw an Error.
        """
        raise NotImplementedError("Too ambiguous, therefore not implemented.")

    def __le__(self, other):
        """
        Too ambiguous, throw an Error.
        """
        raise NotImplementedError("Too ambiguous, therefore not implemented.")

    def __gt__(self, other):
        """
        Too ambiguous, throw an Error.
        """
        raise NotImplementedError("Too ambiguous, therefore not implemented.")

    def __ge__(self, other):
        """
        Too ambiguous, throw an Error.
        """
        raise NotImplementedError("Too ambiguous, therefore not implemented.")

    # Explicitely setting Stream object unhashable (mutable object).
    # See also Python Language Reference (3.0 Data Model):
    # http://docs.python.org/reference/datamodel.html
    #
    # Classes which inherit a __hash__() method from a parent class but change
    # the meaning of __cmp__() or __eq__() such that [...] can explicitly flag
    # themselves as being unhashable by setting __hash__ = None in the class
    # definition. Doing so means that not only will instances of the classraise
    # an appropriate TypeError when a program attempts to retrieve their hash
    # value, but they will also be correctly identified as unhashable when
    # checking isinstance(obj, collections.Hashable) (unlike classes which
    # define their own __hash__() to explicitly raise TypeError).
    __hash__ = None

    def __setitem__(self, index, trace):
        """
        __setitem__ method of obspy.Stream objects.

        :return: Trace objects
        """
        self.traces[index] = trace

    def __getitem__(self, index):
        """
        __getitem__ method of obspy.Stream objects.

        :return: Trace objects
        """
        return self.traces[index]

    def __delitem__(self, index):
        """
        Passes on the __delitem__ method to the underlying list of traces.
        """
        return self.traces.__delitem__(index)

    def __getslice__(self, i, j):
        """
        __getslice__ method of obspy.Stream objects.

        :return: Stream object
        """
        return self.__class__(traces=self.traces[i:j])

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

        Example
        -------
        
        Our example stream has no gaps:
        
        >>> from obspy.core import read, UTCDateTime
        >>> st = read()
        >>> st.getGaps()
        []
        >>> st.printGaps()
        Source            Last Sample                 Next Sample                 Delta           Samples 
        Total: 0 gap(s) and 0 overlap(s)
        
        So let's make a copy of the first trace and cut both so that we end up with
        a gappy stream:

        >>> tr = st[0].copy()
        >>> t = UTCDateTime("2009-08-24T00:20:13.0")
        >>> st[0].trim(endtime=t)
        >>> tr.trim(starttime=t+1)
        >>> st.append(tr)
        >>> st.getGaps()
        [['BW', 'RJOB', '', 'EHZ', UTCDateTime(2009, 8, 24, 0, 20, 13), UTCDateTime(2009, 8, 24, 0, 20, 14), 1.0, 99]]
        >>> st.printGaps()
        Source            Last Sample                 Next Sample                 Delta           Samples 
        BW.RJOB..EHZ      2009-08-24T00:20:13.000000Z 2009-08-24T00:20:14.000000Z 1.000000        99      
        Total: 1 gap(s) and 0 overlap(s)

        :param min_gap: All gaps smaller than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.
        :param max_gap: All gaps larger than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.
        """
        # Create shallow copy of the traces to be able to sort them later on.
        copied_traces = copy.copy(self.traces)
        self.sort()
        gap_list = []
        for _i in xrange(len(self.traces) - 1):
            # skip traces with different network, station, location or channel
            if self.traces[_i].id != self.traces[_i + 1].id:
                continue
            # different sampling rates should always result in a gap or overlap
            if self.traces[_i].stats.delta == self.traces[_i + 1].stats.delta:
                flag = True
            else:
                flag = False
            stats = self.traces[_i].stats
            stime = stats['endtime']
            etime = self.traces[_i + 1].stats['starttime']
            delta = etime.timestamp - stime.timestamp
            # Check that any overlap is not larger than the trace coverage
            if delta < 0:
                temp = self.traces[_i + 1].stats['endtime'].timestamp - \
                       etime.timestamp
                if (delta * -1) > temp:
                    delta = -1 * temp
            # Check gap/overlap criteria
            if min_gap and delta < min_gap:
                continue
            if max_gap and delta > max_gap:
                continue
            # Number of missing samples
            nsamples = int(round(math.fabs(delta) * stats['sampling_rate']))
            # skip if is equal to delta (1 / sampling rate)
            if flag and nsamples == 1:
                continue
            elif delta > 0:
                nsamples -= 1
            else:
                nsamples += 1
            gap_list.append([stats['network'], stats['station'],
                             stats['location'], stats['channel'],
                             stime, etime, delta, nsamples])
        # Set the original traces to not alter the stream object.
        self.traces = copied_traces
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

        Example
        -------
        >>> from obspy.core import read
        >>> st = read()
        >>> st.plot() #doctest: +SKIP

        .. plot:
            from obspy.core import read
            st = read()
            st.plot()

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
        :param fig: Use an existing figure instance, default None
        """
        try:
            from obspy.imaging.waveform import WaveformPlotting
        except:
            msg = "Please install module obspy.imaging to be able to " + \
                  "plot ObsPy Stream objects."
            print msg
            raise
        waveform = WaveformPlotting(stream=self, *args, **kwargs)
        return waveform.plotWaveform()

    def spectrogram(self, *args, **kwargs):
        """
        Creates a spectrogram plot for each trace in the stream.

        For details on kwargs that can be used to customize the spectrogram
        plot see :func:`~obspy.imaging.spectrogram.spectrogram`.

        Basic Usage
        -----------
        >>> from obspy.core import read
        >>> st = read()
        >>> st.spectrogram() #doctest: +SKIP

        .. plot::

            from obspy.core import read
            st = read()
            st.spectrogram()
        """
        spec_list = []
        for tr in self:
            spec = tr.spectrogram(*args, **kwargs)
            spec_list.append(spec)

        return spec_list

    def pop(self, index= -1):
        """
        Removes the Trace object specified by index from the Stream object and
        returns it. If no index is given it will remove the last Trace.
        Passes on the pop() to self.traces.

        :param index: Index of the Trace object to be returned and removed.
        :returns: Removed Trace.
        """
        return self.traces.pop(index)

    def printGaps(self, **kwargs):
        """
        Print gap/overlap list summary information of the Stream object.

        Example
        -------
        
        Our example stream has no gaps:
        
        >>> from obspy.core import read, UTCDateTime
        >>> st = read()
        >>> st.getGaps()
        []
        >>> st.printGaps()
        Source            Last Sample                 Next Sample                 Delta           Samples 
        Total: 0 gap(s) and 0 overlap(s)
        
        So let's make a copy of the first trace and cut both so that we end up with
        a gappy stream:

        >>> tr = st[0].copy()
        >>> t = UTCDateTime("2009-08-24T00:20:13.0")
        >>> st[0].trim(endtime=t)
        >>> tr.trim(starttime=t+1)
        >>> st.append(tr)
        >>> st.getGaps()
        [['BW', 'RJOB', '', 'EHZ', UTCDateTime(2009, 8, 24, 0, 20, 13), UTCDateTime(2009, 8, 24, 0, 20, 14), 1.0, 99]]
        >>> st.printGaps()
        Source            Last Sample                 Next Sample                 Delta           Samples 
        BW.RJOB..EHZ      2009-08-24T00:20:13.000000Z 2009-08-24T00:20:14.000000Z 1.000000        99      
        Total: 1 gap(s) and 0 overlap(s)
        """
        result = self.getGaps(**kwargs)
        print "%-17s %-27s %-27s %-15s %-8s" % ('Source', 'Last Sample',
                                                'Next Sample', 'Delta',
                                                'Samples')
        gaps = 0
        overlaps = 0
        for r in result:
            if r[6] > 0:
                gaps += 1
            else:
                overlaps += 1
            print "%-17s %-27s %-27s %-15.6f %-8d" % ('.'.join(r[0:4]),
                                                      r[4], r[5], r[6], r[7])
        print "Total: %d gap(s) and %d overlap(s)" % (gaps, overlaps)

    def remove(self, trace):
        """
        Removes the first occurence of the specified Trace object in the Stream
        object.
        Passes on the remove() call to self.traces.

        Example
        -------

        This example shows how to delete all "E" component traces in a stream:

        >>> from obspy.core import read
        >>> st = read()
        >>> print st
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
        >>> for tr in st.select(component="E"):
        ...     st.remove(tr)
        >>> print st
        2 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples

        :param trace: Trace object to be removed from Stream.
        :returns: None
        """
        return self.traces.remove(trace)

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

        >>> from obspy.core import read
        >>> st = read() # doctest: +SKIP
        >>> st.write("example.mseed", format="MSEED") # doctest: +SKIP

        Writing files with meaningful filenames can be done e.g. using trace.id

        >>> for tr in st: #doctest: +SKIP
        ...     tr.write("%s.MSEED" % tr.id, format="MSEED") #doctest: +SKIP

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
        # Check all traces for masked arrays and raise exception.
        for trace in self.traces:
            if np.ma.is_masked(trace.data):
                msg = 'Masked array writing is not supported. You can use ' + \
                      'np.array.filled() to convert the masked array to a ' + \
                      'normal array.'
                raise Exception(msg)
        format = format.upper()
        # Gets all available formats and the corresponding entry points.
        formats_ep = _getPlugins('obspy.plugin.waveform', 'writeFormat')
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

    def trim(self, starttime=None, endtime=None, pad=False, nearest_sample=True):
        """
        Cuts all traces of this Stream object to given start and end time.
        If nearest_sample=True the closest sample point of the first trace
        is the select, the remaining traces are trimmed according to that
        sample point.
        For more info see :meth:`~obspy.core.trace.Trace.trim.`

        """
        # select starttime/endtime fitting to a sample point of the first trace
        if nearest_sample:
            tr = self.traces[0]
            if starttime:
                delta = round((starttime - tr.stats.starttime) * \
                               tr.stats.sampling_rate)
                starttime = tr.stats.starttime + delta * tr.stats.delta
            if endtime:
                delta = round((endtime - tr.stats.endtime) * \
                               tr.stats.sampling_rate)
                endtime = tr.stats.endtime + delta * tr.stats.delta #delta is negativ!
        for trace in self.traces:
            trace.trim(starttime, endtime, pad,
                       nearest_sample=nearest_sample)
        # remove empty traces after trimming 
        self.traces = [tr for tr in self.traces if tr.stats.npts]

    def ltrim(self, *args, **kwargs):
        """
        Deprecated. Please use :meth:`~obspy.core.stream.Stream.trim` instead.
        This method will be removed in the next major release.
        """
        msg = "Use trim(starttime=starttime, endtime=None, ...) instead"
        warnings.warn(msg, DeprecationWarning)
        self.__ltrim(*args, **kwargs)

    def __ltrim(self, starttime, pad=False, nearest_sample=True):
        """
        Cuts all traces of this Stream object to given start time.
        For more info see :meth:`~obspy.core.trace.Trace._ltrim.`
        """
        for trace in self.traces:
            trace.ltrim(starttime, pad, nearest_sample=nearest_sample)
        # remove empty traces after trimming 
        self.traces = [tr for tr in self.traces if tr.stats.npts]

    def rtrim(self, *args, **kwargs):
        """
        Deprecated. Please use :meth:`~obspy.core.stream.Stream.trim` instead.
        This method will be removed in the next major release.
        """
        msg = "Use trim(starttime=None, endtime=endtime, ...) instead"
        warnings.warn(msg, DeprecationWarning)
        self.__rtrim(*args, **kwargs)

    def __rtrim(self, endtime, pad=False, nearest_sample=True):
        """
        Cuts all traces of this Stream object to given end time.
        For more info see :meth:`~obspy.core.trace.Trace._rtrim.`
        """
        for trace in self.traces:
            trace.rtrim(endtime, pad, nearest_sample=nearest_sample)
        # remove empty traces after trimming 
        self.traces = [tr for tr in self.traces if tr.stats.npts]

    def slice(self, starttime, endtime, keep_empty_traces=False):
        """
        Returns new Stream object cut to the given start- and endtime.

        Does not copy the data but only passes a reference. Will by default
        discard any empty traces. Change the keep_empty_traces parameter to
        True to change this behaviour.
        """
        traces = []
        for trace in self:
            sliced_trace = trace.slice(starttime, endtime)
            if keep_empty_traces is False and not sliced_trace.stats.npts:
                continue
            traces.append(sliced_trace)
        return self.__class__(traces=traces)

    def select(self, network=None, station=None, location=None, channel=None,
               sampling_rate=None, npts=None, component=None):
        """
        Returns new Stream object only with these traces that match the given
        stats criteria (e.g. all traces with channel="EHZ").
        All kwargs except for component are tested directly against the
        respective entry in the trace.stats dictionary.
        If a string for component is given (should be a single letter) it is
        tested (case insensitive) against the last letter of the
        trace.stats.channel entry.

        Does not copy the data but only passes a reference.
        """
        # make given component letter uppercase (if e.g. "z" is given)
        if component:
            component = component.upper()
            if channel and component != channel[-1]:
                msg = "Selection criteria for channel and component are " + \
                      "mutually exclusive!"
                raise Exception(msg)
        traces = []
        for trace in self:
            # skip trace if any given criterion is not matched
            if network and network != trace.stats.network:
                continue
            if station and station != trace.stats.station:
                continue
            if location and location != trace.stats.location:
                continue
            if channel and channel != trace.stats.channel:
                continue
            if sampling_rate and float(sampling_rate) != trace.stats.sampling_rate:
                continue
            if npts and int(npts) != trace.stats.npts:
                continue
            if component and component != trace.stats.channel[-1]:
                continue
            traces.append(trace)
        return self.__class__(traces=traces)

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

    def _mergeChecks(self):
        """
        Sanity checks for merging.
        """
        sr = {}
        dtype = {}
        calib = {}
        for trace in self.traces:
            # Check sampling rate.
            sr.setdefault(trace.id, trace.stats.sampling_rate)
            if trace.stats.sampling_rate != sr[trace.id]:
                msg = "Can't merge traces with same ids but differing " + \
                      "sampling rates!"
                raise Exception(msg)
            # Check dtype.
            dtype.setdefault(trace.id, trace.data.dtype)
            if trace.data.dtype != dtype[trace.id]:
                msg = "Can't merge traces with same ids but differing " + \
                      "data types!"
                raise Exception(msg)
            # Check calibration factor. 
            calib.setdefault(trace.id, trace.stats.calib)
            if trace.stats.calib != calib[trace.id]:
                msg = "Can't merge traces with same ids but differing " + \
                      "calibration factors.!"
                raise Exception(msg)

    def merge(self, method=0, fill_value=None, interpolation_samples=0):
        """
        Merges ObsPy Trace objects with same IDs.

        Gaps and overlaps are usually separated in distinct traces. This method
        tries to merge them and to create distinct traces within this 
        :class:`~Stream` object. The method is working on the stream object
        itself (inplace), thus returns nothing. Merged trace data will be
        converted into a NumPy masked array data type if any gaps are
        present. This behavior may be prevented by setting the
        ``fill_value`` parameter. The ``method`` argument controls the
        handling of overlapping data values.

        Parameters
        ----------
        method : [ 0 | 1 ], optional
            Methodology to handle overlaps of traces (default is 0).
            See :meth:`obspy.core.trace.Trace.__add__` for details
        fill_value : int or float, optional
            Fill value for gaps (default is None). Traces will be converted to
            NumPy masked arrays if no value is given and gaps are present.
        interpolation_samples : int, optional
            Used only for method 1. It specifies the number of samples which
            are used to interpolate between overlapping traces (default is 0).
            If set to -1 all overlapping samples are interpolated.
        """
        # check sampling rates and dtypes
        self._mergeChecks()
        # order matters!
        self.sort(keys=['network', 'station', 'location', 'channel',
                        'starttime', 'endtime'])
        # build up dictionary with with lists of traces with same ids
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
        # clear traces of current stream
        self.traces = []
        # loop through ids
        for id in traces_dict.keys():
            cur_trace = traces_dict[id].pop(0)
            # loop through traces of same id
            for _i in xrange(len(traces_dict[id])):
                trace = traces_dict[id].pop(0)
                # disable sanity checks because there are already done
                cur_trace = cur_trace.__add__(trace, method,
                    fill_value=fill_value, sanity_checks=False,
                    interpolation_samples=interpolation_samples)
            self.traces.append(cur_trace)

    def filter(self, type, filter_options, in_place=True):
        """
        Filters the data of all traces in the ``Stream``. This is performed in
        place on the actual data arrays. The raw data is not accessible anymore
        afterwards.
        This also makes an entry with information on the applied processing
        in ``stats.processing`` of every trace.

        Example
        -------

        The default is in place filtering to avoid unintentional excessive
        memory consumption.

        >>> from obspy.core import read
        >>> st = read()
        >>> st.filter("highpass", {'freq': 1.0})
        >>> st.plot() # doctest: +SKIP

        If the original stream is needed for further work, too, set
        ``ìn_place=False`` to return a new trace object:

        >>> from obspy.core import read
        >>> st = read()
        >>> st2 = st.filter("highpass", {'freq': 1.0}, in_place=False)
        >>> st.plot() # doctest: +SKIP
        >>> st2.plot() # doctest: +SKIP

        .. plot::
            
            from obspy.core import read
            st = read()
            st2 = st.filter("highpass", {'freq': 1.0}, in_place=False)
            st.plot()
            st2.plot()

        :param type: String that specifies which filter is applied (e.g.
                "bandpass").
        :param filter_options: Dictionary that contains arguments that will
                be passed to the respective filter function as kwargs.
                (e.g. {'freqmin': 1.0, 'freqmax': 20.0})
        :param in_place: Determines if the filter is applied in place or if
                a new ``Stream`` object is returned.
        :return: ``None`` if ``in_place=True``, new ``Stream`` with filtered
                traces otherwise.
        """
        new_traces = []
        for trace in self:
            new_tr = trace.filter(type=type, filter_options=filter_options,
                                  in_place=in_place)
            new_traces.append(new_tr)

        if in_place:
            return
        else:
            return self.__class__(traces=new_traces)

    def trigger(self, type, trigger_options, in_place=True):
        """
        Runs a triggering algorithm on all traces in the stream. This is
        performed in place on the actual data arrays by default. The raw data
        is not accessible anymore afterwards. This can be avoided by specifying
        ``in_place=False``.
        This also makes an entry with information on the applied processing
        in ``stats.processing`` of every trace.

        Example
        -------

        >>> from obspy.core import read
        >>> st = read()
        >>> st.filter("highpass", {'freq': 1.0})
        >>> st.plot() # doctest: +SKIP
        >>> st.trigger('recStalta', {'sta': 3, 'lta': 10})
        >>> st.plot() # doctest: +SKIP

        .. plot::
            
            from obspy.core import read
            st = read()
            st.filter("highpass", {'freq': 1.0})
            st.plot()
            st.trigger('recStalta', {'sta': 3, 'lta': 10})
            st.plot()

        :param type: String that specifies which trigger is applied (e.g.
                'recStalta').
        :param trigger_options: Dictionary that contains arguments that will
                be passed to the respective trigger function as kwargs.
                Arguments ``sta`` and ``lta`` will be mapped to ``nsta`` and
                ``nlta`` by multiplying with sampling rate of each trace.
                (e.g. {'sta': 3, 'lta': 10} would call the trigger with 3 and
                10 seconds average, respectively)
        :param in_place: Determines if the filter is applied in place or if
                a new ``Stream`` object is returned.
        :return: ``None`` if ``in_place=True``, new ``Stream`` with triggered
                traces otherwise.
        """
        new_traces = []
        for trace in self:
            new_tr = trace.trigger(type=type, trigger_options=trigger_options,
                                   in_place=in_place)
            new_traces.append(new_tr)

        if in_place:
            return
        else:
            return self.__class__(traces=new_traces)

    def downsample(self, decimation_factor, no_filter=False,
                   strict_length=True, in_place=True):
        """
        Downsample data in all traces of stream.

        Currently a simple integer decimation is implemented.
        Only every decimation_factor-th sample remains in the trace, all other
        samples are thrown away. Prior to decimation a lowpass filter is
        applied to ensure no aliasing artifacts are introduced. The automatic
        filtering can be deactivated with ``no_filter=True``.
        If the length of the data array modulo ``decimation_factor`` is not
        zero then the endtime of the trace is changing on sub-sample scale. The
        downsampling is aborted in this case but can be forced by setting
        ``strict_length=False``.
        Per default downsampling is done in place. By setting ``in_place=False`` a
        new ``Stream`` object is returned.

        Basic Usage
        -----------

        By default all processing is done in place to avoid unintentional
        excessive memory consumption. For the example we switch off the
        automatic pre-filtering:

        >>> tr = Trace(data=np.arange(10))
        >>> st = Stream(traces=[tr])
        >>> tr.stats.sampling_rate
        1.0
        >>> tr.data
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> st.downsample(4, strict_length=False, no_filter=True)
        >>> tr.stats.sampling_rate
        0.25
        >>> tr.data
        array([0, 4])
        
        If the original trace is needed for further work the ``in_place``
        option can be set to ``False`` to return a new trace:
        
        >>> tr = Trace(data=np.arange(10))
        >>> st = Stream(traces=[tr])
        >>> tr.stats.sampling_rate
        1.0
        >>> tr.data
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> st2 = st.downsample(2, no_filter=True, in_place=False)
        >>> tr2 = st2[0]
        >>> tr.stats.sampling_rate
        1.0
        >>> tr.data
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> tr2.stats.sampling_rate
        0.5
        >>> tr2.data
        array([0, 2, 4, 6, 8])

        :param decimation_factor: integer factor by which the sampling rate is
            lowered by decimation.
        :param no_filter: deactivate automatic filtering
        :param strict_length: leave traces unchanged for which endtime of trace
            would change
        :param in_place: perform operation in place or return new ``Stream``
            object.
        :return: ``None`` if ``in_place=True``, new ``Stream`` with downsampled
            data otherwise.
        """
        new_traces = []
        for trace in self:
            new_tr = trace.downsample(decimation_factor=decimation_factor,
                    no_filter=no_filter, strict_length=strict_length,
                    in_place=in_place)
            new_traces.append(new_tr)

        if in_place:
            return
        else:
            return self.__class__(traces=new_traces)

    def max(self):
        """
        Method to get the values of the absolute maximum amplitudes of all
        traces in the stream.

        >>> tr1 = Trace(data=[0, -3, 9, 6, 4])
        >>> tr2 = Trace(data=[0, -3, -9, 6, 4])
        >>> tr3 = Trace(data=[0.3, -3.5, 9.0, 6.4, 4.3])
        >>> st = Stream(traces=[tr1, tr2, tr3])
        >>> st.max()
        [9, -9, 9.0]

        :return: List of values of absolute maxima of all traces
        """
        values = []

        for tr in self:
            values.append(tr.max())

        return values

    def std(self):
        """
        Method to get the standard deviations of amplitudes in all trace in the
        stream.
        Standard deviations are calculated by numpy method on every
        ``trace.data`` in the stream.
        
        >>> tr1 = Trace(data=[0, -3, 9, 6, 4])
        >>> tr2 = Trace(data=[0.3, -3.5, 9.0, 6.4, 4.3])
        >>> st = Stream(traces=[tr1, tr2])
        >>> st.std()
        [4.2614551505325036, 4.4348618918744247]

        :return: List of standard deviations of all traces.
        """
        values = []

        for tr in self:
            values.append(tr.std())

        return values

    def normalize(self, global_max=False, in_place=True):
        """
        Normalizes all trace in the stream. By default all traces are
        normalized separately to their respective absolute maximum. By setting
        ``global_max=True`` all traces get normalized to the global maximum of
        all traces.
        By default normalization is done in place. By setting
        ``in_place=False`` a new Trace object is returned.
        Note: If self.data.dtype was integer it is changing to float.
        This also makes an entry with information on the applied processing
        in ``stats.processing`` of every trace.
        Note: If ``data.dtype`` of a trace was integer it is changing to float.

        Example
        -------
        
        Make a Stream with two Traces:

        >>> tr1 = Trace(data=[0, -3, 9, 6, 4])
        >>> tr2 = Trace(data=[0.3, -0.5, -0.8, 0.4, 0.3])
        >>> st = Stream(traces=[tr1, tr2])

        Normalize and return a new stream:

        >>> st2 = st.normalize(in_place=False)

        All traces are normalized to their absolute maximum and processing
        information is added:

        >>> st2[0].data
        array([ 0.        , -0.33333333,  1.        ,  0.66666667,  0.44444444])
        >>> st2[0].stats.processing
        ['normalize:9']
        >>> st2[1].data
        array([ 0.375, -0.625, -1.   ,  0.5  ,  0.375])
        >>> st2[1].stats.processing
        ['normalize:-0.8']

        Now do the operation on the original file in place and normalize all
        traces to the stream's global maximum:

        >>> st.normalize(global_max=True)
        >>> st[0].data
        array([ 0.        , -0.33333333,  1.        ,  0.66666667,  0.44444444])
        >>> st[0].stats.processing
        ['normalize:9']
        >>> st[1].data
        array([ 0.03333333, -0.05555556, -0.08888889,  0.04444444,  0.03333333])
        >>> st[1].stats.processing
        ['normalize:9']

        :param global_max: If set to ``True``, all traces are normalized with
                respect to the global maximum of all traces in the stream
                instead of normalizing every trace separately.
        :param in_place: perform operation in place or return new ``Stream``
                object.
        :return: ``None`` if ``in_place=True``, new ``Trace`` with normalized
                data otherwise.
        """
        new_traces = []

        # use the same value for normalization on all traces?
        if global_max:
            norm = max([abs(value) for value in self.max()])
        else:
            norm = None

        # normalize all traces
        for trace in self:
            new_tr = trace.normalize(norm=norm, in_place=in_place)
            new_traces.append(new_tr)

        if in_place:
            return
        else:
            return self.__class__(traces=new_traces)

    def copy(self):
        """
        Returns a deepcopy of the stream.

        Example
        -------

        Make a Trace and copy it:

        >>> from obspy.core import read
        >>> st = read()
        >>> st2 = st.copy()

        The two objects are not the same:

        >>> st2 is st
        False

        But they have equal data (before applying further processing):

        >>> st2 == st
        True

        The following example shows how to make an alias but not copy the data.
        Any changes on ``st3`` would also change the contents of ``st``.

        >>> st3 = st
        >>> st3 is st
        True
        >>> st3 == st
        True

        :return: Copy of stream.
        """
        return copy.deepcopy(self)


def createDummyStream(stream_string):
    """
    Creates a dummy stream object from the output of the print method of any
    Stream or Trace object.

    If the __str__ method of the Stream or Trace objects changes, than this
    method has to be adjusted too.
    """
    stream_io = StringIO(stream_string)
    traces = []
    for line in stream_io:
        line = line.strip()
        # Skip first line.
        if not line or 'Stream' in line:
            continue
        items = line.split(' ')
        items = [item for item in items if len(item) > 1]
        # Map them.
        try:
            id = items[0]
            network, station, location, channel = id.split('.')
            starttime = UTCDateTime(items[1])
            endtime = UTCDateTime(items[2])
            npts = int(items[5])
        except:
            continue
        tr = Trace(data=np.random.ranf(npts))
        tr.stats.network = network
        tr.stats.station = station
        tr.stats.location = location
        tr.stats.channel = channel
        tr.stats.starttime = starttime
        delta = (endtime - starttime) / (npts - 1)
        tr.stats.delta = delta
        # Set as a preview Trace if it is a preview.
        if '[preview]' in line:
            tr.stats.preview = True
        traces.append(tr)
    return Stream(traces=traces)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
