# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Stream objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from glob import glob, iglob, has_magic
from obspy.core.trace import Trace
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile, _getPlugins, interceptDict, \
    getExampleFile, getEntryPoints, uncompressFile
from pkg_resources import load_entry_point
import copy
import fnmatch
import math
import numpy as np
import os
import urllib2
import warnings


ENTRY_POINTS = getEntryPoints()


def read(pathname_or_url=None, format=None, headonly=False,
         nearest_sample=True, **kwargs):
    """
    Read waveform files into an ObsPy Stream object.

    The :func:`~obspy.core.stream.read` function opens either one or multiple
    waveform files given via file name or URL using the ``pathname_or_url``
    attribute.

    The format of the waveform file will be automatically detected if not
    given. Allowed formats mainly depend on ObsPy packages installed. See the
    `Supported Formats` section below.

    This function returns an ObsPy :class:`~obspy.core.stream.Stream` object, a
    ``list``-like object of multiple ObsPy :class:`~obspy.core.trace.Trace`
    objects.

    :type pathname_or_url: string, optional
    :param pathname_or_url: String containing a file name or a URL. Wildcards
        are allowed for a file name. If this attribute is omitted, a Stream
        object with an example data set will be created.
    :type format: string, optional
    :param format: Format of the file to read, e.g. ``"GSE2"``, ``"MSEED"``,
        ``"SAC"``, ``"SEISAN"``, ``"WAV"``, ``"Q"``, ``"SH_ASC"``, etc. See
        the `Supported Formats` section below for a full list of supported
        formats. If format is set to ``None`` it will be automatically detected
        which results in a slightly slower reading. If you specify a format no
        further format checking is done.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the data header. This is most
        useful for scanning available meta information of huge data sets.
    :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
    :param starttime: Specify the start time to read.
    :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
    :param endtime: Specify the end time to read.
    :type nearest_sample: bool, optional
    :param nearest_sample: Only applied if `starttime` or `endtime` is given.
        Select nearest sample or the one containing the specified time. For
        more info, see :meth:`~obspy.core.trace.Trace.trim`.
    :return: A ObsPy :class:`~obspy.core.stream.Stream` object.

    .. rubric:: Basic Usage

    In most cases a filename is specified as the only argument to
    :func:`~obspy.core.stream.read`. For a quick start you may omit all
    arguments and ObsPy will create and return a basic example seismogram.
    Further usages of the :func:`~obspy.core.stream.read` function can
    be seen in the `Further Examples` section underneath.

    >>> from obspy.core import read
    >>> st = read()
    >>> print(st)  # doctest: +ELLIPSIS
    3 Trace(s) in Stream:
    BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
    BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
    BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples

    .. rubric:: Supported Formats

    Additional ObsPy modules extend the functionality of the
    :func:`~obspy.core.stream.read` function. The following table summarizes
    all known file formats currently supported by ObsPy. The table order also
    reflects the order of the autodetection routine if no format option is
    specified.

    Please refer to the `Linked Function Call` of each module for any extra
    options available at the import stage.

    =======  ===================  ====================================
    Format   Required Module      Linked Function Call
    =======  ===================  ====================================
    MSEED    :mod:`obspy.mseed`   :func:`obspy.mseed.core.readMSEED`
    SAC      :mod:`obspy.sac`     :func:`obspy.sac.core.readSAC`
    GSE2     :mod:`obspy.gse2`    :func:`obspy.gse2.core.readGSE2`
    SEISAN   :mod:`obspy.seisan`  :func:`obspy.seisan.core.readSEISAN`
    SACXY    :mod:`obspy.sac`     :func:`obspy.sac.core.readSACXY`
    GSE1     :mod:`obspy.gse2`    :func:`obspy.gse2.core.readGSE1`
    Q        :mod:`obspy.sh`      :func:`obspy.sh.core.readQ`
    SH_ASC   :mod:`obspy.sh`      :func:`obspy.sh.core.readASC`
    SLIST    :mod:`obspy.core`    :func:`obspy.core.ascii.readSLIST`
    TSPAIR   :mod:`obspy.core`    :func:`obspy.core.ascii.readTSPAIR`
    SEGY     :mod:`obspy.segy`    :func:`obspy.segy.core.readSEGY`
    SU       :mod:`obspy.segy`    :func:`obspy.segy.core.readSU`
    WAV      :mod:`obspy.wav`     :func:`obspy.wav.core.readWAV`
    =======  ===================  ====================================

    Next to the :func:`~obspy.core.stream.read` function the
    :meth:`~obspy.core.stream.Stream.write` method of the returned
    :class:`~obspy.core.stream.Stream` object can be used to export the data
    to the file system.

    .. rubric:: Further Examples

    Example waveform files may be retrieved via http://examples.obspy.org.

    (1) Reading multiple local files using wildcards.

        The following code uses wildcards, in this case it matches two files.
        Both files are then read into a single
        :class:`~obspy.core.stream.Stream` object.

        >>> from obspy.core import read  # doctest: +SKIP
        >>> st = read("/path/to/loc_R*.z")  # doctest: +SKIP
        >>> print(st)  # doctest: +SKIP
        2 Trace(s) in Stream:
        .RJOB..Z | 2005-08-31T02:33:49.849998Z - ... | 200.0 Hz, 12000 samples
        .RNON..Z | 2004-06-09T20:05:59.849998Z - ... | 200.0 Hz, 12000 samples

    (2) Reading a local file without format detection.

        Using the ``format`` parameter disables the automatic detection and
        enforces reading a file in a given format.

        >>> from obspy.core import read
        >>> st = read("/path/to/loc_RJOB20050831023349.z", format="GSE2")
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        .RJOB..Z | 2005-08-31T02:33:49.849998Z - ... | 200.0 Hz, 12000 samples

    (3) Reading a remote file via HTTP protocol.

        >>> from obspy.core import read
        >>> st = read("http://examples.obspy.org/loc_RJOB20050831023349.z")
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        .RJOB..Z | 2005-08-31T02:33:49.849998Z - ... | 200.0 Hz, 12000 samples

    (4) Reading a compressed files.

        >>> from obspy.core import read
        >>> st = read("/path/to/tspair.ascii.gz")
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        XX.TEST..BHZ | 2008-01-15T00:00:00.025000Z - ... | 40.0 Hz, 635 samples

        >>> st = read("http://examples.obspy.org/slist.ascii.bz2")
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        XX.TEST..BHZ | 2008-01-15T00:00:00.025000Z - ... | 40.0 Hz, 635 samples
    """
    # if no pathname or URL specified, make example stream
    if not pathname_or_url:
        return _createExampleStream(headonly=headonly)
    # if pathname starts with /path/to/ try to search in examples
    if isinstance(pathname_or_url, basestring) and \
       pathname_or_url.startswith('/path/to/'):
        try:
            pathname_or_url = getExampleFile(pathname_or_url[9:])
        except:
            # otherwise just try to read the given /path/to folder
            pass
    # create stream
    st = Stream()
    if "://" in pathname_or_url:
        # extract extension if any
        suffix = os.path.basename(pathname_or_url).partition('.')[2] or '.tmp'
        # some URL
        fh = NamedTemporaryFile(suffix=suffix)
        fh.write(urllib2.urlopen(pathname_or_url).read())
        fh.close()
        st.extend(_read(fh.name, format, headonly, **kwargs).traces)
        os.remove(fh.name)
    else:
        # file name
        pathname = pathname_or_url
        for file in iglob(pathname):
            st.extend(_read(file, format, headonly, **kwargs).traces)
        if len(st) == 0:
            # try to give more specific information why the stream is empty
            if has_magic(pathname) and not glob(pathname):
                raise Exception("No file matching file pattern: %s" % pathname)
            elif not has_magic(pathname) and not os.path.isfile(pathname):
                raise IOError(2, "No such file or directory", pathname)
            # Only raise error if no starttime/endtime has been set. This
            # will return an empty stream if the user chose a time window with
            # no data in it.
            # XXX: Might cause problems if the data is faulty and the user
            # set starttime/endtime. Not sure what to do in this case.
            elif not 'starttime' in kwargs and not 'endtime' in kwargs:
                raise Exception("Cannot open file/files: %s" % pathname)
    # Trim if times are given.
    starttime = kwargs.get('starttime')
    endtime = kwargs.get('endtime')
    if headonly and (starttime or endtime):
        msg = "Keyword headonly cannot be combined with starttime or endtime."
        raise Exception(msg)
    if starttime:
        st._ltrim(starttime, nearest_sample=nearest_sample)
    if endtime:
        st._rtrim(endtime, nearest_sample=nearest_sample)
    return st


@uncompressFile
def _read(filename, format=None, headonly=False, **kwargs):
    """
    Reads a single file into a ObsPy Stream object.
    """
    if isinstance(filename, basestring) and not os.path.exists(filename):
        msg = "File not found '%s'" % (filename)
        raise IOError(msg)
#XXX: not used in trunk
#    # Hack to make StringIO reading work.
#    # XXX: Needs to be unified for all modules, e.g. maybe a module property
#    # whether nor not it can read memory files.
#    elif hasattr(filename, 'read') and hasattr(filename, 'tell') and \
#         hasattr(filename, 'seek') and (format.upper() == 'MSEED' or \
#         format is None):
#        format = 'MSEED'
    # go through all known formats
    format_ep = None
    if not format:
        # detect format
        eps = ENTRY_POINTS
        for ep in eps:
            try:
                # search isFormat for given entry point
                isFormat = load_entry_point(ep.dist.key,
                                            'obspy.plugin.waveform.' + ep.name,
                                            'isFormat')
            except ImportError, e:
                # verbose error handling/parsing
                msg = "Cannot load module %s:\n%s" % (ep.dist.key, str(e))
                warnings.warn(msg, category=ImportWarning)
                continue
            if isFormat(filename):
                format_ep = ep
                break
    else:
        # format given via argument
        format = format.upper()
        try:
            format_ep = [_i for _i in ENTRY_POINTS if _i.name == format][0]
        except IndexError:
            msg = "Format is not supported. Supported Formats: "
            raise TypeError(msg + ', '.join([_i.name for _i in ENTRY_POINTS]))
    # file format should be known by now
    try:
        # search readFormat for given entry point
        readFormat = load_entry_point(format_ep.dist.key,
                                      'obspy.plugin.waveform.' + \
                                      format_ep.name, 'readFormat')
    except:
        msg = "Format is not supported. Supported Formats: "
        raise TypeError(msg + ', '.join([_i.name for _i in ENTRY_POINTS]))
    if headonly:
        stream = readFormat(filename, headonly=True, **kwargs)
    else:
        stream = readFormat(filename, **kwargs)
    # set a format keyword for each trace
    for trace in stream:
        trace.stats._format = format_ep.name
    return stream


def _createExampleStream(headonly=False):
    """
    Create an example stream.

    Data arrays are stored in NumPy's NPZ format. The header information are
    fixed values.

    PAZ of the used instrument, needed to demonstrate seisSim() etc.:
    paz = {'gain': 60077000.0,
           'poles': [-0.037004+0.037016j, -0.037004-0.037016j, -251.33+0j,
                     -131.04-467.29j, -131.04+467.29j],
           'sensitivity': 2516778400.0,
           'zeros': [0j, 0j]}}
    """
    if not headonly:
        path = os.path.dirname(__file__)
        path = os.path.join(path, "tests", "data", "example.npz")
        data = np.load(path)
    st = Stream()
    for channel in ["EHZ", "EHN", "EHE"]:
        header = {'network': "BW",
                  'station': "RJOB",
                  'location': "",
                  'npts': 3000,
                  'starttime': UTCDateTime(2009, 8, 24, 0, 20, 3),
                  'sampling_rate': 100.0,
                  'calib': 1.0}
        header['channel'] = channel
        if not headonly:
            st.append(Trace(data=data[channel], header=header))
        else:
            st.append(Trace(header=header))
    return st


class Stream(object):
    """
    List like object of multiple ObsPy trace objects.

    :type traces: list of :class:`~obspy.core.trace.Trace`, optional
    :param traces: Initial list of ObsPy Trace objects.

    .. rubric:: Basic Usage

    >>> trace1 = Trace()
    >>> trace2 = Trace()
    >>> stream = Stream(traces=[trace1, trace2])
    >>> print(stream)  # doctest: +ELLIPSIS
    2 Trace(s) in Stream:
    ...

    .. rubric:: Supported Operations

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

    def __add__(self, other):
        """
        Method to add two streams or a stream with a single trace.

        :type other: :class:`~obspy.core.stream.Stream` or
            :class:`~obspy.core.trace.Trace`
        :param other: Stream or Trace object to add.
        :rtype: :class:`~obspy.core.stream.Stream`
        :returns: New Stream object containing references to the traces of the
            original streams

        .. rubric:: Example

        >>> st1 = Stream([Trace(), Trace(), Trace()])
        >>> len(st1)
        3
        >>> st2 = Stream([Trace(), Trace()])
        >>> len(st2)
        2

        >>> stream = st1 + st2  # adding two Streams
        >>> len(stream)
        5

        >>> stream2 = st1 + Trace()  # adding Stream and Trace
        >>> len(stream2)
        4
        """
        if isinstance(other, Trace):
            other = Stream([other])
        if not isinstance(other, Stream):
            raise TypeError
        traces = self.traces + other.traces
        return self.__class__(traces=traces)

    def __iadd__(self, other):
        """
        Method to add two streams with self += other.

        It will extend the current Stream object with the traces of the given
        Stream. Traces will not be copied but references to the original traces
        will be appended.

        :type other: :class:`~obspy.core.stream.Stream` or
            :class:`~obspy.core.trace.Trace`
        :param other: Stream or Trace object to add.

        .. rubric:: Example

        >>> stream = Stream([Trace(), Trace(), Trace()])
        >>> len(stream)
        3

        >>> stream += Stream([Trace(), Trace()])
        >>> len(stream)
        5

        >>> stream += Trace()
        >>> len(stream)
        6
        """
        if isinstance(other, Trace):
            other = Stream([other])
        if not isinstance(other, Stream):
            raise TypeError
        self.extend(other.traces)
        return self

    def __iter__(self):
        """
        Return a robust iterator for stream.traces.

        Doing this it is safe to remove traces from streams inside of
        for-loops using stream's :meth:`~obspy.core.stream.Stream.remove`
        method. Actually this creates a new iterator every time a trace is
        removed inside the for-loop.

        .. rubric:: Example

        >>> from obspy.core import Stream
        >>> st = Stream()
        >>> for component in ["1", "Z", "2", "3", "Z", "N", "E", "4", "5"]:
        ...     channel = "EH" + component
        ...     tr = Trace(header={'station': 'TEST', 'channel': channel})
        ...     st.append(tr)
        >>> print(st)  # doctest: +ELLIPSIS
        9 Trace(s) in Stream:
        .TEST..EH1 | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EHZ | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EH2 | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EH3 | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EHZ | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EHN | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EHE | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EH4 | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EH5 | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples

        >>> for tr in st:
        ...     if tr.stats.channel[-1] not in ["Z", "N", "E"]:
        ...         st.remove(tr)
        >>> print(st)  # doctest: +ELLIPSIS
        4 Trace(s) in Stream:
        .TEST..EHZ | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EHZ | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EHN | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EHE | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        """
        return list(self.traces).__iter__()

    def __len__(self):
        """
        Returns the number of Traces in the Stream object.

        .. rubric:: Example

        >>> stream = Stream([Trace(), Trace(), Trace()])
        >>> len(stream)
        3
        """
        return len(self.traces)

    count = __len__

    def __str__(self, extended=False):
        """
        Returns short summary string of the current stream.

        It will contain the number of Traces in the Stream and the return value
        of each Trace's :meth:`~obspy.core.trace.Trace.__str__` method.

        :type extended: bool, optional
        :param extended: This method will show only 20 traces by default.
            Enable this option to show all entries.

        .. rubric:: Example

        >>> stream = Stream([Trace(), Trace()])
        >>> print(stream)  # doctest: +ELLIPSIS
        2 Trace(s) in Stream:
        ...
        """
        # get longest id
        id_length = self and max(len(tr.id) for tr in self) or 0
        out = str(len(self.traces)) + ' Trace(s) in Stream:\n'
        if len(self.traces) <= 20 or extended is True:
            out = out + "\n".join([tr.__str__(id_length) for tr in self])
        else:
            out = out + "\n" + self.traces[0].__str__() + "\n" + \
                    '...\n(%i other traces)\n...\n' % (len(self.traces) - \
                    2) + self.traces[-1].__str__() + '\n\n[Use "print(' + \
                    'Stream.__str__(extended=True))" to print all Traces]'
        return out

    def __eq__(self, other):
        """
        Implements rich comparison of Stream objects for "==" operator.

        :type other: :class:`~obspy.core.stream.Stream`
        :param other: Stream object for comparison.
        :rtype: bool
        :return: ``True`` if both Streams contain the same traces, i.e. after a
            sort operation going through both streams every trace should be
            equal according to Trace's
            :meth:`~obspy.core.trace.Trace.__eq__` operator.

        .. rubric:: Example

        >>> from obspy.core import read
        >>> st = read()
        >>> st2 = st.copy()
        >>> st is st2
        False
        >>> st == st2
        True
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

        .. rubric:: Example

        >>> from obspy.core import read
        >>> st = read()
        >>> st2 = st.copy()
        >>> st is st2
        False
        >>> st != st2
        False
        """
        # Calls __eq__() and returns the opposite.
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

    # Explicitly setting Stream object unhashable (mutable object).
    # See also Python Language Reference (3.0 Data Model):
    # http://docs.python.org/reference/datamodel.html
    #
    # Classes which inherit a __hash__() method from a parent class but change
    # the meaning of __cmp__() or __eq__() such that [...] can explicitly flag
    # themselves as being unhashable by setting __hash__ = None in the class
    # definition. Doing so means that not only will instances of the class
    # raise an appropriate TypeError when a program attempts to retrieve their
    # hash value, but they will also be correctly identified as unhashable when
    # checking isinstance(obj, collections.Hashable) (unlike classes which
    # define their own __hash__() to explicitly raise TypeError).
    __hash__ = None

    def __setitem__(self, index, trace):
        """
        __setitem__ method of obspy.Stream objects.

        :return: Trace objects
        """
        self.traces.__setitem__(index, trace)

    def __getitem__(self, index):
        """
        __getitem__ method of obspy.Stream objects.

        :return: Trace objects
        """
        if isinstance(index, slice):
            return self.__class__(traces=self.traces.__getitem__(index))
        else:
            return self.traces.__getitem__(index)

    def __delitem__(self, index):
        """
        Passes on the __delitem__ method to the underlying list of traces.
        """
        return self.traces.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        """
        __getslice__ method of obspy.Stream objects.

        :return: Stream object
        """
        # see also http://docs.python.org/reference/datamodel.html
        return self.__class__(traces=self.traces[max(0, i):max(0, j):k])

    def append(self, trace):
        """
        Appends a single Trace object to the current Stream object.

        :param trace: :class:`~obspy.core.stream.Trace` object.

        .. rubric:: Example

        >>> from obspy.core import read, Trace
        >>> st = read()
        >>> tr = Trace()
        >>> tr.stats.station = 'TEST'
        >>> st.append(tr)
        >>> print(st)  # doctest: +ELLIPSIS
        4 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        .TEST..      | 1970-01-01T00:00:00.000000Z ... | 1.0 Hz, 0 samples
        """
        if isinstance(trace, Trace):
            self.traces.append(trace)
        else:
            msg = 'Append only supports a single Trace object as an argument.'
            raise TypeError(msg)

    def extend(self, trace_list):
        """
        Extends the current Stream object with a list of Trace objects.

        :param trace_list: list of :class:`~obspy.core.stream.Trace` objects or
            :class:`~obspy.core.stream.Stream`.

        .. rubric:: Example

        >>> from obspy.core import read, Trace
        >>> st = read()
        >>> tr1 = Trace()
        >>> tr1.stats.station = 'TEST1'
        >>> tr2 = Trace()
        >>> tr2.stats.station = 'TEST2'
        >>> st.extend([tr1, tr2])
        >>> print(st)  # doctest: +ELLIPSIS
        5 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        .TEST1..     | 1970-01-01T00:00:00.000000Z ... | 1.0 Hz, 0 samples
        .TEST2..     | 1970-01-01T00:00:00.000000Z ... | 1.0 Hz, 0 samples
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

        :param min_gap: All gaps smaller than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.
        :param max_gap: All gaps larger than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.

        The returned list contains one item in the following form for each gap/
        overlap: [network, station, location, channel, starttime of the gap,
        endtime of the gap, duration of the gap, number of missing samples]

        Please be aware that no sorting and checking of stations, channels, ...
        is done. This method only compares the start- and endtimes of the
        Traces.

        .. rubric:: Example

        Our example stream has no gaps:

        >>> from obspy.core import read, UTCDateTime
        >>> st = read()
        >>> st.getGaps()
        []
        >>> st.printGaps() # doctest: +ELLIPSIS
        Source            Last Sample                 ...
        Total: 0 gap(s) and 0 overlap(s)

        So let's make a copy of the first trace and cut both so that we end up
        with a gappy stream:

        >>> tr = st[0].copy()
        >>> t = UTCDateTime("2009-08-24T00:20:13.0")
        >>> st[0].trim(endtime=t)
        >>> tr.trim(starttime=t+1)
        >>> st.append(tr)
        >>> st.getGaps()  # doctest: +ELLIPSIS
        [['BW', 'RJOB', '', 'EHZ', UTCDateTime(2009, 8, 24, 0, 20, 13), ...
        >>> st.printGaps()  # doctest: +ELLIPSIS
        Source            Last Sample                 ...
        BW.RJOB..EHZ      2009-08-24T00:20:13.000000Z ...
        Total: 1 gap(s) and 0 overlap(s)
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
        Creates a waveform plot of the current ObsPy Stream object.

        :param outfile: Output file string. Also used to automatically
            determine the output format. Supported file formats depend on your
            matplotlib backend. Most backends support png, pdf, ps, eps and
            svg. Defaults to ``None``.
        :param format: Format of the graph picture. If no format is given the
            outfile parameter will be used to try to automatically determine
            the output format. If no format is found it defaults to png output.
            If no outfile is specified but a format is, than a binary
            imagestring will be returned.
            Defaults to ``None``.
        :param starttime: Starttime of the graph as a
            :class:`~obspy.core.utcdatetime.UTCDateTime` object. If not set
            the graph will be plotted from the beginning.
            Defaults to ``False``.
        :param endtime: Endtime of the graph as a
            :class:`~obspy.core.utcdatetime.UTCDateTime` object. If not set
            the graph will be plotted until the end.
            Defaults to ``False``.
        :param fig: Use an existing matplotlib figure instance.
            Default to ``None``.
        :param automerge: If automerge is True, Traces with the same id will be
            merged.
            Defaults to ``True``.
        :param size: Size tuple in pixel for the output file. This corresponds
            to the resolution of the graph for vector formats.
            Defaults to ``(800, 250)`` pixel per channel.
        :param dpi: Dots per inch of the output file. This also affects the
            size of most elements in the graph (text, linewidth, ...).
            Defaults to ``100``.
        :param color: Color of the graph as a matplotlib color string as
            described below. If ``type='dayplot'`` a list/tuple of color
            strings is expected that will be periodically repeated for each
            line plotted.
            Defaults to ``'black'`` or to ``('#B2000F', '#004C12', '#847200',
            '#0E01FF')`` for ``type='dayplot'``.
        :param bgcolor: Background color of the graph.
            Defaults to ``'white'``.
        :param face_color: Facecolor of the matplotlib canvas.
            Defaults to ``'white'``.
        :param transparent: Make all backgrounds transparent (True/False). This
            will overwrite the bgcolor and face_color arguments.
            Defaults to ``False``.
        :param number_of_ticks: The number of ticks on the x-axis.
            Defaults to ``5``.
        :param tick_format: The way the time axis is formated.
            Defaults to ``'%H:%M:%S'``.
        :param tick_rotation: Tick rotation in degrees.
            Default to ``0``.
        :param handle: Whether or not to return the matplotlib figure instance
            after the plot has been created.
            Defaults to ``False``.
        :param method: By default, all traces with more than 400.000 samples
            will be plotted with a fast method that cannot be zoomed.
            Setting this argument to ``'full'`` will straight up plot the data.
            This results in a potentially worse performance but the interactive
            matplotlib view can be used properly.
            Defaults to 'fast'.
        :param type: Type may be set to either ``'dayplot'`` in order to create
            a one-day plot for a single Trace or ``'relative'`` to convert all
            date/time information to a relative scale, effectively starting
            the seismogram at 0 seconds. ``'normal'`` will produce a standard
            plot.
            Defaults to ``'normal'``.
        :param vertical_scaling_range: Only used if ``type='dayplot'``.
            Determines how each line is scaled in its given space. Every line
            will be centered around its mean value and then clamped to fit its
            given space. This argument is the range in data units that will be
            used to clamp the data. If the range is smaller than the actual
            range, the lines' data may overshoot to other lines which is
            usually a desired effect. Larger ranges will result in a vertical
            padding.
            If ``0``, the actual range of the data will be used and no
            overshooting or additional padding will occur.
            If ``None`` the range will be chosen to be the 99.5-percentile of
            the actual range - so some values will overshoot.
            Defaults to None.
        :param interval: Only used if ``type='dayplot'``. This defines the
            interval length in minutes for one line.
        :param time_offset: Only used if ``type='dayplot'``. The dayplot will
            have two vertical scales. One showing UTC time and the other a user
            defined timezone. This argument specifies the offset of the other
            time scale in hours relative to UTC time.
            Defaults to the current offset of the system time to UTC time.
        :param timezone: Only used if ``type='dayplot'``. Defines the name of
            the user defined time scale.
            Defaults to ``'local time'``.

        .. rubric:: Color Options

        Colors can be specified as defined in the :mod:`matplotlib.colors`
        documentation.

        Short Version: For all color values, you can either use:

        * legit `HTML color names <http://www.w3.org/TR/css3-color/#html4>`_,
          e.g. ``'blue'``,
        * HTML hex strings, e.g. ``'#EE00FF'``,
        * pass an string of a R, G, B tuple, where each of the component is a
          float value in the range of 0 to 1, e.g. ``'(1, 0.25, 0.5)'``, or
        * use a single letters for the basic built-in colors, such as ``'b'``
          (blue), ``'g'`` (green), ``'r'`` (red), ``'c'`` (cyan), ``'m'``
          (magenta), ``'y'`` (yellow), ``'k'`` (black), ``'w'`` (white).

        .. rubric:: Example

        >>> from obspy.core import read
        >>> st = read()
        >>> st.plot() # doctest: +SKIP

        .. plot::

            from obspy.core import read
            st = read()
            st.plot()
        """
        try:
            from obspy.imaging.waveform import WaveformPlotting
        except:
            msg = "Please install module obspy.imaging to be able to " + \
                  "plot ObsPy Stream objects."
            warnings.warn(msg, category=ImportWarning)
            raise
        waveform = WaveformPlotting(stream=self, *args, **kwargs)
        return waveform.plotWaveform(*args, **kwargs)

    def spectrogram(self, *args, **kwargs):
        """
        Creates a spectrogram plot for each trace in the stream.

        For details on kwargs that can be used to customize the spectrogram
        plot see :func:`obspy.imaging.spectrogram.spectrogram`.

        .. rubric:: Example

        >>> from obspy.core import read
        >>> st = read()
        >>> st[0].spectrogram() # doctest: +SKIP

        .. plot::

            from obspy.core import read
            st = read()
            st[0].spectrogram()
        """
        spec_list = []
        for tr in self:
            spec = tr.spectrogram(*args, **kwargs)
            spec_list.append(spec)

        return spec_list

    def pop(self, index=(-1)):
        """
        Removes the Trace object specified by index from the Stream object and
        returns it. If no index is given it will remove the last Trace.
        Passes on the pop() to self.traces.

        :param index: Index of the Trace object to be returned and removed.
        :returns: Removed Trace.

        .. rubric:: Example

        >>> from obspy.core import read
        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> tr = st.pop()
        >>> print(st)  # doctest: +ELLIPSIS
        2 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> print(tr)  # doctest: +ELLIPSIS
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        """
        return self.traces.pop(index)

    def printGaps(self, **kwargs):
        """
        Print gap/overlap list summary information of the Stream object.

        .. rubric:: Example

        Our example stream has no gaps:

        >>> from obspy.core import read, UTCDateTime
        >>> st = read()
        >>> st.getGaps()
        []
        >>> st.printGaps()  # doctest: +ELLIPSIS
        Source            Last Sample                 Next Sample ...
        Total: 0 gap(s) and 0 overlap(s)

        So let's make a copy of the first trace and cut both so that we end up
        with a gappy stream:

        >>> tr = st[0].copy()
        >>> t = UTCDateTime("2009-08-24T00:20:13.0")
        >>> st[0].trim(endtime=t)
        >>> tr.trim(starttime=t+1)
        >>> st.append(tr)
        >>> st.getGaps()  # doctest: +ELLIPSIS
        [['BW', 'RJOB', '', 'EHZ', UTCDateTime(2009, 8, 24, 0, 20, 13), ...
        >>> st.printGaps()  # doctest: +ELLIPSIS
        Source            Last Sample                 ...
        BW.RJOB..EHZ      2009-08-24T00:20:13.000000Z ...
        Total: 1 gap(s) and 0 overlap(s)
        """
        result = self.getGaps(**kwargs)
        print("%-17s %-27s %-27s %-15s %-8s" % ('Source', 'Last Sample',
                                                'Next Sample', 'Delta',
                                                'Samples'))
        gaps = 0
        overlaps = 0
        for r in result:
            if r[6] > 0:
                gaps += 1
            else:
                overlaps += 1
            print("%-17s %-27s %-27s %-15.6f %-8d" % ('.'.join(r[0:4]),
                                                      r[4], r[5], r[6], r[7]))
        print("Total: %d gap(s) and %d overlap(s)" % (gaps, overlaps))

    def remove(self, trace):
        """
        Removes the first occurrence of the specified Trace object in the
        Stream object. Passes on the remove() call to self.traces.

        :param trace: Trace object to be removed from Stream.

        .. rubric:: Example

        This example shows how to delete all "E" component traces in a stream:

        >>> from obspy.core import read
        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> for tr in st.select(component="E"):
        ...     st.remove(tr)
        >>> print(st)  # doctest: +ELLIPSIS
        2 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        """
        return self.traces.remove(trace)

    def reverse(self):
        """
        Reverses the Traces of the Stream object in place.

        .. rubric:: Example

        >>> from obspy.core import read
        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> st.reverse()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        """
        self.traces.reverse()

    def sort(self, keys=['network', 'station', 'location', 'channel',
                         'starttime', 'endtime']):
        """
        Method to sort the traces in the Stream object.

        The traces will be sorted according to the keys list. It will be sorted
        by the first item first, then by the second and so on. It will always
        be sorted from low to high and from A to Z.

        :type keys: list, optional
        :param keys: List containing the values according to which the traces
             will be sorted. They will be sorted by the first item first and
             then by the second item and so on.
             Available items: 'network', 'station', 'channel', 'location',
             'starttime', 'endtime', 'sampling_rate', 'npts', 'dataquality'
             Defaults to ['network', 'station', 'location', 'channel',
             'starttime', 'endtime'].

        .. rubric:: Example

        >>> from obspy.core import read
        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> st.sort()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
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

    def write(self, filename, format, **kwargs):
        """
        Saves stream into a file.

        :type filename: string
        :param filename: The name of the file to write.
        :type format: string
        :param format: The format to write must be specified. Depending on your
            ObsPy installation one of ``"MSEED"``, ``"GSE2"``, ``"SAC"``,
            ``"SACXY"``, ``"Q"``, ``"SH_ASC"``, ``"SEGY"``, ``"SU"``,
            ``"WAV"``. See the `Supported Formats` section below for a full
            list of supported formats.

        .. rubric:: Example

        >>> from obspy.core import read
        >>> st = read() # doctest: +SKIP
        >>> st.write("example.mseed", format="MSEED") # doctest: +SKIP

        Writing files with meaningful filenames can be done e.g. using trace.id

        >>> for tr in st: #doctest: +SKIP
        ...     tr.write("%s.MSEED" % tr.id, format="MSEED") #doctest: +SKIP

        .. rubric:: Supported Formats

        Additional ObsPy modules extend the parameters of the
        :meth:`~obspy.core.stream.Stream.write` method. The following
        table summarizes all known formats currently available for ObsPy.

        Please refer to the *Linked Function Call* of each module for any extra
        options available.

        =======  ===================  ====================================
        Format   Required Module      Linked Function Call
        =======  ===================  ====================================
        MSEED    :mod:`obspy.mseed`   :func:`obspy.mseed.core.writeMSEED`
        GSE2     :mod:`obspy.gse2`    :func:`obspy.gse2.core.writeGSE2`
        SAC      :mod:`obspy.sac`     :func:`obspy.sac.core.writeSAC`
        SACXY    :mod:`obspy.sac`     :func:`obspy.sac.core.writeSACXY`
        Q        :mod:`obspy.sh`      :func:`obspy.sh.core.writeQ`
        SH_ASC   :mod:`obspy.sh`      :func:`obspy.sh.core.writeASC`
        SEGY     :mod:`obspy.segy`    :func:`obspy.segy.core.writeSEGY`
        SLIST    :mod:`obspy.core`    :func:`obspy.core.ascii.writeSLIST`
        SU       :mod:`obspy.segy`    :func:`obspy.segy.core.writeSU`
        TSPAIR   :mod:`obspy.core`    :func:`obspy.core.ascii.writeTSPAIR`
        WAV      :mod:`obspy.wav`     :func:`obspy.wav.core.writeWAV`
        =======  ===================  ====================================
        """
        # Check all traces for masked arrays and raise exception.
        for trace in self.traces:
            if isinstance(trace.data, np.ma.masked_array):
                msg = 'Masked array writing is not supported. You can use ' + \
                      'np.array.filled() to convert the masked array to a ' + \
                      'normal array.'
                raise NotImplementedError(msg)
        format = format.upper()
        # Gets all available formats and the corresponding entry points.
        formats_ep = _getPlugins('obspy.plugin.waveform', 'writeFormat')
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

    def trim(self, starttime=None, endtime=None, pad=False,
             nearest_sample=True, fill_value=None):
        """
        Cuts all traces of this Stream object to given start and end time.
        If nearest_sample=True the closest sample point of the first trace
        is the select, the remaining traces are trimmed according to that
        sample point.
        For more info see :meth:`~obspy.core.trace.Trace.trim`.

        .. rubric:: Example

        >>> from obspy.core import read
        >>> st = read()
        >>> len(st[0])
        3000
        >>> dt = st[0].stats.starttime
        >>> dt
        UTCDateTime(2009, 8, 24, 0, 20, 3)
        >>> st.trim(dt + 2, dt + 4)
        >>> len(st[0])
        201
        >>> st[0].stats.starttime
        UTCDateTime(2009, 8, 24, 0, 20, 5)
        """
        if not self:
            return
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
                # delta is negative!
                endtime = tr.stats.endtime + delta * tr.stats.delta
        for trace in self.traces:
            trace.trim(starttime, endtime, pad,
                       nearest_sample=nearest_sample, fill_value=fill_value)
        # remove empty traces after trimming
        self.traces = [tr for tr in self.traces if tr.stats.npts]

    def _ltrim(self, starttime, pad=False, nearest_sample=True):
        """
        Cuts all traces of this Stream object to given start time.
        For more info see :meth:`~obspy.core.trace.Trace._ltrim.`
        """
        for trace in self.traces:
            trace.trim(starttime=starttime, pad=pad,
                       nearest_sample=nearest_sample)
        # remove empty traces after trimming
        self.traces = [tr for tr in self.traces if tr.stats.npts]

    def _rtrim(self, endtime, pad=False, nearest_sample=True):
        """
        Cuts all traces of this Stream object to given end time.
        For more info see :meth:`~obspy.core.trace.Trace._rtrim.`
        """
        for trace in self.traces:
            trace.trim(endtime=endtime, pad=pad, nearest_sample=nearest_sample)
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
               sampling_rate=None, npts=None, component=None, id=None):
        """
        Returns new Stream object only with these traces that match the given
        stats criteria (e.g. all traces with ``channel="EHZ"``).

        .. rubric:: Examples

        >>> from obspy.core import read
        >>> st = read()
        >>> st2 = st.select(station="R*")
        >>> print(st2)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples

        >>> st2 = st.select(id="BW.RJOB..EHZ")
        >>> print(st2)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples

        >>> st2 = st.select(component="Z")
        >>> print(st2)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples

        >>> st2 = st.select(network="CZ")
        >>> print(st2) # doctest: +NORMALIZE_WHITESPACE
        0 Trace(s) in Stream:

        .. warning::
            A new Stream object is returned but the traces it contains are
            just aliases to the traces of the original stream. Does not copy
            the data but only passes a reference.

        All keyword arguments except for ``component`` are tested directly
        against the respective entry in the
        :class:`Trace.stats <obspy.core.trace.Stats>`` dictionary.
        If a string for ``component`` is given (should be a single letter) it
        is tested (case insensitive) against the last letter of the
        ``Trace.stats.channel`` entry.

        Alternatively, ``channel`` may have the last one or two letters
        wildcarded (e.g. ``channel="EH*"``) to select all components with a
        common band/instrument code.

        All other selection criteria that accept strings (network, station,
        location) may also contain Unix style wildcards (``*``, ``?``, ...).
        """
        # make given component letter uppercase (if e.g. "z" is given)
        if component:
            component = component.upper()
            if channel and channel[-1] != "*" and component != channel[-1]:
                msg = "Selection criteria for channel and component are " + \
                      "mutually exclusive!"
                raise ValueError(msg)
        traces = []
        for trace in self:
            # skip trace if any given criterion is not matched
            if id and not fnmatch.fnmatch(trace.id, id):
                continue
            if network and not fnmatch.fnmatch(trace.stats.network, network):
                continue
            if station and not fnmatch.fnmatch(trace.stats.station, station):
                continue
            if location and not fnmatch.fnmatch(trace.stats.location,
                                                location):
                continue
            if channel and not fnmatch.fnmatch(trace.stats.channel, channel):
                continue
            if sampling_rate and \
               float(sampling_rate) != trace.stats.sampling_rate:
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

        .. rubric:: Example

        >>> from obspy.core import Trace, Stream
        >>> tr = Trace(data=np.array([1, 2, 3, 4]))
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

        :type method: ``-1``, ``0`` or ``1``, optional
        :param method: Methodology to handle overlaps of traces. Defaults
            to ``0``.
            See :meth:`obspy.core.trace.Trace.__add__` for details on
            methods ``0`` and ``1``,
            see :meth:`obspy.core.stream.Stream._cleanup` for details on
            method ``-1``.
        :type fill_value: int or float, ``'latest'`` or ``'interpolate'``,
            optional
        :param fill_value: Fill value for gaps. Defaults to ``None``. Traces
            will be converted to NumPy masked arrays if no value is given and
            gaps are present. The value ``'latest'`` will use the latest value
            before the gap. If value ``'interpolate'`` is provided, missing
            values are linearly interpolated (not changing the data
            type e.g. of integer valued traces). Not used for ``method=-1``.
        :type interpolation_samples: int, optional
        :param interpolation_samples: Used only for ``method=1``. It specifies
            the number of samples which are used to interpolate between
            overlapping traces. Default to ``0``. If set to ``-1`` all
            overlapping samples are interpolated.

        Importing waveform data containing gaps or overlaps results into
        a :class:`~obspy.core.stream.Stream` object with multiple traces having
        the same identifier. This method tries to merge such traces inplace,
        thus returning nothing. Merged trace data will be converted into a
        NumPy :class:`~numpy.ma.MaskedArray` type if any gaps are present. This
        behavior may be prevented by setting the ``fill_value`` parameter.
        The ``method`` argument controls the handling of overlapping data
        values.
        """
        if method == -1:
            self._cleanup()
            return
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

    def simulate(self, paz_remove=None, paz_simulate=None,
                 remove_sensitivity=True, simulate_sensitivity=True, **kwargs):
        """
        Correct for instrument response / Simulate new instrument response.

        :type paz_remove: dict, None
        :param paz_remove: Dictionary containing keys ``'poles'``, ``'zeros'``,
            ``'gain'`` (A0 normalization factor). Poles and zeros must be a
            list of complex floating point numbers, gain must be of type float.
            Poles and Zeros are assumed to correct to m/s, SEED convention.
            Use ``None`` for no inverse filtering.
            Use ``'self'`` to use paz AttribDict in ``trace.stats`` for every
            trace in stream.
        :type paz_simulate: dict, None
        :param paz_simulate: Dictionary containing keys ``'poles'``,
            ``'zeros'``, ``'gain'``. Poles and zeros must be a list of complex
            floating point numbers, gain must be of type float. Or ``None`` for
            no simulation.
        :type remove_sensitivity: bool
        :param remove_sensitivity: Determines if data is divided by
            ``paz_remove['sensitivity']`` to correct for overall sensitivity of
            recording instrument (seismometer/digitizer) during instrument
            correction.
        :type simulate_sensitivity: bool
        :param simulate_sensitivity: Determines if data is multiplied with
            ``paz_simulate['sensitivity']`` to simulate overall sensitivity of
            new instrument (seismometer/digitizer) during instrument
            simulation.

        This function corrects for the original instrument response given by
        ``paz_remove`` and/or simulates a new instrument response given by
        ``paz_simulate``.

        For additional information and more options to control the instrument
        correction/simulation (e.g. water level, demeaning, tapering, ...) see
        :func:`~obspy.signal.invsim.seisSim`.

        The keywords `paz_remove` and `paz_simulate` are expected to be
        dictionaries containing information on poles, zeros and gain (and
        usually also sensitivity).

        If both ``paz_remove`` and ``paz_simulate`` are specified, both steps
        are performed in one go in the frequency domain, otherwise only the
        specified step is performed.

        Processing is performed in place on the actual data array.
        To keep your original data, use :meth:`~obspy.core.stream.Stream.copy`
        to make a copy of your trace.
        This also makes an entry with information on the applied processing
        in ``trace.stats.processing`` of every trace.

        .. rubric:: Example

        >>> from obspy.core import read
        >>> from obspy.signal import cornFreq2Paz
        >>> st = read()
        >>> st.plot() # doctest: +SKIP
        >>> paz_sts2 = {'poles': [-0.037004+0.037016j, -0.037004-0.037016j,
        ...                       -251.33+0j,
        ...                       -131.04-467.29j, -131.04+467.29j],
        ...             'zeros': [0j, 0j],
        ...             'gain': 60077000.0,
        ...             'sensitivity': 2516778400.0}
        >>> paz_1hz = cornFreq2Paz(1.0, damp=0.707)
        >>> st.simulate(paz_remove=paz_sts2, paz_simulate=paz_1hz)
        >>> st.plot() # doctest: +SKIP

        .. plot::

            from obspy.core import read
            from obspy.signal import cornFreq2Paz
            st = read()
            st.plot()
            paz_sts2 = {'poles': [-0.037004+0.037016j, -0.037004-0.037016j,
                                  -251.33+0j,
                                  -131.04-467.29j, -131.04+467.29j],
                        'zeros': [0j, 0j],
                        'gain': 60077000.0,
                        'sensitivity': 2516778400.0}
            paz_1hz = cornFreq2Paz(1.0, damp=0.707)
            paz_1hz['sensitivity'] = 1.0
            st.simulate(paz_remove=paz_sts2, paz_simulate=paz_1hz)
            st.plot()
        """
        for tr in self:
            tr.simulate(paz_remove=paz_remove, paz_simulate=paz_simulate,
                        remove_sensitivity=remove_sensitivity,
                        simulate_sensitivity=simulate_sensitivity, **kwargs)
        return

    @interceptDict
    def filter(self, type, **options):
        """
        Filters the data of all traces in the Stream. This is performed in
        place on the actual data arrays. The raw data is not accessible anymore
        afterwards.

        :type type: str
        :param type: String that specifies which filter is applied (e.g.
            ``"bandpass"``).
        :param options: Necessary keyword arguments for the respective filter
            that will be passed on. (e.g. ``freqmin=1.0``, ``freqmax=20.0`` for
            ``"bandpass"``)

        To keep your original data, use :meth:`~obspy.core.stream.Stream.copy`
        to make a copy of your trace.
        This also makes an entry with information on the applied processing
        in ``stats.processing`` of every trace.

        .. rubric:: Example

        >>> from obspy.core import read
        >>> st = read()
        >>> st.filter("highpass", freq=1.0)
        >>> st.plot() # doctest: +SKIP

        .. plot::

            from obspy.core import read
            st = read()
            st.filter("highpass", freq=1.0)
            st.plot()
        """
        for tr in self:
            tr.filter(type, **options)
        return

    @interceptDict
    def trigger(self, type, **options):
        """
        Runs a triggering algorithm on all traces in the stream. This is
        performed in place on the actual data arrays. The raw data
        is not accessible anymore afterwards.

        :param type: String that specifies which trigger is applied (e.g.
            ``'recStalta'``).
        :param options: Necessary keyword arguments for the respective trigger
            that will be passed on.
            (e.g. ``sta=3``, ``lta=10``)
            Arguments ``sta`` and ``lta`` (seconds) will be mapped to ``nsta``
            and ``nlta`` (samples) by multiplying with sampling rate of trace.
            (e.g. ``sta=3``, ``lta=10`` would call the trigger with 3 and 10
            seconds average, respectively)

        To keep your original data, use :meth:`~obspy.core.stream.Stream.copy`
        to make a copy of your trace.
        This also makes an entry with information on the applied processing
        in ``stats.processing`` of every trace.

        .. rubric:: Example

        >>> from obspy.core import read
        >>> st = read()
        >>> st.filter("highpass", freq=1.0)
        >>> st.plot() # doctest: +SKIP
        >>> st.trigger('recStalta', sta=3, lta=10)
        >>> st.plot() # doctest: +SKIP

        .. plot::

            from obspy.core import read
            st = read()
            st.filter("highpass", freq=1.0)
            st.plot()
            st.trigger('recStalta', sta=3, lta=10)
            st.plot()
        """
        for tr in self:
            tr.trigger(type, **options)
        return

    def downsample(self, decimation_factor, no_filter=False,
                   strict_length=False):
        """
        Downsample data in all traces of stream.

        :type decimation_factor: int
        :param decimation_factor: Factor by which the sampling rate is lowered
            by decimation.
        :type no_filter: bool, optional
        :param no_filter: Deactivates automatic filtering if set to ``True``.
            Defaults to ``False``.
        :type strict_length: bool, optional
        :param strict_length: Leave traces unchanged for which endtime of trace
            would change. Defaults to ``False``.

        Currently a simple integer decimation is implemented.
        Only every decimation_factor-th sample remains in the trace, all other
        samples are thrown away. Prior to decimation a lowpass filter is
        applied to ensure no aliasing artifacts are introduced. The automatic
        filtering can be deactivated with ``no_filter=True``.

        If the length of the data array modulo ``decimation_factor`` is not
        zero then the endtime of the trace is changing on sub-sample scale. To
        abort downsampling in case of changing endtimes set
        ``strict_length=True``.

        This operation is performed in place on the actual data arrays. The raw
        data is not accessible anymore afterwards.
        To keep your original data, use :meth:`~obspy.core.stream.Stream.copy`
        to make a copy of your trace.

        This also makes an entry with information on the applied processing
        in ``stats.processing`` of every trace.

        For the example we switch off the automatic pre-filtering so that
        the effect of the downsampling routine becomes clearer:

        .. rubric:: Example

        >>> from obspy.core import Trace, Stream
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
        array([0, 4, 8])
        """
        for tr in self:
            tr.downsample(decimation_factor=decimation_factor,
                    no_filter=no_filter, strict_length=strict_length)
        return

    def max(self):
        """
        Method to get the values of the absolute maximum amplitudes of all
        traces in the stream. See :meth:`~obspy.core.trace.Trace.max`.

        :return: List of values of absolute maxima of all traces

        .. rubric:: Example

        >>> from obspy.core import Trace, Stream
        >>> tr1 = Trace(data=np.array([0, -3, 9, 6, 4]))
        >>> tr2 = Trace(data=np.array([0, -3, -9, 6, 4]))
        >>> tr3 = Trace(data=np.array([0.3, -3.5, 9.0, 6.4, 4.3]))
        >>> st = Stream(traces=[tr1, tr2, tr3])
        >>> st.max()
        [9, -9, 9.0]
        """
        return [tr.max() for tr in self]

    def std(self):
        """
        Method to get the standard deviations of amplitudes in all trace in the
        stream.

        Standard deviations are calculated by NumPy method
        :meth:`~numpy.ndarray.std` on ``trace.data`` of every trace in the
        stream.

        :return: List of standard deviations of all traces.

        .. rubric:: Example

        >>> from obspy.core import Trace, Stream
        >>> tr1 = Trace(data=np.array([0, -3, 9, 6, 4]))
        >>> tr2 = Trace(data=np.array([0.3, -3.5, 9.0, 6.4, 4.3]))
        >>> st = Stream(traces=[tr1, tr2])
        >>> st.std()
        [4.2614551505325036, 4.4348618918744247]
        """
        return [tr.std() for tr in self]

    def normalize(self, global_max=False):
        """
        Normalizes all trace in the stream. By default all traces are
        normalized separately to their respective absolute maximum. By setting
        ``global_max=True`` all traces get normalized to the global maximum of
        all traces.

        :param global_max: If set to ``True``, all traces are normalized with
                respect to the global maximum of all traces in the stream
                instead of normalizing every trace separately.

        This operation is performed in place on the actual data arrays. The raw
        data is not accessible anymore afterwards.
        To keep your original data, use :meth:`~obspy.core.stream.Stream.copy`
        to make a copy of your trace.
        This also makes an entry with information on the applied processing
        in ``stats.processing`` of every trace.

        Note: If ``data.dtype`` of a trace was integer it is changing to float.

        .. rubric:: Example

        Make a Stream with two Traces:

        >>> from obspy.core import Trace, Stream
        >>> tr1 = Trace(data=np.array([0, -3, 9, 6, 4]))
        >>> tr2 = Trace(data=np.array([0.3, -0.5, -0.8, 0.4, 0.3]))
        >>> st = Stream(traces=[tr1, tr2])

        All traces are normalized to their absolute maximum and processing
        information is added:

        >>> st.normalize()
        >>> st[0].data  # doctest: +ELLIPSIS
        array([ 0.        , -0.33333333,  1.        ,  0.66666667,  ...])
        >>> st[0].stats.processing
        ['normalize:9']
        >>> st[1].data
        array([ 0.375, -0.625, -1.   ,  0.5  ,  0.375])
        >>> st[1].stats.processing
        ['normalize:-0.8']

        Now let's do it again normalize all traces to the stream's global
        maximum:

        >>> tr1 = Trace(data=np.array([0, -3, 9, 6, 4]))
        >>> tr2 = Trace(data=np.array([0.3, -0.5, -0.8, 0.4, 0.3]))
        >>> st = Stream(traces=[tr1, tr2])

        >>> st.normalize(global_max=True)
        >>> st[0].data  # doctest: +ELLIPSIS
        array([ 0.        , -0.33333333,  1.        ,  0.66666667,  ...])
        >>> st[0].stats.processing
        ['normalize:9']
        >>> st[1].data  # doctest: +ELLIPSIS
        array([ 0.03333333, -0.05555556, -0.08888889,  0.04444444,  ...])
        >>> st[1].stats.processing
        ['normalize:9']
        """
        # use the same value for normalization on all traces?
        if global_max:
            norm = max([abs(value) for value in self.max()])
        else:
            norm = None
        # normalize all traces
        for tr in self:
            tr.normalize(norm=norm)
        return

    def copy(self):
        """
        Returns a deepcopy of the stream.

        :rtype: :class:`~obspy.core.stream.Stream`
        :return: Copy of current stream.

        .. rubric:: Examples

        1. Make a Trace and copy it

            >>> from obspy.core import read
            >>> st = read()
            >>> st2 = st.copy()

           The two objects are not the same:

            >>> st2 is st
            False

           But they have equal data (before applying further processing):

            >>> st2 == st
            True

        2. The following example shows how to make an alias but not copy the
           data. Any changes on ``st3`` would also change the contents of
           ``st``.

            >>> st3 = st
            >>> st3 is st
            True
            >>> st3 == st
            True
        """
        return copy.deepcopy(self)

    def clear(self):
        """
        Clear trace list (convenient method).

        Replaces Stream's trace list by an empty one creating an empty
        Stream object. Useful if there are references to the current
        Stream object that should not break. Otherwise simply use a new
        Stream() instance.
        """
        self.traces = []

    def _cleanup(self):
        """
        Merge consistent trace objects but leave everything else alone.

        This can mean traces with matching header that are directly adjacent or
        are contained/equal/overlapping traces with exactly the same waveform
        data in the overlapping part.

        .. rubric:: Notes

        Traces with overlapping data parts that do not match are not merged::

            before:
            Trace 1: AAAAAAAA
            Trace 2:     BBBBBBBB

            after:
            Trace 1: AAAAAAAA
            Trace 2:     BBBBBBBB

        Traces with overlapping data parts that do match are merged::

            before:
            Trace 1: AAAAAAAA
            Trace 2:     AAAABBBB

            after:
            Trace 1: AAAAAAAABBBB

        Contained traces are handled the same way.
        If common data does not match, nothing is done::

            before:
            Trace 1: AAAAAAAAAAAA
            Trace 2:     BBBB

            after:
            Trace 1: AAAAAAAAAAAA
            Trace 2:     BBBB

        If the common data part matches they are merged::

            before:
            Trace 1: AAAAAAAAAAAA
            Trace 2:     AAAA

            after:
            Trace 1: AAAAAAAAAAAA

        Directly adjacent traces are merged::

            before:
            Trace 1: AAAAAAA
            Trace 2:        BBBBB

            after:
            Trace 1: AAAAAAABBBBB
        """
        # check sampling rates and dtypes
        try:
            self._mergeChecks()
        except Exception, e:
            if "Can't merge traces with same ids but" in str(e):
                msg = "Incompatible traces (sampling_rate, dtype, ...) " + \
                      "with same id detected. Doing nothing."
                warnings.warn(msg)
                return
        # order matters!
        self.sort(keys=['network', 'station', 'location', 'channel',
                        'starttime', 'endtime'])
        # build up dictionary with lists of traces with same ids
        traces_dict = {}
        # using pop() and try-except saves memory
        try:
            while True:
                trace = self.traces.pop(0)
                id = trace.getId()
                # add trace to respective list or create that list
                try:
                    traces_dict[id].append(trace)
                except:
                    traces_dict[id] = [trace]
        except IndexError:
            pass
        # clear traces of current stream
        self.traces = []
        # loop through ids
        for id in traces_dict.keys():
            trace_list = traces_dict[id]
            cur_trace = trace_list.pop(0)
            # work through all traces of same id
            while trace_list:
                trace = trace_list.pop(0)
                # we have some common parts: check if consistent
                if trace.stats.starttime <= cur_trace.stats.endtime:
                    # check if common time slice [t1 --> t2] is equal:
                    t1 = trace.stats.starttime
                    t2 = min(cur_trace.stats.endtime, trace.stats.endtime)
                    # if consistent: add them together
                    if cur_trace.slice(t1, t2) == trace.slice(t1, t2):
                        cur_trace += trace
                    # if not consistent: leave them alone
                    else:
                        self.traces.append(cur_trace)
                        cur_trace = trace
                # traces are perfectly adjacent: add them together
                elif trace.stats.starttime == cur_trace.stats.endtime + \
                     cur_trace.stats.delta:
                    cur_trace += trace
                # no common parts (gap):
                # leave traces alone and add current to list
                else:
                    self.traces.append(cur_trace)
                    cur_trace = trace
            self.traces.append(cur_trace)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
