# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Stream objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str
from future import standard_library
with standard_library.hooks():
    import urllib.request

from glob import glob, has_magic
from obspy.core import compatibility
from obspy.core.trace import Trace
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.core.util.decorator import map_example_filename
from obspy.core.util.base import ENTRY_POINTS, _readFromPlugin, \
    _getFunctionFromEntryPoint
from obspy.core.util.decorator import uncompressFile, raiseIfMasked
from pkg_resources import load_entry_point
import pickle
import copy
import fnmatch
import math
import numpy as np
import os
import warnings


@map_example_filename("pathname_or_url")
def read(pathname_or_url=None, format=None, headonly=False, starttime=None,
         endtime=None, nearest_sample=True, dtype=None, apply_calib=False,
         **kwargs):
    """
    Read waveform files into an ObsPy Stream object.

    The :func:`~obspy.core.stream.read` function opens either one or multiple
    waveform files given via file name or URL using the ``pathname_or_url``
    attribute.

    The format of the waveform file will be automatically detected if not
    given. See the `Supported Formats`_ section below for available formats.

    This function returns an ObsPy :class:`~obspy.core.stream.Stream` object, a
    ``list``-like object of multiple ObsPy :class:`~obspy.core.trace.Trace`
    objects.

    :type pathname_or_url: str or io.BytesIO, optional
    :param pathname_or_url: String containing a file name or a URL or a open
        file-like object. Wildcards are allowed for a file name. If this
        attribute is omitted, an example :class:`~obspy.core.stream.Stream`
        object will be returned.
    :type format: str, optional
    :param format: Format of the file to read (e.g. ``"MSEED"``). See
        the `Supported Formats`_ section below for a list of supported formats.
        If format is set to ``None`` it will be automatically detected which
        results in a slightly slower reading. If a format is specified, no
        further format checking is done.
    :type headonly: bool, optional
    :param headonly: If set to ``True``, read only the data header. This is
        most useful for scanning available meta information of huge data sets.
    :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
    :param starttime: Specify the start time to read.
    :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
    :param endtime: Specify the end time to read.
    :type nearest_sample: bool, optional
    :param nearest_sample: Only applied if `starttime` or `endtime` is given.
        Select nearest sample or the one containing the specified time. For
        more info, see :meth:`~obspy.core.trace.Trace.trim`.
    :type dtype: :class:`numpy.dtype`, optional
    :param dtype: Convert data of all traces into given numpy.dtype.
    :type apply_calib: bool, optional
    :param apply_calib: Automatically applies the calibration factor
        ``trace.stats.calib`` for each trace, if set. Defaults to ``False``.
    :param kwargs: Additional keyword arguments passed to the underlying
        waveform reader method.
    :return: An ObsPy :class:`~obspy.core.stream.Stream` object.

    .. rubric:: Basic Usage

    In most cases a filename is specified as the only argument to
    :func:`~obspy.core.stream.read`. For a quick start you may omit all
    arguments and ObsPy will create and return a basic example seismogram.
    Further usages of the :func:`~obspy.core.stream.read` function can
    be seen in the `Further Examples`_ section underneath.

    >>> from obspy import read
    >>> st = read()
    >>> print(st)  # doctest: +ELLIPSIS
    3 Trace(s) in Stream:
    BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
    BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
    BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples

    .. rubric:: _`Supported Formats`

    Additional ObsPy modules extend the functionality of the
    :func:`~obspy.core.stream.read` function. The following table summarizes
    all known file formats currently supported by ObsPy. The table order also
    reflects the order of the autodetection routine if no format option is
    specified.

    Please refer to the `Linked Function Call`_ of each module for any extra
    options available at the import stage.

    %s

    Next to the :func:`~obspy.core.stream.read` function the
    :meth:`~obspy.core.stream.Stream.write` method of the returned
    :class:`~obspy.core.stream.Stream` object can be used to export the data
    to the file system.

    .. rubric:: _`Further Examples`

    Example waveform files may be retrieved via http://examples.obspy.org.

    (1) Reading multiple local files using wildcards.

        The following code uses wildcards, in this case it matches two files.
        Both files are then read into a single
        :class:`~obspy.core.stream.Stream` object.

        >>> from obspy import read  # doctest: +SKIP
        >>> st = read("/path/to/loc_R*.z")  # doctest: +SKIP
        >>> print(st)  # doctest: +SKIP
        2 Trace(s) in Stream:
        .RJOB..Z | 2005-08-31T02:33:49.850000Z - ... | 200.0 Hz, 12000 samples
        .RNON..Z | 2004-06-09T20:05:59.850000Z - ... | 200.0 Hz, 12000 samples

    (2) Reading a local file without format detection.

        Using the ``format`` parameter disables the automatic detection and
        enforces reading a file in a given format.

        >>> from obspy import read
        >>> st = read("/path/to/loc_RJOB20050831023349.z", format="GSE2")
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        .RJOB..Z | 2005-08-31T02:33:49.850000Z - ... | 200.0 Hz, 12000 samples

    (3) Reading a remote file via HTTP protocol.

        >>> from obspy import read
        >>> st = read("http://examples.obspy.org/loc_RJOB20050831023349.z")
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        .RJOB..Z | 2005-08-31T02:33:49.850000Z - ... | 200.0 Hz, 12000 samples

    (4) Reading a compressed files.

        >>> from obspy import read
        >>> st = read("/path/to/tspair.ascii.gz")
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        XX.TEST..BHZ | 2008-01-15T00:00:00.025000Z - ... | 40.0 Hz, 635 samples

        >>> st = read("http://examples.obspy.org/slist.ascii.bz2")
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        XX.TEST..BHZ | 2008-01-15T00:00:00.025000Z - ... | 40.0 Hz, 635 samples

    (5) Reading a file-like object.

        >>> from future import standard_library
        >>> with standard_library.hooks(): from urllib import request
        >>> import io
        >>> example_url = "http://examples.obspy.org/loc_RJOB20050831023349.z"
        >>> stringio_obj = io.BytesIO(request.urlopen(example_url).read())
        >>> st = read(stringio_obj)
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        .RJOB..Z | 2005-08-31T02:33:49.850000Z - ... | 200.0 Hz, 12000 samples

    (6) Using 'starttime' and 'endtime' parameters

        >>> from obspy import read
        >>> dt = UTCDateTime("2005-08-31T02:34:00")
        >>> st = read("http://examples.obspy.org/loc_RJOB20050831023349.z",
        ...           starttime=dt, endtime=dt+10)
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        .RJOB..Z | 2005-08-31T02:34:00.000000Z - ... | 200.0 Hz, 2001 samples
    """
    # add default parameters to kwargs so sub-modules may handle them
    kwargs['starttime'] = starttime
    kwargs['endtime'] = endtime
    kwargs['nearest_sample'] = nearest_sample
    # create stream
    st = Stream()
    if pathname_or_url is None:
        # if no pathname or URL specified, return example stream
        st = _createExampleStream(headonly=headonly)
    elif not isinstance(pathname_or_url, (str, native_str)):
        # not a string - we assume a file-like object
        pathname_or_url.seek(0)
        try:
            # first try reading directly
            stream = _read(pathname_or_url, format, headonly, **kwargs)
            st.extend(stream.traces)
        except TypeError:
            # if this fails, create a temporary file which is read directly
            # from the file system
            pathname_or_url.seek(0)
            with NamedTemporaryFile() as fh:
                fh.write(pathname_or_url.read())
                st.extend(_read(fh.name, format, headonly, **kwargs).traces)
        pathname_or_url.seek(0)
    elif "://" in pathname_or_url:
        # some URL
        # extract extension if any
        suffix = os.path.basename(pathname_or_url).partition('.')[2] or '.tmp'
        with NamedTemporaryFile(suffix=suffix) as fh:
            fh.write(urllib.request.urlopen(pathname_or_url).read())
            st.extend(_read(fh.name, format, headonly, **kwargs).traces)
    else:
        # some file name
        pathname = pathname_or_url
        for file in sorted(glob(pathname)):
            st.extend(_read(file, format, headonly, **kwargs).traces)
        if len(st) == 0:
            # try to give more specific information why the stream is empty
            if has_magic(pathname) and not glob(pathname):
                raise Exception("No file matching file pattern: %s" % pathname)
            elif not has_magic(pathname) and not os.path.isfile(pathname):
                raise IOError(2, "No such file or directory", pathname)
            # Only raise error if no start/end time has been set. This
            # will return an empty stream if the user chose a time window with
            # no data in it.
            # XXX: Might cause problems if the data is faulty and the user
            # set start/end time. Not sure what to do in this case.
            elif not starttime and not endtime:
                raise Exception("Cannot open file/files: %s" % pathname)
    # Trim if times are given.
    if headonly and (starttime or endtime or dtype):
        msg = "Keyword headonly cannot be combined with starttime, endtime" + \
            " or dtype."
        warnings.warn(msg, UserWarning)
        return st
    if starttime:
        st._ltrim(starttime, nearest_sample=nearest_sample)
    if endtime:
        st._rtrim(endtime, nearest_sample=nearest_sample)
    # convert to dtype if given
    if dtype:
        # For compatibility with NumPy 1.4
        if isinstance(dtype, str):
            dtype = native_str(dtype)
        for tr in st:
            tr.data = np.require(tr.data, dtype)
    # applies calibration factor
    if apply_calib:
        for tr in st:
            tr.data = tr.data * tr.stats.calib
    return st


@uncompressFile
def _read(filename, format=None, headonly=False, **kwargs):
    """
    Reads a single file into a ObsPy Stream object.
    """
    stream, format = _readFromPlugin('waveform', filename, format=format,
                                     headonly=headonly, **kwargs)
    # set _format identifier for each element
    for trace in stream:
        trace.stats._format = format
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
                  'calib': 1.0,
                  'back_azimuth': 100.0,
                  'inclination': 30.0}
        header['channel'] = channel
        if not headonly:
            st.append(Trace(data=data[channel], header=header))
        else:
            st.append(Trace(header=header))
    from obspy.station import read_inventory
    st.attach_response(read_inventory("/path/to/BW_RJOB.xml"))
    return st


class Stream(object):
    """
    List like object of multiple ObsPy Trace objects.

    :type traces: list of :class:`~obspy.core.trace.Trace`, optional
    :param traces: Initial list of ObsPy :class:`~obspy.core.trace.Trace`
        objects.

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
        if isinstance(traces, Trace):
            traces = [traces]
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

        .. rubric:: Examples

        1. Adding two Streams

            >>> st1 = Stream([Trace(), Trace(), Trace()])
            >>> len(st1)
            3
            >>> st2 = Stream([Trace(), Trace()])
            >>> len(st2)
            2
            >>> stream = st1 + st2
            >>> len(stream)
            5

        2. Adding Stream and Trace

            >>> stream2 = st1 + Trace()
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

    def __mul__(self, num):
        """
        Creates a new Stream containing num copies of this stream.

        :rtype num: int
        :param num: Number of copies.
        :returns: New ObsPy Stream object.

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> len(st)
        3
        >>> st2 = st * 5
        >>> len(st2)
        15
        """
        if not isinstance(num, int):
            raise TypeError("Integer expected")
        from obspy import Stream
        st = Stream()
        for _i in range(num):
            st += self.copy()
        return st

    def __iter__(self):
        """
        Return a robust iterator for stream.traces.

        Doing this it is safe to remove traces from streams inside of
        for-loops using stream's :meth:`~obspy.core.stream.Stream.remove`
        method. Actually this creates a new iterator every time a trace is
        removed inside the for-loop.

        .. rubric:: Example

        >>> from obspy import Stream
        >>> st = Stream()
        >>> for component in ["1", "Z", "2", "3", "Z", "N", "E", "4", "5"]:
        ...     channel = "EH" + component
        ...     tr = Trace(header={'station': 'TEST', 'channel': channel})
        ...     st.append(tr)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
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
        ...         st.remove(tr)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> print(st)  # doctest: +ELLIPSIS
        4 Trace(s) in Stream:
        .TEST..EHZ | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EHZ | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EHN | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        .TEST..EHE | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples
        """
        return list(self.traces).__iter__()

    def __nonzero__(self):
        """
        A Stream is considered zero if has no Traces.
        """
        return bool(len(self.traces))

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
        if self.traces:
            id_length = self and max(len(tr.id) for tr in self) or 0
        else:
            id_length = 0
        out = str(len(self.traces)) + ' Trace(s) in Stream:\n'
        if len(self.traces) <= 20 or extended is True:
            out = out + "\n".join([_i.__str__(id_length) for _i in self])
        else:
            out = out + "\n" + self.traces[0].__str__() + "\n" + \
                '...\n(%i other traces)\n...\n' % (len(self.traces) - 2) + \
                self.traces[-1].__str__() + '\n\n[Use "print(' + \
                'Stream.__str__(extended=True))" to print all Traces]'
        return out

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__(extended=p.verbose))

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

        >>> from obspy import read
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
        if self_sorted.traces != other_sorted.traces:
            return False

        return True

    def __ne__(self, other):
        """
        Implements rich comparison of Stream objects for "!=" operator.

        .. rubric:: Example

        >>> from obspy import read
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

    def __setitem__(self, index, trace):
        """
        __setitem__ method of obspy.Stream objects.
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

        >>> from obspy import read, Trace
        >>> st = read()
        >>> tr = Trace()
        >>> tr.stats.station = 'TEST'
        >>> st.append(tr)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
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
        return self

    def extend(self, trace_list):
        """
        Extends the current Stream object with a list of Trace objects.

        :param trace_list: list of :class:`~obspy.core.trace.Trace` objects or
            :class:`~obspy.core.stream.Stream`.

        .. rubric:: Example

        >>> from obspy import read, Trace
        >>> st = read()
        >>> tr1 = Trace()
        >>> tr1.stats.station = 'TEST1'
        >>> tr2 = Trace()
        >>> tr2.stats.station = 'TEST2'
        >>> st.extend([tr1, tr2])  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
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
            self.traces.extend(trace_list.traces)
        else:
            msg = 'Extend only supports a list of Trace objects as argument.'
            raise TypeError(msg)
        return self

    def getGaps(self, min_gap=None, max_gap=None):
        """
        Returns a list of all trace gaps/overlaps of the Stream object.

        :param min_gap: All gaps smaller than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.
        :param max_gap: All gaps larger than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.

        The returned list contains one item in the following form for each gap/
        overlap: [network, station, location, channel, starttime of the gap,
        end time of the gap, duration of the gap, number of missing samples]

        Please be aware that no sorting and checking of stations, channels, ...
        is done. This method only compares the start and end times of the
        Traces.

        .. rubric:: Example

        Our example stream has no gaps:

        >>> from obspy import read, UTCDateTime
        >>> st = read()
        >>> st.getGaps()
        []
        >>> st.printGaps()  # doctest: +ELLIPSIS
        Source            Last Sample                 ...
        Total: 0 gap(s) and 0 overlap(s)

        So let's make a copy of the first trace and cut both so that we end up
        with a gappy stream:

        >>> tr = st[0].copy()
        >>> t = UTCDateTime("2009-08-24T00:20:13.0")
        >>> st[0].trim(endtime=t)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.trim(starttime=t + 1)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> st.append(tr)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.getGaps()[0]  # doctest: +SKIP
        [['BW', 'RJOB', '', 'EHZ', UTCDateTime(2009, 8, 24, 0, 20, 13),
          UTCDateTime(2009, 8, 24, 0, 20, 14), 1.0, 99]]
        >>> st.printGaps()  # doctest: +ELLIPSIS
        Source            Last Sample                 ...
        BW.RJOB..EHZ      2009-08-24T00:20:13.000000Z ...
        Total: 1 gap(s) and 0 overlap(s)
        """
        # Create shallow copy of the traces to be able to sort them later on.
        copied_traces = copy.copy(self.traces)
        self.sort()
        gap_list = []
        for _i in range(len(self.traces) - 1):
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
            nsamples = int(compatibility.round_away(math.fabs(delta) *
                                                    stats['sampling_rate']))
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
            for _i in range(len(object)):
                self.traces.insert(position + _i, object[_i])
        elif isinstance(object, Stream):
            self.insert(position, object.traces)
        else:
            msg = 'Only accepts a Trace object or a list of Trace objects.'
            raise TypeError(msg)
        return self

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
        :param starttime: Start time of the graph as a
            :class:`~obspy.core.utcdatetime.UTCDateTime` object. If not set
            the graph will be plotted from the beginning.
            Defaults to ``None``.
        :param endtime: End time of the graph as a
            :class:`~obspy.core.utcdatetime.UTCDateTime` object. If not set
            the graph will be plotted until the end.
            Defaults to ``None``.
        :param fig: Use an existing matplotlib figure instance.
            Defaults to ``None``.
        :param automerge: If automerge is True, Traces with the same id will be
            merged.
            Defaults to ``True``.
        :param size: Size tuple in pixel for the output file. This corresponds
            to the resolution of the graph for vector formats.
            Defaults to ``(800, 250)`` pixel per channel for ``type='normal'``
            or ``type='relative'``, ``(800, 600)`` for ``type='dayplot'``, and
            ``(1000, 600)`` for ``type='section'``.
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
        :param face_color: Face color of the matplotlib canvas.
            Defaults to ``'white'``.
        :param transparent: Make all backgrounds transparent (True/False). This
            will override the ``bgcolor`` and ``face_color`` arguments.
            Defaults to ``False``.
        :param number_of_ticks: The number of ticks on the x-axis.
            Defaults to ``4``.
        :param tick_format: The way the time axis is formatted.
            Defaults to ``'%H:%M:%S'`` or ``'%.2f'`` if ``type='relative'``.
        :param tick_rotation: Tick rotation in degrees.
            Defaults to ``0``.
        :param handle: Whether or not to return the matplotlib figure instance
            after the plot has been created.
            Defaults to ``False``.
        :param method: By default, all traces with more than 400,000 samples
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
        :param equal_scale: If enabled all plots are equally scaled.
            Defaults to ``True``.
        :param show: If True, show the plot interactively after plotting. This
            is ignored if any of ``outfile``, ``format``, ``handle``, or
            ``fig`` are specified.
            Defaults to ``True``.
        :param draw: If True, the figure canvas is explicitly re-drawn, which
            ensures that *existing* figures are fresh. It makes no difference
            for figures that are not yet visible.
            Defaults to ``True``.
        :param block: If True block call to showing plot. Only works if the
            active matplotlib backend supports it.
            Defaults to ``True``.
        :param linewidth: Float value in points of the line width.
            Defaults to ``1.0``.
        :param linestyle: Line style.
            Defaults to ``'-'``
        :param grid_color: Color of the grid.
            Defaults to ``'black'``.
        :param grid_linewidth: Float value in points of the grid line width.
            Defaults to ``0.5``.
        :param grid_linestyle: Grid line style.
            Defaults to ``':'``

        **Dayplot Parameters**

        The following parameters are only available if ``type='dayplot'`` is
        set.

        :param vertical_scaling_range: Determines how each line is scaled in
            its given space. Every line will be centered around its mean value
            and then clamped to fit its given space. This argument is the range
            in data units that will be used to clamp the data. If the range is
            smaller than the actual range, the lines' data may overshoot to
            other lines which is usually a desired effect. Larger ranges will
            result in a vertical padding.
            If ``0``, the actual range of the data will be used and no
            overshooting or additional padding will occur.
            If ``None`` the range will be chosen to be the 99.5-percentile of
            the actual range - so some values will overshoot.
            Defaults to ``None``.
        :param interval: This defines the interval length in minutes for one
            line.
            Defaults to ``15``.
        :param time_offset: Only used if ``type='dayplot'``. The difference
            between the timezone of the data (specified with the kwarg
            ``timezone``) and UTC time in hours. Will be displayed in a string.
            Defaults to the current offset of the system time to UTC time.
        :param timezone: Defines the name of the user defined time scale. Will
            be displayed in a string together with the actual offset defined in
            the kwarg ``time_offset``.
            Defaults to ``'local time'``.
        :param localization_dict: Enables limited localization of the dayplot
            through the usage of a dictionary. To change the labels to, e.g.
            German, use the following::

                localization_dict={'time in': 'Zeit in', 'seconds': 'Sekunden',
                                   'minutes': 'Minuten', 'hours': 'Stunden'}

        :param data_unit: If given, the scale of the data will be drawn on the
            right hand side in the form ``"%f {data_unit}"``. The unit is
            supposed to be a string containing the actual unit of the data. Can
            be a LaTeX expression if matplotlib has been built with LaTeX
            support, e.g., ``"$\\\\frac{m}{s}$"``. Be careful to escape the
            backslashes, or use r-prefixed strings, e.g.,
            ``r"$\\\\frac{m}{s}$"``.
            Defaults to ``None``, meaning no scale is drawn.
        :param events: An optional list of events can be drawn on the plot if
            given.  They will be displayed as yellow stars with optional
            annotations.  They are given as a list of dictionaries. Each
            dictionary at least needs to have a "time" key, containing a
            UTCDateTime object with the origin time of the event. Furthermore
            every event can have an optional "text" key which will then be
            displayed as an annotation.
            Example::

                events=[{"time": UTCDateTime(...), "text": "Event A"}, {...}]

            It can also be a :class:`~obspy.core.event.Catalog` object. In this
            case each event will be annotated with its corresponding
            Flinn-Engdahl region and the magnitude.
            Events can also be automatically downloaded with the help of
            obspy.neries. Just pass a dictionary with a "min_magnitude" key,
            e.g. ::

                events={"min_magnitude": 5.5}

            Defaults to ``[]``.
        :param x_labels_size: Size of x labels in points or fontsize.
            Defaults to ``8``.
        :param y_labels_size: Size of y labels in points or fontsize.
            Defaults to ``8``.
        :param title_size: Size of the title in points or fontsize.
            Defaults to ``10``.
        :param subplots_adjust_left: The left side of the subplots of the
            figure in fraction of the figure width.
            Defaults to ``0.12``.
        :param subplots_adjust_right: The right side of the subplots of the
            figure in fraction of the figure width.
            Defaults to ``0.88``.
        :param subplots_adjust_top: The top side of the subplots of the figure
            in fraction of the figure width.
            Defaults to ``0.95``.
        :param subplots_adjust_bottom: The bottom side of the subplots of the
            figure in fraction of the figure width.
            Defaults to ``0.1``.
        :param right_vertical_labels: Whether or not to display labels on the
            right side of the dayplot.
            Defaults to ``False``.
        :param one_tick_per_line: Whether or not to display one tick per line.
            Defaults to ``False``.
        :param show_y_UTC_label: Whether or not to display the Y UTC vertical
            label.
            Defaults to ``True``.
        :param title: The title to display on top of the plot.
            Defaults to ``self.stream[0].id``.

        **Section Parameters**

        These parameters are only available if ``type='section'`` is set. To
        plot a record section the ObsPy header ``trace.stats.distance`` must be
        defined in meters (Default). Or ``trace.stats.coordinates.latitude`` &
        ``trace.stats.coordinates.longitude`` must be set if plotted in
        azimuthal distances (``dist_degree=True``) along with ``ev_coord``.

        :type scale: float, optional
        :param scale: Scale the traces width with this factor.
            Defaults to ``1.0``.
        :type vred: float, optional
        :param vred: Perform velocity reduction, in m/s.
        :type norm_method: str, optional
        :param norm_method: Defines how the traces are normalized, either
            against each ``trace`` or against the global maximum ``stream``.
            Defaults to ``trace``.
        :type offset_min: float or None, optional
        :param offset_min: Minimum offset in meters to plot.
            Defaults to minimum offset of all traces.
        :type offset_max: float or None, optional
        :param offset_max: Maximum offset in meters to plot.
            Defaults to maximum offset of all traces.
        :type dist_degree: bool, optional
        :param dist_degree: Plot trace distance in degree from epicenter. If
            ``True``, parameter ``ev_coord`` has to be defined.
            Defaults to ``False``.
        :type ev_coord: tuple or None, optional
        :param ev_coord: Event's coordinates as tuple
            ``(latitude, longitude)``.
        :type plot_dx: int, optional
        :param plot_dx: Spacing of ticks on the spatial x-axis.
            Either km or degree, depending on ``dist_degree``.
        :type recordstart: int or float, optional
        :param recordstart: Seconds to crop from the beginning.
        :type recordlength: int or float, optional
        :param recordlength: Length of the record section in seconds.
        :type alpha: float, optional
        :param alpha: Transparency of the traces between 0.0 - 1.0.
            Defaults to ``0.5``.
        :type time_down: bool, optional
        :param time_down: Flip the plot horizontally, time goes down.
            Defaults to ``False``, i.e., time goes up.
        :type reftime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param reftime: The reference time to which the time scale will refer.
            Defaults to the minimum start time of the visible traces.

        **Relative Parameters**

        The following parameters are only available if ``type='relative'`` is
        set.

        :type reftime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param reftime: The reference time to which the relative scale will
            refer.
            Defaults to ``starttime``.

        .. rubric:: Color Options

        Colors can be specified as defined in the :mod:`matplotlib.colors`
        documentation.

        Short Version: For all color values, you can either use:

        * legal `HTML color names <http://www.w3.org/TR/css3-color/#html4>`_,
          e.g. ``'blue'``,
        * HTML hex strings, e.g. ``'#EE00FF'``,
        * pass an string of a R, G, B tuple, where each of the components is a
          float value in the range of 0 to 1, e.g. ``'(1, 0.25, 0.5)'``, or
        * use single letters for the basic built-in colors, such as ``'b'``
          (blue), ``'g'`` (green), ``'r'`` (red), ``'c'`` (cyan), ``'m'``
          (magenta), ``'y'`` (yellow), ``'k'`` (black), ``'w'`` (white).

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            st.plot()
        """
        from obspy.imaging.waveform import WaveformPlotting
        waveform = WaveformPlotting(stream=self, *args, **kwargs)
        return waveform.plotWaveform(*args, **kwargs)

    def spectrogram(self, **kwargs):
        """
        Creates a spectrogram plot for each trace in the stream.

        For details on kwargs that can be used to customize the spectrogram
        plot see :func:`obspy.imaging.spectrogram.spectrogram`.

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> st[0].spectrogram()  # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            st[0].spectrogram()
        """
        spec_list = []
        for tr in self:
            spec = tr.spectrogram(**kwargs)
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

        >>> from obspy import read
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

    def printGaps(self, min_gap=None, max_gap=None):
        """
        Print gap/overlap list summary information of the Stream object.

        :param min_gap: All gaps smaller than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.
        :param max_gap: All gaps larger than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.

        .. rubric:: Example

        Our example stream has no gaps:

        >>> from obspy import read, UTCDateTime
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
        >>> st[0].trim(endtime=t)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.trim(starttime=t+1)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> st.append(tr)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.getGaps()  # doctest: +ELLIPSIS
        [[..., UTCDateTime(2009, 8, 24, 0, 20, 13), ...
        >>> st.printGaps()  # doctest: +ELLIPSIS
        Source            Last Sample                 ...
        BW.RJOB..EHZ      2009-08-24T00:20:13.000000Z ...
        Total: 1 gap(s) and 0 overlap(s)


        And finally let us create some overlapping traces:

        >>> st = read()
        >>> tr = st[0].copy()
        >>> t = UTCDateTime("2009-08-24T00:20:13.0")
        >>> st[0].trim(endtime=t)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.trim(starttime=t-1)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> st.append(tr)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.getGaps()  # doctest: +ELLIPSIS
        [[...'EHZ', UTCDateTime(2009, 8, 24, 0, 20, 13), ...
        >>> st.printGaps()  # doctest: +ELLIPSIS
        Source            Last Sample                 ...
        BW.RJOB..EHZ      2009-08-24T00:20:13.000000Z ...
        Total: 0 gap(s) and 1 overlap(s)
        """
        result = self.getGaps(min_gap, max_gap)
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

        >>> from obspy import read
        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> for tr in st.select(component="E"):
        ...     st.remove(tr)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> print(st)  # doctest: +ELLIPSIS
        2 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        """
        self.traces.remove(trace)
        return self

    def reverse(self):
        """
        Reverses the Traces of the Stream object in place.

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> st.reverse()  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        """
        self.traces.reverse()
        return self

    def sort(self, keys=['network', 'station', 'location', 'channel',
                         'starttime', 'endtime'], reverse=False):
        """
        Method to sort the traces in the Stream object.

        The traces will be sorted according to the keys list. It will be sorted
        by the first item first, then by the second and so on. It will always
        be sorted from low to high and from A to Z.

        :type keys: list, optional
        :param keys: List containing the values according to which the traces
             will be sorted. They will be sorted by the first item first and
             then by the second item and so on.
             Always available items: 'network', 'station', 'channel',
             'location', 'starttime', 'endtime', 'sampling_rate', 'npts',
             'dataquality'
             Defaults to ['network', 'station', 'location', 'channel',
             'starttime', 'endtime'].
        :type reverse: bool
        :param reverse: Reverts sorting order to descending.

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> st.sort()  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        """
        # check if list
        msg = "keys must be a list of strings. Always available items to " + \
            "sort after: \n'network', 'station', 'channel', 'location', " + \
            "'starttime', 'endtime', 'sampling_rate', 'npts', 'dataquality'"
        if not isinstance(keys, list):
            raise TypeError(msg)
        # Loop over all keys in reversed order.
        for _i in keys[::-1]:
            self.traces.sort(key=lambda x: x.stats[_i], reverse=reverse)
        return self

    def write(self, filename, format, **kwargs):
        """
        Saves stream into a file.

        :type filename: str
        :param filename: The name of the file to write.
        :type format: str
        :param format: The file format to use (e.g. ``"MSEED"``). See
            the `Supported Formats`_ section below for a list of supported
            formats.
        :param kwargs: Additional keyword arguments passed to the underlying
            waveform writer method.

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()  # doctest: +SKIP
        >>> st.write("example.mseed", format="MSEED")  # doctest: +SKIP

        Writing single traces into files with meaningful filenames can be done
        e.g. using trace.id

        >>> for tr in st: #doctest: +SKIP
        ...     tr.write(tr.id + ".MSEED", format="MSEED") #doctest: +SKIP

        .. rubric:: _`Supported Formats`

        Additional ObsPy modules extend the parameters of the
        :meth:`~obspy.core.stream.Stream.write` method. The following
        table summarizes all known formats currently available for ObsPy.

        Please refer to the `Linked Function Call`_ of each module for any
        extra options available.

        %s
        """
        # Check all traces for masked arrays and raise exception.
        for trace in self.traces:
            if isinstance(trace.data, np.ma.masked_array):
                msg = 'Masked array writing is not supported. You can use ' + \
                      'np.array.filled() to convert the masked array to a ' + \
                      'normal array.'
                raise NotImplementedError(msg)
        format = format.upper()
        try:
            # get format specific entry point
            format_ep = ENTRY_POINTS['waveform_write'][format]
            # search writeFormat method for given entry point
            writeFormat = load_entry_point(
                format_ep.dist.key,
                'obspy.plugin.waveform.%s' % (format_ep.name), 'writeFormat')
        except (IndexError, ImportError, KeyError):
            msg = "Writing format \"%s\" is not supported. Supported types: %s"
            raise TypeError(msg % (format,
                                   ', '.join(ENTRY_POINTS['waveform_write'])))
        writeFormat(self, filename, **kwargs)

    def trim(self, starttime=None, endtime=None, pad=False,
             nearest_sample=True, fill_value=None):
        """
        Cuts all traces of this Stream object to given start and end time.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param starttime: Specify the start time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: Specify the end time.
        :type pad: bool, optional
        :param pad: Gives the possibility to trim at time points outside the
            time frame of the original trace, filling the trace with the
            given ``fill_value``. Defaults to ``False``.
        :type nearest_sample: bool, optional
        :param nearest_sample: If set to ``True``, the closest sample is
            selected, if set to ``False``, the next sample containing the time
            is selected. Defaults to ``True``.

            Given the following trace containing 4 samples, "|" are the
            sample points, "A" is the requested starttime::

                |        A|         |         |

            ``nearest_sample=True`` will select the second sample point,
            ``nearest_sample=False`` will select the first sample point.

        :type fill_value: int, float or ``None``, optional
        :param fill_value: Fill value for gaps. Defaults to ``None``. Traces
            will be converted to NumPy masked arrays if no value is given and
            gaps are present.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data will no longer be accessible afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.

        .. rubric:: Example

        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> dt = UTCDateTime("2009-08-24T00:20:20")
        >>> st.trim(dt, dt + 5)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:20.000000Z ... | 100.0 Hz, 501 samples
        BW.RJOB..EHN | 2009-08-24T00:20:20.000000Z ... | 100.0 Hz, 501 samples
        BW.RJOB..EHE | 2009-08-24T00:20:20.000000Z ... | 100.0 Hz, 501 samples
        """
        if not self:
            return
        # select start/end time fitting to a sample point of the first trace
        if nearest_sample:
            tr = self.traces[0]
            if starttime:
                delta = compatibility.round_away(
                    (starttime - tr.stats.starttime) * tr.stats.sampling_rate)
                starttime = tr.stats.starttime + delta * tr.stats.delta
            if endtime:
                delta = compatibility.round_away(
                    (endtime - tr.stats.endtime) * tr.stats.sampling_rate)
                # delta is negative!
                endtime = tr.stats.endtime + delta * tr.stats.delta
        for trace in self.traces:
            trace.trim(starttime, endtime, pad=pad,
                       nearest_sample=nearest_sample, fill_value=fill_value)
        # remove empty traces after trimming
        self.traces = [_i for _i in self.traces if _i.stats.npts]
        return self

    def _ltrim(self, starttime, pad=False, nearest_sample=True):
        """
        Cuts all traces of this Stream object to given start time.
        For more info see :meth:`~obspy.core.trace.Trace._ltrim`.
        """
        for trace in self.traces:
            trace.trim(starttime=starttime, pad=pad,
                       nearest_sample=nearest_sample)
        # remove empty traces after trimming
        self.traces = [tr for tr in self.traces if tr.stats.npts]
        return self

    def _rtrim(self, endtime, pad=False, nearest_sample=True):
        """
        Cuts all traces of this Stream object to given end time.
        For more info see :meth:`~obspy.core.trace.Trace._rtrim`.
        """
        for trace in self.traces:
            trace.trim(endtime=endtime, pad=pad, nearest_sample=nearest_sample)
        # remove empty traces after trimming
        self.traces = [tr for tr in self.traces if tr.stats.npts]
        return self

    def cutout(self, starttime, endtime):
        """
        Cuts the given time range out of all traces of this Stream object.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of time span to remove from stream.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of time span to remove from stream.

        .. rubric:: Example

        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> t1 = UTCDateTime("2009-08-24T00:20:06")
        >>> t2 = UTCDateTime("2009-08-24T00:20:11")
        >>> st.cutout(t1, t2)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> print(st)  # doctest: +ELLIPSIS
        6 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 301 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 301 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 301 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:11.000000Z ... | 100.0 Hz, 2200 samples
        BW.RJOB..EHN | 2009-08-24T00:20:11.000000Z ... | 100.0 Hz, 2200 samples
        BW.RJOB..EHE | 2009-08-24T00:20:11.000000Z ... | 100.0 Hz, 2200 samples
        """
        tmp = self.slice(endtime=starttime, keep_empty_traces=False)
        tmp += self.slice(starttime=endtime, keep_empty_traces=False)
        self.traces = tmp.traces
        return self

    def slice(self, starttime=None, endtime=None, keep_empty_traces=False):
        """
        Returns new Stream object cut to the given start and end time.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Specify the start time of all traces.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Specify the end time of all traces.
        :type keep_empty_traces: bool, optional
        :param keep_empty_traces: Empty traces will be kept if set to ``True``.
            Defaults to ``False``.
        :return: :class:`~obspy.core.stream.Stream`

        .. note::

            The basic idea of :meth:`~obspy.core.stream.Stream.slice`
            is to avoid copying the sample data in memory. So sample data in
            the resulting :class:`~obspy.core.stream.Stream` object contains
            only a reference to the original traces.

        .. rubric:: Example

        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> dt = UTCDateTime("2009-08-24T00:20:20")
        >>> st = st.slice(dt, dt + 5)
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:20.000000Z ... | 100.0 Hz, 501 samples
        BW.RJOB..EHN | 2009-08-24T00:20:20.000000Z ... | 100.0 Hz, 501 samples
        BW.RJOB..EHE | 2009-08-24T00:20:20.000000Z ... | 100.0 Hz, 501 samples
        """
        tmp = copy.copy(self)
        tmp.traces = []
        new = tmp.copy()
        for trace in self:
            sliced_trace = trace.slice(starttime=starttime, endtime=endtime)
            if keep_empty_traces is False and not sliced_trace.stats.npts:
                continue
            new.append(sliced_trace)
        return new

    def select(self, network=None, station=None, location=None, channel=None,
               sampling_rate=None, npts=None, component=None, id=None):
        """
        Returns new Stream object only with these traces that match the given
        stats criteria (e.g. all traces with ``channel="EHZ"``).

        .. rubric:: Examples

        >>> from obspy import read
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
        >>> print(st2)  # doctest: +NORMALIZE_WHITESPACE
        0 Trace(s) in Stream:

        .. warning::
            A new Stream object is returned but the traces it contains are
            just aliases to the traces of the original stream. Does not copy
            the data but only passes a reference.

        All keyword arguments except for ``component`` are tested directly
        against the respective entry in the :class:`~obspy.core.trace.Stats`
        dictionary.

        If a string for ``component`` is given (should be a single letter) it
        is tested against the last letter of the ``Trace.stats.channel`` entry.

        Alternatively, ``channel`` may have the last one or two letters
        wildcarded (e.g. ``channel="EH*"``) to select all components with a
        common band/instrument code.

        All other selection criteria that accept strings (network, station,
        location) may also contain Unix style wildcards (``*``, ``?``, ...).
        """
        # make given component letter uppercase (if e.g. "z" is given)
        if component and channel:
            component = component.upper()
            channel = channel.upper()
            if channel[-1] != "*" and component != channel[-1]:
                msg = "Selection criteria for channel and component are " + \
                      "mutually exclusive!"
                raise ValueError(msg)
        traces = []
        for trace in self:
            # skip trace if any given criterion is not matched
            if id and not fnmatch.fnmatch(trace.id.upper(), id.upper()):
                continue
            if network is not None:
                if not fnmatch.fnmatch(trace.stats.network.upper(),
                                       network.upper()):
                    continue
            if station is not None:
                if not fnmatch.fnmatch(trace.stats.station.upper(),
                                       station.upper()):
                    continue
            if location is not None:
                if not fnmatch.fnmatch(trace.stats.location.upper(),
                                       location.upper()):
                    continue
            if channel is not None:
                if not fnmatch.fnmatch(trace.stats.channel.upper(),
                                       channel.upper()):
                    continue
            if sampling_rate is not None:
                if float(sampling_rate) != trace.stats.sampling_rate:
                    continue
            if npts is not None and int(npts) != trace.stats.npts:
                continue
            if component is not None:
                if len(trace.stats.channel) < 3:
                    continue
                if not fnmatch.fnmatch(trace.stats.channel[-1].upper(),
                                       component.upper()):
                    continue
            traces.append(trace)
        return self.__class__(traces=traces)

    def verify(self):
        """
        Verifies all traces of current Stream against available meta data.

        .. rubric:: Example

        >>> from obspy import Trace, Stream
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
        return self

    def _mergeChecks(self):
        """
        Sanity checks for merging.
        """
        sr = {}
        dtype = {}
        calib = {}
        for trace in self.traces:
            # skip empty traces
            if len(trace) == 0:
                continue
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

        :type method: int, optional
        :param method: Methodology to handle overlaps of traces. Defaults
            to ``0``.
            See :meth:`obspy.core.trace.Trace.__add__` for details on
            methods ``0`` and ``1``,
            see :meth:`obspy.core.stream.Stream._cleanup` for details on
            method ``-1``.
        :type fill_value: int, float, str or ``None``, optional
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
        def listsort(order, current):
            """
            Helper method for keeping trace's ordering
            """
            try:
                return order.index(current)
            except ValueError:
                return -1

        if method == -1:
            self._cleanup()
            return
        # check sampling rates and dtypes
        self._mergeChecks()
        # remember order of traces
        order = [id(i) for i in self.traces]
        # order matters!
        self.sort(keys=['network', 'station', 'location', 'channel',
                        'starttime', 'endtime'])
        # build up dictionary with with lists of traces with same ids
        traces_dict = {}
        # using pop() and try-except saves memory
        try:
            while True:
                trace = self.traces.pop(0)
                # skip empty traces
                if len(trace) == 0:
                    continue
                _id = trace.getId()
                if _id not in traces_dict:
                    traces_dict[_id] = [trace]
                else:
                    traces_dict[_id].append(trace)
        except IndexError:
            pass
        # clear traces of current stream
        self.traces = []
        # loop through ids
        for _id in traces_dict.keys():
            cur_trace = traces_dict[_id].pop(0)
            # loop through traces of same id
            for _i in range(len(traces_dict[_id])):
                trace = traces_dict[_id].pop(0)
                # disable sanity checks because there are already done
                cur_trace = cur_trace.__add__(
                    trace, method, fill_value=fill_value, sanity_checks=False,
                    interpolation_samples=interpolation_samples)
            self.traces.append(cur_trace)

        # trying to restore order, newly created traces are placed at
        # start
        self.traces.sort(key=lambda x: listsort(order, id(x)))
        return self

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

        .. note::

            Instead of the builtin deconvolution based on Poles and Zeros
            information, the deconvolution can be performed using evalresp
            instead by using the option `seedresp` (see documentation of
            :func:`~obspy.signal.invsim.seisSim` and the `ObsPy Tutorial
            <http://docs.obspy.org/master/tutorial/code_snippets/\
seismometer_correction_simulation.html#using-a-resp-file>`_.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.

        .. rubric:: Example

        >>> from obspy import read
        >>> from obspy.signal import cornFreq2Paz
        >>> st = read()
        >>> paz_sts2 = {'poles': [-0.037004+0.037016j, -0.037004-0.037016j,
        ...                       -251.33+0j,
        ...                       -131.04-467.29j, -131.04+467.29j],
        ...             'zeros': [0j, 0j],
        ...             'gain': 60077000.0,
        ...             'sensitivity': 2516778400.0}
        >>> paz_1hz = cornFreq2Paz(1.0, damp=0.707)
        >>> st.simulate(paz_remove=paz_sts2, paz_simulate=paz_1hz)
        ... # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read
            from obspy.signal import cornFreq2Paz
            st = read()
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
        return self

    def filter(self, type, **options):
        """
        Filters the data of all traces in the Stream.

        :type type: str
        :param type: String that specifies which filter is applied (e.g.
            ``"bandpass"``). See the `Supported Filter`_ section below for
            further details.
        :param options: Necessary keyword arguments for the respective filter
            that will be passed on. (e.g. ``freqmin=1.0``, ``freqmax=20.0`` for
            ``"bandpass"``)

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.

        .. rubric:: _`Supported Filter`

        ``'bandpass'``
            Butterworth-Bandpass (uses :func:`obspy.signal.filter.bandpass`).

        ``'bandstop'``
            Butterworth-Bandstop (uses :func:`obspy.signal.filter.bandstop`).

        ``'lowpass'``
            Butterworth-Lowpass (uses :func:`obspy.signal.filter.lowpass`).

        ``'highpass'``
            Butterworth-Highpass (uses :func:`obspy.signal.filter.highpass`).

        ``'lowpassCheby2'``
            Cheby2-Lowpass (uses :func:`obspy.signal.filter.lowpassCheby2`).

        ``'lowpassFIR'`` (experimental)
            FIR-Lowpass (uses :func:`obspy.signal.filter.lowpassFIR`).

        ``'remezFIR'`` (experimental)
            Minimax optimal bandpass using Remez algorithm (uses
            :func:`obspy.signal.filter.remezFIR`).

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> st.filter("highpass", freq=1.0)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            st.filter("highpass", freq=1.0)
            st.plot()
        """
        for tr in self:
            tr.filter(type, **options)
        return self

    def trigger(self, type, **options):
        """
        Runs a triggering algorithm on all traces in the stream.

        :param type: String that specifies which trigger is applied (e.g.
            ``'recstalta'``). See the `Supported Trigger`_ section below for
            further details.
        :param options: Necessary keyword arguments for the respective
            trigger that will be passed on. (e.g. ``sta=3``, ``lta=10``)
            Arguments ``sta`` and ``lta`` (seconds) will be mapped to ``nsta``
            and ``nlta`` (samples) by multiplying with sampling rate of trace.
            (e.g. ``sta=3``, ``lta=10`` would call the trigger with 3 and 10
            seconds average, respectively)

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.

        .. rubric:: _`Supported Trigger`

        ``'classicstalta'``
            Computes the classic STA/LTA characteristic function (uses
            :func:`obspy.signal.trigger.classicSTALTA`).

        ``'recstalta'``
            Recursive STA/LTA (uses :func:`obspy.signal.trigger.recSTALTA`).

        ``'recstaltapy'``
            Recursive STA/LTA written in Python (uses
            :func:`obspy.signal.trigger.recSTALTAPy`).

        ``'delayedstalta'``
            Delayed STA/LTA. (uses :func:`obspy.signal.trigger.delayedSTALTA`).

        ``'carlstatrig'``
            Computes the carlSTATrig characteristic function (uses
            :func:`obspy.signal.trigger.carlSTATrig`).

        ``'zdetect'``
            Z-detector (uses :func:`obspy.signal.trigger.zDetect`).

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> st.filter("highpass", freq=1.0)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.plot()  # doctest: +SKIP
        >>> st.trigger('recstalta', sta=1, lta=4)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            st.filter("highpass", freq=1.0)
            st.plot()
            st.trigger('recstalta', sta=1, lta=4)
            st.plot()
        """
        for tr in self:
            tr.trigger(type, **options)
        return self

    def resample(self, sampling_rate, window='hanning', no_filter=True,
                 strict_length=False):
        """
        Resample data in all traces of stream using Fourier method.

        :type sampling_rate: float
        :param sampling_rate: The sampling rate of the resampled signal.
        :type window: array_like, callable, str, float, or tuple, optional
        :param window: Specifies the window applied to the signal in the
            Fourier domain. Defaults ``'hanning'`` window. See
            :func:`scipy.signal.resample` for details.
        :type no_filter: bool, optional
        :param no_filter: Deactivates automatic filtering if set to ``True``.
            Defaults to ``True``.
        :type strict_length: bool, optional
        :param strict_length: Leave traces unchanged for which end time of
            trace would change. Defaults to ``False``.

        .. note::

            The :class:`~Stream` object has three different methods to change
            the sampling rate of its data: :meth:`~.resample`,
            :meth:`~.decimate`, and :meth:`~.interpolate`

            Make sure to choose the most appropriate one for the problem at
            hand.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.

        Uses :func:`scipy.signal.resample`. Because a Fourier method is used,
        the signal is assumed to be periodic.

        .. rubric:: Example

        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> st.resample(10.0)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 10.0 Hz, 300 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z ... | 10.0 Hz, 300 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z ... | 10.0 Hz, 300 samples
        """
        for tr in self:
            tr.resample(sampling_rate, window=native_str(window),
                        no_filter=no_filter, strict_length=strict_length)
        return self

    def decimate(self, factor, no_filter=False, strict_length=False):
        """
        Downsample data in all traces of stream by an integer factor.

        :type factor: int
        :param factor: Factor by which the sampling rate is lowered by
            decimation.
        :type no_filter: bool, optional
        :param no_filter: Deactivates automatic filtering if set to ``True``.
            Defaults to ``False``.
        :type strict_length: bool, optional
        :param strict_length: Leave traces unchanged for which end time of
            trace would change. Defaults to ``False``.

        Currently a simple integer decimation is implemented.
        Only every decimation_factor-th sample remains in the trace, all other
        samples are thrown away. Prior to decimation a lowpass filter is
        applied to ensure no aliasing artifacts are introduced. The automatic
        filtering can be deactivated with ``no_filter=True``.

        If the length of the data array modulo ``decimation_factor`` is not
        zero then the end time of the trace is changing on sub-sample scale. To
        abort downsampling in case of changing end times set
        ``strict_length=True``.

        .. note::

            The :class:`~Stream` object has three different methods to change
            the sampling rate of its data: :meth:`~.resample`,
            :meth:`~.decimate`, and :meth:`~.interpolate`

            Make sure to choose the most appropriate one for the problem at
            hand.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.

        .. rubric:: Example

        For the example we switch off the automatic pre-filtering so that
        the effect of the downsampling routine becomes clearer.

        >>> from obspy import Trace, Stream
        >>> tr = Trace(data=np.arange(10))
        >>> st = Stream(traces=[tr])
        >>> tr.stats.sampling_rate
        1.0
        >>> tr.data
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> st.decimate(4, strict_length=False, no_filter=True)
        ... # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> tr.stats.sampling_rate
        0.25
        >>> tr.data
        array([0, 4, 8])
        """
        for tr in self:
            tr.decimate(factor, no_filter=no_filter,
                        strict_length=strict_length)
        return self

    def max(self):
        """
        Method to get the values of the absolute maximum amplitudes of all
        traces in the stream. See :meth:`~obspy.core.trace.Trace.max`.

        :return: List of values of absolute maxima of all traces

        .. rubric:: Example

        >>> from obspy import Trace, Stream
        >>> tr1 = Trace(data=np.array([0, -3, 9, 6, 4]))
        >>> tr2 = Trace(data=np.array([0, -3, -9, 6, 4]))
        >>> tr3 = Trace(data=np.array([0.3, -3.5, 9.0, 6.4, 4.3]))
        >>> st = Stream(traces=[tr1, tr2, tr3])
        >>> st.max()
        [9, -9, 9.0]
        """
        return [tr.max() for tr in self]

    def differentiate(self, type='gradient'):
        """
        Method to differentiate all traces with respect to time.

        :type type: str, optional
        :param type: Method to use for differentiation. Defaults to
            ``'gradient'``. See the `Supported Methods`_ section below for
            further details.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.

        .. rubric:: _`Supported Methods`

        ``'gradient'``
            The gradient is computed using central differences in the interior
            and first differences at the boundaries. The returned gradient
            hence has the same shape as the input array. (uses
            :func:`numpy.gradient`)
        """
        for tr in self:
            tr.differentiate(type=type)
        return self

    def integrate(self, **options):
        """
        Integrate all traces with respect to time.

        For details see the corresponding
        :meth:`~obspy.core.trace.Trace.integrate` method of
        :class:`~obspy.core.trace.Trace`.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.
        """
        for tr in self:
            tr.integrate(**options)
        return self

    @raiseIfMasked
    def detrend(self, type='simple'):
        """
        Method to remove a linear trend from all traces.

        :type type: str, optional
        :param type: Method to use for detrending. Defaults to ``'simple'``.
            See the `Supported Methods`_ section below for further details.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.

        .. rubric:: _`Supported Methods`

        ``'simple'``
            Subtracts a linear function defined by first/last sample of the
            trace (uses :func:`obspy.signal.detrend.simple`).

        ``'linear'``
            Fitting a linear function to the trace with least squares and
            subtracting it (uses :func:`scipy.signal.detrend`).

        ``'constant'`` or ``'demean'``
            Mean of data is subtracted (uses :func:`scipy.signal.detrend`).
        """
        for tr in self:
            tr.detrend(type=type)
        return self

    def taper(self, *args, **kwargs):
        """
        Method to taper all Traces in Stream.

        For details see the corresponding :meth:`~obspy.core.trace.Trace.taper`
        method of :class:`~obspy.core.trace.Trace`.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
        """
        for tr in self:
            tr.taper(*args, **kwargs)
        return self

    def interpolate(self, *args, **kwargs):
        """
        Method to interpolate all Traces in a Stream.

        For details see the corresponding
        :meth:`~obspy.core.trace.Trace.interpolate` method of
        :class:`~obspy.core.trace.Trace`.

        .. note::

            The :class:`~Stream` object has three different methods to change
            the sampling rate of its data: :meth:`~.resample`,
            :meth:`~.decimate`, and :meth:`~.interpolate`

            Make sure to choose the most appropriate one for the problem at
            hand.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data will no longer be accessible afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.

        >>> from obspy import read
        >>> st = read()
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03... - ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03... - ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03... - ... | 100.0 Hz, 3000 samples
        >>> st.interpolate(sampling_rate=111.1)  # doctest: +ELLIPSIS
        <obspy.core.stream.Stream object at 0x...>
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03... - ... | 111.1 Hz, 3332 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03... - ... | 111.1 Hz, 3332 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03... - ... | 111.1 Hz, 3332 samples
        """
        for tr in self:
            tr.interpolate(*args, **kwargs)
        return self

    def std(self):
        """
        Method to get the standard deviations of amplitudes in all trace in the
        stream.

        Standard deviations are calculated by NumPy method
        :meth:`~numpy.ndarray.std` on ``trace.data`` of every trace in the
        stream.

        :return: List of standard deviations of all traces.

        .. rubric:: Example

        >>> from obspy import Trace, Stream
        >>> tr1 = Trace(data=np.array([0, -3, 9, 6, 4]))
        >>> tr2 = Trace(data=np.array([0.3, -3.5, 9.0, 6.4, 4.3]))
        >>> st = Stream(traces=[tr1, tr2])
        >>> st.std()
        [4.2614551505325036, 4.4348618918744247]
        """
        return [tr.std() for tr in self]

    def normalize(self, global_max=False):
        """
        Normalizes all trace in the stream.

        By default all traces are normalized separately to their respective
        absolute maximum. By setting ``global_max=True`` all traces get
        normalized to the global maximum of all traces.

        :param global_max: If set to ``True``, all traces are normalized with
                respect to the global maximum of all traces in the stream
                instead of normalizing every trace separately.

        .. note::
            If ``data.dtype`` of a trace was integer it is changing to float.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.

        .. rubric:: Example

        Make a Stream with two Traces:

        >>> from obspy import Trace, Stream
        >>> tr1 = Trace(data=np.array([0, -3, 9, 6, 4]))
        >>> tr2 = Trace(data=np.array([0.3, -0.5, -0.8, 0.4, 0.3]))
        >>> st = Stream(traces=[tr1, tr2])

        All traces are normalized to their absolute maximum and processing
        information is added:

        >>> st.normalize()  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st[0].data  # doctest: +ELLIPSIS
        array([ 0.        , -0.33333333,  1.        ,  0.66666667,  ...])
        >>> print(st[0].stats.processing[0])  # doctest: +ELLIPSIS
        ObsPy ... normalize(norm=None)
        >>> st[1].data
        array([ 0.375, -0.625, -1.   ,  0.5  ,  0.375])
        >>> print(st[1].stats.processing[0])  # doctest: +ELLIPSIS
        ObsPy ...: normalize(norm=None)

        Now let's do it again normalize all traces to the stream's global
        maximum:

        >>> tr1 = Trace(data=np.array([0, -3, 9, 6, 4]))
        >>> tr2 = Trace(data=np.array([0.3, -0.5, -0.8, 0.4, 0.3]))
        >>> st = Stream(traces=[tr1, tr2])

        >>> st.normalize(global_max=True)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st[0].data  # doctest: +ELLIPSIS
        array([ 0.        , -0.33333333,  1.        ,  0.66666667,  ...])
        >>> print(st[0].stats.processing[0])  # doctest: +ELLIPSIS
        ObsPy ...: normalize(norm=9)
        >>> st[1].data  # doctest: +ELLIPSIS
        array([ 0.03333333, -0.05555556, -0.08888889,  0.04444444,  ...])
        >>> print(st[1].stats.processing[0])  # doctest: +ELLIPSIS
        ObsPy ...: normalize(norm=9)
        """
        # use the same value for normalization on all traces?
        if global_max:
            norm = max([abs(value) for value in self.max()])
        else:
            norm = None
        # normalize all traces
        for tr in self:
            tr.normalize(norm=norm)
        return self

    def rotate(self, method, back_azimuth=None, inclination=None):
        """
        Convenience method for rotating stream objects.

        :type method: str
        :param method: Determines the rotation method.

            ``'NE->RT'``: Rotates the North- and East-components of a
                seismogram to radial and transverse components.
            ``'RT->NE'``: Rotates the radial and transverse components of a
                seismogram to North- and East-components.
            ``'ZNE->LQT'``: Rotates from left-handed Z, North, and  East system
                to LQT, e.g. right-handed ray coordinate system.
            ``'LQR->ZNE'``: Rotates from LQT, e.g. right-handed ray coordinate
                system to left handed Z, North, and East system.

        :type back_azimuth: float, optional
        :param back_azimuth: Depends on the chosen method.
            A single float, the back azimuth from station to source in degrees.
            If not given, ``stats.back_azimuth`` will be used. It will also be
            written after the rotation is done.
        :type inclination: float, optional
        :param inclination: Inclination of the ray at the station in degrees.
            Only necessary for three component rotations. If not given,
            ``stats.inclination`` will be used. It will also be written after
            the rotation is done.
        """
        if method == "NE->RT":
            func = "rotate_NE_RT"
        elif method == "RT->NE":
            func = "rotate_RT_NE"
        elif method == "ZNE->LQT":
            func = "rotate_ZNE_LQT"
        elif method == "LQT->ZNE":
            func = "rotate_LQT_ZNE"
        else:
            raise ValueError("Method has to be one of ('NE->RT', 'RT->NE', "
                             "'ZNE->LQT', or 'LQT->ZNE').")
        # Retrieve function call from entry points
        func = _getFunctionFromEntryPoint("rotate", func)
        # Split to get the components. No need for further checks for the
        # method as invalid methods will be caught by previous conditional.
        input_components, output_components = method.split("->")
        # Figure out inclination and back-azimuth.
        if back_azimuth is None:
            try:
                back_azimuth = self[0].stats.back_azimuth
            except:
                msg = "No back-azimuth specified."
                raise TypeError(msg)
        if len(input_components) == 3 and inclination is None:
            try:
                inclination = self[0].stats.inclination
            except:
                msg = "No inclination specified."
                raise TypeError(msg)
        # Do one of the two-component rotations.
        if len(input_components) == 2:
            input_1 = self.select(component=input_components[0])
            input_2 = self.select(component=input_components[1])
            for i_1, i_2 in zip(input_1, input_2):
                dt = 0.5 * i_1.stats.delta
                if (len(i_1) != len(i_2)) or \
                        (abs(i_1.stats.starttime - i_2.stats.starttime) > dt) \
                        or (i_1.stats.sampling_rate !=
                            i_2.stats.sampling_rate):
                    msg = "All components need to have the same time span."
                    raise ValueError(msg)
            for i_1, i_2 in zip(input_1, input_2):
                output_1, output_2 = func(i_1.data, i_2.data, back_azimuth)
                i_1.data = output_1
                i_2.data = output_2
                # Rename the components.
                i_1.stats.channel = i_1.stats.channel[:-1] + \
                    output_components[0]
                i_2.stats.channel = i_2.stats.channel[:-1] + \
                    output_components[1]
                # Add the azimuth and inclination to the stats object.
                for comp in (i_1, i_2):
                    comp.stats.back_azimuth = back_azimuth
        # Do one of the three-component rotations.
        else:
            input_1 = self.select(component=input_components[0])
            input_2 = self.select(component=input_components[1])
            input_3 = self.select(component=input_components[2])
            for i_1, i_2, i_3 in zip(input_1, input_2, input_3):
                dt = 0.5 * i_1.stats.delta
                if (len(i_1) != len(i_2)) or (len(i_1) != len(i_3)) or \
                        (abs(i_1.stats.starttime -
                             i_2.stats.starttime) > dt) or \
                        (abs(i_1.stats.starttime -
                             i_3.stats.starttime) > dt) or \
                        (i_1.stats.sampling_rate !=
                            i_2.stats.sampling_rate) or \
                        (i_1.stats.sampling_rate != i_3.stats.sampling_rate):
                    msg = "All components need to have the same time span."
                    raise ValueError(msg)
            for i_1, i_2, i_3 in zip(input_1, input_2, input_3):
                output_1, output_2, output_3 = func(
                    i_1.data, i_2.data, i_3.data, back_azimuth, inclination)
                i_1.data = output_1
                i_2.data = output_2
                i_3.data = output_3
                # Rename the components.
                i_1.stats.channel = i_1.stats.channel[:-1] + \
                    output_components[0]
                i_2.stats.channel = i_2.stats.channel[:-1] + \
                    output_components[1]
                i_3.stats.channel = i_3.stats.channel[:-1] + \
                    output_components[2]
                # Add the azimuth and inclination to the stats object.
                for comp in (i_1, i_2, i_3):
                    comp.stats.back_azimuth = back_azimuth
                    comp.stats.inclination = inclination
        return self

    def copy(self):
        """
        Returns a deepcopy of the Stream object.

        :rtype: :class:`~obspy.core.stream.Stream`
        :return: Copy of current stream.

        .. rubric:: Examples

        1. Create a Stream and copy it

            >>> from obspy import read
            >>> st = read()
            >>> st2 = st.copy()

           The two objects are not the same:

            >>> st is st2
            False

           But they have equal data (before applying further processing):

            >>> st == st2
            True

        2. The following example shows how to make an alias but not copy the
           data. Any changes on ``st3`` would also change the contents of
           ``st``.

            >>> st3 = st
            >>> st is st3
            True
            >>> st == st3
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

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> len(st)
        3
        >>> st.clear()  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.traces
        []
        """
        self.traces = []
        return self

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
        except Exception as e:
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
                # add trace to respective list or create that list
                traces_dict.setdefault(trace.id, []).append(trace)
        except IndexError:
            pass
        # clear traces of current stream
        self.traces = []
        # loop through ids
        for id_ in traces_dict.keys():
            trace_list = traces_dict[id_]
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
                    if np.array_equal(cur_trace.slice(t1, t2).data,
                                      trace.slice(t1, t2).data):
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
        self.traces = [tr for tr in self.traces if tr.stats.npts]

    def split(self):
        """
        Splits any trace containing gaps into contiguous unmasked traces.

        :rtype: :class:`obspy.core.stream.Stream`
        :returns: Returns a new stream object containing only contiguous
            unmasked.
        """
        new_stream = Stream()
        for trace in self.traces:
            new_stream.extend(trace.split())
        return new_stream

    @map_example_filename("inventories")
    def attach_response(self, inventories):
        """
        Search for and attach channel response to each trace as
        trace.stats.response. Does not raise an exception but shows a warning
        if response information can not be found for all traces. Returns a
        list of traces for which no response could be found.
        To subsequently deconvolve the instrument response use
        :meth:`Stream.remove_response`.

        >>> from obspy import read, read_inventory
        >>> st = read()
        >>> inv = read_inventory("/path/to/BW_RJOB.xml")
        >>> st.attach_response(inv)
        []
        >>> tr = st[0]
        >>> print(tr.stats.response)  \
                # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Channel Response
           From M/S (Velocity in Meters Per Second) to COUNTS (Digital Counts)
           Overall Sensitivity: 2.5168e+09 defined at 0.020 Hz
           4 stages:
              Stage 1: PolesZerosResponseStage from M/S to V, gain: 1500
              Stage 2: CoefficientsTypeResponseStage from V to COUNTS, ...
              Stage 3: FIRResponseStage from COUNTS to COUNTS, gain: 1
              Stage 4: FIRResponseStage from COUNTS to COUNTS, gain: 1

        :type inventories: :class:`~obspy.station.inventory.Inventory` or
            :class:`~obspy.station.network.Network` or a list containing
            objects of these types.
        :param inventories: Station metadata to use in search for response for
            each trace in the stream.
        :rtype: list of :class:`~obspy.core.trace.Trace`
        :returns: list of traces for which no response information could be
            found.
        """
        skipped_traces = []
        for tr in self.traces:
            try:
                tr.attach_response(inventories)
            except Exception as e:
                if str(e) == "No matching response information found.":
                    warnings.warn(str(e))
                    skipped_traces.append(tr)
                else:
                    raise
        return skipped_traces

    def remove_response(self, *args, **kwargs):
        """
        Method to deconvolve instrument response for all Traces in Stream.

        For details see the corresponding
        :meth:`~obspy.core.trace.Trace.remove_response` method of
        :class:`~obspy.core.trace.Trace`.

        >>> from obspy import read
        >>> st = read()
        >>> # Response object is already attached to example data:
        >>> resp = st[0].stats.response
        >>> print(resp)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Channel Response
            From M/S (Velocity in Meters Per Second) to COUNTS (Digital Counts)
            Overall Sensitivity: 2.5168e+09 defined at 0.020 Hz
            4 stages:
                Stage 1: PolesZerosResponseStage from M/S to V, gain: 1500
                Stage 2: CoefficientsTypeResponseStage from V to COUNTS, ...
                Stage 3: FIRResponseStage from COUNTS to COUNTS, gain: 1
                Stage 4: FIRResponseStage from COUNTS to COUNTS, gain: 1
        >>> st.remove_response()  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            st.remove_response()
            st.plot()

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
        """
        for tr in self:
            tr.remove_response(*args, **kwargs)
        return self


def isPickle(filename):  # @UnusedVariable
    """
    Checks whether a file is a pickled ObsPy Stream file.

    :type filename: str
    :param filename: Name of the pickled ObsPy Stream file to be checked.
    :rtype: bool
    :return: ``True`` if pickled file.

    .. rubric:: Example

    >>> isPickle('/path/to/pickle.file')  # doctest: +SKIP
    True
    """
    if isinstance(filename, (str, native_str)):
        try:
            with open(filename, 'rb') as fp:
                st = pickle.load(fp)
        except:
            return False
    else:
        try:
            st = pickle.load(filename)
        except:
            return False
    return isinstance(st, Stream)


def readPickle(filename, **kwargs):  # @UnusedVariable
    """
    Reads and returns Stream from pickled ObsPy Stream file.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: Name of the pickled ObsPy Stream file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.
    """
    if isinstance(filename, (str, native_str)):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)
    else:
        return pickle.load(filename)


def writePickle(stream, filename, protocol=2, **kwargs):  # @UnusedVariable
    """
    Writes a Python pickle of current stream.

    .. note::
        Writing into PICKLE format allows to store additional attributes
        appended to the current Stream object or any contained Trace.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of file to write.
    :type protocol: int, optional
    :param protocol: Pickle protocol, defaults to ``2``.
    """
    if isinstance(filename, (str, native_str)):
        with open(filename, 'wb') as fp:
            pickle.dump(stream, fp, protocol=protocol)
    else:
        pickle.dump(stream, filename, protocol=protocol)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
