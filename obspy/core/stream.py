# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Stream objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import PY3, native_str

import collections
import copy
import fnmatch
import math
import os
import pickle
import re
import warnings
from glob import glob, has_magic

import numpy as np

from obspy.core import compatibility
from obspy.core.trace import Trace
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.attribdict import AttribDict
from obspy.core.util.base import (ENTRY_POINTS, _get_function_from_entry_point,
                                  _read_from_plugin, _generic_reader)
from obspy.core.util.decorator import (map_example_filename,
                                       raise_if_masked, uncompress_file)
from obspy.core.util.misc import get_window_times, buffered_load_entry_point
from obspy.core.util.obspy_types import ObsPyException


_headonly_warning_msg = (
    "Keyword headonly cannot be combined with starttime, endtime or dtype.")


@map_example_filename("pathname_or_url")
def read(pathname_or_url=None, format=None, headonly=False, starttime=None,
         endtime=None, nearest_sample=True, dtype=None, apply_calib=False,
         check_compression=True, **kwargs):
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
    :param check_compression: Check for compression on file and decompress
        if needed. This may be disabled for a moderate speed up.
    :type check_compression: bool, optional
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

    Example waveform files may be retrieved via https://examples.obspy.org.

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
        >>> st = read("https://examples.obspy.org/loc_RJOB20050831023349.z")
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        .RJOB..Z | 2005-08-31T02:33:49.850000Z - ... | 200.0 Hz, 12000 samples

    (4) Reading a compressed files.

        >>> from obspy import read
        >>> st = read("/path/to/tspair.ascii.gz")
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        XX.TEST..BHZ | 2008-01-15T00:00:00.025000Z - ... | 40.0 Hz, 635 samples

        >>> st = read("https://examples.obspy.org/slist.ascii.bz2")
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        XX.TEST..BHZ | 2008-01-15T00:00:00.025000Z - ... | 40.0 Hz, 635 samples

    (5) Reading a file-like object.

        >>> import requests
        >>> import io
        >>> example_url = "https://examples.obspy.org/loc_RJOB20050831023349.z"
        >>> stringio_obj = io.BytesIO(requests.get(example_url).content)
        >>> st = read(stringio_obj)
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        .RJOB..Z | 2005-08-31T02:33:49.850000Z - ... | 200.0 Hz, 12000 samples

    (6) Using 'starttime' and 'endtime' parameters

        >>> from obspy import read
        >>> dt = UTCDateTime("2005-08-31T02:34:00")
        >>> st = read("https://examples.obspy.org/loc_RJOB20050831023349.z",
        ...           starttime=dt, endtime=dt+10)
        >>> print(st)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        .RJOB..Z | 2005-08-31T02:34:00.000000Z - ... | 200.0 Hz, 2001 samples
    """
    # add default parameters to kwargs so sub-modules may handle them
    kwargs['starttime'] = starttime
    kwargs['endtime'] = endtime
    kwargs['nearest_sample'] = nearest_sample
    kwargs['check_compression'] = check_compression
    kwargs['headonly'] = headonly
    kwargs['format'] = format

    if pathname_or_url is None:
        # if no pathname or URL specified, return example stream
        st = _create_example_stream(headonly=headonly)
    else:
        st = _generic_reader(pathname_or_url, _read, **kwargs)

    if len(st) == 0:
        # try to give more specific information why the stream is empty
        if has_magic(pathname_or_url) and not glob(pathname_or_url):
            raise Exception("No file matching file pattern: %s" %
                            pathname_or_url)
        elif not has_magic(pathname_or_url) and \
                not os.path.isfile(pathname_or_url):
            raise IOError(2, "No such file or directory", pathname_or_url)
        # Only raise error if no start/end time has been set. This
        # will return an empty stream if the user chose a time window with
        # no data in it.
        # XXX: Might cause problems if the data is faulty and the user
        # set start/end time. Not sure what to do in this case.
        elif not starttime and not endtime:
            raise Exception("Cannot open file/files: %s" % pathname_or_url)
    # Trim if times are given.
    if headonly and (starttime or endtime or dtype):
        warnings.warn(_headonly_warning_msg, UserWarning)
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


@uncompress_file
def _read(filename, format=None, headonly=False, **kwargs):
    """
    Read a single file into a ObsPy Stream object.
    """
    stream, format = _read_from_plugin('waveform', filename, format=format,
                                       headonly=headonly, **kwargs)
    # set _format identifier for each element
    for trace in stream:
        trace.stats._format = format
    return stream


def _create_example_stream(headonly=False):
    """
    Create an example stream.

    Data arrays are stored in NumPy's NPZ format. The header information are
    fixed values.

    PAZ of the used instrument, needed to demonstrate simulate_seismometer()
    etc.::

        paz = {'gain': 60077000.0,
               'poles': [-0.037004+0.037016j, -0.037004-0.037016j, -251.33+0j,
                         -131.04-467.29j, -131.04+467.29j],
               'sensitivity': 2516778400.0,
               'zeros': [0j, 0j]}}

    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not headonly:
        path = os.path.join(data_dir, "example.npz")
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
    from obspy import read_inventory
    inv = read_inventory(os.path.join(data_dir, "BW_RJOB.xml"))
    st.attach_response(inv)
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
        Add two streams or a stream with a single trace.

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
        Add two streams with self += other.

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
        Create a new Stream containing num copies of this stream.

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
        Return the number of Traces in the Stream object.

        .. rubric:: Example

        >>> stream = Stream([Trace(), Trace(), Trace()])
        >>> len(stream)
        3
        """
        return len(self.traces)

    count = __len__

    def __str__(self, extended=False):
        """
        Return short summary string of the current stream.

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

        Trace order does not effect the comparison because the traces are
        sorted beforehand.

        This function strictly compares the data and stats objects of each
        trace contained by the streams. If less strict behavior is desired,
        which may be the case for testing, consider using the
        :func:`~obspy.core.util.testing.stream_almost_equal` function.

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
        # see also https://docs.python.org/3/reference/datamodel.html
        return self.__class__(traces=self.traces[max(0, i):max(0, j):k])

    def append(self, trace):
        """
        Append a single Trace object to the current Stream object.

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
        Extend the current Stream object with a list of Trace objects.

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

    def get_gaps(self, min_gap=None, max_gap=None):
        """
        Determine all trace gaps/overlaps of the Stream object.

        :param min_gap: All gaps smaller than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.
        :param max_gap: All gaps larger than this value will be omitted. The
            value is assumed to be in seconds. Defaults to None.

        The returned list contains one item in the following form for each gap/
        overlap: [network, station, location, channel, starttime of the gap,
        end time of the gap, duration of the gap, number of missing samples]

        Please be aware that no sorting and checking of stations, channels, ...
        is done. This method only compares the start and end times of the
        Traces and the start and end times of segments within Traces that
        contain masked arrays (i.e., Traces that were merged without a fill
        value).

        .. rubric:: Example

        Our example stream has no gaps:

        >>> from obspy import read, UTCDateTime
        >>> st = read()
        >>> st.get_gaps()
        []
        >>> st.print_gaps()  # doctest: +ELLIPSIS
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
        >>> st.get_gaps()[0]  # doctest: +SKIP
        [['BW', 'RJOB', '', 'EHZ', UTCDateTime(2009, 8, 24, 0, 20, 13),
          UTCDateTime(2009, 8, 24, 0, 20, 14), 1.0, 99]]
        >>> st.print_gaps()  # doctest: +ELLIPSIS
        Source            Last Sample                 ...
        BW.RJOB..EHZ      2009-08-24T00:20:13.000000Z ...
        Total: 1 gap(s) and 0 overlap(s)
        """
        # Create shallow copy of the traces to be able to sort them later on.
        copied_traces = copy.copy(self.traces)
        self.sort()
        gap_list = []
        for _i in range(len(self.traces)):
            # if the trace is masked, break it up and run get_gaps on the
            # resulting stream
            if isinstance(self.traces[_i].data, np.ma.masked_array):
                gap_list.extend(self.traces[_i].split().get_gaps())
            if _i + 1 == len(self.traces):
                # reached the last trace
                break
            # skip traces with different network, station, location or channel
            if self.traces[_i].id != self.traces[_i + 1].id:
                continue
            # different sampling rates should always result in a gap or overlap
            if self.traces[_i].stats.delta == self.traces[_i + 1].stats.delta:
                same_sampling_rate = True
            else:
                same_sampling_rate = False
            stats = self.traces[_i].stats
            stime = min(stats['endtime'], self.traces[_i + 1].stats['endtime'])
            etime = self.traces[_i + 1].stats['starttime']
            # last sample of earlier trace represents data up to time of last
            # sample (stats.endtime) plus one delta
            delta = etime.timestamp - (stime.timestamp + stats.delta)
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
            if delta < 0:
                nsamples = -nsamples
            # skip if is equal to delta (1 / sampling rate)
            if same_sampling_rate and nsamples == 0:
                continue
            # check if gap is already covered in trace before:
            covered = False
            # only need to check previous traces because the traces are sorted
            for prev_trace in self.traces[:_i]:
                prev_stats = prev_trace.stats
                # look if trace is contained in other trace
                prev_start = prev_stats['starttime']
                prev_end = prev_stats['endtime']
                if not (prev_start < stime < etime < prev_end):
                    continue
                # don't look in traces of other measurements
                elif self.traces[_i].id != prev_trace.id:
                    continue
                else:
                    covered = True
                    break
            if covered:
                continue
            gap_list.append([stats['network'], stats['station'],
                             stats['location'], stats['channel'],
                             stime, etime, delta, nsamples])
        # Set the original traces to not alter the stream object.
        self.traces = copied_traces
        return gap_list

    def insert(self, position, object):
        """
        Insert either a single Trace or a list of Traces before index.

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
        Create a waveform plot of the current ObsPy Stream object.

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
            line plotted. If ``type='section'`` then the values ``'network'``,
            ``'station'`` or ``'channel'`` are also accepted, and traces will
            be uniquely colored by the given information.
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
        :param type: Type may be set to either: ``'normal'`` to produce the
            standard plot; ``'dayplot'`` to create a one-day plot for a single
            Trace; ``'relative'`` to convert all date/time information to a
            relative scale starting the seismogram at 0 seconds; ``'section'``
            to plot all seismograms in a single coordinate system shifted
            according to their distance from a reference point. Defaults to
            ``'normal'``.
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
            obspy.clients.fdsn. Just pass a dictionary with a "min_magnitude"
            key, e.g. ::

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
            Either m or degree, depending on ``dist_degree``.
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
        :type orientation: str, optional
        :param orientation: The orientation of the time axis, either
            ``'vertical'`` or ``'horizontal'``. Defaults to ``'vertical'``.
        :type fillcolors: tuple, optional
        :param fillcolors:  Fill the inside of the lines (wiggle plot),
            for both the positive and negative sides; use ``None`` to omit
            one of the sides. Defaults to ``(None,None)``.

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

        * legal `HTML color names <https://www.w3.org/TR/css3-color/#html4>`_,
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
        return waveform.plot_waveform(*args, **kwargs)

    def spectrogram(self, **kwargs):
        """
        Create a spectrogram plot for each trace in the stream.

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
        Remove and return the Trace object specified by index from the Stream.

        If no index is given, remove the last Trace. Passes on the pop() to
        self.traces.

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

    def print_gaps(self, min_gap=None, max_gap=None):
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
        >>> st.get_gaps()
        []
        >>> st.print_gaps()  # doctest: +ELLIPSIS
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
        >>> st.get_gaps()  # doctest: +ELLIPSIS
        [[..., UTCDateTime(2009, 8, 24, 0, 20, 13), ...
        >>> st.print_gaps()  # doctest: +ELLIPSIS
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
        >>> st.get_gaps()  # doctest: +ELLIPSIS
        [[...'EHZ', UTCDateTime(2009, 8, 24, 0, 20, 13), ...
        >>> st.print_gaps()  # doctest: +ELLIPSIS
        Source            Last Sample                 ...
        BW.RJOB..EHZ      2009-08-24T00:20:13.000000Z ...
        Total: 0 gap(s) and 1 overlap(s)
        """
        result = self.get_gaps(min_gap, max_gap)
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
        Remove the first occurrence of the specified Trace object in the
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
        Reverse the Traces of the Stream object in place.

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
        Sort the traces in the Stream object.

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

    def write(self, filename, format=None, **kwargs):
        """
        Save stream into a file.

        :type filename: str
        :param filename: The name of the file to write.
        :type format: str, optional
        :param format: The file format to use (e.g. ``"MSEED"``). See
            the `Supported Formats`_ section below for a list of supported
            formats. If format is set to ``None`` it will be deduced from
            file extension, whenever possible.
        :param kwargs: Additional keyword arguments passed to the underlying
            waveform writer method.

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()  # doctest: +SKIP
        >>> st.write("example.mseed", format="MSEED")  # doctest: +SKIP

        The ``format`` argument can be omitted, and the file format will be
        deduced from file extension, whenever possible.

        >>> st.write("example.mseed")  # doctest: +SKIP

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
        if not self.traces:
            msg = 'Can not write empty stream to file.'
            raise ObsPyException(msg)

        # Check all traces for masked arrays and raise exception.
        for trace in self.traces:
            if isinstance(trace.data, np.ma.masked_array):
                msg = 'Masked array writing is not supported. You can use ' + \
                      'np.array.filled() to convert the masked array to a ' + \
                      'normal array.'
                raise NotImplementedError(msg)
        if format is None:
            # try to guess format from file extension
            _, format = os.path.splitext(filename)
            format = format[1:]
        format = format.upper()
        try:
            # get format specific entry point
            format_ep = ENTRY_POINTS['waveform_write'][format]
            # search writeFormat method for given entry point
            write_format = buffered_load_entry_point(
                format_ep.dist.key,
                'obspy.plugin.waveform.%s' % (format_ep.name), 'writeFormat')
        except (IndexError, ImportError, KeyError):
            msg = "Writing format \"%s\" is not supported. Supported types: %s"
            raise ValueError(msg % (format,
                                    ', '.join(ENTRY_POINTS['waveform_write'])))
        write_format(self, filename, **kwargs)

    def trim(self, starttime=None, endtime=None, pad=False,
             nearest_sample=True, fill_value=None):
        """
        Cut all traces of this Stream object to given start and end time.

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
            selected, if set to ``False``, the inner (next sample for a
            start time border, previous sample for an end time border) sample
            containing the time is selected. Defaults to ``True``.

            Given the following trace containing 6 samples, "|" are the
            sample points, "A" is the requested starttime::

                |         |A        |         |       B |         |
                1         2         3         4         5         6

            ``nearest_sample=True`` will select samples 2-5,
            ``nearest_sample=False`` will select samples 3-4 only.

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
            try:
                if starttime is not None:
                    delta = compatibility.round_away(
                        (starttime - tr.stats.starttime) *
                        tr.stats.sampling_rate)
                    starttime = tr.stats.starttime + delta * tr.stats.delta
                if endtime is not None:
                    delta = compatibility.round_away(
                        (endtime - tr.stats.endtime) * tr.stats.sampling_rate)
                    # delta is negative!
                    endtime = tr.stats.endtime + delta * tr.stats.delta
            except TypeError:
                msg = ('starttime and endtime must be UTCDateTime objects '
                       'or None for this call to Stream.trim()')
                raise TypeError(msg)
        for trace in self.traces:
            trace.trim(starttime, endtime, pad=pad,
                       nearest_sample=nearest_sample, fill_value=fill_value)
        # remove empty traces after trimming
        self.traces = [_i for _i in self.traces if _i.stats.npts]
        return self

    def _ltrim(self, starttime, pad=False, nearest_sample=True):
        """
        Cut all traces of this Stream object to given start time.
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
        Cut all traces of this Stream object to given end time.
        For more info see :meth:`~obspy.core.trace.Trace._rtrim`.
        """
        for trace in self.traces:
            trace.trim(endtime=endtime, pad=pad, nearest_sample=nearest_sample)
        # remove empty traces after trimming
        self.traces = [tr for tr in self.traces if tr.stats.npts]
        return self

    def cutout(self, starttime, endtime):
        """
        Cut the given time range out of all traces of this Stream object.

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

    def slice(self, starttime=None, endtime=None, keep_empty_traces=False,
              nearest_sample=True):
        """
        Return new Stream object cut to the given start and end time.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Specify the start time of all traces.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Specify the end time of all traces.
        :type keep_empty_traces: bool, optional
        :param keep_empty_traces: Empty traces will be kept if set to ``True``.
            Defaults to ``False``.
        :type nearest_sample: bool, optional
        :param nearest_sample: If set to ``True``, the closest sample is
            selected, if set to ``False``, the inner (next sample for a
            start time border, previous sample for an end time border) sample
            containing the time is selected. Defaults to ``True``.

            Given the following trace containing 6 samples, "|" are the
            sample points, "A" is the requested starttime::

                |         |A        |         |       B |         |
                1         2         3         4         5         6

            ``nearest_sample=True`` will select samples 2-5,
            ``nearest_sample=False`` will select samples 3-4 only.

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
            sliced_trace = trace.slice(starttime=starttime, endtime=endtime,
                                       nearest_sample=nearest_sample)
            if keep_empty_traces is False and not sliced_trace.stats.npts:
                continue
            new.append(sliced_trace)
        return new

    def slide(self, window_length, step, offset=0,
              include_partial_windows=False, nearest_sample=True):
        """
        Generator yielding equal length sliding windows of the Stream.

        Please keep in mind that it only returns a new view of the original
        data. Any modifications are applied to the original data as well. If
        you don't want this you have to create a copy of the yielded
        windows. Also be aware that if you modify the original data and you
        have overlapping windows, all following windows are affected as well.

        Not all yielded windows must have the same number of traces. The
        algorithm will determine the maximal temporal extents by analysing
        all Traces and then creates windows based on these times.

        .. rubric:: Example

        >>> import obspy
        >>> st = obspy.read()
        >>> for windowed_st in st.slide(window_length=10.0, step=10.0):
        ...     print(windowed_st)
        ...     print("---")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        3 Trace(s) in Stream:
        ... | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:13.000000Z | ...
        ... | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:13.000000Z | ...
        ... | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:13.000000Z | ...
        ---
        3 Trace(s) in Stream:
        ... | 2009-08-24T00:20:13.000000Z - 2009-08-24T00:20:23.000000Z | ...
        ... | 2009-08-24T00:20:13.000000Z - 2009-08-24T00:20:23.000000Z | ...
        ... | 2009-08-24T00:20:13.000000Z - 2009-08-24T00:20:23.000000Z | ...


        :param window_length: The length of each window in seconds.
        :type window_length: float
        :param step: The step between the start times of two successive
            windows in seconds. Can be negative if an offset is given.
        :type step: float
        :param offset: The offset of the first window in seconds relative to
            the start time of the whole interval.
        :type offset: float
        :param include_partial_windows: Determines if windows that are
            shorter then 99.9 % of the desired length are returned.
        :type include_partial_windows: bool
        :param nearest_sample: If set to ``True``, the closest sample is
            selected, if set to ``False``, the inner (next sample for a
            start time border, previous sample for an end time border) sample
            containing the time is selected. Defaults to ``True``.

            Given the following trace containing 6 samples, "|" are the
            sample points, "A" is the requested starttime::

                |         |A        |         |       B |         |
                1         2         3         4         5         6

            ``nearest_sample=True`` will select samples 2-5,
            ``nearest_sample=False`` will select samples 3-4 only.
        :type nearest_sample: bool, optional
        """
        starttime = min(tr.stats.starttime for tr in self)
        endtime = max(tr.stats.endtime for tr in self)
        windows = get_window_times(
            starttime=starttime,
            endtime=endtime,
            window_length=window_length,
            step=step,
            offset=offset,
            include_partial_windows=include_partial_windows)

        if len(windows) < 1:
            return

        for start, stop in windows:
            temp = self.slice(start, stop,
                              nearest_sample=nearest_sample)
            # It might happen that there is a time frame where there are no
            # windows, e.g. two traces separated by a large gap.
            if not temp:
                continue
            yield temp

    def select(self, network=None, station=None, location=None, channel=None,
               sampling_rate=None, npts=None, component=None, id=None):
        """
        Return new Stream object only with these traces that match the given
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
                if not fnmatch.fnmatch(trace.stats.channel[-1].upper(),
                                       component.upper()):
                    continue
            traces.append(trace)
        return self.__class__(traces=traces)

    def verify(self):
        """
        Verify all traces of current Stream against available meta data.

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

    def _merge_checks(self):
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

    def merge(self, method=0, fill_value=None, interpolation_samples=0,
              **kwargs):
        """
        Merge ObsPy Trace objects with same IDs.

        :type method: int, optional
        :param method: Methodology to handle overlaps/gaps of traces. Defaults
            to ``0``.
            See :meth:`obspy.core.trace.Trace.__add__` for details on
            methods ``0`` and ``1``,
            see :meth:`obspy.core.stream.Stream._cleanup` for details on
            method ``-1``. Any merge operation performs a cleanup merge as
            a first step (method ``-1``).
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

        self._cleanup(**kwargs)
        if method == -1:
            return
        # check sampling rates and dtypes
        self._merge_checks()
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
                _id = trace.get_id()
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
        :func:`~obspy.signal.invsim.simulate_seismometer`.

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
            :func:`~obspy.signal.invsim.simulate_seismometer` and the
            `ObsPy Tutorial
            <https://docs.obspy.org/master/tutorial/code_snippets/\
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
        >>> from obspy.signal.invsim import corn_freq_2_paz
        >>> st = read()
        >>> paz_sts2 = {'poles': [-0.037004+0.037016j, -0.037004-0.037016j,
        ...                       -251.33+0j,
        ...                       -131.04-467.29j, -131.04+467.29j],
        ...             'zeros': [0j, 0j],
        ...             'gain': 60077000.0,
        ...             'sensitivity': 2516778400.0}
        >>> paz_1hz = corn_freq_2_paz(1.0, damp=0.707)
        >>> st.simulate(paz_remove=paz_sts2, paz_simulate=paz_1hz)
        ... # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read
            from obspy.signal.invsim import corn_freq_2_paz
            st = read()
            paz_sts2 = {'poles': [-0.037004+0.037016j, -0.037004-0.037016j,
                                  -251.33+0j,
                                  -131.04-467.29j, -131.04+467.29j],
                        'zeros': [0j, 0j],
                        'gain': 60077000.0,
                        'sensitivity': 2516778400.0}
            paz_1hz = corn_freq_2_paz(1.0, damp=0.707)
            paz_1hz['sensitivity'] = 1.0
            st.simulate(paz_remove=paz_sts2, paz_simulate=paz_1hz)
            st.plot()
        """
        for tr in self:
            tr.simulate(paz_remove=paz_remove, paz_simulate=paz_simulate,
                        remove_sensitivity=remove_sensitivity,
                        simulate_sensitivity=simulate_sensitivity, **kwargs)
        return self

    @raise_if_masked
    def filter(self, type, **options):
        """
        Filter the data of all traces in the Stream.

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

        ``'lowpass_cheby_2'``
            Cheby2-Lowpass (uses :func:`obspy.signal.filter.lowpass_cheby_2`).

        ``'lowpass_fir'`` (experimental)
            FIR-Lowpass (uses :func:`obspy.signal.filter.lowpass_fir`).

        ``'remez_fir'`` (experimental)
            Minimax optimal bandpass using Remez algorithm (uses
            :func:`obspy.signal.filter.remez_fir`).

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
        Run a triggering algorithm on all traces in the stream.

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
            :func:`obspy.signal.trigger.classic_sta_lta`).

        ``'recstalta'``
            Recursive STA/LTA
            (uses :func:`obspy.signal.trigger.recursive_sta_lta`).

        ``'recstaltapy'``
            Recursive STA/LTA written in Python (uses
            :func:`obspy.signal.trigger.recursive_sta_lta_py`).

        ``'delayedstalta'``
            Delayed STA/LTA.
            (uses :func:`obspy.signal.trigger.delayed_sta_lta`).

        ``'carlstatrig'``
            Computes the carl_sta_trig characteristic function (uses
            :func:`obspy.signal.trigger.carl_sta_trig`).

        ``'zdetect'``
            Z-detector (uses :func:`obspy.signal.trigger.z_detect`).

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
        Get the values of the absolute maximum amplitudes of all traces in the
        stream. See :meth:`~obspy.core.trace.Trace.max`.

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

    def differentiate(self, method='gradient'):
        """
        Differentiate all traces with respect to time.

        :type method: str, optional
        :param method: Method to use for differentiation. Defaults to
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
            tr.differentiate(method=method)
        return self

    def integrate(self, method='cumtrapz', **options):
        """
        Integrate all traces with respect to time.

        For details see the corresponding
        :meth:`~obspy.core.trace.Trace.integrate` method of
        :class:`~obspy.core.trace.Trace`.

        :type method: str, optional
        :param type: Method to use for integration. Defaults to
            ``'cumtrapz'``. See :meth:`~obspy.core.trace.Trace.integrate` for
            further details.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of every trace.
        """
        for tr in self:
            tr.integrate(method=method, **options)
        return self

    @raise_if_masked
    def detrend(self, type='simple', **options):
        """
        Remove a trend from all traces.

        For details on supported methods and parameters see the corresponding
        :meth:`~obspy.core.trace.Trace.detrend` method of
        :class:`~obspy.core.trace.Trace`.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data will no longer be accessible afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
        """
        for tr in self:
            tr.detrend(type=type, **options)
        return self

    def taper(self, *args, **kwargs):
        """
        Taper all Traces in Stream.

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
        Interpolate all Traces in a Stream.

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
        Calculate standard deviations of all Traces in the Stream.

        Standard deviations are calculated by NumPy method
        :meth:`~numpy.ndarray.std` on ``trace.data`` for every trace in the
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
        Normalize all Traces in the Stream.

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

    def rotate(self, method, back_azimuth=None, inclination=None,
               inventory=None, **kwargs):
        """
        Rotate stream objects.

        :type method: str
        :param method: Determines the rotation method.

            ``'->ZNE'``: Rotates data from three components into Z, North- and
                East-components based on the station metadata (e.g. borehole
                stations). Uses mandatory ``inventory`` parameter (provide
                either an :class:`~obspy.core.inventory.inventory.Inventory` or
                :class:`~obspy.io.xseed.parser.Parser` object) and ignores
                ``back_azimuth`` and ``inclination`` parameters. Additional
                kwargs will be passed on to :meth:`_rotate_to_zne()` (use if
                other components than ``["Z", "1", "2"]`` and
                ``["1", "2", "3"]`` need to be rotated).
                Trims common channels used in rotation to time spans that are
                available for all three channels (i.e. cuts away parts for
                which one or two channels used in rotation do not have data).
            ``'NE->RT'``: Rotates the North- and East-components of a
                seismogram to radial and transverse components.
            ``'RT->NE'``: Rotates the radial and transverse components of a
                seismogram to North- and East-components.
            ``'ZNE->LQT'``: Rotates from left-handed Z, North, and  East system
                to LQT, e.g. right-handed ray coordinate system.
            ``'LQT->ZNE'``: Rotates from LQT, e.g. right-handed ray coordinate
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
        :type inventory: :class:`~obspy.core.inventory.inventory.Inventory` or
            :class:`~obspy.io.xseed.parser.Parser`
        :param inventory: Inventory or SEED Parser with metadata of channels.

        Example to rotate unaligned borehole instrument data based on station
        inventory (a dataless SEED :class:`~obspy.io.xseed.parser.Parser` can
        also be provided, see details for option ``inventory``):

        >>> from obspy import read, read_inventory
        >>> st = read("/path/to/ffbx_unrotated_gaps.mseed")
        >>> inv = read_inventory("/path/to/ffbx.stationxml")
        >>> st.rotate(method="->ZNE", inventory=inv)  # doctest: +ELLIPSIS
        <obspy.core.stream.Stream object at 0x...>
        """
        if method == "->ZNE":
            if inventory is None:
                msg = ("With method '->ZNE' station metadata has to be "
                       "provided as 'inventory' parameter.")
                raise ValueError(msg)
            return self._rotate_to_zne(inventory, **kwargs)
        elif method == "NE->RT":
            func = "rotate_ne_rt"
        elif method == "RT->NE":
            func = "rotate_rt_ne"
        elif method == "ZNE->LQT":
            func = "rotate_zne_lqt"
        elif method == "LQT->ZNE":
            func = "rotate_lqt_zne"
        else:
            msg = ("Method has to be one of ('->ZNE', 'NE->RT', 'RT->NE', "
                   "'ZNE->LQT', or 'LQT->ZNE').")
            raise ValueError(msg)
        # Retrieve function call from entry points
        func = _get_function_from_entry_point("rotate", func)
        # Split to get the components. No need for further checks for the
        # method as invalid methods will be caught by previous conditional.
        input_components, output_components = method.split("->")
        # Figure out inclination and back-azimuth.
        if back_azimuth is None:
            try:
                back_azimuth = self[0].stats.back_azimuth
            except Exception:
                msg = "No back-azimuth specified."
                raise TypeError(msg)
        if len(input_components) == 3 and inclination is None:
            try:
                inclination = self[0].stats.inclination
            except Exception:
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
        Return a deepcopy of the Stream object.

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
        Clear trace list (convenience method).

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

    def _cleanup(self, misalignment_threshold=1e-2):
        """
        Merge consistent trace objects but leave everything else alone.

        This can mean traces with matching header that are directly adjacent or
        are contained/equal/overlapping traces with exactly the same waveform
        data in the overlapping part.
        If option `misalignment_threshold` is non-zero then
        contained/overlapping/directly adjacent traces with the sampling points
        misaligned by less than `misalignment_threshold` times the sampling
        interval are aligned on the same sampling points (see example below).

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

        Misaligned traces are aligned, depending on set parameters, e.g. for a
        directly adjacent trace with slight misalignment (with two common
        samples at start of Trace 2 for better visualization)::

            before:
            Trace 1: A---------A---------A
            Trace 2:            A---------A---------B---------B

            after:
            Trace 1: A---------A---------A---------B---------B

        :type misalignment_threshold: float
        :param misalignment_threshold: Threshold value for sub-sample
            misalignments of sampling points of two traces that should be
            merged together (fraction of sampling interval, from 0 to 0.5).
            ``0`` means traces with even just the slightest misalignment will
            not be merged together, ``0.5`` means traces will be merged
            together disregarding of any sub-sample shifts of sampling points.
        """
        # first of all throw away all empty traces
        self.traces = [_i for _i in self.traces if _i.stats.npts]
        # check sampling rates and dtypes
        try:
            self._merge_checks()
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
            delta = cur_trace.stats.delta
            allowed_micro_shift = misalignment_threshold * delta
            # work through all traces of same id
            while trace_list:
                trace = trace_list.pop(0)
                # `gap` is the deviation (in seconds) of the actual start
                # time of the second trace from the expected start time
                # (for the ideal case of directly adjacent and perfectly
                # aligned traces).
                gap = trace.stats.starttime - (cur_trace.stats.endtime + delta)
                # if `gap` is larger than the designated allowed shift,
                # we treat it as a real gap and leave as is.
                if misalignment_threshold > 0 and gap <= allowed_micro_shift:
                    # `gap` is smaller than allowed shift (or equal),
                    #  the traces could be
                    #  - overlapping without being misaligned or..
                    #  - overlapping with misalignment or..
                    #  - misaligned with a micro gap
                    # check if the sampling points are misaligned:
                    misalignment = gap % delta
                    if misalignment != 0:
                        # determine the position of the second trace's
                        # sampling points in the interval between two
                        # sampling points of first trace.
                        # a `misalign_percentage` of close to 0.0 means a
                        # sampling point of the first trace is just a bit
                        # to the left of our sampling point:
                        #
                        #  Trace 1: --|---------|---------|---------|--
                        #  Trace 2: ---|---------|---------|---------|-
                        # misalign_percentage:  0.........1
                        #
                        # a `misalign_percentage` of close to 1.0 means a
                        # sampling point of the first trace is just a bit
                        # to the right of our sampling point:
                        #
                        #  Trace 1: --|---------|---------|---------|--
                        #  Trace 2: -|---------|---------|---------|---
                        # misalign_percentage:  0.........1
                        misalign_percentage = misalignment / delta
                        if (misalign_percentage <= misalignment_threshold or
                                misalign_percentage >=
                                1 - misalignment_threshold):
                            # now we align the sampling points of both traces
                            trace.stats.starttime = (
                                cur_trace.stats.starttime +
                                round((trace.stats.starttime -
                                       cur_trace.stats.starttime) / delta) *
                                delta)
                # we have some common parts: check if consistent
                # (but only if sampling points are matching to specified
                #  accuracy, which is checked and conditionally corrected in
                #  previous code block)
                subsample_shift_percentage = (
                    trace.stats.starttime.timestamp -
                    cur_trace.stats.starttime.timestamp) % delta / delta
                subsample_shift_percentage = min(
                    subsample_shift_percentage, 1 - subsample_shift_percentage)
                if (trace.stats.starttime <= cur_trace.stats.endtime and
                        subsample_shift_percentage < misalignment_threshold):
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
        return self

    def split(self):
        """
        Split any trace containing gaps into contiguous unmasked traces.

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
        >>> inv = read_inventory()
        >>> st.attach_response(inv)
        []
        >>> tr = st[0]
        >>> print(tr.stats.response)  \
                # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Channel Response
           From M/S (Velocity in Meters per Second) to COUNTS (Digital Counts)
           Overall Sensitivity: 2.5168e+09 defined at 0.020 Hz
           4 stages:
              Stage 1: PolesZerosResponseStage from M/S to V, gain: 1500
              Stage 2: CoefficientsTypeResponseStage from V to COUNTS, ...
              Stage 3: FIRResponseStage from COUNTS to COUNTS, gain: 1
              Stage 4: FIRResponseStage from COUNTS to COUNTS, gain: 1

        :type inventories: :class:`~obspy.core.inventory.inventory.Inventory`
            or :class:`~obspy.core.inventory.network.Network` or a list
            containing objects of these types.
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
        Deconvolve instrument response for all Traces in Stream.

        For details see the corresponding
        :meth:`~obspy.core.trace.Trace.remove_response` method of
        :class:`~obspy.core.trace.Trace`.

        >>> from obspy import read, read_inventory
        >>> st = read()
        >>> inv = read_inventory()
        >>> st.remove_response(inventory=inv)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read, read_inventory
            st = read()
            inv = read_inventory()
            st.remove_response(inventory=inv)
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

    def remove_sensitivity(self, *args, **kwargs):
        """
        Remove instrument sensitivity for all Traces in Stream.

        For details see the corresponding
        :meth:`~obspy.core.trace.Trace.remove_sensitivity` method of
        :class:`~obspy.core.trace.Trace`.

        >>> from obspy import read, read_inventory
        >>> st = read()
        >>> inv = read_inventory()
        >>> st.remove_sensitivity(inv)  # doctest: +ELLIPSIS
        <...Stream object at 0x...>

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
        """
        for tr in self:
            tr.remove_sensitivity(*args, **kwargs)
        return self

    def stack(self, group_by='all', stack_type='linear', npts_tol=0,
              time_tol=0):
        """
        Stack traces by the same selected metadata.

        The metadata of each trace (including starttime) corresponds to the
        metadata of the original traces if those are the same.
        Additionaly, the entry ``stack`` is written to the stats object(s).
        It contains the fields ``group``
        (result of the format operation on the ``group_by`` parameter),
        ``count`` (number of stacked traces) and ``type``
        (``stack_type`` argument).

        :type group_by: str
        :param group_by: Stack waveforms together which have the same metadata
            given by this parameter. The parameter should name the
            corresponding keys of the stats object,
            e.g. ``'{network}.{station}'`` for stacking all
            locations and channels of the stations and returning a stream
            consisting of one stacked trace for each station.
            This parameter can take two special values,
            ``'id'`` which stacks the waveforms by SEED id and
            ``'all'`` (default) which stacks together all traces in the stream.
        :type stack_type: str or tuple
        :param stack_type: Type of stack, one of the following:
            ``'linear'``: average stack (default),
            ``('pw', order)``: phase weighted stack of given order,
            see [Schimmel1997]_,
            ``('root', order)``: root stack of given order.
        :type npts_tol: int
        :param npts_tol: Tolerate traces with different number of points
            with a difference up to this value. Surplus samples are discarded.
        :type time_tol: float (seconds)
        :param time_tol: Tolerate difference in startime when setting the
            new starttime of the stack. If starttimes differs more than this
            value it will be set to timestamp 0.

        >>> from obspy import read
        >>> st = read()
        >>> stack = st.stack()
        >>> print(stack)  # doctest: +ELLIPSIS
        1 Trace(s) in Stream:
        BW.RJOB.. | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data will no longer be accessible afterwards. To keep your
            original data, use :meth:`~obspy.core.stream.Stream.copy` to create
            a copy of your stream object.
        """
        from obspy.signal.util import stack as stack_func
        groups = self._groupby(group_by)
        stacks = []
        for groupid, traces in groups.items():
            header = {k: v for k, v in traces[0].stats.items()
                      if all(np.all(tr.stats.get(k) == v) for tr in traces)}
            header.pop('endtime', None)
            if 'sampling_rate' not in header:
                msg = 'Sampling rate of traces to stack is different'
                raise ValueError(msg)
            if 'starttime' not in header and time_tol > 0:
                times = [tr.stats.starttime for tr in traces]
                if np.ptp(times) <= time_tol:
                    # use high median as starttime
                    header['starttime'] = sorted(times)[len(times) // 2]
            header['stack'] = AttribDict(group=groupid, count=len(traces),
                                         type=stack_type)
            npts_all = [len(tr) for tr in traces]
            npts_dif = np.ptp(npts_all)
            npts = min(npts_all)
            if npts_dif > npts_tol:
                msg = ('Difference of number of points of the traces is higher'
                       ' than requested tolerance ({} > {})')
                raise ValueError(msg.format(npts_dif, npts_tol))
            data = np.array([tr.data[:npts] for tr in traces])
            stack = stack_func(data, stack_type=stack_type)
            stacks.append(traces[0].__class__(data=stack, header=header))
        self.traces = stacks
        return self

    @staticmethod
    def _dummy_stream_from_string(s):
        """
        Helper method to create a dummy Stream object (with data always equal
        to one) from a string representation of the Stream, mostly for
        debugging purposes.

        >>> s = ['', '', '3 Trace(s) in Stream:',
        ...      'IU.GRFO..HH2 | 2016-01-07T00:00:00.008300Z - '
        ...      '2016-01-07T00:00:30.098300Z | 10.0 Hz, 301 samples',
        ...      'XX.GRFO..HH1 | 2016-01-07T00:00:02.668393Z - '
        ...      '2016-01-07T00:00:09.518393Z | 100.0 Hz, 686 samples',
        ...      'IU.ABCD..EH2 | 2016-01-07T00:00:09.528393Z - '
        ...      '2016-01-07T00:00:50.378393Z | 100.0 Hz, 4086 samples',
        ...      '', '']
        >>> s = os.linesep.join(s)
        >>> st = Stream._dummy_stream_from_string(s)
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        IU.GRFO..HH2 | 2016-01-07T00:00:00.008300Z ... | 10.0 Hz, 301 samples
        XX.GRFO..HH1 | 2016-01-07T00:00:02.668393Z ... | 100.0 Hz, 686 samples
        IU.ABCD..EH2 | 2016-01-07T00:00:09.528393Z ... | 100.0 Hz, 4086 samples
        """
        st = Stream()
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.match(r'[0-9]+ Trace\(s\) in Stream:', line):
                continue
            items = line.split()
            net, sta, loc, cha = items[0].split(".")
            starttime = UTCDateTime(items[2])
            sampling_rate = float(items[6])
            npts = int(items[8])
            tr = Trace()
            tr.data = np.ones(npts, dtype=np.float_)
            tr.stats.station = sta
            tr.stats.network = net
            tr.stats.location = loc
            tr.stats.channel = cha
            tr.stats.starttime = starttime
            tr.stats.sampling_rate = sampling_rate
            st += tr
        return st

    def _get_common_channels_info(self):
        """
        Returns a dictionary with information on common channels.
        """
        # get all ids down to location code
        ids_ = set([tr.id.rsplit(".", 1)[0] for tr in self])
        all_channels = {}
        # work can be separated by net.sta.loc, so iterate over each
        for id_ in ids_:
            net, sta, loc = id_.split(".")
            channels = {}
            st_ = self.select(network=net, station=sta, location=loc)
            # for each individual channel collect earliest start time and
            # latest endtime
            for tr in st_:
                cha = tr.stats.channel
                cha_common = cha and cha[:-1] or None
                cha_common_dict = channels.setdefault(cha_common, {})
                cha_dict = cha_common_dict.setdefault(cha, {})
                cha_dict["start"] = min(tr.stats.starttime.timestamp,
                                        cha_dict.get("start", np.inf))
                cha_dict["end"] = max(tr.stats.endtime.timestamp,
                                      cha_dict.get("end", -np.inf))
            # convert all timestamp objects back to UTCDateTime
            for cha_common_dict in channels.values():
                for cha_dict in cha_common_dict.values():
                    cha_dict["start"] = UTCDateTime(cha_dict["start"])
                    cha_dict["end"] = UTCDateTime(cha_dict["end"])
            # now for every combination of common channels determine earliest
            # common start time and latest common end time, as well as gap
            # information in between
            for cha_common, channels_ in channels.items():
                if cha_common is None:
                    cha_pattern = ""
                else:
                    cha_pattern = cha_common + "?"
                st__ = self.select(network=net, station=sta, location=loc,
                                   channel=cha_pattern)
                start = max(
                    [cha_dict_["start"] for cha_dict_ in channels_.values()])
                end = min(
                    [cha_dict_["end"] for cha_dict_ in channels_.values()])
                gaps = st__.get_gaps()
                all_channels[(net, sta, loc, cha_pattern)] = {
                    "start": start, "end": end, "gaps": gaps,
                    "channels": channels_}
        return all_channels

    def _groupby(self, group_by):
        """
        Group traces by same metadata.

        :param group_by: Group traces together which have the same metadata
            given by this parameter. The parameter should name the
            corresponding keys of the stats object,
            e.g. ``'{network}.{station}'``
            This parameter can take the value
            ``'id'`` which stacks groups the traces by SEED id

        :return: dictionary {group: stream}
        """
        if group_by == 'id':
            group_by = '{network}.{station}.{location}.{channel}'
        groups = collections.defaultdict(self.__class__)
        for tr in self:
            groups[group_by.format(**tr.stats)].append(tr)
        return dict(groups)

    def _trim_common_channels(self):
        """
        Trim all channels that have the same ID down to the component character
        to the earliest common start time and latest common end time. Works in
        place.
        """
        self._cleanup()
        channel_infos = self._get_common_channels_info()
        new_traces = []
        for (net, sta, loc, cha_pattern), infos in channel_infos.items():
            st = self.select(network=net, station=sta, location=loc,
                             channel=cha_pattern)
            st.trim(infos["start"], infos["end"])
            for _, _, _, _, start_, end_, _, _ in infos["gaps"]:
                st = st.cutout(start_, end_)
            new_traces += st.traces
        self.traces = new_traces

    def _rotate_to_zne(
            self, inventory, components=("Z12", "123")):
        """
        Rotate all matching traces to ZNE, specifying sets of component codes.

        >>> from obspy import read, read_inventory
        >>> st = read("/path/to/ffbx_unrotated_gaps.mseed")
        >>> inv = read_inventory("/path/to/ffbx.stationxml")
        >>> st._rotate_to_zne(inv)  # doctest: +ELLIPSIS
        <obspy.core.stream.Stream object at 0x...>

        :type inventory: :class:`~obspy.core.inventory.inventory.Inventory` or
            :class:`~obspy.io.xseed.parser.Parser`
        :param inventory: Inventory or Parser with metadata of channels.
        :type components: list or tuple or str
        :param components: List of combinations of three (case sensitive)
            component characters. Rotations are executed in this order, so
            order might matter in very strange cases (e.g. if traces with more
            than three component codes are present for the same SEED ID down to
            the component code). For example, specifying components ``"Z12"``
            would rotate sets of "BHZ", "BH1", "BH2" (and "HHZ", "HH1", "HH2",
            etc.) channels at the same station. If only a single set of
            component codes is used, this option can also be specified as a
            string (e.g. ``components='Z12'``).
        """
        # be nice to users that specify e.g. ``components='ZNE'``..
        # compare http://lists.swapbytes.de/archives/obspy-users/
        # 2018-March/002692.html
        if isinstance(components, (str, native_str)):
            components = [components]

        for component_pair in components:
            st = self.select(component="[{}]".format(component_pair))
            netstaloc = sorted(set(
                [(tr.stats.network, tr.stats.station, tr.stats.location)
                 for tr in st]))
            for net, sta, loc in netstaloc:
                channels = set(
                    [tr.stats.channel
                     for tr in st.select(network=net, station=sta,
                                         location=loc)])
                common_channels = {}
                for channel in channels:
                    if channel == "":
                        continue
                    cha_without_comp = channel[:-1]
                    component = channel[-1]
                    common_channels.setdefault(
                        cha_without_comp, set()).add(component)
                for cha_without_comp, components in sorted(
                        common_channels.items()):
                    if components == set(component_pair):
                        channels_ = [cha_without_comp + comp
                                     for comp in component_pair]
                        self._rotate_specific_channels_to_zne(
                            net, sta, loc, channels_, inventory)
        return self

    def _rotate_specific_channels_to_zne(
            self, network, station, location, channels, inventory):
        """
        Rotate three explicitly specified channels to ZNE.

        >>> from obspy import read, read_inventory
        >>> st = read("/path/to/ffbx_unrotated_gaps.mseed")
        >>> inv = read_inventory("/path/to/ffbx.stationxml")
        >>> st._rotate_specific_channels_to_zne(
        ...     "BW", "FFB1", "", ["HHZ", "HH1", "HH2"],
        ...     inv)  # doctest: +ELLIPSIS
        <obspy.core.stream.Stream object at 0x...>

        :type network: str
        :param network: Network code of channels that should be rotated.
        :type station: str
        :param station: Station code of channels that should be rotated.
        :type location: str
        :param location: Location code of channels that should be rotated.
        :type channels: list
        :param channels: The three channel codes of channels that should be
            rotated.
        :type inventory: :class:`~obspy.core.inventory.inventory.Inventory` or
            :class:`~obspy.io.xseed.parser.Parser`
        :param inventory: Inventory or Parser with metadata of channels.
        """
        from obspy.signal.rotate import rotate2zne
        from obspy.core.inventory import Inventory, Network
        from obspy.io.xseed import Parser

        if isinstance(inventory, (Inventory, Network)):
            metadata_getter = inventory.get_channel_metadata
        elif isinstance(inventory, Parser):
            # xseed Parser has everything in get_coordinates method due to
            # historic reasons..
            metadata_getter = inventory.get_coordinates
        else:
            msg = 'Wrong type for "inventory": {}'.format(str(type(inventory)))
            raise TypeError(msg)
        # build temporary stream that has only those traces that are supposed
        # to be used in rotation
        st = self.select(network=network, station=station, location=location)
        st = (st.select(channel=channels[0]) + st.select(channel=channels[1]) +
              st.select(channel=channels[2]))
        # remove the original unrotated traces from the stream
        for tr in st.traces:
            self.remove(tr)
        # cut data so that we end up with a set of matching pieces for the tree
        # components (i.e. cut away any parts where one of the three components
        # has no data)
        st._trim_common_channels()
        # sort by start time, so each three consecutive traces can then be used
        # in one rotation run
        st.sort(keys=["starttime"])
        # woooops, that's unexpected. must be a bug in the trimming helper
        # routine
        if len(st) % 3 != 0:
            msg = ("Unexpected behavior in rotation. Please file a bug "
                   "report on github.")
            raise NotImplementedError(msg)
        num_pieces = int(len(st) / 3)
        for i in range(num_pieces):
            # three consecutive traces are always the ones that combine for one
            # rotation run
            traces = [st.pop() for i in range(3)]
            # paranoid.. do a quick check of the channels again.
            if set([tr.stats.channel for tr in traces]) != set(channels):
                msg = ("Unexpected behavior in rotation. Please file a bug "
                       "report on github.")
                raise NotImplementedError(msg)
            # `.get_orientation()` works the same for Inventory and Parser
            orientation = [metadata_getter(tr.id, tr.stats.starttime)
                           for tr in traces]
            zne = rotate2zne(
                traces[0], orientation[0]["azimuth"], orientation[0]["dip"],
                traces[1], orientation[1]["azimuth"], orientation[1]["dip"],
                traces[2], orientation[2]["azimuth"], orientation[2]["dip"])
            for tr, new_data, component in zip(traces, zne, "ZNE"):
                tr.data = new_data
                tr.stats.channel = tr.stats.channel[:-1] + component
            self.traces += traces
        return self


def _is_pickle(filename):  # @UnusedVariable
    """
    Check whether a file is a pickled ObsPy Stream file.

    :type filename: str
    :param filename: Name of the pickled ObsPy Stream file to be checked.
    :rtype: bool
    :return: ``True`` if pickled file.

    .. rubric:: Example

    >>> _is_pickle('/path/to/pickle.file')  # doctest: +SKIP
    True
    """
    if isinstance(filename, (str, native_str)):
        try:
            with open(filename, 'rb') as fp:
                st = pickle.load(fp)
        except Exception:
            return False
    else:
        try:
            st = pickle.load(filename)
        except Exception:
            return False
    return isinstance(st, Stream)


def _read_pickle(filename, **kwargs):  # @UnusedVariable
    """
    Read and return Stream from pickled ObsPy Stream file.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: Name of the pickled ObsPy Stream file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.
    """
    kwargs = {}
    if PY3:
        # see seishub/client.py
        # https://api.mongodb.org/python/current/\
        # python3.html#why-can-t-i-share-pickled-objectids-\
        # between-some-versions-of-python-2-and-3
        kwargs['encoding'] = "latin-1"

    if isinstance(filename, (str, native_str)):
        with open(filename, 'rb') as fp:
            return pickle.load(fp, **kwargs)
    else:
        return pickle.load(filename, **kwargs)


def _write_pickle(stream, filename, protocol=2, **kwargs):  # @UnusedVariable
    """
    Write a Python pickle of current stream.

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
