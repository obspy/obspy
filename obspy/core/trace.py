# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Trace objects.

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

import functools
import inspect
import math
import warnings
from copy import copy, deepcopy

import numpy as np
import matplotlib.pyplot as plt

from obspy.core import compatibility
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import AttribDict, create_empty_data_chunk
from obspy.core.util.base import _get_function_from_entry_point
from obspy.core.util.decorator import (deprecated_keywords, raise_if_masked,
                                       skip_if_no_data)
from obspy.core.util.misc import flat_not_masked_contiguous, get_window_times


class Stats(AttribDict):
    """
    A container for additional header information of a ObsPy Trace object.

    A ``Stats`` object may contain all header information (also known as meta
    data) of a :class:`~obspy.core.trace.Trace` object. Those headers may be
    accessed or modified either in the dictionary style or directly via a
    corresponding attribute. There are various default attributes which are
    required by every waveform import and export modules within ObsPy such as
    :mod:`obspy.io.mseed`.

    :type header: dict or :class:`~obspy.core.trace.Stats`, optional
    :param header: Dictionary containing meta information of a single
        :class:`~obspy.core.trace.Trace` object. Possible keywords are
        summarized in the following `Default Attributes`_ section.

    .. rubric:: Basic Usage

    >>> stats = Stats()
    >>> stats.network = 'BW'
    >>> print(stats['network'])
    BW
    >>> stats['station'] = 'MANZ'
    >>> print(stats.station)
    MANZ

    .. rubric:: _`Default Attributes`

    ``sampling_rate`` : float, optional
        Sampling rate in hertz (default value is 1.0).
    ``delta`` : float, optional
        Sample distance in seconds (default value is 1.0).
    ``calib`` : float, optional
        Calibration factor (default value is 1.0).
    ``npts`` : int, optional
        Number of sample points (default value is 0, which implies that no data
        is present).
    ``network`` : string, optional
        Network code (default is an empty string).
    ``location`` : string, optional
        Location code (default is an empty string).
    ``station`` : string, optional
        Station code (default is an empty string).
    ``channel`` : string, optional
        Channel code (default is an empty string).
    ``starttime`` : :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        Date and time of the first data sample given in UTC (default value is
        "1970-01-01T00:00:00.0Z").
    ``endtime`` : :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        Date and time of the last data sample given in UTC
        (default value is "1970-01-01T00:00:00.0Z").

    .. rubric:: Notes

    (1) The attributes ``sampling_rate`` and ``delta`` are linked to each
        other. If one of the attributes is modified the other will be
        recalculated.

        >>> stats = Stats()
        >>> stats.sampling_rate
        1.0
        >>> stats.delta = 0.005
        >>> stats.sampling_rate
        200.0

    (2) The attributes ``starttime``, ``npts``, ``sampling_rate`` and ``delta``
        are monitored and used to automatically calculate the ``endtime``.

        >>> stats = Stats()
        >>> stats.npts = 60
        >>> stats.delta = 1.0
        >>> stats.starttime = UTCDateTime(2009, 1, 1, 12, 0, 0)
        >>> stats.endtime
        UTCDateTime(2009, 1, 1, 12, 0, 59)
        >>> stats.delta = 0.5
        >>> stats.endtime
        UTCDateTime(2009, 1, 1, 12, 0, 29, 500000)

    (3) The attribute ``endtime`` is read only and can not be modified.

        >>> stats = Stats()
        >>> stats.endtime = UTCDateTime(2009, 1, 1, 12, 0, 0)
        Traceback (most recent call last):
        ...
        AttributeError: Attribute "endtime" in Stats object is read only!
        >>> stats['endtime'] = UTCDateTime(2009, 1, 1, 12, 0, 0)
        Traceback (most recent call last):
        ...
        AttributeError: Attribute "endtime" in Stats object is read only!

    (4)
        The attribute ``npts`` will be automatically updated from the
        :class:`~obspy.core.trace.Trace` object.

        >>> trace = Trace()
        >>> trace.stats.npts
        0
        >>> trace.data = np.array([1, 2, 3, 4])
        >>> trace.stats.npts
        4
    """
    readonly = ['endtime']
    defaults = {
        'sampling_rate': 1.0,
        'delta': 1.0,
        'starttime': UTCDateTime(0),
        'endtime': UTCDateTime(0),
        'npts': 0,
        'calib': 1.0,
        'network': '',
        'station': '',
        'location': '',
        'channel': '',
    }

    def __init__(self, header={}):
        """
        """
        super(Stats, self).__init__(header)

    def __setitem__(self, key, value):
        """
        """
        # keys which need to refresh derived values
        if key in ['delta', 'sampling_rate', 'starttime', 'npts']:
            # ensure correct data type
            if key == 'delta':
                key = 'sampling_rate'
                value = 1.0 / float(value)
            elif key == 'sampling_rate':
                value = float(value)
            elif key == 'starttime':
                value = UTCDateTime(value)
            elif key == 'npts':
                value = int(value)
            # set current key
            super(Stats, self).__setitem__(key, value)
            # set derived value: delta
            try:
                delta = 1.0 / float(self.sampling_rate)
            except ZeroDivisionError:
                delta = 0
            self.__dict__['delta'] = delta
            # set derived value: endtime
            if self.npts == 0:
                timediff = 0
            else:
                timediff = (self.npts - 1) * delta
            self.__dict__['endtime'] = self.starttime + timediff
            return
        # prevent a calibration factor of 0
        if key == 'calib' and value == 0:
            msg = 'Calibration factor set to 0.0!'
            warnings.warn(msg, UserWarning)
        # all other keys
        if isinstance(value, dict):
            super(Stats, self).__setitem__(key, AttribDict(value))
        else:
            super(Stats, self).__setitem__(key, value)

    __setattr__ = __setitem__

    def __str__(self):
        """
        Return better readable string representation of Stats object.
        """
        priorized_keys = ['network', 'station', 'location', 'channel',
                          'starttime', 'endtime', 'sampling_rate', 'delta',
                          'npts', 'calib']
        return self._pretty_str(priorized_keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


def _add_processing_info(func):
    """
    This is a decorator that attaches information about a processing call as a
    string to the Trace.stats.processing list.
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        callargs = inspect.getcallargs(func, *args, **kwargs)
        callargs.pop("self")
        kwargs_ = callargs.pop("kwargs", {})
        from obspy import __version__
        info = "ObsPy {version}: {function}(%s)".format(
            version=__version__,
            function=func.__name__)
        arguments = []
        arguments += \
            ["%s=%s" % (k, v) if not isinstance(v, native_str) else
             "%s='%s'" % (k, v) for k, v in callargs.items()]
        arguments += \
            ["%s=%s" % (k, v) if not isinstance(v, native_str) else
             "%s='%s'" % (k, v) for k, v in kwargs_.items()]
        arguments.sort()
        info = info % "::".join(arguments)
        self = args[0]
        result = func(*args, **kwargs)
        # Attach after executing the function to avoid having it attached
        # while the operation failed.
        self._addProcessingInfo(info)
        return result

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


class Trace(object):
    """
    An object containing data of a continuous series, such as a seismic trace.

    :type data: :class:`~numpy.ndarray` or :class:`~numpy.ma.MaskedArray`
    :param data: Array of data samples
    :type header: dict or :class:`~obspy.core.trace.Stats`
    :param header: Dictionary containing header fields

    :var id: A SEED compatible identifier of the trace.
    :var stats: A container :class:`~obspy.core.trace.Stats` for additional
        header information of the trace.
    :var data: Data samples in a :class:`~numpy.ndarray` or
        :class:`~numpy.ma.MaskedArray`

    .. rubric:: Supported Operations

    ``trace = traceA + traceB``
        Merges traceA and traceB into one new trace object.
        See also: :meth:`Trace.__add__`.
    ``len(trace)``
        Returns the number of samples contained in the trace. That is
        it es equal to ``len(trace.data)``.
        See also: :meth:`Trace.__len__`.
    ``str(trace)``
        Returns basic information about the trace object.
        See also: :meth:`Trace.__str__`.
    """

    def __init__(self, data=np.array([]), header=None):
        # make sure Trace gets initialized with suitable ndarray as self.data
        # otherwise we could end up with e.g. a list object in self.data
        _data_sanity_checks(data)
        # set some defaults if not set yet
        if header is None:
            # Default values: For detail see
            # http://www.obspy.org/wiki/\
            # KnownIssues#DefaultParameterValuesinPython
            header = {}
        header.setdefault('npts', len(data))
        self.stats = Stats(header)
        # set data without changing npts in stats object (for headonly option)
        super(Trace, self).__setattr__('data', data)

    @property
    def meta(self):
        return self.stats

    @meta.setter
    def meta(self, value):
        self.stats = value

    def __eq__(self, other):
        """
        Implements rich comparison of Trace objects for "==" operator.

        Traces are the same, if both their data and stats are the same.
        """
        # check if other object is a Trace
        if not isinstance(other, Trace):
            return False
        # comparison of Stats objects is supported by underlying AttribDict
        if not self.stats == other.stats:
            return False
        # comparison of ndarrays is supported by NumPy
        if not np.array_equal(self, other):
            return False

        return True

    def __ne__(self, other):
        """
        Implements rich comparison of Trace objects for "!=" operator.

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

    def __nonzero__(self):
        """
        No data means no trace.
        """
        return bool(len(self.data))

    def __str__(self, id_length=None):
        """
        Return short summary string of the current trace.

        :rtype: str
        :return: Short summary string of the current trace containing the SEED
            identifier, start time, end time, sampling rate and number of
            points of the current trace.

        .. rubric:: Example

        >>> tr = Trace(header={'station':'FUR', 'network':'GR'})
        >>> str(tr)  # doctest: +ELLIPSIS
        'GR.FUR.. | 1970-01-01T00:00:00.000000Z - ... | 1.0 Hz, 0 samples'
        """
        # set fixed id width
        if id_length:
            out = "%%-%ds" % (id_length)
            trace_id = out % self.id
        else:
            trace_id = "%s" % self.id
        out = ''
        # output depending on delta or sampling rate bigger than one
        if self.stats.sampling_rate < 0.1:
            if hasattr(self.stats, 'preview') and self.stats.preview:
                out = out + ' | '\
                    "%(starttime)s - %(endtime)s | " + \
                    "%(delta).1f s, %(npts)d samples [preview]"
            else:
                out = out + ' | '\
                    "%(starttime)s - %(endtime)s | " + \
                    "%(delta).1f s, %(npts)d samples"
        else:
            if hasattr(self.stats, 'preview') and self.stats.preview:
                out = out + ' | '\
                    "%(starttime)s - %(endtime)s | " + \
                    "%(sampling_rate).1f Hz, %(npts)d samples [preview]"
            else:
                out = out + ' | '\
                    "%(starttime)s - %(endtime)s | " + \
                    "%(sampling_rate).1f Hz, %(npts)d samples"
        # check for masked array
        if np.ma.count_masked(self.data):
            out += ' (masked)'
        return trace_id + out % (self.stats)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __len__(self):
        """
        Return number of data samples of the current trace.

        :rtype: int
        :return: Number of data samples.

        .. rubric:: Example

        >>> trace = Trace(data=np.array([1, 2, 3, 4]))
        >>> trace.count()
        4
        >>> len(trace)
        4
        """
        return len(self.data)

    count = __len__

    def __setattr__(self, key, value):
        """
        __setattr__ method of Trace object.
        """
        # any change in Trace.data will dynamically set Trace.stats.npts
        if key == 'data':
            _data_sanity_checks(value)
            self.stats.npts = len(value)
        return super(Trace, self).__setattr__(key, value)

    def __getitem__(self, index):
        """
        __getitem__ method of Trace object.

        :rtype: list
        :return: List of data points
        """
        return self.data[index]

    def __mul__(self, num):
        """
        Create a new Stream containing num copies of this trace.

        :type num: int
        :param num: Number of copies.
        :returns: New ObsPy Stream object.

        .. rubric:: Example

        >>> from obspy import read
        >>> tr = read()[0]
        >>> st = tr * 5
        >>> len(st)
        5
        """
        if not isinstance(num, int):
            raise TypeError("Integer expected")
        from obspy import Stream
        st = Stream()
        for _i in range(num):
            st += self.copy()
        return st

    def __div__(self, num):
        """
        Split Trace into new Stream containing num Traces of the same size.

        :type num: int
        :param num: Number of traces in returned Stream. Last trace may contain
            lesser samples.
        :returns: New ObsPy Stream object.

        .. rubric:: Example

        >>> from obspy import read
        >>> tr = read()[0]
        >>> print(tr)  # doctest: +ELLIPSIS
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> st = tr / 7
        >>> print(st)  # doctest: +ELLIPSIS
        7 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 429 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:07.290000Z ... | 100.0 Hz, 429 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:11.580000Z ... | 100.0 Hz, 429 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:15.870000Z ... | 100.0 Hz, 429 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:20.160000Z ... | 100.0 Hz, 429 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:24.450000Z ... | 100.0 Hz, 429 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:28.740000Z ... | 100.0 Hz, 426 samples
        """
        if not isinstance(num, int):
            raise TypeError("Integer expected")
        from obspy import Stream
        total_length = np.size(self.data)
        rest_length = total_length % num
        if rest_length:
            packet_length = (total_length // num)
        else:
            packet_length = (total_length // num) - 1
        tstart = self.stats.starttime
        tend = tstart + (self.stats.delta * packet_length)
        st = Stream()
        for _i in range(num):
            st.append(self.slice(tstart, tend).copy())
            tstart = tend + self.stats.delta
            tend = tstart + (self.stats.delta * packet_length)
        return st

    # Py3k: '/' does not map to __div__ anymore in Python 3
    __truediv__ = __div__

    def __mod__(self, num):
        """
        Split Trace into new Stream containing Traces with num samples.

        :type num: int
        :param num: Number of samples in each trace in returned Stream. Last
            trace may contain lesser samples.
        :returns: New ObsPy Stream object.

        .. rubric:: Example

        >>> from obspy import read
        >>> tr = read()[0]
        >>> print(tr)  # doctest: +ELLIPSIS
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> st = tr % 800
        >>> print(st)  # doctest: +ELLIPSIS
        4 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 800 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:11.000000Z ... | 100.0 Hz, 800 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:19.000000Z ... | 100.0 Hz, 800 samples
        BW.RJOB..EHZ | 2009-08-24T00:20:27.000000Z ... | 100.0 Hz, 600 samples
        """
        if not isinstance(num, int):
            raise TypeError("Integer expected")
        elif num <= 0:
            raise ValueError("Positive Integer expected")
        from obspy import Stream
        st = Stream()
        total_length = np.size(self.data)
        if num >= total_length:
            st.append(self.copy())
            return st
        tstart = self.stats.starttime
        tend = tstart + (self.stats.delta * (num - 1))
        while True:
            st.append(self.slice(tstart, tend).copy())
            tstart = tend + self.stats.delta
            tend = tstart + (self.stats.delta * (num - 1))
            if tstart > self.stats.endtime:
                break
        return st

    def __add__(self, trace, method=0, interpolation_samples=0,
                fill_value=None, sanity_checks=True):
        """
        Add another Trace object to current trace.

        :type method: int, optional
        :param method: Method to handle overlaps of traces. Defaults to ``0``.
            See the `Handling Overlaps`_ section below for further details.
        :type fill_value: int, float, str or ``None``, optional
        :param fill_value: Fill value for gaps. Defaults to ``None``. Traces
            will be converted to NumPy masked arrays if no value is given and
            gaps are present. If the keyword ``'latest'`` is provided it will
            use the latest value before the gap. If keyword ``'interpolate'``
            is provided, missing values are linearly interpolated (not
            changing the data type e.g. of integer valued traces).
            See the `Handling Gaps`_ section below for further details.
        :type interpolation_samples: int, optional
        :param interpolation_samples: Used only for ``method=1``. It specifies
            the number of samples which are used to interpolate between
            overlapping traces. Defaults to ``0``. If set to ``-1`` all
            overlapping samples are interpolated.
        :type sanity_checks: bool, optional
        :param sanity_checks: Enables some sanity checks before merging traces.
            Defaults to ``True``.

        Trace data will be converted into a NumPy masked array data type if
        any gaps are present. This behavior may be prevented by setting the
        ``fill_value`` parameter. The ``method`` argument controls the
        handling of overlapping data values.

        Sampling rate, data type and trace.id of both traces must match.

        .. rubric:: _`Handling Overlaps`

        ======  ===============================================================
        Method  Description
        ======  ===============================================================
        0       Discard overlapping data. Overlaps are essentially treated the
                same way as gaps::

                    Trace 1: AAAAAAAA
                    Trace 2:     FFFFFFFF
                    1 + 2  : AAAA----FFFF

                Contained traces with differing data will be marked as gap::

                    Trace 1: AAAAAAAAAAAA
                    Trace 2:     FF
                    1 + 2  : AAAA--AAAAAA

                Missing data can be merged in from a different trace::

                    Trace 1: AAAA--AAAAAA (contained trace, missing samples)
                    Trace 2:     FF
                    1 + 2  : AAAAFFAAAAAA
        1       Discard data of the previous trace assuming the following trace
                contains data with a more correct time value. The parameter
                ``interpolation_samples`` specifies the number of samples used
                to linearly interpolate between the two traces in order to
                prevent steps. Note that if there are gaps inside, the
                returned array is still a masked array, only if ``fill_value``
                is set, the returned array is a normal array and gaps are
                filled with fill value.

                No interpolation (``interpolation_samples=0``)::

                    Trace 1: AAAAAAAA
                    Trace 2:     FFFFFFFF
                    1 + 2  : AAAAFFFFFFFF

                Interpolate first two samples (``interpolation_samples=2``)::

                    Trace 1: AAAAAAAA
                    Trace 2:     FFFFFFFF
                    1 + 2  : AAAACDFFFFFF (interpolation_samples=2)

                Interpolate all samples (``interpolation_samples=-1``)::

                    Trace 1: AAAAAAAA
                    Trace 2:     FFFFFFFF
                    1 + 2  : AAAABCDEFFFF

                Any contained traces with different data will be discarded::

                    Trace 1: AAAAAAAAAAAA (contained trace)
                    Trace 2:     FF
                    1 + 2  : AAAAAAAAAAAA

                Missing data can be merged in from a different trace::

                    Trace 1: AAAA--AAAAAA (contained trace, missing samples)
                    Trace 2:     FF
                    1 + 2  : AAAAFFAAAAAA
        ======  ===============================================================

        .. rubric:: _`Handling gaps`

        1. Traces with gaps and ``fill_value=None`` (default)::

            Trace 1: AAAA
            Trace 2:         FFFF
            1 + 2  : AAAA----FFFF

        2. Traces with gaps and given ``fill_value=0``::

            Trace 1: AAAA
            Trace 2:         FFFF
            1 + 2  : AAAA0000FFFF

        3. Traces with gaps and given ``fill_value='latest'``::

            Trace 1: ABCD
            Trace 2:         FFFF
            1 + 2  : ABCDDDDDFFFF

        4. Traces with gaps and given ``fill_value='interpolate'``::

            Trace 1: AAAA
            Trace 2:         FFFF
            1 + 2  : AAAABCDEFFFF
        """
        if sanity_checks:
            if not isinstance(trace, Trace):
                raise TypeError
            #  check id
            if self.getId() != trace.getId():
                raise TypeError("Trace ID differs")
            #  check sample rate
            if self.stats.sampling_rate != trace.stats.sampling_rate:
                raise TypeError("Sampling rate differs")
            #  check calibration factor
            if self.stats.calib != trace.stats.calib:
                raise TypeError("Calibration factor differs")
            # check data type
            if self.data.dtype != trace.data.dtype:
                raise TypeError("Data type differs")
        # check times
        if self.stats.starttime <= trace.stats.starttime:
            lt = self
            rt = trace
        else:
            rt = self
            lt = trace
        # check whether to use the latest value to fill a gap
        if fill_value == "latest":
            fill_value = lt.data[-1]
        elif fill_value == "interpolate":
            fill_value = (lt.data[-1], rt.data[0])
        sr = self.stats.sampling_rate
        delta = (rt.stats.starttime - lt.stats.endtime) * sr
        delta = int(compatibility.round_away(delta)) - 1
        delta_endtime = lt.stats.endtime - rt.stats.endtime
        # create the returned trace
        out = self.__class__(header=deepcopy(lt.stats))
        # check if overlap or gap
        if delta < 0 and delta_endtime < 0:
            # overlap
            delta = abs(delta)
            if np.all(np.equal(lt.data[-delta:], rt.data[:delta])):
                # check if data are the same
                data = [lt.data[:-delta], rt.data]
            elif method == 0:
                overlap = create_empty_data_chunk(delta, lt.data.dtype,
                                                  fill_value)
                data = [lt.data[:-delta], overlap, rt.data[delta:]]
            elif method == 1 and interpolation_samples >= -1:
                try:
                    ls = lt.data[-delta - 1]
                except:
                    ls = lt.data[0]
                if interpolation_samples == -1:
                    interpolation_samples = delta
                elif interpolation_samples > delta:
                    interpolation_samples = delta
                try:
                    rs = rt.data[interpolation_samples]
                except IndexError:
                    # contained trace
                    data = [lt.data]
                else:
                    # include left and right sample (delta + 2)
                    interpolation = np.linspace(ls, rs,
                                                interpolation_samples + 2)
                    # cut ls and rs and ensure correct data type
                    interpolation = np.require(interpolation[1:-1],
                                               lt.data.dtype)
                    data = [lt.data[:-delta], interpolation,
                            rt.data[interpolation_samples:]]
            else:
                raise NotImplementedError
        elif delta < 0 and delta_endtime >= 0:
            # contained trace
            delta = abs(delta)
            lenrt = len(rt)
            t1 = len(lt) - delta
            t2 = t1 + lenrt
            # check if data are the same
            data_equal = (lt.data[t1:t2] == rt.data)
            # force a masked array and fill it for check of equality of valid
            # data points
            if np.all(np.ma.masked_array(data_equal).filled()):
                # if all (unmasked) data are equal,
                if isinstance(data_equal, np.ma.masked_array):
                    x = np.ma.masked_array(lt.data[t1:t2])
                    y = np.ma.masked_array(rt.data)
                    data_same = np.choose(x.mask, [x, y])
                    data = np.choose(x.mask & y.mask, [data_same, np.nan])
                    if np.any(np.isnan(data)):
                        data = np.ma.masked_invalid(data)
                    # convert back to maximum dtype of original data
                    dtype = np.max((x.dtype, y.dtype))
                    data = data.astype(dtype)
                    data = [lt.data[:t1], data, lt.data[t2:]]
                else:
                    data = [lt.data]
            elif method == 0:
                gap = create_empty_data_chunk(lenrt, lt.data.dtype, fill_value)
                data = [lt.data[:t1], gap, lt.data[t2:]]
            elif method == 1:
                data = [lt.data]
            else:
                raise NotImplementedError
        elif delta == 0:
            # exact fit - merge both traces
            data = [lt.data, rt.data]
        else:
            # gap
            # use fixed value or interpolate in between
            gap = create_empty_data_chunk(delta, lt.data.dtype, fill_value)
            data = [lt.data, gap, rt.data]
        # merge traces depending on NumPy array type
        if True in [isinstance(_i, np.ma.masked_array) for _i in data]:
            data = np.ma.concatenate(data)
        else:
            data = np.concatenate(data)
            data = np.require(data, dtype=lt.data.dtype)
        # Check if we can downgrade to normal ndarray
        if isinstance(data, np.ma.masked_array) and \
           np.ma.count_masked(data) == 0:
            data = data.compressed()
        out.data = data
        return out

    def getId(self):
        """
        Return a SEED compatible identifier of the trace.

        :rtype: str
        :return: SEED identifier

        The SEED identifier contains the network, station, location and channel
        code for the current Trace object.

        .. rubric:: Example

        >>> meta = {'station': 'MANZ', 'network': 'BW', 'channel': 'EHZ'}
        >>> tr = Trace(header=meta)
        >>> print(tr.getId())
        BW.MANZ..EHZ
        >>> print(tr.id)
        BW.MANZ..EHZ
        """
        out = "%(network)s.%(station)s.%(location)s.%(channel)s"
        return out % (self.stats)

    id = property(getId)

    def plot(self, **kwargs):
        """
        Create a simple graph of the current trace.

        Various options are available to change the appearance of the waveform
        plot. Please see :meth:`~obspy.core.stream.Stream.plot` method for all
        possible options.

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> tr = st[0]
        >>> tr.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            tr = st[0]
            tr.plot()
        """
        from obspy.imaging.waveform import WaveformPlotting
        waveform = WaveformPlotting(stream=self, **kwargs)
        return waveform.plot_waveform()

    def spectrogram(self, **kwargs):
        """
        Create a spectrogram plot of the trace.

        For details on kwargs that can be used to customize the spectrogram
        plot see :func:`~obspy.imaging.spectrogram.spectrogram`.

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> tr = st[0]
        >>> tr.spectrogram()  # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            tr = st[0]
            tr.spectrogram(sphinx=True)
        """
        # set some default values
        if 'samp_rate' not in kwargs:
            kwargs['samp_rate'] = self.stats.sampling_rate
        if 'title' not in kwargs:
            kwargs['title'] = str(self)
        from obspy.imaging.spectrogram import spectrogram
        return spectrogram(data=self.data, **kwargs)

    def write(self, filename, format, **kwargs):
        """
        Save current trace into a file.

        :type filename: str
        :param filename: The name of the file to write.
        :type format: str
        :param format: The format to write must be specified. See
            :meth:`obspy.core.stream.Stream.write` method for possible
            formats.
        :param kwargs: Additional keyword arguments passed to the underlying
            waveform writer method.

        .. rubric:: Example

        >>> tr = Trace()
        >>> tr.write("out.mseed", format="MSEED")  # doctest: +SKIP
        """
        # we need to import here in order to prevent a circular import of
        # Stream and Trace classes
        from obspy import Stream
        Stream([self]).write(filename, format, **kwargs)

    def _ltrim(self, starttime, pad=False, nearest_sample=True,
               fill_value=None):
        """
        Cut current trace to given start time. For more info see
        :meth:`~obspy.core.trace.Trace.trim`.

        .. rubric:: Example

        >>> tr = Trace(data=np.arange(0, 10))
        >>> tr.stats.delta = 1.0
        >>> tr._ltrim(tr.stats.starttime + 8)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.data
        array([8, 9])
        >>> tr.stats.starttime
        UTCDateTime(1970, 1, 1, 0, 0, 8)
        """
        org_dtype = self.data.dtype
        if isinstance(starttime, float) or isinstance(starttime, int):
            starttime = UTCDateTime(self.stats.starttime) + starttime
        elif not isinstance(starttime, UTCDateTime):
            raise TypeError
        # check if in boundary
        if nearest_sample:
            delta = compatibility.round_away(
                (starttime - self.stats.starttime) * self.stats.sampling_rate)
            # due to rounding and npts starttime must always be right of
            # self.stats.starttime, rtrim relies on it
            if delta < 0 and pad:
                npts = abs(delta) + 10  # use this as a start
                newstarttime = self.stats.starttime - npts / \
                    float(self.stats.sampling_rate)
                newdelta = compatibility.round_away(
                    (starttime - newstarttime) * self.stats.sampling_rate)
                delta = newdelta - npts
            delta = int(delta)
        else:
            delta = int(math.floor(round((self.stats.starttime - starttime) *
                                   self.stats.sampling_rate, 7))) * -1
        # Adjust starttime only if delta is greater than zero or if the values
        # are padded with masked arrays.
        if delta > 0 or pad:
            self.stats.starttime += delta * self.stats.delta
        if delta == 0 or (delta < 0 and not pad):
            return self
        elif delta < 0 and pad:
            try:
                gap = create_empty_data_chunk(abs(delta), self.data.dtype,
                                              fill_value)
            except ValueError:
                # create_empty_data_chunk returns negative ValueError ?? for
                # too large number of points, e.g. 189336539799
                raise Exception("Time offset between starttime and "
                                "trace.starttime too large")
            self.data = np.ma.concatenate((gap, self.data))
            return self
        elif starttime > self.stats.endtime:
            self.data = np.empty(0, dtype=org_dtype)
            return self
        elif delta > 0:
            try:
                self.data = self.data[delta:]
            except IndexError:
                # a huge numbers for delta raises an IndexError
                # here we just create empty array with same dtype
                self.data = np.empty(0, dtype=org_dtype)
        return self

    def _rtrim(self, endtime, pad=False, nearest_sample=True, fill_value=None):
        """
        Cut current trace to given end time. For more info see
        :meth:`~obspy.core.trace.Trace.trim`.

        .. rubric:: Example

        >>> tr = Trace(data=np.arange(0, 10))
        >>> tr.stats.delta = 1.0
        >>> tr._rtrim(tr.stats.starttime + 2)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.data
        array([0, 1, 2])
        >>> tr.stats.endtime
        UTCDateTime(1970, 1, 1, 0, 0, 2)
        """
        org_dtype = self.data.dtype
        if isinstance(endtime, float) or isinstance(endtime, int):
            endtime = UTCDateTime(self.stats.endtime) - endtime
        elif not isinstance(endtime, UTCDateTime):
            raise TypeError
        # check if in boundary
        if nearest_sample:
            delta = compatibility.round_away(
                (endtime - self.stats.starttime) *
                self.stats.sampling_rate) - self.stats.npts + 1
            delta = int(delta)
        else:
            # solution for #127, however some tests need to be changed
            # delta = -1*int(math.floor(compatibility.round_away(
            #     (self.stats.endtime - endtime) * \
            #     self.stats.sampling_rate, 7)))
            delta = int(math.floor(round((endtime - self.stats.endtime) *
                                   self.stats.sampling_rate, 7)))
        if delta == 0 or (delta > 0 and not pad):
            return self
        if delta > 0 and pad:
            try:
                gap = create_empty_data_chunk(delta, self.data.dtype,
                                              fill_value)
            except ValueError:
                # create_empty_data_chunk returns negative ValueError ?? for
                # too large number of points, e.g. 189336539799
                raise Exception("Time offset between starttime and " +
                                "trace.starttime too large")
            self.data = np.ma.concatenate((self.data, gap))
            return self
        elif endtime < self.stats.starttime:
            self.stats.starttime = self.stats.endtime + \
                delta * self.stats.delta
            self.data = np.empty(0, dtype=org_dtype)
            return self
        # cut from right
        delta = abs(delta)
        total = len(self.data) - delta
        if endtime == self.stats.starttime:
            total = 1
        self.data = self.data[:total]
        return self

    @_add_processing_info
    def trim(self, starttime=None, endtime=None, pad=False,
             nearest_sample=True, fill_value=None):
        """
        Cut current trace to given start and end time.

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
            selected, if set to ``False``, the outer (previous sample for a
            start time border, next sample for an end time border) sample
            containing the time is selected. Defaults to ``True``.

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
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.

        .. rubric:: Example

        >>> tr = Trace(data=np.arange(0, 10))
        >>> tr.stats.delta = 1.0
        >>> t = tr.stats.starttime
        >>> tr.trim(t + 2.000001, t + 7.999999)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.data
        array([2, 3, 4, 5, 6, 7, 8])
        """
        # check time order and swap eventually
        if starttime and endtime and starttime > endtime:
            raise ValueError("startime is larger than endtime")
        # cut it
        if starttime:
            self._ltrim(starttime, pad, nearest_sample=nearest_sample,
                        fill_value=fill_value)
        if endtime:
            self._rtrim(endtime, pad, nearest_sample=nearest_sample,
                        fill_value=fill_value)
        # if pad=True and fill_value is given convert to NumPy ndarray
        if pad is True and fill_value is not None:
            try:
                self.data = self.data.filled()
            except AttributeError:
                # numpy.ndarray object has no attribute 'filled' - ignoring
                pass
        return self

    def slice(self, starttime=None, endtime=None, nearest_sample=True):
        """
        Return a new Trace object with data going from start to end time.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Specify the start time of slice.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Specify the end time of slice.
        :type nearest_sample: bool, optional
        :param nearest_sample: If set to ``True``, the closest sample is
            selected, if set to ``False``, the outer (previous sample for a
            start time border, next sample for an end time border) sample
            containing the time is selected. Defaults to ``True``.

            Given the following trace containing 4 samples, "|" are the
            sample points, "A" is the requested starttime::

                |        A|         |         |

            ``nearest_sample=True`` will select the second sample point,
            ``nearest_sample=False`` will select the first sample point.

        :return: New :class:`~obspy.core.trace.Trace` object. Does not copy
            data but just passes a reference to it.

        .. rubric:: Example

        >>> tr = Trace(data=np.arange(0, 10))
        >>> tr.stats.delta = 1.0
        >>> t = tr.stats.starttime
        >>> tr2 = tr.slice(t + 2, t + 8)
        >>> tr2.data
        array([2, 3, 4, 5, 6, 7, 8])
        """
        tr = copy(self)
        tr.stats = deepcopy(self.stats)
        tr.trim(starttime=starttime, endtime=endtime,
                nearest_sample=nearest_sample)
        return tr

    def slide(self, window_length, step, offset=0,
              include_partial_windows=False, nearest_sample=True):
        """
        Generator yielding equal length sliding windows of the Trace.

        Please keep in mind that it only returns a new view of the original
        data. Any modifications are applied to the original data as well. If
        you don't want this you have to create a copy of the yielded
        windows. Also be aware that if you modify the original data and you
        have overlapping windows, all following windows are affected as well.

        .. rubric:: Example

        >>> import obspy
        >>> tr = obspy.read()[0]
        >>> for windowed_tr in tr.slide(window_length=10.0, step=10.0):
        ...     print("---")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        ...     print(windowed_tr)
        ---
        ... | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:13.000000Z | ...
        ---
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
            selected, if set to ``False``, the outer (previous sample for a
            start time border, next sample for an end time border) sample
            containing the time is selected. Defaults to ``True``.

            Given the following trace containing 4 samples, "|" are the
            sample points, "A" is the requested starttime::

                |        A|         |         |

            ``nearest_sample=True`` will select the second sample point,
            ``nearest_sample=False`` will select the first sample point.
        :type nearest_sample: bool, optional
        """
        windows = get_window_times(
            starttime=self.stats.starttime,
            endtime=self.stats.endtime,
            window_length=window_length,
            step=step,
            offset=offset,
            include_partial_windows=include_partial_windows)

        if len(windows) < 1:
            raise StopIteration

        for start, stop in windows:
            yield self.slice(start, stop,
                             nearest_sample=nearest_sample)

        raise StopIteration

    def verify(self):
        """
        Verify current trace object against available meta data.

        .. rubric:: Example

        >>> tr = Trace(data=np.array([1,2,3,4]))
        >>> tr.stats.npts = 100
        >>> tr.verify()  #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        Exception: ntps(100) differs from data size(4)
        """
        if len(self) != self.stats.npts:
            msg = "ntps(%d) differs from data size(%d)"
            raise Exception(msg % (self.stats.npts, len(self.data)))
        delta = self.stats.endtime - self.stats.starttime
        if delta < 0:
            msg = "End time(%s) before start time(%s)"
            raise Exception(msg % (self.stats.endtime, self.stats.starttime))
        sr = self.stats.sampling_rate
        if self.stats.starttime != self.stats.endtime:
            if int(compatibility.round_away(delta * sr)) + 1 != len(self.data):
                msg = "Sample rate(%f) * time delta(%.4lf) + 1 != data len(%d)"
                raise Exception(msg % (sr, delta, len(self.data)))
            # Check if the endtime fits the starttime, npts and sampling_rate.
            if self.stats.endtime != self.stats.starttime + \
                    (self.stats.npts - 1) / float(self.stats.sampling_rate):
                msg = "End time is not the time of the last sample."
                raise Exception(msg)
        elif self.stats.npts not in [0, 1]:
            msg = "Data size should be 0, but is %d"
            raise Exception(msg % self.stats.npts)
        if not isinstance(self.stats, Stats):
            msg = "Attribute stats must be an instance of obspy.core.Stats"
            raise Exception(msg)
        if isinstance(self.data, np.ndarray) and \
           self.data.dtype.byteorder not in ["=", "|"]:
            msg = "Trace data should be stored as numpy.ndarray in the " + \
                  "system specific byte order."
            raise Exception(msg)
        return self

    @_add_processing_info
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
        `paz_remove` and/or simulates a new instrument response given by
        `paz_simulate`.
        For additional information and more options to control the instrument
        correction/simulation (e.g. water level, demeaning, tapering, ...) see
        :func:`~obspy.signal.invsim.simulate_seismometer`.

        `paz_remove` and `paz_simulate` are expected to be dictionaries
        containing information on poles, zeros and gain (and usually also
        sensitivity).

        If both `paz_remove` and `paz_simulate` are specified, both steps are
        performed in one go in the frequency domain, otherwise only the
        specified step is performed.

        .. note::

            Instead of the builtin deconvolution based on Poles and Zeros
            information, the deconvolution can be performed using evalresp
            instead by using the option `seedresp` (see documentation of
            :func:`~obspy.signal.invsim.simulate_seismometer` and the `ObsPy
            Tutorial <http://docs.obspy.org/master/tutorial/code_snippets/\
seismometer_correction_simulation.html#using-a-resp-file>`_.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.

        .. rubric:: Example

        >>> from obspy import read
        >>> from obspy.signal.invsim import corn_freq_2_paz
        >>> st = read()
        >>> tr = st[0]
        >>> paz_sts2 = {'poles': [-0.037004+0.037016j, -0.037004-0.037016j,
        ...                       -251.33+0j,
        ...                       -131.04-467.29j, -131.04+467.29j],
        ...             'zeros': [0j, 0j],
        ...             'gain': 60077000.0,
        ...             'sensitivity': 2516778400.0}
        >>> paz_1hz = corn_freq_2_paz(1.0, damp=0.707)
        >>> paz_1hz['sensitivity'] = 1.0
        >>> tr.simulate(paz_remove=paz_sts2, paz_simulate=paz_1hz)
        ... # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read
            from obspy.signal.invsim import corn_freq_2_paz
            st = read()
            tr = st[0]
            paz_sts2 = {'poles': [-0.037004+0.037016j, -0.037004-0.037016j,
                                  -251.33+0j,
                                  -131.04-467.29j, -131.04+467.29j],
                        'zeros': [0j, 0j],
                        'gain': 60077000.0,
                        'sensitivity': 2516778400.0}
            paz_1hz = corn_freq_2_paz(1.0, damp=0.707)
            paz_1hz['sensitivity'] = 1.0
            tr.simulate(paz_remove=paz_sts2, paz_simulate=paz_1hz)
            tr.plot()
        """
        # XXX accepting string "self" and using attached PAZ then
        if paz_remove == 'self':
            paz_remove = self.stats.paz

        # some convenience handling for evalresp type instrument correction
        if "seedresp" in kwargs:
            seedresp = kwargs["seedresp"]
            # if date is missing use trace's starttime
            seedresp.setdefault("date", self.stats.starttime)
            # if a Parser object is provided, get corresponding RESP
            # information
            from obspy.io.xseed import Parser
            if isinstance(seedresp['filename'], Parser):
                seedresp = deepcopy(seedresp)
                kwargs['seedresp'] = seedresp
                resp_key = ".".join(("RESP", self.stats.network,
                                     self.stats.station, self.stats.location,
                                     self.stats.channel))
                for key, stringio in seedresp['filename'].get_RESP():
                    if key == resp_key:
                        stringio.seek(0, 0)
                        seedresp['filename'] = stringio
                        break
                else:
                    msg = "Response for %s not found in Parser" % self.id
                    raise ValueError(msg)
            # Set the SEED identifiers!
            for item in ["network", "station", "location", "channel"]:
                seedresp[item] = self.stats[item]

        from obspy.signal.invsim import simulate_seismometer
        self.data = simulate_seismometer(
            self.data, self.stats.sampling_rate, paz_remove=paz_remove,
            paz_simulate=paz_simulate, remove_sensitivity=remove_sensitivity,
            simulate_sensitivity=simulate_sensitivity, **kwargs)

        return self

    @_add_processing_info
    def filter(self, type, **options):
        """
        Filter the data of the current trace.

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
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.

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

        ``'lowpassFIR'`` (experimental)
            FIR-Lowpass (uses :func:`obspy.signal.filter.lowpassFIR`).

        ``'remezFIR'`` (experimental)
            Minimax optimal bandpass using Remez algorithm (uses
            :func:`obspy.signal.filter.remezFIR`).

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> tr = st[0]
        >>> tr.filter("highpass", freq=1.0)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            tr = st[0]
            tr.filter("highpass", freq=1.0)
            tr.plot()
        """
        type = type.lower()
        # retrieve function call from entry points
        func = _get_function_from_entry_point('filter', type)
        # filtering
        # the options dictionary is passed as kwargs to the function that is
        # mapped according to the filter_functions dictionary
        self.data = func(self.data, df=self.stats.sampling_rate, **options)
        return self

    @_add_processing_info
    def trigger(self, type, **options):
        """
        Run a triggering algorithm on the data of the current trace.

        :param type: String that specifies which trigger is applied (e.g.
            ``'recstalta'``). See the `Supported Trigger`_ section below for
            further details.
        :param options: Necessary keyword arguments for the respective trigger
            that will be passed on.
            (e.g. ``sta=3``, ``lta=10``)
            Arguments ``sta`` and ``lta`` (seconds) will be mapped to ``nsta``
            and ``nlta`` (samples) by multiplying with sampling rate of trace.
            (e.g. ``sta=3``, ``lta=10`` would call the trigger with 3 and 10
            seconds average, respectively)

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.

        .. rubric:: _`Supported Trigger`

        ``'classicstalta'``
            Computes the classic STA/LTA characteristic function (uses
            :func:`obspy.signal.trigger.classic_STALTA`).

        ``'recstalta'``
            Recursive STA/LTA
            (uses :func:`obspy.signal.trigger.recursive_STALTA`).

        ``'recstaltapy'``
            Recursive STA/LTA written in Python (uses
            :func:`obspy.signal.trigger.recursive_STALTA_py`).

        ``'delayedstalta'``
            Delayed STA/LTA.
            (uses :func:`obspy.signal.trigger.delayed_STALTA`).

        ``'carlstatrig'``
            Computes the carl_STA_trig characteristic function (uses
            :func:`obspy.signal.trigger.carl_STA_trig`).

        ``'zdetect'``
            Z-detector (uses :func:`obspy.signal.trigger.z_detect`).

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> tr = st[0]
        >>> tr.filter("highpass", freq=1.0)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.plot()  # doctest: +SKIP
        >>> tr.trigger("recstalta", sta=1, lta=4)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            tr = st[0]
            tr.filter("highpass", freq=1.0)
            tr.plot()
            tr.trigger('recstalta', sta=1, lta=4)
            tr.plot()
        """
        type = type.lower()
        # retrieve function call from entry points
        func = _get_function_from_entry_point('trigger', type)
        # convert the two arguments sta and lta to nsta and nlta as used by
        # actual triggering routines (needs conversion to int, as samples are
        # used in length of trigger averages)...
        spr = self.stats.sampling_rate
        for key in ['sta', 'lta']:
            if key in options:
                options['n%s' % (key)] = int(options.pop(key) * spr)
        # triggering
        # the options dictionary is passed as kwargs to the function that is
        # mapped according to the trigger_functions dictionary
        self.data = func(self.data, **options)
        return self

    @skip_if_no_data
    @_add_processing_info
    def resample(self, sampling_rate, window='hanning', no_filter=True,
                 strict_length=False):
        """
        Resample trace data using Fourier method. Spectra are linearly
        interpolated if required.

        :type sampling_rate: float
        :param sampling_rate: The sampling rate of the resampled signal.
        :type window: array_like, callable, str, float, or tuple, optional
        :param window: Specifies the window applied to the signal in the
            Fourier domain. Defaults to ``'hanning'`` window. See
            :func:`scipy.signal.resample` for details.
        :type no_filter: bool, optional
        :param no_filter: Deactivates automatic filtering if set to ``True``.
            Defaults to ``True``.
        :type strict_length: bool, optional
        :param strict_length: Leave traces unchanged for which end time of
            trace would change. Defaults to ``False``.

        .. note::

            The :class:`~Trace` object has three different methods to change
            the sampling rate of its data: :meth:`~.resample`,
            :meth:`~.decimate`, and :meth:`~.interpolate`

            Make sure to choose the most appropriate one for the problem at
            hand.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.

        Uses :func:`scipy.signal.resample`. Because a Fourier method is used,
        the signal is assumed to be periodic.

        .. rubric:: Example

        >>> tr = Trace(data=np.array([0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1]))
        >>> len(tr)
        8
        >>> tr.stats.sampling_rate
        1.0
        >>> tr.resample(4.0)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> len(tr)
        32
        >>> tr.stats.sampling_rate
        4.0
        >>> tr.data  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        array([ 0.5       ,  0.40432914,  0.3232233 ,  0.26903012,  0.25 ...
        """
        from scipy.signal import get_window
        from scipy.fftpack import rfft, irfft
        factor = self.stats.sampling_rate / float(sampling_rate)
        # check if end time changes and this is not explicitly allowed
        if strict_length:
            if len(self.data) % factor != 0.0:
                msg = "End time of trace would change and strict_length=True."
                raise ValueError(msg)
        # do automatic lowpass filtering
        if not no_filter:
            # be sure filter still behaves good
            if factor > 16:
                msg = "Automatic filter design is unstable for resampling " + \
                      "factors (current sampling rate/new sampling rate) " + \
                      "above 16. Manual resampling is necessary."
                raise ArithmeticError(msg)
            freq = self.stats.sampling_rate * 0.5 / float(factor)
            self.filter('lowpass_cheby_2', freq=freq, maxorder=12)

        orig_dtype = self.data.dtype
        new_dtype = np.float32 if orig_dtype.itemsize == 4 else np.float64

        # resample in the frequency domain
        X = rfft(np.require(self.data, dtype=new_dtype))
        X = np.insert(X, 1, 0)
        if self.stats.npts % 2 == 0:
            X = np.append(X, [0])
        Xr = X[::2]
        Xi = X[1::2]

        if window is not None:
            if callable(window):
                W = window(np.fft.fftfreq(self.stats.npts))
            elif isinstance(window, np.ndarray):
                if window.shape != (self.stats.npts,):
                    msg = "Window has the wrong shape. Window length must " + \
                          "equal the number of points."
                    raise ValueError(msg)
                W = window
            else:
                W = np.fft.ifftshift(get_window(native_str(window),
                                                self.stats.npts))
            Xr *= W[:self.stats.npts//2+1]
            Xi *= W[:self.stats.npts//2+1]

        # interpolate
        num = int(self.stats.npts / factor)
        df = 1.0 / (self.stats.npts * self.stats.delta)
        dF = 1.0 / num * sampling_rate
        f = df * np.arange(0, self.stats.npts // 2 + 1, dtype=np.int32)
        nF = num // 2 + 1
        F = dF * np.arange(0, nF, dtype=np.int32)
        Y = np.zeros((2*nF))
        Y[::2] = np.interp(F, f, Xr)
        Y[1::2] = np.interp(F, f, Xi)

        Y = np.delete(Y, 1)
        if num % 2 == 0:
            Y = np.delete(Y, -1)
        self.data = irfft(Y) * (float(num) / float(self.stats.npts))
        self.data = np.require(self.data, dtype=orig_dtype)
        self.stats.sampling_rate = sampling_rate

        return self

    @_add_processing_info
    def decimate(self, factor, no_filter=False, strict_length=False):
        """
        Downsample trace data by an integer factor.

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
        Only every ``decimation_factor``-th sample remains in the trace, all
        other samples are thrown away. Prior to decimation a lowpass filter is
        applied to ensure no aliasing artifacts are introduced. The automatic
        filtering can be deactivated with ``no_filter=True``.

        If the length of the data array modulo ``decimation_factor`` is not
        zero then the end time of the trace is changing on sub-sample scale. To
        abort downsampling in case of changing end times set
        ``strict_length=True``.

        .. note::

            The :class:`~Trace` object has three different methods to change
            the sampling rate of its data: :meth:`~.resample`,
            :meth:`~.decimate`, and :meth:`~.interpolate`

            Make sure to choose the most appropriate one for the problem at
            hand.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.

        .. rubric:: Example

        For the example we switch off the automatic pre-filtering so that
        the effect of the downsampling routine becomes clearer:

        >>> tr = Trace(data=np.arange(10))
        >>> tr.stats.sampling_rate
        1.0
        >>> tr.data
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> tr.decimate(4, strict_length=False,
        ...    no_filter=True)  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.stats.sampling_rate
        0.25
        >>> tr.data
        array([0, 4, 8])
        """
        # check if end time changes and this is not explicitly allowed
        if strict_length and len(self.data) % factor:
            msg = "End time of trace would change and strict_length=True."
            raise ValueError(msg)

        # do automatic lowpass filtering
        if not no_filter:
            # be sure filter still behaves good
            if factor > 16:
                msg = "Automatic filter design is unstable for decimation " + \
                      "factors above 16. Manual decimation is necessary."
                raise ArithmeticError(msg)
            freq = self.stats.sampling_rate * 0.5 / float(factor)
            self.filter('lowpass_cheby_2', freq=freq, maxorder=12)

        # actual downsampling, as long as sampling_rate is a float we would not
        # need to convert to float, but let's do it as a safety measure
        from obspy.signal.filter import integer_decimation
        self.data = integer_decimation(self.data, factor)
        self.stats.sampling_rate = self.stats.sampling_rate / float(factor)
        return self

    def max(self):
        """
        Returns the value of the absolute maximum amplitude in the trace.

        :return: Value of absolute maximum of ``trace.data``.

        .. rubric:: Example

        >>> tr = Trace(data=np.array([0, -3, 9, 6, 4]))
        >>> tr.max()
        9
        >>> tr = Trace(data=np.array([0, -3, -9, 6, 4]))
        >>> tr.max()
        -9
        >>> tr = Trace(data=np.array([0.3, -3.5, 9.0, 6.4, 4.3]))
        >>> tr.max()
        9.0
        """
        value = self.data.max()
        _min = self.data.min()
        if abs(_min) > abs(value):
            value = _min
        return value

    def std(self):
        """
        Method to get the standard deviation of amplitudes in the trace.

        :return: Standard deviation of ``trace.data``.

        Standard deviation is calculated by NumPy method
        :meth:`~numpy.ndarray.std` on ``trace.data``.

        .. rubric:: Example

        >>> tr = Trace(data=np.array([0, -3, 9, 6, 4]))
        >>> tr.std()
        4.2614551505325036
        >>> tr = Trace(data=np.array([0.3, -3.5, 9.0, 6.4, 4.3]))
        >>> tr.std()
        4.4348618918744247
        """
        return self.data.std()

    @deprecated_keywords({'type': 'method'})
    @skip_if_no_data
    @_add_processing_info
    def differentiate(self, method='gradient', **options):
        """
        Differentiate the trace with respect to time.

        :type method: str, optional
        :param method: Method to use for differentiation. Defaults to
            ``'gradient'``. See the `Supported Methods`_ section below for
            further details.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.

        .. rubric:: _`Supported Methods`

        ``'gradient'``
            The gradient is computed using central differences in the interior
            and first differences at the boundaries. The returned gradient
            hence has the same shape as the input array. (uses
            :func:`numpy.gradient`)
        """
        method = method.lower()
        # retrieve function call from entry points
        func = _get_function_from_entry_point('differentiate', method)
        # differentiate
        self.data = func(self.data, self.stats.delta, **options)
        return self

    @deprecated_keywords({'type': 'method'})
    @skip_if_no_data
    @_add_processing_info
    def integrate(self, method="cumtrapz", **options):
        """
        Integrate the trace with respect to time.

        .. rubric:: _`Supported Methods`

        ``'cumtrapz'``
            First order integration of data using the trapezoidal rule. Uses
            :func:`obspy.signal.differentiate_and_integrate.integrate_cumtrapz`

        ``'spline'``
            Integrates by generating an interpolating spline and integrating
            that. Uses
            :func:`obspy.signal.differentiate_and_integrate.integrate_spline`

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.
        """
        method = method.lower()
        # retrieve function call from entry points
        func = _get_function_from_entry_point('integrate', method)

        self.data = func(data=self.data, dx=self.stats.delta, **options)
        return self

    @skip_if_no_data
    @raise_if_masked
    @_add_processing_info
    def detrend(self, type='simple', **options):
        """
        Remove a linear trend from the trace.

        :type type: str, optional
        :param type: Method to use for detrending. Defaults to ``'simple'``.
            See the `Supported Methods`_ section below for further details.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.

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
        type = type.lower()
        # retrieve function call from entry points
        func = _get_function_from_entry_point('detrend', type)
        # handle function specific settings
        if func.__module__.startswith('scipy'):
            # SciPy need to set the type keyword
            if type == 'demean':
                type = 'constant'
            options['type'] = type
        # detrending
        self.data = func(self.data, **options)
        return self

    @skip_if_no_data
    @_add_processing_info
    def taper(self, max_percentage, type='hann', max_length=None,
              side='both', **kwargs):
        """
        Taper the trace.

        Optional (and sometimes necessary) options to the tapering function can
        be provided as kwargs. See respective function definitions in
        `Supported Methods`_ section below.

        :type type: str
        :param type: Type of taper to use for detrending. Defaults to
            ``'cosine'``.  See the `Supported Methods`_ section below for
            further details.
        :type max_percentage: None, float
        :param max_percentage: Decimal percentage of taper at one end (ranging
            from 0. to 0.5).
        :type max_length: None, float
        :param max_length: Length of taper at one end in seconds.
        :type side: str
        :param side: Specify if both sides should be tapered (default, "both")
            or if only the left half ("left") or right half ("right") should be
            tapered.

        .. note::

            To get the same results as the default taper in SAC, use
            `max_percentage=0.05` and leave `type` as `hann`.

        .. note::

            If both `max_percentage` and `max_length` are set to a float, the
            shorter tape length is used. If both `max_percentage` and
            `max_length` are set to `None`, the whole trace will be tapered.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.

        .. rubric:: _`Supported Methods`

        ``'cosine'``
            Cosine taper, for additional options like taper percentage see:
            :func:`obspy.signal.invsim.cosine_taper`.
        ``'barthann'``
            Modified Bartlett-Hann window. (uses:
            :func:`scipy.signal.barthann`)
        ``'bartlett'``
            Bartlett window. (uses: :func:`scipy.signal.bartlett`)
        ``'blackman'``
            Blackman window. (uses: :func:`scipy.signal.blackman`)
        ``'blackmanharris'``
            Minimum 4-term Blackman-Harris window. (uses:
            :func:`scipy.signal.blackmanharris`)
        ``'bohman'``
            Bohman window. (uses: :func:`scipy.signal.bohman`)
        ``'boxcar'``
            Boxcar window. (uses: :func:`scipy.signal.boxcar`)
        ``'chebwin'``
            Dolph-Chebyshev window. (uses: :func:`scipy.signal.chebwin`)
        ``'flattop'``
            Flat top window. (uses: :func:`scipy.signal.flattop`)
        ``'gaussian'``
            Gaussian window with standard-deviation std. (uses:
            :func:`scipy.signal.gaussian`)
        ``'general_gaussian'``
            Generalized Gaussian window. (uses:
            :func:`scipy.signal.general_gaussian`)
        ``'hamming'``
            Hamming window. (uses: :func:`scipy.signal.hamming`)
        ``'hann'``
            Hann window. (uses: :func:`scipy.signal.hann`)
        ``'kaiser'``
            Kaiser window with shape parameter beta. (uses:
            :func:`scipy.signal.kaiser`)
        ``'nuttall'``
            Minimum 4-term Blackman-Harris window according to Nuttall.
            (uses: :func:`scipy.signal.nuttall`)
        ``'parzen'``
            Parzen window. (uses: :func:`scipy.signal.parzen`)
        ``'slepian'``
            Slepian window. (uses: :func:`scipy.signal.slepian`)
        ``'triang'``
            Triangular window. (uses: :func:`scipy.signal.triang`)
        """
        type = type.lower()
        side = side.lower()
        side_valid = ['both', 'left', 'right']
        npts = self.stats.npts
        if side not in side_valid:
            raise ValueError("'side' has to be one of: %s" % side_valid)
        # retrieve function call from entry points
        func = _get_function_from_entry_point('taper', type)
        # store all constraints for maximum taper length
        max_half_lenghts = []
        if max_percentage is not None:
            max_half_lenghts.append(int(max_percentage * npts))
        if max_length is not None:
            max_half_lenghts.append(int(max_length * self.stats.sampling_rate))
        if np.all([2 * mhl > npts for mhl in max_half_lenghts]):
            msg = "The requested taper is longer than the trace. " \
                  "The taper will be shortened to trace length."
            warnings.warn(msg)
        # add full trace length to constraints
        max_half_lenghts.append(int(npts / 2))
        # select shortest acceptable window half-length
        wlen = min(max_half_lenghts)
        # obspy.signal.cosine_taper has a default value for taper percentage,
        # we need to override is as we control percentage completely via npts
        # of taper function and insert ones in the middle afterwards
        if type == "cosine":
            kwargs['p'] = 1.0
        # tapering. tapering functions are expected to accept the number of
        # samples as first argument and return an array of values between 0 and
        # 1 with the same length as the data
        if 2 * wlen == npts:
            taper_sides = func(2 * wlen, **kwargs)
        else:
            taper_sides = func(2 * wlen + 1, **kwargs)
        if side == 'left':
            taper = np.hstack((taper_sides[:wlen], np.ones(npts - wlen)))
        elif side == 'right':
            taper = np.hstack((np.ones(npts - wlen),
                               taper_sides[len(taper_sides) - wlen:]))
        else:
            taper = np.hstack((taper_sides[:wlen], np.ones(npts - 2 * wlen),
                               taper_sides[len(taper_sides) - wlen:]))
        self.data = self.data * taper
        return self

    @_add_processing_info
    def normalize(self, norm=None):
        """
        Normalize the trace to its absolute maximum.

        :type norm: ``None`` or float
        :param norm: If not ``None``, trace is normalized by dividing by
            specified value ``norm`` instead of dividing by its absolute
            maximum. If a negative value is specified then its absolute value
            is used.

        If ``trace.data.dtype`` was integer it is changing to float.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.

        .. rubric:: Example

        >>> tr = Trace(data=np.array([0, -3, 9, 6]))
        >>> tr.normalize()  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.data
        array([ 0.        , -0.33333333,  1.        ,  0.66666667])
        >>> print(tr.stats.processing[0])  # doctest: +ELLIPSIS
        ObsPy ...: normalize(norm=None)
        >>> tr = Trace(data=np.array([0.3, -3.5, -9.2, 6.4]))
        >>> tr.normalize()  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.data
        array([ 0.0326087 , -0.38043478, -1.        ,  0.69565217])
        >>> print(tr.stats.processing[0])  # doctest: +ELLIPSIS
        ObsPy ...: normalize(norm=None)
        """
        # normalize, use norm-kwarg otherwise normalize to 1
        if norm:
            norm = norm
            if norm < 0:
                msg = "Normalizing with negative values is forbidden. " + \
                      "Using absolute value."
                warnings.warn(msg)
        else:
            norm = self.max()

        self.data = self.data.astype(np.float64)
        self.data /= abs(norm)

        return self

    def copy(self):
        """
        Returns a deepcopy of the trace.

        :return: Copy of trace.

        This actually copies all data in the trace and does not only provide
        another pointer to the same data. At any processing step if the
        original data has to be available afterwards, this is the method to
        use to make a copy of the trace.

        .. rubric:: Example

        Make a Trace and copy it:

        >>> tr = Trace(data=np.random.rand(10))
        >>> tr2 = tr.copy()

        The two objects are not the same:

        >>> tr2 is tr
        False

        But they have equal data (before applying further processing):

        >>> tr2 == tr
        True

        The following example shows how to make an alias but not copy the
        data. Any changes on ``tr3`` would also change the contents of ``tr``.

        >>> tr3 = tr
        >>> tr3 is tr
        True
        >>> tr3 == tr
        True
        """
        return deepcopy(self)

    def _addProcessingInfo(self, info):
        """
        Add the given informational string to the `processing` field in the
        trace's :class:`~obspy.core.trace.Stats` object.
        """
        proc = self.stats.setdefault('processing', [])
        proc.append(info)

    @_add_processing_info
    def split(self):
        """
        Split Trace object containing gaps using a NumPy masked array into
        several traces.

        :rtype: :class:`~obspy.core.stream.Stream`
        :returns: Stream containing all split traces. A gapless trace will
            still be returned as Stream with only one entry.
        """
        from obspy import Stream
        if not isinstance(self.data, np.ma.masked_array):
            # no gaps
            return Stream([self])
        slices = flat_not_masked_contiguous(self.data)
        trace_list = []
        for slice in slices:
            if slice.step:
                raise NotImplementedError("step not supported")
            stats = self.stats.copy()
            tr = Trace(header=stats)
            tr.stats.starttime += (stats.delta * slice.start)
            # return the underlying data not the masked array
            tr.data = self.data.data[slice.start:slice.stop]
            trace_list.append(tr)
        return Stream(trace_list)

    @skip_if_no_data
    @raise_if_masked
    @_add_processing_info
    def interpolate(self, sampling_rate, method="weighted_average_slopes",
                    starttime=None, npts=None, time_shift=0.0,
                    *args, **kwargs):
        """
        Interpolate the data using various interpolation techniques.

        Be careful when downsampling data and make sure to apply an appropriate
        anti-aliasing lowpass filter before interpolating in case it's
        necessary.

        .. note::

            The :class:`~Trace` object has three different methods to change
            the sampling rate of its data: :meth:`~.resample`,
            :meth:`~.decimate`, and :meth:`~.interpolate`.

            Make sure to choose the most appropriate one for the problem at
            hand.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data will no longer be accessible afterwards. To keep your
            original data, use :meth:`~.copy` to create a copy of your Trace
            object.

        .. rubric:: _`Interpolation Methods:`

        The chosen method is crucial and we will elaborate a bit about the
        choices here:

        * ``"lanczos"``: This offers the highest quality interpolation and
          should be chosen whenever possible. It is only due to legacy
          reasons that this is not the default method. The one downside it
          has is that it can be fairly expensive. See the
          :func:`~obspy.signal.interpolation.lanczos_interpolation` function
          for more details.
        * ``"weighted_average_slopes"``: This is the interpolation method used
          by SAC. Refer to
          :func:`~obspy.signal.interpolation.weighted_average_slopes` for
          more details.
        * ``"slinear"``, ``"quadratic"`` and ``"cubic"``: spline interpolation
          of first, second or third order.
        * ``"linear"``: Linear interpolation.
        * ``"nearest"``: Nearest neighbour interpolation.
        * ``"zero"``: Last encountered value interpolation.

        .. rubric:: _`Parameters:`

        :param sampling_rate: The new sampling rate in ``Hz``.
        :param method: The kind of interpolation to perform as a string. One of
            ``"linear"``, ``"nearest"``, ``"zero"``, ``"slinear"``,
            ``"quadratic"``, ``"cubic"``, ``"lanczos"``, or
            ``"weighted_average_slopes"``. Alternatively an integer
            specifying the order of the spline interpolator to use also works.
            Defaults to ``"weighted_average_slopes"``.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime` or int
        :param starttime: The start time (or timestamp) for the new
            interpolated stream. Will be set to current start time of the
            trace if not given.
        :type npts: int
        :param npts: The new number of samples. Will be set to the best
            fitting  number to retain the current end time of the trace if
            not given.
        :type time_shift: float
        :param time_shift: Interpolation can also shift the data with
            subsample accuracy. The time shift is always given in seconds. A
            positive shift means the data is shifted towards the future,
            e.g. a positive time delta. Please note that a time shift in
            the Fourier domain is always more accurate than this. When using
            Lanczos interpolation with large values of ``a`` and away from the
            boundaries this is nonetheless pretty good.

        .. rubric:: _`New in version 0.11:`

        * New parameter ``time_shift``.
        * New interpolation method ``lanczos``.


        .. rubric:: _`Usage Examples`

        >>> from obspy import read
        >>> tr = read()[0]
        >>> print(tr)  # doctest: +ELLIPSIS
        BW.RJOB..EHZ | 2009-08-24T00:20:03... - ... | 100.0 Hz, 3000 samples
        >>> tr.interpolate(sampling_rate=111.1)  # doctest: +ELLIPSIS
        <obspy.core.trace.Trace object at 0x...>
        >>> print(tr)  # doctest: +ELLIPSIS
        BW.RJOB..EHZ | 2009-08-24T00:20:03... - ... | 111.1 Hz, 3332 samples

        Setting ``starttime`` and/or ``npts`` will interpolate to sampling
        points with the given start time and/or number of samples.
        Extrapolation will not be performed.

        >>> tr = read()[0]
        >>> print(tr)  # doctest: +ELLIPSIS
        BW.RJOB..EHZ | 2009-08-24T00:20:03... - ... | 100.0 Hz, 3000 samples
        >>> tr.interpolate(sampling_rate=111.1,
        ...                starttime=tr.stats.starttime + 10) \
        # doctest:  +ELLIPSIS
        <obspy.core.trace.Trace object at 0x...>
        >>> print(tr)  # doctest: +ELLIPSIS
        BW.RJOB..EHZ | 2009-08-24T00:20:13... - ... | 111.1 Hz, 2221 samples
        """
        try:
            method = method.lower()
        except:
            pass

        dt = float(sampling_rate)
        if dt <= 0.0:
            raise ValueError("The time step must be positive.")
        dt = 1.0 / sampling_rate

        # We just shift the old start time. The interpolation will take care
        # of the rest.
        if time_shift:
            self.stats.starttime += time_shift

        try:
            if isinstance(method, int) or \
                    method in ["linear", "nearest", "zero", "slinear",
                               "quadratic", "cubic"]:
                func = _get_function_from_entry_point('interpolate',
                                                      'interpolate_1d')
            else:
                func = _get_function_from_entry_point('interpolate', method)
            old_start = self.stats.starttime.timestamp
            old_dt = self.stats.delta

            if starttime is not None:
                try:
                    starttime = starttime.timestamp
                except AttributeError:
                    pass
            else:
                starttime = self.stats.starttime.timestamp
            endtime = self.stats.endtime.timestamp
            if npts is None:
                npts = int(math.floor((endtime - starttime) / dt)) + 1

            self.data = np.atleast_1d(func(
                np.require(self.data, dtype=np.float64), old_start, old_dt,
                starttime, dt, npts, type=method, *args, **kwargs))
            self.stats.starttime = UTCDateTime(starttime)
            self.stats.delta = dt
        except:
            # Revert the start time change if something went wrong.
            if time_shift:
                self.stats.starttime -= time_shift
            # re-raise last exception.
            raise

        return self

    def times(self):
        """
        For convenient plotting compute a NumPy array of seconds since
        starttime corresponding to the samples in Trace.

        :rtype: :class:`~numpy.ndarray` or :class:`~numpy.ma.MaskedArray`
        :returns: An array of time samples in an :class:`~numpy.ndarray` if
            the trace doesn't have any gaps or a :class:`~numpy.ma.MaskedArray`
            otherwise.
        """
        timeArray = np.arange(self.stats.npts)
        timeArray = timeArray / self.stats.sampling_rate
        # Check if the data is a ma.maskedarray
        if isinstance(self.data, np.ma.masked_array):
            timeArray = np.ma.array(timeArray, mask=self.data.mask)
        return timeArray

    def attach_response(self, inventories):
        """
        Search for and attach channel response to the trace as
        :class:`Trace`.stats.response. Raises an exception if no matching
        response can be found.
        To subsequently deconvolve the instrument response use
        :meth:`Trace.remove_response`.

        >>> from obspy import read, read_inventory
        >>> st = read()
        >>> tr = st[0]
        >>> inv = read_inventory("/path/to/BW_RJOB.xml")
        >>> tr.attach_response(inv)
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

        :type inventories: :class:`~obspy.core.inventory.inventory.Inventory`
            or :class:`~obspy.core.inventory.network.Network` or a list
            containing objects of these types or a string with a filename of
            a StationXML file.
        :param inventories: Station metadata to use in search for response for
            each trace in the stream.
        """
        from obspy.core.inventory import Inventory, Network, read_inventory
        if isinstance(inventories, Inventory) or \
           isinstance(inventories, Network):
            inventories = [inventories]
        elif isinstance(inventories, (str, native_str)):
            inventories = [read_inventory(inventories)]
        responses = []
        for inv in inventories:
            try:
                responses.append(inv.get_response(self.id,
                                                  self.stats.starttime))
            except:
                pass
        if len(responses) > 1:
            msg = "Found more than one matching response. Attaching first."
            warnings.warn(msg)
        elif len(responses) < 1:
            msg = "No matching response information found."
            raise Exception(msg)
        self.stats.response = responses[0]

    @_add_processing_info
    def remove_response(self, output="VEL", water_level=60, pre_filt=None,
                        zero_mean=True, taper=True, taper_fraction=0.05,
                        plot=False, **kwargs):
        """
        Deconvolve instrument response.

        Uses the :class:`obspy.core.inventory.response.Response` object
        attached as :class:`Trace`.stats.response to deconvolve the
        instrument response from the trace's time series data. Raises an
        exception if the response is not present. Use e.g.
        :meth:`Trace.attach_response` to attach response to trace providing
        :class:`obspy.core.inventory.inventory.Inventory` data.

        Note that there are two ways to prevent overamplification
        while convolving the inverted instrument spectrum: One possibility is
        to specify a water level which represents a clipping of the inverse
        spectrum and limits amplification to a certain maximum cut-off value
        (`water_level` in dB). The other possibility is to taper the waveform
        data in the frequency domain prior to multiplying with the inverse
        spectrum, i.e. perform a pre-filtering in the frequency domain
        (specifying the four corner frequencies of the frequency taper as a
        tuple in `pre_filt`).

        .. note::

            Any additional kwargs will be passed on to
            :meth:`obspy.core.inventory.response.Response.get_evalresp_response`,
            see documentation of that method for further customization (e.g.
            start/stop stage).

        .. note::

            Using :meth:`~Trace.remove_response` is equivalent to using
            :meth:`~Trace.simulate` with the identical response provided as
            a (dataless) SEED or RESP file and when using the same
            `water_level` and `pre_filt` (and options `sacsim=True` and
            `pitsasim=False` which influence very minor details in detrending
            and tapering).

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> tr = st[0].copy()
        >>> tr.plot()  # doctest: +SKIP
        >>> # Response object is already attached to example data:
        >>> print(tr.stats.response)  \
                # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Channel Response
            From M/S (Velocity in Meters Per Second) to COUNTS (Digital Counts)
            Overall Sensitivity: 2.5168e+09 defined at 0.020 Hz
            4 stages:
                Stage 1: PolesZerosResponseStage from M/S to V, gain: 1500
                Stage 2: CoefficientsTypeResponseStage from V to COUNTS, ...
                Stage 3: FIRResponseStage from COUNTS to COUNTS, gain: 1
                Stage 4: FIRResponseStage from COUNTS to COUNTS, gain: 1
        >>> tr.remove_response()  # doctest: +ELLIPSIS
        <...Trace object at 0x...>
        >>> tr.plot()  # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            tr = st[0]
            tr.remove_response()
            tr.plot()

        Using the `plot` option it is possible to visualize the individual
        steps during response removal in the frequency domain to check the
        chosen `pre_filt` and `water_level` options to stabilize the
        deconvolution of the inverted instrument response spectrum:

        >>> from obspy import read, read_inventory
        >>> st = read("/path/to/IU_ULN_00_LH1_2015-07-18T02.mseed")
        >>> tr = st[0]
        >>> inv = read_inventory("/path/to/IU_ULN_00_LH1.xml")
        >>> tr.attach_response(inv)
        >>> pre_filt = [0.001, 0.005, 45, 50]
        >>> tr.remove_response(pre_filt=pre_filt, output="DISP",
        ...                    water_level=60, plot=True)  # doctest: +SKIP
        <...Trace object at 0x...>

        .. plot::

            from obspy import read, read_inventory
            st = read("http://examples.obspy.org/IU_ULN_2015-07-18T02.mseed")
            tr = st[0]
            inv = read_inventory("http://examples.obspy.org/IU_ULN.xml")
            tr.attach_response(inv)
            pre_filt = [0.001, 0.005, 45, 50]
            output = "DISP"
            tr.remove_response(pre_filt=pre_filt, output=output,
                               water_level=60, plot=True)

        :type output: str
        :param output: Output units. One of:

            ``"DISP"``
                displacement, output unit is meters
            ``"VEL"``
                velocity, output unit is meters/second
            ``"ACC"``
                acceleration, output unit is meters/second**2

        :type water_level: float
        :param water_level: Water level for deconvolution.
        :type pre_filt: list or tuple of four float
        :param pre_filt: Apply a bandpass filter in frequency domain to the
            data before deconvolution. The list or tuple defines
            the four corner frequencies `(f1, f2, f3, f4)` of a cosine taper
            which is one between `f2` and `f3` and tapers to zero for
            `f1 < f < f2` and `f3 < f < f4`.
        :type zero_mean: bool
        :param zero_mean: If `True`, the mean of the waveform data is
            subtracted in time domain prior to deconvolution.
        :type taper: bool
        :param taper: If `True`, a cosine taper is applied to the waveform data
            in time domain prior to deconvolution.
        :type taper_fraction: float
        :param taper_fraction: Taper fraction of cosine taper to use.
        :type plot: bool or str
        :param plot: If `True`, brings up a plot that illustrates how the
            data are processed in the frequency domain in three steps. First by
            `pre_filt` frequency domain tapering, then by inverting the
            instrument response spectrum with or without `water_level` and
            finally showing data with inverted instrument response multiplied
            on it in frequency domain. It also shows the comparison of
            raw/corrected data in time domain. If a `str` is provided then the
            plot is saved to file (filename must have a valid image suffix
            recognizable by matplotlib e.g. '.png').
        """
        from obspy.core.inventory import Response, PolynomialResponseStage
        from obspy.signal.invsim import (cosine_taper, cosine_sac_taper,
                                         invert_spectrum)

        if "response" not in self.stats:
            msg = ("No response information attached to trace "
                   "(as Trace.stats.response).")
            raise KeyError(msg)
        if not isinstance(self.stats.response, Response):
            msg = ("Response must be of type "
                   "obspy.core.inventory.response.Response "
                   "(but is of type %s).") % type(self.stats.response)
            raise TypeError(msg)

        response = self.stats.response
        # polynomial response using blockette 62 stage 0
        if not response.response_stages and response.instrument_polynomial:
            coefficients = response.instrument_polynomial.coefficients
            self.data = np.poly1d(coefficients[::-1])(self.data)
            return self

        # polynomial response using blockette 62 stage 1 and no other stages
        if len(response.response_stages) == 1 and \
           isinstance(response.response_stages[0], PolynomialResponseStage):
            # check for gain
            if response.response_stages[0].stage_gain is None:
                msg = 'Stage gain not defined for %s - setting it to 1.0'
                warnings.warn(msg % self.id)
                gain = 1
            else:
                gain = response.response_stages[0].stage_gain
            coefficients = response.response_stages[0].coefficients[:]
            for i in range(len(coefficients)):
                coefficients[i] /= math.pow(gain, i)
            self.data = np.poly1d(coefficients[::-1])(self.data)
            return self

        # use evalresp
        data = self.data.astype(np.float64)
        npts = len(data)
        # time domain pre-processing
        if zero_mean:
            data -= data.mean()
        if taper:
            data *= cosine_taper(npts, taper_fraction,
                                 sactaper=True, halfcosine=False)

        if plot:
            color1 = "blue"
            color2 = "red"
            bbox = dict(boxstyle="round", fc="w", alpha=1, ec="w")
            bbox1 = dict(boxstyle="round", fc="blue", alpha=0.15)
            bbox2 = dict(boxstyle="round", fc="red", alpha=0.15)
            fig = plt.figure(figsize=(14, 10))
            fig.suptitle(str(self))
            ax1 = fig.add_subplot(321)
            ax1b = ax1.twinx()
            ax2 = fig.add_subplot(323, sharex=ax1)
            ax2b = ax2.twinx()
            ax3 = fig.add_subplot(325, sharex=ax1)
            ax3b = ax3.twinx()
            ax4 = fig.add_subplot(322)
            ax5 = fig.add_subplot(324, sharex=ax4)
            ax6 = fig.add_subplot(326, sharex=ax4)
            for ax_ in (ax1, ax2, ax3, ax4, ax5, ax6):
                ax_.grid(zorder=-10)
            ax1.text(0.05, 0.1, 'pre_filt: %s' % pre_filt,
                     ha="left", va="bottom", transform=ax1.transAxes,
                     fontsize="large", bbox=bbox, zorder=5)
            ax1.set_ylabel("Data spectrum, raw", bbox=bbox1)
            ax1b.set_ylabel("'pre_filt' taper fraction", bbox=bbox2)
            evalresp_info = "\n".join(
                ['output: %s' % output] +
                ['%s: %s' % (key, value) for key, value in kwargs.items()])
            ax2.text(0.05, 0.1, evalresp_info, ha="left",
                     va="bottom", transform=ax2.transAxes,
                     fontsize="large", zorder=5, bbox=bbox)
            ax2.set_ylabel("Data spectrum,\n"
                           "'pre_filt' applied", bbox=bbox1)
            ax2b.set_ylabel("Instrument response", bbox=bbox2)
            ax3.text(0.05, 0.1, 'water_level: %s' % water_level,
                     ha="left", va="bottom", transform=ax3.transAxes,
                     fontsize="large", zorder=5, bbox=bbox)
            ax3.set_ylabel("Data spectrum,\nmultiplied with inverted\n"
                           "instrument response", bbox=bbox1)
            ax3b.set_ylabel("Inverted instrument response,\n"
                            "water level applied", bbox=bbox2)
            ax3.set_xlabel("Frequency [Hz]")
            times = self.times()
            ax4.plot(times, self.data, color="k")
            ax4.set_ylabel("Raw")
            ax4.yaxis.set_ticks_position("right")
            ax4.yaxis.set_label_position("right")
            ax5.plot(times, data, color="k")
            ax5.set_ylabel("Raw, after time\ndomain pre-processing")
            ax5.yaxis.set_ticks_position("right")
            ax5.yaxis.set_label_position("right")
            ax6.set_ylabel("Response removed")
            ax6.set_xlabel("Time [s]")
            ax6.yaxis.set_ticks_position("right")
            ax6.yaxis.set_label_position("right")

        # smart calculation of nfft dodging large primes
        from obspy.signal.util import _npts2nfft
        nfft = _npts2nfft(npts)
        # Transform data to Frequency domain
        data = np.fft.rfft(data, n=nfft)
        # calculate and apply frequency response,
        # optionally prefilter in frequency domain and/or apply water level
        freq_response, freqs = \
            self.stats.response.get_evalresp_response(self.stats.delta, nfft,
                                                      output=output, **kwargs)

        if plot:
            ax1.loglog(freqs, np.abs(data), color=color1, zorder=9)

        # frequency domain pre-filtering of data spectrum
        # (apply cosine taper in frequency domain)
        if pre_filt:
            freq_domain_taper = cosine_sac_taper(freqs, flimit=pre_filt)
            data *= freq_domain_taper

        if plot:
            try:
                freq_domain_taper
            except NameError:
                freq_domain_taper = np.ones(len(freqs))
            ax1b.semilogx(freqs, freq_domain_taper, color=color2, zorder=10)
            ax1b.set_ylim(-0.05, 1.05)
            ax2.loglog(freqs, np.abs(data), color=color1, zorder=9)
            ax2b.loglog(freqs, np.abs(freq_response), color=color2, zorder=10)

        if water_level is None:
            # No water level used, so just directly invert the response.
            # First entry is at zero frequency and value is zero, too.
            # Just do not invert the first value (and set to 0 to make sure).
            freq_response[0] = 0.0
            freq_response[1:] = 1.0 / freq_response[1:]
        else:
            # Invert spectrum with specified water level.
            invert_spectrum(freq_response, water_level)

        data *= freq_response
        data[-1] = abs(data[-1]) + 0.0j

        if plot:
            ax3.loglog(freqs, np.abs(data), color=color1, zorder=9)
            ax3b.loglog(freqs, np.abs(freq_response), color=color2, zorder=10)

        # transform data back into the time domain
        data = np.fft.irfft(data)[0:npts]

        if plot:
            ax6.plot(times, data, color="k")
            plt.subplots_adjust(wspace=0.4)
            if plot is True:
                plt.show()
            else:
                plt.savefig(plot)
                plt.close(fig)

        # assign processed data and store processing information
        self.data = data
        info = ":".join(["remove_response"] +
                        [str(x) for x in (output, water_level, pre_filt,
                                          zero_mean, taper, taper_fraction)] +
                        ["%s=%s" % (k, v) for k, v in kwargs.items()])
        self._addProcessingInfo(info)
        return self


def _data_sanity_checks(value):
    """
    Check if a given input is suitable to be used for Trace.data. Raises the
    corresponding exception if it is not, otherwise silently passes.
    """
    if not isinstance(value, np.ndarray):
        msg = "Trace.data must be a NumPy array."
        raise ValueError(msg)
    if value.ndim != 1:
        msg = ("NumPy array for Trace.data has bad shape ('%s'). Only 1-d "
               "arrays are allowed for initialization.") % str(value.shape)
        raise ValueError(msg)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
