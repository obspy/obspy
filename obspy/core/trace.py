# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Trace objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from copy import deepcopy, copy
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import AttribDict, createEmptyDataChunk
from obspy.core.util.base import _getFunctionFromEntryPoint
from obspy.core.util.misc import flatnotmaskedContiguous
import math
import numpy as np
import warnings


class Stats(AttribDict):
    """
    A container for additional header information of a ObsPy Trace object.

    A ``Stats`` object may contain all header information (also known as meta
    data) of a :class:`~obspy.core.trace.Trace` object. Those headers may be
    accessed or modified either in the dictionary style or directly via a
    corresponding attribute. There are various default attributes which are
    required by every waveform import and export modules within ObsPy such as
    :mod:`obspy.mseed`.

    :type header: dict or :class:`~obspy.core.trace.Stats`, optional
    :param header: Dictionary containing meta information of a single
        :class:`~obspy.core.trace.Trace` object. Possible keywords are
        summarized in the following `Default Attributes`_ section.

    .. rubric:: Basic Usage

    >>> stats = Stats()
    >>> stats.network = 'BW'
    >>> stats['network']
    'BW'
    >>> stats['station'] = 'MANZ'
    >>> stats.station
    'MANZ'

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

        .. note::
            The attribute ``endtime`` is currently calculated as
            ``endtime = starttime + (npts-1) * delta``. This behaviour may
            change in the future to ``endtime = starttime + npts * delta``.

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
        # make sure Trace gets initialized with ndarray as self.data
        # otherwise we could end up with e.g. a list object in self.data
        if not isinstance(data, np.ndarray):
            msg = "Trace.data must be a NumPy array."
            raise ValueError(msg)
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

    def __str__(self, id_length=None):
        """
        Returns short summary string of the current trace.

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

    def __len__(self):
        """
        Returns number of data samples of the current trace.

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
            if not isinstance(value, np.ndarray):
                msg = "Trace.data must be a NumPy array."
                ValueError(msg)
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
        Creates a new Stream containing num copies of this trace.

        :rtype num: int
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
        Splits Trace into new Stream containing num Traces of the same size.

        :type num: int
        :param num: Number of traces in returned Stream. Last trace may contain
            lesser samples.
        :returns: New ObsPy Stream object.

        .. rubric:: Example

        >>> from obspy import read
        >>> tr = read()[0]
        >>> print tr  # doctest: +ELLIPSIS
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> st = tr / 7
        >>> print st  # doctest: +ELLIPSIS
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

    def __mod__(self, num):
        """
        Splits Trace into new Stream containing Traces with num samples.

        :type num: int
        :param num: Number of samples in each trace in returned Stream. Last
            trace may contain lesser samples.
        :returns: New ObsPy Stream object.

        .. rubric:: Example

        >>> from obspy import read
        >>> tr = read()[0]
        >>> print tr  # doctest: +ELLIPSIS
        BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z ... | 100.0 Hz, 3000 samples
        >>> st = tr % 800
        >>> print st  # doctest: +ELLIPSIS
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
        Adds another Trace object to current trace.

        :type method: ``0`` or ``1``, optional
        :param method: Method to handle overlaps of traces. Defaults to ``0``.
            See the `Handling Overlaps`_ section below for further details.
        :type fill_value: int or float, ``'latest'`` or ``'interpolate'``,
            optional
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
        :type sanity_checks: boolean, optional
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
        delta = int(round(delta)) - 1
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
                overlap = createEmptyDataChunk(delta, lt.data.dtype,
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
            if np.all(lt.data[t1:t2] == rt.data):
                # check if data are the same
                data = [lt.data]
            elif method == 0:
                gap = createEmptyDataChunk(lenrt, lt.data.dtype, fill_value)
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
            gap = createEmptyDataChunk(delta, lt.data.dtype, fill_value)
            data = [lt.data, gap, rt.data]
        # merge traces depending on numpy array type
        if True in [isinstance(_i, np.ma.masked_array) for _i in data]:
            data = np.ma.concatenate(data)
        else:
            data = np.concatenate(data)
            data = np.require(data, dtype=lt.data.dtype)
        out.data = data
        return out

    def getId(self):
        """
        Returns a SEED compatible identifier of the trace.

        :rtype: str
        :return: SEED identifier

        The SEED identifier contains the network, station, location and channel
        code for the current Trace object.

        .. rubric:: Example

        >>> meta = {'station': 'MANZ', 'network': 'BW', 'channel': 'EHZ'}
        >>> tr = Trace(header=meta)
        >>> tr.getId()
        'BW.MANZ..EHZ'
        >>> tr.id
        'BW.MANZ..EHZ'
        """
        out = "%(network)s.%(station)s.%(location)s.%(channel)s"
        return out % (self.stats)

    id = property(getId)

    def plot(self, **kwargs):
        """
        Creates a simple graph of the current trace.

        Various options are available to change the appearance of the waveform
        plot. Please see :meth:`~obspy.core.stream.Stream.plot` method for all
        possible options.

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> tr = st[0]
        >>> tr.plot() # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            tr = st[0]
            tr.plot()
        """
        from obspy.imaging.waveform import WaveformPlotting
        waveform = WaveformPlotting(stream=self, **kwargs)
        return waveform.plotWaveform()

    def spectrogram(self, **kwargs):
        """
        Creates a spectrogram plot of the trace.

        For details on kwargs that can be used to customize the spectrogram
        plot see :func:`~obspy.imaging.spectrogram.spectrogram`.

        .. rubric:: Example

        >>> from obspy import read
        >>> st = read()
        >>> tr = st[0]
        >>> tr.spectrogram() # doctest: +SKIP

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
        Saves current trace into a file.

        :type filename: string
        :param filename: The name of the file to write.
        :type format: string
        :param format: The format to write must be specified. One of
            ``"MSEED"``, ``"GSE2"``, ``"SAC"``, ``"SACXY"``, ``"Q"``,
            ``"SH_ASC"``, ``"SEGY"``, ``"SU"``, ``"WAV"``, ``"PICKLE"``. See
            :meth:`obspy.core.stream.Stream.write` method for all possible
            formats.
        :param kwargs: Additional keyword arguments passed to the underlying
            waveform writer method.

        .. rubric:: Example

        >>> tr = Trace()
        >>> tr.write("out.mseed", format="MSEED") # doctest: +SKIP
        """
        # we need to import here in order to prevent a circular import of
        # Stream and Trace classes
        from obspy import Stream
        Stream([self]).write(filename, format, **kwargs)

    def _ltrim(self, starttime, pad=False, nearest_sample=True,
               fill_value=None):
        """
        Cuts current trace to given start time. For more info see
        :meth:`~obspy.core.trace.Trace.trim`.

        .. rubric:: Example

        >>> tr = Trace(data=np.arange(0, 10))
        >>> tr.stats.delta = 1.0
        >>> tr._ltrim(tr.stats.starttime + 8)
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
            delta = round((starttime - self.stats.starttime) *
                          self.stats.sampling_rate)
            # due to rounding and npts starttime must always be right of
            # self.stats.starttime, rtrim relies on it
            if delta < 0 and pad:
                npts = abs(delta) + 10  # use this as a start
                newstarttime = self.stats.starttime - npts / \
                        float(self.stats.sampling_rate)
                newdelta = round((starttime - newstarttime) *
                                 self.stats.sampling_rate)
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
            return
        elif delta < 0 and pad:
            try:
                gap = createEmptyDataChunk(abs(delta), self.data.dtype,
                                           fill_value)
            except ValueError:
                # createEmptyDataChunk returns negative ValueError ?? for
                # too large number of points, e.g. 189336539799
                raise Exception("Time offset between starttime and "
                                "trace.starttime too large")
            self.data = np.ma.concatenate((gap, self.data))
            return
        elif starttime > self.stats.endtime:
            self.data = np.empty(0, dtype=org_dtype)
            return
        elif delta > 0:
            self.data = self.data[delta:]

    def _rtrim(self, endtime, pad=False, nearest_sample=True, fill_value=None):
        """
        Cuts current trace to given end time. For more info see
        :meth:`~obspy.core.trace.Trace.trim`.

        .. rubric:: Example

        >>> tr = Trace(data=np.arange(0, 10))
        >>> tr.stats.delta = 1.0
        >>> tr._rtrim(tr.stats.starttime + 2)
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
            delta = round((endtime - self.stats.starttime) *
                           self.stats.sampling_rate) - self.stats.npts + 1
            delta = int(delta)
        else:
            # solution for #127, however some tests need to be changed
            #delta = -1*int(math.floor(round((self.stats.endtime - endtime) * \
            #                       self.stats.sampling_rate, 7)))
            delta = int(math.floor(round((endtime - self.stats.endtime) *
                                   self.stats.sampling_rate, 7)))
        if delta == 0 or (delta > 0 and not pad):
            return
        if delta > 0 and pad:
            try:
                gap = createEmptyDataChunk(delta, self.data.dtype, fill_value)
            except ValueError:
                # createEmptyDataChunk returns negative ValueError ?? for
                # too large number of pointes, e.g. 189336539799
                raise Exception("Time offset between starttime and " +
                                "trace.starttime too large")
            self.data = np.ma.concatenate((self.data, gap))
            return
        elif endtime < self.stats.starttime:
            self.stats.starttime = self.stats.endtime + \
                                   delta * self.stats.delta
            self.data = np.empty(0, dtype=org_dtype)
            return
        # cut from right
        delta = abs(delta)
        total = len(self.data) - delta
        if endtime == self.stats.starttime:
            total = 1
        self.data = self.data[:total]

    def trim(self, starttime=None, endtime=None, pad=False,
             nearest_sample=True, fill_value=None):
        """
        Cuts current trace to given start and end time.

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
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.

        .. rubric:: Example

        >>> tr = Trace(data=np.arange(0, 10))
        >>> tr.stats.delta = 1.0
        >>> t = tr.stats.starttime
        >>> tr.trim(t + 2.000001, t + 7.999999)
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

    def slice(self, starttime=None, endtime=None):
        """
        Returns a new Trace object with data going from start to end time.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Specify the start time of slice.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Specify the end time of slice.
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
        tr.trim(starttime=starttime, endtime=endtime)
        return tr

    def verify(self):
        """
        Verifies current trace object against available meta data.

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
            if int(round(delta * sr)) + 1 != len(self.data):
                msg = "Sample rate(%f) * time delta(%.4lf) + 1 != data len(%d)"
                raise Exception(msg % (sr, delta, len(self.data)))
            # Check if the endtime fits the starttime, npts and sampling_rate.
            if self.stats.endtime != self.stats.starttime + \
                (self.stats.npts - 1) / float(self.stats.sampling_rate):
                msg = "Endtime is not the time of the last sample."
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
        :func:`~obspy.signal.invsim.seisSim`.

        `paz_remove` and `paz_simulate` are expected to be dictionaries
        containing information on poles, zeros and gain (and usually also
        sensitivity).

        If both `paz_remove` and `paz_simulate` are specified, both steps are
        performed in one go in the frequency domain, otherwise only the
        specified step is performed.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.

        .. rubric:: Example

        >>> from obspy import read
        >>> from obspy.signal import cornFreq2Paz
        >>> st = read()
        >>> tr = st[0]
        >>> tr.plot() # doctest: +SKIP
        >>> paz_sts2 = {'poles': [-0.037004+0.037016j, -0.037004-0.037016j,
        ...                       -251.33+0j,
        ...                       -131.04-467.29j, -131.04+467.29j],
        ...             'zeros': [0j, 0j],
        ...             'gain': 60077000.0,
        ...             'sensitivity': 2516778400.0}
        >>> paz_1hz = cornFreq2Paz(1.0, damp=0.707)
        >>> paz_1hz['sensitivity'] = 1.0
        >>> tr.simulate(paz_remove=paz_sts2, paz_simulate=paz_1hz)
        >>> tr.plot() # doctest: +SKIP

        .. plot::

            from obspy import read
            from obspy.signal import cornFreq2Paz
            st = read()
            tr = st[0]
            tr.plot()
            paz_sts2 = {'poles': [-0.037004+0.037016j, -0.037004-0.037016j,
                                  -251.33+0j,
                                  -131.04-467.29j, -131.04+467.29j],
                        'zeros': [0j, 0j],
                        'gain': 60077000.0,
                        'sensitivity': 2516778400.0}
            paz_1hz = cornFreq2Paz(1.0, damp=0.707)
            paz_1hz['sensitivity'] = 1.0
            tr.simulate(paz_remove=paz_sts2, paz_simulate=paz_1hz)
            tr.plot()
        """
        # XXX accepting string "self" and using attached PAZ then
        if paz_remove == 'self':
            paz_remove = self.stats.paz

        from obspy.signal import seisSim
        self.data = seisSim(self.data, self.stats.sampling_rate,
                paz_remove=paz_remove, paz_simulate=paz_simulate,
                remove_sensitivity=remove_sensitivity,
                simulate_sensitivity=simulate_sensitivity, **kwargs)

        # add processing information to the stats dictionary
        if paz_remove:
            proc_info = "simulate:inverse:%s:sensitivity=%s" % \
                    (paz_remove, remove_sensitivity)
            self._addProcessingInfo(proc_info)
        if paz_simulate:
            proc_info = "simulate:forward:%s:sensitivity=%s" % \
                    (paz_simulate, simulate_sensitivity)
            self._addProcessingInfo(proc_info)

    def filter(self, type, **options):
        """
        Filters the data of the current trace.

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
        >>> tr = st[0]
        >>> tr.filter("highpass", freq=1.0)
        >>> tr.plot() # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            tr = st[0]
            tr.filter("highpass", freq=1.0)
            tr.plot()
        """
        type = type.lower()
        # retrieve function call from entry points
        func = _getFunctionFromEntryPoint('filter', type)
        # filtering
        # the options dictionary is passed as kwargs to the function that is
        # mapped according to the filter_functions dictionary
        self.data = func(self.data, df=self.stats.sampling_rate, **options)
        # add processing information to the stats dictionary
        proc_info = "filter:%s:%s" % (type, options)
        self._addProcessingInfo(proc_info)

    def trigger(self, type, **options):
        """
        Runs a triggering algorithm on the data of the current trace.

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
        >>> tr = st[0]
        >>> tr.filter("highpass", freq=1.0)
        >>> tr.plot() # doctest: +SKIP
        >>> tr.trigger("recstalta", sta=3, lta=10)
        >>> tr.plot() # doctest: +SKIP

        .. plot::

            from obspy import read
            st = read()
            tr = st[0]
            tr.filter("highpass", freq=1.0)
            tr.plot()
            tr.trigger('recstalta', sta=3, lta=10)
            tr.plot()
        """
        type = type.lower()
        # retrieve function call from entry points
        func = _getFunctionFromEntryPoint('trigger', type)
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
        # add processing information to the stats dictionary
        proc_info = "trigger:%s:%s" % (type, options)
        self._addProcessingInfo(proc_info)

    def resample(self, sampling_rate, window='hanning', no_filter=True,
                 strict_length=False):
        """
        Resample trace data using Fourier method.

        :type sampling_rate: float
        :param sampling_rate: The sampling rate of the resampled signal.
        :type window: array_like, callable, string, float, or tuple, optional
        :param window: Specifies the window applied to the signal in the
            Fourier domain. Defaults to ``'hanning'`` window. See
            :func:`scipy.signal.resample` for details.
        :type strict_length: bool, optional
        :param strict_length: Leave traces unchanged for which endtime of trace
            would change. Defaults to ``False``.

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
        >>> tr.resample(4.0)
        >>> len(tr)
        32
        >>> tr.stats.sampling_rate
        4.0
        >>> tr.data  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        array([ 0.5       ,  0.40432914,  0.3232233 ,  0.26903012,  0.25 ...
        """
        from scipy.signal import resample
        factor = self.stats.sampling_rate / float(sampling_rate)
        # check if endtime changes and this is not explicitly allowed
        if strict_length and len(self.data) % factor != 0.0:
            msg = "Endtime of trace would change and strict_length=True."
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
            self.filter('lowpassCheby2', freq=freq, maxorder=12)
        # resample
        num = int(self.stats.npts / factor)
        self.data = resample(self.data, num, window=window)
        self.stats.sampling_rate = sampling_rate
        # add processing information to the stats dictionary
        proc_info = "resample:%d:%s" % (sampling_rate, window)
        self._addProcessingInfo(proc_info)

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
        :param strict_length: Leave traces unchanged for which endtime of trace
            would change. Defaults to ``False``.

        Currently a simple integer decimation is implemented.
        Only every ``decimation_factor``-th sample remains in the trace, all
        other samples are thrown away. Prior to decimation a lowpass filter is
        applied to ensure no aliasing artifacts are introduced. The automatic
        filtering can be deactivated with ``no_filter=True``.

        If the length of the data array modulo ``decimation_factor`` is not
        zero then the endtime of the trace is changing on sub-sample scale. To
        abort downsampling in case of changing endtimes set
        ``strict_length=True``.

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
        >>> tr.decimate(4, strict_length=False, no_filter=True)
        >>> tr.stats.sampling_rate
        0.25
        >>> tr.data
        array([0, 4, 8])
        """
        # check if endtime changes and this is not explicitly allowed
        if strict_length and len(self.data) % factor:
            msg = "Endtime of trace would change and strict_length=True."
            raise ValueError(msg)

        # do automatic lowpass filtering
        if not no_filter:
            # be sure filter still behaves good
            if factor > 16:
                msg = "Automatic filter design is unstable for decimation " + \
                      "factors above 16. Manual decimation is necessary."
                raise ArithmeticError(msg)
            freq = self.stats.sampling_rate * 0.5 / float(factor)
            self.filter('lowpassCheby2', freq=freq, maxorder=12)

        # actual downsampling, as long as sampling_rate is a float we would not
        # need to convert to float, but let's do it as a safety measure
        from obspy.signal import integerDecimation
        self.data = integerDecimation(self.data, factor)
        self.stats.sampling_rate = self.stats.sampling_rate / float(factor)

        # add processing information to the stats dictionary
        proc_info = "downsample:integerDecimation:%s" % factor
        self._addProcessingInfo(proc_info)

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

        Standard deviation is calculated by numpy method
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

    def differentiate(self, type='gradient', **options):
        """
        Method to differentiate the trace with respect to time.

        :type type: ``'gradient'``, optional
        :param type: Method to use for differentiation. Defaults to
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
        type = type.lower()
        # retrieve function call from entry points
        func = _getFunctionFromEntryPoint('differentiate', type)
        # differentiate
        self.data = func(self.data, self.stats.delta, **options)
        # add processing information to the stats dictionary
        proc_info = "differentiate:%s" % type
        self._addProcessingInfo(proc_info)

    def integrate(self, type='cumtrapz', **options):
        """
        Method to integrate the trace with respect to time.

        :type type: ``'cumtrapz'``, optional
        :param type: Method to use for integration. Defaults to
            ``'cumtrapz'``. See the `Supported Methods`_ section below for
            further details.

        .. note::

            This operation is performed in place on the actual data arrays. The
            raw data is not accessible anymore afterwards. To keep your
            original data, use :meth:`~obspy.core.trace.Trace.copy` to create
            a copy of your trace object.
            This also makes an entry with information on the applied processing
            in ``stats.processing`` of this trace.

        .. rubric:: _`Supported Methods`

        ``'cumtrapz'``
            Trapezoidal rule to cumulatively compute integral (uses
            :func:`scipy.integrate.cumtrapz`). Result has one sample less then
            the input!

        ``'trapz'``
            Trapezoidal rule to compute integral from samples (uses
            :func:`scipy.integrate.trapz`).

        ``'simps'``
            Simpson's rule to compute integral from samples (uses
            :func:`scipy.integrate.simps`).

        ``'romb'``
            Romberg Integration to compute integral from (2**k + 1)
            evenly-spaced samples. (uses :func:`scipy.integrate.romb`).
        """
        type = type.lower()
        # retrieve function call from entry points
        func = _getFunctionFromEntryPoint('integrate', type)
        # handle function specific settings
        if func.__module__.startswith('scipy'):
            # scipy needs to set dx keyword if not given in options
            if 'dx' not in options:
                options['dx'] = self.stats.delta
            args = [self.data]
        else:
            args = [self.data, self.stats.delta]
        # integrating
        self.data = func(*args, **options)
        # add processing information to the stats dictionary
        proc_info = "integrate:%s" % (type)
        self._addProcessingInfo(proc_info)

    def detrend(self, type='simple', **options):
        """
        Method to remove a linear trend from the trace.

        :type type: ``'linear'``, ``'constant'``, ``'demean'`` or ``'simple'``,
            optional
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
        func = _getFunctionFromEntryPoint('detrend', type)
        # handle function specific settings
        if func.__module__.startswith('scipy'):
            # scipy need to set the type keyword
            if type == 'demean':
                type = 'constant'
            options['type'] = type
        # detrending
        self.data = func(self.data, **options)
        # add processing information to the stats dictionary
        proc_info = "detrend:%s:%s" % (type, options)
        self._addProcessingInfo(proc_info)

    def taper(self, type='cosine', *args, **kwargs):
        """
        Method to taper the trace.

        Optional (and sometimes necessary) options to the tapering function can
        be provided as args and kwargs. See respective function definitions in
        `Supported Methods`_ section below.

        :type type: str
        :param type: Type of taper to use for detrending. Defaults to
            ``'cosine'``.  See the `Supported Methods`_ section below for
            further details.

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
            :func:`obspy.signal.invsim.cosTaper`.
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
        # retrieve function call from entry points
        func = _getFunctionFromEntryPoint('taper', type)
        # tapering. tapering functions are expected to accept the number of
        # samples as first argument and return an array of values between 0 and
        # 1 with the same length as the data
        self.data = self.data * func(self.stats.npts, *args, **kwargs)
        # add processing information to the stats dictionary
        proc_info = "taper:%s:%s:%s" % (type, args, kwargs)
        self._addProcessingInfo(proc_info)

    def normalize(self, norm=None):
        """
        Method to normalize the trace to its absolute maximum.

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
        >>> tr.normalize()
        >>> tr.data
        array([ 0.        , -0.33333333,  1.        ,  0.66666667])
        >>> tr.stats.processing
        ['normalize:9']
        >>> tr = Trace(data=np.array([0.3, -3.5, -9.2, 6.4]))
        >>> tr.normalize()
        >>> tr.data
        array([ 0.0326087 , -0.38043478, -1.        ,  0.69565217])
        >>> tr.stats.processing
        ['normalize:-9.2']
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

        self.data = self.data.astype("float64")
        self.data /= abs(norm)

        # add processing information to the stats dictionary
        proc_info = "normalize:%s" % norm
        self._addProcessingInfo(proc_info)

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
        Adds the given informational string to the `processing` field in the
        trace's :class:`~obspy.core.trace.stats.Stats` object.
        """
        proc = self.stats.setdefault('processing', [])
        proc.append(info)

    def split(self):
        """
        Splits Trace object containing gaps using a NumPy masked array into
        several traces.

        :rtype: list
        :returns: List of split traces. A gapless trace will still be
            returned as list with only one entry.
        """
        if not isinstance(self.data, np.ma.masked_array):
            # no gaps
            return [self]
        slices = flatnotmaskedContiguous(self.data)
        trace_list = []
        for slice in slices:
            if slice.step:
                raise NotImplementedError("step not supported")
            stats = self.stats.copy()
            tr = Trace(header=stats)
            tr.stats.starttime += (stats.delta * slice.start)
            tr.data = self.data[slice.start:slice.stop]
            trace_list.append(tr)
        return trace_list

    def times(self):
        """
        For convenient plotting compute a Numpy array of seconds since
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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
