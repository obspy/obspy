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
import numpy as np
import math


class Stats(AttribDict):
    """
    A container for additional header information of a ObsPy Trace object.

    A ``Stats`` object may contain all header information (also known as meta
    data) of a :class:`~obspy.core.trace.Trace` object. Those headers may be
    accessed or modified either in the dictionary style or directly via a
    corresponding attribute. There are various default attributes which are
    required by every waveform import and export modules within ObsPy such as
    :mod:`obspy.mseed`.

    Basic Usage
    -----------
    >>> stats = Stats()
    >>> stats.network = 'BW'
    >>> stats['network']
    'BW'
    >>> stats['station'] = 'MANZ'
    >>> stats.station
    'MANZ'

    Parameters
    ----------
    header : dict or :class:`~obspy.core.trace.Stats`, optional
        Dictionary containing meta information of a single
        :class:`~obspy.core.trace.Trace` object. Possible keywords are
        summarized in the following attributes section.

    Attributes
    ----------
    sampling_rate : float, optional
        Sampling rate in hertz (default value is 1.0).
    delta : float, optional
        Sample distance in seconds (default value is 1.0).
    calib : float, optional
        Calibration factor (default value is 1.0).
    npts : int, optional
        Number of sample points (default value is 0, which implies that no data
        is present).
    network : string, optional
        Network code (default is an empty string).
    location : string, optional
        Location code (default is an empty string).
    station : string, optional
        Station code (default is an empty string).
    channel : string, optional
        Channel code (default is an empty string).
    starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        Date and time of the first data sample given in UTC (default value is
        "1970-01-01T00:00:00.0Z").
    endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        Date and time of the last data sample given in UTC
        (default value is "1970-01-01T00:00:00.0Z").

    Notes
    -----
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
            Endtime is currently calculated as
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
        >>> trace.data = [1, 2, 3, 4]
        >>> trace.stats.npts
        4
    """

    readonly = ['endtime']
    sampling_rate = 1.0
    starttime = UTCDateTime(0)
    npts = 0

    def __init__(self, header={}):
        """
        """
        # set default values without calculating derived entries
        super(Stats, self).__setitem__('sampling_rate', 1.0)
        super(Stats, self).__setitem__('starttime', UTCDateTime(0))
        super(Stats, self).__setitem__('npts', 0)
        # set default values for all other headers
        header.setdefault('calib', 1.0)
        for default in ['station', 'network', 'location', 'channel']:
            header.setdefault(default, '')
        # initialize
        super(Stats, self).__init__(header)
        # calculate derived values
        self._calculateDerivedValues()

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
            # set derived values
            self._calculateDerivedValues()
            return
        # all other keys
        if isinstance(value, dict):
            super(Stats, self).__setitem__(key, AttribDict(value))
        else:
            super(Stats, self).__setitem__(key, value)

    __setattr__ = __setitem__

    def _calculateDerivedValues(self):
        """
        Calculates derived headers such as `delta` and `endtime`.
        """
        # set delta
        delta = 1.0 / float(self.sampling_rate)
        super(Stats, self).__setitem__('delta', delta)
        # set endtime
        if self.npts == 0:
            # XXX: inconsistent
            delta = 0
        else:
            delta = (self.npts - 1) / float(self.sampling_rate)
        endtime = self.starttime + delta
        self.__dict__['endtime'] = endtime

    def setEndtime(self, value):
        msg = "Attribute \"endtime\" in Stats object is read only!"
        raise AttributeError(msg)

    def getEndtime(self):
        return self.__dict__['endtime']

    endtime = property(getEndtime, setEndtime)


class Trace(object):
    """
    An object containing data of a continuous series, such as a seismic trace.

    Parameters
    ----------
    data : `numpy.array` or `ma.masked_array`
        Numpy array of data samples
    header : dict or :class:`~obspy.core.trace.Stats`
        Dictionary containing header fields

    Supported Operations
    --------------------
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
        # set some defaults if not set yet
        if header == None:
            # Default values: For detail see
            # http://svn.geophysik.uni-muenchen.de/trac/obspy/wiki/\
            # KnownIssues#DefaultParameterValuesinPython
            header = {}
        header.setdefault('npts', len(data))
        self.stats = Stats(header)
        # set data without changing npts in stats object (for headonly option)
        super(Trace, self).__setattr__('data', data)

    def __str__(self):
        """
        Returns short summary string of the current trace.

        Returns
        -------
        string
            short summary string of the current trace containing the SEED
            identifier, start time, end time, sampling rate and number of
            points of the current trace.

        Example
        -------
        >>> tr = Trace(header={'station':'FUR', 'network':'GR'})
        >>> str(tr) #doctest: +ELLIPSIS
        'GR.FUR.. | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:00.00000...'
        """
        out = "%(network)s.%(station)s.%(location)s.%(channel)s | " + \
              "%(starttime)s - %(endtime)s | " + \
              "%(sampling_rate).1f Hz, %(npts)d samples"
        return out % (self.stats)

    def __len__(self):
        """
        Returns number of data samples of the current trace.

        Returns
        -------
        int
            Number of data samples.

        Example
        -------
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
            self.stats.npts = len(value)
        return super(Trace, self).__setattr__(key, value)

    def __getitem__(self, index):
        """
        __getitem__ method of Trace object.

        :rtype: list
        :return: List of data points
        """
        return self.data[index]

    def __add__(self, trace, method=0, interpolation_samples=0,
                fill_value=None):
        """
        Adds another Trace object to current trace.

        Trace data will be converted into a NumPy masked array data type if
        any gaps are present. This behavior may be prevented by setting the
        ``fill_value`` parameter. The ``method`` argument controls the
        handling of overlapping data values.

        Sampling rate, data type and trace.id of both traces must match.

        Parameters
        ----------
        method : [ 0 | 1 ], optional
            Method to handle overlaps of traces (default is 0). See the
            table given in the notes section below for further details.
        fill_value : int or float, optional
            Fill value for gaps (default is None). Traces will be converted to
            NumPy masked arrays if no value is given and gaps are present.
        interpolation_samples : int, optional
            Used only for method 1. It specifies the number of samples which
            are used to interpolate between overlapping traces (default is 0).
            If set to -1 all overlapping samples are interpolated.

        Notes
        -----
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
                prevent steps.
                
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
                    1 + 2  : AAAABCDEFFFF ()
                
                Any contained traces with different data will be discarted::
                
                    Trace 1: AAAAAAAAAAAA (contained trace)
                    Trace 2:     FF
                    1 + 2  : AAAAAAAAAAAA
        ======  ===============================================================
        """
        if not isinstance(trace, Trace):
            raise TypeError
        #  check id
        if self.getId() != trace.getId():
            raise TypeError("Trace ID differs")
        #  check sample rate
        if self.stats.sampling_rate != trace.stats.sampling_rate:
            raise TypeError("Sampling rate differs")
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
        sr = self.stats.sampling_rate
        delta = int(math.floor(round((rt.stats.starttime - \
                                      lt.stats.endtime) * sr, 7))) - 1
        delta_endtime = lt.stats.endtime - rt.stats.endtime
        # create the returned trace
        out = Trace(header=deepcopy(lt.stats))
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
                ls = lt.data[-delta - 1]
                if interpolation_samples == -1:
                    interpolate_samples = delta
                rs = rt.data[interpolate_samples]
                # include left and right sample (delta + 2)
                interpolation = ls + np.arange(interpolate_samples + 2) * \
                                (ls - rs) / float(interpolate_samples - 1 + 2)
                # cut ls and rs and ensure correct data type
                interpolation = interpolation[1:-1].require(lt.data.dtype)
                data = [lt.data[:-delta], interpolation,
                        rt.data[interpolate_samples:]]
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
            data = [lt.data, rt.data]
        else:
            # gap
            gap = createEmptyDataChunk(delta, lt.data.dtype, fill_value)
            data = [lt.data, gap, rt.data]
        # merge traces depending on numpy array type
        if True in [np.ma.is_masked(_i) for _i in data]:
            data = np.ma.concatenate(data)
        else:
            data = np.concatenate(data)
        out.data = data
        return out

    def getId(self):
        """
        Returns a SEED compatible identifier of the trace.

        The SEED identifier contains the network, station, location and channel
        code for the current Trace object.

        Returns
        -------
        string
            SEED identifier

        Example
        -------
        >>> meta = {'station':'MANZ', 'network':'BW', 'channel':'EHZ'}
        >>> trace = Trace(header=meta)
        >>> trace.getId()
        'BW.MANZ..EHZ'
        >>> trace.id
        'BW.MANZ..EHZ'
        """
        out = "%(network)s.%(station)s.%(location)s.%(channel)s"
        return out % (self.stats)

    id = property(getId)

    def plot(self, *args, **kwargs):
        """
        Creates a simple graph of the current trace.

        Basic Usage
        -----------
        >>> data = np.sin(np.linspace(0,2*np.pi,1000))
        >>> tr = Trace(data=data)
        >>> tr.plot() # doctest: +SKIP

        .. plot::

            import numpy as np
            from obspy.core import Trace
            data = np.sin(np.linspace(0,2*np.pi,1000))
            tr = Trace(data=data)
            tr.plot()
        """
        try:
            from obspy.imaging.waveform import WaveformPlotting
        except:
            msg = "Please install module obspy.imaging to be able to " + \
                  "plot ObsPy Trace objects."
            raise Exception(msg)
        waveform = WaveformPlotting(stream=self, *args, **kwargs)
        return waveform.plotWaveform()

    def write(self, filename, format, **kwargs):
        """
        Saves current trace into a file.

        Parameters
        ----------
        filename : string
            Name of the output file.
        format : string
            Name of the output format.
            See :func:`~obspy.core.stream.read` for all possible formats.

        Basic Usage
        -----------
        >>> tr = Trace()
        >>> tr.write("out.mseed", format="MSEED") # doctest: +SKIP
        """
        # we need to import here in order to prevent a circular import of 
        # Stream and Trace classes
        from obspy.core import Stream
        Stream([self]).write(filename, format, **kwargs)

    def ltrim(self, starttime, pad=False):
        """
        Cuts current trace to given start time.

        Basic Usage
        -----------
        >>> tr = Trace(data=np.arange(0, 10))
        >>> tr.stats.delta = 1.0
        >>> tr.ltrim(tr.stats.starttime + 8)
        >>> tr.data
        array([8, 9])
        >>> tr.stats.starttime
        UTCDateTime(1970, 1, 1, 0, 0, 8)
        """
        if isinstance(starttime, float) or isinstance(starttime, int):
            starttime = UTCDateTime(self.stats.starttime) + starttime
        elif not isinstance(starttime, UTCDateTime):
            raise TypeError
        # check if in boundary
        delta = int(math.floor(round((starttime - self.stats.starttime) * \
                                     self.stats.sampling_rate, 7)))
        # Adjust starttime only if delta is greater than zero or if the values
        # are padded with masked arrays.
        if delta > 0 or pad:
            self.stats.starttime += delta * self.stats.delta
        if delta == 0 or (delta < 0 and not pad):
            return
        elif delta < 0 and pad:
            try:
                gap = createEmptyDataChunk(abs(delta), self.data.dtype)
            except ValueError:
                # createEmptyDataChunk returns negative ValueError ?? for
                # too large number of pointes, e.g. 189336539799
                raise Exception("Time offset between starttime and " + \
                                "trace.starttime too large")
            self.data = np.ma.concatenate((gap, self.data))
            return
        elif starttime > self.stats.endtime:
            self.data = np.empty(0)
            return
        elif delta > 0:
            self.data = self.data[delta:]

    def rtrim(self, endtime, pad=False):
        """
        Cuts current trace to given end time.

        Basic Usage
        -----------
        >>> tr = Trace(data=np.arange(0, 10))
        >>> tr.stats.delta = 1.0
        >>> tr.rtrim(tr.stats.starttime + 2)
        >>> tr.data
        array([0, 1, 2])
        >>> tr.stats.endtime
        UTCDateTime(1970, 1, 1, 0, 0, 2)
        """
        if isinstance(endtime, float) or isinstance(endtime, int):
            endtime = UTCDateTime(self.stats.endtime) - endtime
        elif not isinstance(endtime, UTCDateTime):
            raise TypeError
        # check if in boundary
        delta = int(math.floor(round((endtime - self.stats.endtime) * \
                               self.stats.sampling_rate, 7)))
        if delta == 0 or (delta > 0 and not pad):
            return
        if delta > 0 and pad:
            try:
                gap = createEmptyDataChunk(delta, self.data.dtype)
            except ValueError:
                # createEmptyDataChunk returns negative ValueError ?? for
                # too large number of pointes, e.g. 189336539799
                raise Exception("Time offset between starttime and " + \
                                "trace.starttime too large")
            self.data = np.ma.concatenate((self.data, gap))
            return
        elif endtime < self.stats.starttime:
            self.stats.starttime = self.stats.endtime + \
                                   delta * self.stats.delta
            self.data = np.empty(0)
            return
        # cut from right
        if pad:
            delta = abs(delta)
            total = len(self.data) - delta
            if endtime == self.stats.starttime:
                total = 1
            self.data = self.data[:total]
        else:
            delta = abs(delta)
            total = len(self.data) - delta
            if endtime == self.stats.starttime:
                total = 1
            self.data = self.data[:total]

    def trim(self, starttime, endtime, pad=False):
        """
        Cuts current trace to given start and end time.

        Basic Usage
        -----------
        >>> tr = Trace(data=np.arange(0, 10))
        >>> tr.stats.delta = 1.0
        >>> t = tr.stats.starttime
        >>> tr.trim(t + 2.000001, t + 7.999999)
        >>> tr.data
        array([2, 3, 4, 5, 6, 7])
        """
        # check time order and swap eventually
        if starttime > endtime:
            raise Exception("startime is larger than endtime")
        # cut it
        self.ltrim(starttime, pad)
        self.rtrim(endtime, pad)

    cut = trim
    lcut = ltrim
    rcut = rtrim

    def slice(self, starttime, endtime):
        """
        Returns a new Trace object with data going from start to end time.

        Does not copy data but just passes a reference to it.

        Basic Usage
        -----------
        >>> tr = Trace(data=np.arange(0, 10))
        >>> tr.stats.delta = 1.0
        >>> t = tr.stats.starttime
        >>> tr2 = tr.slice(t + 2, t + 8)
        >>> tr2.data
        array([2, 3, 4, 5, 6, 7, 8])
        """
        tr = copy(self)
        tr.stats = deepcopy(self.stats)
        tr.trim(starttime, endtime)
        return tr

    def verify(self):
        """
        Verifies current trace object against available meta data.

        Basic Usage
        -----------
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
        elif self.stats.npts != 0:
            msg = "Data size should be 0, but is %d"
            raise Exception(msg % self.stats.npts)
        if not isinstance(self.stats, Stats):
            msg = "Attribute stats must be an instance of obspy.core.Stats"
            raise Exception(msg)
        if isinstance(self.data, np.ndarray) and \
           self.data.dtype.byteorder not in ["=", "|"]:
            msg = "Trace data should be stored as numpy.ndarray in the " + \
                  "system specific byte order."
            print self.data.dtype.byteorder
            raise Exception(msg)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
