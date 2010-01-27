# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Trace objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from copy import deepcopy
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import AttribDict
import numpy as np


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
        # filter read only attributes
        if key in self.readonly:
            msg = "Attribute \"%s\" in Stats object is read only!" % (key)
            raise AttributeError(msg)
        # keys which need to refresh derived values
        if key in ['delta', 'sampling_rate', 'starttime', 'npts']:
            if key == 'delta':
                key = 'sampling_rate'
                value = 1.0 / float(value)
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
        super(Stats, self).__setitem__('endtime', endtime)


class Trace(object):
    """
    An object containing data of a continuous series, such as a seismic trace.

    Parameters
    ----------
    data : `numpy.array` or `ma.masked_array`
        Numpy array of data samples
    header : dict or :class:`~obspy.core.trace.Stats`
        Dictionary containing header fields
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
        >>> trace = Trace(data=[1, 2, 3, 4])
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

    def __add__(self, trace):
        """
        Adds another Trace object to current trace.

        Data is automatically appended by interpolating overlaps or filling
        gaps with numpy.NaN samples. Sampling rate and trace.id of both
        traces must match.
        """
        if not isinstance(trace, Trace):
            raise TypeError
        #  check id
        if self.getId() != trace.getId():
            raise TypeError("Trace ID differs")
        #  check sample rate
        if self.stats.sampling_rate != trace.stats.sampling_rate:
            raise TypeError("Sampling rate differs")
        # check times
        if self.stats.starttime <= trace.stats.starttime and \
           self.stats.endtime >= trace.stats.endtime:
            # new trace is within this trace
            return deepcopy(self)
        elif self.stats.starttime >= trace.stats.starttime and \
           self.stats.endtime <= trace.stats.endtime:
            # this trace is within new trace
            return deepcopy(trace)
        # shortcuts
        if self.stats.starttime <= trace.stats.starttime:
            lt = self
            rt = trace
        else:
            rt = self
            lt = trace
        sr = self.stats.sampling_rate
        delta = int(round((rt.stats.starttime - lt.stats.endtime) * sr)) - 1
        out = deepcopy(lt)
        # check if overlap or gap
        if delta <= 0:
            # overlap
            delta = abs(delta)
            ltotal = len(lt)
            lend = ltotal - delta
            ldata = np.asanyarray(lt.data)
            rdata = np.asanyarray(rt.data)
            samples = (ldata[lend:] + rdata[0:delta]) / 2
            if np.ma.is_masked(ldata) or np.ma.is_masked(rdata):
                out.data = np.ma.concatenate([ldata[0:lend], samples,
                                              rdata[delta:]])
            else:
                out.data = np.concatenate([ldata[0:lend], samples,
                                           rdata[delta:]])
        else:
            # gap
            # get number of missing samples
            nans = np.empty(delta)
            nans[:] = np.NaN
            out.data = np.concatenate([lt.data, nans, rt.data])
            # Create masked array.
            out.data = np.ma.masked_array(out.data, np.isnan(out.data))
        out.stats.npts = out.data.size
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
        waveform.plotWaveform()

    def write(self, filename, format, **kwargs):
        """
        Saves current trace into a file.

        Parameters
        ----------
        filename : string
            Name of the output file.
        format : string
            Name of the output format.
            .. :seealso:: 
                :func:`~obspy.core.stream.read()` for all possible formats.

        Basic Usage
        -----------
        >>> tr = Trace()
        >>> tr.write("out.mseed", format="MSEED") # doctest: +SKIP
        """
        # we need to import here in order to prevent a circular import of 
        # Stream and Trace classes
        from obspy.core import Stream
        Stream([self]).write(filename, format, **kwargs)

    def ltrim(self, starttime):
        """
        Cuts current trace to given start time.
        """
        if isinstance(starttime, float) or isinstance(starttime, int):
            starttime = UTCDateTime(self.stats.starttime) + starttime
        elif not isinstance(starttime, UTCDateTime):
            raise TypeError
        # check if in boundary
        if starttime <= self.stats.starttime or \
           starttime >= self.stats.endtime:
            return
        # cut from left
        delta = (starttime - self.stats.starttime)
        samples = int(round(delta * self.stats.sampling_rate))
        self.data = self.data[samples:]
        self.stats.npts = len(self.data)
        self.stats.starttime = starttime

    def rtrim(self, endtime):
        """
        Cuts current trace to given end time.
        """
        if isinstance(endtime, float) or isinstance(endtime, int):
            endtime = UTCDateTime(self.stats.endtime) - endtime
        elif not isinstance(endtime, UTCDateTime):
            raise TypeError
        # check if in boundary
        if endtime >= self.stats.endtime or endtime < self.stats.starttime:
            return
        # cut from right
        delta = (self.stats.endtime - endtime)
        samples = int(round(delta * self.stats.sampling_rate))
        total = len(self.data) - samples
        if endtime == self.stats.starttime:
            total = 1
        self.data = self.data[0:total]
        self.stats.npts = total

    def trim(self, starttime, endtime):
        """
        Cuts current trace to given start and end time.
        """
        # check time order and switch eventually
        if starttime > endtime:
            endtime, starttime = starttime, endtime
        # cut it
        self.ltrim(starttime)
        self.rtrim(endtime)

    def verify(self):
        """
        Verifies current trace object against available meta data.
        """
        if len(self) != self.stats.npts:
            msg = "ntps(%d) differs from data size(%d)"
            raise Exception(msg % (self.stats.npts, len(self.data)))
        delta = self.stats.endtime - self.stats.starttime
        if delta < 0:
            msg = "End time(%s) before start time(%s)"
            raise Exception(msg % (self.stats.endtime, self.stats.starttime))
        sr = self.stats.sampling_rate
        if int(round(delta * sr)) + 1 != len(self.data):
            msg = "Sample rate(%f) * time delta(%.4lf) + 1 != data size(%d)"
            raise Exception(msg % (sr, delta, len(self.data)))
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
