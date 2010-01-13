# -*- coding: utf-8 -*-

from copy import deepcopy
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import AttribDict
import numpy as np


class Stats(AttribDict):
    """
    A class containing header information for a single :class:`Trace` object.

    Default header attributes are:

    ============= =================== ============ ================= ====
    Name          Description         Data type    Default value     Mode
    ============= =================== ============ ================= ====
    sampling_rate Sampling rate [Hz]  float        1.0               RW
    delta         Sample distance [s] float        1.0               RW
    calib         Calibration factor  float        1.0               RW
    npts          Number of points    int          0                 RW
    network       Network code        string       ''                RW
    location      Location code       string       ''                RW
    station       Station code        string       ''                RW
    channel       Channel code        string       ''                RW
    starttime     Start time [UTC]    UTCDateTime  1970-01-01 00:00  RW
    endtime       End time [UTC]      UTCDateTime  1970-01-01 00:00  R
    ============= =================== ============ ================= ====

    The attributes `starttime`, `sampling_rate` and `delta` are dynamically
    monitored and used to recalculated `endtime`, `delta` and `sampling_rate`
    on any change. Attribute `endtime` is read only and can not be modified.

        >>> stats = Stats()
        >>> stats.sampling_rate
        1.0
        >>> stats.delta = 0.005
        >>> stats.sampling_rate
        200.0

    All header information of the `Stats` class may be accessed or modified
    either in the dictionary style or directly via the respective attribute.

        >>> stats = Stats()
        >>> stats.network = 'BW'
        >>> stats['network']
        'BW'
        >>> stats['station'] = 'MANZ'
        >>> stats.station
        'MANZ'
    """
    readonly = ['endtime']

    def __init__(self, header={}):
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
        endtime = self.starttime + (self.npts - 1) / float(self.sampling_rate)
        super(Stats, self).__setitem__('endtime', endtime)


class Trace(object):
    """
    A class containing data and meta data about a single continuous trace.

    :type data: `numpy.array`
    :param data: Numpy array of data samples
    :type header: `dict` or :class:`Stats`
    :param header: Dictionary containing header fields
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
        out = "%(network)s.%(station)s.%(location)s.%(channel)s | " + \
              "%(starttime)s - %(endtime)s | " + \
              "%(sampling_rate).1f Hz, %(npts)d samples"
        return out % (self.stats)

    def __len__(self):
        """
        Returns the number of data samples of a L{Trace} object.

        :rtype: int
        :return: Number of data samples.

        Usage:
            >>> tr = Trace(data=[1, 2, 3, 4])
            >>> tr.count()
            4
            >>> len(tr)
            4
        """
        return len(self.data)

    count = __len__

    def __setattr__(self, key, value):
        """
        __setattr__ method of L{Trace} object.
        
        Any change in Trace.data will dynamically set Trace.stats.npts.
        """
        if key == 'data':
            self.stats.npts = len(value)
        return super(Trace, self).__setattr__(key, value)

    def __getitem__(self, index):
        """
        __getitem__ method of L{Trace} object.

        :rtype: list
        :return: List of data points
        """
        return self.data[index]

    def __add__(self, trace):
        """
        Adds a Trace object to this Trace

        It will automatically append the data by interpolating overlaps or
        filling gaps with numpy.NaN samples. Sampling rate and Trace ID must
        be the same.
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
        """
        out = "%(network)s.%(station)s.%(location)s.%(channel)s"
        return out % (self.stats)

    id = property(getId)

    def plot(self, *args, **kwargs):
        """
        Creates a graph of this L{Trace} object.
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
        Saves trace into a file.
        """
        # we need to import here in order to prevent a circular import of 
        # Stream and Trace classes
        from obspy.core import Stream
        Stream([self]).write(filename, format, **kwargs)

    def ltrim(self, starttime):
        """
        Cuts L{Trace} object to given start time.
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
        Cuts L{Trace} object to given end time.
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
        Cuts L{Trace} object to given start and end time.
        """
        # check time order and switch eventually
        if starttime > endtime:
            endtime, starttime = starttime, endtime
        # cut it
        self.ltrim(starttime)
        self.rtrim(endtime)

    def verify(self):
        """
        Verifies current Trace object with header values in stats attribute.
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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
