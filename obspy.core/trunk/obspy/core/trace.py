# -*- coding: utf-8 -*-

from copy import deepcopy
# from numpy.ma import masked_array, is_nan does not work with some
# Python/NumPy combinations.
import numpy as np
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import AttribDict
import obspy


class Stats(AttribDict):
    pass


class Trace(object):
    """
    ObsPy Trace class.
    
    This class contains information about a single trace.
    
    @type data: Numpy ndarray 
    @param data: Numpy ndarray of data samples
    @param header: Dictionary containing header fields
    @param address: Address of data to be freed when trace is deleted
    """
    def __init__(self, data=np.array([]), header=None):
        if header == None:
            # Default values: For detail see
            # http://svn.geophysik.uni-muenchen.de/trac/obspy/wiki/\
            # KnownIssues#DefaultParameterValuesinPython
            header = {}
        # set some defaults if not set yet
        for default in ['station', 'network', 'location', 'channel']:
            header.setdefault(default, '')
        header.setdefault('npts', len(data))
        self.stats = Stats(header)
        for key, value in header.iteritems():
            if not isinstance(value, dict):
                continue
            self.stats[key] = Stats(value)
        self.data = data

    def __str__(self):
        out = "%(network)s.%(station)s.%(location)s.%(channel)s | " + \
              "%(starttime)s - %(endtime)s | " + \
              "%(sampling_rate).1f Hz, %(npts)d samples"
        return out % (self.stats)

    def __len__(self):
        """
        Returns the number of data samples of a L{Trace} object.
        
        @rtype: int 
        @return: Number of data samples.
        """
        return len(self.data)

    count = __len__

    def __getitem__(self, index):
        """ 
        __getitem__ method of L{Trace} object.
        
        @rtype: list 
        @return: List of data points 
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
        out.stats.endtime = rt.stats.endtime
        return out

    def getId(self):
        out = "%(network)s.%(station)s.%(location)s.%(channel)s"
        return out % (self.stats)

    id = property(getId)

    def plot(self, **kwargs):
        """
        Creates a graph of this L{Trace} object.
        """
        try:
            from obspy.imaging import waveform
        except:
            msg = "Please install module obspy.imaging to be able to " + \
                  "plot ObsPy Trace objects."
            raise Exception(msg)
        waveform.plotWaveform(self, **kwargs)

    def write(self, filename, format, **kwargs):
        """
        Saves trace into a file.
        """
        obspy.Stream([self]).write(filename, format, **kwargs)

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
        self.stats.npts = len(self.data)
        self.stats.endtime = endtime

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

    def _verify(self):
        """
        Verifies this L{Trace} object with saved stats values.
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
