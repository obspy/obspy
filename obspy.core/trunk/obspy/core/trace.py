# -*- coding: utf-8 -*-

from copy import deepcopy
from numpy import array, NaN, concatenate
from obspy.core import UTCDateTime, Stats
from obspy.core.util import libc

class Trace(object):
    """
    ObsPy Trace class.
    
    This class contains information about a single trace.
    
    @type data: Numpy ndarray 
    @ivar data: Data samples 
    @param data: Numpy ndarray of data samples
    @param header: Dictionary containing header fields
    @param address: Address of data to be freed when trace is deleted
    """
    def __init__(self, data=array([]), header={}, address=None):
        self.address = address
        self.stats = Stats()
        self.stats.update(header)
        for key, value in header.iteritems():
            if not isinstance(value, dict):
                continue
            self.stats[key] = Stats(value)
        self.data = data
        self.stats.setdefault('npts', len(self.data))
        # set some defaults if not set yet
        for default in ['station', 'network', 'location', 'channel']:
            self.stats.setdefault(default, '')

    def __del__(self):
        if self.address:
            libc.free(self.address)

    def __str__(self):
        out = "%(network)s.%(station)s.%(location)s.%(channel)s | " + \
              "%(starttime)s - %(endtime)s | " + \
              "%(sampling_rate).1f Hz, %(npts)d samples"
        return out % (self.stats)

    def __len__(self):
        """
        Returns the number of data samples of a L{Trace} object.
        """
        return len(self.data)

    count = __len__

    def __getitem__(self, index):
        """ 
        __getitem__ method of L{Trace} object.
          
        @return: List of data points 
        """
        return self.data[index]

    def __add__(self, trace):
        """
        Adds a Trace object to this Trace
        
        It will automatically append the data by interpolating overlaps or
        filling gaps with NaN samples. Sampling rate and Trace ID must be 
        the same.
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
        # check if overlap or gap
        if delta <= 0:
            # overlap
            delta = abs(delta)
            out = deepcopy(lt)
            ltotal = len(lt)
            lend = ltotal - delta
            ldata = array(lt.data[0:lend])
            rdata = array(rt.data[delta:])
            samples = (array(lt.data[lend:]) + array(rt.data[0:delta])) / 2
            out.data = concatenate([ldata, samples, rdata])
            out.stats.endtime = rt.stats.endtime
            out.stats.npts = len(out.data)
        else:
            # gap
            out = deepcopy(lt)
            # get number of missing samples
            nans = array([NaN] * delta)
            out.data = concatenate([lt.data, nans, rt.data])
            out.stats.endtime = rt.stats.endtime
            out.stats.npts = len(out.data)
        return out

    def getId(self):
        out = "%(network)s.%(station)s.%(location)s.%(channel)s"
        return out % (self.stats)

    def plot(self, **kwargs):
        """
        Creates a graph of this L{Trace} object.
        """
        try:
            from obspy.imaging import waveform
        except:
            msg = "Please install module obspy.imaging to be able to " + \
                  "plot ObsPy Trace objects."
            print msg
            raise
        waveform.plotWaveform(self, **kwargs)

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

    def verify(self):
        """
        Verifies this L{Trace} object with saved stats values.
        """
        if len(self.data) != self.stats.npts:
            msg = "ntps(%d) differs from data size(%d)"
            raise Exception(msg % (self.stats.npts, len(self.data)))
        delta = self.stats.endtime - self.stats.starttime
        if delta < 0:
            msg = "End time(%s) before start time(%s)"
            raise Exception(msg % (self.stats.endtime, self.stats.starttime))
        sr = self.stats.sampling_rate
        if int(round(delta * sr)) + 1 != len(self.data):
            msg = "Sample rate(%d) * time delta(%.4lf) + 1 != data size(%d)"
            raise Exception(msg % (sr, delta, len(self.data)))
