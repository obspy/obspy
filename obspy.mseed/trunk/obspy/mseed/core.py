# -*- coding: utf-8 -*-

from obspy.mseed import libmseed
from obspy.numpy import array
from obspy.util import Stats
import os



class MSEEDTrace(object):
    __format__ = 'MSEED'
    
    def __init__(self, filename = None, data = None, header = None, **kwargs):
        if filename:
            self.read(filename, **kwargs)
        if data != None:
            self.data = data
        if header:
            # reset header information
            self.stats = Stats()
            # station name
            self.stats.station = header['station']
            # start time of seismogram in seconds since 1970 (float)
            self.stats.julday = float(header['starttime']/1000000)
            self.stats.starttime = header['starttime']
            # sampling rate in Hz (float)
            self.stats.sampling_rate = header['samprate']
            # number of samples/data points (int)
            self.stats.npts = header['samplecnt']
            # network ID
            self.stats.network = header['network']
            # location ID
            self.stats.location = header['location']
            # channel ID
            self.stats.channel = header['channel']
            # data quality indicator
            self.stats.dataquality = header['dataquality']
            # ent time of seismogram in seconds since 1970 (float)
            self.stats.endtime = float(header['endtime']/1000000)
            # type, not actually used by libmseed
            self.stats.type = header['type']
    
    def read(self, filename, **kwargs):
        __libmseed__ = libmseed()
        if not os.path.exists(filename):
            msg = "File not found '%s'" % (filename)
            raise IOError(msg)
        # read MiniSEED file
        trace_list = __libmseed__.readMSTraces(filename)
        header = trace_list[0][0]
        data = trace_list[0][1]
        # reset header information
        self.stats = Stats()
        # station name
        self.stats.station = header['station']
        # start time of seismogram in seconds since 1970 (float)
        self.stats.julday = float(header['starttime']/1000000)
        self.stats.starttime = header['starttime']
        # sampling rate in Hz (float)
        self.stats.sampling_rate = header['samprate']
        # number of samples/data points (int)
        self.stats.npts = header['samplecnt']
        # network ID
        self.stats.network = header['network']
        # location ID
        self.stats.location = header['location']
        # channel ID
        self.stats.channel = header['channel']
        # data quality indicator
        self.stats.dataquality = header['dataquality']
        # ent time of seismogram in seconds since 1970 (float)
        self.stats.endtime = float(header['endtime']/1000000)
        # type, not actually used by libmseed
        self.stats.type = header['type']
        # the actual seismogram data
        self.data = data
    
    def write(self, filename=None, **kwargs):
        raise NotImplementedError
    
    
    
class MSEEDStream(object):
    __format__ = 'MSEED'
    
    def __init__(self, filename=None, **kwargs):
        if filename:
            self.read(filename, **kwargs)
    
    def read(self, filename, **kwargs):
        __libmseed__ = libmseed()
        if not os.path.exists(filename):
            msg = "File not found '%s'" % (filename)
            raise IOError(msg)
        # read MiniSEED file
        trace_list = __libmseed__.readMSTraces(filename)
        # Create traces attribut of the stream object.
        self.traces = []
        # Loop over all traces found in the file.
        for _i in xrange(len(trace_list)):
            # Create new MSEEDTrace object for each trace in the list.
            evalstring = 'one_trace'+str(_i)+' = MSEEDTrace(header = trace_list['+\
                        str(_i)+'][0], data = trace_list['+str(_i)+'][1])'
            exec(evalstring)
            evalstring = 'self.traces.append( one_trace'+str(_i)+')'
            exec(evalstring)
    
    def write(self, filename=None, **kwargs):
        raise NotImplementedError
