# -*- coding: utf-8 -*-

from obspy.mseed import libmseed
from obspy.numpy import array
from obspy.util import Stats
from obspy.core import Stream, Trace
import os


def isMSEED(filename):
    """
    """
    __libmseed__ = libmseed()
    return __libmseed__.isMSEED(filename)

def readMSEED(filename):
    __libmseed__ = libmseed()
    # read MiniSEED file
    trace_list = __libmseed__.readMSTraces(filename)
    # Create a list containing all the traces.
    traces = []
    # Loop over all traces found in the file.
    for _i in trace_list:
        # Convert header to be compatible with obspy.core.
        old_header = _i[0]
        header = {}
        convert_dict = {'station': 'station', 'sampling_rate':'samprate',
                        'npts': 'numsamples', 'network': 'network',
                        'location': 'location', 'channel': 'channel',
                        'dataquality': 'dataquality', 'starttime' :
                        'starttime', 'endtime' : 'endtime'}
        for _j in convert_dict.keys():
            header[_j] = old_header[convert_dict[_j]]
        header['extra'] = {'sampletype' : old_header['sampletype']}
        header['starttime'] =\
            __libmseed__._convertMSTimeToDatetime(header['starttime'])
        header['endtime'] =\
            __libmseed__._convertMSTimeToDatetime(header['endtime'])
        traces.append(Trace(header = header, data = _i[1]))
    return Stream(traces = traces)