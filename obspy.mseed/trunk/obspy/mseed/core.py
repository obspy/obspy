# -*- coding: utf-8 -*-

from obspy.core import Stream, Trace
from obspy.mseed import libmseed


def isMSEED(filename):
    """
    Returns true if the file is a Mini-SEED file and false otherwise.
    
    @param filename: File to be read.
    """
    __libmseed__ = libmseed()
    return __libmseed__.isMSEED(filename)


def readMSEED(filename):
    """
    Reads a given Mini-SEED file and returns an obspy.Stream object.
    
    @param filename: Mini-SEED file to be read.
    """
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
        # Create a dictionary to specify how to convert keys.
        convert_dict = {'station': 'station', 'sampling_rate':'samprate',
                        'npts': 'numsamples', 'network': 'network',
                        'location': 'location', 'channel': 'channel',
                        'dataquality': 'dataquality', 'starttime' :
                        'starttime', 'endtime' : 'endtime'}
        # Convert header.
        for _j in convert_dict.keys():
            header[_j] = old_header[convert_dict[_j]]
        # Convert times to obspy.UTCDateTime objects.
        header['starttime'] = \
            __libmseed__._convertMSTimeToDatetime(header['starttime'])
        header['endtime'] = \
            __libmseed__._convertMSTimeToDatetime(header['endtime'])
        # Append traces.
        traces.append(Trace(header=header, data=_i[1]))
    return Stream(traces=traces)


def writeMSEED(stream_object, filename, reclen= -1, encoding= -1,
               byteorder= -1, flush= -1, verbose=0):
    """
    Write Miniseed file from a Stream objext.
    
    All kwargs are passed directly to obspy.mseed.writeMSTraces.
    
    @param stream_object: obspy.Stream object.
    @param filename: Name of the output file
    @param reclen: should be set to the desired data record length in bytes
        which must be expressible as 2 raised to the power of X where X is
        between (and including) 8 to 20. -1 defaults to 4096
    @param encoding: should be set to one of the following supported
        Mini-SEED data encoding formats: DE_ASCII (0), DE_INT16 (1),
        DE_INT32 (3), DE_FLOAT32 (4), DE_FLOAT64 (5), DE_STEIM1 (10)
        and DE_STEIM2 (11). -1 defaults to STEIM-2 (11)
    @param byteorder: must be either 0 (LSBF or little-endian) or 1 (MBF or 
        big-endian). -1 defaults to big-endian (1)
    @param flush: if it is not zero all of the data will be packed into 
        records, otherwise records will only be packed while there are
        enough data samples to completely fill a record.
    @param verbose: controls verbosity, a value of zero will result in no 
        diagnostic output.
    """
    __libmseed__ = libmseed()
    traces = stream_object.traces
    trace_list = []
    convert_dict = {'station': 'station', 'samprate':'sampling_rate',
                'numsamples': 'npts', 'network': 'network',
                'location': 'location', 'channel': 'channel',
                'dataquality': 'dataquality', 'starttime' :
                'starttime', 'endtime' : 'endtime'}
    for _i in traces:
        header = {}
        for _j in convert_dict.keys():
            header[_j] = _i.stats[convert_dict[_j]]
        # Convert obspy.UTCDateTime times to Mini-SEED times.
        header['starttime'] = \
            __libmseed__._convertDatetimeToMSTime(header['starttime'])
        header['endtime'] = \
            __libmseed__._convertDatetimeToMSTime(header['endtime'])
        # Save samples as integers.
        header['sampletype'] = 'i'
        # Set Dataquality to indeterminate if it is not already set.
        if len(header['dataquality']) == 0:
            header['dataquality'] = 'D'
        # Fill the samplecnt attribute.
        header['samplecnt'] = len(_i.data)
        trace_list.append([header, _i.data])
    # Write resulting trace_list to Mini-SEED file.
    __libmseed__.writeMSTraces(trace_list, outfile=filename, reclen=reclen,
                               encoding=encoding, byteorder=byteorder,
                               flush=flush, verbose=verbose)
