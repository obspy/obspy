# -*- coding: utf-8 -*-

from obspy.core import Stream, Trace
from obspy.mseed import LibMSEED
from obspy.mseed.headers import ENCODINGS
import numpy as np
import sys


def isMSEED(filename):
    """
    Returns true if the file is a Mini-SEED file and false otherwise.
    
    :param filename: File to be read.
    """
    __libmseed__ = LibMSEED()
    return __libmseed__.isMSEED(filename)

def readMSEED(filename, headonly=False, starttime=None, endtime=None,
              reclen= -1, **kwargs):
    """
    Reads a given Mini-SEED file and returns an obspy.Stream object.
    
    :param filename: Mini-SEED file to be read.
    """
    __libmseed__ = LibMSEED()
    # read MiniSEED file
    if headonly:
        trace_list = __libmseed__.readMSHeader(filename, reclen=reclen)
    else:
        kwargs['starttime'] = kwargs.get('starttime', None)
        kwargs['endtime'] = kwargs.get('endtime', None)
        if starttime or endtime:
            trace_list = __libmseed__.readMSTracesViaRecords(filename,
                         starttime=starttime, endtime=endtime, reclen=reclen)
        else:
            #10% faster, problem on windows
            trace_list = __libmseed__.readMSTraces(filename, reclen=reclen)
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
                        'starttime' : 'starttime', 'endtime' : 'endtime'}
        # Convert header.
        for _j, _k in convert_dict.iteritems():
            header[_j] = old_header[_k]
        # Dataquality is Mini-SEED only and thus has an extra Stats attribute.
        header['mseed'] = {}
        header['mseed']['dataquality'] = old_header['dataquality']
        # Convert times to obspy.UTCDateTime objects.
        header['starttime'] = \
            __libmseed__._convertMSTimeToDatetime(header['starttime'])
        header['endtime'] = \
            __libmseed__._convertMSTimeToDatetime(header['endtime'])
        # Append traces.
        if headonly:
            header['npts'] = int((header['endtime'] - header['starttime']) *
                                   header['sampling_rate'] + 1 + 0.5)
            traces.append(Trace(header=header))
        else:
            traces.append(Trace(header=header, data=_i[1]))
    return Stream(traces=traces)


def writeMSEED(stream_object, filename, encoding=None, **kwargs):
    """
    Write Mini-SEED file from a Stream object.
    
    All kwargs are passed directly to obspy.mseed.writeMSTraces.
    
    :param stream_object: obspy.Stream object. Data in stream object must
        be of type int32. NOTE: They are automatically adapted if necessary
    :param filename: Name of the output file
    :param reclen: should be set to the desired data record length in bytes
        which must be expressible as 2 raised to the power of X where X is
        between (and including) 8 to 20. -1 defaults to 4096
    :type encoding: Integer or String
    :param encoding: should be set to one of the following supported
        Mini-SEED data encoding formats: ASCII (0)*, INT16 (1), INT32 (3), 
        FLOAT32 (4)*, FLOAT64 (5)*, STEIM1 (10) and STEIM2 (11)*. Default 
        data types a marked with an asterisk.
    :param byteorder: must be either 0 (LSBF or little-endian) or 1 (MBF or 
        big-endian). -1 defaults to big-endian (1)
    :param flush: if it is not zero all of the data will be packed into 
        records, otherwise records will only be packed while there are
        enough data samples to completely fill a record.
    :param verbose: controls verbosity, a value of zero will result in no 
        diagnostic output.
    """
    # Check if encoding kwarg is set and catch invalid encodings.
    # XXX: Currently INT24 is not working due to lacking numpy support.
    encoding_strings = dict([(v[0], k) for (k, v) in ENCODINGS.iteritems()])
    if not encoding:
        encoding = -1
    elif isinstance(encoding, int) and encoding in ENCODINGS:
        encoding = encoding
    elif isinstance(encoding, basestring) and encoding in encoding_strings:
        encoding = encoding_strings[encoding]
    else:
        msg = 'Invalid encoding %s. Valid encodings: %s'
        raise ValueError(msg % (encoding, encoding_strings))
    # Catch invalid record length.
    valid_record_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                            65536, 131072, 262144, 524288, 1048576]
    if 'reclen' in kwargs.keys() and \
                not kwargs['reclen'] in valid_record_lengths:
        msg = 'Invalid record length. The record length must be a value\n' + \
              'of 2 to the power of X where 8 <= X <= 20.'
        raise ValueError(msg)
    # libmseed instance.
    __libmseed__ = LibMSEED()
    traces = stream_object.traces
    trace_list = []
    convert_dict = {'station': 'station', 'samprate':'sampling_rate',
                    'numsamples': 'npts', 'network': 'network',
                    'location': 'location', 'channel': 'channel',
                    'starttime': 'starttime', 'endtime': 'endtime'}
    for trace in traces:
        header = {}
        for _j, _k in convert_dict.iteritems():
            header[_j] = trace.stats[_k]
        # Set data quality to indeterminate (= D) if it is not already set.
        try:
            header['dataquality'] = trace.stats['mseed']['dataquality']
        except:
            header['dataquality'] = 'D'
        # Convert UTCDateTime times to Mini-SEED times.
        header['starttime'] = \
            __libmseed__._convertDatetimeToMSTime(header['starttime'])
        header['endtime'] = \
            __libmseed__._convertDatetimeToMSTime(header['endtime'])
        # Check that data are numpy.ndarrays
        if not isinstance(trace.data, np.ndarray):
            msg = "Unsupported data type %s" % type(trace.data)
            raise Exception(msg)
        # autodetect format if no global encoding is given
        if encoding == -1:
            if trace.data.dtype.type == np.int32:
                enc = 11
            elif trace.data.dtype.type == np.float32:
                enc = 4
            elif trace.data.dtype.type == np.float64:
                enc = 5
            elif trace.data.dtype.type == np.int16:
                enc = 1
            elif trace.data.dtype.type == np.dtype('|S1').type:
                enc = 0
            else:
                msg = "Unsupported data type %s" % trace.data.dtype
                raise Exception(msg)
            _, sampletype, _ = ENCODINGS[enc]
        else:
            # global encoding given
            enc = encoding
            id, sampletype, dtype = ENCODINGS[enc]
            # Check if supported data type
            if trace.data.dtype.type != dtype:
                msg = "Data type for encoding %s must be of %s" % (id, dtype)
                raise Exception(msg)
        # INT16 needs INT32 data type
        if enc == 1:
            trace.data = trace.data.astype(np.int32)
        header['sampletype'] = sampletype
        header['encoding'] = enc
        # Fill the samplecnt attribute.
        header['samplecnt'] = len(trace.data)
        trace_list.append([header, trace.data])
    # Write resulting trace_list to Mini-SEED file.
    __libmseed__.writeMSTraces(trace_list, outfile=filename, **kwargs)
