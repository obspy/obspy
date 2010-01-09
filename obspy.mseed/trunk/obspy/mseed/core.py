# -*- coding: utf-8 -*-

from obspy.core import Stream, Trace
from obspy.mseed import libmseed
from obspy.mseed.headers import ENCODINGS
import sys


def isMSEED(filename):
    """
    Returns true if the file is a Mini-SEED file and false otherwise.
    
    :param filename: File to be read.
    """
    __libmseed__ = libmseed()
    return __libmseed__.isMSEED(filename)


def readMSEED(filename, headonly=False, starttime=None, endtime=None,
              **kwargs):
    """
    Reads a given Mini-SEED file and returns an obspy.Stream object.
    
    :param filename: Mini-SEED file to be read.
    """
    __libmseed__ = libmseed()
    # read MiniSEED file
    if headonly:
        trace_list = __libmseed__.readMSHeader(filename)
    else:
        #if True: #Uncomment to emulate windows behaviour
        if starttime or endtime or sys.platform == 'win32':
            trace_list = __libmseed__.readMSTracesViaRecords(filename,
                    starttime=starttime, endtime=endtime)
        else:
            #10% faster, problem on windows
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


def writeMSEED(stream_object, filename, encoding= -1, **kwargs):
    """
    Write Miniseed file from a Stream objext.
    
    All kwargs are passed directly to obspy.mseed.writeMSTraces.
    
    :param stream_object: obspy.Stream object. Data in stream object must
        be of type int32. NOTE: They are automatically adapted if necessary
    :param filename: Name of the output file
    :param reclen: should be set to the desired data record length in bytes
        which must be expressible as 2 raised to the power of X where X is
        between (and including) 8 to 20. -1 defaults to 4096
    :type encoding: Integer
    :param encoding: should be set to one of the following supported
        Mini-SEED data encoding formats: ASCII (0), INT16 (1) INT32 (3), 
        FLOAT32 (4), FLOAT64 (5), STEIM1 (10) and STEIM2 (11). Defaults 
        to STEIM2 (11)
    :param byteorder: must be either 0 (LSBF or little-endian) or 1 (MBF or 
        big-endian). -1 defaults to big-endian (1)
    :param flush: if it is not zero all of the data will be packed into 
        records, otherwise records will only be packed while there are
        enough data samples to completely fill a record.
    :param verbose: controls verbosity, a value of zero will result in no 
        diagnostic output.
    """
    # Check if encoding kwarg is set and catch invalid encodings.
    #XXX: Currently INT16 and INT24 are not working. INT24 due to lacking
    #     numpy support, INT16 due to conversion problems from signed and
    #     unsigned integers {1: "i"}
    valid_encodings = dict([(item[0], key) \
                            for (key, item) in ENCODINGS.iteritems()])
    if encoding == -1:
        encoding = 11
    if isinstance(encoding, int) and encoding in ENCODINGS:
        encoding = encoding
    elif isinstance(encoding, basestring) and encoding in valid_encodings:
        encoding = valid_encodings[encoding]
    else:
        msg = 'Invalid encoding %s. Valid encodings: %s'
        raise ValueError(msg % (encoding, valid_encodings))
    kwargs['encoding'] = encoding
    # Catch invalid record length.
    valid_record_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                            65536, 131072, 262144, 524288, 1048576]
    if 'reclen' in kwargs.keys() and \
                not kwargs['reclen'] in valid_record_lengths:
        msg = 'Invalid record length. The record length must be a value\n' + \
              'of 2 to the power of X where 8 <= X <= 20.'
        raise ValueError(msg)
    # libmseed instance.
    __libmseed__ = libmseed()
    traces = stream_object.traces
    trace_list = []
    convert_dict = {'station': 'station', 'samprate':'sampling_rate',
                'numsamples': 'npts', 'network': 'network',
                'location': 'location', 'channel': 'channel',
                'starttime' : 'starttime', 'endtime' : 'endtime'}
    for _i in traces:
        header = {}
        for _j, _k in convert_dict.iteritems():
            header[_j] = _i.stats[_k]
        # Dataquality is extra.
        # Set Dataquality to indeterminate (= D) if it is not already set.
        try:
            header['dataquality'] = _i.stats['mseed']['dataquality']
        except:
            header['dataquality'] = 'D'
        # Convert obspy.UTCDateTime times to Mini-SEED times.
        header['starttime'] = \
            __libmseed__._convertDatetimeToMSTime(header['starttime'])
        header['endtime'] = \
            __libmseed__._convertDatetimeToMSTime(header['endtime'])
        header['sampletype'] = ENCODINGS[encoding][1][0]
        # Fill the samplecnt attribute.
        header['samplecnt'] = len(_i.data)
        trace_list.append([header, _i.data])
    # Write resulting trace_list to Mini-SEED file.
    __libmseed__.writeMSTraces(trace_list, outfile=filename, **kwargs)
