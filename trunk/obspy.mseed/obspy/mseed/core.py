# -*- coding: utf-8 -*-
"""
MSEED bindings to ObsPy core module.
"""

from headers import clibmseed, ENCODINGS, HPTMODULUS, SAMPLETYPE, DATATYPES, \
    SAMPLESIZES, VALID_RECORD_LENGTHS, HPTERROR, SelectTime, Selections, \
    blkt_1001_s, VALID_CONTROL_HEADERS, SEED_CONTROL_HEADERS
from itertools import izip
from math import log
from obspy.core import Stream, Trace, UTCDateTime
from obspy.core.util import NATIVE_BYTEORDER
import ctypes as C
import numpy as np
import os
import util
import warnings


def isMSEED(filename):
    """
    Checks whether a file is Mini-SEED/full SEED or not.

    :type filename: string
    :param filename: Mini-SEED/full SEED file to be checked.
    :rtype: bool
    :return: ``True`` if a Mini-SEED file.

    This method only reads the first seven bytes of the file and checks
    whether its a MiniSEED or fullSEED file.

    It also is true for fullSEED files because libmseed can read the data
    part of fullSEED files. If the method finds a fullSEED file it also
    checks if it has a data part and returns False otherwise.

    Thus it cannot be used to validate a MiniSEED or SEED file.
    """
    fp = open(filename, 'rb')
    header = fp.read(7)
    # File has less than 7 characters
    if len(header) != 7:
        return False
    # Sequence number must contains a single number or be empty
    seqnr = header[0:6].replace('\x00', ' ').strip()
    if not seqnr.isdigit() and seqnr != '':
        return False
    # Check for any valid control header types.
    if header[6] in ['D', 'R', 'Q', 'M']:
        return True
    # Check if Full-SEED
    if not header[6] == 'V':
        return False
    # Parse the whole file and check whether it has has a data record.
    fp.seek(1, 1)
    _i = 0
    # search for blockettes 010 or 008
    while True:
        if fp.read(3) in ['010', '008']:
            break
        # the next for bytes are the record length
        # as we are currently at position 7 (fp.read(3) fp.read(4))
        # we need to subtract this first before we seek
        # to the appropriate position
        try:
            fp.seek(int(fp.read(4)) - 7, 1)
        except:
            return False
        _i += 1
        # break after 3 cycles
        if _i == 3:
            return False
    # Try to get a record length.
    fp.seek(8, 1)
    try:
        record_length = pow(2, int(fp.read(2)))
    except:
        return False
    file_size = os.path.getsize(filename)
    # Jump to the second record.
    fp.seek(record_length + 6)
    # Loop over all records and return True if one record is a data
    # record
    while fp.tell() < file_size:
        flag = fp.read(1)
        if flag in ['D', 'R', 'Q', 'M']:
            return True
        fp.seek(record_length - 1, 1)
    return False


def readMSEED(mseed_object, starttime=None, endtime=None, sourcename=None,
              readMSInfo=True, headonly=False, reclen=None, **kwargs):
    """
    Reads a Mini-SEED file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param mseed_object: Filename or open file like object that contains the
        binary MiniSEED data. Any object that provides a read() method will be
        considered to be a file like object.
    :param starttime: UTCDateTime object
        Only read data samples after or at the starttime.
    :param endtime: UTCDateTime object
        Only read data samples before or at the starttime.
    :param sourcename: String
        sourcename has to have the structure 'network.station.location.channel'
        and can contain globbing characters
        Defaults to None
    readMSInfo : bool, optional
        If True the byteorder, record length and the encoding of the file will
        be read and stored in every Trace's stats.mseed AttribDict. These
        stored attributes will also be used while writing a MiniSEED file. Only
        the very first record of the file will be read and all following
        records are assumed to be the same. Defaults to True.
    :param headonly: Determines whether or not to unpack the data or just
        read the headers.
    :param reclen: If it is None, it will be automatically determined for every
        record. If it is known, just set it to the record length in bytes which
        will increase the reading speed slightly.

    .. rubric:: Example

    >>> from obspy.core import read
    >>> st = read("/path/to/test.mseed")

    The following example will read all EHZ channels from the BW network from
    the binary data in mseed_data. Only the first hour of 2010 will be read.

    >> from cStringIO import StringIO
    >> f = StringIO(mseed_data)
    >> selection = {'starttime': UTCDateTime(2010, 1, 1, 0, 0, 0),
                    'endtime': UTCDateTime(2010, 1, 1, 1, 0, 0),
                    'sourcename': 'BW.*.*.EHZ'}
    >> st = readMSEED(f, selection)
    """
    # Parse the headonly and reclen flags.
    if headonly is True:
        unpack_data = 0
    else:
        unpack_data = 1
    if reclen is None:
        reclen = -1
    elif reclen is not None and reclen not in VALID_RECORD_LENGTHS:
        msg = 'Invalid record length. Autodetection will be used.'
        warnings.warn(msg)
        reclen = -1
    else:
        reclen = int(log(reclen, 2))

    # The quality flag is no more supported. Raise a warning.
    if 'quality' in kwargs:
        msg = 'The quality flag is no more supported in this version of ' + \
        'obspy.mseed. obspy.mseed.util has some functions with similar ' + \
        'behaviour.'
        warnings.warn(msg, category=DeprecationWarning)

    # XXX: Make work with StringIO, ...
    # Read fileformat information if necessary.
    if readMSInfo:
        if type(mseed_object) is str:
            info = util.getFileformatInformation(mseed_object)
            # Better readability.
            if 'reclen' in info:
                info['record_length'] = info['reclen']
                del info['reclen']
            if 'encoding' in info:
                info['encoding'] = ENCODINGS[info['encoding']][0]
            if 'byteorder' in info:
                if info['byteorder'] == 1:
                    info['byteorder'] = '>'
                else:
                    info['byteorder'] = '<'
        else:
            msg = 'readMSInfo currently only enabled for real files.'
            warnings.warn(msg)
            readMSInfo = False

    # If its a filename just read it.
    if isinstance(mseed_object,  basestring):
        # Read to NumPy array which is used as a buffer.
        buffer = np.fromfile(mseed_object, dtype='b')
    elif hasattr(mseed_object, 'read'):
        buffer = np.fromstring(mseed_object.read(), dtype='b')

    # Get the record length
    try:
        record_length = pow(2, int(''.join([chr(_i) for _i in buffer[19:21]])))
    except ValueError:
        record_length = 4096

    # Search for data records and pass only the data part to the underlying C
    # routine.
    offset = 0
    # 0 to 9 are defined in a row in the ASCII charset.
    min_ascii = ord('0')
    # Small function to check whether an array of ASCII values contains only
    # digits.
    isdigit = lambda x: True if (x - min_ascii).max() <= 9 else False
    while True:
        # This should never happen
        if (isdigit(buffer[offset:offset + 6]) is False) or \
            (buffer[offset + 6] not in VALID_CONTROL_HEADERS):
            msg = 'Not a valid (Mini-)SEED file'
            raise Exception(msg)
        elif buffer[offset + 6] in SEED_CONTROL_HEADERS:
            offset += record_length
            continue
        break
    buffer = buffer[offset:]
    buflen = len(buffer)

    # If no selection is given pass None to the C function.
    if starttime is None and endtime is None and sourcename is None:
        selections = None
    else:
        select_time = SelectTime()
        selections = Selections()
        selections.timewindows.contents = select_time
        if starttime is not None:
            if type(starttime) != UTCDateTime:
                msg = 'starttime needs to be a UTCDateTime object'
                raise ValueError(msg)
            selections.timewindows.contents.starttime = \
                util._convertDatetimeToMSTime(starttime)
        else:
            # HPTERROR results in no starttime.
            selections.timewindows.contents.starttime = HPTERROR
        if endtime is not None:
            if type(endtime) != UTCDateTime:
                msg = 'endtime needs to be a UTCDateTime object'
                raise ValueError(msg)
            selections.timewindows.contents.endtime = \
                util._convertDatetimeToMSTime(endtime)
        else:
            # HPTERROR results in no starttime.
            selections.timewindows.contents.endtime = HPTERROR
        if sourcename is not None:
            if type(sourcename) != str:
                msg = 'sourcename needs to be a string'
                raise ValueError(msg)
            # libmseed uses underscores as separators and allows filtering
            # after the dataquality which is disabled here to not confuse
            # users.
            selections.srcname = sourcename.replace('.', '_') + '_D'
        else:
            selections.srcname = '*'

    all_data = []

    # Use a callback function to allocate the memory and keep track of the
    # data.
    def allocate_data(samplecount, sampletype):
        data = np.empty(samplecount, dtype=DATATYPES[sampletype])
        all_data.append(data)
        return data.ctypes.data
    # XXX: Do this properly!
    # Define Python callback function for use in C function. Return a long so
    # it hopefully works on 32 and 64 bit systems.
    allocData = C.CFUNCTYPE(C.c_long, C.c_int, C.c_char)(allocate_data)

    lil = clibmseed.readMSEEDBuffer(buffer, buflen, selections, unpack_data,
                              reclen, 0, allocData)

    # XXX: Check if the freeing works.
    del selections

    traces = []
    try:
        currentID = lil.contents
    # Return stream if not traces are found.
    except ValueError:
        return Stream()

    while True:
        # Init header with the essential information.
        header = {'network': currentID.network,
                  'station': currentID.station,
                  'location': currentID.location,
                  'channel': currentID.channel,
                  'mseed': {'dataquality': currentID.dataquality}}
        # Loop over segments.
        try:
            currentSegment = currentID.firstSegment.contents
        except ValueError:
            break
        while True:
            header['sampling_rate'] = currentSegment.samprate
            header['starttime'] = \
                util._convertMSTimeToDatetime(currentSegment.starttime)
            if headonly is False:
                # The data always will be in sequential order.
                data = all_data.pop(0)
                header['npts'] = len(data)
            else:
                data = np.array([])
                header['npts'] = currentSegment.samplecnt
            # Make sure to init the number of samples.
            trace = Trace(header=header, data=data)
            # Append information if necessary.
            if readMSInfo:
                for key, value in info.iteritems():
                    setattr(trace.stats.mseed, key, value)
            traces.append(trace)
            # A Null pointer access results in a ValueError
            try:
                currentSegment = currentSegment.next.contents
            except ValueError:
                break
        try:
            currentID = currentID.next.contents
        except ValueError:
            break

    clibmseed.lil_free(lil)
    del lil
    return Stream(traces=traces)


def writeMSEED(stream, filename, encoding=None, reclen=None, byteorder=None,
               flush=1, verbose=0, **kwargs):
    """
    Write Mini-SEED file from a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an

    stream_object : :class:`~obspy.core.stream.Stream`
        A Stream object. Data in stream object must be of type int32.
        NOTE: They are automatically adapted if necessary
    filename : string
        Name of the output file
    encoding : int or string, optional
        Should be set to one of the following supported Mini-SEED data encoding
        formats: ASCII (0)*, INT16 (1), INT32 (3), FLOAT32 (4)*, FLOAT64 (5)*,
        STEIM1 (10) and STEIM2 (11)*. Default data types a marked with an
        asterisk. Currently INT24 (2) is not supported due to lacking NumPy
        support.
    reclen : int, optional
        Should be set to the desired data record length in bytes
        which must be expressible as 2 raised to the power of X where X is
        between (and including) 8 to 20.
        Defaults to 4096
    byteorder : [ 0 or '<' | '1 or '>' | '='], optional
        Must be either 0 or '<' for LSBF or little-endian, 1 or '>' for MBF or
        big-endian. '=' is the native byteorder. If -1 it will be passed
        directly to libmseed which will also default it to big endian.
        Defaults to big endian.
    flush : int, optional
        If it is not zero all of the data will be packed into
        records, otherwise records will only be packed while there are
        enough data samples to completely fill a record.
    verbose : int, optional
        Controls verbosity, a value of zero will result in no diagnostic
        output.

    .. note::
        The reclen, encoding and byteorder keyword arguments can be set
        in the trace.stats.mseed AttribDict as well as as kwargs. If both are
        set the kwargs will be used.

    .. rubric:: Example

    >>> from obspy.core import read
    >>> st = read()
    >>> st.write('filename.mseed', format='MSEED') #doctest: +SKIP
    """
    # Some sanity checks for the keyword arguments.
    if reclen is not None and reclen not in VALID_RECORD_LENGTHS:
        msg = 'Invalid record length. The record length must be a value\n' + \
                'of 2 to the power of X where 8 <= X <= 20.'
        raise ValueError(msg)
    if byteorder is not None and byteorder not in [0, 1, -1]:
        if byteorder == '=':
            byteorder = NATIVE_BYTEORDER
        # If not elif because NATIVE_BYTEORDER is '<' or '>'.
        if byteorder == '<':
            byteorder = 0
        elif byteorder == '>':
            byteorder = 1
        else:
            msg = "Invalid byteorder. It must be either '<', '>', '=', " + \
                  "0, 1 or -1"
            raise ValueError(msg)

    # Check if encoding kwarg is set and catch invalid encodings.
    # XXX: Currently INT24 is not working due to lacking NumPy support.
    encoding_strings = dict([(v[0], k) for (k, v) in ENCODINGS.iteritems()])

    if encoding is not None:
        if isinstance(encoding, int) and encoding in ENCODINGS:
            pass
        elif encoding and isinstance(encoding, basestring) and encoding \
                in encoding_strings:
            encoding = encoding_strings[encoding]
        else:
            msg = 'Invalid encoding %s. Valid encodings: %s'
            raise ValueError(msg % (encoding, encoding_strings))

    trace_attributes = []
    use_blkt_1001 = 0
    # The data might need to be modified. To not modify the input data keep
    # references of which data to finally write.
    trace_data = []
    # Loop over every trace and figure out the correct settings.
    for _i, trace in enumerate(stream):
        # Create temporary dict for storing information while writing.
        trace_attr = {}
        trace_attributes.append(trace_attr)
        stats = trace.stats

        # Figure out whether or not to use Blockette 1001. This check is done
        # once to ensure that Blockette 1001 is either written for every record
        # in the file or for none. It checks the starttime as well as the
        # sampling rate. If either one has a precision of more than 100
        # microseconds, Blockette 1001 will be written for every record.
        starttime = util._convertDatetimeToMSTime(trace.stats.starttime)
        if starttime % 100 != 0 or \
           (1.0 / trace.stats.sampling_rate * HPTMODULUS) % 100 != 0:
            use_blkt_1001 += 1

        # Set data quality to indeterminate (= D) if it is not already set.
        try:
            trace_attr['dataquality'] = \
                trace.stats['mseed']['dataquality'].upper()
        except:
            trace_attr['dataquality'] = 'D'
        # Sanity check for the dataquality to get a nice Python exception
        # instead of a C error.
        if trace_attr['dataquality'] not in ['D', 'R', 'Q', 'M']:
            msg = 'Invalid dataquality in Stream[%i].stats' % _i + \
                  '.mseed.dataquality\n' + \
                  'The dataquality for Mini-SEED must be either D, R, Q ' + \
                  'or M. See the SEED manual for further information.'
            raise ValueError(msg)

        # Check that data is of the right type.
        if not isinstance(trace.data, np.ndarray):
            msg = "Unsupported data type %s" % type(trace.data) + \
                  " for Stream[%i].data." % _i
            raise ValueError(msg)

        # Check if ndarray is contiguous (see #192, #193)
        if not trace.data.flags.c_contiguous:
            msg = "Detected non contiguous data array in Stream[%i]" % _i + \
                  ".data. Trying to fix array."
            warnings.warn(msg)
            trace.data = np.require(trace.data, requirements=('C_CONTIGUOUS',))

        # Handle the record length.
        if reclen is not None:
            trace_attr['reclen'] = reclen
        elif hasattr(stats, 'mseed') and \
                hasattr(stats.mseed, 'record_length'):
            if stats.mseed.record_length in VALID_RECORD_LENGTHS:
                trace_attr['reclen'] = stats.mseed.record_length
            else:
                msg = 'Invalid record length in Stream[%i].stats.' % _i + \
                      'mseed.reclen.\nThe record length must be a value ' + \
                      'of 2 to the power of X where 8 <= X <= 20.'
                raise ValueError(msg)
        else:
            trace_attr['reclen'] = 4096

        # Handle the byteorder.
        if byteorder is not None:
            trace_attr['byteorder'] = byteorder
        elif hasattr(stats, 'mseed') and \
                hasattr(stats.mseed, 'byteorder'):
            if stats.mseed.byteorder in [0, 1, -1]:
                trace_attr['byteorder'] = stats.mseed.byteorder
            elif stats.mseed.byteorder == '=':
                if NATIVE_BYTEORDER == '<':
                    trace_attr['byteorder'] = 0
                else:
                    trace_attr['byteorder'] = 1
            elif stats.mseed.byteorder == '<':
                trace_attr['byteorder'] = 0
            elif stats.mseed.byteorder == '>':
                trace_attr['byteorder'] = 1
            else:
                msg = "Invalid byteorder in Stream[%i].stats." % _i + \
                       "mseed.byteorder. It must be either '<', '>', '='," + \
                       " 0, 1 or -1"
                raise ValueError(msg)
        else:
            trace_attr['byteorder'] = 1
        if trace_attr['byteorder'] == -1:
            if NATIVE_BYTEORDER == '<':
                trace_attr['byteorder'] = 0
            else:
                trace_attr['byteorder'] = 1

        # Handle the encoding.
        if encoding is not None:
            # Check if the dtype for all traces is compatible with the enforced
            # encoding.
            id, _, dtype = ENCODINGS[encoding]
            if trace.data.dtype.type != dtype:
                msg = """
                    Wrong dtype for Stream[%i].data for encoding %s.
                    Please change the dtype of your data or use an appropriate
                    encoding. See the obspy.mseed documentation for more
                    information.
                    """ % (_i, id)
                raise Exception(msg)
            trace_attr['encoding'] = encoding
        elif hasattr(trace.stats, 'mseed') and hasattr(trace.stats.mseed,
                                                       'encoding'):
            mseed_encoding = stats.mseed.encoding
            # Check if the encoding is valid.
            if isinstance(mseed_encoding, int) and mseed_encoding in ENCODINGS:
                trace_attr['encoding'] = mseed_encoding
            elif isinstance(mseed_encoding, basestring) and \
                    mseed_encoding in encoding_strings:
                trace_attr['encoding'] = encoding_strings[mseed_encoding]
            else:
                msg = 'Invalid encoding %s in ' + \
                      'Stream[%i].stats.mseed.encoding. Valid encodings: %s'
                raise ValueError(msg % (mseed_encoding, _i, encoding_strings))
            # Check if the encoding matches the data's dtype.
            if trace.data.dtype.type != ENCODINGS[trace_attr['encoding']][2]:
                msg = 'The encoding specified in ' + \
                      'trace.stats.mseed.encoding does not match the ' + \
                      'dtype of the data.\nA suitable encoding will ' + \
                      'be chosen.'
                warnings.warn(msg, UserWarning)
        else:
            # automatically detect encoding if no encoding is given.
            if trace.data.dtype.type == np.dtype("int32"):
                trace_attr['encoding'] = 11
            elif trace.data.dtype.type == np.dtype("float32"):
                trace_attr['encoding'] = 4
            elif trace.data.dtype.type == np.dtype("float64"):
                trace_attr['encoding'] = 5
            elif trace.data.dtype.type == np.dtype("int16"):
                trace_attr['encoding'] = 1
            elif trace.data.dtype.type == np.dtype('|S1').type:
                trace_attr['encoding'] = 0
            else:
                msg = "Unsupported data type %s in Stream[%i].data" % \
                    (trace.data.dtype, _i)
                raise Exception(msg)

        # Convert data if necessary, otherwise store references in list.
        if trace_attr['encoding'] == 1:
            # INT16 needs INT32 data type
            trace_data.append(trace.data.copy().astype(np.int32))
        else:
            trace_data.append(trace.data)

    # Do some final sanity checks and raise a warning if a file will be written
    # with more than one different encoding, record length or byteorder.
    encodings = set([_i['encoding'] for _i in trace_attributes])
    reclens = set([_i['reclen'] for _i in trace_attributes])
    byteorders = set([_i['byteorder'] for _i in trace_attributes])
    msg = 'File will be written with more than one different %s.\n' + \
          'This might have a negative influence on the compatibility ' + \
          'with other programs.'
    if len(encodings) != 1:
        warnings.warn(msg % 'encodings')
    if len(reclens) != 1:
        warnings.warn(msg % 'record lengths')
    if len(byteorders) != 1:
        warnings.warn(msg % 'byteorders')

    # Open filehandler or use an existing file like object.
    if not hasattr(filename, 'write'):
        f = open(filename, 'wb')
    else:
        f = filename

    # Loop over every trace and finally write it to the filehandler.
    for trace, data, trace_attr in izip(stream, trace_data, trace_attributes):
        # Create C struct MSTraceGroup.
        mstg = MSTG(trace, data, dataquality=trace_attr['dataquality'],
                    byteorder=trace_attr['byteorder'])
        # Initialize packedsamples pointer for the mst_pack function
        packedsamples = C.c_int()

        # Callback function for mst_pack to actually write the file
        def record_handler(record, reclen, _stream):
            f.write(record[0:reclen])
        # Define Python callback function for use in C function
        recHandler = C.CFUNCTYPE(C.c_void_p, C.POINTER(C.c_char), C.c_int,
                                 C.c_void_p)(record_handler)

        # Fill up msr record structure, this is already contained in
        # mstg, however if blk1001 is set we need it anyway
        msr = clibmseed.msr_init(None)
        msr.contents.network = trace.stats.network
        msr.contents.station = trace.stats.station
        msr.contents.location = trace.stats.location
        msr.contents.channel = trace.stats.channel
        msr.contents.dataquality = trace_attr['dataquality']

        # Only use Blockette 1001 if necessary.
        if use_blkt_1001:
            size = C.sizeof(blkt_1001_s)
            blkt1001 = C.c_char(' ')
            C.memset(C.pointer(blkt1001), 0, size)
            ret_val = clibmseed.msr_addblockette(msr, C.pointer(blkt1001),
                                       size, 1001, 0)
            # Usually returns a pointer to the added blockette in the
            # blockette link chain and a NULL pointer if it fails.
            # NULL pointers have a false boolean value according to the
            # ctypes manual.
            if bool(ret_val) is False:
                raise Exception('Error in msr_addblockette')

        # Pack mstg into a MSEED file using the callback record_handler as
        # write method.
        errcode = clibmseed.mst_packgroup(mstg.getMstg(), recHandler, None,
                          trace_attr['reclen'], trace_attr['encoding'],
                          trace_attr['byteorder'], C.byref(packedsamples),
                          flush, verbose, msr)
        if errcode == -1:
            raise Exception('Error in mst_packgroup')
        # Deallocating msr is not necessary because no data is allocated. The
        # memory management of mstg is handled by the class.
        del mstg, msr
    # Close if its a file handler.
    if isinstance(f, file):
        f.close()


class MSTG(object):
    """
    Class that transforms a ObsPy Trace object to a libmseed internal
    MSTraceGroup struct.

    The class works on a Trace instead of a Stream because that way it is
    possible to write MiniSEED files with a different encoding per Trace.

    The class is mainly used to achieve a clean memory management.
    """
    def __init__(self, trace, data, dataquality, byteorder):
        """
        The init function requires a ObsPy Trace object which will be used to
        fill self.mstg.
        """
        # Initialize MSTraceGroup
        mstg = clibmseed.mst_initgroup(None)
        self.mstg = mstg
        # Set numtraces.
        mstg.contents.numtraces = 1
        # Initialize MSTrace object and connect with group
        mstg.contents.traces = clibmseed.mst_init(None)
        chain = mstg.contents.traces

        # Figure out the datatypes.
        sampletype = SAMPLETYPE[data.dtype.type]
        c_dtype = DATATYPES[sampletype]

        # Set the header values.
        chain.contents.network = trace.stats.network
        chain.contents.station = trace.stats.station
        chain.contents.location = trace.stats.location
        chain.contents.channel = trace.stats.channel
        chain.contents.dataquality = dataquality
        chain.contents.type = '\x00'
        chain.contents.starttime = \
                util._convertDatetimeToMSTime(trace.stats.starttime)
        chain.contents.endtime = \
                util._convertDatetimeToMSTime(trace.stats.endtime)
        chain.contents.samprate = trace.stats.sampling_rate
        chain.contents.samplecnt = trace.stats.npts
        chain.contents.numsamples = trace.stats.npts
        chain.contents.sampletype = sampletype

        # Create a single datapoint and resize its memory to be able to
        # hold all datapoints.
        tempdatpoint = c_dtype()
        datasize = SAMPLESIZES[sampletype] * trace.stats.npts
        # XXX: Ugly workaround for bug writing ASCII.
        if sampletype == 'a' and datasize < 17:
            datasize = 17
        C.resize(tempdatpoint, datasize)
        # The datapoints in the MSTG structure are a pointer to the memory
        # area reserved for tempdatpoint.
        chain.contents.datasamples = C.cast(C.pointer(tempdatpoint),
                                            C.c_void_p)
        # Swap if wrong byte order because libmseed expects native byteorder.
        if data.dtype.byteorder != "=":
            data = data.byteswap()
        # Pointer to the NumPy data buffer.
        datptr = data.ctypes.get_data()
        # Manually move the contents of the NumPy data buffer to the
        # address of the previously created memory area.
        C.memmove(chain.contents.datasamples, datptr, datasize)

    def __del__(self):
        """
        Frees the MSTraceGroup struct. Therefore Python garbage collection can
        work with this class.
        """
        clibmseed.mst_freegroup(C.pointer(self.mstg))
        del self.mstg

    def getMstg(self):
        """
        Simply returns the mstg.
        """
        return self.mstg


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
