# -*- coding: utf-8 -*-
"""
MSEED bindings to ObsPy core module.
"""

from headers import clibmseed, ENCODINGS, HPTMODULUS, SAMPLETYPE, DATATYPES, \
    VALID_RECORD_LENGTHS, HPTERROR, SelectTime, Selections, blkt_1001_s, \
    VALID_CONTROL_HEADERS, SEED_CONTROL_HEADERS
from itertools import izip
from obspy import Stream, Trace, UTCDateTime
from obspy.core.util import NATIVE_BYTEORDER
from obspy.mseed.headers import blkt_100_s
import ctypes as C
import numpy as np
import os
import util
import warnings


class InternalMSEEDReadingError(Exception):
    pass


class InternalMSEEDReadingWarning(UserWarning):
    pass


def isMSEED(filename):
    """
    Checks whether a file is Mini-SEED/full SEED or not.

    :type filename: string
    :param filename: Mini-SEED/full SEED file to be checked.
    :rtype: bool
    :return: ``True`` if a Mini-SEED file.

    This method only reads the first seven bytes of the file and checks
    whether its a Mini-SEED or full SEED file.

    It also is true for fullSEED files because libmseed can read the data
    part of fullSEED files. If the method finds a fullSEED file it also
    checks if it has a data part and returns False otherwise.

    Thus it cannot be used to validate a Mini-SEED or SEED file.
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


def readMSEED(mseed_object, starttime=None, endtime=None, headonly=False,
              sourcename=None, reclen=None, recinfo=True, details=False,
              header_byteorder=None, verbose=None, **kwargs):
    """
    Reads a Mini-SEED file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param mseed_object: Filename or open file like object that contains the
        binary Mini-SEED data. Any object that provides a read() method will be
        considered to be a file like object.
    :type starttime: UTCDateTime
    :param starttime: Only read data samples after or at the starttime.
    :type endtime: UTCDateTime
    :param endtime: Only read data samples before or at the starttime.
    :param headonly: Determines whether or not to unpack the data or just
        read the headers.
    :type sourcename: str
    :param sourcename: Sourcename has to have the structure
        'network.station.location.channel' and can contain globbing characters.
        Defaults to ``None``.
    :param reclen: If it is None, it will be automatically determined for every
        record. If it is known, just set it to the record length in bytes which
        will increase the reading speed slightly.
    :type recinfo: bool, optional
    :param recinfo: If ``True`` the byteorder, record length and the
        encoding of the file will be read and stored in every Trace's
        stats.mseed AttribDict. These stored attributes will also be used while
        writing a Mini-SEED file. Only the very first record of the file will
        be read and all following records are assumed to be the same. Defaults
        to ``True``.
    :type details: bool, optional
    :param details: If ``True`` read additional information: timing quality
        and availability of calibration information.
        Note, that the traces are then also split on these additional
        information. Thus the number of traces in a stream will change.
        Details are stored in the mseed stats AttribDict of each trace.
        -1 specifies for both cases, that these information is not available.
        ``timing_quality`` specifies the timing quality from 0 to 100 [%].
        ``calibration_type`` specifies the type of available calibration
        information: 1 == Step Calibration, 2 == Sine Calibration, 3 ==
        Pseudo-random Calibration, 4 == Generic Calibration and -2 ==
        Calibration Abort.
    :type header_byteorder: [``0`` or ``'<'`` | ``1`` or ``'>'`` | ``'='``],
        optional
    :param header_byteorder: Must be either ``0`` or ``'<'`` for LSBF or
        little-endian, ``1`` or ``'>'`` for MBF or big-endian. ``'='`` is the
        native byteorder. Used to enforce the header byteorder. Useful in some
        rare cases where the automatic byte order detection fails.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/two_channels.mseed")
    >>> print(st)  # doctest: +ELLIPSIS
    2 Trace(s) in Stream:
    BW.UH3..EHE | 2010-06-20T00:00:00.279999Z - ... | 200.0 Hz, 386 samples
    BW.UH3..EHZ | 2010-06-20T00:00:00.279999Z - ... | 200.0 Hz, 386 samples

    >>> from obspy import UTCDateTime
    >>> st = read("/path/to/test.mseed",
    ...           starttime=UTCDateTime("2003-05-29T02:16:00"),
    ...           selection="NL.*.*.?HZ")
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    NL.HGN.00.BHZ | 2003-05-29T02:15:59.993400Z - ... | 40.0 Hz, 5629 samples
    """
    # Parse the headonly and reclen flags.
    if headonly is True:
        unpack_data = 0
    else:
        unpack_data = 1
    if reclen is None:
        reclen = -1
    elif reclen not in VALID_RECORD_LENGTHS:
        msg = 'Invalid record length. Autodetection will be used.'
        warnings.warn(msg)
        reclen = -1

    # Determine the byteorder.
    if header_byteorder == "=":
        header_byteorder = NATIVE_BYTEORDER

    if header_byteorder is None:
        header_byteorder = -1
    elif header_byteorder in [0, "0", "<"]:
        header_byteorder = 0
    elif header_byteorder in [1, "1", ">"]:
        header_byteorder = 1

    # The quality flag is no more supported. Raise a warning.
    if 'quality' in kwargs:
        msg = 'The quality flag is no more supported in this version of ' + \
            'obspy.mseed. obspy.mseed.util has some functions with similar' + \
            ' behaviour.'
        warnings.warn(msg, category=DeprecationWarning)

    # Parse some information about the file.
    if recinfo:
        # Pass the byteorder if enforced.
        if header_byteorder == 0:
            bo = "<"
        elif header_byteorder > 0:
            bo = ">"
        else:
            bo = None

        info = util.getRecordInformation(mseed_object, endian=bo)
        info['encoding'] = ENCODINGS[info['encoding']][0]
        # Only keep information relevant for the whole file.
        info = {'encoding': info['encoding'],
                'filesize': info['filesize'],
                'record_length': info['record_length'],
                'byteorder': info['byteorder'],
                'number_of_records': info['number_of_records']}

    # If its a filename just read it.
    if isinstance(mseed_object, basestring):
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
            if not isinstance(starttime, UTCDateTime):
                msg = 'starttime needs to be a UTCDateTime object'
                raise ValueError(msg)
            selections.timewindows.contents.starttime = \
                util._convertDatetimeToMSTime(starttime)
        else:
            # HPTERROR results in no starttime.
            selections.timewindows.contents.starttime = HPTERROR
        if endtime is not None:
            if not isinstance(endtime, UTCDateTime):
                msg = 'endtime needs to be a UTCDateTime object'
                raise ValueError(msg)
            selections.timewindows.contents.endtime = \
                util._convertDatetimeToMSTime(endtime)
        else:
            # HPTERROR results in no starttime.
            selections.timewindows.contents.endtime = HPTERROR
        if sourcename is not None:
            if not isinstance(sourcename, basestring):
                msg = 'sourcename needs to be a string'
                raise ValueError(msg)
            # libmseed uses underscores as separators and allows filtering
            # after the dataquality which is disabled here to not confuse
            # users. (* == all data qualities)
            selections.srcname = sourcename.replace('.', '_') + '_*'
        else:
            selections.srcname = '*'
    all_data = []

    # Use a callback function to allocate the memory and keep track of the
    # data.
    def allocate_data(samplecount, sampletype):
        # Enhanced sanity checking for libmseed 2.10 can result in the
        # sampletype not being set. Just return an empty array in this case.
        if sampletype == "\x00":
            data = np.empty(0)
        else:
            data = np.empty(samplecount, dtype=DATATYPES[sampletype])
        all_data.append(data)
        return data.ctypes.data
    # XXX: Do this properly!
    # Define Python callback function for use in C function. Return a long so
    # it hopefully works on 32 and 64 bit systems.
    allocData = C.CFUNCTYPE(C.c_long, C.c_int, C.c_char)(allocate_data)

    def log_error_or_warning(msg):
        if msg.startswith("ERROR: "):
            raise InternalMSEEDReadingError(msg[7:].strip())
        if msg.startswith("INFO: "):
            warnings.warn(msg[6:].strip(), InternalMSEEDReadingWarning)
    diag_print = C.CFUNCTYPE(C.c_void_p, C.c_char_p)(log_error_or_warning)

    def log_message(msg):
        print msg[6:].strip()
    log_print = C.CFUNCTYPE(C.c_void_p, C.c_char_p)(log_message)

    try:
        verbose = int(verbose)
    except:
        verbose = 0

    lil = clibmseed.readMSEEDBuffer(
        buffer, buflen, selections, C.c_int8(unpack_data),
        reclen, C.c_int8(verbose), C.c_int8(details), header_byteorder,
        allocData, diag_print, log_print)

    # XXX: Check if the freeing works.
    del selections

    traces = []
    try:
        currentID = lil.contents
    # Return stream if not traces are found.
    except ValueError:
        clibmseed.lil_free(lil)
        del lil
        return Stream()

    while True:
        # Init header with the essential information.
        header = {'network': currentID.network.strip(),
                  'station': currentID.station.strip(),
                  'location': currentID.location.strip(),
                  'channel': currentID.channel.strip(),
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
            # TODO: write support is missing
            if details:
                timing_quality = currentSegment.timing_quality
                if timing_quality == 0xFF:  # 0xFF is mask for not known timing
                    timing_quality = -1
                header['mseed']['timing_quality'] = timing_quality
                header['mseed']['calibration_type'] = \
                    currentSegment.calibration_type

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
            if recinfo:
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

    clibmseed.lil_free(lil)  # NOQA
    del lil  # NOQA
    return Stream(traces=traces)


def writeMSEED(stream, filename, encoding=None, reclen=None, byteorder=None,
               flush=1, verbose=0, **_kwargs):
    """
    Write Mini-SEED file from a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: A Stream object.
    :type filename: str
    :param filename: Name of the output file
    :type encoding: int or str, optional
    :param encoding: Should be set to one of the following supported Mini-SEED
        data encoding formats: ASCII (``0``)*, INT16 (``1``), INT32 (``3``),
        FLOAT32 (``4``)*, FLOAT64 (``5``)*, STEIM1 (``10``) and STEIM2
        (``11``)*. Default data types a marked with an asterisk. Currently
        INT24 (``2``) is not supported due to lacking NumPy support.
    :type reclen: int, optional
    :param reclen: Should be set to the desired data record length in bytes
        which must be expressible as 2 raised to the power of X where X is
        between (and including) 8 to 20.
        Defaults to 4096
    :type byteorder: [``0`` or ``'<'`` | ``1`` or ``'>'`` | ``'='``], optional
    :param byteorder: Must be either ``0`` or ``'<'`` for LSBF or
        little-endian, ``1`` or ``'>'`` for MBF or big-endian. ``'='`` is the
        native byteorder. If ``-1`` it will be passed directly to libmseed
        which will also default it to big endian. Defaults to big endian.
    :type flush: int, optional
    :param flush: If it is not zero all of the data will be packed into
        records, otherwise records will only be packed while there are
        enough data samples to completely fill a record.
    :type verbose: int, optional
    :param verbose: Controls verbosity, a value of zero will result in no
        diagnostic output.

    .. note::
        The reclen, encoding and byteorder keyword arguments can be set
        in the stats.mseed of each :class:`~obspy.core.trace.Trace` as well as
        as kwargs of this function. If both are given the kwargs will be used.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read()
    >>> st.write('filename.mseed', format='MSEED')  # doctest: +SKIP
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

        # Determine if a blockette 100 will be needed to represent the input
        # sample rate or if the sample rate in the fixed section of the data
        # header will suffice (see ms_genfactmult in libmseed/genutils.c)
        if trace.stats.sampling_rate >= 32727.0 or \
           trace.stats.sampling_rate <= (1.0 / 32727.0):
            use_blkt_100 = True
        else:
            use_blkt_100 = False

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
        trace_attr['encoding'] = None
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
                trace_attr['encoding'] = None
        # automatically detect encoding if no encoding is given.
        if not trace_attr['encoding']:
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
        if not len(data):
            msg = 'Skipping empty trace "%s".' % (trace)
            warnings.warn(msg)
            continue
        # Create C struct MSTrace.
        mst = MST(trace, data, dataquality=trace_attr['dataquality'])

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
                clibmseed.msr_free(C.pointer(msr))
                del msr
                raise Exception('Error in msr_addblockette')
        # Only use Blockette 100 if necessary.
        if use_blkt_100:
            size = C.sizeof(blkt_100_s)
            blkt100 = C.c_char(' ')
            C.memset(C.pointer(blkt100), 0, size)
            ret_val = clibmseed.msr_addblockette(
                msr, C.pointer(blkt100), size, 100, 0)  # NOQA
            # Usually returns a pointer to the added blockette in the
            # blockette link chain and a NULL pointer if it fails.
            # NULL pointers have a false boolean value according to the
            # ctypes manual.
            if bool(ret_val) is False:
                clibmseed.msr_free(C.pointer(msr))  # NOQA
                del msr  # NOQA
                raise Exception('Error in msr_addblockette')

        # Pack mstg into a MSEED file using the callback record_handler as
        # write method.
        errcode = clibmseed.mst_pack(
            mst.mst, recHandler, None, trace_attr['reclen'],
            trace_attr['encoding'], trace_attr['byteorder'],
            C.byref(packedsamples), flush, verbose, msr)  # NOQA

        if errcode == 0:
            msg = ("Did not write any data for trace '%s' even though it "
                   "contains data values.") % trace
            raise ValueError(msg)
        if errcode == -1:
            clibmseed.msr_free(C.pointer(msr))  # NOQA
            del mst, msr  # NOQA
            raise Exception('Error in mst_pack')
        # Deallocate any allocated memory.
        clibmseed.msr_free(C.pointer(msr))  # NOQA
        del mst, msr  # NOQA
    # Close if its a file handler.
    if isinstance(f, file):
        f.close()


class MST(object):
    """
    Class that transforms a ObsPy Trace object to a libmseed internal MSTrace
    struct.
    """
    def __init__(self, trace, data, dataquality):
        """
        The init function requires a ObsPy Trace object which will be used to
        fill self.mstg.
        """
        self.mst = clibmseed.mst_init(None)
        # Figure out the datatypes.
        sampletype = SAMPLETYPE[data.dtype.type]

        # Set the header values.
        self.mst.contents.network = trace.stats.network
        self.mst.contents.station = trace.stats.station
        self.mst.contents.location = trace.stats.location
        self.mst.contents.channel = trace.stats.channel
        self.mst.contents.dataquality = dataquality
        self.mst.contents.type = '\x00'
        self.mst.contents.starttime = \
            util._convertDatetimeToMSTime(trace.stats.starttime)
        self.mst.contents.endtime = \
            util._convertDatetimeToMSTime(trace.stats.endtime)
        self.mst.contents.samprate = trace.stats.sampling_rate
        self.mst.contents.samplecnt = trace.stats.npts
        self.mst.contents.numsamples = trace.stats.npts
        self.mst.contents.sampletype = sampletype

        # libmseed expects data in the native byteorder.
        if data.dtype.byteorder != "=":
            data = data.byteswap()

        # Copy the data. The copy appears to be necessary so that Python's
        # garbage collection does not interfere it.
        bytecount = data.itemsize * data.size

        self.mst.contents.datasamples = clibmseed.allocate_bytes(bytecount)
        C.memmove(self.mst.contents.datasamples, data.ctypes.get_data(),
                  bytecount)

    def __del__(self):
        """
        Frees all allocated memory.
        """
        # This also frees the data of the associated datasamples pointer.
        clibmseed.mst_free(C.pointer(self.mst))
        del self.mst


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
