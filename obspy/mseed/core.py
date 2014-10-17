# -*- coding: utf-8 -*-
"""
MSEED bindings to ObsPy core module.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str
from struct import pack

from obspy.mseed.headers import clibmseed, ENCODINGS, HPTMODULUS, \
    SAMPLETYPE, DATATYPES, UNSUPPORTED_ENCODINGS, \
    VALID_RECORD_LENGTHS, HPTERROR, SelectTime, Selections, blkt_1001_s, \
    VALID_CONTROL_HEADERS, SEED_CONTROL_HEADERS, blkt_100_s
from obspy.mseed import util

from obspy import Stream, Trace, UTCDateTime
from obspy.core.util import NATIVE_BYTEORDER
import ctypes as C
import numpy as np
import os
import warnings


class InternalMSEEDReadingError(Exception):
    pass


class InternalMSEEDReadingWarning(UserWarning):
    pass


def isMSEED(filename):
    """
    Checks whether a file is Mini-SEED/full SEED or not.

    :type filename: str
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
    with open(filename, 'rb') as fp:
        header = fp.read(7)
        # File has less than 7 characters
        if len(header) != 7:
            return False
        # Sequence number must contains a single number or be empty
        seqnr = header[0:6].replace(b'\x00', b' ').strip()
        if not seqnr.isdigit() and seqnr != b'':
            return False
        # Check for any valid control header types.
        if header[6:7] in [b'D', b'R', b'Q', b'M']:
            return True
        # Check if Full-SEED
        if not header[6:7] == b'V':
            return False
        # Parse the whole file and check whether it has has a data record.
        fp.seek(1, 1)
        _i = 0
        # search for blockettes 010 or 008
        while True:
            if fp.read(3) in [b'010', b'008']:
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
            if flag in [b'D', b'R', b'Q', b'M']:
                return True
            fp.seek(record_length - 1, 1)
        return False


def readMSEED(mseed_object, starttime=None, endtime=None, headonly=False,
              sourcename=None, reclen=None, details=False,
              header_byteorder=None, verbose=None, **kwargs):
    """
    Reads a Mini-SEED file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param mseed_object: Filename or open file like object that contains the
        binary Mini-SEED data. Any object that provides a read() method will be
        considered to be a file like object.
    :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param starttime: Only read data samples after or at the start time.
    :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param endtime: Only read data samples before or at the end time.
    :param headonly: Determines whether or not to unpack the data or just
        read the headers.
    :type sourcename: str
    :param sourcename: Source name has to have the structure
        'network.station.location.channel' and can contain globbing characters.
        Defaults to ``None``.
    :param reclen: If it is None, it will be automatically determined for every
        record. If it is known, just set it to the record length in bytes which
        will increase the reading speed slightly.
    :type details: bool, optional
    :param details: If ``True`` read additional information: timing quality
        and availability of calibration information.
        Note, that the traces are then also split on these additional
        information. Thus the number of traces in a stream will change.
        Details are stored in the mseed stats AttribDict of each trace.
        ``False`` specifies for both cases, that this information is not
        available. ``blkt1001.timing_quality`` specifies the timing quality
        from 0 to 100 [%]. ``calibration_type`` specifies the type of available
        calibration information blockettes:

        - ``1``: Step Calibration (Blockette 300)
        - ``2``: Sine Calibration (Blockette 310)
        - ``3``: Pseudo-random Calibration (Blockette 320)
        - ``4``: Generic Calibration  (Blockette 390)
        - ``-2``: Calibration Abort (Blockette 395)

    :type header_byteorder: int or str, optional
    :param header_byteorder: Must be either ``0`` or ``'<'`` for LSBF or
        little-endian, ``1`` or ``'>'`` for MBF or big-endian. ``'='`` is the
        native byte order. Used to enforce the header byte order. Useful in
        some rare cases where the automatic byte order detection fails.

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

    Read with ``details=True`` to read more details of the file if present.

    >>> st = read("/path/to/timingquality.mseed", details=True)
    >>> print(st[0].stats.mseed.blkt1001.timing_quality)
    55

    ``False`` means that the necessary information could not be found in the
    file.

    >>> print(st[0].stats.mseed.calibration_type)
    False

    Note that each change in timing quality from record to record may trigger a
    new Trace object to be created so the Stream object may contain many Trace
    objects if ``details=True`` is used.

    >>> print(len(st))
    101
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

    # Determine the byte order.
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
        msg = 'The quality flag is no longer supported in this version of ' + \
            'obspy.mseed. obspy.mseed.util has some functions with similar' + \
            ' behavior.'
        warnings.warn(msg, category=DeprecationWarning)

    # Parse some information about the file.
    if header_byteorder == 0:
        bo = "<"
    elif header_byteorder > 0:
        bo = ">"
    else:
        bo = None

    info = util.getRecordInformation(mseed_object, endian=bo)

    # Map the encoding to a readable string value.
    if info["encoding"] in ENCODINGS:
        info['encoding'] = ENCODINGS[info['encoding']][0]
    elif info["encoding"] in UNSUPPORTED_ENCODINGS:
        msg = ("Encoding '%s' (%i) is not supported by ObsPy. Please send "
               "the file to the ObsPy developers so that we can add "
               "support for it.") % \
            (UNSUPPORTED_ENCODINGS[info['encoding']], info['encoding'])
        raise ValueError(msg)
    else:
        msg = "Encoding '%i' is not a valid MiniSEED encoding." % \
            info['encoding']
        raise ValueError(msg)

    # Only keep information relevant for the whole file.
    info = {'encoding': info['encoding'],
            'filesize': info['filesize'],
            'record_length': info['record_length'],
            'byteorder': info['byteorder'],
            'number_of_records': info['number_of_records']}

    # If it's a file name just read it.
    if isinstance(mseed_object, (str, native_str)):
        # Read to NumPy array which is used as a buffer.
        bfrNp = np.fromfile(mseed_object, dtype=np.int8)
    elif hasattr(mseed_object, 'read'):
        bfrNp = np.fromstring(mseed_object.read(), dtype=np.int8)

    # Get the record length
    try:
        record_length = pow(2, int(''.join([chr(_i) for _i in bfrNp[19:21]])))
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
        if (isdigit(bfrNp[offset:offset + 6]) is False) or \
                (bfrNp[offset + 6] not in VALID_CONTROL_HEADERS):
            msg = 'Not a valid (Mini-)SEED file'
            raise Exception(msg)
        elif bfrNp[offset + 6] in SEED_CONTROL_HEADERS:
            offset += record_length
            continue
        break
    bfrNp = bfrNp[offset:]
    buflen = len(bfrNp)

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
            if not isinstance(sourcename, (str, native_str)):
                msg = 'sourcename needs to be a string'
                raise ValueError(msg)
            # libmseed uses underscores as separators and allows filtering
            # after the dataquality which is disabled here to not confuse
            # users. (* == all data qualities)
            selections.srcname = (sourcename.replace('.', '_') + '_*').\
                encode('ascii', 'ignore')
        else:
            selections.srcname = b'*'
    all_data = []

    # Use a callback function to allocate the memory and keep track of the
    # data.
    def allocate_data(samplecount, sampletype):
        # Enhanced sanity checking for libmseed 2.10 can result in the
        # sampletype not being set. Just return an empty array in this case.
        if sampletype == b"\x00":
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
        if msg.startswith(b"ERROR: "):
            raise InternalMSEEDReadingError(msg[7:].strip())
        if msg.startswith(b"INFO: "):
            msg = msg[6:].strip()
            # Append the offset of the full SEED header if necessary. That way
            # the C code does not have to deal with it.
            if offset and "offset" in msg:
                msg = ("%s The file contains a %i byte dataless part at the "
                       "beginning. Make sure to add that to the reported "
                       "offset to get the actual location in the file." % (
                           msg, offset))
            warnings.warn(msg, InternalMSEEDReadingWarning)
    diag_print = C.CFUNCTYPE(C.c_void_p, C.c_char_p)(log_error_or_warning)

    def log_message(msg):
        print(msg[6:].strip())
    log_print = C.CFUNCTYPE(C.c_void_p, C.c_char_p)(log_message)

    try:
        verbose = int(verbose)
    except:
        verbose = 0

    lil = clibmseed.readMSEEDBuffer(
        bfrNp, buflen, selections, C.c_int8(unpack_data),
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
            if details:
                timing_quality = currentSegment.timing_quality
                if timing_quality == 0xFF:  # 0xFF is mask for not known timing
                    timing_quality = False
                header['mseed']['blkt1001'] = {}
                header['mseed']['blkt1001']['timing_quality'] = timing_quality
                header['mseed']['calibration_type'] = \
                    currentSegment.calibration_type \
                    if currentSegment.calibration_type != -1 else False

            if headonly is False:
                # The data always will be in sequential order.
                data = all_data.pop(0)
                header['npts'] = len(data)
            else:
                data = np.array([])
                header['npts'] = currentSegment.samplecnt
            # Make sure to init the number of samples.
            # Py3k: convert to unicode
            header['mseed'] = dict((k, v.decode())
                                   if isinstance(v, bytes) else (k, v)
                                   for k, v in header['mseed'].items())
            header = dict((k, v.decode()) if isinstance(v, bytes) else (k, v)
                          for k, v in header.items())
            trace = Trace(header=header, data=data)
            # Append information.
            for key, value in info.items():
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
               sequence_number=None, flush=True, verbose=0, **_kwargs):
    """
    Write Mini-SEED file from a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: A Stream object.
    :type filename: str
    :param filename: Name of the output file or a file-like object.
    :type encoding: int or str, optional
    :param encoding: Should be set to one of the following supported Mini-SEED
        data encoding formats: ``ASCII`` (``0``)*, ``INT16`` (``1``),
        ``INT32`` (``3``), ``FLOAT32`` (``4``)*, ``FLOAT64`` (``5``)*,
        ``STEIM1`` (``10``) and ``STEIM2`` (``11``)*. If no encoding is given
        it will be derived from the dtype of the data and the appropriate
        default encoding (depicted with an asterix) will be chosen.
    :type reclen: int, optional
    :param reclen: Should be set to the desired data record length in bytes
        which must be expressible as 2 raised to the power of X where X is
        between (and including) 8 to 20.
        Defaults to 4096
    :type byteorder: int or str, optional
    :param byteorder: Must be either ``0`` or ``'<'`` for LSBF or
        little-endian, ``1`` or ``'>'`` for MBF or big-endian. ``'='`` is the
        native byte order. If ``-1`` it will be passed directly to libmseed
        which will also default it to big endian. Defaults to big endian.
    :type sequence_number: int, optional
    :param sequence_number: Must be an integer ranging between 1 and 999999.
        Represents the sequence count of the first record of each Trace.
        Defaults to 1.
    :type flush: bool, optional
    :param flush: If ``True``, all data will be packed into records. If
        ``False`` new records will only be created when there is enough data to
        completely fill a record. Be careful with this. If in doubt, choose
        ``True`` which is also the default value.
    :type verbose: int, optional
    :param verbose: Controls verbosity, a value of ``0`` will result in no
        diagnostic output.

    .. note::
        The ``reclen``, ``encoding``, ``byteorder`` and ``sequence_count``
        keyword arguments can be set in the ``stats.mseed`` of
        each :class:`~obspy.core.trace.Trace` as well as ``kwargs`` of this
        function. If both are given the ``kwargs`` will be used.

        The ``stats.mseed.blkt1001.timing_quality`` value will also be written
        if it is set.

        The ``stats.mseed.blkt1001.timing_quality`` value will also be written
        if it is set.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read()
    >>> st.write('filename.mseed', format='MSEED')  # doctest: +SKIP
    """
    # Map flush and verbose flags.
    if flush:
        flush = 1
    else:
        flush = 0

    if not verbose:
        verbose = 0
    if verbose is True:
        verbose = 1

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
            msg = "Invalid byte order. It must be either '<', '>', '=', " + \
                  "0, 1 or -1"
            raise ValueError(msg)

    if encoding is not None:
        encoding = util._convert_and_check_encoding_for_writing(encoding)

    if sequence_number is not None:
        # Check sequence number type
        try:
            sequence_number = int(sequence_number)
            # Check sequence number value
            if sequence_number < 1 or sequence_number > 999999:
                raise ValueError("Sequence number out of range. It must be " +
                                 " between 1 and 999999.")
        except (TypeError, ValueError):
            msg = "Invalid sequence number. It must be an integer ranging " +\
                  "from 1 to 999999."
            raise ValueError(msg)

    trace_attributes = []
    use_blkt_1001 = False

    # The data might need to be modified. To not modify the input data keep
    # references of which data to finally write.
    trace_data = []
    # Loop over every trace and figure out the correct settings.
    for _i, trace in enumerate(stream):
        # Create temporary dict for storing information while writing.
        trace_attr = {}
        trace_attributes.append(trace_attr)

        # Figure out whether or not to use Blockette 1001. This check is done
        # once to ensure that Blockette 1001 is either written for every record
        # in the file or for none. It checks the starttime, the sampling rate
        # and the timing quality. If starttime or sampling rate has a precision
        # of more than 100 microseconds, or if timing quality is set, \
        # Blockette 1001 will be written for every record.
        starttime = util._convertDatetimeToMSTime(trace.stats.starttime)
        if starttime % 100 != 0 or \
           (1.0 / trace.stats.sampling_rate * HPTMODULUS) % 100 != 0:
            use_blkt_1001 = True

        if hasattr(trace.stats, 'mseed') and \
           hasattr(trace.stats['mseed'], 'blkt1001') and \
           hasattr(trace.stats['mseed']['blkt1001'], 'timing_quality'):

            timing_quality = trace.stats['mseed']['blkt1001']['timing_quality']
            # Check timing quality type
            try:
                timing_quality = int(timing_quality)
                if timing_quality < 0 or timing_quality > 100:
                    raise ValueError("Timing quality out of range. It must be "
                                     + "between 0 and 100.")
            except ValueError:
                msg = "Invalid timing quality in Stream[%i].stats." % _i + \
                    "mseed.timing_quality. It must be an integer ranging" + \
                    " from 0 to 100"
                raise ValueError(msg)

            trace_attr['timing_quality'] = timing_quality
            use_blkt_1001 = True
        else:
            trace_attr['timing_quality'] = timing_quality = 0

        # Determine if a blockette 100 will be needed to represent the input
        # sample rate or if the sample rate in the fixed section of the data
        # header will suffice (see ms_genfactmult in libmseed/genutils.c)
        if trace.stats.sampling_rate >= 32727.0 or \
           trace.stats.sampling_rate <= (1.0 / 32727.0):
            use_blkt_100 = True
        else:
            use_blkt_100 = False

        if sequence_number is not None:
            trace_attr['sequence_number'] = sequence_number
        elif hasattr(trace.stats, 'mseed') and \
                hasattr(trace.stats['mseed'], 'sequence_number'):

            sequence_number = trace.stats['mseed']['sequence_number']
            # Check sequence number type
            try:
                sequence_number = int(sequence_number)
                # Check sequence number value
                if sequence_number < 1 or sequence_number > 999999:
                    raise ValueError("Sequence number out of range in " +
                                     "Stream[%i].stats. It must be between " +
                                     "1 and 999999.")
            except (TypeError, ValueError):
                msg = "Invalid sequence number in Stream[%i].stats." % _i +\
                      "mseed.sequence_number. It must be an integer ranging" +\
                      " from 1 to 999999."
                raise ValueError(msg)
            trace_attr['sequence_number'] = sequence_number
        else:
            trace_attr['sequence_number'] = sequence_number = 1

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
            trace.data = np.ascontiguousarray(trace.data)

        # Handle the record length.
        if reclen is not None:
            trace_attr['reclen'] = reclen
        elif hasattr(trace.stats, 'mseed') and \
                hasattr(trace.stats.mseed, 'record_length'):
            if trace.stats.mseed.record_length in VALID_RECORD_LENGTHS:
                trace_attr['reclen'] = trace.stats.mseed.record_length
            else:
                msg = 'Invalid record length in Stream[%i].stats.' % _i + \
                      'mseed.reclen.\nThe record length must be a value ' + \
                      'of 2 to the power of X where 8 <= X <= 20.'
                raise ValueError(msg)
        else:
            trace_attr['reclen'] = 4096

        # Handle the byte order.
        if byteorder is not None:
            trace_attr['byteorder'] = byteorder
        elif hasattr(trace.stats, 'mseed') and \
                hasattr(trace.stats.mseed, 'byteorder'):
            if trace.stats.mseed.byteorder in [0, 1, -1]:
                trace_attr['byteorder'] = trace.stats.mseed.byteorder
            elif trace.stats.mseed.byteorder == '=':
                if NATIVE_BYTEORDER == '<':
                    trace_attr['byteorder'] = 0
                else:
                    trace_attr['byteorder'] = 1
            elif trace.stats.mseed.byteorder == '<':
                trace_attr['byteorder'] = 0
            elif trace.stats.mseed.byteorder == '>':
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
        # If encoding arrives here it is already guaranteed to be a valid
        # integer encoding.
        if encoding is not None:
            # Check if the dtype for all traces is compatible with the enforced
            # encoding.
            ident, _, dtype, _ = ENCODINGS[encoding]
            if trace.data.dtype.type != dtype:
                msg = """
                    Wrong dtype for Stream[%i].data for encoding %s.
                    Please change the dtype of your data or use an appropriate
                    encoding. See the obspy.mseed documentation for more
                    information.
                    """ % (_i, ident)
                raise Exception(msg)
            trace_attr['encoding'] = encoding
        elif hasattr(trace.stats, 'mseed') and hasattr(trace.stats.mseed,
                                                       'encoding'):
            trace_attr["encoding"] = \
                util._convert_and_check_encoding_for_writing(
                    trace.stats.mseed.encoding)
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
            if trace.data.dtype.type == np.int32:
                trace_attr['encoding'] = 11
            elif trace.data.dtype.type == np.float32:
                trace_attr['encoding'] = 4
            elif trace.data.dtype.type == np.float64:
                trace_attr['encoding'] = 5
            elif trace.data.dtype.type == np.int16:
                trace_attr['encoding'] = 1
            elif trace.data.dtype.type == np.dtype(native_str('|S1')).type:
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
    # with more than one different encoding, record length or byte order.
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
    for trace, data, trace_attr in zip(stream, trace_data, trace_attributes):
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
        msr.contents.network = trace.stats.network.encode('ascii', 'strict')
        msr.contents.station = trace.stats.station.encode('ascii', 'strict')
        msr.contents.location = trace.stats.location.encode('ascii', 'strict')
        msr.contents.channel = trace.stats.channel.encode('ascii', 'strict')
        msr.contents.dataquality = trace_attr['dataquality'].\
            encode('ascii', 'strict')

        # Set starting sequence number
        msr.contents.sequence_number = trace_attr['sequence_number']

        # Only use Blockette 1001 if necessary.
        if use_blkt_1001:
            # Timing quality has been set in trace_attr

            size = C.sizeof(blkt_1001_s)
            # Only timing quality matters here, other blockette attributes will
            # be filled by libmseed.msr_normalize_header
            blkt_value = pack(native_str("BBBB"), trace_attr['timing_quality'],
                              0, 0, 0)
            blkt_ptr = C.create_string_buffer(blkt_value, len(blkt_value))

            # Usually returns a pointer to the added blockette in the
            # blockette link chain and a NULL pointer if it fails.
            # NULL pointers have a false boolean value according to the
            # ctypes manual.
            ret_val = clibmseed.msr_addblockette(msr, blkt_ptr,
                                                 size, 1001, 0)

            if bool(ret_val) is False:
                clibmseed.msr_free(C.pointer(msr))
                del msr
                raise Exception('Error in msr_addblockette')
        # Only use Blockette 100 if necessary.
        if use_blkt_100:
            size = C.sizeof(blkt_100_s)
            blkt100 = C.c_char(b' ')
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
    if not hasattr(filename, 'write'):
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
        self.mst.contents.network = trace.stats.network.\
            encode('ascii', 'strict')
        self.mst.contents.station = trace.stats.station.\
            encode('ascii', 'strict')
        self.mst.contents.location = trace.stats.location.\
            encode('ascii', 'strict')
        self.mst.contents.channel = trace.stats.channel.\
            encode('ascii', 'strict')
        self.mst.contents.dataquality = dataquality.encode('ascii', 'strict')
        self.mst.contents.type = b'\x00'
        self.mst.contents.starttime = \
            util._convertDatetimeToMSTime(trace.stats.starttime)
        self.mst.contents.endtime = \
            util._convertDatetimeToMSTime(trace.stats.endtime)
        self.mst.contents.samprate = trace.stats.sampling_rate
        self.mst.contents.samplecnt = trace.stats.npts
        self.mst.contents.numsamples = trace.stats.npts
        self.mst.contents.sampletype = sampletype.encode('ascii', 'strict')

        # libmseed expects data in the native byte order.
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
