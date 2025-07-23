# -*- coding: utf-8 -*-
"""
MSEED bindings to ObsPy core module.
"""
import ctypes as C  # NOQA
import io
import sys
import os
import struct
import traceback
import warnings
from pathlib import Path
from struct import pack

import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core.util import AttribDict
from obspy.core.compatibility import from_buffer
from obspy.core.util import NATIVE_BYTEORDER
from . import (util, InternalMSEEDError, ObsPyMSEEDFilesizeTooSmallError,
               ObsPyMSEEDFilesizeTooLargeError, ObsPyMSEEDError)
from .headers import (DATATYPES, ENCODINGS, HPTERROR, HPTMODULUS, SAMPLETYPE,
                      SEED_CONTROL_HEADERS, UNSUPPORTED_ENCODINGS,
                      VALID_CONTROL_HEADERS, VALID_RECORD_LENGTHS, Selections,
                      SelectTime, Blkt100S, Blkt1001S, clibmseed)


def _is_mseed(file):
    """
    Checks whether a file is Mini-SEED/full SEED or not.

    :type file: str or file-like object
    :param file: Mini-SEED/full SEED file to be checked.
    :rtype: bool
    :return: ``True`` if a Mini-SEED file.

    This method only reads the first seven bytes of the file and checks
    whether its a Mini-SEED or full SEED file.

    It also is true for fullSEED files because libmseed can read the data
    part of fullSEED files. If the method finds a fullSEED file it also
    checks if it has a data part and returns False otherwise.

    Thus it cannot be used to validate a Mini-SEED or SEED file.
    """
    # Open filehandler or use an existing file like object.
    if not hasattr(file, 'read'):
        file_size = os.path.getsize(file)
        with io.open(file, 'rb') as fh:
            return __is_mseed(fh, file_size=file_size)
    else:
        initial_pos = file.tell()
        try:
            if hasattr(file, "getbuffer"):  # BytesIO
                file_size = file.getbuffer().nbytes
            else:
                try:
                    file_size = os.fstat(file.fileno()).st_size
                except Exception:
                    _p = file.tell()
                    file.seek(0, 2)
                    file_size = file.tell()
                    file.seek(_p, 0)
            return __is_mseed(file, file_size)
        finally:
            # Reset pointer.
            file.seek(initial_pos, 0)


def __is_mseed(fp, file_size):  # NOQA
    """
    Internal version of _is_mseed working only with open file-like object.
    """
    header = fp.read(7)
    # File has less than 7 characters
    if len(header) != 7:
        return False
    # Sequence number must contains a single number or be empty
    seqnr = header[0:6].replace(b'\x00', b' ').strip()
    if not seqnr.isdigit() and seqnr != b'':
        # This might be a completely empty sequence - in that case jump 128
        # bytes and try again.
        fp.seek(-7, 1)
        try:
            _t = fp.read(128).decode().strip()
        except Exception:
            return False
        if not _t:
            return __is_mseed(fp=fp, file_size=file_size)
        return False
    # Check for any valid control header types.
    if header[6:7] in [b'D', b'R', b'Q', b'M']:
        return True
    elif header[6:7] == b" ":
        # If empty, it might be a noise record. Check the rest of 128 bytes
        # (min record size) and try again.
        try:
            _t = fp.read(128 - 7).decode().strip()
        except Exception:
            return False
        if not _t:
            return __is_mseed(fp=fp, file_size=file_size)
        return False
    # Check if Full-SEED
    elif header[6:7] != b'V':
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
        except Exception:
            return False
        _i += 1
        # break after 3 cycles
        if _i == 3:
            return False

    # Try to get a record length.
    fp.seek(8, 1)
    try:
        record_length = pow(2, int(fp.read(2)))
    except Exception:
        return False

    # Jump to the second record.
    fp.seek(record_length + 6, 0)
    # Loop over all records and return True if one record is a data
    # record
    while fp.tell() < file_size:
        flag = fp.read(1)
        if flag in [b'D', b'R', b'Q', b'M']:
            return True
        fp.seek(record_length - 1, 1)
    return False


# def _read_mseed(mseed_object, starttime=None, endtime=None, headonly=False,
#                 sourcename=None, reclen=None, details=False,
#                 header_byteorder=None, verbose=None, **kwargs):
#     """
#     Reads a Mini-SEED file and returns a Stream object.
#
#     .. warning::
#         This function should NOT be called directly, it registers via the
#         ObsPy :func:`~obspy.core.stream.read` function, call this instead.
#
#     :param mseed_object: Filename or open file like object that contains the
#         binary Mini-SEED data. Any object that provides a read() method will be
#         considered to be a file like object.
#     :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
#     :param starttime: Only read data samples after or at the start time.
#     :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
#     :param endtime: Only read data samples before or at the end time.
#     :param headonly: Determines whether or not to unpack the data or just
#         read the headers.
#     :type sourcename: str
#     :param sourcename: Only read data with matching SEED ID (can contain
#         wildcards "?" and "*", e.g. "BW.UH2.*" or "*.??Z").
#         Defaults to ``None`` .
#     :param reclen: If it is None, it will be automatically determined for every
#         record. If it is known, just set it to the record length in bytes which
#         will increase the reading speed slightly.
#     :type details: bool, optional
#     :param details: If ``True`` read additional information: timing quality
#         and availability of calibration information.
#         Note, that the traces are then also split on these additional
#         information. Thus the number of traces in a stream will change.
#         Details are stored in the mseed stats AttribDict of each trace.
#         ``False`` specifies for both cases, that this information is not
#         available. ``blkt1001.timing_quality`` specifies the timing quality
#         from 0 to 100 [%]. ``calibration_type`` specifies the type of available
#         calibration information blockettes:
#
#         - ``1`` : Step Calibration (Blockette 300)
#         - ``2`` : Sine Calibration (Blockette 310)
#         - ``3`` : Pseudo-random Calibration (Blockette 320)
#         - ``4`` : Generic Calibration  (Blockette 390)
#         - ``-2`` : Calibration Abort (Blockette 395)
#
#     :type header_byteorder: int or str, optional
#     :param header_byteorder: Must be either ``0`` or ``'<'`` for LSBF or
#         little-endian, ``1`` or ``'>'`` for MBF or big-endian. ``'='`` is the
#         native byte order. Used to enforce the header byte order. Useful in
#         some rare cases where the automatic byte order detection fails.
#
#     .. rubric:: Example
#
#     >>> from obspy import read
#     >>> st = read("/path/to/two_channels.mseed")
#     >>> print(st)  # doctest: +ELLIPSIS
#     2 Trace(s) in Stream:
#     BW.UH3..EHE | 2010-06-20T00:00:00.279999Z - ... | 200.0 Hz, 386 samples
#     BW.UH3..EHZ | 2010-06-20T00:00:00.279999Z - ... | 200.0 Hz, 386 samples
#
#     >>> from obspy import UTCDateTime
#     >>> st = read("/path/to/two_channels.mseed",
#     ...           starttime=UTCDateTime("2010-06-20T00:00:01"),
#     ...           sourcename="*.?HZ")
#     >>> print(st)  # doctest: +ELLIPSIS
#     1 Trace(s) in Stream:
#     BW.UH3..EHZ | 2010-06-20T00:00:00.999999Z - ... | 200.0 Hz, 242 samples
#
#     Read with ``details=True`` to read more details of the file if present.
#
#     >>> st = read("/path/to/timingquality.mseed", details=True)
#     >>> print(st[0].stats.mseed.blkt1001.timing_quality)
#     55
#
#     ``False`` means that the necessary information could not be found in the
#     file.
#
#     >>> print(st[0].stats.mseed.calibration_type)
#     False
#
#     Note that each change in timing quality from record to record may trigger a
#     new Trace object to be created so the Stream object may contain many Trace
#     objects if ``details=True`` is used.
#
#     >>> print(len(st))
#     101
#     """
#     if isinstance(mseed_object, Path):
#         mseed_object = str(mseed_object)
#     # Parse the headonly and reclen flags.
#     if headonly is True:
#         unpack_data = 0
#     else:
#         unpack_data = 1
#     if reclen is None:
#         reclen = -1
#     elif reclen not in VALID_RECORD_LENGTHS:
#         msg = 'Invalid record length. Autodetection will be used.'
#         warnings.warn(msg)
#         reclen = -1
#
#     # Determine the byte order.
#     if header_byteorder == "=":
#         header_byteorder = NATIVE_BYTEORDER
#
#     if header_byteorder is None:
#         header_byteorder = -1
#     elif header_byteorder in [0, "0", "<"]:
#         header_byteorder = 0
#     elif header_byteorder in [1, "1", ">"]:
#         header_byteorder = 1
#
#     # Parse some information about the file.
#     if header_byteorder == 0:
#         bo = "<"
#     elif header_byteorder > 0:
#         bo = ">"
#     else:
#         bo = None
#
#     # Determine total size. Either its a file-like object.
#     if hasattr(mseed_object, "tell") and hasattr(mseed_object, "seek"):
#         cur_pos = mseed_object.tell()
#         mseed_object.seek(0, 2)
#         length = mseed_object.tell() - cur_pos
#         mseed_object.seek(cur_pos, 0)
#     # Or a file name.
#     else:
#         length = os.path.getsize(mseed_object)
#
#     if length < 128:
#         msg = "The smallest possible mini-SEED record is made up of 128 " \
#               "bytes. The passed buffer or file contains only %i." % length
#         raise ObsPyMSEEDFilesizeTooSmallError(msg)
#     elif length > 2 ** 31:
#         msg = ("ObsPy can currently not directly read mini-SEED files that "
#                "are larger than 2^31 bytes (2048 MiB). To still read it, "
#                "please read the file in chunks as documented here: "
#                "https://github.com/obspy/obspy/pull/1419"
#                "#issuecomment-221582369")
#         raise ObsPyMSEEDFilesizeTooLargeError(msg)
#
#     info = util.get_record_information(mseed_object, endian=bo)
#
#     # Map the encoding to a readable string value.
#     if "encoding" not in info:
#         # Hopefully detected by libmseed.
#         info["encoding"] = None
#     elif info["encoding"] in ENCODINGS:
#         info['encoding'] = ENCODINGS[info['encoding']][0]
#     elif info["encoding"] in UNSUPPORTED_ENCODINGS:
#         msg = ("Encoding '%s' (%i) is not supported by ObsPy. Please send "
#                "the file to the ObsPy developers so that we can add "
#                "support for it.") % \
#             (UNSUPPORTED_ENCODINGS[info['encoding']], info['encoding'])
#         raise ValueError(msg)
#     else:
#         msg = "Encoding '%i' is not a valid MiniSEED encoding." % \
#             info['encoding']
#         raise ValueError(msg)
#
#     record_length = info["record_length"]
#
#     # Only keep information relevant for the whole file.
#     info = {'filesize': info['filesize']}
#
#     # If it's a file name just read it.
#     if isinstance(mseed_object, str):
#         # Read to NumPy array which is used as a buffer.
#         bfr_np = np.fromfile(mseed_object, dtype=np.int8)
#     elif hasattr(mseed_object, 'read'):
#         bfr_np = from_buffer(mseed_object.read(), dtype=np.int8)
#
#     # Search for data records and pass only the data part to the underlying C
#     # routine.
#     offset = 0
#     # 0 to 9 are defined in a row in the ASCII charset.
#     min_ascii = ord('0')
#
#     # Small function to check whether an array of ASCII values contains only
#     # digits.
#     def isdigit(x):
#         return True if (x - min_ascii).max() <= 9 else False
#
#     while True:
#         # This should never happen
#         if (isdigit(bfr_np[offset:offset + 6]) is False) or \
#                 (bfr_np[offset + 6] not in VALID_CONTROL_HEADERS):
#             msg = 'Not a valid (Mini-)SEED file'
#             raise Exception(msg)
#         elif bfr_np[offset + 6] in SEED_CONTROL_HEADERS:
#             offset += record_length
#             continue
#         break
#     bfr_np = bfr_np[offset:]
#     buflen = len(bfr_np)
#
#     # If no selection is given pass None to the C function.
#     if starttime is None and endtime is None and sourcename is None:
#         selections = None
#     else:
#         select_time = SelectTime()
#         selections = Selections()
#         selections.timewindows.contents = select_time
#         if starttime is not None:
#             if not isinstance(starttime, UTCDateTime):
#                 msg = 'starttime needs to be a UTCDateTime object'
#                 raise ValueError(msg)
#             selections.timewindows.contents.starttime = \
#                 util._convert_datetime_to_mstime(starttime)
#         else:
#             # HPTERROR results in no starttime.
#             selections.timewindows.contents.starttime = HPTERROR
#         if endtime is not None:
#             if not isinstance(endtime, UTCDateTime):
#                 msg = 'endtime needs to be a UTCDateTime object'
#                 raise ValueError(msg)
#             selections.timewindows.contents.endtime = \
#                 util._convert_datetime_to_mstime(endtime)
#         else:
#             # HPTERROR results in no starttime.
#             selections.timewindows.contents.endtime = HPTERROR
#         if sourcename is not None:
#             if not isinstance(sourcename, str):
#                 msg = 'sourcename needs to be a string'
#                 raise ValueError(msg)
#             # libmseed uses underscores as separators and allows filtering
#             # after the dataquality which is disabled here to not confuse
#             # users. (* == all data qualities)
#             selections.srcname = (sourcename.replace('.', '_') + '_*').\
#                 encode('ascii', 'ignore')
#         else:
#             selections.srcname = b'*'
#     all_data = []
#
#     # Use a callback function to allocate the memory and keep track of the
#     # data.
#     def allocate_data(samplecount, sampletype):
#         # Enhanced sanity checking for libmseed 2.10 can result in the
#         # sampletype not being set. Just return an empty array in this case.
#         if sampletype == b"\x00":
#             data = np.empty(0)
#         else:
#             data = np.empty(samplecount, dtype=DATATYPES[sampletype])
#         all_data.append(data)
#         return data.ctypes.data
#     # XXX: Do this properly!
#     # Define Python callback function for use in C function. Return a long so
#     # it hopefully works on 32 and 64 bit systems.
#     alloc_data = C.CFUNCTYPE(C.c_longlong, C.c_int, C.c_char)(allocate_data)
#
#     try:
#         verbose = int(verbose)
#     except Exception:
#         verbose = 0
#
#     clibmseed.verbose = bool(verbose)
#     try:
#         lil = clibmseed.readMSEEDBuffer(
#             bfr_np, buflen, selections, C.c_int8(unpack_data),
#             reclen, C.c_int8(verbose), C.c_int8(details), header_byteorder,
#             alloc_data)
#     except InternalMSEEDError as e:
#         msg = e.args[0]
#         if offset and offset in str(e):
#             # Append the offset of the full SEED header if necessary. That way
#             # the C code does not have to deal with it.
#             if offset and "offset" in msg:
#                 msg = ("%s\nThe file contains a %i byte dataless part at the "
#                        "beginning. Make sure to add that to the reported "
#                        "offset to get the actual location in the file." % (
#                            msg, offset))
#                 raise InternalMSEEDError(msg)
#         else:
#             raise
#     finally:
#         # Make sure to reset the verbosity.
#         clibmseed.verbose = True
#
#     del selections
#
#     traces = []
#     try:
#         current_id = lil.contents
#     # Return stream if not traces are found.
#     except ValueError:
#         clibmseed.lil_free(lil)
#         del lil
#         return Stream()
#
#     while True:
#         # Init header with the essential information.
#         header = {'network': current_id.network.strip(),
#                   'station': current_id.station.strip(),
#                   'location': current_id.location.strip(),
#                   'channel': current_id.channel.strip(),
#                   'mseed': {'dataquality': current_id.dataquality}}
#         # Loop over segments.
#         try:
#             current_segment = current_id.firstSegment.contents
#         except ValueError:
#             break
#         while True:
#             header['sampling_rate'] = current_segment.samprate
#             header['starttime'] = \
#                 util._convert_mstime_to_datetime(current_segment.starttime)
#             header['mseed']['number_of_records'] = current_segment.recordcnt
#             header['mseed']['encoding'] = \
#                 ENCODINGS[current_segment.encoding][0]
#             header['mseed']['byteorder'] = \
#                 "<" if current_segment.byteorder == 0 else ">"
#             header['mseed']['record_length'] = current_segment.reclen
#             if details:
#                 timing_quality = current_segment.timing_quality
#                 if timing_quality == 0xFF:  # 0xFF is mask for not known timing
#                     timing_quality = False
#                 header['mseed']['blkt1001'] = {}
#                 header['mseed']['blkt1001']['timing_quality'] = timing_quality
#                 header['mseed']['calibration_type'] = \
#                     current_segment.calibration_type \
#                     if current_segment.calibration_type != -1 else False
#
#             if headonly is False:
#                 # The data always will be in sequential order.
#                 data = all_data.pop(0)
#                 header['npts'] = len(data)
#             else:
#                 data = np.array([])
#                 header['npts'] = current_segment.samplecnt
#             # Make sure to init the number of samples.
#             # Py3k: convert to unicode
#             header['mseed'] = dict((k, v.decode())
#                                    if isinstance(v, bytes) else (k, v)
#                                    for k, v in header['mseed'].items())
#             header = dict((k, util._decode_header_field(k, v))
#                           if isinstance(v, bytes) else (k, v)
#                           for k, v in header.items())
#             trace = Trace(header=header, data=data)
#             # Append global information.
#             for key, value in info.items():
#                 setattr(trace.stats.mseed, key, value)
#             traces.append(trace)
#             # A Null pointer access results in a ValueError
#             try:
#                 current_segment = current_segment.next.contents
#             except ValueError:
#                 break
#         try:
#             current_id = current_id.next.contents
#         except ValueError:
#             break
#
#     clibmseed.lil_free(lil)  # NOQA
#     del lil  # NOQA
#     return Stream(traces=traces)

from mseedlib import MS3RecordReader, MSTraceList, TimeFormat
from obspy.core import Stats


def _get_mseed_input_file(filename):
    """
    Utility function to handle different types of input (filename, file object, BytesIO)
    and return a suitable filename for mseedlib functions.

    :param filename: Input source, can be a filename, file object, or BytesIO object.
    :type filename: str, file, or BytesIO
    :return: Tuple of (filename to use, temp_file object if created or None)
    """
    temp_file = None
    input_filename = filename

    # Check if it's a file-like object without fileno() method (like BytesIO)
    if hasattr(filename, 'read'):
        import tempfile

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        input_filename = temp_file.name

        # Write the content to the temporary file
        filename.seek(0)
        temp_file.write(filename.read())
        temp_file.seek(0)
        temp_file.close()
        # bubu = temp_file.fileno()


    else:
        input_filename = str(filename)

    return input_filename


def extract_mseed_metadata(record, stats):
    """
    Extract all available metadata from a mseedlib record and add it to ObsPy stats object.

    :param record: mseedlib record object (MS3Record or segment from MSTraceID)
    :param stats: ObsPy Stats object to populate
    :return: Updated stats object
    """
    # Create mseed and mseed3 attribute dictionaries if they don't exist
    if not hasattr(stats, 'mseed'):
        stats.mseed = {}


    # Common attributes for both MSEED2 and MSEED3
    # 1. Basic timing information
    # if not hasattr(stats.mseed, 'records'):
    #     stats.mseed.records = []
    # if hasattr(record, 'starttime'):
        # Convert from nanoseconds to seconds for UTCDateTime
        # start_ns = record.starttime
        # start_sec = start_ns / 1_000_000_000
        # stats.mseed.records.append({"startime": UTCDateTime(start_sec)})

    if hasattr(record, 'samplerate'):
        stats.sampling_rate = record.samplerate

    # 2. Record info
    if hasattr(record, 'reclen'):
        stats.mseed['record_length'] = record.reclen

    if hasattr(record, 'samplecnt'):
        stats.npts = record.samplecnt

    # # 3. Sample encoding
    if hasattr(record, 'sampletype'):
        stats.mseed['encoding'] = record.sampletype

    # 4. Byte order (extracted from MSEED record)
    try:
        # Extract byte order from the actual MSEED record
        # The record object should have byte order information
        if hasattr(record, 'byteorder'):
            stats.mseed['byteorder'] = record.byteorder
        elif hasattr(record, 'byte_order'):
            stats.mseed['byteorder'] = record.byte_order
        elif hasattr(record, 'header') and hasattr(record.header, 'byteorder'):
            stats.mseed['byteorder'] = record.header.byteorder
        else:
            # Try to determine from record data/header structure
            # If record has raw data, examine the header bytes directly
            if hasattr(record, 'data') and len(record.data) >= 22:
                from struct import unpack

                # Check year field in MSEED header (bytes 20-21)
                year_be = unpack('>H', record.data[20:22])[0]  # big-endian
                year_le = unpack('<H', record.data[20:22])[0]  # little-endian

                # The year should be reasonable (1970-2070 range)
                if 1970 <= year_be <= 2070:
                    stats.mseed['byteorder'] = '>'
                elif 1970 <= year_le <= 2070:
                    stats.mseed['byteorder'] = '<'
                else:
                    # Default fallback
                    stats.mseed['byteorder'] = '>'
            else:
                # Final fallback
                stats.mseed['byteorder'] = '>'

    except (AttributeError, IndexError, struct.error):
        # Fallback to big-endian if detection fails
        stats.mseed['byteorder'] = '>'  # Default to big-endian for MSEED

    # 5. Data quality indicator
    if hasattr(record, 'quality'):
        stats.mseed['dataquality'] = record.quality

    # MSEED version-specific attributes

    # MSEED3-specific attributes
    # 1. Record publishing/format attributes
    if hasattr(record, 'pubversion'):
        stats.mseed['dataquality'] = {1: "R", 2: "D", 3: "Q", 4: "M"}[record.pubversion]

    if hasattr(record, 'formatversion') and record.formatversion >= 3:
        if hasattr(record, 'formatversion'):
            stats.mseed['miniseed_version'] = record.formatversion

        # These attributes are specific to MSEED3
        if not hasattr(stats, 'mseed3'):
            stats.mseed3 = {}
        stats.mseed3['publication_version'] = record.pubversion
        # 2. Source identifier details
        if hasattr(record, 'sourceid'):
            stats.mseed3['source_id'] = record.sourceid

        # 3. CRC values (data integrity)
        if hasattr(record, 'crc'):
            stats.mseed3['crc'] = record.crc

        # 4. Extra headers (JSON metadata)
        if hasattr(record, 'extra_headers') and record.extra_headers:
            stats.mseed3['extra_headers'] = record.extra_headers

        # 5. Record identifier attributes
        if hasattr(record, 'recid'):
            stats.mseed3['record_id'] = record.recid

        # 6. Timing quality
        if hasattr(record, 'timingquality'):
            stats.mseed3['timing_quality'] = record.timingquality

        # 7. Publication time
        if hasattr(record, 'pubtime'):
            pub_ns = record.pubtime
            pub_sec = pub_ns / 1_000_000_000
            stats.mseed3['publication_time'] = UTCDateTime(pub_sec)


        # 8. Activity/IO/Data quality flags
        if hasattr(record, 'flags'):
            stats.mseed3['flags'] = record.flags

    # MSEED2-specific attributes
    if hasattr(record, 'formatversion') and record.formatversion < 3:
        # MSEED2 specific attributes

        # 1. Blockette presence
        if hasattr(record, 'blockettecount'):
            stats.mseed['blockette_count'] = record.blockettecount

        # 2. Network/station/location/channel codes (if stored separately)
        if hasattr(record, 'network'):
            stats.network = record.network
        if hasattr(record, 'station'):
            stats.station = record.station
        if hasattr(record, 'location'):
            stats.location = record.location
        if hasattr(record, 'channel'):
            stats.channel = record.channel

        # 3. Sequence number
        if hasattr(record, 'sequence_number'):
            stats.mseed['sequence_number'] = record.sequence_number

        # 4. Activity/IO/Data quality flags
        if hasattr(record, 'act_flags'):
            stats.mseed['activity_flags'] = record.act_flags
        if hasattr(record, 'io_flags'):
            stats.mseed['io_flags'] = record.io_flags
        if hasattr(record, 'dq_flags'):
            stats.mseed['data_quality_flags'] = record.dq_flags

    # Add any additional attributes available that we didn't explicitly handle
    # This ensures we don't miss any metadata in future mseedlib versions
    # for attr_name in dir(record):
    #     # Skip private attributes, methods, and ones we've already handled
    #     if (attr_name.startswith('_') or
    #             attr_name in ['starttime', 'samplerate', 'reclen', 'samplecnt',
    #                           'sampletype', 'formatversion',
    #                           'sourceid', 'crc', 'extra_headers', 'recid',
    #                           'timingquality', 'flags', 'blockettecount',
    #                           'network', 'station', 'location', 'channel', 'msr',
    #                           'sequence_number', 'act_flags', 'io_flags', 'dq_flags',
    #                            'filename', 'datasamples'
    #                           ]):
    #         continue
    #
    #     # Add qto the appropriate attribute dict based on version
    #     if hasattr(record, 'formatversion') and record.formatversion >= 3:
    #         stats.mseed3[attr_name] = getattr(record, attr_name)
    #     else:
    #         stats.mseed[attr_name] = getattr(record, attr_name)

    return stats


def _matches_sourcename(sourceid, sourcename):
    """Check if sourceid matches the sourcename pattern."""
    if sourcename is None:
        return True

    # Convert FDSN format to standard SEED format for comparison
    if sourceid.startswith("FDSN:"):
        parts = sourceid[5:].split('_')
        if len(parts) >= 4:
            seed_id = f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}{parts[4]}{parts[5]}"
        else:
            seed_id = sourceid
    else:
        # Convert underscores to dots
        seed_id = sourceid.replace('_', '.')

    # Use fnmatch for wildcard matching
    import fnmatch
    return fnmatch.fnmatch(seed_id, sourcename)


def _read_mseed(filename, starttime=None, endtime=None, headonly=False,
                sourcename=None, reclen=None, details=False,
                header_byteorder=None, verbose=None, **kwargs):
    """
    Reads a MSEED file and returns an ObsPy Stream object.

    :param filename: MSEED3 file to be read.
    :type filename: str
    :param headonly: If True, read only the headers. Default is False.
    :type headonly: bool
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: An ObsPy Stream object containing the data.
    """
    # Enhanced validation with proper error types
    if reclen is not None:
        if not isinstance(reclen, int) or reclen not in VALID_RECORD_LENGTHS:
            import warnings
            warnings.warn('Invalid record length. Autodetection will be used.')
            reclen = None

    # Process and normalize header_byteorder parameter
    processed_header_byteorder = None
    if header_byteorder is not None:
        if header_byteorder == "=":
            processed_header_byteorder = NATIVE_BYTEORDER
        elif header_byteorder in [0, "0", "<", "little"]:
            processed_header_byteorder = "<"
        elif header_byteorder in [1, "1", ">", "big"]:
            processed_header_byteorder = ">"
        else:
            import warnings
            warnings.warn(f"Invalid header_byteorder '{header_byteorder}', using autodetection")
            processed_header_byteorder = None

    try:
        # More robust file size detection with better error handling
        if hasattr(filename, "read"):
            # File-like object - handle various types more safely
            cur_pos = filename.tell()
            try:
                if hasattr(filename, "getbuffer"):  # BytesIO
                    length = filename.getbuffer().nbytes - cur_pos
                elif hasattr(filename, "fileno"):  # Real file objects
                    file_stat = os.fstat(filename.fileno())
                    length = file_stat.st_size - cur_pos
                else:  # Fallback for other file-like objects
                    filename.seek(0, 2)
                    length = filename.tell() - cur_pos
                    filename.seek(cur_pos, 0)
            except (OSError, IOError, AttributeError) as e:
                # If we can't determine size, let the library handle it
                length = None
        else:
            # File path - more robust path handling
            try:
                if isinstance(filename, (str, bytes, os.PathLike)):
                    length = os.path.getsize(filename)
                else:
                    raise TypeError(f"Invalid filename type: {type(filename)}")
            except (OSError, IOError) as e:
                raise IOError(f"Cannot access file {filename}: {str(e)}")

        # Only check size if we could determine it
        if length is not None:
            if length < 128:
                msg = ("The smallest possible mini-SEED record is made up of 128 "
                       "bytes. The passed buffer or file contains only %i." % length)
                raise ObsPyMSEEDFilesizeTooSmallError(msg)

    except (OSError, IOError):
        raise
    except Exception as e:
        raise IOError(f"Error accessing file {filename}: {str(e)}")

    # Handle different types of input
    temp_file = None
    input_filename = None

    try:
        input_filename = _get_mseed_input_file(filename)

        # Add validation that the file was properly created/accessed
        if not input_filename or not os.path.exists(input_filename):
            raise IOError("Failed to prepare input file for reading")

        # Initialize with better error context
        try:
            mstracelist = MSTraceList(input_filename, unpack_data=not headonly, record_list=True)
        except Exception as e:
            # More specific error classification
            error_msg = str(e).lower()
            if any(phrase in error_msg for phrase in ['not a valid', 'invalid format', 'corrupt']):
                raise ObsPyMSEEDError('Not a valid (Mini-)SEED file')
            elif 'permission' in error_msg or 'access' in error_msg:
                raise IOError(f"Permission denied accessing file: {filename}")
            elif 'memory' in error_msg or 'allocation' in error_msg:
                raise MemoryError(f"Insufficient memory to read file: {filename}")
            elif 'No miniSEED data detected' in error_msg:
                raise IOError(f"No MiniSEED data record found in file. {filename}")
            else:
                raise IOError(f"Error reading MSEED file: {str(e)}")

    except (ObsPyMSEEDError, IOError, MemoryError):
        raise
    except Exception as e:
        raise IOError(f"Unexpected error reading MSEED file {filename}: {str(e)}")

    traces = []

    # Iterate through each trace ID in the trace list
    for traceid in mstracelist.traceids():
        # Parse network, station, location, channel from sourceid
        sourceid = traceid.sid.decode("utf8")
        if not _matches_sourcename(sourceid, sourcename):
            continue

        # Try to parse FDSN ID format: FDSN:NET_STA_LOC_CHAN
        if sourceid.startswith("FDSN:"):
            parts = sourceid[5:].split('_')
            if len(parts) >= 4:
                network = parts[0]
                station = parts[1]
                location = parts[2]
                channel = "".join(parts[3:6]).strip()
            else:
                # Handle non-standard IDs
                network, station, location, channel = "", "", "", sourceid
        else:
            # Handle other ID formats or try to parse based on underscore separators
            parts = sourceid.split('_')
            if len(parts) >= 4:
                network = parts[0]
                station = parts[1]
                location = parts[2]
                channel = parts[3]
            else:
                # Default when can't parse
                network, station, location, channel = "", "", "", sourceid

        # Process each segment for this trace ID
        for segment in traceid.segments():
            # Create Stats object
            stats = Stats()
            stats.network = network
            stats.station = station
            stats.location = location
            stats.channel = channel

            # Set timing information - convert from nanoseconds to UTCDateTime
            start_ns = segment.starttime
            start_sec = start_ns / 1_000_000_000
            stats.starttime = UTCDateTime(start_sec)

            # Set sampling rate
            stats.sampling_rate = segment.samprate

            # Initialize mseed stats with defaults
            stats.mseed = AttribDict()

            # Process records to extract metadata and determine byteorder
            record_count = 0
            detected_byteorder = None
            record_length = None

            for rec in segment.recordlist.records():
                record_count += 1

                # Extract metadata from MSRecord
                stats = extract_mseed_metadata(rec.msr, stats)

                # Get byteorder from the first record
                if record_count == 1:
                    # Try to get byteorder from MSRecord structure
                    if hasattr(rec.msr, 'byteorder'):
                        # MSRecord byteorder: 0 = little-endian, 1 = big-endian
                        detected_byteorder = '<' if rec.msr.byteorder == 0 else '>'
                    elif hasattr(rec.msr, 'Blkt1000'):
                        # Extract from Blockette 1000 if available
                        blkt1000 = rec.msr.Blkt1000
                        if blkt1000:
                            # Blockette 1000 word order: 0 = little, 1 = big
                            word_order = getattr(blkt1000, 'word_order', None)
                            if word_order is not None:
                                detected_byteorder = '<' if word_order == 0 else '>'

                    # Get record length from first record
                    if hasattr(rec.msr, 'reclen'):
                        record_length = rec.msr.reclen

                # Validate encoding
                encoding_code = rec.msr.encoding
                if encoding_code not in ENCODINGS:
                    if encoding_code in UNSUPPORTED_ENCODINGS:
                        msg = ("Encoding '%s' (%i) is not supported by ObsPy. Please send "
                               "the file to the ObsPy developers so that we can add "
                               "support for it.") % (UNSUPPORTED_ENCODINGS[encoding_code], encoding_code)
                        raise ValueError(msg)
                    else:
                        msg = "Encoding '%i' is not a valid MiniSEED encoding." % encoding_code
                        raise ValueError(msg)

                # Store encoding from first record
                if record_length:
                    stats.mseed["encoding"] = ENCODINGS[rec.msr.encoding][0]

            # Set MSeed metadata
            stats.mseed["number_of_records"] = record_count

            # Determine final byteorder to use
            # Handle byteorder precedence: user specified takes priority
            if processed_header_byteorder is not None:
                # User specified byteorder overrides detected one
                stats.mseed["byteorder"] = processed_header_byteorder
            # If no byteorder was set by extract_mseed_metadata, set a default
            elif "byteorder" not in stats.mseed:
                stats.mseed["byteorder"] = NATIVE_BYTEORDER

            # Store record length if detected
            if record_length is not None:
                stats.mseed["record_length"] = record_length

            # Get file size
            try:
                with open(input_filename, 'rb') as f:
                    f.seek(0, 2)
                    stats.mseed["filesize"] = f.tell()
            except Exception:
                stats.mseed["filesize"] = 0

            # If headonly, create an empty trace
            if headonly:
                trace = Trace(data=np.array([]), header=stats)
                trace.stats.npts = segment.samplecnt if hasattr(segment, 'samplecnt') else 0
                traces.append(trace)
                continue

            # Get the data samples
            if segment.numsamples:
                data = np.array(segment.datasamples)
            else:
                # If no data samples but we know the count, create zeros
                sample_count = segment.samplecnt if hasattr(segment, 'samplecnt') else 0
                data = np.zeros(sample_count)

            # Handle data byteorder properly
            if len(data) > 0:
                # Get the original data byteorder
                original_data_byteorder = data.dtype.byteorder
                if original_data_byteorder == '=':
                    original_data_byteorder = NATIVE_BYTEORDER

                # If user specified a byteorder that differs from data, we may need to swap
                if processed_header_byteorder is not None and processed_header_byteorder != original_data_byteorder:
                    # Convert data to the requested byteorder
                    if processed_header_byteorder == '<' and original_data_byteorder == '>':
                        data = data.byteswap().newbyteorder('<')
                    elif processed_header_byteorder == '>' and original_data_byteorder == '<':
                        data = data.byteswap().newbyteorder('>')

                # Always ensure data is in native byte order for processing
                if data.dtype.byteorder not in ('=', NATIVE_BYTEORDER):
                    data = data.astype(data.dtype.newbyteorder('='))

            trace = Trace(data=data, header=stats)
            trace.stats.npts = len(data)

            if details:
                # Extract detailed information from records
                timing_quality = False  # Default
                calibration_type = False  # Default

                # Try to extract from MSRecord if available
                for rec in segment.recordlist.records():
                    # Extract blockette information if present
                    if hasattr(rec.msr, 'timing_quality'):
                        timing_quality = rec.msr.timing_quality
                        if timing_quality == 0xFF:
                            timing_quality = False
                        break  # Use first record's timing quality

                    if hasattr(rec.msr, 'calibration_type'):
                        cal_type = rec.msr.calibration_type
                        calibration_type = cal_type if cal_type != -1 else False

                # Store detailed information
                if 'blkt1001' not in trace.stats.mseed:
                    trace.stats.mseed['blkt1001'] = AttribDict()
                trace.stats.mseed['blkt1001']['timing_quality'] = timing_quality
                trace.stats.mseed['calibration_type'] = calibration_type

            traces.append(trace)

    stream = Stream(traces=traces)

    # Apply time window filtering if specified
    if starttime or endtime:
        stream.trim(starttime=starttime, endtime=endtime)

    return stream.sort()


def _write_mseed(stream, filename, encoding=None, reclen=None, byteorder=None,
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
        starttime = util._convert_datetime_to_mstime(trace.stats.starttime)
        if starttime % 100 != 0 or (
                trace.stats.sampling_rate and
                (1.0 / trace.stats.sampling_rate * HPTMODULUS) % 100 != 0):
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
                                     "between 0 and 100.")
            except ValueError:
                msg = "Invalid timing quality in Stream[%i].stats." % _i + \
                    "mseed.timing_quality. It must be an integer ranging" + \
                    " from 0 to 100"
                raise ValueError(msg)

            trace_attr['timing_quality'] = timing_quality
            use_blkt_1001 = True
        else:
            trace_attr['timing_quality'] = timing_quality = 0

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
        except Exception:
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
                    encoding. See the obspy.io.mseed documentation for more
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
        if trace_attr['encoding'] is None:
            if trace.data.dtype.type == np.int32:
                trace_attr['encoding'] = 11
            elif trace.data.dtype.type == np.float32:
                trace_attr['encoding'] = 4
            elif trace.data.dtype.type == np.float64:
                trace_attr['encoding'] = 5
            elif trace.data.dtype.type == np.int16:
                trace_attr['encoding'] = 1
            elif trace.data.dtype.type == np.dtype('|S1').type:
                trace_attr['encoding'] = 0
            # int64 data not supported; if possible downcast to int32, else
            # create error message. After bumping up to numpy 1.9.0 this check
            # can be replaced by numpy.can_cast()
            # -- actually not sure, it even looks like can_cast() does not
            # check individual values in arrays, so it might be better to keep
            # the current check using iinfo().
            elif trace.data.dtype.type == np.int64:
                # check if data can be safely downcast to int32
                ii32 = np.iinfo(np.int32)
                if abs(trace.max()) <= ii32.max:
                    trace_data.append(trace.data.astype(np.int32, copy=True))
                    trace_attr['encoding'] = 11
                else:
                    msg = ("int64 data only supported when writing MSEED if "
                           "it can be downcast to int32 type data.")
                    raise ObsPyMSEEDError(msg)
            else:
                msg = "Unsupported data type %s in Stream[%i].data" % \
                    (trace.data.dtype, _i)
                raise Exception(msg)

        # Convert data if necessary, otherwise store references in list.
        if trace_attr['encoding'] == 1:
            # INT16 needs INT32 data type
            trace_data.append(trace.data.astype(np.int32, copy=True))
        else:
            trace_data.append(trace.data)

    # Do some final sanity checks and raise a warning if a file will be written
    # with more than one different encoding, record length or byte order.
    encodings = {_i['encoding'] for _i in trace_attributes}
    reclens = {_i['reclen'] for _i in trace_attributes}
    byteorders = {_i['byteorder'] for _i in trace_attributes}
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
        rec_handler = C.CFUNCTYPE(C.c_void_p, C.POINTER(C.c_char), C.c_int,
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

            size = C.sizeof(Blkt1001S)
            # Only timing quality matters here, other blockette attributes will
            # be filled by libmseed.msr_normalize_header
            blkt_value = pack("BBBB", trace_attr['timing_quality'],
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
        # Determine if a blockette 100 will be needed to represent the input
        # sample rate or if the sample rate in the fixed section of the data
        # header will suffice (see ms_genfactmult in libmseed/genutils.c)
        use_blkt_100 = False

        _factor = C.c_int16()
        _multiplier = C.c_int16()
        _retval = clibmseed.ms_genfactmult(
            trace.stats.sampling_rate, C.pointer(_factor),
            C.pointer(_multiplier))
        # Use blockette 100 if ms_genfactmult() failed.
        if _retval != 0:
            use_blkt_100 = True
        # Otherwise figure out if ms_genfactmult() found exact factors.
        # Otherwise write blockette 100.
        else:
            ms_sr = clibmseed.ms_nomsamprate(_factor.value, _multiplier.value)

            # It is also necessary if the libmseed calculated sampling rate
            # would result in a loss of accuracy - the floating point
            # comparision is on purpose here as it will always try to
            # preserve all accuracy.
            # Cast to float32 to not add blockette 100 for values
            # that cannot be represented with 32bits.
            if np.float32(ms_sr) != np.float32(trace.stats.sampling_rate):
                use_blkt_100 = True

        if use_blkt_100:
            size = C.sizeof(Blkt100S)
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
            mst.mst, rec_handler, None, trace_attr['reclen'],
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
            util._convert_datetime_to_mstime(trace.stats.starttime)
        self.mst.contents.endtime = \
            util._convert_datetime_to_mstime(trace.stats.endtime)
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
        C.memmove(self.mst.contents.datasamples, data.ctypes.data,
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
