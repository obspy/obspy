# -*- coding: utf-8 -*-
"""
MSEED bindings to ObsPy core module.
"""
import ctypes as C  # NOQA
import fnmatch
import io
import os
import struct

import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core import Stats
from obspy.core.util import AttribDict
from obspy.core.util import NATIVE_BYTEORDER

from ..mseed import ObsPyMSEEDFilesizeTooSmallError, ObsPyMSEEDError
from ..mseed.headers import (ENCODINGS, UNSUPPORTED_ENCODINGS,
                             VALID_RECORD_LENGTHS)

from pymseed import MS3TraceList


def _is_mseed3(file):
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
            return __is_mseed3(fh, file_size=file_size)
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
            return __is_mseed3(file, file_size)
        finally:
            # Reset pointer.
            file.seek(initial_pos, 0)


def __is_mseed3(fp, file_size):  # NOQA
    """
    Internal version of _is_mseed3 working only with open file-like object.
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
            return __is_mseed3(fp=fp, file_size=file_size)
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
            return __is_mseed3(fp=fp, file_size=file_size)
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


def _get_mseed3_input_file(filename):
    """
    Utility function to handle different types of input (filename, file object,
     BytesIO)
    and return a suitable filename for mseedlib functions.

    :param filename: Input source, can be a filename, file object, or BytesIO
     object.
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

    else:
        input_filename = str(filename)

    return input_filename


def extract_mseed3_metadata(record, stats):
    """
    Extract all available metadata from a mseedlib record and add it to ObsPy
    stats object.

    :param record: mseedlib record object (MS3Record or segment from MSTraceID)
    :param stats: ObsPy Stats object to populate
    :return: Updated stats object
    """
    # Create mseed and mseed3 attribute dictionaries if they don't exist

    if not hasattr(stats, 'mseed'):
        stats.mseed = {}

    # Define attribute mappings for common MSEED attributes
    # Format: 'record_attribute': ('stats_location', 'stats_key')
    # where stats_location can be 'root', 'mseed', or 'mseed3'
    common_mseed_attrs = {
        'samplerate': ('root', 'sampling_rate'),
        'reclen': ('mseed', 'record_length'),
        'samplecnt': ('root', 'npts'),
        'sampletype': ('mseed', 'encoding'),
        'quality': ('mseed', 'dataquality'),
    }

    # MSEED3-specific attributes
    mseed3_attrs = {
        'sourceid': ('mseed3', 'source_id'),
        'crc': ('mseed3', 'crc'),
        'extra_headers': ('mseed3', 'extra_headers'),
        'recid': ('mseed3', 'record_id'),
        'timingquality': ('mseed3', 'timing_quality'),
        'flags': ('mseed3', 'flags'),
    }

    # MSEED2-specific attributes
    mseed2_attrs = {
        'blockettecount': ('mseed', 'blockette_count'),
        'network': ('root', 'network'),
        'station': ('root', 'station'),
        'location': ('root', 'location'),
        'channel': ('root', 'channel'),
        'sequence_number': ('mseed', 'sequence_number'),
        'act_flags': ('mseed', 'activity_flags'),
        'io_flags': ('mseed', 'io_flags'),
        'dq_flags': ('mseed', 'data_quality_flags'),
    }

    # Process common MSEED attributes
    for attr_name, (target_location, target_key) in common_mseed_attrs.items():
        if hasattr(record, attr_name):
            value = getattr(record, attr_name)
            if target_location == 'root':
                setattr(stats, target_key, value)
            elif target_location == 'mseed':
                stats.mseed[target_key] = value

    # Handle byte order detection (more complex logic)
    try:
        # Check multiple possible attribute names for byte order
        byteorder_attrs = ['byteorder', 'byte_order']
        byteorder_found = False

        for attr_name in byteorder_attrs:
            if hasattr(record, attr_name):
                stats.mseed['byteorder'] = getattr(record, attr_name)
                byteorder_found = True
                break

        if not byteorder_found and hasattr(record, 'header'):
            for attr_name in byteorder_attrs:
                if hasattr(record.header, attr_name):
                    stats.mseed['byteorder'] = getattr(record.header,
                                                       attr_name)
                    byteorder_found = True
                    break

        if not byteorder_found:
            # Try to determine from record data/header structure
            if hasattr(record, 'data') and len(record.data) >= 22:
                from struct import unpack
                # Check year field in MSEED header (bytes 20-21)
                # big-endian
                year_be = unpack('>H', record.data[20:22])[0]
                # little-endian
                year_le = unpack('<H', record.data[20:22])[0]
                # The year should be reasonable (1970-2070 range)
                if 1970 <= year_be <= 2070:
                    stats.mseed['byteorder'] = '>'
                elif 1970 <= year_le <= 2070:
                    stats.mseed['byteorder'] = '<'
                else:
                    stats.mseed['byteorder'] = '>'
            else:
                stats.mseed['byteorder'] = '>'

    except (AttributeError, IndexError, struct.error):
        # Fallback to big-endian if detection fails
        stats.mseed['byteorder'] = '>'  # Default to big-endian for MSEED

    # Handle special case for pubversion -> dataquality mapping
    if hasattr(record, 'pubversion'):
        stats.mseed['dataquality'] = {0: "X", 1: "R",
                                      2: "D", 3: "Q", 4: "M"}[record.pubversion]

    # Check if this is MSEED3 format
    is_mseed3 = hasattr(record, 'formatversion') and record.formatversion >= 3

    if is_mseed3:
        if hasattr(record, 'formatversion'):
            stats.mseed['miniseed_version'] = record.formatversion

        # Create mseed3 dict and set publication_version
        if not hasattr(stats, 'mseed3'):
            stats.mseed3 = {}
        if hasattr(record, 'pubversion'):
            stats.mseed3['publication_version'] = record.pubversion

        # Process MSEED3-specific attributes
        for attr_name, (target_location, target_key) in mseed3_attrs.items():
            if hasattr(record, attr_name):
                value = getattr(record, attr_name)
                if target_location == 'mseed3':
                    stats.mseed3[target_key] = value

        # Handle special time conversion for pubtime
        if hasattr(record, 'pubtime'):
            pub_ns = record.pubtime
            pub_sec = pub_ns / 1_000_000_000
            stats.mseed3['publication_time'] = UTCDateTime(pub_sec)

    # Check if this is MSEED2 format
    is_mseed2 = hasattr(record, 'formatversion') and record.formatversion < 3

    if is_mseed2:
        # Process MSEED2-specific attributes
        for attr_name, (target_location, target_key) in mseed2_attrs.items():
            if hasattr(record, attr_name):
                value = getattr(record, attr_name)
                if target_location == 'root':
                    setattr(stats, target_key, value)
                elif target_location == 'mseed':
                    stats.mseed[target_key] = value

    # Add any additional attributes available that we didn't explicitly handle
    # This ensures we don't miss any metadata in future mseedlib versions
    # for attr_name in dir(record):
    #     # Skip private attributes, methods, and ones we've already handled
    #     if (attr_name.startswith('_') or
    #             attr_name in ['starttime', 'samplerate', 'reclen',
    #                           'samplecnt',
    #                           'sampletype', 'formatversion',
    #                           'sourceid', 'crc', 'extra_headers', 'recid',
    #                           'timingquality', 'flags', 'blockettecount',
    #                           'network', 'station', 'location', 'channel',
    #                           'msr',
    #                           'sequence_number', 'act_flags', 'io_flags',
    #                           'dq_flags',
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
            seed_id = (f"{parts[0]}.{parts[1]}.{parts[2]}."
                       f"{parts[3]}{parts[4]}{parts[5]}")
        else:
            seed_id = sourceid
    else:
        # Convert underscores to dots
        seed_id = sourceid.replace('_', '.')

    # Use fnmatch for wildcard matching

    return fnmatch.fnmatch(seed_id, sourcename)


def _read_mseed3(filename, starttime=None, endtime=None, headonly=False,
                sourcename=None, reclen=None, details=False,
                header_byteorder=None, verbose=None, **kwargs):
    """
    Reads a MSEED3 file and returns an ObsPy Stream object.

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
            warnings.warn(f"Invalid header_byteorder '{header_byteorder}',"
                          f" using autodetection")
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
                msg = ("The smallest possible mini-SEED record is made up of"
                       " 128 bytes. The passed buffer or file contains"
                       " only %i." % length)
                raise ObsPyMSEEDFilesizeTooSmallError(msg)

    except (OSError, IOError):
        raise
    except Exception as e:
        raise IOError(f"Error accessing file {filename}: {str(e)}")

    try:
        input_filename = _get_mseed3_input_file(filename)

        # Add validation that the file was properly created/accessed
        if not input_filename or not os.path.exists(input_filename):
            raise IOError("Failed to prepare input file for reading")

        # Initialize with better error context
        try:
            mstracelist = MS3TraceList(input_filename, unpack_data=not headonly,
                                       record_list=True)
        except Exception as e:
            # More specific error classification
            error_msg = str(e).lower()
            if any(phrase in error_msg for phrase in ['not a valid',
                                                      'invalid format',
                                                      'corrupt']):
                raise ObsPyMSEEDError('Not a valid (Mini-)SEED file')
            elif 'permission' in error_msg or 'access' in error_msg:
                raise IOError(f"Permission denied accessing file:"
                              f" {filename}")
            elif 'memory' in error_msg or 'allocation' in error_msg:
                raise MemoryError(f"Insufficient memory to read file:"
                                  f" {filename}")
            elif 'No miniSEED data detected' in error_msg:
                raise IOError(f"No MiniSEED data record found in file."
                              f" {filename}")
            else:
                raise IOError(f"Error reading MSEED file: {str(e)}")

    except (ObsPyMSEEDError, IOError, MemoryError):
        raise
    except Exception as e:
        raise IOError(f"Unexpected error reading MSEED file {filename}:"
                      f" {str(e)}")

    traces = []

    # Iterate through each trace ID in the trace list
    for traceid in mstracelist.traceids():
        # Parse network, station, location, channel from sourceid
        sourceid = traceid.sourceid
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
            # Handle other ID formats or try to parse based on underscore
            # separators
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
                stats = extract_mseed3_metadata(rec.msr, stats)

                # Get byteorder from the first record
                if record_count == 1:
                    # Try to get byteorder from MSRecord structure
                    if hasattr(rec.msr, 'byteorder'):
                        # MSRecord byteorder: 0 = little-endian, 1 = big-endian
                        detected_byteorder = '<' \
                            if rec.msr.byteorder == 0 else '>'
                    elif hasattr(rec.msr, 'Blkt1000'):
                        # Extract from Blockette 1000 if available
                        blkt1000 = rec.msr.Blkt1000
                        if blkt1000:
                            # Blockette 1000 word order: 0 = little, 1 = big
                            word_order = getattr(blkt1000, 'word_order', None)
                            if word_order is not None:
                                detected_byteorder = '<' \
                                    if word_order == 0 else '>'

                    # Get record length from first record
                    if hasattr(rec.msr, 'reclen'):
                        record_length = rec.msr.reclen

                # Validate encoding
                encoding_code = rec.msr.encoding
                if encoding_code not in ENCODINGS:
                    if encoding_code in UNSUPPORTED_ENCODINGS:
                        msg = (("Encoding '%s' (%i) is not supported by ObsPy."
                               " Please send the file to the ObsPy developers"
                               " so that we can add "
                               "support for it.") %
                       (UNSUPPORTED_ENCODINGS[encoding_code], encoding_code))
                        raise ValueError(msg)
                    else:
                        msg = ("Encoding '%i' is not a valid MiniSEED"
                               "encoding.") % encoding_code
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
                trace.stats.npts = segment.samplecnt\
                    if hasattr(segment,'samplecnt') else 0
                traces.append(trace)
                continue

            # Get the data samples
            if segment.numsamples:
                data = np.array(segment.datasamples)
            else:
                # If no data samples but we know the count, create zeros
                sample_count = segment.samplecnt\
                    if hasattr(segment, 'samplecnt') else 0
                data = np.zeros(sample_count)

            # Handle data byteorder properly
            if len(data) > 0:
                # Get the original data byteorder
                original_data_byteorder = data.dtype.byteorder
                if original_data_byteorder == '=':
                    original_data_byteorder = NATIVE_BYTEORDER

                # If user specified a byteorder that differs from data,
                # we may need to swap
                if (processed_header_byteorder is not None and
                        processed_header_byteorder != original_data_byteorder):
                    # Convert data to the requested byteorder
                    if (processed_header_byteorder == '<' and
                            original_data_byteorder == '>'):
                        data = data.byteswap().newbyteorder('<')
                    elif (processed_header_byteorder == '>' and
                          original_data_byteorder == '<'):
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


def _write_mseed3(stream, filename, encoding=None, byteorder=None, **kwargs):
    """
    Write Stream object to MiniSEED 3 format.

    :param stream: ObsPy Stream object to write
    :param filename: Output filename
    :param encoding: Data encoding to use
    :param byteorder: Byte order for output
    """
    raise NotImplementedError(
        "MiniSEED 3 writing is not yet implemented. "
        "This functionality requires additional development."
    )


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
