# -*- coding: utf-8 -*-
"""
MSEED bindings to ObsPy core module.
"""
import ctypes as C  # NOQA
import fnmatch
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

from pymseed import MS3TraceList, sample_time


def _is_mseed3(file):
    """
    Checks whether a file is a valid MiniSEED file (version 2 or 3).

    :type file: str or file-like object
    :param file: MiniSEED file to be checked.
    :rtype: bool
    :return: ``True`` if a valid MiniSEED file.
    """
    try:
        from pymseed import MS3TraceList

        # Handle file-like objects vs file paths
        if hasattr(file, 'read'):
            # For file-like objects, we need to save to a temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.seek(0)
                temp_file.write(file.read())
                temp_filename = temp_file.name

            try:
                # Try to read with pymseed - if it works, it's valid MiniSEED
                MS3TraceList(temp_filename, unpack_data=False)
                return True
            except Exception:
                return False
            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(temp_filename)
                except OSError:
                    pass
        else:
            # For file paths, directly try to read
            try:
                MS3TraceList(str(file), unpack_data=False)
                return True
            except Exception:
                return False

    except ImportError:
        # If pymseed is not available, return False
        return False
    except Exception:
        return False


def __is_mseed3(fp, file_size):
    """
    Internal version of _is_mseed3 working only with open file-like object.
    """
    try:
        from pymseed import MS3TraceList
        import tempfile

        # Save current position
        initial_pos = fp.tell()

        try:
            # Read all data to temp file
            fp.seek(0)
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(fp.read())
                temp_filename = temp_file.name

            try:
                # Try to parse with pymseed
                MS3TraceList(temp_filename, unpack_data=False)
                return True
            except Exception:
                return False
            finally:
                # Clean up
                import os
                try:
                    os.unlink(temp_filename)
                except OSError:
                    pass

        finally:
            # Always restore file position
            fp.seek(initial_pos)

    except ImportError:
        return False
    except Exception:
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
                                      2: "D", 3: "Q",
                                      4: "M"}[record.pubversion]

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
                print(e)
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
            mstracelist = MS3TraceList(input_filename,
                                       unpack_data=not headonly,
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
                detected_byteorder = None
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
                               (UNSUPPORTED_ENCODINGS[encoding_code],
                                encoding_code))
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
            elif detected_byteorder:
                stats.mseed["byteorder"] = detected_byteorder
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
                    if hasattr(segment, 'samplecnt') else 0
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
                        calibration_type = cal_type \
                            if cal_type != -1 else False

                # Store detailed information
                if 'blkt1001' not in trace.stats.mseed:
                    trace.stats.mseed['blkt1001'] = AttribDict()
                trace.stats.mseed['blkt1001']['timing_quality'] =\
                    timing_quality
                trace.stats.mseed['calibration_type'] = calibration_type

            traces.append(trace)

    stream = Stream(traces=traces)

    # Apply time window filtering if specified
    if starttime or endtime:
        stream.trim(starttime=starttime, endtime=endtime)

    return stream.sort()


def _write_mseed3(stream, filename, encoding=None, byteorder=None,
                  record_length=512, format_version=3, flush_data=True,
                  **kwargs):
    """
    Write Stream object to MiniSEED 3 format using pymseed.

    :param stream: ObsPy Stream object to write
    :type stream: :class:`~obspy.core.stream.Stream`
    :param filename: Output filename or file-like object
    :type filename: str or file-like object
    :param encoding: Data encoding to use (default: auto-detect from trace)
    :type encoding: str or int, optional
    :param byteorder: Byte order for output (not used in MSEED3, kept for
     compatibility)
    :type byteorder: str, optional
    :param record_length: Record length in bytes (default: 512)
    :type record_length: int, optional
    :param format_version: MiniSEED format version (default: 3)
    :type format_version: int, optional
    :param flush_data: Whether to flush remaining data in buffers
     (default: True)
    :type flush_data: bool, optional
    """
    # Validate format version
    if format_version not in [2, 3]:
        raise ValueError(f"Unsupported format version: {format_version}."
                         f" Use 2 or 3.")

    # Validate record length
    if record_length not in VALID_RECORD_LENGTHS:
        raise ValueError(f"Invalid record length: {record_length}. "
                         f"Valid lengths are: {VALID_RECORD_LENGTHS}")

    # Determine if we're writing to a file or file-like object
    close_file = False
    temp_file = None

    if hasattr(filename, 'write'):
        # File-like object
        if hasattr(filename, 'mode') and 'b' not in filename.mode:
            raise ValueError("File must be opened in binary mode for"
                             " MiniSEED writing")
        file_handle = filename
    elif isinstance(filename, (str, bytes, os.PathLike)):
        # File path - open for writing
        file_handle = open(filename, 'wb')
        close_file = True
    else:
        raise TypeError(f"Invalid filename type: {type(filename)}")

    # Define encoding mappings from ObsPy to pymseed
    # ObsPy encoding name/code -> pymseed sample_type character
    encoding_map = {
        'ASCII': 'a',
        'INT16': 'i',
        'INT32': 'i',
        'FLOAT32': 'f',
        'FLOAT64': 'd',
        'STEIM1': 'i',
        'STEIM2': 'i',
        0: 'a',   # ASCII
        1: 'i',   # 16-bit integers
        3: 'i',   # 32-bit integers
        4: 'f',   # IEEE float32
        5: 'd',   # IEEE float64
        10: 'i',  # STEIM1
        11: 'i',  # STEIM2
    }

    def record_handler(buffer, handlerdata):
        """Write buffer to the file handle."""
        handlerdata["fh"].write(buffer)

    # Initialize MS3TraceList
    traces = MS3TraceList()

    total_samples = 0
    total_records = 0

    try:
        # Group traces by source identifier for efficient processing
        trace_groups = {}
        for trace in stream:
            # Create source ID from ObsPy trace stats
            if (hasattr(trace.stats, 'mseed3') and 'source_id' in
                    trace.stats.mseed3):
                # Use existing MSEED3 source ID if available
                sourceid = trace.stats.mseed3['source_id']
            else:
                # Create FDSN source ID from NSLC codes
                network = trace.stats.network or ""
                station = trace.stats.station or ""
                location = trace.stats.location or ""
                channel = trace.stats.channel or ""

                # For MSEED3, use FDSN format: FDSN:NET_STA_LOC_CHAN_SUB_SUBSUB
                sourceid = f"FDSN:{network}_{station}_{location}_{channel}__"

            if sourceid not in trace_groups:
                trace_groups[sourceid] = []
            trace_groups[sourceid].append(trace)

        # Process each trace group
        for sourceid, trace_list in trace_groups.items():
            # Sort traces by start time to ensure proper chronological order
            trace_list.sort(key=lambda t: t.stats.starttime)

            for trace in trace_list:
                # Determine encoding for this trace
                trace_encoding = encoding
                if trace_encoding is None:
                    # Auto-detect from trace stats
                    if (hasattr(trace.stats, 'mseed') and 'encoding' in
                            trace.stats.mseed):
                        obspy_encoding = trace.stats.mseed['encoding']
                        trace_encoding = encoding_map.get(obspy_encoding, 'i')
                    else:
                        # Default based on data type
                        if trace.data.dtype == np.float32:
                            trace_encoding = 'f'
                        elif trace.data.dtype == np.float64:
                            trace_encoding = 'd'
                        else:
                            trace_encoding = 'i'
                elif isinstance(trace_encoding, (int, str)):
                    # Convert encoding to pymseed format
                    trace_encoding = encoding_map.get(trace_encoding, 'i')

                # Convert start time to nanoseconds
                start_time_ns = int(trace.stats.starttime.timestamp
                                    * 1_000_000_000)

                # Handle data in chunks if trace is very large
                chunk_size = 10000  # Process in chunks to avoid memory issues
                data = trace.data
                sample_rate = trace.stats.sampling_rate

                # Process data in chunks
                for i in range(0, len(data), chunk_size):
                    chunk_data = data[i:i+chunk_size]
                    chunk_start_ns = sample_time(start_time_ns, i, sample_rate)

                    # Convert numpy array to list for pymseed
                    if isinstance(chunk_data, np.ndarray):
                        chunk_data = chunk_data.tolist()

                    # Add data to MS3TraceList
                    traces.add_data(
                        sourceid=sourceid,
                        data_samples=chunk_data,
                        sample_type=trace_encoding,
                        sample_rate=sample_rate,
                        start_time=chunk_start_ns,
                    )

                    # Pack records periodically to avoid excessive memory usage
                    if i > 0 and i % (chunk_size * 5) == 0:
                        packed_samples, packed_records = traces.pack(
                            record_handler,
                            handlerdata={"fh": file_handle},
                            format_version=format_version,
                            record_length=record_length,
                            flush_data=False,
                        )
                        total_samples += packed_samples
                        total_records += packed_records

        # Final pack to flush all remaining data
        packed_samples, packed_records = traces.pack(
            record_handler,
            handlerdata={"fh": file_handle},
            format_version=format_version,
            record_length=record_length,
            flush_data=flush_data,
        )
        total_samples += packed_samples
        total_records += packed_records

    finally:
        # Clean up file handle if we opened it
        if close_file and file_handle:
            file_handle.close()

        # Clean up temporary file if created
        if temp_file:
            try:
                os.unlink(temp_file)
            except OSError:
                pass

    return {"samples_written": total_samples, "records_written": total_records}


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
