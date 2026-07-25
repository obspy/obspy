# -*- coding: utf-8 -*-
"""
MSEED3 bindings to ObsPy core module.
"""

import os
import warnings
from typing import IO, Union

import numpy as np
from pymseed import (
    DataEncoding,
    MS3Record,
    MS3TraceList,
    nslc2sourceid,
    sourceid2nslc,
)

from obspy import Stream, Trace, UTCDateTime
from obspy.core import Stats
from obspy.core.util import AttribDict

# Upper bound on bytes read for detecting if a file contains miniSEED.
_MSEED3_ISFORMAT_PROBE_BYTES = 1 << 20


def _supports_buffer_protocol(obj) -> bool:
    """
    Return True if ``obj`` exposes the Python buffer protocol.

    Used to route inputs through pymseed's ``from_buffer`` (zero-copy)
    rather than ``from_filelike``. Probes ``memoryview(obj)`` cheaply;
    accepts bytes, bytearray, memoryview, numpy arrays, mmap objects,
    ctypes buffers, and similar.
    """
    try:
        memoryview(obj)
    except TypeError:
        return False
    return True


def _describe_source(source) -> str:
    """
    Return a short description of a read source, for use in error messages.

    A path is named directly, as is an open file that reports one. Anything
    else is described by type and size only, so that an error message never
    interpolates the contents of a buffer.
    """
    if isinstance(source, (str, os.PathLike)):
        return f"file '{os.fspath(source)}'"

    name = getattr(source, "name", None)
    if isinstance(name, (str, bytes, os.PathLike)):
        return f"file '{os.fsdecode(name)}'"

    try:
        return f"{type(source).__name__} of {memoryview(source).nbytes} bytes"
    except TypeError:
        return type(source).__name__


def _is_mseed3(
    source: Union[str, os.PathLike, bytes, bytearray, memoryview, IO[bytes]],
) -> bool:
    """
    Checks whether data at the start of ``source`` is readable as miniSEED
    (version 2 or 3) by parsing the first record with pymseed.

    Only the first record header is parsed; samples are not unpacked and CRC
    validation is skipped for a lightweight format check.

    :type source: str, os.PathLike, bytes-like, or file-like object
    :param source: miniSEED data to be checked.
    :rtype: bool
    :return: ``True`` if the first record parses successfully.
    """
    parse_kwargs = {"unpack_data": False, "validate_crc": False}

    try:
        if isinstance(source, (bytes, bytearray, memoryview)):
            chunk = bytes(source[:_MSEED3_ISFORMAT_PROBE_BYTES])
            return MS3Record.parse(buffer=chunk, **parse_kwargs) is not None
        if hasattr(source, "read"):
            chunk = source.read(_MSEED3_ISFORMAT_PROBE_BYTES)
            return MS3Record.parse(buffer=chunk, **parse_kwargs) is not None
        for _ in MS3Record.from_file(
            os.fspath(source),
            end_byte_offset=_MSEED3_ISFORMAT_PROBE_BYTES,
            **parse_kwargs,
        ):
            return True
        return False
    except Exception:
        # Any parse/IO error means "not mseed"; format probes must not raise.
        return False


def _read_mseed3(
    source: Union[str, os.PathLike, bytes, bytearray, memoryview, IO[bytes]],
    starttime: Union[UTCDateTime, None] = None,
    endtime: Union[UTCDateTime, None] = None,
    headonly: bool = False,
    sourceid: Union[str, None] = None,
    sourcename: Union[str, None] = None,
    twopass: bool = False,
    details: bool = False,
    verbose: Union[bool, int] = 0,
    **kwargs: dict,
) -> Stream:
    """
    Reads a miniSEED file and returns an ObsPy Stream object.

    :param source: File path, in-memory buffer, or file-like object to be
        read. Any object supporting the Python buffer protocol (bytes,
        bytearray, memoryview, numpy.ndarray, mmap, etc.) is read directly
        (zero-copy); otherwise a ``read()`` method is required (e.g.
        ``io.BytesIO``, ``io.BufferedReader``).
    :type source: str, os.PathLike, bytes-like (supports the buffer
        protocol), or file-like (typing.IO[bytes])
    :param starttime: Only read data samples after or at the start time.
    :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param endtime: Only read data samples before or at the end time.
    :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param headonly: If True, do not decompress data samples. Default is False.
    :type headonly: bool
    :param sourceid: Only read data with matching FDSN Source ID.
        Value can contain wildcards "?" and "*" and other common globbing
        patterns, e.g. "FDSN:BW_UH2_*", "*_L_H_[EN]"
        Defaults to ``None``.
    :type sourceid: str
    :param sourcename: Only read data with matching SEED ID. Value can contain
        wildcards "?" and "*", e.g. "BW.UH2.*" or "*.??Z".
        LIMITATION: Unanchored patterns with channel codes, e.g. "*.LH*",
        are not supported.  Use "*.LH?" or sourceid patterns instead.
        Defaults to ``None``.
    :type sourcename: str
    :param twopass: If True, the data will be read in two passes.  During
        the first pass, the data are read without unpacking the data samples.
        During the second pass, data are unpacked directly into numpy arrays.
        This unpack-on-demand approach avoids duplicating the data in memory;
        the cost is reading the file twice.  The value of this tradeoff between
        memory and I/O will vary depending on the use case and system
        resources.  The second pass re-reads the source, which is only possible
        for a file path or an in-memory buffer.  For a file-like object the
        data are read in a single pass instead, with a warning.
        Default is False.
    :type twopass: bool
    :param details: If True, read additional information: timing quality,
        publication versions, and encodings. Stored in the mseed3 stats
        dictionary of each trace as run-length-deduplicated lists, which
        retains the order that the values were encountered.
        Default is False.
    :type details: bool
    :param verbose: If True, print verbose output at level 2.  If an integer,
        print verbose output at the given level. Default is False (aka 0).
    :type verbose: bool, int

    :rtype: :class:`~obspy.core.stream.Stream`
    :return: An ObsPy Stream object containing the data.
    """

    if sourceid and sourcename:
        raise ValueError(
            "Cannot specify both sourceid and sourcename. Use only one."
        )

    # Convert sourcename pattern to FDSN sourceid pattern, best effort
    if sourcename:
        parts = sourcename.split(".")
        # If four parts, convert to FDSN sourceid pattern
        if len(parts) == 4:
            sourceid = nslc2sourceid(*parts)
        # Front-anchored: known network, optionally station and location;
        # missing trailing components become wildcards.
        elif len(parts) > 1 and not parts[0].startswith("*"):
            nslc = list(parts) + ["*"] * (4 - len(parts))
            sourceid = nslc2sourceid(*nslc)
        # End-anchored: known channel (+ optionally location/station);
        # missing leading components become wildcards.
        elif len(parts) > 1 and not parts[-1].endswith("*"):
            nslc = ["*"] * (4 - len(parts)) + list(parts)
            sourceid = nslc2sourceid(*nslc)
        # Otherwise no anchors, do a naive conversion to sourceid-like pattern
        else:
            sourceid = sourcename.replace(".", "_")

    # Select the reader for this source. A file-like object is consumed as it
    # is read, unlike a path or buffer, which can be re-read for a second pass.
    if isinstance(source, (str, os.PathLike)):
        read_tracelist, rereadable = MS3TraceList.from_file, True
    elif _supports_buffer_protocol(source):
        read_tracelist, rereadable = MS3TraceList.from_buffer, True
    elif callable(getattr(source, "read", None)):
        read_tracelist, rereadable = MS3TraceList.from_filelike, False
    else:
        raise IOError(f"Unsupported input source: {type(source).__name__}")

    # Common arguments for MS3TraceList factory functions
    common_kwargs = {
        "unpack_data": not headonly,
    }

    # Without a re-readable source the record list has nothing to unpack
    # samples from in the second pass, so read in a single pass instead.
    if twopass and not rereadable:
        warnings.warn(
            "twopass reading requires a file path or in-memory buffer, "
            f"reading {type(source).__name__} in a single pass instead"
        )
        twopass = False

    # If twopass, read the record list first and unpack data later
    if twopass:
        common_kwargs["record_list"] = True
        common_kwargs["unpack_data"] = False

    # Details requires a record list
    if details:
        common_kwargs["record_list"] = True

    # Set verbose level to 2 if True, otherwise as integer level
    if isinstance(verbose, bool):
        common_kwargs["verbose"] = 2 if verbose else 0
    elif isinstance(verbose, int):
        common_kwargs["verbose"] = verbose

    if starttime:
        common_kwargs["starttime"] = str(starttime)
    if endtime:
        common_kwargs["endtime"] = str(endtime)
    if sourceid:
        common_kwargs["sourceid"] = sourceid

    try:
        mstracelist = read_tracelist(source, **common_kwargs)
    except IOError:
        raise
    except Exception as e:
        raise IOError(
            f"Error reading miniSEED from {_describe_source(source)}: {e}"
        ) from e

    traces = []

    # Iterate through each trace ID in the trace list
    for traceid in mstracelist:
        try:
            (network, station, location, channel) = sourceid2nslc(
                traceid.sourceid
            )
        except ValueError:
            network = station = location = channel = ""

        # Process each continuous segment for this trace ID
        for segment in traceid:
            # Create Stats object
            stats = Stats()
            stats.network = network
            stats.station = station
            stats.location = location
            stats.channel = channel
            stats.sampling_rate = segment.samprate
            stats.npts = segment.samplecnt

            # Convert segment start time to UTCDateTime
            stats.starttime = UTCDateTime(ns=segment.starttime)

            # Add mseeds stats dictionary to stats object
            stats.mseed3 = AttribDict()
            stats.mseed3["source_id"] = traceid.sourceid
            if segment.recordlist:
                stats.mseed3["number_of_records"] = (
                    segment.recordlist.recordcnt
                )

            # Summarize record details
            if details and segment.recordlist:
                # Walk record list and collect run-length-deduplicated values:
                # a value is only appended when it differs from the previous.
                timing_qualities = []
                publication_versions = []
                encodings = []
                for record_ptr in segment.recordlist:
                    record = record_ptr.record
                    timing_quality = record.get_extra_header(
                        "/FDSN/Time/Quality"
                    )
                    if timing_quality is not None and timing_quality != (
                        timing_qualities[-1] if timing_qualities else object()
                    ):
                        timing_qualities.append(timing_quality)
                    pubver = record.pubversion
                    if pubver != (
                        publication_versions[-1]
                        if publication_versions
                        else object()
                    ):
                        publication_versions.append(pubver)
                    encoding = record.encoding_str()
                    if encoding != (encodings[-1] if encodings else object()):
                        encodings.append(encoding)

                stats.mseed3["timing_qualities"] = timing_qualities
                stats.mseed3["publication_versions"] = publication_versions
                stats.mseed3["encodings"] = encodings

            # If header-only mode create an empty trace
            if headonly:
                trace = Trace(data=np.array([]), header=stats)
            # If twopass & no samples, unpack data samples into a numpy array
            elif twopass and not segment.datasamples:
                data = segment.create_numpy_array_from_recordlist()
                trace = Trace(data=data, header=stats)
            # Create a trace with the data samples
            else:
                data = segment.np_datasamples.copy()
                trace = Trace(data=data, header=stats)

            traces.append(trace)

    stream = Stream(traces=traces)

    # If time window is specified, the data have already been limited
    # at the record-level; trim() applies sample-level filtering.
    if starttime or endtime:
        stream.trim(starttime=starttime, endtime=endtime)

    return stream


# Map encoding aliases (case-sensitive strings and SEED integer codes) to
# pymseed DataEncoding values that are writable.
_ENCODING_ALIASES = {
    "ASCII": DataEncoding.TEXT,
    "TEXT": DataEncoding.TEXT,
    0: DataEncoding.TEXT,
    "INT16": DataEncoding.INT16,
    1: DataEncoding.INT16,
    "INT32": DataEncoding.INT32,
    3: DataEncoding.INT32,
    "FLOAT32": DataEncoding.FLOAT32,
    4: DataEncoding.FLOAT32,
    "FLOAT64": DataEncoding.FLOAT64,
    5: DataEncoding.FLOAT64,
    "STEIM1": DataEncoding.STEIM1,
    10: DataEncoding.STEIM1,
    "STEIM2": DataEncoding.STEIM2,
    11: DataEncoding.STEIM2,
}


def _write_mseed3(
    stream: Stream,
    destination: Union[str, os.PathLike, IO[bytes]],
    encoding: Union[str, int, None] = None,
    max_record_length: int = 4096,
    format_version: int = 3,
    overwrite: bool = True,
    **kwargs,
) -> None:
    """
    Write Stream object to miniSEED version 2 or 3 format.

    :param stream: ObsPy Stream object to write
    :type stream: :class:`~obspy.core.stream.Stream`
    :param destination: Output filename or file-like object
    :type destination: str or file-like object
    :param encoding: Data encoding to use: ``ASCII`` (``0``)*,
        ``INT16`` (``1``), ``INT32`` (``3``), ``FLOAT32`` (``4``)*,
        ``FLOAT64`` (``5``)*,``STEIM1`` (``10``) and ``STEIM2`` (``11``)*.
        If no encoding is given it will be derived from the dtype of the data
        and the appropriate default encoding (depicted with an asterix)
        will be chosen.
    :type encoding: str or int, optional
    :param max_record_length: Maximum record length in bytes (default: 4096)
    :type max_record_length: int, optional
    :param format_version: miniSEED format version (default: 3)
    :type format_version: int, optional
    :param overwrite: Overwrite destination file if it exists (default: True)
    :type overwrite: bool, optional
    """

    # Determine pymseed encoding. ``None`` defers to dtype-derived defaults.
    if encoding is None:
        pymseed_encoding = None
    elif encoding in _ENCODING_ALIASES:
        pymseed_encoding = _ENCODING_ALIASES[encoding]
    else:
        raise ValueError(
            f"Unsupported encoding: {encoding!r}. Use ASCII/TEXT (0), "
            f"INT16 (1), INT32 (3), FLOAT32 (4), FLOAT64 (5), "
            f"STEIM1 (10), or STEIM2 (11)."
        )

    # Initialize MS3Record with common header fields
    msrecord = MS3Record()
    msrecord.reclen = max_record_length
    msrecord.formatversion = format_version

    first_path_write = True

    for trace in stream:
        # Create source ID from codes
        network = trace.stats.network or ""
        station = trace.stats.station or ""
        location = trace.stats.location or ""
        channel = trace.stats.channel or "__"

        msrecord.sourceid = nslc2sourceid(network, station, location, channel)

        # Start time in nanoseconds
        msrecord.starttime = trace.stats.starttime.ns

        # Sample rate in Hz
        msrecord.samprate = trace.stats.sampling_rate

        # Determine pymseed sample type and type specific default encoding
        sample_type_code = None
        type_default_encoding = None

        data = trace.data

        # Normalize non-native byte order to native byte order
        if data.dtype.byteorder not in ("=", "|"):
            data = data.astype(data.dtype.newbyteorder("="))

        if data.dtype == np.int32:
            sample_type_code = "i"
            type_default_encoding = DataEncoding.STEIM2
        elif data.dtype == np.float32:
            sample_type_code = "f"
            type_default_encoding = DataEncoding.FLOAT32
        elif data.dtype == np.float64:
            sample_type_code = "d"
            type_default_encoding = DataEncoding.FLOAT64
        elif data.dtype == np.dtype("|S1"):
            sample_type_code = "t"
            type_default_encoding = DataEncoding.TEXT
        elif data.dtype == np.int16:
            sample_type_code = "i"
            type_default_encoding = DataEncoding.STEIM2
            data = data.astype(np.int32, copy=True)
        elif data.dtype == np.int64:
            ii32 = np.iinfo(np.int32)
            if data.min() >= ii32.min and data.max() <= ii32.max:
                sample_type_code = "i"
                type_default_encoding = DataEncoding.STEIM2
                data = data.astype(np.int32, copy=True)
            else:
                raise ValueError(
                    "int64 data only supported when writing miniSEED "
                    "if it can be downcast to int32 type data."
                )
        else:
            raise ValueError(f"Unsupported data type: {data.dtype}")

        msrecord.encoding = (
            pymseed_encoding
            if pymseed_encoding is not None
            else type_default_encoding
        )

        # Write records using a zero-copy view of the data samples
        with msrecord.with_datasamples(data, sample_type_code):
            # Write to a path-like destination
            if isinstance(destination, (str, os.PathLike)):
                msrecord.to_file(
                    destination, overwrite=overwrite and first_path_write
                )
                first_path_write = False
            # Write to a file-like destination
            else:
                for record in msrecord.generate():
                    destination.write(record)


if __name__ == "__main__":
    import doctest

    doctest.testmod(exclude_empty=True)
