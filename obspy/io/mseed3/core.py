# -*- coding: utf-8 -*-
"""
MSEED3 bindings to ObsPy core module.
"""

import os
from typing import IO, Union

import numpy as np
from pymseed import DataEncoding, MS3Record, MS3TraceList, nslc2sourceid, sourceid2nslc

from obspy import Stream, Trace, UTCDateTime
from obspy.core import Stats
from obspy.core.util import AttribDict

# Upper bound on bytes read for detecting if a file contains miniSEED.
_MSEED3_ISFORMAT_PROBE_BYTES = 1 << 20


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
    verbose: Union[bool, int] = 0,
    **kwargs: dict,
) -> Stream:
    """
    Reads a miniSEED file and returns an ObsPy Stream object.

    :param source: File path, in-memory buffer, or file-like object to be
        read. File-like objects must provide a ``read()`` method
        (e.g. ``io.BytesIO``, ``io.BufferedReader``).
    :type source: str, os.PathLike, bytes, bytearray, memoryview, or
        file-like (typing.IO[bytes])
    :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param starttime: Only read data samples after or at the start time.
    :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param endtime: Only read data samples before or at the end time.
    :param headonly: If True, do not decompress data samples. Default is False.
    :type details: bool
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
    :param verbose: If True, print verbose output at level 2.  If an integer,
        print verbose output at the given level. Default is False (aka level 0).
    :type verbose: bool, int
    :param twopass: If True, the data will be read in two passes.  During
        the first pass, the data will be read without unpacking the data samples.
        During the second pass, the data will be unpacked directly into a numpy array.
        This unpack-on-demand approach avoids duplicating the data in memory; the
        cost is reading the file twice.  The value of this tradeoff between memory
        and I/O will vary depending on the use case and system resources.  This
        option cannot be used in combination with sources that are not persistent,
        such as a file-like object that is not seekable (e.g. a network stream).
        Default is False.
    :type twopass: bool

    :rtype: :class:`~obspy.core.stream.Stream`
    :return: An ObsPy Stream object containing the data.
    """

    if sourceid and sourcename:
        raise ValueError(
            "Cannot specify both sourceid and sourcename. Use one or the other."
        )

    # Convert sourcename pattern to FDSN sourceid pattern
    if sourcename:
        parts = sourcename.split(".")
        # If four parts, convert to FDSN sourceid pattern
        if len(parts) == 4:
            sourceid = nslc2sourceid(*parts)
        # Special case: if the first part does not start with '*' it must be a network
        # code anchoring the pattern.  Use this information to convert.
        elif len(parts[0]) > 0 and not parts[0].startswith("*"):
            nslc = [parts[0]] + ["*"] * (4 - len(parts)) + list(parts[1:])
            sourceid = nslc2sourceid(*nslc)
        # Special case: if the last part does not end in '*' it must be a channel
        # code anchoring the pattern.  Use this information to convert.
        elif len(parts[-1]) > 0 and not parts[-1].endswith("*"):
            nslc = ["*"] * (4 - len(parts)) + list(parts)
            sourceid = nslc2sourceid(*nslc)
        # Otherwise no anchors, do a naive conversion to sourceid-like pattern
        else:
            sourceid = sourcename.replace(".", "_")

    # Common arguments for MS3TraceList factory functions
    common_kwargs = {
        "unpack_data": not headonly,
        "record_list": twopass,
    }

    # Set verbose level as integer level, otherwise if True, set to 2
    if isinstance(verbose, int):
        common_kwargs["verbose"] = verbose
    elif verbose:
        common_kwargs["verbose"] = 2

    if starttime:
        common_kwargs["starttime"] = str(starttime)
    if endtime:
        common_kwargs["endtime"] = str(endtime)
    if sourceid:
        common_kwargs["sourceid"] = sourceid

    try:
        if isinstance(source, (str, os.PathLike)):
            mstracelist = MS3TraceList.from_file(source, **common_kwargs)
        elif isinstance(source, (bytes, bytearray, memoryview)):
            mstracelist = MS3TraceList.from_buffer(source, **common_kwargs)
        elif callable(getattr(source, "read", None)):
            mstracelist = MS3TraceList.from_filelike(source, **common_kwargs)
        else:
            raise IOError(f"Unsupported input source: {type(source).__name__}")
    except IOError:
        raise
    except Exception as e:
        raise IOError(f"Error reading MSEED file {source}: {e}") from e

    traces = []

    # Iterate through each trace ID in the trace list
    for traceid in mstracelist:
        (network, station, location, channel) = sourceid2nslc(traceid.sourceid)

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
                stats.mseed3["number_of_records"] = segment.recordlist.recordcnt

            # If header-only mode create an empty trace
            if headonly:
                trace = Trace(data=np.array([]), header=stats)
            # If twopass and no data samples, unpack data samples into a numpy array
            elif twopass and not segment.datasamples:
                data = segment.create_numpy_array_from_recordlist()
                trace = Trace(data=data, header=stats)
            # Create a trace with the data samples
            else:
                data = segment.np_datasamples.copy()
                trace = Trace(data=data, header=stats)

            traces.append(trace)

    stream = Stream(traces=traces)

    # Apply sample-level time window filtering if specified
    if starttime or endtime:
        stream.trim(starttime=starttime, endtime=endtime)

    return stream.sort()


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
    :param encoding: Data encoding to use: ``ASCII`` (``0``)*, ``INT16`` (``1``),
        ``INT32`` (``3``), ``FLOAT32`` (``4``)*, ``FLOAT64`` (``5``)*,
        ``STEIM1`` (``10``) and ``STEIM2`` (``11``)*. If no encoding is given
        it will be derived from the dtype of the data and the appropriate
        default encoding (depicted with an asterix) will be chosen.
    :type encoding: str or int, optional
    :param max_record_length: Maximum record length in bytes (default: 4096)
    :type max_record_length: int, optional
    :param format_version: miniSEED format version (default: 3)
    :type format_version: int, optional
    :param overwrite: Overwrite the destination file if it exists (default: True)
    :type overwrite: bool, optional
    """

    # Determine pymseed encoding
    pymseed_encoding = None
    if encoding == "ASCII" or encoding == "TEXT" or encoding == 0:
        pymseed_encoding = DataEncoding.TEXT
    elif encoding == "INT16" or encoding == 1:
        pymseed_encoding = DataEncoding.INT16
    elif encoding == "INT32" or encoding == 3:
        pymseed_encoding = DataEncoding.INT32
    elif encoding == "FLOAT32" or encoding == 4:
        pymseed_encoding = DataEncoding.FLOAT32
    elif encoding == "FLOAT64" or encoding == 5:
        pymseed_encoding = DataEncoding.FLOAT64
    elif encoding == "STEIM1" or encoding == 10:
        pymseed_encoding = DataEncoding.STEIM1
    elif encoding == "STEIM2" or encoding == 11:
        pymseed_encoding = DataEncoding.STEIM2
    elif encoding is not None:
        raise ValueError(
            f"Unsupported encoding: {encoding}. Use ASCII/TEXT, INT16, INT32, FLOAT32, FLOAT64, STEIM1, or STEIM2."
        )

    # Initialize MS3Record with common header fields
    msrecord = MS3Record()
    msrecord.reclen = max_record_length
    msrecord.formatversion = format_version

    first_path_write = True

    for trace in stream:
        # Create source ID from codes
        network = trace.stats.network
        station = trace.stats.station
        location = trace.stats.location or ""
        channel = trace.stats.channel

        if not network or not station or not channel:
            raise ValueError("Network, station, and channel codes are required.")

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
                    "int64 data only supported when writing miniSEED if it can be downcast to int32 type data."
                )
        else:
            raise ValueError(f"Unsupported data type: {data.dtype}")

        msrecord.encoding = (
            pymseed_encoding if pymseed_encoding is not None else type_default_encoding
        )

        # Write records using a zero-copy view of the data samples
        with msrecord.with_datasamples(data, sample_type_code):
            # Write to a path-like destination
            if isinstance(destination, (str, os.PathLike)):
                msrecord.to_file(destination, overwrite=overwrite and first_path_write)
                first_path_write = False
            # Write to a file-like destination
            else:
                for record in msrecord.generate():
                    destination.write(record)


if __name__ == "__main__":
    import doctest

    doctest.testmod(exclude_empty=True)
