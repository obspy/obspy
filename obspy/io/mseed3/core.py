# -*- coding: utf-8 -*-
"""
MSEED3 bindings to ObsPy core module.
"""
import io
import os
import warnings
from pathlib import Path
from struct import pack

from obspy import Stream, Trace, UTCDateTime
from obspy.core import Stats
from obspy.core.compatibility import from_buffer
from obspy import ObsPyException, ObsPyReadingError

from simplemseed import (
    unpackMSeed3FixedHeader,
    FDSNSourceId,
    FIXED_HEADER_SIZE,
    readMSeed3Records,
    FDSNSourceId,
    MSeed3Record,
    MSeed3Header,
    canDecompress,
)


def _is_mseed3(filename):
    """
    Checks whether a file is mseed3 or not.

    :type filename: str
    :param filename: mseed3 file to be checked.
    :rtype: bool
    :return: ``True`` if a mseed3 file.

    This method reads the first three bytes of the file and checks
    whether it is 'M' 'S' 3, then reads the fixed header and performs
    simple sanity checks on the start time fields.

    """
    # Open filehandler or use an existing file like object.
    if not hasattr(filename, "read"):
        file_size = os.path.getsize(filename)
        with io.open(filename, "rb") as fh:
            return __internal_is_mseed3(fh, file_size=file_size)
    else:
        initial_pos = filename.tell()
        try:
            if hasattr(filename, "getbuffer"):
                file_size = filename.getbuffer().nbytes
            try:
                file_size = os.fstat(filename.fileno()).st_size
            except Exception:
                _p = filename.tell()
                filename.seek(0, 2)
                file_size = filename.tell()
                filename.seek(_p, 0)
            return __internal_is_mseed3(filename, file_size)
        finally:
            # Reset pointer.
            filename.seek(initial_pos, 0)


def __internal_is_mseed3(fp, file_size):
    """
    Internal version of _is_mseed3 working only with open file-like object.
    """
    if file_size < FIXED_HEADER_SIZE:
        return False
    headerBytes = fp.read(FIXED_HEADER_SIZE)
    # File has less than FIXED_HEADER_SIZE characters
    if len(headerBytes) != FIXED_HEADER_SIZE:
        return False
    # M is ascii 77 and S is 83
    if headerBytes[0] != 77 or headerBytes[1] != 83 or headerBytes[2] != 3:
        return False
    header = unpackMSeed3FixedHeader(headerBytes)
    return header.sanityCheck()


def _read_mseed3(
    mseed3_file,
    starttime=None,
    endtime=None,
    headonly=False,
    matchsid=None,
    merge=True,
    verbose=None,
    **kwargs,
):
    """
    Reads a mseed3 file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param mseed3_file: Filename or open file like object that contains the
        binary mseed3 data. Any object that provides a read() method will be
        considered to be a file like object.
    :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param starttime: Only read data records after or at the start time.
    :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param endtime: Only read data records before or at the end time.
    :param headonly: Determines whether or not to unpack the data or just
        read the headers.
    :type matchsid: str
    :param matchsid: Only read data with matching FDSN Source Id
        (can be regular expression, e.g. "BW_UH2.*" or ".*_._._Z").
        Defaults to ``None`` .

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/casee_two.ms3")
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    CO.CASEE.00.HHZ | 2023-06-17T04:53:50.008392Z - 2023-06-17T04:53:55.498392Z | 100.0 Hz, 550 samples

    >>> from obspy import UTCDateTime
    >>> st = read("/path/to/casee_two.ms3",
    ...           starttime=UTCDateTime("2010-06-20T00:00:01"),
    ...           matchsid="_._H_Z")
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    CO.CASEE.00.HHZ | 2023-06-17T04:53:50.008392Z - 2023-06-17T04:53:55.498392Z | 100.0 Hz, 550 samples
    """

    if isinstance(mseed3_file, Path):
        mseed3_file = str(mseed3_file)
    # Parse the headonly and reclen flags.
    if headonly is True:
        unpack_data = 0
    else:
        unpack_data = 1

    # Determine total size. Either its a file-like object.
    if hasattr(mseed3_file, "tell") and hasattr(mseed3_file, "seek"):
        cur_pos = mseed3_file.tell()
        mseed3_file.seek(0, 2)
        length = mseed3_file.tell() - cur_pos
        mseed3_file.seek(cur_pos, 0)
    # Or a file name.
    else:
        length = os.path.getsize(mseed3_file)

    if length < FIXED_HEADER_SIZE:
        msg = (
            f"The smallest possible mseed3 record is made up of {FIXED_HEADER_SIZE} "
            f"bytes. The passed buffer or file contains only {length}."
        )
        raise ObsPyMSEED3FilesizeTooSmallError(msg)
    elif length > 2**31:
        msg = (
            "ObsPy can currently not directly read mseed3 files that "
            "are larger than 2^31 bytes (2048 MiB). To still read it, "
            "please read the file in chunks as documented here: "
            "https://github.com/obspy/obspy/pull/1419"
            "#issuecomment-221582369"
        )
        raise ObsPyMSEED3FilesizeTooLargeError(msg)

    if isinstance(mseed3_file, io.BufferedIOBase):
        return _internal_read_mseed3(
            mseed3_file, starttime=starttime, endtime=endtime, matchsid=matchsid
        )
    elif isinstance(mseed3_file, (str, bytes)):
        with open(mseed3_file, "rb") as fh:
            return _internal_read_mseed3(
                fh, starttime=starttime, endtime=endtime, matchsid=matchsid
            )
    else:
        raise ValueError("Cannot open '%s'." % filename)


def _internal_read_mseed3(
    fp, starttime=None, endtime=None, headonly=False, matchsid=None
):
    traces = []
    for ms3 in readMSeed3Records(fp, matchsid=matchsid, merge=True):
        if starttime is not None and ms3.endtime < starttime:
            continue
        if endtime is not None and endtime < ms3.starttime:
            continue
        stats = mseed3_to_obspy_header(ms3)
        if canDecompress(ms3.header.encoding):
            data = ms3.decompress()
            trace = Trace(data=data, header=stats)
            traces.append(trace)
        else:
            trace = Trace(header=stats)
            traces.append(trace)
    return Stream(traces=traces)


def _write_mseed3(stream, filename, encoding=None, flush=True, verbose=0, **_kwargs):
    """
    Write miniseed3 file from a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: A Stream object.
    :type filename: str
    :param filename: Name of the output file or a file-like object.
    :type encoding: int or str, optional
    :param encoding: Should be set to one of the following supported mseed3
        data encoding formats: ``ASCII`` (``0``)*, ``INT16`` (``1``),
        ``INT32`` (``3``), ``FLOAT32`` (``4``)*, ``FLOAT64`` (``5``)*,
        ``STEIM1`` (``10``) and ``STEIM2`` (``11``)*. If no encoding is given
        it will be derived from the dtype of the data and the appropriate
        default encoding (depicted with an asterix) will be chosen.
    :type verbose: int, optional
    :param verbose: Controls verbosity, a value of ``0`` will result in no
        diagnostic output

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read()
    >>> st.write('filename.ms3', format='MSEED3')  # doctest: +SKIP
    """

    # Open filehandler or use an existing file like object.
    if not hasattr(filename, "write"):
        f = open(filename, "wb")
    else:
        f = filename

    for trace in stream:
        ms3Header = MSeed3Header()
        st = trace.stats["starttime"]
        ms3Header.year = st.year
        ms3Header.dayOfYear = st.julday
        ms3Header.hour = st.hour
        ms3Header.minute = st.minute
        ms3Header.second = st.second
        ms3Header.nanosecond = st.microsecond * 1000 + st.ns % 1000
        ms3Header.sampleRatePeriod = trace.stats["sampling_rate"]
        if len(trace.stats["channel"]) == 0:
            identifier = FDSNSourceId.createUnknown(
                ms3Header.sampleRate,
                networkCode=trace.stats["network"],
                stationCode=trace.stats["station"],
                locationCode=trace.stats["location"],
            )
        else:
            identifier = FDSNSourceId.fromNslc(
                trace.stats["network"],
                trace.stats["station"],
                trace.stats["location"],
                trace.stats["channel"],
            )
        eh = {}
        if "mseed3" in trace.stats:
            ms3stats = trace.stats["mseed3"]
            if "publicationVersion" in ms3stats:
                ms3Header.publicationVersion = parseInt(ms3stats["publicationVersion"])
            if "extraHeaders" in ms3stats:
                eh = ms3stats["extraHeaders"]
        ms3 = MSeed3Record(ms3Header, identifier, trace.data, eh)
        f.write(ms3.pack())
    # Close if its a file handler.
    if not hasattr(filename, "write"):
        f.close()


def mseed3_to_obspy_header(ms3):
    stats = {}
    h = ms3.header
    stats["npts"] = h.numSamples
    stats["sampling_rate"] = h.sampleRate
    sid = FDSNSourceId.parse(ms3.identifier)
    nslc = sid.asNslc()
    stats["network"] = nslc.networkCode
    stats["station"] = nslc.stationCode
    stats["location"] = nslc.locationCode
    stats["channel"] = nslc.channelCode
    micros = int(round(h.nanosecond / 1000))
    stats["starttime"] = UTCDateTime(
        h.year,
        julday=h.dayOfYear,
        hour=h.hour,
        minute=h.minute,
        second=h.second,
        microsecond=micros,
    )

    # store extra header values
    eh = {}
    if len(ms3.eh) > 0:
        eh = ms3.eh
    stats["mseed3"] = {
        "publicationVersion": h.publicationVersion,
        "extraHeaders": eh,
    }

    return Stats(stats)


class ObsPyMSEED3Error(ObsPyException):
    pass


class ObsPyMSEED3ReadingError(ObsPyMSEED3Error, ObsPyReadingError):
    pass


class ObsPyMSEED3FilesizeTooSmallError(ObsPyMSEED3ReadingError):
    pass


class ObsPyMSEED3FilesizeTooLargeError(ObsPyMSEED3ReadingError):
    pass


if __name__ == "__main__":
    import doctest

    doctest.testmod(exclude_empty=True)
