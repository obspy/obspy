# -*- coding: utf-8 -*-
"""
MSEED3 bindings to ObsPy core module.
"""
import io
import os
import warnings
from pathlib import Path
from struct import pack

import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core.compatibility import from_buffer
from obspy.core.util import NATIVE_BYTEORDER
from . import (util, InternalMSEEDError, ObsPyMSEEDFilesizeTooSmallError,
               ObsPyMSEEDFilesizeTooLargeError, ObsPyMSEEDError)
from .headers import (DATATYPES, ENCODINGS, HPTERROR, HPTMODULUS, SAMPLETYPE,
                      SEED_CONTROL_HEADERS, UNSUPPORTED_ENCODINGS,
                      VALID_CONTROL_HEADERS, VALID_RECORD_LENGTHS, Selections,
                      SelectTime, Blkt100S, Blkt1001S, clibmseed)


def _is_mseed3(filename):
    """
    Checks whether a file is mseed3 or not.

    :type filename: str
    :param filename: mseed3 file to be checked.
    :rtype: bool
    :return: ``True`` if a mseed3 file.

    This method only reads the first three bytes of the file and checks
    whether it is 'M' 'S' 3.

    """
    # Open filehandler or use an existing file like object.
    if not hasattr(filename, 'read'):
        file_size = os.path.getsize(filename)
        with io.open(filename, 'rb') as fh:
            return __is_mseed3(fh, file_size=file_size)
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
    if file_size < simplemseed.mseed.FIXED_HEADER_SIZE:
        return False
    headerBytes = fp.read(simplemseed.mseed.FIXED_HEADER_SIZE)
    # File has less than FIXED_HEADER_SIZE characters
    if len(headerBytes) != simplemseed.mseed.FIXED_HEADER_SIZE:
        return False
    if headerBytes[0] != b"M" or headerBytes[1] != b"S" or headerBytes[2] != 3:
        return False
    header = simplemseed.unpackMSeed3FixedHeader(headerBytes)
    return header.sanityCheck()


def _read_mseed3(mseed_object, starttime=None, endtime=None, headonly=False,
                sourcename=None, reclen=None, details=False,
                header_byteorder=None, verbose=None, **kwargs):
    """
    Reads a mseed3 file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param mseed_object: Filename or open file like object that contains the
        binary mseed3 data. Any object that provides a read() method will be
        considered to be a file like object.
    :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param starttime: Only read data samples after or at the start time.
    :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param endtime: Only read data samples before or at the end time.
    :param headonly: Determines whether or not to unpack the data or just
        read the headers.
    :type sourcename: str
    :param sourcename: Only read data with matching SEED ID (can contain
        wildcards "?" and "*", e.g. "BW.UH2.*" or "*.??Z").
        Defaults to ``None`` .
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

        - ``1`` : Step Calibration (Blockette 300)
        - ``2`` : Sine Calibration (Blockette 310)
        - ``3`` : Pseudo-random Calibration (Blockette 320)
        - ``4`` : Generic Calibration  (Blockette 390)
        - ``-2`` : Calibration Abort (Blockette 395)

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
    >>> st = read("/path/to/two_channels.mseed",
    ...           starttime=UTCDateTime("2010-06-20T00:00:01"),
    ...           sourcename="*.?HZ")
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    BW.UH3..EHZ | 2010-06-20T00:00:00.999999Z - ... | 200.0 Hz, 242 samples

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
    if isinstance(mseed_object, Path):
        mseed_object = str(mseed_object)
    # Parse the headonly and reclen flags.
    if headonly is True:
        unpack_data = 0
    else:
        unpack_data = 1

    # Determine total size. Either its a file-like object.
    if hasattr(mseed_object, "tell") and hasattr(mseed_object, "seek"):
        cur_pos = mseed_object.tell()
        mseed_object.seek(0, 2)
        length = mseed_object.tell() - cur_pos
        mseed_object.seek(cur_pos, 0)
    # Or a file name.
    else:
        length = os.path.getsize(mseed_object)

    if length < simplemseed.mseed.FIXED_HEADER_SIZE:
        msg = f"The smallest possible mseed3 record is made up of {simplemseed.mseed.FIXED_HEADER_SIZE} " \
              f"bytes. The passed buffer or file contains only {length}."
        raise ObsPyMSEEDFilesizeTooSmallError(msg)
    elif length > 2 ** 31:
        msg = ("ObsPy can currently not directly read mseed3 files that "
               "are larger than 2^31 bytes (2048 MiB). To still read it, "
               "please read the file in chunks as documented here: "
               "https://github.com/obspy/obspy/pull/1419"
               "#issuecomment-221582369")
        raise ObsPyMSEEDFilesizeTooLargeError(msg)

    traces = []
    for ms3 in simplemseed.readMSeed3Records():
        stats = mseed3_to_obspy_header(ms3)
        data = ms3.decompress()
        assert dtype in data # make sure numpy array
        trace = Trace(data=data, header=stats)
        traces.append(trace)
    return Stream(traces=traces)


def _write_mseed(stream, filename, encoding=None, flush=True, verbose=0, **_kwargs):
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
    :type flush: bool, optional
    :param flush: If ``True``, all data will be packed into records. If
        ``False`` new records will only be created when there is enough data to
        completely fill a record. Be careful with this. If in doubt, choose
        ``True`` which is also the default value.
    :type verbose: int, optional
    :param verbose: Controls verbosity, a value of ``0`` will result in no
        diagnostic output

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read()
    >>> st.write('filename.ms3', format='MSEED3')  # doctest: +SKIP
    """

    # Open filehandler or use an existing file like object.
    if not hasattr(filename, 'write'):
        f = open(filename, 'wb')
    else:
        f = filename

    for trace in stream:
        ms3Header = MSeed3Header()
        header.starttime = trace.stats['starttime']
        header.sampleRatePeriod = trace.stats['sampling_rate']
        if len(trace.stats['channel']) == 0:
            identifier = simplemseed.FDSNSourceId.createUnknown(
                header.sampleRate,
                networkCode=trace.stats['network'],
                stationCode=trace.stats['station'],
                locationCode=trace.stats['location'])
        else:
            identifier = simplemseed.FDSNSourceId.fromNslc(
                trace.stats['network'],
                trace.stats['station'],
                trace.stats['location'],
                trace.stats['channel']
                )
        eh = {}
        if 'mseed3' in trace.stats:
            ms3stats = trace.stats['mseed3']
            eh = ms3stats
        ms3 = simplemseed.MSeed3Record(header, data, identifier, eh)
        f.write(ms3.pack())
    # Close if its a file handler.
    if not hasattr(filename, 'write'):
        f.close()

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
