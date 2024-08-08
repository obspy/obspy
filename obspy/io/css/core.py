# -*- coding: utf-8 -*-
"""
CSS bindings to ObsPy core module.
"""
from pathlib import Path
import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core.compatibility import from_buffer

import gzip

DTYPE = {
    # Big-endian integers
    b's4': b'>i',
    b's2': b'>h',
    # Little-endian integers
    b'i4': b'<i',
    b'i2': b'<h',
    # ASCII integers
    b'c0': (b'S12', int),
    b'c#': (b'S12', int),
    # Big-endian floating point
    b't4': b'>f',
    b't8': b'>d',
    # Little-endian floating point
    b'f4': b'<f',
    b'f8': b'<d',
    # ASCII floating point
    b'a0': (b'S15', np.float32),
    b'a#': (b'S15', np.float32),
    b'b0': (b'S24', np.float64),
    b'b#': (b'S24', np.float64),
}


def _is_css(filename):
    """
    Checks whether a file is CSS waveform data (header) or not.

    :type filename: str
    :param filename: CSS file to be checked.
    :rtype: bool
    :return: ``True`` if a CSS waveform header file.
    """
    # Fixed file format.
    # Tests:
    #  - the length of each line (283 chars)
    #  - two epochal time fields
    #    (for position of dot and if they convert to UTCDateTime)
    #  - supported data type descriptor
    try:
        with open(filename, "rb") as fh:
            lines = fh.readlines()
            # check for empty file
            if not lines:
                return False
            # check every line
            for line in lines:
                assert len(line.rstrip(b"\n\r")) == 283
                assert b"." in line[26:28]
                UTCDateTime(float(line[16:33]))
                assert b"." in line[71:73]
                UTCDateTime(float(line[61:78]))
                assert line[143:145] in DTYPE
    except Exception:
        return False
    return True


def _is_nnsa_kb_core(filename):
    """
    Checks whether a file is NNSA KB Core waveform data (header) or not.

    :type filename: str
    :param filename: NNSA KB Core file to be checked.
    :rtype: bool
    :return: ``True`` if a NNSA KB Core waveform header file.
    """
    # Fixed file format.
    # Tests:
    #  - the length of each line (287 chars)
    #  - two epochal time fields
    #    (for position of dot and if they convert to UTCDateTime)
    #  - supported data type descriptor
    try:
        with open(filename, "rb") as fh:
            lines = fh.readlines()
            # check for empty file
            if not lines:
                return False
            # check every line
            for line in lines:
                assert len(line.rstrip(b"\n\r")) == 287
                assert line[27:28] == b"."
                UTCDateTime(float(line[16:33]))
                assert line[73:74] == b"."
                UTCDateTime(float(line[62:79]))
                assert line[144:146] in DTYPE
    except Exception:
        return False
    return True


def _read_css(filename, **kwargs):
    """
    Reads a CSS waveform file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: CSS file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream with Traces specified by given file.
    """
    # read metafile with info on single traces
    with open(filename, "rb") as fh:
        lines = fh.readlines()
    basedir = Path(filename).parent
    traces = []
    # read single traces
    for line in lines:
        npts = int(line[79:87])
        dirname = line[148:212].strip().decode()
        dfilename = Path(basedir) / dirname / line[213:245].strip().decode()
        offset = int(line[246:256])
        dtype = DTYPE[line[143:145]]
        if isinstance(dtype, tuple):
            read_fmt = np.dtype(dtype[0])
            fmt = dtype[1]
        else:
            read_fmt = np.dtype(dtype)
            fmt = read_fmt

        try:
            # assumed that the waveform file is not compressed
            fh = open(dfilename, "rb")
        except FileNotFoundError as e:
            # If does not find the waveform file referenced in the wfdisc,
            # it will try to open a compressed .gz suffix file instead.
            try:
                fh = gzip.open(str(dfilename) + '.gz', "rb")
            except FileNotFoundError:
                raise e

        # Read one segment of binary data
        fh.seek(offset)
        data = fh.read(read_fmt.itemsize * npts)
        fh.close()
        data = from_buffer(data, dtype=read_fmt)
        data = np.require(data, dtype=fmt)

        header = {}
        header['station'] = line[0:6].strip().decode()
        header['channel'] = line[7:15].strip().decode()
        header['starttime'] = UTCDateTime(float(line[16:33]))
        header['sampling_rate'] = float(line[88:99])
        header['calib'] = float(line[100:116])
        header['calper'] = float(line[117:133])
        tr = Trace(data, header=header)
        traces.append(tr)
    return Stream(traces=traces)


def _read_nnsa_kb_core(filename, **kwargs):
    """
    Reads a NNSA KB Core waveform file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: NNSA KB Core file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream with Traces specified by given file.
    """
    # read metafile with info on single traces
    with open(filename, "rb") as fh:
        lines = fh.readlines()
    basedir = Path(filename).parent
    traces = []
    # read single traces
    for line in lines:
        npts = int(line[80:88])
        dirname = line[149:213].strip().decode()
        filename = Path(basedir) / dirname / \
            line[214:246].strip().decode()

        offset = int(line[247:257])
        dtype = DTYPE[line[144:146]]
        if isinstance(dtype, tuple):
            read_fmt = np.dtype(dtype[0])
            fmt = dtype[1]
        else:
            read_fmt = np.dtype(dtype)
            fmt = read_fmt
        with open(filename, "rb") as fh:
            fh.seek(offset)
            data = fh.read(read_fmt.itemsize * npts)
            data = from_buffer(data, dtype=read_fmt)
            data = np.require(data, dtype=fmt)
        header = {}
        header['station'] = line[0:6].strip().decode()
        header['channel'] = line[7:15].strip().decode()
        header['starttime'] = UTCDateTime(float(line[16:33]))
        header['sampling_rate'] = float(line[89:100])
        header['calib'] = float(line[101:117])
        header['calper'] = float(line[118:134])
        tr = Trace(data, header=header)
        traces.append(tr)
    return Stream(traces=traces)
