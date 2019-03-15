# -*- coding: utf-8 -*-
"""
Simple ASCII time series formats

* ``SLIST``, a ASCII time series format represented with a header line
  followed by a sample lists (see also
  :func:`SLIST format description<obspy.io.ascii.core._write_slist>`)::

    TIMESERIES BW_RJOB__EHZ_D, 6001 samples, 200 sps, 2009-08-24T00:20:03.0000\
00, SLIST, INTEGER,
    288 300 292 285 265 287
    279 250 278 278 268 258
    ...

* ``TSPAIR``, a ASCII format where data is written in time-sample pairs (see
  also :func:`TSPAIR format description<obspy.io.ascii.core._write_tspair>`)::

    TIMESERIES BW_RJOB__EHZ_D, 6001 samples, 200 sps, 2009-08-24T00:20:03.0000\
00, TSPAIR, INTEGER,
    2009-08-24T00:20:03.000000  288
    2009-08-24T00:20:03.005000  300
    2009-08-24T00:20:03.010000  292
    2009-08-24T00:20:03.015000  285
    2009-08-24T00:20:03.020000  265
    2009-08-24T00:20:03.025000  287
    ...

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import io

import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core import Stats
from obspy.core.util import AttribDict, loadtxt


HEADER = ("TIMESERIES {network}_{station}_{location}_{channel}_{dataquality}, "
          "{npts:d} samples, {sampling_rate} sps, {starttime!s:.26s}, "
          "{format}, {dtype}, {unit}\n")


def _format_header(stats, format, dataquality, dtype, unit):
    sampling_rate = str(stats.sampling_rate)
    if "." in sampling_rate and "E" not in sampling_rate.upper():
        sampling_rate = sampling_rate.rstrip('0').rstrip('.')
    header = HEADER.format(
        network=stats.network, station=stats.station, location=stats.location,
        channel=stats.channel, dataquality=dataquality, npts=stats.npts,
        sampling_rate=sampling_rate, starttime=stats.starttime,
        format=format, dtype=dtype, unit=unit)
    return header


def _is_slist(filename):
    """
    Checks whether a file is ASCII SLIST format.

    :type filename: str
    :param filename: Name of the ASCII SLIST file to be checked.
    :rtype: bool
    :return: ``True`` if ASCII SLIST file.

    .. rubric:: Example

    >>> _is_slist('/path/to/slist.ascii')  # doctest: +SKIP
    True
    """
    try:
        with open(filename, 'rt') as f:
            temp = f.readline()
    except Exception:
        return False
    if not temp.startswith('TIMESERIES'):
        return False
    if 'SLIST' not in temp:
        return False
    return True


def _is_tspair(filename):
    """
    Checks whether a file is ASCII TSPAIR format.

    :type filename: str
    :param filename: Name of the ASCII TSPAIR file to be checked.
    :rtype: bool
    :return: ``True`` if ASCII TSPAIR file.

    .. rubric:: Example

    >>> _is_tspair('/path/to/tspair.ascii')  # doctest: +SKIP
    True
    """
    try:
        with open(filename, 'rt') as f:
            temp = f.readline()
    except Exception:
        return False
    if not temp.startswith('TIMESERIES'):
        return False
    if 'TSPAIR' not in temp:
        return False
    return True


def _read_slist(filename, headonly=False, **kwargs):  # @UnusedVariable
    """
    Reads a ASCII SLIST file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: ASCII file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read('/path/to/slist.ascii')
    """
    with open(filename, 'rt') as fh:
        # read file and split text into channels
        buf = []
        key = False
        for line in fh:
            if line.isspace():
                # blank line
                continue
            elif line.startswith('TIMESERIES'):
                # new header line
                key = True
                buf.append((line, io.StringIO()))
            elif headonly:
                # skip data for option headonly
                continue
            elif key:
                # data entry - may be written in multiple columns
                buf[-1][1].write(line.strip() + ' ')
    # create ObsPy stream object
    stream = Stream()
    for header, data in buf:
        # create Stats
        stats = Stats()
        parts = header.replace(',', '').split()
        temp = parts[1].split('_')
        stats.network = temp[0]
        stats.station = temp[1]
        stats.location = temp[2]
        stats.channel = temp[3]
        stats.sampling_rate = parts[4]
        # quality only used in MSEED
        # don't put blank quality code into 'mseed' dictionary
        # (quality code is mentioned as optional by format specs anyway)
        if temp[4]:
            stats.mseed = AttribDict({'dataquality': temp[4]})
        stats.ascii = AttribDict({'unit': parts[-1]})
        stats.starttime = UTCDateTime(parts[6])
        stats.npts = parts[2]
        if headonly:
            # skip data
            stream.append(Trace(header=stats))
        else:
            data = _parse_data(data, parts[8])
            stream.append(Trace(data=data, header=stats))
    return stream


def _read_tspair(filename, headonly=False, **kwargs):  # @UnusedVariable
    """
    Reads a ASCII TSPAIR file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: ASCII file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the headers. This is most useful
        for scanning available data in huge (temporary) data sets.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read('/path/to/tspair.ascii')
    """
    with open(filename, 'rt') as fh:
        # read file and split text into channels
        buf = []
        key = False
        for line in fh:
            if line.isspace():
                # blank line
                continue
            elif line.startswith('TIMESERIES'):
                # new header line
                key = True
                buf.append((line, io.StringIO()))
            elif headonly:
                # skip data for option headonly
                continue
            elif key:
                # data entry - may be written in multiple columns
                buf[-1][1].write(line.strip().split()[-1] + ' ')
    # create ObsPy stream object
    stream = Stream()
    for header, data in buf:
        # create Stats
        stats = Stats()
        parts = header.replace(',', '').split()
        temp = parts[1].split('_')
        stats.network = temp[0]
        stats.station = temp[1]
        stats.location = temp[2]
        stats.channel = temp[3]
        stats.sampling_rate = parts[4]
        # quality only used in MSEED
        # don't put blank quality code into 'mseed' dictionary
        # (quality code is mentioned as optional by format specs anyway)
        if temp[4]:
            stats.mseed = AttribDict({'dataquality': temp[4]})
        stats.ascii = AttribDict({'unit': parts[-1]})
        stats.starttime = UTCDateTime(parts[6])
        stats.npts = parts[2]
        if headonly:
            # skip data
            stream.append(Trace(header=stats))
        else:
            data = _parse_data(data, parts[8])
            stream.append(Trace(data=data, header=stats))
    return stream


def _write_slist(stream, filename, custom_fmt=None,
                 **kwargs):  # @UnusedVariable
    """
    Writes a ASCII SLIST file.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of file to write.
    :type custom_fmt: str
    :param custom_fmt: formatter for writing sample values. Defaults to None.
        Using this parameter will set ``TYPE`` value in header to ``CUSTOM``
        and ObsPy will raise an exception while trying to read that file.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read()
    >>> st.write("slist.ascii", format="SLIST")  #doctest: +SKIP

    .. rubric:: SLIST Format Description

    SLIST is a simple ASCII time series format. Each contiguous time series
    segment (no gaps or overlaps) is represented with a header line followed by
    a sample lists. There are no restrictions on how the segments are organized
    into files, a file might contain a single segment or many, concatenated
    segments either for the same channel or many different channels.

    Header lines have the general form::

        TIMESERIES SourceName, # samples, # sps, Time, Format, Type, Units

    with

    ``SourceName``
        "Net_Sta_Loc_Chan_Qual", no spaces, quality code optional
    ``# samples``
        Number of samples following header
    ``# sps``
        Sampling rate in samples per second
    ``Time``
        Time of first sample in ISO YYYY-MM-DDTHH:MM:SS.FFFFFF format
    ``Format``
        'TSPAIR' (fixed)
    ``Type``
        Sample type 'INTEGER', 'FLOAT' or 'ASCII'
    ``Units``
        Units of time-series, e.g. Counts, M/S, etc., may not contain
        spaces

    Samples are listed in 6 columns with the time-series incrementing from left
    to right and wrapping to the next line. The time of the first sample is the
    time listed in the header.

    *Example SLIST file*::

        TIMESERIES NL_HGN_00_BHZ_R, 12 samples, 40 sps, 2003-05-29T02:13:22.04\
3400, SLIST, INTEGER, Counts
        2787        2776        2774        2780        2783        2782
        2776        2766        2759        2760        2765        2767
        ...
    """
    with open(filename, 'wb') as fh:
        for trace in stream:
            stats = trace.stats
            # quality code
            try:
                dataquality = stats.mseed.dataquality
            except Exception:
                dataquality = ''
            # sample type
            if trace.data.dtype.name.startswith('int'):
                dtype = 'INTEGER'
                fmt = '%d'
            elif trace.data.dtype.name.startswith('float'):
                dtype = 'FLOAT'
                fmt = '%+.10e'

            else:
                raise NotImplementedError
            # fmt
            if custom_fmt is not None:
                dtype = _determine_dtype(custom_fmt)
                fmt = custom_fmt
            # unit
            try:
                unit = stats.ascii.unit
            except Exception:
                unit = ''
            # write trace header
            header = _format_header(stats, 'SLIST', dataquality, dtype, unit)
            fh.write(header.encode('ascii', 'strict'))
            # write data
            rest = stats.npts % 6
            if rest:
                data = trace.data[:-rest]
            else:
                data = trace.data
            data = data.reshape((-1, 6))
            np.savetxt(fh, data, delimiter=b'\t',
                       fmt=fmt.encode('ascii', 'strict'))
            if rest:
                fh.write(('\t'.join([fmt % d for d in trace.data[-rest:]]) +
                         '\n').encode('ascii', 'strict'))


def _write_tspair(stream, filename, custom_fmt=None,
                  **kwargs):  # @UnusedVariable
    """
    Writes a ASCII TSPAIR file.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of file to write.
    :type custom_fmt: str
    :param custom_fmt: formatter for writing sample values. Defaults to None.
        Using this parameter will set ``TYPE`` value in header to ``CUSTOM``
        and ObsPy will raise an exception while trying to read that file.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read()
    >>> st.write("tspair.ascii", format="TSPAIR")  #doctest: +SKIP

    .. rubric:: TSPAIR Format Description

    TSPAIR is a simple ASCII time series format. Each contiguous time series
    segment (no gaps or overlaps) is represented with a header line followed by
    data samples in time-sample pairs. There are no restrictions on how the
    segments are organized into files, a file might contain a single segment
    or many, concatenated segments either for the same channel or many
    different channels.

    Header lines have the general form::

        TIMESERIES SourceName, # samples, # sps, Time, Format, Type, Units

    with

    ``SourceName``
        "Net_Sta_Loc_Chan_Qual", no spaces, quality code optional
    ``# samples``
        Number of samples following header
    ``# sps``
        Sampling rate in samples per second
    ``Time``
        Time of first sample in ISO YYYY-MM-DDTHH:MM:SS.FFFFFF format
    ``Format``
        'TSPAIR' (fixed)
    ``Type``
        Sample type 'INTEGER', 'FLOAT' or 'ASCII'
    ``Units``
        Units of time-series, e.g. Counts, M/S, etc., may not contain
        spaces

    *Example TSPAIR file*::

        TIMESERIES NL_HGN_00_BHZ_R, 12 samples, 40 sps, 2003-05-29T02:13:22.04\
3400, TSPAIR, INTEGER, Counts
        2003-05-29T02:13:22.043400  2787
        2003-05-29T02:13:22.068400  2776
        2003-05-29T02:13:22.093400  2774
        2003-05-29T02:13:22.118400  2780
        2003-05-29T02:13:22.143400  2783
        2003-05-29T02:13:22.168400  2782
        2003-05-29T02:13:22.193400  2776
        2003-05-29T02:13:22.218400  2766
        2003-05-29T02:13:22.243400  2759
        2003-05-29T02:13:22.268400  2760
        2003-05-29T02:13:22.293400  2765
        2003-05-29T02:13:22.318400  2767
        ...
    """
    with open(filename, 'wb') as fh:
        for trace in stream:
            stats = trace.stats
            # quality code
            try:
                dataquality = stats.mseed.dataquality
            except Exception:
                dataquality = ''
            # sample type
            if trace.data.dtype.name.startswith('int'):
                dtype = 'INTEGER'
                fmt = '%d'
            elif trace.data.dtype.name.startswith('float'):
                dtype = 'FLOAT'
                fmt = '%+.10e'
            # fmt
            if custom_fmt is not None:
                dtype = _determine_dtype(custom_fmt)
                fmt = custom_fmt
            # unit
            try:
                unit = stats.ascii.unit
            except Exception:
                unit = ''
            # write trace header
            header = _format_header(stats, 'TSPAIR', dataquality, dtype, unit)
            fh.write(header.encode('ascii', 'strict'))
            # write data
            for t, d in zip(trace.times(type='utcdatetime'), trace.data):
                # .26s cuts the Z from the time string
                line = ('%.26s  ' + fmt + '\n') % (t, d)
                fh.write(line.encode('ascii', 'strict'))


def _determine_dtype(custom_fmt):
    """
    :type custom_fmt: str
    :param custom_fmt: Python string formatter.
    :rtype: str
    :return: Datatype string for writing in header. Currently supported
        are 'INTEGER', 'FLOAT' and `CUSTOM`.
    :raises ValueError: if provided string is empty.
    """
    floats = ('e', 'f', 'g')
    ints = ('d', 'i')
    try:
        if custom_fmt[-1].lower() in floats:
            return 'FLOAT'
        elif custom_fmt[-1].lower() in ints:
            return 'INTEGER'
        else:
            return 'CUSTOM'
    except IndexError:
        raise ValueError('Provided string is not valid for determining ' +
                         'datatype. Provide a proper Python string formatter')


def _parse_data(data, data_type):
    """
    Simple function to read data contained in a StringIO object to a NumPy
    array.

    :type data: io.StringIO
    :param data: The actual data.
    :type data_type: str
    :param data_type: The data type of the expected data. Currently supported
        are 'INTEGER' and 'FLOAT'.
    """
    if data_type == "INTEGER":
        dtype = np.int_
    elif data_type == "FLOAT":
        dtype = np.float64
    else:
        raise NotImplementedError
    # Seek to the beginning of the StringIO.
    data.seek(0)
    # Data will always be a StringIO. Avoid to send empty StringIOs to
    # numpy.readtxt() which raises a warning.
    if len(data.read(1)) == 0:
        return np.array([], dtype=dtype)
    data.seek(0)
    return loadtxt(data, dtype=dtype, ndmin=1)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
