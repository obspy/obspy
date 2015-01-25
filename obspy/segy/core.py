# -*- coding: utf-8 -*-
"""
SEG Y bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy import Stream, Trace, UTCDateTime
from obspy.core import AttribDict
from obspy.segy.segy import readSEGY as readSEGYrev1
from obspy.segy.segy import readSU as readSUFile
from obspy.segy.segy import SEGYError, SEGYFile, SEGYBinaryFileHeader
from obspy.segy.segy import SEGYTrace, autodetectEndianAndSanityCheckSU
from obspy.segy.segy import SUFile, SEGYTraceHeader
from obspy.segy.header import BINARY_FILE_HEADER_FORMAT, TRACE_HEADER_FORMAT
from obspy.segy.header import DATA_SAMPLE_FORMAT_CODE_DTYPE, TRACE_HEADER_KEYS
from obspy.segy.header import ENDIAN
from obspy.segy.util import unpack_header_value

import numpy as np
from struct import unpack
from copy import deepcopy
import warnings


# Valid data format codes as specified in the SEGY rev1 manual.
VALID_FORMATS = [1, 2, 3, 4, 5, 8]

# This is the maximum possible interval between two samples due to the nature
# of the SEG Y format.
MAX_INTERVAL_IN_SECONDS = 0.065535


class SEGYCoreWritingError(SEGYError):
    """
    Raised if the writing of the Stream object fails due to some reason.
    """
    pass


class SEGYSampleIntervalError(SEGYError):
    """
    Raised if the interval between two samples is too large.
    """
    pass


def isSEGY(filename):
    """
    Checks whether or not the given file is a SEG Y file.

    :type filename: str
    :param filename: SEG Y file to be checked.
    :rtype: bool
    :return: ``True`` if a SEG Y file.
    """
    # This is a very weak test. It tests two things: First if the data sample
    # format code is valid. This is also used to determine the endianness. This
    # is then used to check if the sampling interval is set to any sane number
    # greater than 0 and that the number of samples per trace is greater than
    # 0.
    try:
        with open(filename, 'rb') as fp:
            fp.seek(3212)
            _number_of_data_traces = fp.read(2)
            _number_of_auxiliary_traces = fp.read(2)
            _sample_interval = fp.read(2)
            fp.seek(2, 1)
            _samples_per_trace = fp.read(2)
            fp.seek(2, 1)
            data_format_code = fp.read(2)
            fp.seek(3500, 0)
            _format_number = fp.read(2)
            _fixed_length = fp.read(2)
            _extended_number = fp.read(2)
    except:
        return False
    # Unpack using big endian first and check if it is valid.
    try:
        format = unpack(b'>h', data_format_code)[0]
    except:
        return False
    if format in VALID_FORMATS:
        _endian = '>'
    # It can only every be one. It is impossible for little and big endian to
    # both yield a valid data sample format code because they are restricted to
    # be between 1 and 8.
    else:
        format = unpack(b'<h', data_format_code)[0]
        if format in VALID_FORMATS:
            _endian = '<'
        else:
            return False
    # Check if the sample interval and samples per Trace make sense.
    fmt = ('%sh' % _endian).encode('ascii', 'strict')
    _sample_interval = unpack(fmt, _sample_interval)[0]
    _samples_per_trace = unpack(fmt, _samples_per_trace)[0]
    _number_of_data_traces = unpack(fmt, _number_of_data_traces)[0]
    _number_of_auxiliary_traces = unpack(fmt,
                                         _number_of_auxiliary_traces)[0]
    _format_number = unpack(fmt, _format_number)[0]
    _fixed_length = unpack(fmt, _fixed_length)[0]
    _extended_number = unpack(fmt, _extended_number)[0]
    # Make some sanity checks and return False if they fail.
    # Unfortunately the format number is 0 in many files so it cannot be truly
    # tested.
    if _sample_interval <= 0 or _samples_per_trace <= 0 \
       or _number_of_data_traces < 0 or _number_of_auxiliary_traces < 0 \
       or _format_number < 0 or _fixed_length < 0 or _extended_number < 0:
        return False
    return True


def readSEGY(filename, headonly=False, byteorder=None,
             textual_header_encoding=None, unpack_trace_headers=False,
             **kwargs):  # @UnusedVariable
    """
    Reads a SEG Y file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: SEG Y rev1 file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the header and omit the waveform
        data.
    :type byteorder: str or ``None``
    :param byteorder: Determines the endianness of the file. Either ``'>'`` for
        big endian or ``'<'`` for little endian. If it is ``None``, it will try
        to autodetect the endianness. The endianness is always valid for the
        whole file. Defaults to ``None``.
    :type textual_header_encoding: str or ``None``
    :param textual_header_encoding: The encoding of the textual header. Can be
        ``'EBCDIC'``, ``'ASCII'`` or ``None``. If it is ``None``, autodetection
        will be attempted. Defaults to ``None``.
    :type unpack_trace_headers: bool, optional
    :param unpack_trace_headers: Determines whether or not all trace header
        values will be unpacked during reading. If ``False`` it will greatly
        enhance performance and especially memory usage with large files. The
        header values can still be accessed and will be calculated on the fly
        but tab completion will no longer work. Look in the headers.py for a
        list of all possible trace header values. Defaults to ``False``.
    :returns: A ObsPy :class:`~obspy.core.stream.Stream` object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/00001034.sgy_first_trace")
    >>> st  # doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    Seq. No. in line:    1 | 2009-06-22T14:47:37.000000Z - ... 2001 samples
    """
    # Read file to the internal segy representation.
    segy_object = readSEGYrev1(filename, endian=byteorder,
                               textual_header_encoding=textual_header_encoding,
                               unpack_headers=unpack_trace_headers)
    # Create the stream object.
    stream = Stream()
    # SEGY has several file headers that apply to all traces. They will be
    # stored in Stream.stats.
    stream.stats = AttribDict()
    # Get the textual file header.
    textual_file_header = segy_object.textual_file_header
    # The binary file header will be a new AttribDict
    binary_file_header = AttribDict()
    for key, value in segy_object.binary_file_header.__dict__.items():
        setattr(binary_file_header, key, value)
    # Get the data encoding and the endianness from the first trace.
    data_encoding = segy_object.traces[0].data_encoding
    endian = segy_object.traces[0].endian
    textual_file_header_encoding = segy_object.textual_header_encoding.upper()
    # Add the file wide headers.
    stream.stats.textual_file_header = textual_file_header
    stream.stats.binary_file_header = binary_file_header
    # Also set the data encoding, endianness and the encoding of the
    # textual_file_header.
    stream.stats.data_encoding = data_encoding
    stream.stats.endian = endian
    stream.stats.textual_file_header_encoding = \
        textual_file_header_encoding
    # Loop over all traces.
    for tr in segy_object.traces:
        # Create new Trace object for every segy trace and append to the Stream
        # object.
        trace = Trace()
        stream.append(trace)
        # skip data if headonly is set
        if headonly:
            trace.stats.npts = tr.npts
        else:
            trace.data = tr.data
        trace.stats.segy = AttribDict()
        # If all values will be unpacked create a normal dictionary.
        if unpack_trace_headers:
            # Add the trace header as a new attrib dictionary.
            header = AttribDict()
            for key, value in tr.header.__dict__.items():
                setattr(header, key, value)
        # Otherwise use the LazyTraceHeaderAttribDict.
        else:
            # Add the trace header as a new lazy attrib dictionary.
            header = LazyTraceHeaderAttribDict(tr.header.unpacked_header,
                                               tr.header.endian)
        trace.stats.segy.trace_header = header
        # The sampling rate should be set for every trace. It is a sample
        # interval in microseconds. The only sanity check is that is should be
        # larger than 0.
        tr_header = trace.stats.segy.trace_header
        if tr_header.sample_interval_in_ms_for_this_trace > 0:
            trace.stats.delta = \
                float(tr.header.sample_interval_in_ms_for_this_trace) / \
                1E6
        # If the year is not zero, calculate the start time. The end time is
        # then calculated from the start time and the sampling rate.
        if tr_header.year_data_recorded > 0:
            year = tr_header.year_data_recorded
            # The SEG Y rev 0 standard specifies the year to be a 4 digit
            # number.  Before that it was unclear if it should be a 2 or 4
            # digit number. Old or wrong software might still write 2 digit
            # years. Every number <30 will be mapped to 2000-2029 and every
            # number between 30 and 99 will be mapped to 1930-1999.
            if year < 100:
                if year < 30:
                    year += 2000
                else:
                    year += 1900
            julday = tr_header.day_of_year
            hour = tr_header.hour_of_day
            minute = tr_header.minute_of_hour
            second = tr_header.second_of_minute
            trace.stats.starttime = UTCDateTime(
                year=year, julday=julday, hour=hour, minute=minute,
                second=second)
    return stream


def writeSEGY(stream, filename, data_encoding=None, byteorder=None,
              textual_header_encoding=None, **kwargs):  # @UnusedVariable
    """
    Writes a SEG Y file from given ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of file to write.
    :type data_encoding: int
    :param data_encoding: The data encoding is an integer with the following
        currently supported meaning:

        ``1``
            4 byte IBM floating points (float32)
        ``2``
            4 byte Integers (int32)
        ``3``
            2 byte Integer (int16)
        ``5``
            4 byte IEEE floating points (float32)

        The value in the brackets is the necessary dtype of the data. ObsPy
        will now automatically convert the data because data might change/loose
        precision during the conversion so the user has to take care of the
        correct dtype.

        If it is ``None``, the value of the first Trace will be used for all
        consecutive Traces. If it is None for the first Trace, 1 (IBM floating
        point numbers) will be used. Different data encodings for different
        traces are currently not supported because these will most likely not
        be readable by other software.
    :type byteorder: str or ``None``
    :param byteorder: Determines the endianness of the file. Either ``'>'`` for
        big endian or ``'<'`` for little endian. If is ``None``, it will either
        be the endianness of the first Trace or if that is also not set, it
        will be big endian. A mix between little and big endian for the headers
        and traces is currently not supported.
    :type textual_header_encoding: str or ``None``
    :param textual_header_encoding: The encoding of the textual header. Can be
        ``'EBCDIC'``, ``'ASCII'`` or ``None``. If it is ``None``, the
        textual_file_header_encoding attribute in the stats.segy dictionary of
        the first Trace is used and if that is not set, ASCII will be used.

    This function will automatically set the data encoding field of the binary
    file header so the user does not need to worry about it.

    The starttime of every trace is not a required field in the SEG Y
    specification. If the starttime of a trace is UTCDateTime(0) it will be
    interpreted as a not-set starttime and no time is written to the trace
    header. Every other time will be written.

    SEG Y supports a sample interval from 1 to 65535 microseconds in steps of 1
    microsecond. Larger intervals cannot be supported due to the definition of
    the SEG Y format. Therefore the smallest possible sampling rate is ~ 15.26
    Hz. Please keep that in mind.
    """
    # Some sanity checks to catch invalid arguments/keyword arguments.
    if data_encoding is not None and data_encoding not in VALID_FORMATS:
        msg = "Invalid data encoding."
        raise SEGYCoreWritingError(msg)
    # Figure out the data encoding if it is not set.
    if data_encoding is None:
        if hasattr(stream, 'stats') and hasattr(stream.stats, 'data_encoding'):
            data_encoding = stream.stats.data_encoding
        if hasattr(stream, 'stats') and hasattr(stream.stats,
                                                'binary_file_header'):
            data_encoding = \
                stream.stats.binary_file_header.data_sample_format_code
        # Set it to float if it in not given.
        else:
            data_encoding = 1

    # Create empty file wide headers if they do not exist.
    if not hasattr(stream, 'stats'):
        stream.stats = AttribDict()
    if not hasattr(stream.stats, 'textual_file_header'):
        stream.stats.textual_file_header = b""
    if not hasattr(stream.stats, 'binary_file_header'):
        stream.stats.binary_file_header = SEGYBinaryFileHeader()

    # Valid dtype for the data encoding.
    valid_dtype = DATA_SAMPLE_FORMAT_CODE_DTYPE[data_encoding]
    # Makes sure that the dtype is for every Trace is correct.
    for trace in stream:
        # Check the dtype.
        if trace.data.dtype != valid_dtype:
            msg = """
            The dtype of the data and the chosen data_encoding do not match.
            You need to manually convert the dtype if you want to use that
            data_encoding. Please refer to the obspy.segy manual for more
            details.
            """.strip()
            raise SEGYCoreWritingError(msg)
        # Check the sample interval.
        if trace.stats.delta > MAX_INTERVAL_IN_SECONDS:
            msg = """
            SEG Y supports a maximum interval of %s seconds in between two
            samples (trace.stats.delta value).
            """.strip()
            msg = msg % MAX_INTERVAL_IN_SECONDS
            raise SEGYSampleIntervalError(msg)

    # Figure out endianness and the encoding of the textual file header.
    if byteorder is None:
        if hasattr(stream, 'stats') and hasattr(stream.stats, 'endian'):
            byteorder = stream.stats.endian
        else:
            byteorder = '>'
    # Map the byte order.
    byteorder = ENDIAN[byteorder]
    if textual_header_encoding is None:
        if hasattr(stream, 'stats') and hasattr(
                stream.stats, 'textual_file_header_encoding'):
            textual_header_encoding = \
                stream.stats.textual_file_header_encoding
        else:
            textual_header_encoding = 'ASCII'

    # Loop over all Traces and create a SEGY File object.
    segy_file = SEGYFile()
    # Set the file wide headers.
    segy_file.textual_file_header = stream.stats.textual_file_header
    segy_file.textual_header_encoding = \
        textual_header_encoding
    binary_header = SEGYBinaryFileHeader()
    this_binary_header = stream.stats.binary_file_header
    # Loop over all items and if they exists set them. Ignore all other
    # attributes.
    for _, item, _ in BINARY_FILE_HEADER_FORMAT:
        if hasattr(this_binary_header, item):
            setattr(binary_header, item, getattr(this_binary_header, item))
    # Set the data encoding.
    binary_header.data_sample_format_code = data_encoding
    segy_file.binary_file_header = binary_header
    # Add all traces.
    for trace in stream:
        new_trace = SEGYTrace()
        new_trace.data = trace.data
        # Create empty trace header if none is there.
        if not hasattr(trace.stats, 'segy'):
            warnings.warn("CREATING TRACE HEADER")
            trace.stats.segy = {}
            trace.stats.segy.trace_header = SEGYTraceHeader(endian=byteorder)
        elif not hasattr(trace.stats.segy, 'trace_header'):
            warnings.warn("CREATING TRACE HEADER")
            trace.stats.segy.trace_header = SEGYTraceHeader()
        this_trace_header = trace.stats.segy.trace_header
        new_trace_header = new_trace.header
        # Again loop over all field of the trace header and if they exists, set
        # them. Ignore all additional attributes.
        for _, item, _, _ in TRACE_HEADER_FORMAT:
            if hasattr(this_trace_header, item):
                setattr(new_trace_header, item,
                        getattr(this_trace_header, item))
        starttime = trace.stats.starttime
        # Set the date of the Trace if it is not UTCDateTime(0).
        if starttime == UTCDateTime(0):
            new_trace.header.year_data_recorded = 0
            new_trace.header.day_of_year = 0
            new_trace.header.hour_of_day = 0
            new_trace.header.minute_of_hour = 0
            new_trace.header.second_of_minute = 0
        else:
            new_trace.header.year_data_recorded = starttime.year
            new_trace.header.day_of_year = starttime.julday
            new_trace.header.hour_of_day = starttime.hour
            new_trace.header.minute_of_hour = starttime.minute
            new_trace.header.second_of_minute = starttime.second
        # Set the sampling rate.
        new_trace.header.sample_interval_in_ms_for_this_trace = \
            int(trace.stats.delta * 1E6)
        # Set the data encoding and the endianness.
        new_trace.data_encoding = data_encoding
        new_trace.endian = byteorder
        # Add the trace to the SEGYFile object.
        segy_file.traces.append(new_trace)
    # Write the file
    segy_file.write(filename, data_encoding=data_encoding, endian=byteorder)


def isSU(filename):
    """
    Checks whether or not the given file is a Seismic Unix (SU) file.

    :type filename: str
    :param filename: Seismic Unix file to be checked.
    :rtype: bool
    :return: ``True`` if a Seismic Unix file.

    .. note::
        This test is rather shaky because there is no reliable identifier in a
        Seismic Unix file.
    """
    with open(filename, 'rb') as f:
        stat = autodetectEndianAndSanityCheckSU(f)
    if stat is False:
        return False
    else:
        return True


def readSU(filename, headonly=False, byteorder=None,
           unpack_trace_headers=False, **kwargs):  # @UnusedVariable
    """
    Reads a Seismic Unix (SU) file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: SU file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the header and omit the waveform
        data.
    :type byteorder: str or ``None``
    :param byteorder: Determines the endianness of the file. Either ``'>'`` for
        big endian or ``'<'`` for little endian. If it is ``None``, it will try
        to autodetect the endianness. The endianness is always valid for the
        whole file. Defaults to ``None``.
    :type unpack_trace_headers: bool, optional
    :param unpack_trace_headers: Determines whether or not all trace header
        values will be unpacked during reading. If ``False`` it will greatly
        enhance performance and especially memory usage with large files. The
        header values can still be accessed and will be calculated on the fly
        but tab completion will no longer work. Look in the headers.py for a
        list of all possible trace header values. Defaults to ``False``.
    :returns: A ObsPy :class:`~obspy.core.stream.Stream` object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/1.su_first_trace")
    >>> st #doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  #doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    ... | 2005-12-19T15:07:54.000000Z - ... | 4000.0 Hz, 8000 samples
    """
    # Read file to the internal segy representation.
    su_object = readSUFile(filename, endian=byteorder,
                           unpack_headers=unpack_trace_headers)

    # Create the stream object.
    stream = Stream()

    # Get the endianness from the first trace.
    endian = su_object.traces[0].endian
    # Loop over all traces.
    for tr in su_object.traces:
        # Create new Trace object for every segy trace and append to the Stream
        # object.
        trace = Trace()
        stream.append(trace)
        # skip data if headonly is set
        if headonly:
            trace.stats.npts = tr.npts
        else:
            trace.data = tr.data
        trace.stats.su = AttribDict()
        # If all values will be unpacked create a normal dictionary.
        if unpack_trace_headers:
            # Add the trace header as a new attrib dictionary.
            header = AttribDict()
            for key, value in tr.header.__dict__.items():
                setattr(header, key, value)
        # Otherwise use the LazyTraceHeaderAttribDict.
        else:
            # Add the trace header as a new lazy attrib dictionary.
            header = LazyTraceHeaderAttribDict(tr.header.unpacked_header,
                                               tr.header.endian)
        trace.stats.su.trace_header = header
        # Also set the endianness.
        trace.stats.su.endian = endian
        # The sampling rate should be set for every trace. It is a sample
        # interval in microseconds. The only sanity check is that is should be
        # larger than 0.
        tr_header = trace.stats.su.trace_header
        if tr_header.sample_interval_in_ms_for_this_trace > 0:
            trace.stats.delta = \
                float(tr.header.sample_interval_in_ms_for_this_trace) / \
                1E6
        # If the year is not zero, calculate the start time. The end time is
        # then calculated from the start time and the sampling rate.
        # 99 is often used as a placeholder.
        if tr_header.year_data_recorded > 0:
            year = tr_header.year_data_recorded
            # The SEG Y rev 0 standard specifies the year to be a 4 digit
            # number.  Before that it was unclear if it should be a 2 or 4
            # digit number. Old or wrong software might still write 2 digit
            # years. Every number <30 will be mapped to 2000-2029 and every
            # number between 30 and 99 will be mapped to 1930-1999.
            if year < 100:
                if year < 30:
                    year += 2000
                else:
                    year += 1900
            julday = tr_header.day_of_year
            julday = tr_header.day_of_year
            hour = tr_header.hour_of_day
            minute = tr_header.minute_of_hour
            second = tr_header.second_of_minute
            trace.stats.starttime = UTCDateTime(
                year=year, julday=julday, hour=hour, minute=minute,
                second=second)
    return stream


def writeSU(stream, filename, byteorder=None, **kwargs):  # @UnusedVariable
    """
    Writes a Seismic Unix (SU) file from given ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of file to write.
    :type byteorder: str or ``None``
    :param byteorder: Determines the endianness of the file. Either ``'>'`` for
        big endian or ``'<'`` for little endian. If is ``None``, it will either
        be the endianness of the first Trace or if that is also not set, it
        will be big endian. A mix between little and big endian for the headers
        and traces is currently not supported.

    This function will automatically set the data encoding field of the binary
    file header so the user does not need to worry about it.
    """
    # Check that the dtype for every Trace is correct.
    for trace in stream:
        # Check the dtype.
        if trace.data.dtype != np.float32:
            msg = """
            The dtype of the data is not float32.  You need to manually convert
            the dtype. Please refer to the obspy.segy manual for more details.
            """.strip()
            raise SEGYCoreWritingError(msg)
        # Check the sample interval.
        if trace.stats.delta > MAX_INTERVAL_IN_SECONDS:
            msg = """
            Seismic Unix supports a maximum interval of %s seconds in between
            two samples (trace.stats.delta value).
            """.strip()
            msg = msg % MAX_INTERVAL_IN_SECONDS
            raise SEGYSampleIntervalError(msg)

    # Figure out endianness and the encoding of the textual file header.
    if byteorder is None:
        if hasattr(stream[0].stats, 'su') and hasattr(stream[0].stats.su,
                                                      'endian'):
            byteorder = stream[0].stats.su.endian
        else:
            byteorder = '>'

    # Loop over all Traces and create a SEGY File object.
    su_file = SUFile()
    # Add all traces.
    for trace in stream:
        new_trace = SEGYTrace()
        new_trace.data = trace.data
        # Use header saved in stats if one exists.
        if hasattr(trace.stats, 'su') and \
           hasattr(trace.stats.su, 'trace_header'):
            this_trace_header = trace.stats.su.trace_header
        else:
            this_trace_header = AttribDict()
        new_trace_header = new_trace.header
        # Again loop over all field of the trace header and if they exists, set
        # them. Ignore all additional attributes.
        for _, item, _, _ in TRACE_HEADER_FORMAT:
            if hasattr(this_trace_header, item):
                setattr(new_trace_header, item,
                        getattr(this_trace_header, item))
        starttime = trace.stats.starttime
        # Set some special attributes, e.g. the sample count and other stuff.
        new_trace_header.number_of_samples_in_this_trace = trace.stats.npts
        new_trace_header.sample_interval_in_ms_for_this_trace = \
            int(round((trace.stats.delta * 1E6)))
        # Set the date of the Trace if it is not UTCDateTime(0).
        if starttime == UTCDateTime(0):
            new_trace.header.year_data_recorded = 0
            new_trace.header.day_of_year = 0
            new_trace.header.hour_of_day = 0
            new_trace.header.minute_of_hour = 0
            new_trace.header.second_of_minute = 0
        else:
            new_trace.header.year_data_recorded = starttime.year
            new_trace.header.day_of_year = starttime.julday
            new_trace.header.hour_of_day = starttime.hour
            new_trace.header.minute_of_hour = starttime.minute
            new_trace.header.second_of_minute = starttime.second
        # Set the data encoding and the endianness.
        new_trace.endian = byteorder
        # Add the trace to the SEGYFile object.
        su_file.traces.append(new_trace)
    # Write the file
    su_file.write(filename, endian=byteorder)


def __segy_trace__str__(self, *args, **kwargs):
    """
    Monkey patch for the __str__ method of the Trace object. SEGY object do not
    have network, station, channel codes. It just prints the trace sequence
    number within the line.
    """
    try:
        out = "%s" % (
            'Seq. No. in line: %4i' %
            self.stats.segy.trace_header.trace_sequence_number_within_line)
    except (KeyError, AttributeError):
        # fall back if for some reason the segy attribute does not exists
        return getattr(Trace, '__original_str__')(self, *args, **kwargs)
    # output depending on delta or sampling rate bigger than one
    if self.stats.sampling_rate < 0.1:
        if hasattr(self.stats, 'preview') and self.stats.preview:
            out = out + ' | '\
                "%(starttime)s - %(endtime)s | " + \
                "%(delta).1f s, %(npts)d samples [preview]"
        else:
            out = out + ' | '\
                "%(starttime)s - %(endtime)s | " + \
                "%(delta).1f s, %(npts)d samples"
    else:
        if hasattr(self.stats, 'preview') and self.stats.preview:
            out = out + ' | '\
                "%(starttime)s - %(endtime)s | " + \
                "%(sampling_rate).1f Hz, %(npts)d samples [preview]"
        else:
            out = out + ' | '\
                "%(starttime)s - %(endtime)s | " + \
                "%(sampling_rate).1f Hz, %(npts)d samples"
    # check for masked array
    if np.ma.count_masked(self.data):
        out += ' (masked)'
    return out % (self.stats)


class LazyTraceHeaderAttribDict(AttribDict):
    """
    This version of AttribDict will unpack header values only if needed.

    This saves a huge amount of memory. The disadvantage is that it is no
    longer possible to use tab completion in e.g. ipython.

    This version is used for the SEGY/SU trace headers.
    """
    readonly = []

    def __init__(self, unpacked_header, unpacked_header_endian, data={}):
        dict.__init__(data)
        self.update(data)
        self.__dict__['unpacked_header'] = unpacked_header
        self.__dict__['endian'] = unpacked_header_endian

    def __getitem__(self, name):
        # Return if already set.
        if name in self.__dict__:
            if name in self.readonly:
                return self.__dict__[name]
            return super(AttribDict, self).__getitem__(name)
        # Otherwise try to unpack them.
        try:
            index = TRACE_HEADER_KEYS.index(name)
        # If not found raise an attribute error.
        except ValueError:
            msg = "'%s' object has no attribute '%s'" % \
                (self.__class__.__name__, name)
            raise AttributeError(msg)
        # Unpack the one value and set the class attribute so it will does not
        # have to unpacked again if accessed in the future.
        length, name, special_format, start = TRACE_HEADER_FORMAT[index]
        string = self.__dict__['unpacked_header'][start: start + length]
        attribute = unpack_header_value(self.__dict__['endian'], string,
                                        length, special_format)
        setattr(self, name, attribute)
        return attribute

    __getattr__ = __getitem__

    def __deepcopy__(self, *args, **kwargs):  # @UnusedVariable, see #689
        ad = self.__class__(
            unpacked_header=deepcopy(self.__dict__['unpacked_header']),
            unpacked_header_endian=deepcopy(self.__dict__['endian']),
            data=dict((k, deepcopy(v)) for k, v in self.__dict__.items()
                      if k not in ('unpacked_data', 'endian')))
        return ad

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)


# Monkey patch the __str__ method for the all Trace instances used in the
# following.
# XXX: Check if this is not messing anything up. Patching every single
# instance did not reliably work.
setattr(Trace, '__original_str__', Trace.__str__)
setattr(Trace, '__str__', __segy_trace__str__)
