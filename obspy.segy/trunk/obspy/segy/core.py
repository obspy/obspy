# -*- coding: utf-8 -*-
"""
SEG Y bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""


from obspy.core import Stream, Trace, UTCDateTime, Stats, AttribDict
from obspy.segy.segy import readSEGY as readSEGYrev1
from obspy.segy.segy import SEGYError, SEGYFile, SEGYBinaryFileHeader
from obspy.segy.segy import SEGYTrace
from obspy.segy.header import BINARY_FILE_HEADER_FORMAT, TRACE_HEADER_FORMAT
from obspy.segy.header import DATA_SAMPLE_FORMAT_CODE_DTYPE

from StringIO import StringIO

from copy import deepcopy
import numpy as np
from struct import unpack
import os

# Valid data format codes as specified in the SEGY rev1 manual.
VALID_FORMATS = [1, 2, 3, 4, 5, 8]


class SEGYCoreWritingError(SEGYError):
    """
    Raised if the writing of the Stream object fails due to some reason.
    """
    pass


def isSEGY(filename):
    """
    Checks whether a file is a SEGY file or not. Returns True or False.

    Parameters
    ----------

    filename : string
        Name of the SEGY file to be checked.
    """
    # This is a very weak test. It tests two things: First if the data sample
    # format code is valid. This is also used to determine the endianness. This
    # is then used to check if the sampling interval is set to any sane number
    # greater than 0 and that the number of samples per trace is greater than
    # 0.
    try:
        temp = open(filename, 'rb')
        temp.seek(3216)
        sample_interval = temp.read(2)
        temp.seek(2, 1)
        samples_per_trace = temp.read(2)
        temp.seek(2, 1)
        data_format_code = temp.read(2)
        temp.close()
    except:
        return False
    # Unpack using big endian first and check if it is valid.
    format = unpack('>h', data_format_code)[0]
    if format in VALID_FORMATS:
        endian = '>'
    else:
        format = unpack('<h', data_format_code)[0]
        if format in VALID_FORMATS:
            endian = '<'
        else:
            return False
    return True


def readSEGY(filename, endian=None, textual_header_encoding=None):
    """
    Reads a SEGY file and returns an ObsPy Stream object.

    This function should NOT be called directly, it registers via the
    ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    Parameters
    ----------
    filename : string
        SEG Y rev1 file to be read.
    endian : string
        Determines the endianness of the file. Either '>' for big endian or '<'
        for little endian. If it is None, obspy.segy will try to autodetect the
        endianness. The endianness is always valid for the whole file.
    textual_header_encoding :
        The encoding of the textual header.  Either 'EBCDIC', 'ASCII' or None.
        If it is None, autodetection will be attempted.

    Returns
    -------
    stream : :class:`~obspy.core.stream.Stream`
        A ObsPy Stream object.

    Example
    -------
    >>> from obspy.core import read # doctest: +SKIP
    >>> st = read("segy_file") # doctest: +SKIP
    """
    # Monkey patch the __str__ method for the all Trace instances used in the
    # following.
    # XXX: Check if this is not messing anything up. Patching every single
    # instance did not reliably work.
    setattr(Trace, '__str__', segy_trace__str__)

    # Read file to the internal segy representation.
    segy_object = readSEGYrev1(filename, endian=endian,
                               textual_header_encoding=textual_header_encoding)

    # Create the stream object.
    stream = Stream()
    # SEGY has several file headers that apply to all traces. Currently every
    # header will be written to every single Trace and writing will be
    # supported only if the headers are identical for every Trace.
    # Get the textual file header.
    textual_file_header = segy_object.textual_file_header
    # The binary file header will be a new AttribDict
    binary_file_header = AttribDict()
    for key, value in segy_object.binary_file_header.__dict__.iteritems():
        setattr(binary_file_header, key, value)
    # Get the data encoding and the endianness from the first trace.
    data_encoding = segy_object.traces[0].data_encoding
    endian = segy_object.traces[0].endian
    textual_file_header_encoding = segy_object.textual_header_encoding.upper()
    # Loop over all traces.
    for tr in segy_object.traces:
        # Create new Trace object for every segy trace and append to the Stream
        # object.
        trace = Trace()
        stream.append(trace)
        trace.data = tr.data
        trace.stats.segy = AttribDict()
        # Add the trace header as a new attrib dictionary.
        header = AttribDict()
        for key, value in tr.header.__dict__.iteritems():
            setattr(header, key, value)
        trace.stats.segy.trace_header = header
        # Add copies of the file wide headers. Deepcopies are necessary because
        # a change in one part of the header should only affect that one Trace.
        trace.stats.segy.textual_file_header = deepcopy(textual_file_header)
        trace.stats.segy.binary_file_header = deepcopy(binary_file_header)
        # Also set the data encoding, endianness and the encoding of the
        # textual_file_header.
        trace.stats.segy.data_encoding = data_encoding
        trace.stats.segy.endian = endian
        trace.stats.segy.textual_file_header_encoding = \
            textual_file_header_encoding
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
            julday = tr_header.day_of_year
            hour = tr_header.hour_of_day
            minute = tr_header.minute_of_hour
            second = tr_header.second_of_minute
            trace.stats.starttime = UTCDateTime(year=year, julday=julday,
                                    hour=hour, minute=minute, second=second)
    return stream


def writeSEGY(stream, filename, data_encoding=None, endian=None,
              textual_header_encoding=None):
    """
    Writes a SEGY file from given ObsPy Stream object.

    This function should NOT be called directly, it registers via the ObsPy
    :meth:`~obspy.core.stream.Stream.write` method of an ObsPy Stream object,
    call this instead.

    This function will automatically set the data encoding field of the binary
    file header so the user does not need to worry about it.

    Parameters
    ----------
    stream : :class:`~obspy.core.stream.Stream`
        A ObsPy Stream object.
    filename : string
        Name of SEGY file to be written.
    data_encoding : int
        The data encoding is an integer with the following currently supported
        meaning.

        1: 4 byte IBM floating points (float32)
        2: 4 byte Integers (int32)
        3: 2 byte Integer (int16)
        5: 4 byte IEEE floating points (float32)

        The value in the brackets is the necessary dtype of the data. ObsPy
        will now automatically convert the data because data might change/loose
        precision during the conversion so the user has to take care of the
        correct dtype.

        If it is None, the value of the first Trace will be used for all
        consecutive Traces. If it is None for the first Trace, 1 (IBM floating
        point numbers) will be used. Different data encodings for different
        traces are currently not supported because these will most likely not
        be readable by other software.
    endian : string
        Either '<' (little endian), '>' (big endian), or None

        If is None, it will either be the endianness of the first Trace or if
        that is also not set, it will be big endian. A mix between little and
        big endian for the headers and traces is currently not supported.
    textual_header_encoding : string
        The encoding of the textual header. Either 'EBCDIC', 'ASCII' or None.

        If it is None, the textual_file_header_encoding attribute in the
        stats.segy dictionary of the first Trace is used and if that is not
        set, ASCII will be used.
    """
    # Some sanity checks to catch invalid arguments/keyword arguments.
    if data_encoding is not None and data_encoding not in VALID_FORMATS:
        msg = "Invalid data encoding."
        raise SEGYCoreWritingError(msg)
    # Figure out the data encoding if it is not set.
    if data_encoding is None:
        if hasattr(stream[0].stats, 'segy') and hasattr(stream[0].stats.segy,
                                                        'data_encoding'):
            data_encoding = stream[0].stats.segy.data_encoding
        if hasattr(stream[0].stats, 'segy') and hasattr(stream[0].stats.segy,
                                                        'binary_file_header'):
            data_encoding = \
            stream[0].stats.segy.binary_file_header.data_sample_format_code
        else:
            data_encoding = 1
    # Valid dtype for the data encoding. If None is given the encoding of the
    # first trace is used.
    valid_dtype = DATA_SAMPLE_FORMAT_CODE_DTYPE[data_encoding]
    # Check that the textual file header and the binary file header exist for
    # each Trace and that they are identical for each Trace. Also makes sure
    # that the dtype is for every Trace is correct.
    for trace in stream:
        if not hasattr(trace.stats, 'segy') or \
           not hasattr(trace.stats.segy, 'textual_file_header') or \
           not hasattr(trace.stats.segy, 'binary_file_header'):
            msg = """
            Trace.stats.segy.textual_file_header and
            Trace.stats.segy.binary_file_header need to exists for every Trace
            to be able to write SEGY.

            Please refer to the ObsPy documentation for further information.
            """.strip()
            raise SEGYCoreWritingError(msg)
        # Check if all textual headers are identical.
        if trace.stats.segy.textual_file_header != \
           stream[0].stats.segy.textual_file_header:
            msg = """
            The Trace.stats.segy.textual_header needs to be identical for every
            Trace in the Stream object to be able to write SEGY.

            Please refer to the ObsPy documentation for further information.
            """.strip()
            raise SEGYCoreWritingError(msg)
        # Some for the binary file headers.
        if trace.stats.segy.binary_file_header != \
           stream[0].stats.segy.binary_file_header:
            msg = """
            The Trace.stats.segy.textual_header needs to be identical for every
            Trace in the Stream object to be able to write SEGY.

            Please refer to the ObsPy documentation for further information.
            """.strip()
            raise SEGYCoreWritingError(msg)
        # Check the dtype.
        if trace.data.dtype != valid_dtype:
            msg = """
            The dtype of the data and the chosen data_encoding do not match.
            You need to manually convert the dtype if you want to use that
            data_encoding. Please refer to the obspy.segy manual for more
            details.
            """.strip()
            raise SEGYCoreWritingError(msg)

    # Figure out endianness and the encoding of the textual file header.
    if endian is None:
        if hasattr(stream[0].stats, 'segy') and hasattr(stream[0].stats.segy,
                                                        'endian'):
            endian = stream[0].stats.segy.endian
        else:
            endian = '>'
    if textual_header_encoding is None:
        if hasattr(stream[0].stats, 'segy') and hasattr(stream[0].stats.segy,
                                            'textual_file_header_encoding'):
            textual_header_encoding = \
                stream[0].stats.segy.textual_file_header_encoding
        else:
            textual_header_encoding = 'ASCII'

    # Loop over all Traces and create a SEGY File object.
    segy_file = SEGYFile()
    # Set the file wide headers.
    segy_file.textual_file_header = stream[0].stats.segy.textual_file_header
    segy_file.textual_header_encoding = \
            textual_header_encoding
    binary_header = SEGYBinaryFileHeader()
    this_binary_header = stream[0].stats.segy.binary_file_header
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
        this_trace_header = trace.stats.segy.trace_header
        new_trace_header = new_trace.header
        # Again loop over all field of the trace header and if they exists, set
        # them. Ignore all additional attributes.
        for _, item in TRACE_HEADER_FORMAT:
            if hasattr(this_trace_header, item):
                setattr(new_trace_header, item,
                        getattr(this_trace_header, item))
        # Set the data encoding and the endianness.
        new_trace.data_encoding = data_encoding
        new_trace.endian = endian
        # Add the trace to the SEGYFile object.
        segy_file.traces.append(new_trace)
    # Write the file
    segy_file.write(filename, data_encoding=data_encoding, endian=endian)


def segy_trace__str__(self, *args, **kwargs):
    """
    Monkey patch for the __str__ method of the Trace object. SEGY object to not
    have network, station, channel codes. It just prints the trace sequence
    number within the line.
    """
    out = "%s" % ('Seq. No. in line: %4i' % \
             self.stats.segy.trace_header.trace_sequence_number_within_line)
    # output depending on delta or sampling rate bigger than one
    if self.stats.sampling_rate < 0.1:
        if hasattr(self.stats, 'preview')  and self.stats.preview:
            out = out + ' | '\
                  "%(starttime)s - %(endtime)s | " + \
                  "%(delta).1f s, %(npts)d samples [preview]"
        else:
            out = out + ' | '\
                  "%(starttime)s - %(endtime)s | " + \
                  "%(delta).1f s, %(npts)d samples"
    else:
        if hasattr(self.stats, 'preview')  and self.stats.preview:
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
