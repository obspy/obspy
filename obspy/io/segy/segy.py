# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
#  Filename: seg.py
#  Purpose: Routines for reading and writing SEG Y files.
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#
# Copyright (C) 2010 Lion Krischer
# --------------------------------------------------------------------
"""
Routines to read and write SEG Y rev 1 encoded seismic data files.
"""
import io
import os
from struct import pack, unpack
import warnings

import numpy as np

from obspy import Trace, UTCDateTime
from obspy.core import AttribDict

from .header import (BINARY_FILE_HEADER_FORMAT,
                     DATA_SAMPLE_FORMAT_PACK_FUNCTIONS,
                     DATA_SAMPLE_FORMAT_SAMPLE_SIZE,
                     DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS, ENDIAN,
                     TRACE_HEADER_FORMAT, TRACE_HEADER_KEYS)
from .unpack import OnTheFlyDataUnpacker
from .util import unpack_header_value, _pack_attribute_nicer_exception


class SEGYError(Exception):
    """
    Base SEGY exception class.
    """
    pass


class SEGYTraceHeaderTooSmallError(SEGYError):
    """
    Raised if the trace header is not the required 240 byte long.
    """
    pass


class SEGYTraceReadingError(SEGYError):
    """
    Raised if there is not enough data left in the file to unpack the data
    according to the values read from the header.
    """
    pass


class SEGYTraceOnTheFlyDataUnpackingError(SEGYError):
    """
    Raised if attempting to unpack trace data but no ``unpack_data()`` function
    exists.
    """
    pass


class SEGYWritingError(SEGYError):
    """
    Raised if the trace header is not the required 240 byte long.
    """
    pass


class SEGYWarning(UserWarning):
    """
    SEG Y warnings base class.
    """
    pass


class SEGYInvalidTextualHeaderWarning(SEGYWarning):
    """
    Warning that is raised if an invalid textual header is about to be written.
    """
    pass


class SEGYFile(object):
    """
    Class that internally handles SEG Y files.
    """
    def __init__(self, file=None, endian=None, textual_header_encoding=None,
                 unpack_headers=False, headonly=False, read_traces=True):
        """
        Class that internally handles SEG Y files.

        :param file: A file like object with the file pointer set at the
            beginning of the SEG Y file. If file is None, an empty SEGYFile
            object will be initialized.
        :param endian: The endianness of the file. If None, autodetection will
            be used.
        :param textual_header_encoding: The encoding of the textual header.
            Either 'EBCDIC', 'ASCII' or None. If it is None, autodetection will
            be attempted. If it is None and file is also None, it will default
            to 'ASCII'.
        :type unpack_headers: bool
        :param unpack_headers: Determines whether or not all headers will be
            unpacked during reading the file. Has a huge impact on the memory
            usage and the performance. They can be unpacked on-the-fly after
            being read. Defaults to False.
        :type headonly: bool
        :param headonly: Determines whether or not the actual data records
            will be read and unpacked. Has a huge impact on memory usage. Data
            will not be unpackable on-the-fly. Defaults to False.
        :type read_traces: bool
        :param read_traces: Data traces will only be read if this is set to
            ``True``. The data will be completely ignored if this is set to
            ``False``.
        """
        if file is None:
            self._create_empty_segy_file_object()
            # Set the endianness to big.
            if endian is None:
                self.endian = '>'
            else:
                self.endian = ENDIAN[endian]
            # And the textual header encoding to ASCII.
            if textual_header_encoding is None:
                self.textual_header_encoding = 'ASCII'
            self.textual_header = b''
            return
        self.file = file
        # If endian is None autodetect is.
        if not endian:
            self._autodetect_endianness()
        else:
            self.endian = ENDIAN[endian]
        # If the textual header encoding is None, autodetection will be used.
        self.textual_header_encoding = textual_header_encoding
        # Read the headers.
        self._read_headers()
        # Read the actual traces.
        if read_traces:
            [i for i in self._read_traces(
                unpack_headers=unpack_headers, headonly=headonly)]

    def __str__(self):
        """
        Prints some information about the SEG Y file.
        """
        return '%i traces in the SEG Y structure.' % len(self.traces)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def _autodetect_endianness(self):
        """
        Tries to automatically determine the endianness of the file at hand.
        """
        pos = self.file.tell()
        # Jump to the data sample format code.
        self.file.seek(3224, 1)
        format = unpack(b'>h', self.file.read(2))[0]
        # Check if valid.
        if format in DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS.keys():
            self.endian = '>'
        # Else test little endian.
        else:
            self.file.seek(-2, 1)
            format = unpack(b'<h', self.file.read(2))[0]
            if format in DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS.keys():
                self.endian = '<'
            else:
                msg = 'Unable to determine the endianness of the file. ' + \
                      'Please specify it.'
                raise SEGYError(msg)
        # Jump to previous position.
        self.file.seek(pos, 0)

    def _create_empty_segy_file_object(self):
        """
        Creates an empty SEGYFile object.
        """
        self.textual_file_header = b''
        self.binary_file_header = None
        self.traces = []

    def _read_textual_header(self):
        """
        Reads the textual header.
        """
        # The first 3200 byte are the textual header.
        textual_header = self.file.read(3200)
        # The data can either be saved as plain ASCII or EBCDIC. The first
        # character always is mostly 'C' and therefore used to check the
        # encoding. Sometimes is it not C but also cannot be decoded from
        # EBCDIC so it is treated as ASCII and all empty symbols are removed.
        #
        # Also check the revision number and textual header end markers for
        # the "C" as they might be set when the first byte is not.
        if not self.textual_header_encoding:
            if not (b'C' in (textual_header[0:1], textual_header[3040:3041],
                             textual_header[3120:3120])):
                try:
                    textual_header = \
                        textual_header.decode('EBCDIC-CP-BE').encode('ascii')
                    # If this worked, the encoding is EBCDIC.
                    self.textual_header_encoding = 'EBCDIC'
                except UnicodeEncodeError:
                    textual_header = textual_header
                    # Otherwise it is ASCII.
                    self.textual_header_encoding = 'ASCII'
            else:
                # Otherwise the encoding will also be ASCII.
                self.textual_header_encoding = 'ASCII'
        elif self.textual_header_encoding.upper() == 'EBCDIC':
            textual_header = \
                textual_header.decode('EBCDIC-CP-BE').encode('ascii')
        elif self.textual_header_encoding.upper() != 'ASCII':
            msg = """
            The textual_header_encoding has to be either ASCII, EBCDIC or None
            for autodetection. ASCII, EBCDIC or None for autodetection.
            """.strip()
            raise SEGYError(msg)
        # Finally set it.
        self.textual_file_header = textual_header

    def _read_headers(self):
        """
        Reads the textual and binary file headers starting at the current file
        pointer position.
        """
        # Read the textual header.
        self._read_textual_header()
        # The next 400 bytes are from the Binary File Header.
        binary_file_header = self.file.read(400)
        bfh = SEGYBinaryFileHeader(binary_file_header, self.endian)
        self.binary_file_header = bfh
        self.data_encoding = self.binary_file_header.data_sample_format_code
        # If bytes 3506-3506 are not zero, an extended textual header follows
        # which is not supported so far.
        if bfh.number_of_3200_byte_ext_file_header_records_following != 0:
            msg = 'Extended textual headers are not yet supported. ' + \
                'Please contact the developers.'
            raise NotImplementedError(msg)

    def write(self, file, data_encoding=None, endian=None):
        """
        Write a SEG Y file to file which is either a file like object with a
        write method or a filename string.

        If data_encoding or endian is set, these values will be enforced.
        """
        if not hasattr(file, 'write'):
            with open(file, 'wb') as file:
                self._write(file, data_encoding=data_encoding, endian=endian)
            return
        self._write(file, data_encoding=data_encoding, endian=endian)

    def _write(self, file, data_encoding=None, endian=None):
        """
        Writes SEG Y to a file like object.

        If data_encoding or endian is set, these values will be enforced.
        """
        # Write the textual header.
        self._write_textual_header(file)

        # Write certain fields in the binary header if they are not set. Most
        # fields will be written using the data from the first trace. It is
        # usually better to set the header manually!
        if self.binary_file_header.number_of_data_traces_per_ensemble <= 0:
            self.binary_file_header.number_of_data_traces_per_ensemble = \
                len(self.traces)
        if self.binary_file_header.sample_interval_in_microseconds <= 0:
            self.binary_file_header.sample_interval_in_microseconds = \
                self.traces[0].header.sample_interval_in_ms_for_this_trace
        if self.binary_file_header.number_of_samples_per_data_trace <= 0:
            self.binary_file_header.number_of_samples_per_data_trace = \
                len(self.traces[0].data)

        # Always set the SEGY Revision number to 1.0 (hex-coded).
        self.binary_file_header.seg_y_format_revision_number = 256
        # Set the fixed length flag to zero if all traces have NOT the same
        # length. Leave unchanged otherwise.
        if len(set([len(tr.data) for tr in self.traces])) != 1:
            self.binary_file_header.fixed_length_trace_flag = 0
        # Extended textual headers are not supported by ObsPy so far.
        self.binary_file_header.\
            number_of_3200_byte_ext_file_header_records_following = 0
        # Enforce the encoding
        if data_encoding:
            self.binary_file_header.data_sample_format_code = data_encoding

        # Write the binary header.
        self.binary_file_header.write(file, endian=endian)
        # Write all traces.
        for trace in self.traces:
            trace.write(file, data_encoding=data_encoding, endian=endian)

    def _write_textual_header(self, file):
        """
        Write the textual header in various encodings. The encoding will depend
        on self.textual_header_encoding. If self.textual_file_header is too
        small it will be padded with zeros. If it is too long or an invalid
        encoding is specified an exception will be raised.
        """
        textual_header = self.textual_file_header

        # Convert to ASCII bytes if necessary - this will raise an error in
        # case the textual file header has no representation in ASCII - this
        # is then the users responsibility.
        if hasattr(textual_header, "encode"):
            textual_header = textual_header.encode()

        length = len(textual_header)
        # Append spaces to the end if its too short.
        if length < 3200:
            textual_header = textual_header + b' ' * (3200 - length)
        elif length == 3200:
            textual_header = textual_header
        # The length must not exceed 3200 byte.
        else:
            msg = 'self.textual_file_header is not allowed to be longer ' + \
                  'than 3200 bytes'
            raise SEGYWritingError(msg)

        # Assert the encoding.
        enc = self.textual_header_encoding.upper()

        # Make sure revision number and end header marker are present. If
        # not: add them - if something else is already present, raise a
        # warning but don't do anything.

        # Make sure the textual header has the required fields.
        revision_number = textual_header[3200 - 160:3200 - 146].decode()
        end_header_mark = textual_header[3200 - 80:3200 - 58]
        if revision_number != "C39 SEG Y REV1":
            if revision_number.strip() in ("", "C", "C39"):
                textual_header = textual_header[:3200 - 160] + \
                    b"C39 SEG Y REV1" + textual_header[3200 - 146:]
            else:
                # Raise warning but don't do anything.
                msg = ("The revision number in the textual header should be "
                       "set as 'C39 SEG Y REV1' for a fully valid SEG-Y "
                       "file. It is set to '%s' which will be written to the "
                       "file. Please change it if you want a fully valid file."
                       % revision_number)
                warnings.warn(msg, SEGYInvalidTextualHeaderWarning)

        desired_end_header_mark = b"C40 END TEXTUAL HEADER" if enc == "ASCII" \
            else b"C40 END EBCDIC        "

        if end_header_mark != desired_end_header_mark:
            if end_header_mark.strip() in (b"", b"C", b"C40"):
                textual_header = textual_header[:3200 - 80] + \
                    desired_end_header_mark + textual_header[3200 - 58:]
            else:
                # Raise warning but don't do anything.
                msg = ("The end header mark in the textual header should be "
                       "set as 'C40 END TEXTUAL HEADER' or as "
                       "'C40 END EBCDIC        ' for a fully valid "
                       "SEG-Y file. It is set to '%s' which will be written "
                       "to the file. Please change it if you want a fully "
                       "valid file."
                       % end_header_mark.decode())
                warnings.warn(msg, SEGYInvalidTextualHeaderWarning)

        # Finally encode the header if necessary.
        if enc == 'ASCII':
            pass
        elif enc == 'EBCDIC':
            textual_header = \
                textual_header.decode('ascii').encode('EBCDIC-CP-BE')
        # Should not happen.
        else:
            msg = 'self.textual_header_encoding has to be either ASCII or ' + \
                  'EBCDIC.'
            raise SEGYWritingError(msg)

        file.write(textual_header)

    def _read_traces(self, unpack_headers=False, headonly=False,
                     yield_each_trace=False):
        """
        Reads the actual traces starting at the current file pointer position
        to the end of the file.

        :type unpack_headers: bool
        :param unpack_headers: Determines whether or not all headers will be
            unpacked during reading the file. Has a huge impact on the memory
            usage and the performance. They can be unpacked on-the-fly after
            being read. Defaults to False.

        :type headonly: bool
        :param headonly: Determines whether or not the actual data records
            will be read and unpacked. Has a huge impact on memory usage. Data
            will not be unpackable on-the-fly after reading the file.
            Defaults to False.

        :type yield_each_trace: bool
        :param yield_each_trace: If True, it will yield each trace after it
            has been read. This enables a simple implementation of a
            streaming interface to read SEG-Y files. Read traces will no
            longer be collected in ``self.traces`` list if this is set to
            ``True``.
        """
        self.traces = []
        # Determine the filesize once.
        if isinstance(self.file, io.BytesIO):
            pos = self.file.tell()
            self.file.seek(0, 2)  # go t end of file
            filesize = self.file.tell()
            self.file.seek(pos, 0)
        else:
            filesize = os.fstat(self.file.fileno())[6]
        # Big loop to read all data traces.
        while True:
            # Read and as soon as the trace header is too small abort.
            try:
                trace = SEGYTrace(self.file, self.data_encoding, self.endian,
                                  unpack_headers=unpack_headers,
                                  filesize=filesize, headonly=headonly)
                if yield_each_trace:
                    yield trace
                else:
                    self.traces.append(trace)
            except SEGYTraceHeaderTooSmallError:
                break


class SEGYBinaryFileHeader(object):
    """
    Parses the binary file header at the given starting position.
    """
    def __init__(self, header=None, endian='>'):
        """
        """
        self.endian = endian
        if header is None:
            self._create_empty_binary_file_header()
            return
        self._read_binary_file_header(header)

    def _read_binary_file_header(self, header):
        """
        Reads the binary file header and stores every value in a class
        attribute.
        """
        pos = 0
        for item in BINARY_FILE_HEADER_FORMAT:
            length, name, _ = item
            string = header[pos: pos + length]
            pos += length
            # Unpack according to different lengths.
            if length == 2:
                format = ('%sh' % self.endian).encode('ascii', 'strict')
                # Set the class attribute.
                setattr(self, name, unpack(format, string)[0])
            # Update: Seems to be correct. Two's complement integers seem to be
            # the common way to store integer values.
            elif length == 4:
                format = ('%si' % self.endian).encode('ascii', 'strict')
                # Set the class attribute.
                setattr(self, name, unpack(format, string)[0])
            # The other value are the unassigned values. As it is unclear how
            # these are formatted they will be stored as strings.
            elif name.startswith('unassigned'):
                # These are only the unassigned fields.
                format = 'h' * (length // 2)
                # Set the class attribute.
                setattr(self, name, string)
            # Should not happen.
            else:
                raise Exception

    def __str__(self):
        """
        Convenience method to print the binary file header.
        """
        final_str = ["Binary File Header:"]
        for item in BINARY_FILE_HEADER_FORMAT:
            final_str.append("\t%s: %s" % (item[1],
                                           str(getattr(self, item[1]))))
        return "\n".join(final_str)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def write(self, file, endian=None):
        """
        Writes the header to an open file like object.
        """
        if endian is None:
            endian = self.endian
        for item in BINARY_FILE_HEADER_FORMAT:
            length, name, _ = item
            # Unpack according to different lengths.
            if length == 2:
                format = ('%sh' % endian).encode('ascii', 'strict')
                # Write to file.
                file.write(_pack_attribute_nicer_exception(self, name, format))
            # Update: Seems to be correct. Two's complement integers seem to be
            # the common way to store integer values.
            elif length == 4:
                format = ('%si' % endian).encode('ascii', 'strict')
                # Write to file.
                file.write(_pack_attribute_nicer_exception(self, name, format))
            # These are the two unassigned values in the binary file header.
            elif name.startswith('unassigned'):
                temp = getattr(self, name)
                if not isinstance(temp, bytes):
                    temp = str(temp).encode('ascii', 'strict')
                temp_length = len(temp)
                # Pad to desired length if necessary.
                if temp_length != length:
                    temp += b'\x00' * (length - temp_length)
                file.write(temp)
            # Should not happen.
            else:
                raise Exception

    def _create_empty_binary_file_header(self):
        """
        Just fills all necessary class attributes with zero.
        """
        for _, name, _ in BINARY_FILE_HEADER_FORMAT:
            setattr(self, name, 0)


class SEGYTrace(object):
    """
    Convenience class that internally handles a single SEG Y trace.
    """
    def __init__(self, file=None, data_encoding=4, endian='>',
                 unpack_headers=False, filesize=None, headonly=False):
        """
        Convenience class that internally handles a single SEG Y trace.

        :param file: Open file like object with the file pointer of the
            beginning of a trace. If it is None, an empty trace will be
            created.
        :param data_encoding: The data sample format code as defined in the
            binary file header:

            1:
                4 byte IBM floating point
            2:
                4 byte Integer, two's complement
            3:
                2 byte Integer, two's complement
            4:
                4 byte Fixed point with gain
            5:
                4 byte IEEE floating point
            8:
                1 byte Integer, two's complement

            Defaults to 4.
        :type big_endian: bool
        :param big_endian: True means the header is encoded in big endian and
            False corresponds to a little endian header.
        :type unpack_headers: bool
        :param unpack_headers: Determines whether or not all headers will be
            unpacked during reading the file. Has a huge impact on the memory
            usage and the performance. They can be unpacked on-the-fly after
            being read. Defaults to False.
        :type filesize: int
        :param filesize: Filesize of the file. If not given it will be
            determined using fstat which is slow.
        :param headonly: bool
        :param headonly: Determines whether or not the actual data records
            will be read and unpacked. Has a huge impact on memory usage. Data
            will not be unpackable on-the-fly after reading the file.
            Defaults to False.
        """
        self.endian = endian
        self.data_encoding = data_encoding
        # If None just return empty structure.
        if file is None:
            self._create_empty_trace()
            return
        self.file = file
        # Set the filesize if necessary.
        if filesize:
            self.filesize = filesize
        else:
            if isinstance(self.file, io.BytesIO):
                _pos = self.file.tell()
                self.file.seek(0, 2)
                self.filesize = self.file.tell()
                self.file.seek(_pos)
            else:
                self.filesize = os.fstat(self.file.fileno())[6]
        # Otherwise read the file.
        self._read_trace(unpack_headers=unpack_headers, headonly=headonly)

    def _read_trace(self, unpack_headers=False, headonly=False):
        """
        Reads the complete next header starting at the file pointer at
        self.file.

        :type unpack_headers: bool
        :param unpack_headers: Determines whether or not all headers will be
            unpacked during reading the file. Has a huge impact on the memory
            usage and the performance. They can be unpacked on-the-fly after
            being read. Defaults to False.
        :type headonly: bool
        :param headonly: Determines whether or not the actual data records
            will be read and unpacked. Has a huge impact on memory usage. Data
            will not be unpackable on-the-fly after reading the file.
            Defaults to False.
        """
        trace_header = self.file.read(240)
        # Check if it is smaller than 240 byte.
        if len(trace_header) != 240:
            msg = 'The trace header needs to be 240 bytes long'
            raise SEGYTraceHeaderTooSmallError(msg)
        self.header = SEGYTraceHeader(trace_header,
                                      endian=self.endian,
                                      unpack_headers=unpack_headers)
        # The number of samples in the current trace.
        npts = self.header.number_of_samples_in_this_trace
        self.npts = npts
        # Do a sanity check if there is enough data left.
        pos = self.file.tell()
        data_left = self.filesize - pos
        data_needed = DATA_SAMPLE_FORMAT_SAMPLE_SIZE[self.data_encoding] * \
            npts
        if npts < 1 or data_needed > data_left:
            msg = """
                  Too little data left in the file to unpack it according to
                  its trace header. This is most likely either due to a wrong
                  byte order or a corrupt file.
                  """.strip()
            raise SEGYTraceReadingError(msg)
        if headonly:
            # skip reading the data, but still advance the file
            self.file.seek(data_needed, 1)
            # build a function for reading data from the disk on the fly
            self.unpack_data = OnTheFlyDataUnpacker(
                DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[self.data_encoding],
                self.file.name, self.file.mode, pos, npts, endian=self.endian)
        else:
            # Unpack the data.
            self.data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[
                self.data_encoding](self.file, npts, endian=self.endian)

    def write(self, file, data_encoding=None, endian=None):
        """
        Writes the Trace to a file like object.

        If endian or data_encoding is set, these values will be enforced.
        Otherwise use the values of the SEGYTrace object.
        """
        # Set the data length in the header before writing it.
        self.header.number_of_samples_in_this_trace = len(self.data)

        # Write the header.
        self.header.write(file, endian=endian)
        if data_encoding is None:
            data_encoding = self.data_encoding
        if endian is None:
            endian = self.endian
        # Write the data.
        if self.data is None:
            msg = "No data in the SEGYTrace."
            raise SEGYWritingError(msg)
        DATA_SAMPLE_FORMAT_PACK_FUNCTIONS[data_encoding](file, self.data,
                                                         endian=endian)

    def _create_empty_trace(self):
        """
        Creates an empty trace with an empty header.
        """
        self.data = np.zeros(0, dtype=np.float32)
        self.header = SEGYTraceHeader(header=None, endian=self.endian)

    def __str__(self):
        """
        Print some information about the trace.
        """
        ret_val = 'Trace sequence number within line: %i\n' % \
            self.header.trace_sequence_number_within_line
        ret_val += '%i samples, dtype=%s, %.2f Hz' % (
            len(self.data),
            self.data.dtype, 1.0 /
            (self.header.sample_interval_in_ms_for_this_trace /
                float(1E6)))
        return ret_val

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __getattr__(self, name):
        """
        This method is only called if the attribute is not found in the usual
        places (i.e. not an instance attribute or not found in the class tree
        for self).
        """
        if name == 'data':
            # Use data unpack function to unpack data on the fly
            if hasattr(self, 'unpack_data'):
                return self.unpack_data()
            else:
                msg = """
                      Attempted to unpack trace data on the fly with
                      self.unpack_data(), but function does not exist.
                      """.strip()
                raise SEGYTraceOnTheFlyDataUnpackingError(msg)
        else:
            msg = "'%s' object has no attribute '%s'" % \
                  (self.__class__.__name__, name)
            raise AttributeError(msg)

    def to_obspy_trace(self, unpack_trace_headers=False, headonly=False):
        """
        Convert the current Trace to an ObsPy Trace object.

        :param unpack_trace_headers:
        """
        # Import here to avoid circular imports.
        from .core import LazyTraceHeaderAttribDict  # NOQA

        # Create new Trace object for every segy trace and append to the Stream
        # object.
        trace = Trace()
        # skip data if headonly is set
        if headonly:
            trace.stats.npts = self.npts
        else:
            trace.data = self.data
        trace.stats.segy = AttribDict()
        # If all values will be unpacked create a normal dictionary.
        if unpack_trace_headers:
            # Add the trace header as a new attrib dictionary.
            header = AttribDict()
            for key, value in self.header.__dict__.items():
                setattr(header, key, value)
        # Otherwise use the LazyTraceHeaderAttribDict.
        else:
            # Add the trace header as a new lazy attrib dictionary.
            header = LazyTraceHeaderAttribDict(self.header.unpacked_header,
                                               self.header.endian)
        trace.stats.segy.trace_header = header
        # The sampling rate should be set for every trace. It is a sample
        # interval in microseconds. The only sanity check is that is should be
        # larger than 0.
        tr_header = trace.stats.segy.trace_header
        if tr_header.sample_interval_in_ms_for_this_trace > 0:
            trace.stats.delta = \
                float(self.header.sample_interval_in_ms_for_this_trace) / \
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
            # work around some strange SEGY files that don't store proper
            # start date/time but only a year (see #1722)
            if julday == 0 and hour == 0 and minute == 0 and second == 0:
                msg = ('Trace starttime does not store a proper date (day '
                       'of year is zero). Using January 1st 00:00 as '
                       'trace start time.')
                warnings.warn(msg)
                julday = 1
            trace.stats.starttime = UTCDateTime(
                year=year, julday=julday, hour=hour, minute=minute,
                second=second)
        return trace


class SEGYTraceHeader(object):
    """
    Convenience class that handles reading and writing of the trace headers.
    """
    def __init__(self, header=None, endian='>', unpack_headers=False):
        """
        Will take the 240 byte of the trace header and unpack all values with
        the given endianness.

        :type header: str
        :param header: String that contains the packed binary header values.
            If header is None, a trace header with all values set to 0 will be
            created
        :type big_endian: bool
        :param big_endian: True means the header is encoded in big endian and
            False corresponds to a little endian header.
        :type unpack_headers: bool
        :param unpack_headers: Determines whether or not all headers will be
            unpacked during reading the file. Has a huge impact on the memory
            usage and the performance. They can be unpacked on-the-fly after
            being read. Defaults to False.
        """
        self.endian = endian
        if header is None:
            self._create_empty_trace_header()
            return
        # Check the length of the string,
        if len(header) != 240:
            msg = 'The trace header needs to be 240 bytes long'
            raise SEGYTraceHeaderTooSmallError(msg)
        # Either unpack the header or just append the unpacked header. This is
        # much faster and can later be unpacked on the fly.
        if not unpack_headers:
            self.unpacked_header = header
        else:
            self.unpacked_header = None
            self._read_trace_header(header)

    def _read_trace_header(self, header):
        """
        Reads the 240 byte long header and unpacks all values into
        corresponding class attributes.
        """
        # Set the start position.
        pos = 0
        # Loop over all items in the TRACE_HEADER_FORMAT list which is supposed
        # to be in the correct order.
        for item in TRACE_HEADER_FORMAT:
            length, name, special_format, _ = item
            string = header[pos: pos + length]
            pos += length
            setattr(self, name, unpack_header_value(self.endian, string,
                                                    length, special_format))

    def write(self, file, endian=None):
        """
        Writes the header to an open file like object.
        """
        if endian is None:
            endian = self.endian
        for item in TRACE_HEADER_FORMAT:
            length, name, special_format, _ = item
            # Use special format if necessary.
            if special_format:
                format = ('%s%s' % (endian,
                                    special_format)).encode('ascii',
                                                            'strict')
                file.write(pack(format, getattr(self, name)))
            # Pack according to different lengths.
            elif length == 2:
                format = ('%sh' % endian).encode('ascii', 'strict')
                file.write(pack(format, getattr(self, name)))
            # Update: Seems to be correct. Two's complement integers seem to be
            # the common way to store integer values.
            elif length == 4:
                format = ('%si' % endian).encode('ascii', 'strict')
                file.write(pack(format, getattr(self, name)))
            # Just the one unassigned field.
            elif length == 8:
                field = getattr(self, name)
                # An empty field will have a zero.
                if field == 0:
                    field = 2 * pack(('%si' % endian).encode('ascii',
                                                             'strict'), 0)
                file.write(field)
            # Should not happen.
            else:
                raise Exception

    def __getattr__(self, name):
        """
        This method is only called if the attribute is not found in the usual
        places (i.e. not an instance attribute or not found in the class tree
        for self).
        """
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
        string = self.unpacked_header[start: start + length]
        attribute = unpack_header_value(self.endian, string, length,
                                        special_format)
        setattr(self, name, attribute)
        return attribute

    def __str__(self):
        """
        Just returns all header values.
        """
        retval = ''
        for _, name, _, _ in TRACE_HEADER_FORMAT:
            # Do not print the unassigned value.
            if name == 'unassigned':
                continue
            retval += '%s: %i\n' % (name, getattr(self, name))
        return retval

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def _create_empty_trace_header(self):
        """
        Init the trace header with zeros.
        """
        # First set all fields to zero.
        for field in TRACE_HEADER_FORMAT:
            setattr(self, field[1], 0)


def _read_segy(file, endian=None, textual_header_encoding=None,
               unpack_headers=False, headonly=False):
    """
    Reads a SEG Y file and returns a SEGYFile object.

    :param file: Open file like object or a string which will be assumed to be
        a filename.
    :type endian: str
    :param endian: String that determines the endianness of the file. Either
        '>' for big endian or '<' for little endian. If it is None,
        obspy.io.segy will try to autodetect the endianness. The endianness
        is always valid for the whole file.
    :param textual_header_encoding: The encoding of the textual header.
        Either 'EBCDIC', 'ASCII' or None. If it is None, autodetection will
        be attempted.
    :type unpack_headers: bool
    :param unpack_headers: Determines whether or not all headers will be
        unpacked during reading the file. Has a huge impact on the memory usage
        and the performance. They can be unpacked on-the-fly after being read.
        Defaults to False.
    :type headonly: bool
    :param headonly: Determines whether or not the actual data records will be
        read and unpacked. Has a huge impact on memory usage. Data will not be
        unpackable on-the-fly after reading the file. Defaults to False.
    """
    # Open the file if it is not a file like object.
    if not hasattr(file, 'read') or not hasattr(file, 'tell') or not \
            hasattr(file, 'seek'):
        with open(file, 'rb') as open_file:
            return _internal_read_segy(
                open_file, endian=endian,
                textual_header_encoding=textual_header_encoding,
                unpack_headers=unpack_headers, headonly=headonly)
    # Otherwise just read it.
    return _internal_read_segy(file, endian=endian,
                               textual_header_encoding=textual_header_encoding,
                               unpack_headers=unpack_headers,
                               headonly=headonly)


def _internal_read_segy(file, endian=None, textual_header_encoding=None,
                        unpack_headers=False, headonly=False):
    """
    Reads on open file object and returns a SEGYFile object.

    :param file: Open file like object.
    :type endian: str
    :param endian: String that determines the endianness of the file. Either
        '>' for big endian or '<' for little endian. If it is None,
        obspy.io.segy will try to autodetect the endianness. The endianness
        is always valid for the whole file.
    :param textual_header_encoding: The encoding of the textual header.
        Either 'EBCDIC', 'ASCII' or None. If it is None, autodetection will
        be attempted.
    :type unpack_headers: bool
    :param unpack_headers: Determines whether or not all headers will be
        unpacked during reading the file. Has a huge impact on the memory usage
        and the performance. They can be unpacked on-the-fly after being read.
        Defaults to False.
    :type headonly: bool
    :param headonly: Determines whether or not the actual data records will be
        read and unpacked. Has a huge impact on memory usage. Data will not be
        unpackable on-the-fly after reading the file. Defaults to False.
    """
    return SEGYFile(file, endian=endian,
                    textual_header_encoding=textual_header_encoding,
                    unpack_headers=unpack_headers, headonly=headonly)


def iread_segy(file, endian=None, textual_header_encoding=None,
               unpack_headers=False, headonly=False):
    """
    Iteratively read a SEG-Y field and yield single ObsPy Traces.

    The function iteratively loops over the whole file and yields single
    ObsPy Traces. The next Trace will be read after the current loop has
    finished - this function is thus suitable for reading arbitrarily large
    SEG-Y files without running into memory problems.

    >>> from obspy.core.util import get_example_file
    >>> filename = get_example_file("00001034.sgy_first_trace")
    >>> from obspy.io.segy.segy import iread_segy
    >>> for tr in iread_segy(filename):
    ...     # Each Trace's stats attribute will have references to the file
    ...     # headers and some more information.
    ...     tf = tr.stats.segy.textual_file_header
    ...     bf = tr.stats.segy.binary_file_header
    ...     tfe = tr.stats.segy.textual_file_header_encoding
    ...     de = tr.stats.segy.data_encoding
    ...     e = tr.stats.segy.endian
    ...     # Also do something meaningful with each Trace.
    ...     print(int(tr.data.sum() * 1E9))
    -5

    :param file: Open file like object or a string which will be assumed to be
        a filename.
    :type endian: str
    :param endian: String that determines the endianness of the file. Either
        '>' for big endian or '<' for little endian. If it is None,
        obspy.io.segy will try to autodetect the endianness. The endianness
        is always valid for the whole file.
    :param textual_header_encoding: The encoding of the textual header.
        Either 'EBCDIC', 'ASCII' or None. If it is None, autodetection will
        be attempted.
    :type unpack_headers: bool
    :param unpack_headers: Determines whether or not all headers will be
        unpacked during reading the file. Has a huge impact on the memory usage
        and the performance. They can be unpacked on-the-fly after being read.
        Defaults to False.
    :type headonly: bool
    :param headonly: Determines whether or not the actual data records will be
        read and unpacked. Has a huge impact on memory usage. Data will not be
        unpackable on-the-fly after reading the file. Defaults to False.
    """
    # Open the file if it is not a file like object.
    if not hasattr(file, 'read') or not hasattr(file, 'tell') or not \
            hasattr(file, 'seek'):
        with open(file, 'rb') as open_file:
            for tr in _internal_iread_segy(
                    open_file, endian=endian,
                    textual_header_encoding=textual_header_encoding,
                    unpack_headers=unpack_headers, headonly=headonly):
                yield tr
            return
    # Otherwise just read it.
    for tr in _internal_iread_segy(
            file, endian=endian,
            textual_header_encoding=textual_header_encoding,
            unpack_headers=unpack_headers, headonly=headonly):
        yield tr


def _internal_iread_segy(file, endian=None, textual_header_encoding=None,
                         unpack_headers=False, headonly=False):
    """
    Iteratively read a SEG-Y field and yield single ObsPy Traces.
    """
    segy_file = SEGYFile(
        file, endian=endian, textual_header_encoding=textual_header_encoding,
        unpack_headers=unpack_headers, headonly=headonly, read_traces=False)
    for trace in segy_file._read_traces(unpack_headers=unpack_headers,
                                        headonly=headonly,
                                        yield_each_trace=True):
        tr = trace.to_obspy_trace(unpack_trace_headers=unpack_headers,
                                  headonly=headonly)
        # Fill stats that are normally attached to the stream stats.
        tr.stats.segy.textual_file_header = segy_file.textual_file_header
        tr.stats.segy.binary_file_header = segy_file.binary_file_header
        tr.stats.segy.textual_file_header_encoding = \
            segy_file.textual_header_encoding.upper()
        tr.stats.segy.data_encoding = trace.data_encoding
        tr.stats.segy.endian = trace.endian
        tr.stats._format = "SEGY"
        yield tr


def iread_su(file, endian=None, unpack_headers=False, headonly=False):
    """
    Iteratively read a SU field and yield single ObsPy Traces.

    The function iteratively loops over the whole file and yields single
    ObsPy Traces. The next Trace will be read after the current loop has
    finished - this function is thus suitable for reading arbitrarily large
    SU files without running into memory problems.

    >>> from obspy.core.util import get_example_file
    >>> filename = get_example_file("1.su_first_trace")
    >>> from obspy.io.segy.segy import iread_su
    >>> for tr in iread_su(filename):
    ...     # Each Trace's stats attribute will have some file-wide
    ...     # information.
    ...     de = tr.stats.su.data_encoding
    ...     e = tr.stats.su.endian
    ...     # Also do something meaningful with each Trace.
    ...     print(int(tr.data.sum()))
    -26121

    :param file: Open file like object or a string which will be assumed to be
        a filename.
    :type endian: str
    :param endian: String that determines the endianness of the file. Either
        '>' for big endian or '<' for little endian. If it is None,
        obspy.io.segy will try to autodetect the endianness. The endianness
        is always valid for the whole file.
    :type unpack_headers: bool
    :param unpack_headers: Determines whether or not all headers will be
        unpacked during reading the file. Has a huge impact on the memory usage
        and the performance. They can be unpacked on-the-fly after being read.
        Defaults to False.
    :type headonly: bool
    :param headonly: Determines whether or not the actual data records will be
        read and unpacked. Has a huge impact on memory usage. Data will not be
        unpackable on-the-fly after reading the file. Defaults to False.
    """
    # Open the file if it is not a file like object.
    if not hasattr(file, 'read') or not hasattr(file, 'tell') or not \
            hasattr(file, 'seek'):
        with open(file, 'rb') as open_file:
            for tr in _internal_iread_su(
                    open_file, endian=endian,
                    unpack_headers=unpack_headers, headonly=headonly):
                yield tr
            return
    # Otherwise just read it.
    for tr in _internal_iread_su(
            file, endian=endian,
            unpack_headers=unpack_headers, headonly=headonly):
        yield tr


def _internal_iread_su(file, endian=None, unpack_headers=False,
                       headonly=False):
    """
    Iteratively read a SU field and yield single ObsPy Traces.
    """
    su_file = SUFile(
        file, endian=endian, unpack_headers=unpack_headers, headonly=headonly,
        read_traces=False)
    for trace in su_file._read_traces(unpack_headers=unpack_headers,
                                      headonly=headonly,
                                      yield_each_trace=True):
        tr = trace.to_obspy_trace(unpack_trace_headers=unpack_headers,
                                  headonly=headonly)
        tr.stats.su = tr.stats.segy
        del tr.stats.segy
        # Fill stats that are normally attached to the stream stats.
        tr.stats.su.data_encoding = trace.data_encoding
        tr.stats.su.endian = trace.endian
        tr.stats._format = "SU"
        yield tr


class SUFile(object):
    """
    Convenience class that internally handles Seismic Unix data files. It
    currently can only read IEEE 4 byte float encoded SU data files.
    """
    def __init__(self, file=None, endian=None, unpack_headers=False,
                 headonly=False, read_traces=True):
        """
        :param file: A file like object with the file pointer set at the
            beginning of the SEG Y file. If file is None, an empty SEGYFile
            object will be initialized.

        :param endian: The endianness of the file. If None, autodetection will
            be used.
        :type unpack_header: bool
        :param unpack_header: Determines whether or not all headers will be
            unpacked during reading the file. Has a huge impact on the memory
            usage and the performance. They can be unpacked on-the-fly after
            being read. Defaults to False.
        :type headonly: bool
        :param headonly: Determines whether or not the actual data records
            will be read and unpacked. Has a huge impact on memory usage. Data
            will not be unpackable on-the-fly after reading the file.
            Defaults to False.
        :type read_traces: bool
        :param read_traces: Data traces will only be read if this is set to
            ``True``. The data will be completely ignored if this is set to
            ``False``.
        """
        if file is None:
            self._create_empty_su_file_object()
            return
            # Set the endianness to big.
            if endian is None:
                self.endian = '>'
            else:
                self.endian = ENDIAN[endian]
            return
        self.file = file
        # If endian is None autodetect is.
        if not endian:
            self._autodetect_endianness()
        else:
            self.endian = ENDIAN[endian]
        if read_traces:
            # Read the actual traces.
            [i for i in self._read_traces(unpack_headers=unpack_headers,
                                          headonly=headonly)]

    def _autodetect_endianness(self):
        """
        Tries to automatically determine the endianness of the file at hand.
        """
        self.endian = autodetect_endian_and_sanity_check_su(self.file)
        if self.endian is False:
            msg = 'Autodetection of Endianness failed. Please specify it ' + \
                  'by hand or contact the developers.'
            raise Exception(msg)

    def _create_empty_su_file_object(self):
        """
        Creates an empty SUFile object.
        """
        self.traces = []

    def __str__(self):
        """
        Prints some information about the SU file.
        """
        return '%i traces in the SU structure.' % len(self.traces)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def _read_traces(self, unpack_headers=False, headonly=False,
                     yield_each_trace=False):
        """
        Reads the actual traces starting at the current file pointer position
        to the end of the file.

        :type unpack_header: bool
        :param unpack_header: Determines whether or not all headers will be
            unpacked during reading the file. Has a huge impact on the memory
            usage and the performance. They can be unpacked on-the-fly after
            being read. Defaults to False.
        :type headonly: bool
        :param headonly: Determines whether or not the actual data records
            will be unpacked. Useful if one is just interested in the headers.
            Data will not be unpackable on-the-fly after reading the file.
            Defaults to False.
        :type yield_each_trace: bool
        :param yield_each_trace: If True, it will yield each trace after it
            has been read. This enables a simple implementation of a
            streaming interface to read SEG-Y files. Read traces will no
            longer be collected in ``self.traces`` list if this is set to
            ``True``.
        """
        self.traces = []
        # Big loop to read all data traces.
        while True:
            # Read and as soon as the trace header is too small abort.
            try:
                # Always unpack with IEEE
                trace = SEGYTrace(self.file, 5, self.endian,
                                  unpack_headers=unpack_headers,
                                  headonly=headonly)
                if yield_each_trace:
                    yield trace
                else:
                    self.traces.append(trace)
            except SEGYTraceHeaderTooSmallError:
                break

    def write(self, file, endian=None):
        """
        Write a SU Y file to file which is either a file like object with a
        write method or a filename string.

        If endian is set it will be enforced.
        """
        if not hasattr(file, 'write'):
            with open(file, 'wb') as file:
                self._write(file, endian=endian)
            return
        self._write(file, endian=endian)

    def _write(self, file, endian=None):
        """
        Write a SU Y file to file which is either a file like object with a
        write method or a filename string.

        If endian is set it will be enforced.
        """
        # Write all traces.
        for trace in self.traces:
            trace.write(file, data_encoding=5, endian=endian)


def _read_su(file, endian=None, unpack_headers=False, headonly=False):
    """
    Reads a Seismic Unix (SU) file and returns a SUFile object.

    :param file: Open file like object or a string which will be assumed to be
        a filename.
    :type endian: str
    :param endian: String that determines the endianness of the file. Either
        '>' for big endian or '<' for little endian. If it is None,
        obspy.io.segy will try to autodetect the endianness. The endianness
        is always valid for the whole file.
    :type unpack_header: bool
    :param unpack_header: Determines whether or not all headers will be
        unpacked during reading the file. Has a huge impact on the memory usage
        and the performance. They can be unpacked on-the-fly after being read.
        Defaults to False.
    :type headonly: bool
    :param headonly: Determines whether or not the actual data records will be
        unpacked. Useful if one is just interested in the headers. Defaults to
        False.
    """
    # Open the file if it is not a file like object.
    if not hasattr(file, 'read') or not hasattr(file, 'tell') or not \
            hasattr(file, 'seek'):
        with open(file, 'rb') as open_file:
            return _internal_read_su(open_file, endian=endian,
                                     unpack_headers=unpack_headers,
                                     headonly=headonly)
    # Otherwise just read it.
    return _internal_read_su(file, endian=endian,
                             unpack_headers=unpack_headers, headonly=headonly)


def _internal_read_su(file, endian=None, unpack_headers=False, headonly=False):
    """
    Reads on open file object and returns a SUFile object.

    :param file: Open file like object.
    :type endian: str
    :param endian: String that determines the endianness of the file. Either
        '>' for big endian or '<' for little endian. If it is None,
        obspy.io.segy will try to autodetect the endianness. The endianness
        is always valid for the whole file.
    :type unpack_header: bool
    :param unpack_header: Determines whether or not all headers will be
        unpacked during reading the file. Has a huge impact on the memory usage
        and the performance. They can be unpacked on-the-fly after being read.
        Defaults to False.
    :type headonly: bool
    :param headonly: Determines whether or not the actual data records will be
        unpacked. Useful if one is just interested in the headers. Defaults to
        False.
    """
    return SUFile(file, endian=endian, unpack_headers=unpack_headers,
                  headonly=headonly)


def autodetect_endian_and_sanity_check_su(file):
    """
    Takes an open file and tries to determine the endianness of a Seismic
    Unix data file by doing some sanity checks with the unpacked header values.

    Returns False if the sanity checks failed and the endianness otherwise.

    It is assumed that the data is written as 32bit IEEE floating points in
    either little or big endian.

    The test currently can only identify SU files in which all traces have the
    same length. It basically just makes a sanity check for various fields in
    the Trace header.
    """
    pos = file.tell()
    if isinstance(file, io.BytesIO):
        file.seek(0, 2)
        size = file.tell()
        file.seek(pos, 0)
    else:
        size = os.fstat(file.fileno())[6]
    if size < 244:
        return False
    # Also has to be a multiple of 4 in length because every header is 400 long
    # and every data value 4 byte long.
    elif (size % 4) != 0:
        return False
    # Jump to the number of samples field in the trace header.
    file.seek(114, 0)
    sample_count = file.read(2)
    interval = file.read(2)
    # Jump to the beginning of the year fields.
    file.seek(156, 0)
    year = file.read(2)
    jul_day = file.read(2)
    hour = file.read(2)
    minute = file.read(2)
    second = file.read(2)
    # Jump to previous position.
    file.seek(pos, 0)
    # Unpack in little and big endian.
    le_sample_count = unpack(b'<h', sample_count)[0]
    be_sample_count = unpack(b'>h', sample_count)[0]
    # Check if both work.
    working_byteorders = []
    if le_sample_count > 0:
        length = 240 + (le_sample_count * 4)
        if (size % length) == 0:
            working_byteorders.append('<')
    if be_sample_count > 0:
        length = 240 + (be_sample_count * 4)
        if (size % length) == 0:
            working_byteorders.append('>')
    # If None works return False.
    if len(working_byteorders) == 0:
        return False
    # Check if the other header values make sense.
    still_working_byteorders = []
    for bo in working_byteorders:
        fmt = ("%sh" % bo).encode('ascii', 'strict')
        this_interval = unpack(fmt, interval)[0]
        this_year = unpack(fmt, year)[0]
        this_julday = unpack(fmt, jul_day)[0]
        this_hour = unpack(fmt, hour)[0]
        this_minute = unpack(fmt, minute)[0]
        this_second = unpack(fmt, second)[0]
        # Make a sanity check for each.
        # XXX: The arbitrary maximum of the sample interval is 10 seconds.
        if this_interval <= 0 or this_interval > 10E7:
            continue
        # Some programs write two digit years.
        if this_year != 0 and (this_year < 1930 or this_year >= 2030) and \
                (this_year < 0 or this_year >= 100):
            continue
        # 9999 is often used as a placeholder
        if (this_julday > 366 or this_julday < 0) and this_julday != 9999:
            continue
        if this_hour > 24 or this_hour < 0:
            continue
        if this_minute > 60 or this_minute < 0:
            continue
        if this_second > 60 or this_second < 0:
            continue
        still_working_byteorders.append(bo)
    length = len(still_working_byteorders)
    if not length:
        return False
    elif length == 1:
        return still_working_byteorders[0]
    else:
        # XXX: In the unlikely case both byte orders pass the sanity checks
        # something else should be checked. Currently it is not.
        msg = """
            Both possible byte orders passed all sanity checks. Please contact
            the ObsPy developers so they can implement additional tests.
            """.strip()
        raise Exception(msg)
