# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
#  Filename: seg.py
#  Purpose: Routines for reading and writing SEG Y files.
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#
# Copyright (C) 2010 Lion Krischer
#---------------------------------------------------------------------
"""
Routines to read and write SEG Y rev 1 encoded seismic data files.
"""

from __future__ import with_statement
from obspy.segy.header import ENDIAN, DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS, \
    BINARY_FILE_HEADER_FORMAT, DATA_SAMPLE_FORMAT_PACK_FUNCTIONS, \
    TRACE_HEADER_FORMAT
from struct import pack, unpack
import numpy as np


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


class SEGYWritingError(SEGYError):
    """
    Raised if the trace header is not the required 240 byte long.
    """
    pass


class SEGYFile(object):
    """
    Convenience class that internally handles SEG Y files.
    """
    def __init__(self, file=None, endian=None, textual_header_encoding=None):
        """
        :param file: A file like object with the file pointer set at the
            beginning of the SEG Y file. If file is None, an empty SEGYFile
            object will be initialized.

        :param textual_header_encoding: The encoding of the textual header.
            Either 'EBCDIC', 'ASCII' or None. If it is None, autodetection will
            be attempted. If it is None and file is also None, it will default
            to 'ASCII'.
        """
        if file is None:
            self._createEmptySEGYFileObject()
            # Set the endianness to big.
            if endian is None:
                self.endian = ENDIAN['big']
            else:
                self.endian = endian
            # And the textual header encoding to ASCII.
            if textual_header_encoding is None:
                self.textual_header_encoding = 'ASCII'
            self.textual_header = ''
            return
        # Check if file has all the necessary attributes and raise an exception
        # otherwise.
        if not hasattr(file, 'read') or not hasattr(file, 'seek') or not \
            hasattr(file, 'tell'):
            msg = 'file is not a file like object.'
            raise TypeError(msg)
        self.file = file
        # If endian is None autodetect is.
        if not endian:
            self._autodetectEndianness()
        else:
            self.endian = ENDIAN[endian]
        # If the textual header encoding is None, autodetection will be used.
        self.textual_header_encoding = textual_header_encoding
        # Read the headers.
        self._readHeaders()
        # Read the actual traces.
        self._readTraces()

    def __str__(self):
        """
        Prints some information about the SEG Y file.
        """
        return '%i traces in the SEG Y structure.' % len(self.traces)

    def _autodetectEndianness(self):
        """
        Tries to automatically determine the endianness of the file at hand.
        """
        pos = self.file.tell()
        # Jump to the data sample format code.
        self.file.seek(3224, 1)
        format = unpack('>h', self.file.read(2))[0]
        # Check if valid.
        if format in DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS.keys():
            self.endian = '>'
        # Else test little endian.
        else:
            self.file.seek(-2, 1)
            format = unpack('<h', self.file.read(2))[0]
            if format in DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS.keys():
                self.endian = '<'
            else:
                msg = 'Unable to determine the endianness of the file. ' + \
                      'Please specify it.'
                raise SEGYError(msg)
        # Jump to previous position.
        self.file.seek(pos, 0)

    def _createEmptySEGYFileObject(self):
        """
        Creates an empty SEGYFile object.
        """
        self.textual_file_header = ''
        self.binary_file_header = None
        self.traces = []

    def _readTextualHeader(self):
        """
        Reads the textual header.
        """
        # The first 3200 byte are the textual header.
        textual_header = self.file.read(3200)
        # The data can either be saved as plain ASCII or EBCDIC. The first
        # character always is mostly 'C' and therefore used to check the
        # encoding. Sometimes is it not C but also cannot be decoded from
        # EBCDIC so it is treated as ASCII and all empty symbols are removed.
        if not self.textual_header_encoding:
            if textual_header[0] != 'C':
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

    def _readHeaders(self):
        """
        Reads the textual and binary file headers starting at the current file
        pointer position.
        """
        # Read the textual header.
        self._readTextualHeader()
        # The next 400 bytes are from the Binary File Header.
        binary_file_header = self.file.read(400)
        bfh = SEGYBinaryFileHeader(binary_file_header, self.endian)
        self.binary_file_header = bfh
        self.data_encoding = self.binary_file_header.data_sample_format_code
        # If bytes 3506-3506 are not zero, an extended textual header follows
        # which is not supported so far.
        if bfh.number_of_3200_byte_ext_file_header_records_following != 0:
            msg = 'Extended textual headers are supported yet. ' + \
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
        self._writeTextualHeader(file)
        # Write the binary header.
        self.binary_file_header.write(file, endian=endian)
        # Write all traces.
        for trace in self.traces:
            trace.write(file, data_encoding=data_encoding, endian=endian)

    def _writeTextualHeader(self, file):
        """
        Write the textual header in various encodings. The encoding will depend
        on self.textual_header_encoding. If self.textual_file_header is too
        small it will be padded with zeros. If it is too long or an invalid
        encoding is specified an exception will be raised.
        """
        length = len(self.textual_file_header)
        # Append spaces to the end if its too short.
        if length < 3200:
            textual_header = self.textual_file_header + ' ' * (3200 - length)
        elif length == 3200:
            textual_header = self.textual_file_header
        # The length must not exceed 3200 byte.
        else:
            msg = 'self.textual_file_header is not allowed to be longer ' + \
                  'than 3200 bytes'
            raise SEGYWritingError(msg)
        if self.textual_header_encoding.upper() == 'ASCII':
            pass
        elif self.textual_header_encoding.upper() == 'EBCDIC':
            textual_header = \
                textual_header.decode('ascii').encode('EBCDIC-CP-BE')
        # Should not happen.
        else:
            msg = 'self.textual_header_encoding has to be either ASCII or ' + \
                  'EBCDIC.'
            raise SEGYWritingError(msg)
        file.write(textual_header)

    def _readTraces(self):
        """
        Reads the actual traces starting at the current file pointer position
        to the end of the file.
        """
        self.traces = []
        # Big loop to read all data traces.
        while True:
            # Read and as soon as the trace header is too small abort.
            try:
                trace = SEGYTrace(self.file, self.data_encoding, self.endian)
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
            self._createEmptyBinaryFileHeader()
            return
        self._readBinaryFileHeader(header)

    def _readBinaryFileHeader(self, header):
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
                format = '%sh' % self.endian
                # Set the class attribute.
                setattr(self, name, unpack(format, string)[0])
            # Update: Seems to be correct. Two's complement integers seem to be
            # the common way to store integer values.
            elif length == 4:
                format = '%sI' % self.endian
                # Set the class attribute.
                setattr(self, name, unpack(format, string)[0])
            # The other value are the unassigned values. As it is unclear how
            # these are formated they will be stored as strings.
            elif name.startswith('unassigned'):
                # These are only the unassigned fields.
                format = 'h' * (length / 2)
                # Set the class attribute.
                setattr(self, name, string)
            # Should not happen.
            else:
                raise Exception

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
                format = '%sh' % endian
                # Write to file.
                file.write(pack(format, getattr(self, name)))
            # Update: Seems to be correct. Two's complement integers seem to be
            # the common way to store integer values.
            elif length == 4:
                format = '%sI' % endian
                # Write to file.
                file.write(pack(format, getattr(self, name)))
            # These are the two unassigned values in the binary file header.
            elif name.startswith('unassigned'):
                file.write(getattr(self, name))
            # Should not happen.
            else:
                raise Exception

    def _createEmptyBinaryFileHeader(self):
        """
        Just fills all necessary class attributes with zero.
        """
        for item in BINARY_FILE_HEADER_FORMAT:
            _, name, _ = item
            setattr(self, name, 0)


class SEGYTrace(object):
    """
    Convenience class that internally handles a single SEG Y trace.
    """
    def __init__(self, file=None, data_encoding=4, endian='>'):
        """
        :param file: Open file like object with the file pointer of the
            beginning of a trace. If it is None, an empty trace will be
            created.
        :param data_encoding: The data sample format code as defined in the
            binary file header:
                1: 4 byte IBM floating point
                2: 4 byte Integer, two's complement
                3: 2 byte Integer, two's complement
                4: 4 byte Fixed point with gain
                5: 4 byte IEEE floating point
                8: 1 byte Integer, two's complement
            Defaults to 4.
        :param big_endian: Bool. True means the header is encoded in big endian
            and False corresponds to a little endian header.
        """
        self.endian = endian
        self.data_encoding = data_encoding
        # If None just return empty structure.
        if file is None:
            self._createEmptyTrace()
            return
        self.file = file
        # Otherwise read the file.
        self._readTrace()

    def _readTrace(self):
        """
        Reads the complete next header starting at the file pointer at
        self.file.
        """
        trace_header = self.file.read(240)
        # Check if it is smaller than 240 byte.
        if len(trace_header) != 240:
            msg = 'The trace header needs to be 240 bytes long'
            raise SEGYTraceHeaderTooSmallError(msg)
        self.header = SEGYTraceHeader(trace_header,
                                      endian=self.endian)
        # The number of samples in the current trace.
        npts = self.header.number_of_samples_in_this_trace
        # Unpack the data.
        self.data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[\
                        self.data_encoding](self.file, npts,
                        endian=self.endian)

    def write(self, file, data_encoding=None, endian=None):
        """
        Writes the Trace to a file like object.

        If endian or data_encoding is set, these values will be enforced.
        Otherwise use the values of the SEGYTrace object.
        """
        # Write the header.
        self.header.write(file, endian=endian)
        if data_encoding is None:
            data_encoding = self.data_encoding
        if endian is None:
            endian = self.endian
        # Write the data.
        DATA_SAMPLE_FORMAT_PACK_FUNCTIONS[data_encoding](file, self.data,
                                                  endian=endian)

    def _createEmptyTrace(self):
        """
        Creates an empty trace with an empty header.
        """
        self.data = np.zeros(0, dtype='float32')
        self.header = SEGYTraceHeader(header=None, endian=self.endian)

    def __str__(self):
        """
        Print some information about the trace.
        """
        ret_val = 'Trace sequence number within line: %i\n' % \
                self.header.trace_sequence_number_within_line
        ret_val += '%i samples, dtype=%s, %.2f Hz' % (len(self.data),
                self.data.dtype, 1.0 / \
                (self.header.sample_interval_in_ms_for_this_trace / \
                float(1E6)))
        return ret_val


class SEGYTraceHeader(object):
    """
    Convenience class that handles reading and writing of the trace headers.
    """
    def __init__(self, header=None, endian='>'):
        """
        Will take the 240 byte of the trace header and unpack all values with
        the given endianness.

        :param header: String that contains the packed binary header values.
            If header is None, a trace header with all values set to 0 will be
            created
        :param big_endian: Bool. True means the header is encoded in big endian
            and False corresponds to a little endian header.
        """
        self.endian = endian
        if header is None:
            self._createEmptyTraceHeader()
            return
        # Check the length of the string,
        if len(header) != 240:
            msg = 'The trace header needs to be 240 bytes long'
            raise SEGYTraceHeaderTooSmallError(msg)
        # Otherwise read the given file.
        self._readTraceHeader(header)

    def _readTraceHeader(self, header):
        """
        Reads the 240 byte long header and unpacks all values into
        corresponding class attributes.
        """
        # Set the start position.
        pos = 0
        # Loop over all items in the TRACE_HEADER_FORMAT list which is supposed
        # to be in the correct order.
        for item in TRACE_HEADER_FORMAT:
            length, name = item
            string = header[pos: pos + length]
            pos += length
            # Unpack according to different lengths.
            if length == 2:
                format = '%sh' % self.endian
                setattr(self, name, unpack(format, string)[0])
            # Update: Seems to be correct. Two's complement integers seem to be
            # the common way to store integer values.
            elif length == 4:
                format = '%sI' % self.endian
                setattr(self, name, unpack(format, string)[0])
            # The unassigned field. Since it is unclear how this field is
            # encoded it will just be stored as a string.
            elif length == 8:
                format = '%shhhh' % self.endian
                setattr(self, name, string)
            # Should not happen
            else:
                raise Exception

    def write(self, file, endian=None):
        """
        Writes the header to an open file like object.
        """
        if endian is None:
            endian = self.endian
        for item in TRACE_HEADER_FORMAT:
            length, name = item
            # Pack according to different lengths.
            if length == 2:
                format = '%sh' % endian
                file.write(pack(format, getattr(self, name)))
            # Update: Seems to be correct. Two's complement integers seem to be
            # the common way to store integer values.
            elif length == 4:
                format = '%sI' % endian
                file.write(pack(format, getattr(self, name)))
            # Just the one unassigned field.
            elif length == 8:
                file.write(getattr(self, name))
            # Should not happen.
            else:
                raise Exception

    def __str__(self):
        """
        Just returns all header values.
        """
        retval = ''
        for item in TRACE_HEADER_FORMAT:
            _, name = item
            # Do not print the unassigned value.
            if name == 'unassigned':
                continue
            retval += '%s: %i\n' % (name, getattr(self, name))
        return retval

    def _createEmptyTraceHeader(self):
        """
        Sets all trace header values to 0.
        """
        for field in TRACE_HEADER_FORMAT:
            setattr(self, field[1], 0)


def readSEGY(file, endian=None, textual_header_encoding=None):
    """
    :param file: Open file like object or a string which will be assumed to be
        a filename.
    :param endian: String that determines the endianness of the file. Either
        '>' for big endian or '<' for little endian. If it is None, obspy.segy
        will try to autodetect the endianness. The endianness is always valid
        for the whole file.
    :param textual_header_encoding: The encoding of the textual header.
        Either 'EBCDIC', 'ASCII' or None. If it is None, autodetection will
        be attempted.
    """
    # Open the file if it is not a file like object.
    if not hasattr(file, 'read') or not hasattr(file, 'tell') or not \
        hasattr(file, 'seek'):
        with open(file, 'rb') as open_file:
            return _readSEGY(open_file, endian=endian,
                             textual_header_encoding=textual_header_encoding)
    # Otherwise just read it.
    return _readSEGY(file, endian=endian,
                     textual_header_encoding=textual_header_encoding)


def _readSEGY(file, endian=None, textual_header_encoding=None):
    """
    Reads on open file object and returns a SEGYFile object.

    :param file: Open file like object.
    :param endian: String that determines the endianness of the file. Either
        '>' for big endian or '<' for little endian. If it is None, obspy.segy
        will try to autodetect the endianness. The endianness is always valid
        for the whole file.
    :param textual_header_encoding: The encoding of the textual header.
        Either 'EBCDIC', 'ASCII' or None. If it is None, autodetection will
        be attempted.
    """
    return SEGYFile(file, endian=endian,
                    textual_header_encoding=textual_header_encoding)
