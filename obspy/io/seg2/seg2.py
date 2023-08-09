# -*- coding: utf-8 -*-
"""
SEG-2 support for ObsPy.

A file format description is given by [Pullan1990]_.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2011
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from copy import deepcopy
from struct import unpack, unpack_from
import warnings
import re

import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core import AttribDict
from obspy.core.compatibility import from_buffer
from .header import MONTHS


WARNING_HEADER = "Many companies use custom defined SEG2 header variables." + \
    " This might cause basic header information reflected in the single " + \
    "traces' stats to be wrong (e.g. recording delays, first sample " + \
    "number, station code names, ..). Please check the complete list of " + \
    "additional unmapped header fields that gets stored in " + \
    "Trace.stats.seg2 and/or the manual of the source of the SEG2 files " + \
    "for fields that might influence e.g. trace start times."


class SEG2BaseError(Exception):
    """
    Base class for all SEG-2 specific errors.
    """
    pass


class SEG2InvalidFileError(SEG2BaseError):
    """
    Will be raised if something is not correct with the SEG-2 file.
    """
    pass


class SEG2(object):
    """
    Class to read SEG 2 formatted files.

    The main reason this is realized as a class is for the ease of passing
    the various parameters from one function to the next.

    Do not change the file_pointer attribute while using this class. It will
    be used to keep track of which parts have been read yet and which not.
    """
    def __init__(self):
        pass

    def read_file(self, file_object):
        """
        Reads the following file and will return a Stream object. If
        file_object is a string it will be treated as a file name, otherwise it
        will be expected to be a file like object with read(), seek() and
        tell() methods.

        If it is a file_like object, file.seek(0, 0) is expected to be the
        beginning of the SEG-2 file.
        """
        # Read the file if it is a file name.
        if not hasattr(file_object, 'write'):
            self.file_pointer = open(file_object, 'rb')
        else:
            self.file_pointer = file_object
            self.file_pointer.seek(0, 0)

        self.stream = Stream()

        # Read the file descriptor block. This will also determine the
        # endianness.
        self.read_file_descriptor_block()

        # Loop over every trace, read it and append it to the Stream.
        for tr_pointer in self.trace_pointers:
            self.file_pointer.seek(tr_pointer, 0)
            self.stream.append(self.parse_next_trace())

        if not hasattr(file_object, 'write'):
            self.file_pointer.close()
        return self.stream

    def read_file_descriptor_block(self):
        """
        Handles the reading of the file descriptor block and the free form
        section following it.
        """
        file_descriptor_block = self.file_pointer.read(32)

        # Determine the endianness and check if the block id is valid.
        if unpack_from(b'2B', file_descriptor_block) == (0x55, 0x3a):
            self.endian = b'<'
        elif unpack_from(b'2B', file_descriptor_block) == (0x3a, 0x55):
            self.endian = b'>'
        else:
            msg = 'Wrong File Descriptor Block ID'
            raise SEG2InvalidFileError(msg)

        # Check the revision number.
        revision_number, = unpack_from(self.endian + b'H',
                                       file_descriptor_block, 2)
        if revision_number != 1:
            msg = '\nOnly SEG 2 revision 1 is officially supported. This file '
            msg += 'has revision %i. Reading it might fail.' % revision_number
            msg += '\nPlease contact the ObsPy developers with a sample file.'
            warnings.warn(msg)

        # Determine trace counts.
        (size_of_trace_pointer_sub_block,
         number_of_traces
         ) = unpack_from(self.endian + b'HH', file_descriptor_block, 4)
        if number_of_traces * 4 > size_of_trace_pointer_sub_block:
            msg = ('File indicates %d traces, but there are only %d trace '
                   'pointers.') % (number_of_traces,
                                   size_of_trace_pointer_sub_block // 4)
            raise SEG2InvalidFileError(msg)

        # Define the string and line terminators.
        (size_of_string_terminator,
         first_string_terminator_char,
         second_string_terminator_char,
         size_of_line_terminator,
         first_line_terminator_char,
         second_line_terminator_char
         ) = unpack_from(b'BccBcc', file_descriptor_block, 8)

        # Assemble the string terminator.
        if size_of_string_terminator == 1:
            self.string_terminator = first_string_terminator_char
        elif size_of_string_terminator == 2:
            self.string_terminator = first_string_terminator_char + \
                second_string_terminator_char
        else:
            msg = 'Wrong size of string terminator.'
            raise SEG2InvalidFileError(msg)
        # Assemble the line terminator.
        if size_of_line_terminator == 1:
            self.line_terminator = first_line_terminator_char
        elif size_of_line_terminator == 2:
            self.line_terminator = first_line_terminator_char + \
                second_line_terminator_char
        else:
            msg = 'Wrong size of line terminator.'
            raise SEG2InvalidFileError(msg)

        # Read the trace pointer sub-block and retrieve all the pointers.
        trace_pointer_sub_block = \
            self.file_pointer.read(size_of_trace_pointer_sub_block)
        self.trace_pointers = unpack_from(
            self.endian + (b'L' * number_of_traces), trace_pointer_sub_block)

        # The rest of the header up to where the first trace pointer points is
        # a free form section.
        self.stream.stats = AttribDict()
        self.stream.stats.seg2 = AttribDict()
        self.parse_free_form(
            self.file_pointer.read(self.trace_pointers[0] -
                                   self.file_pointer.tell()),
            self.stream.stats.seg2)

        # Get the time information from the file header.
        # XXX: Need some more generic date/time parsers.
        if "ACQUISITION_TIME" in self.stream.stats.seg2 \
                and "ACQUISITION_DATE" in self.stream.stats.seg2:
            time = self.stream.stats.seg2.ACQUISITION_TIME
            date = self.stream.stats.seg2.ACQUISITION_DATE
            # Split on any non numeric character
            time = list(filter(None, re.split(r'\D+', time)))
            # Split on space, dot (.), slash (/), and dash (-)
            date = list(filter(None, re.split("[, ./-]+", date)))
            hour, minute, second = int(time[0]), int(time[1]), float(time[2])
            day, month, year = int(date[0]), MONTHS[date[1].lower()], \
                int(date[2])
            self.starttime = UTCDateTime(year, month, day, hour, minute,
                                         second)
        else:
            self.starttime = UTCDateTime(0)

    def parse_next_trace(self):
        """
        Parse the next trace in the trace pointer list and return a Trace
        object.
        """
        trace_descriptor_block = self.file_pointer.read(32)
        # Check if the trace descriptor block id is valid.
        if unpack(self.endian + b'H', trace_descriptor_block[0:2])[0] != \
           0x4422:
            msg = 'Invalid trace descriptor block id.'
            raise SEG2InvalidFileError(msg)
        size_of_this_block, = unpack_from(self.endian + b'H',
                                          trace_descriptor_block, 2)
        number_of_samples_in_data_block, = \
            unpack_from(self.endian + b'L', trace_descriptor_block, 8)
        data_format_code, = unpack_from(b'B', trace_descriptor_block, 12)

        # Parse the data format code.
        if data_format_code == 4:
            dtype = self.endian + b'f4'
            sample_size = 4
        elif data_format_code == 5:
            dtype = self.endian + b'f8'
            sample_size = 8
        elif data_format_code == 1:
            dtype = self.endian + b'i2'
            sample_size = 2
        elif data_format_code == 2:
            dtype = self.endian + b'i4'
            sample_size = 4
        elif data_format_code == 3:
            dtype = self.endian + b'i2'
            sample_size = 2.5
            if number_of_samples_in_data_block % 4 != 0:
                raise SEG2InvalidFileError(
                    'Data format code 3 requires that the number of samples '
                    'is divisible by 4, but sample count is %d' % (
                        number_of_samples_in_data_block, ))
        else:
            msg = 'Unrecognized data format code'
            raise SEG2InvalidFileError(msg)

        # The rest of the trace block is free form.
        header = {}
        header['seg2'] = AttribDict()
        self.parse_free_form(self.file_pointer.read(size_of_this_block - 32),
                             header['seg2'])
        header['delta'] = float(header['seg2']['SAMPLE_INTERVAL'])
        # Set to the file's start time.
        header['starttime'] = deepcopy(self.starttime)
        if 'DELAY' in header['seg2']:
            if float(header['seg2']['DELAY']) != 0:
                msg = "Non-zero value found in Trace's 'DELAY' field. " + \
                      "This is not supported/tested yet and might lead " + \
                      "to a wrong starttime of the Trace. Please contact " + \
                      "the ObsPy developers with a sample file."
                warnings.warn(msg)

        if "DESCALING_FACTOR" in header["seg2"]:
            header['calib'] = float(header['seg2']['DESCALING_FACTOR'])

        # Unpack the data.
        data = from_buffer(
            self.file_pointer.read(
                int(number_of_samples_in_data_block * sample_size)),
            dtype=dtype)
        if data_format_code == 3:
            # Convert one's complement to two's complement by adding one to
            # negative numbers.
            one_to_two = (data < 0)
            # The first two bytes (1 word) of every 10 bytes (5 words) contains
            # a 4-bit exponent for each of the 4 remaining 2-byte (int16)
            # samples.
            exponents = data[0::5].view(self.endian + b'u2')
            result = np.empty(number_of_samples_in_data_block, dtype=np.int32)
            # Apply the negative correction, then multiply by correct exponent.
            result[0::4] = ((data[1::5] + one_to_two[1::5]) *
                            2**((exponents & 0x000f) >> 0))
            result[1::4] = ((data[2::5] + one_to_two[2::5]) *
                            2**((exponents & 0x00f0) >> 4))
            result[2::4] = ((data[3::5] + one_to_two[3::5]) *
                            2**((exponents & 0x0f00) >> 8))
            result[3::4] = ((data[4::5] + one_to_two[4::5]) *
                            2**((exponents & 0xf000) >> 12))
            data = result

        # Integrate SEG2 file header into each trace header
        tmp = self.stream.stats.seg2.copy()
        tmp.update(header['seg2'])
        header['seg2'] = tmp
        return Trace(data=data, header=header)

    def parse_free_form(self, free_form_str, attrib_dict):
        """
        Parse the free form section stored in free_form_str and save it in
        attrib_dict.
        """
        def cleanup_and_decode_string(value):
            # Some software/hardware produces invalid characters.
            def is_good_char(c):
                return c in (b'0123456789'
                             b'abcdefghijklmnopqrstuvwxyz'
                             b'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                             b'!"#$%&\'()*+,-./:; <=>?@[\\]^_`{|}~ ')

            # A loop over a bytestring in Python 3 returns integers. This can
            # be solved with a number of imports from the python-future module
            # and all kinds of subtle changes throughout this file.
            return "".join(map(chr, filter(is_good_char, value))).strip()

        # Separate the strings. Every string starts with a 2-byte offset to the
        # next string, and ends with a terminator. An offset of 0 indicates the
        # end of the strings.
        offset = 0
        strings = []
        while offset + 2 < len(free_form_str):
            strlen, = unpack_from(self.endian + b'H', free_form_str, offset)
            if strlen == 0:
                break
            curstr = free_form_str[offset + 2:offset + strlen]
            try:
                curstrlen = curstr.index(self.string_terminator)
            except ValueError:
                strings.append(curstr)
            else:
                strings.append(curstr[:curstrlen])
            offset += strlen

        # Every string has the structure OPTION<SPACE>VALUE. Write to
        # stream.stats attribute.
        for string in strings:
            string = string.strip().split(b' ', 1)
            key = cleanup_and_decode_string(string[0])
            try:
                value = string[1]
            except IndexError:
                value = b''
            if key == 'NOTE':
                value = [cleanup_and_decode_string(line)
                         for line in value.split(self.line_terminator)
                         if line]
            else:
                value = cleanup_and_decode_string(value)
            setattr(attrib_dict, key, value)


def _is_seg2(filename):
    if not hasattr(filename, 'write'):
        file_pointer = open(filename, 'rb')
    else:
        file_pointer = filename

    file_descriptor_block = file_pointer.read(4)
    if not hasattr(filename, 'write'):
        file_pointer.close()
    try:
        # Determine the endianness and check if the block id is valid.
        if unpack_from(b'2B', file_descriptor_block) == (0x55, 0x3a):
            endian = b'<'
        elif unpack_from(b'2B', file_descriptor_block) == (0x3a, 0x55):
            endian = b'>'
        else:
            return False
    except Exception:
        return False
    # Check the revision number.
    revision_number, = unpack_from(endian + b'H', file_descriptor_block, 2)
    if revision_number != 1:
        return False
    return True


def _read_seg2(filename, **kwargs):  # @UnusedVariable
    seg2 = SEG2()
    st = seg2.read_file(filename)
    warnings.warn(WARNING_HEADER)
    return st
