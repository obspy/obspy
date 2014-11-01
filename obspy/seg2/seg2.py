# -*- coding: utf-8 -*-
"""
SEG-2 support for ObsPy.

A file format description is given by [Pullan1990]_.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2011
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import PY2

from copy import deepcopy
import numpy as np
from struct import unpack
import warnings

from obspy import Trace, Stream, UTCDateTime
from obspy.core import AttribDict
from obspy.seg2.header import MONTHS


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
    Class to read and write SEG 2 formatted files. The main reason this is
    realized as a class is for the ease of passing the various parameters from
    one function to the next.

    Do not change the file_pointer attribute while using this class. It will
    be used to keep track of which parts have been read yet and which not.
    """
    def __init__(self):
        pass

    def readFile(self, file_object):
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
        self.readFileDescriptorBlock()

        # Loop over every trace, read it and append it to the Stream.
        for tr_pointer in self.trace_pointers:
            self.file_pointer.seek(tr_pointer, 0)
            self.stream.append(self.parseNextTrace())

        if not hasattr(file_object, 'write'):
            self.file_pointer.close()
        return self.stream

    def readFileDescriptorBlock(self):
        """
        Handles the reading of the file descriptor block and the free form
        section following it.
        """
        file_descriptor_block = self.file_pointer.read(32)

        # Determine the endianness and check if the block id is valid.
        if unpack(b'B', file_descriptor_block[0:1])[0] == 0x55 and \
           unpack(b'B', file_descriptor_block[1:2])[0] == 0x3a:
            self.endian = b'<'
        elif unpack(b'B', file_descriptor_block[0:1])[0] == 0x3a and \
                unpack(b'B', file_descriptor_block[1:2])[0] == 0x55:
            self.endian = b'>'
        else:
            msg = 'Wrong File Descriptor Block ID'
            raise SEG2InvalidFileError(msg)

        # Check the revision number.
        revision_number = unpack(self.endian + b'H',
                                 file_descriptor_block[2:4])[0]
        if revision_number != 1:
            msg = '\nOnly SEG 2 revision 1 is officially supported. This file '
            msg += 'has revision %i. Reading it might fail.' % revision_number
            msg += '\nPlease contact the ObsPy developers with a sample file.'
            warnings.warn(msg)
        size_of_trace_pointer_sub_block = unpack(
            self.endian + b'H', file_descriptor_block[4:6])[0]
        number_of_traces = unpack(
            self.endian + b'H', file_descriptor_block[6:8])[0]

        # Define the string and line terminators.
        (size_of_string_terminator,
         first_string_terminator_char,
         second_string_terminator_char,
         size_of_line_terminator,
         first_line_terminator_char,
         second_line_terminator_char
         ) = unpack(b'BccBcc', file_descriptor_block[8:14])

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
        self.trace_pointers = []
        for _i in range(number_of_traces):
            index = _i * 4
            self.trace_pointers.append(
                unpack(self.endian + b'L',
                       trace_pointer_sub_block[index:index + 4])[0])

        # The rest of the header up to where the first trace pointer points is
        # a free form section.
        self.stream.stats = AttribDict()
        self.stream.stats.seg2 = AttribDict()
        self.parseFreeForm(self.file_pointer.read(
                           self.trace_pointers[0] - self.file_pointer.tell()),
                           self.stream.stats.seg2)

        # Get the time information from the file header.
        # XXX: Need some more generic date/time parsers.
        time = self.stream.stats.seg2.ACQUISITION_TIME
        date = self.stream.stats.seg2.ACQUISITION_DATE
        time = time.strip().split(':')
        date = date.strip().split('/')
        hour, minute, second = int(time[0]), int(time[1]), float(time[2])
        day, month, year = int(date[0]), MONTHS[date[1].lower()], int(date[2])
        self.starttime = UTCDateTime(year, month, day, hour, minute, second)

    def parseNextTrace(self):
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
        size_of_this_block = unpack(self.endian + b'H',
                                    trace_descriptor_block[2:4])[0]
        number_of_samples_in_data_block = \
            unpack(self.endian + b'L', trace_descriptor_block[8:12])[0]
        data_format_code = unpack(b'B', trace_descriptor_block[12:13])[0]

        # Parse the data format code.
        if data_format_code == 4:
            dtype = np.float32
            sample_size = 4
        elif data_format_code == 5:
            dtype = np.float64
            sample_size = 8
        elif data_format_code == 1:
            dtype = np.int16
            sample_size = 2
        elif data_format_code == 2:
            dtype = np.int32
            sample_size = 4
        elif data_format_code == 3:
            msg = ('\nData format code 3 (20-bit SEG-D floating point) not '
                   'supported yet.\nPlease contact the ObsPy developers with '
                   'a sample file.')
            raise NotImplementedError(msg)
        else:
            msg = 'Unrecognized data format code'
            raise SEG2InvalidFileError(msg)

        # The rest of the trace block is free form.
        header = {}
        header['seg2'] = AttribDict()
        self.parseFreeForm(self.file_pointer.read(size_of_this_block - 32),
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
        header['calib'] = float(header['seg2']['DESCALING_FACTOR'])
        # Unpack the data.
        data = np.fromstring(
            self.file_pointer.read(number_of_samples_in_data_block *
                                   sample_size),
            dtype=dtype)
        # Integrate SEG2 file header into each trace header
        tmp = self.stream.stats.seg2.copy()
        tmp.update(header['seg2'])
        header['seg2'] = tmp
        return Trace(data=data, header=header)

    def parseFreeForm(self, free_form_str, attrib_dict):
        """
        Parse the free form section stored in free_form_str and save it in
        attrib_dict.
        """
        # Separate the strings.
        strings = free_form_str.split(self.string_terminator)
        # This is not fully according to the SEG-2 format specification (or
        # rather the specification only speaks about on offset of 2 bytes
        # between strings and a string_terminator between two free form
        # strings. The file I have show the following separation between two
        # strings: 'random offset byte', 'string_terminator',
        # 'random offset byte'
        # Therefore every string has to be at least 3 bytes wide to be
        # acceptable after being split at the string terminator.

        def is_good_char(c):
            return c in (b'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN'
                         b'OPQRSTUVWXYZ!"#$%&\'()*+,-./:; <=>?@[\\]^_`{|}~ ')

        # A loop over a bytestring in Python 3 returns integers. This can be
        # solved with a number of imports from the python-future module and
        # all kinds of subtle changes throughout this file. Separating the
        # handling for Python 2 and 3 seems the cleaner and simpler approach.
        if PY2:
            strings = ["".join(filter(is_good_char, _i))
                       for _i in strings
                       if len(_i) >= 3]
        else:
            strings = ["".join(map(chr, filter(is_good_char, _i)))
                       for _i in strings
                       if len(_i) >= 3]

        # Every string has the structure OPTION<SPACE>VALUE. Write to
        # stream.stats attribute.
        for string in strings:
            string = string.strip()
            string = string.split(' ')
            key = string[0].strip()
            value = ' '.join(string[1:]).strip()
            setattr(attrib_dict, key, value)
        # Parse the notes string again.
        if hasattr(attrib_dict, 'NOTE'):
            notes = attrib_dict.NOTE.split(self.line_terminator.decode())
            attrib_dict.NOTE = AttribDict()
            for note in notes:
                note = note.strip()
                note = note.split(' ')
                key = note[0].strip()
                value = ' '.join(note[1:]).strip()
                setattr(attrib_dict.NOTE, key, value)


def isSEG2(filename):
    if not hasattr(filename, 'write'):
        file_pointer = open(filename, 'rb')
    else:
        file_pointer = filename

    file_descriptor_block = file_pointer.read(4)
    if not hasattr(filename, 'write'):
        file_pointer.close()
    try:
        # Determine the endianness and check if the block id is valid.
        if unpack(b'B', file_descriptor_block[0:1])[0] == 0x55 and \
           unpack(b'B', file_descriptor_block[1:2])[0] == 0x3a:
            endian = b'<'
        elif unpack(b'B', file_descriptor_block[0:1])[0] == 0x3a and \
                unpack(b'B', file_descriptor_block[1:2])[0] == 0x55:
            endian = b'>'
        else:
            return False
    except:
        return False
    # Check the revision number.
    revision_number = unpack(endian + b'H',
                             file_descriptor_block[2:4])[0]
    if revision_number != 1:
        return False
    return True


def readSEG2(filename, **kwargs):  # @UnusedVariable
    seg2 = SEG2()
    st = seg2.readFile(filename)
    warnings.warn(WARNING_HEADER)
    return st
