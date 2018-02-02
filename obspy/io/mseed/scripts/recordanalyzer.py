#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
#  Filename: recordanalyzer.py
#  Purpose: A command-line tool to analyze Mini-SEED records for development
#           purposes.
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#
# Copyright (C) 2010-2012 Lion Krischer
# --------------------------------------------------------------------
"""
A command-line tool to analyze Mini-SEED records.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import sys
from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from struct import unpack

from obspy import UTCDateTime, __version__


class RecordAnalyser(object):
    """
    Analyses a Mini-SEED file on a per record basis.

    Basic usage:
        >> rec = RecordAnalyser(filename)
        # Pretty print the information contained in the first record.
        >> print(rec)
        # Jump to the next record.
        >> rex.next()
        >> print(rec)
    """
    def __init__(self, file_object):
        """
        file_object can either be a filename or any file like object that has
        read, seek and tell methods.

        Will automatically read the first record.
        """
        # Read object or file.
        if not hasattr(file_object, 'read'):
            self.filename = file_object
            self.file = open(file_object, 'rb')
        else:
            self.filename = None
            self.file = file_object
        # Set the offset to the record.
        self.record_offset = 0
        self.record_number = 0
        # Parse the header.
        self._parse_header()
        self.did_goto = False

    def __eq__(self, other):
        """
        Compares two records.
        """
        if self.fixed_header != other.fixed_header:
            return False
        if self.blockettes != other.blockettes:
            return False
        return True

    def __ne__(self, other):
        """
        Always needed of __eq__ is defined.
        """
        if self.__eq__(other):
            return False
        return True

    def __next__(self):
        """
        Jumps to the next record and parses the header.
        """
        self.record_offset += 2 ** self.blockettes[1000]['Data Record Length']
        self.record_number += 1
        try:
            self._parse_header()
        except IOError as e:
            msg = "IOError while trying to read record number %i: %s"
            raise StopIteration(msg % (self.record_number, str(e)))

    def goto(self, record_number):
        """
        Jumps to the specified record and parses its header.

        :type record_number: int
        :param record_number: Record number to jump to (first record has record
            number 0).
        """
        self.record_number = record_number
        self.record_offset = (
            record_number * 2 ** self.blockettes[1000]['Data Record Length'])
        try:
            self._parse_header()
        except IOError as e:
            msg = "IOError while trying to read record number %i: %s"
            raise StopIteration(msg % (self.record_number, str(e)))
        self.did_goto = True

    def _parse_header(self):
        """
        Makes all necessary calls to parse the header.
        """
        # Big or little endian for the header.
        self._get_endianess()
        # Read the fixed header.
        self._read_fixed_header()
        # Get the present blockettes.
        self._get_blockettes()
        # Calculate the starttime.
        self._calculate_start_time()

    def _get_endianess(self):
        """
        Tries to figure out whether or not the file has little or big endian
        encoding and sets self.endian to either '<' for little endian or '>'
        for big endian. It works by unpacking the year with big endian and
        checking whether it is between 1900 and 2050. Does not change the
        pointer.
        """
        # Save the file pointer.
        current_pointer = self.file.tell()
        # Seek the year.
        self.file.seek(self.record_offset + 20, 0)
        # Get the year
        year_raw = self.file.read(2)
        try:
            year = unpack(native_str('>H'), year_raw)[0]
        except Exception:
            if len(year_raw) == 0:
                msg = "Unexpected end of file."
                raise IOError(msg)
            raise
        if year >= 1900 and year <= 2050:
            self.endian = '>'
        else:
            self.endian = '<'
        # Reset the pointer.
        self.file.seek(current_pointer, 0)

    def _read_fixed_header(self):
        """
        Reads the fixed header of the Mini-SEED file and writes all entries to
        self.fixed_header, a dictionary.
        """
        # Init empty fixed header dictionary. Use an ordered dictionary to
        # achieve the same order as in the Mini-SEED manual.
        self.fixed_header = OrderedDict()
        # Read and unpack.
        self.file.seek(self.record_offset, 0)
        fixed_header = self.file.read(48)
        encoding = native_str('%s20c2H3Bx4H4Bl2H' % self.endian)
        try:
            header_item = unpack(encoding, fixed_header)
        except Exception:
            if len(fixed_header) == 0:
                msg = "Unexpected end of file."
                raise IOError(msg)
            raise
        # Write values to dictionary.
        self.fixed_header['Sequence number'] = \
            int(''.join(x.decode('ascii') for x in header_item[:6]))
        self.fixed_header['Data header/quality indicator'] = \
            header_item[6].decode('ascii')
        self.fixed_header['Station identifier code'] = \
            ''.join(x.decode('ascii') for x in header_item[8:13]).strip()
        self.fixed_header['Location identifier'] = \
            ''.join(x.decode('ascii') for x in header_item[13:15]).strip()
        self.fixed_header['Channel identifier'] = \
            ''.join(x.decode('ascii') for x in header_item[15:18]).strip()
        self.fixed_header['Network code'] = \
            ''.join(x.decode('ascii') for x in header_item[18:20]).strip()
        # Construct the starttime. This is only the starttime in the fixed
        # header without any offset. See page 31 of the SEED manual for the
        # time definition.
        self.fixed_header['Record start time'] = \
            UTCDateTime(year=header_item[20], julday=header_item[21],
                        hour=header_item[22], minute=header_item[23],
                        second=header_item[24], microsecond=header_item[25] *
                        100)
        self.fixed_header['Number of samples'] = int(header_item[26])
        self.fixed_header['Sample rate factor'] = int(header_item[27])
        self.fixed_header['Sample rate multiplier'] = int(header_item[28])
        self.fixed_header['Activity flags'] = int(header_item[29])
        self.fixed_header['I/O and clock flags'] = int(header_item[30])
        self.fixed_header['Data quality flags'] = int(header_item[31])
        self.fixed_header['Number of blockettes that follow'] = \
            int(header_item[32])
        self.fixed_header['Time correction'] = int(header_item[33])
        self.fixed_header['Beginning of data'] = int(header_item[34])
        self.fixed_header['First blockette'] = int(header_item[35])

    def _get_blockettes(self):
        """
        Loop over header and try to extract all header values!
        """
        self.blockettes = OrderedDict()
        cur_blkt_offset = self.fixed_header['First blockette']
        # Loop until the beginning of the data is reached.
        while True:
            if len(self.blockettes) == \
                    self.fixed_header["Number of blockettes that follow"]:
                break
            # Seek to the offset.
            self.file.seek(self.record_offset + cur_blkt_offset, 0)
            # Unpack the first two values. This is always the blockette type
            # and the beginning of the next blockette.
            encoding = native_str('%s2H' % self.endian)
            _tmp = self.file.read(4)
            try:
                blkt_type, next_blockette = unpack(encoding, _tmp)
            except Exception:
                if len(_tmp) == 0:
                    msg = "Unexpected end of file."
                    raise IOError(msg)
                raise
            blkt_type = int(blkt_type)
            next_blockette = int(next_blockette)
            self.blockettes[blkt_type] = self._parse_blockette(blkt_type)
            # Also break the loop if next_blockette is zero.
            if next_blockette == 0 or next_blockette < 4 or \
                    next_blockette - 4 < cur_blkt_offset:
                break
            cur_blkt_offset = next_blockette

    def _parse_blockette(self, blkt_type):
        """
        Parses the blockette blkt_type. If nothing is known about the blockette
        is will just return an empty dictionary.
        """
        blkt_dict = OrderedDict()
        # Check the blockette number.
        if blkt_type == 100:
            _tmp = self.file.read(8)
            try:
                unpack_values = unpack(native_str('%sfxxxx' % self.endian),
                                       _tmp)
            except Exception:
                if len(_tmp) == 0:
                    msg = "Unexpected end of file."
                    raise IOError(msg)
                raise
            blkt_dict['Sampling Rate'] = float(unpack_values[0])
        elif blkt_type == 1000:
            _tmp = self.file.read(4)
            try:
                unpack_values = unpack(native_str('%sBBBx' % self.endian),
                                       _tmp)
            except Exception:
                if len(_tmp) == 0:
                    msg = "Unexpected end of file."
                    raise IOError(msg)
                raise
            blkt_dict['Encoding Format'] = int(unpack_values[0])
            blkt_dict['Word Order'] = int(unpack_values[1])
            blkt_dict['Data Record Length'] = int(unpack_values[2])
        elif blkt_type == 1001:
            _tmp = self.file.read(4)
            try:
                unpack_values = unpack(native_str('%sbbxb' % self.endian),
                                       _tmp)
            except Exception:
                if len(_tmp) == 0:
                    msg = "Unexpected end of file."
                    raise IOError(msg)
                raise
            blkt_dict['Timing quality'] = int(unpack_values[0])
            blkt_dict['mu_sec'] = int(unpack_values[1])
            blkt_dict['Frame count'] = int(unpack_values[2])
        return blkt_dict

    def _calculate_start_time(self):
        """
        Calculates the true record starttime. See the SEED manual for all
        necessary information.

        Field 8 of the fixed header is the start of the time calculation. Field
        16 in the fixed header might contain a time correction. Depending on
        the setting of bit 1 in field 12 of the fixed header the record start
        time might already have been adjusted. If the bit is 1 the time
        correction has been applied, if 0 then not. Units of the correction is
        in 0.0001 seconds.

        Further time adjustments are possible in Blockette 500 and Blockette
        1001. So far no file with Blockette 500 has been encountered so only
        corrections in Blockette 1001 are applied. Field 4 of Blockette 1001
        stores the offset in microseconds of the starttime.
        """
        self.corrected_starttime = deepcopy(
            self.fixed_header['Record start time'])
        # Check whether or not the time correction has already been applied.
        if not self.fixed_header['Activity flags'] & 2:
            # Apply the correction.
            self.corrected_starttime += \
                self.fixed_header['Time correction'] * 0.0001
        # Check for blockette 1001.
        if 1001 in self.blockettes:
            self.corrected_starttime += self.blockettes[1001]['mu_sec'] * \
                1E-6

    def __str__(self):
        """
        Set the string representation of the class.
        """
        if self.filename:
            filename = self.filename
        else:
            filename = 'Unknown'
        if self.endian == '<':
            endian = 'Little Endian'
        else:
            endian = 'Big Endian'
        if self.did_goto:
            goto_info = (" (records were skipped, number is wrong in case "
                         "of differing record sizes)")
        else:
            goto_info = ""
        ret_val = ('FILE: %s\nRecord Number: %i%s\n' +
                   'Record Offset: %i byte\n' +
                   'Header Endianness: %s\n\n') % \
                  (filename, self.record_number, goto_info, self.record_offset,
                   endian)
        ret_val += 'FIXED SECTION OF DATA HEADER\n'
        for key in self.fixed_header.keys():
            ret_val += '\t%s: %s\n' % (key, self.fixed_header[key])
        ret_val += '\nBLOCKETTES\n'
        for key in self.blockettes.keys():
            ret_val += '\t%i:' % key
            if not len(self.blockettes[key]):
                ret_val += '\tNOT YET IMPLEMENTED\n'
            for _i, blkt_key in enumerate(self.blockettes[key].keys()):
                if _i == 0:
                    tabs = '\t'
                else:
                    tabs = '\t\t'
                ret_val += '%s%s: %s\n' % (tabs, blkt_key,
                                           self.blockettes[key][blkt_key])
        ret_val += '\nCALCULATED VALUES\n'
        ret_val += '\tCorrected Starttime: %s\n' % self.corrected_starttime
        return ret_val

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


def main(argv=None):
    """
    Entry point for setup.py.
    """
    parser = ArgumentParser(prog='obspy-mseed-recordanalyzer',
                            description=__doc__.split('\n')[0])
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + __version__)
    parser.add_argument('-n', default=0, type=int,
                        help='show info about N-th record (default: 0)')
    parser.add_argument('-a', '--all', dest="all",
                        default=False, action="store_true",
                        help=('show info for *all* records '
                              '(option "-n" has no effect in this case)'))
    parser.add_argument('-f', '--fast', dest="fast",
                        default=False, action="store_true",
                        help=('Jump to specified record number. Warning: '
                              'This assumes that all records have the same '
                              'size as the first one.'))
    parser.add_argument('filename', help='file to analyze')
    args = parser.parse_args(argv)

    rec = RecordAnalyser(args.filename)
    # read all records
    if args.all:
        while True:
            print(rec)
            try:
                next(rec)
            except StopIteration:
                sys.exit(0)
    # read single specified record
    if args.fast:
        try:
            rec.goto(args.n)
        except StopIteration as e:
            print(str(e))
            sys.exit(1)
    else:
        i = 0
        while i < args.n:
            i += 1
            try:
                next(rec)
            except StopIteration as e:
                print(str(e))
                sys.exit(1)
    print(rec)


if __name__ == "__main__":
    main()
