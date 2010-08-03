# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
#  Filename: dev.py
#  Purpose: Tools to analyse Mini-SEED records for development
#           purposes.
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#
# Copyright (C) 2010 Lion Krischer
#---------------------------------------------------------------------
"""
Provides some tools to analyse Mini-SEED records.
"""
from copy import deepcopy
from struct import unpack

from obspy.core import UTCDateTime


class RecordAnalyser(object):
    """
    Analyses a Mini-SEED file on a per record basis.

    Basic usage:
        >> rec = RecordAnalyser(filename)
        # Pretty print the information contained in the first record.
        >> print rec
        # Jump to the next record.
        >> rex.next()
        >> print rec
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
        # Parse the header.
        self._parseHeader()

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

    def next(self):
        """
        Jumps to the next record and parses the header.
        """
        self.record_offset += 2 ** self.blockettes[1000]['Data Record Length']
        self._parseHeader()

    def _parseHeader(self):
        """
        Makes all necessary calls to parse the header.
        """
        # Big or little endian for the header.
        self._getEndianess()
        # Read the fixed header.
        self._readFixedHeader()
        # Get the present blockettes.
        self._getBlockettes()
        # Calculate the starttime.
        self._calculateStarttime()

    def _getEndianess(self):
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
        # Get the year.
        year = unpack('>H', self.file.read(2))[0]
        if year >= 1900 and year <= 2050:
            self.endian = '>'
        else:
            self.endian = '<'
        # Reset the pointer.
        self.file.seek(current_pointer, 0)

    def _readFixedHeader(self):
        """
        Reads the fixed header of the Mini-SEED file and writes all entries to
        self.fixed_header, a dictionary.
        """
        # Init empty fixed header dictionary. Use an ordered dictionary to
        # achieve the same order as in the Mini-SEED manual.
        self.fixed_header = SimpleOrderedDict()
        # Read and unpack.
        self.file.seek(self.record_offset, 0)
        fixed_header = self.file.read(48)
        encoding = ('%s20c2H3Bx4H4Bl2H' % self.endian)
        header_item = unpack(encoding, fixed_header)
        # Write values to dictionary.
        self.fixed_header['Sequence number'] = int(''.join(header_item[:6]))
        self.fixed_header['Data header/quality indicator'] = header_item[6]
        self.fixed_header['Station identifier code'] = \
                ''.join(header_item[8:13]).strip()
        self.fixed_header['Location identifier'] = \
                ''.join(header_item[13:15]).strip()
        self.fixed_header['Channel identifier'] = \
                ''.join(header_item[15:18]).strip()
        self.fixed_header['Network code'] = \
                ''.join(header_item[18:20]).strip()
        # Construct the starttime. This is only the starttime in the fixed
        # header without any offset. See page 31 of the SEED manual for the
        # time definition.
        self.fixed_header['Record start time'] = \
                UTCDateTime(year=header_item[20], julday=header_item[21],
                hour=header_item[22], minute=header_item[23],
                second=header_item[24], microsecond=header_item[25] * 100)
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

    def _getBlockettes(self):
        """
        Loop over header and try to extract all header values!
        """
        self.blockettes = SimpleOrderedDict()
        cur_blkt_offset = self.fixed_header['First blockette']
        # Loop until the beginning of the data is reached.
        while True:
            if cur_blkt_offset >= self.fixed_header['Beginning of data']:
                break
            # Seek to the offset.
            self.file.seek(cur_blkt_offset, 0)
            # Unpack the first two values. This is always the blockette type
            # and the beginning of the next blockette.
            blkt_type, next_blockette = unpack('%s2H' % self.endian,
                                               self.file.read(4))
            blkt_type = int(blkt_type)
            next_blockette = int(next_blockette)
            cur_blkt_offset = next_blockette
            self.blockettes[blkt_type] = self._parseBlockette(blkt_type)
            # Also break the loop if next_blockette is zero.
            if next_blockette == 0:
                break

    def _parseBlockette(self, blkt_type):
        """
        Parses the blockette blkt_type. If nothing is known about the blockette
        is will just return an empty dictionary.
        """
        blkt_dict = SimpleOrderedDict()
        # Check the blockette number.
        if blkt_type == 1000:
            unpack_values = unpack('%s3B' % self.endian,
                                               self.file.read(3))
            blkt_dict['Encoding Format'] = int(unpack_values[0])
            blkt_dict['Word Order'] = int(unpack_values[1])
            blkt_dict['Data Record Length'] = int(unpack_values[2])
        elif blkt_type == 1001:
            unpack_values = unpack('%sBBxB' % self.endian,
                                               self.file.read(4))
            blkt_dict['Timing quality'] = int(unpack_values[0])
            blkt_dict['mu_sec'] = int(unpack_values[1])
            blkt_dict['Frame count'] = int(unpack_values[2])
        return blkt_dict

    def _calculateStarttime(self):
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
        self.corrected_starttime = deepcopy(\
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
            ret_val = ('FILE: %s\nRecord Offset: %i byte\n' +
                      'Header Endianness: %s\n\n') % \
                      (filename, self.record_offset, endian)
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


class SimpleOrderedDict(object):
    """
    Provides a very simple ordered dict for pre 2.7 Python versions. Does not
    support much but keys will always be returned in the same order as they
    are entered.
    """
    def __init__(self):
        # Init the list that will keep track of the keys.
        self.ordered_list = []

    def __eq__(self, other):
        """
        Will not check if both are sorted in the same way but just the
        contents.
        """
        if not len(self) == len(other):
            return False
        for key in self.ordered_list:
            if key not in other.ordered_list:
                return False
            if self[key] != other[key]:
                return False
        return True

    def __ne__(self, other):
        if self.__eq__(other):
            return False
        return True

    def __getitem__(self, key):
        if key in self.ordered_list:
            return self.__dict__[key]
        else:
            raise KeyError

    def __contains__(self, key):
        if key in self.ordered_list:
            return True
        else:
            return False

    def __len__(self):
        return len(self.ordered_list)

    def __setitem__(self, key, value):
        """
        Add to list and dictionary to enable sorting.
        """
        if not key in self.ordered_list:
            self.ordered_list.append(key)
        self.__dict__[key] = value

    def keys(self):
        return DictIter(self)


class DictIter(object):
    """
    Iterator for the keys of the ordered dict.
    """
    def __init__(self, ordered_dict):
        """
        Takes a SimpleOrderedDict instance.
        """
        self.cur_item = 0
        self.dict = ordered_dict

    def __iter__(self):
        return self

    def next(self):
        try:
            self.cur_item += 1
            return self.dict.ordered_list[self.cur_item - 1]
        except IndexError:
            raise StopIteration
