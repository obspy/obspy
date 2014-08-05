# -*- coding: utf-8 -*-
"""
Various additional utilities for ObsPy xseed.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import native_str

import sys

from obspy import UTCDateTime


# Ignore Attributes of Blockettes
IGNORE_ATTR = ['blockette_id', 'blockette_name', 'compact', 'debug',
               'seed_version', 'strict', 'xseed_version',
               'length_of_blockette', 'blockette_type']


class SEEDParserException(Exception):
    pass


def toTag(name):
    """
    Creates a XML tag from a given string.
    """
    temp = name.lower().replace(' ', '_')
    temp = temp.replace('fir_', 'FIR_')
    temp = temp.replace('a0_', 'A0_')
    return temp


def toString(tag):
    """
    Creates a pretty string from any given XML tag.
    """
    temp = tag.replace('_', ' ').title()
    temp = temp.replace('Fir', 'FIR')
    return temp


def DateTime2String(dt, compact=False):
    """
    Generates a valid SEED time string from a UTCDateTime object.
    """
    if isinstance(dt, UTCDateTime):
        return dt.formatSEED(compact)
    elif isinstance(dt, (str, native_str)):
        dt = dt.strip()
    if not dt:
        return ""
    try:
        dt = UTCDateTime(dt)
        return dt.formatSEED(compact)
    except:
        raise Exception("Invalid datetime %s: %s" % (type(dt), str(dt)))


def compareSEED(seed1, seed2):
    """
    Compares two SEED files.

    Only works with a record length of 4096 bytes.
    """
    # Each SEED string should be a multiple of the record length.
    if (len(seed1) % 4096) != 0:
        msg = "Length of first SEED string should be a multiple of 4096 bytes"
        raise Exception(msg)
    if (len(seed2) % 4096) != 0:
        msg = "Length of second SEED string should be a multiple of 4096 bytes"
        raise Exception(msg)
    # Loop over each record and remove empty ones. obspy.xseed doesn't write
    # empty records. Redundant code to ease coding...
    recnums = len(seed1) // 4096
    new_seed1 = b''
    for _i in range(recnums):
        cur_record = seed1[_i * 4096 + 8:(_i + 1) * 4096].strip()
        if cur_record == b'':
            continue
        new_seed1 += seed1[_i * 4096:(_i + 1) * 4096]
    seed1 = new_seed1
    recnums = len(seed2) // 4096
    new_seed2 = b''
    for _i in range(recnums):
        cur_record = seed2[_i * 4096 + 8:(_i + 1) * 4096].strip()
        if cur_record == b'':
            continue
        new_seed2 += seed2[_i * 4096:(_i + 1) * 4096]
    seed2 = new_seed2
    # length should be the same
    if len(seed1) != len(seed2):
        msg = "Length of SEED strings differ! (%d != %d)" % (len(seed1),
                                                             len(seed2))
        raise Exception(msg)
    # version string is always ' 2.4' for output
    if seed1[15:19] == b' 2.3':
        seed1 = seed1.replace(b' 2.3', b' 2.4', 1)
    if seed1[15:19] == b'02.3':
        seed1 = seed1.replace(b'02.3', b' 2.4', 1)
    # check for missing '~' in blockette 10 (faulty dataless from BW network)
    l = int(seed1[11:15])
    temp = seed1[0:(l + 8)]
    if temp.count(b'~') == 4:
        # added a '~' and remove a space before the next record
        # record length for now 4096
        b_l = ('%04i' % (l + 1)).encode('ascii', 'strict')
        seed1 = seed1[0:11] + b_l + seed1[15:(l + 8)] + b'~' + \
            seed1[(l + 8):4095] + seed1[4096:]
    # check each byte
    for i in range(0, len(seed1)):
        if seed1[i] == seed2[i]:
            continue
        temp = seed1[i] + seed2[i]
        if temp == b'0+':
            continue
        if temp == b'0 ':
            continue
        if temp == b' +':
            continue
        if temp == b'- ':
            # -056.996398+0031.0
            #  -56.996398  +31.0
            continue


def LookupCode(blockettes, blkt_number, field_name, lookup_code,
               lookup_code_number):
    """
    Loops over a list of blockettes until it finds the blockette with the
    right number and lookup code.
    """
    # List of all possible names for lookup
    for blockette in blockettes:
        if blockette.id != blkt_number:
            continue
        if getattr(blockette, lookup_code) != lookup_code_number:
            continue
        return getattr(blockette, field_name)
    return None


def formatRESP(number, digits=4):
    """
    Formats a number according to the RESP format.
    """
    format_string = "%%-10.%dE" % digits
    return format_string % (number)


def Blockette34Lookup(abbr, lookup):
    """
    Gets certain values from blockette 34. Needed for RESP output.
    """
    try:
        l1 = LookupCode(abbr, 34, 'unit_name', 'unit_lookup_code', lookup)
        l2 = LookupCode(abbr, 34, 'unit_description', 'unit_lookup_code',
                        lookup)
        return l1 + ' - ' + l2
    except:
        msg = '\nWarning: Abbreviation reference not found.'
        sys.stdout.write(msg)
        return 'No Abbreviation Referenced'


def setXPath(blockette, identifier):
    """
    Returns an X-Path String to a blockette with the correct identifier.
    """
    try:
        identifier = int(identifier)
    except:
        msg = 'X-Path identifier needs to be an integer.'
        raise TypeError(msg)
    abbr_path = '/xseed/abbreviation_dictionary_control_header/'
    end_of_path = '[text()="%s"]/parent::*'
    if blockette == 30:
        return abbr_path + \
            'data_format_dictionary/data_format_identifier_code' + \
            end_of_path % identifier
    elif blockette == 31:
        return abbr_path + \
            'comment_description/comment_code_key' + \
            end_of_path % identifier
    elif blockette == 33:
        return abbr_path + \
            'generic_abbreviation/abbreviation_lookup_code' + \
            end_of_path % identifier
    elif blockette == 34:
        return abbr_path + \
            'units_abbreviations/unit_lookup_code' + \
            end_of_path % identifier
    # All dictionary blockettes.
    elif blockette == 'dictionary':
        return abbr_path + \
            '*/response_lookup_key' + \
            end_of_path % identifier
    msg = 'XPath for blockette %d not implemented yet.' % blockette
    raise NotImplementedError(msg)


def getXPath(xpath):
    """
    Returns lookup key of XPath expression on abbreviation dictionary.
    """
    return int(xpath.split('"')[-2])


def uniqueList(seq):
    # Not order preserving
    keys = {}
    for e in seq:
        keys[e] = 1
    return list(keys.keys())
