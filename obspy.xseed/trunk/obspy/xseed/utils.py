# -*- coding: utf-8 -*-

import datetime
from obspy.core import UTCDateTime
from obspy.core.util import formatScientific
import sys


def toAttribute(name):
    """
    Creates a valid attribute name from a given string.
    """
    #return name.lower().replace(' ','_')
    temp = name.lower().replace(' ', '_')
    temp = temp.replace('fir_', 'FIR_')
    temp = temp.replace('a0_', 'A0_')
    return temp


def toXMLTag(name):
    """
    Creates a XML tag from a given string.
    """
    temp = name.lower().replace(' ', '_')
    temp = temp.replace('fir_', 'FIR_')
    temp = temp.replace('a0_', 'A0_')
    return temp


def DateTime2String(t, compact=False):
    """
    Generates a valid SEED time string from a DateTime or Date object.
    
    This function is adopted from fseed.py, the SEED builder for SeisComP 
    written by Andres Heinloo, GFZ Potsdam in 2005.
    """
    if t == None:
        return ""
    elif isinstance(t, datetime.datetime):
        tt = t.utctimetuple()
        if not compact:
            return "%04d,%03d,%02d:%02d:%02d.%04d" % (t.year, tt[7], t.hour,
                                                      t.minute, t.second,
                                                      t.microsecond // 100)
        temp = "%04d,%03d" % (t.year, tt[7])
        if not t.hour:
            return temp
        temp += ",%02d" % t.hour
        if not t.minute:
            return temp
        temp += ":%02d" % t.minute
        if not t.second:
            return temp
        temp += ":%02d" % t.second
        if not t.microsecond:
            return temp
        temp += ".%04d" % (t.microsecond // 100)
        return temp
    elif isinstance(t, datetime.date):
        tt = datetime.datetime.combine(t,
                                       datetime.time(0, 0, 0)).utctimetuple()
        return "%04d,%03d" % (t.year, tt[7])
    raise Exception("Invalid python date object: " + str(t))


def String2DateTime(s):
    """
    Generates either a DateTime or Date object from a valid SEED time string.
    """
    s = s.strip()
    if not s:
        return None
    if s.count(':') == 2 and s.count(',') == 2:
        # w/ seconds
        m = '0000'
        if '.' in s:
            s, m = s.split('.')
        dt = datetime.datetime.strptime(s, "%Y,%j,%H:%M:%S")
        lenm = len(m)
        if lenm > 0 and lenm <= 6:
            dt = dt.replace(microsecond=int(m) * pow(10, 6 - lenm))
    elif s.count(':') == 1 and s.count(',') == 2:
        # w/o seconds
        dt = datetime.datetime.strptime(s, "%Y,%j,%H:%M")
    elif s.count(',') == 2:
        # w/o minutes
        dt = datetime.datetime.strptime(s, "%Y,%j,%H")
    elif s.count(',') == 1:
        # only date
        dt = datetime.datetime.strptime(s, "%Y,%j").date()
    else:
        raise Exception("Invalid SEED date object: " + str(s))
    return dt


def Iso2DateTime(s):
    """
    Generates either a DateTime or Date object from a ISO 8601 time string.
    """
    s = s.strip()
    if not s:
        return None
    if ':' in s:
        m = 0
        if '.' in s:
            s, m = s.split('.')
        dt = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
        dt = dt.replace(microsecond=int(m))
    else:
        dt = datetime.datetime.strptime(s, "%Y-%m-%d").date()
    return dt


def DateTime2Iso(t):
    if t == None:
        return ""
    elif isinstance(t, datetime.datetime):
        return t.isoformat()
    elif isinstance(t, datetime.date):
        return t.isoformat()
    raise Exception("Invalid python date object: " + str(t))


def compareSEED(seed1, seed2):
    """
    Compares two SEED files.
    
    Only works with a record length of 4096 bytes.
    """
    # Each SEED string should be a multiple of the record length.
    assert (len(seed1) % 4096) == 0
    assert (len(seed2) % 4096) == 0
    # Loop over each record and remove empty ones. obspy.xseed doesn't write
    # empty records. Redundant code to ease coding...
    recnums = len(seed1) / 4096
    new_seed1 = ''
    for _i in xrange(recnums):
        cur_record = seed1[_i * 4096 + 8: (_i + 1) * 4096].strip()
        if cur_record == '':
            continue
        new_seed1 += seed1[_i * 4096 : (_i + 1) * 4096]
    seed1 = new_seed1
    recnums = len(seed2) / 4096
    new_seed2 = ''
    for _i in xrange(recnums):
        cur_record = seed2[_i * 4096 + 8: (_i + 1) * 4096].strip()
        if cur_record == '':
            continue
        new_seed2 += seed2[_i * 4096 : (_i + 1) * 4096]
    seed2 = new_seed2
    # length should be the same
    assert len(seed1) == len(seed2)
    # version string is always ' 2.4' for output
    if seed1[15:19] == ' 2.3':
        seed1 = seed1.replace(' 2.3', ' 2.4', 1)
    if seed1[15:19] == '02.3':
        seed1 = seed1.replace('02.3', ' 2.4', 1)
    # check for missing '~' in blockette 10 (faulty dataless from BW network)
    l = int(seed1[11:15])
    temp = seed1[0:(l + 8)]
    if temp.count('~') == 4:
        # added a '~' and remove a space before the next record
        # record length for now 4096
        seed1 = seed1[0:11] + '%04i' % (l + 1) + seed1[15:(l + 8)] + '~' + \
                seed1[(l + 8):4095] + seed1[4096:]
    # check each byte
    for i in xrange(0, len(seed1)):
        if seed1[i] == seed2[i]:
            continue
        temp = seed1[i] + seed2[i]
        if temp == '0+':
            continue
        if temp == '0 ':
            continue
        if temp == ' +':
            continue
        if temp == '- ':
            # -056.996398+0031.0
            #  -56.996398  +31.0
            continue


def SEEDtoRESPTime(seedstring):
    """
    Converts a SEED date into a RESP date string.
    """
    time = UTCDateTime(seedstring)
    if len(seedstring) == 10:
        return time.strftime('%Y,%j')
    else:
        resp_string = time.strftime('%Y,%j,%H:%M:%S.')
        # Microseconds needs a special treatment.
        ms = '%04d' % int(round(int(time.strftime('%f')) / 100.))
    return resp_string + ms


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
    return formatScientific(format_string % number)


def Blockette34Lookup(abbr, lookup):
    try:
        l1 = LookupCode(abbr, 34, 'unit_name', 'unit_lookup_code', lookup)
        l2 = LookupCode(abbr, 34, 'unit_description', 'unit_lookup_code',
                        lookup)
        return l1 + ' - ' + l2
    except:
        msg = '\nWarning: Abbreviation reference not found.'
        sys.stdout.write(msg)
        return 'No Abbreviation Referenced'
