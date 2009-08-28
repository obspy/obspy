# -*- coding: utf-8 -*-

import datetime


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


def DateTime2String(t):
    """
    Generates a valid SEED time string from a DateTime or Date object.
    
    This function is adopted from fseed.py, the SEED builder for SeisComP 
    written by Andres Heinloo, GFZ Potsdam in 2005.
    """
    if t == None:
        return ""
    elif isinstance(t, datetime.datetime):
        tt = t.utctimetuple()
        return "%04d,%03d,%02d:%02d:%02d.%04d" % (t.year, tt[7],
            t.hour, t.minute, t.second, t.microsecond // 100)
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
    """
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
        import pdb;pdb.set_trace()


