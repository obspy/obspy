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
