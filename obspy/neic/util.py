# -*- coding: utf-8 -*-

import string
from datetime import datetime


def asctime():
    """
    Returns the current time as a string hh:mm:ss
    """
    a = str(datetime.utcnow())
    return a[11:19]


def ascdate():
    """
    Returns the current date at yy/mm/dd
    """
    a = str(datetime.utcnow())
    return a[2:10]


def dsecs(dt):
    """
    Given a timedelta object compute it as double seconds.
    """
    d = dt.days * 86400.
    d = d + dt.seconds
    d = d + dt.microseconds / 1000000.0
    return d


def getProperty(file, key):
    """
    Given a property filename get the value of the given key
    """
    file = open(file, 'r')
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = string.strip(lines[i])
        if lines[i][0:len(key)] == key:
            ans = lines[i][len(key) + 1:]
            return ans
    return ""
