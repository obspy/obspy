# -*- coding: utf-8 -*-
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


def get_property(filename, key):
    """
    Given a property filename get the value of the given key
    """
    with open(filename, 'r') as fh:
        lines = fh.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith(key):
            ans = line[len(key) + 1:]
            return ans
    return ""
