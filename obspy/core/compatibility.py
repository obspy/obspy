#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compatibility layer for ObsPy. Enables the same code basis to be used from
Python 2.6 to 3.3.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np
import sys

if sys.version_info[0] == 3:
    PY3K = True
else:
    PY3K = False

# Useful for isinstance(value, string) checks. Py3K does not have seperate
# unicode strings and thus no basestring type.
if PY3K is True:
    string = str
else:
    string = basestring


if PY3K is True:
    raw_input = input
else:
    raw_input = raw_input

# PY3K does not have iteritems(), instead items() is already an iterator.
if PY3K is True:
    __iteritems__ = "items"
else:
    __iteritems__ = "iteritems"

def iteritems(obj):
    return getattr(obj, __iteritems__)()

# range() in PY3K is an iterator.
if PY3K is True:
    range = range
else:
    range = xrange

# range() in PY3K is an iterator.
if PY3K is True:
    izip = zip
else:
    from itertools import izip
    izip = izip

# Should work under Python 2 and 3.
try:
    import cPickle as pickle
except:
    import pickle

if PY3K is True:
    from urllib.request import urlopen
else:
    from urllib2 import urlopen


def round_away(number):
    """
    Simple function that rounds a number to the nearest integer. If the number
    is halfway between two integers, it will round away from zero. Of course
    only works up machine precision. This should hopefully behave like the
    round() function in Python 2.

    This is potentially desired behaviour in the trim functions but some more
    thought should be poured into it.

    The np.round() function rounds towards the even nearest even number in case
    of half-way splits.

    >>> round_away(2.5)
    3
    >>> round_away(-2.5)
    -3

    >>> round_away(10.5)
    11
    >>> round_away(-10.5)
    -11

    >>> round_away(11.0)
    11
    >>> round_away(-11.0)
    -11
    """
    floor = np.floor(number)
    ceil = np.ceil(number)
    if (floor != ceil) and (abs(number - floor) == abs(ceil - number)):
        return int(int(number) + int(np.sign(number)))
    else:
        return int(np.round(number))

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
