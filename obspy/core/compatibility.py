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
