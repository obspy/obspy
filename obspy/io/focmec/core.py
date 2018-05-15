#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FOCMEC file format support for ObsPy

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport


from obspy.core.event import Catalog


def _is_focmec(filename):
    """
    Checks that a file is actually a FOCMEC output data file
    """
    try:
        with open(filename, 'rb') as fh:
            line = fh.readline()
    except Exception:
        return False
    # first line should be ASCII only, something like:
    #   Fri Sep  8 14:54:58 2017 for program Focmec
    try:
        line = line.decode('ASCII')
    except:
        return False
    line = line.split()
    # program name 'focmec' at the end is written slightly differently
    # depending on how focmec was compiled, sometimes all lower case sometimes
    # capitalized..
    line[-1] = line[-1].lower()
    if line[-3:] == ['for', 'program', 'focmec']:
        return True
    return False


def _read_focmec(filename, **kwargs):
    """
    Reads a FOCMEC '.lst' or '.out' file to a
    :class:`~obspy.core.event.Catalog` object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.event.catalog.read_events()` function, call
        this instead.

    :param filename: File or file-like object in text mode.
    :rtype: :class:`~obspy.core.event.Catalog`
    """
    return Catalog()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
