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

from obspy import UTCDateTime, Catalog, __version__
from obspy.core.event import Event


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
    if not hasattr(filename, "read"):
        # Check if it exists, otherwise assume its a string.
        try:
            with open(filename, "rb") as fh:
                data = fh.read()
            data = data.decode("UTF-8")
        except Exception:
            try:
                data = filename.decode("UTF-8")
            except Exception:
                data = str(filename)
            data = data.strip()
    else:
        data = filename.read()
        if hasattr(data, "decode"):
            data = data.decode("UTF-8")

    # split lines
    lines = [line for line in data.splitlines()]

    # line 6 in 'lst' format should look like this:
    # " Statn  Azimuth    TOA   Key  Log10 Ratio  NumPol  DenTOA  Comment"
    if lines[5].split() == [
            'Statn', 'Azimuth', 'TOA', 'Key', 'Log10', 'Ratio', 'NumPol',
            'DenTOA', 'Comment']:
        event = _read_focmec_lst(lines)
    # line 16 in 'out' format should look like this:
    # "    Dip   Strike   Rake    Pol: P     SV    SH  AccR/TotR  RMS RErr..."
    # But on older program version output, it's instead line number 14, so it
    # might depend on input data (polarities and/or amplitude ratios) and thus
    # what the program outputs as info (different settings available depending
    # on input data)
    else:
        for line in lines[4:30]:
            if line.split() == [
                    'Dip', 'Strike', 'Rake', 'Pol:', 'P', 'SV', 'SH',
                    'AccR/TotR', 'RMS', 'RErr', 'AbsMaxDiff']:
                event = _read_focmec_out(lines)
                break
        else:
            msg = ("Input was not recognized as either FOCMEC 'lst' or "
                   "'out' file format. Please contact developers if input "
                   "indeed is one of these two file types.")
            raise ValueError(msg)

    cat = Catalog(events=[event])
    cat.creation_info.creation_time = UTCDateTime()
    cat.creation_info.version = "ObsPy %s" % __version__
    return cat


def _read_focmec_lst(lines):
    """
    Read given data into an :class:`~obspy.core.event.Event` object.

    :type lines: list
    :param lines: List of decoded unicode strings with data from a FOCMEC lst
        file.
    """
    event = _read_common_header(lines)
    event.focal_mechanisms = []
    return event


def _read_focmec_out(lines):
    """
    Read given data into an :class:`~obspy.core.event.Event` object.

    :type lines: list
    :param lines: List of decoded unicode strings with data from a FOCMEC out
        file.
    """
    event = _read_common_header(lines)
    event.focal_mechanisms = []
    return event


def _read_common_header(lines):
    """
    Read given data into an :class:`~obspy.core.event.Event` object.

    Parses the first few common header lines and sets creation time and some
    other basic info.

    :type lines: list
    :param lines: List of decoded unicode strings with data from a FOCMEC out
        file.
    """
    event = Event()
    return event


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
