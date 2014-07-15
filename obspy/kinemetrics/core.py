#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EVT (Kinemetrics files) bindings to ObsPy's core classes.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import obspy.kinemetrics.evt as evt
from .evt_base import EVTBaseError


def is_evt(filename):
    """
    Checks whether a file is EVT or not.

    :type filename: string
    :param filename: EVT file to be checked.
    :rtype: bool
    :return: ``True`` if a EVT file, ``False`` otherwise.
    """
    try:
        Tag = evt.EVT_TAG()
        with open(filename, "rb") as fh:
            Tag.read(fh)
            Tag.verify()
    except (EVTBaseError, IOError):
        return False
    return True


def read_evt(filename, **kwargs):
    """
    Reads a EVT file and returns a Stream object.

    .. warning::
    This function should NOT be called directly, it registers via the
    ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: string
    :param filename: EVT file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: Stream object containing header and data.
    """
    Evt_Obj = evt.EVT()
    stream = Evt_Obj.readFile(filename)
    return stream
