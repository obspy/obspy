#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EVT (Kinemetrics files) bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import unicode_literals
from __future__ import print_function

from future import standard_library  # NOQA
from future.builtins import open

import sys
import warnings
import obspy.kinemetrics.evt as evt
from . evt_base import EVTBaseError


def is_evt(filename):
    """
    Checks whether a file is EVT or not.

    :type filename: string
    :param filename: EVT file to be checked.
    :rtype: bool
    :return: ``True`` if a EVT file.
    """
    try:
        Tag = evt.EVT_TAG()
        fpin = open(filename, "rb")
        Tag.read(fpin)
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

    try:
        Evt_Obj = evt.EVT()
        stream = Evt_Obj.readFile(filename)
    except EVTBaseError:
        (etype, value, traceback) = sys.exc_info()
        message = "Error : " + value.message + str(traceback)
        warnings.warn(message)
        stream = False
    return stream
