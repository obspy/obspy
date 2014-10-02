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


def is_evt(filename_or_object):
    """
    Checks whether a file is EVT or not.

    :type filename_or_object: filename or file-like object
    :param filename_or_object: EVT file to be checked
    :rtype: bool
    :return: ``True`` if a EVT file, ``False`` otherwise
    """
    if hasattr(filename_or_object, "seek") and \
            hasattr(filename_or_object, "tell") and \
            hasattr(filename_or_object, "read"):
        is_fileobject = True
        pos = filename_or_object.tell()
    else:
        is_fileobject = False

    Tag = evt.EVT_TAG()

    if is_fileobject:
        try:
            Tag.read(filename_or_object)
            if Tag.verify(verbose=False) is False:
                return False
            return True
        except EVTBaseError:
            return False
        finally:
            filename_or_object.seek(pos, 0)
    else:
        with open(filename_or_object, "rb") as fh:
            try:
                Tag.read(fh)
                if Tag.verify(verbose=False) is False:
                    return False
                return True
            except (EVTBaseError, IOError):
                return False


def read_evt(filename_or_object, **kwargs):
    """
    Reads a EVT file and returns a Stream object.

    .. warning::
		This function should NOT be called directly, it registers via the
		ObsPy :func:`~obspy.core.stream.read` function, call this instead

    :type filename_or_object: str or file-like object
    :param filename_or_object: EVT file to be read
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: Stream object containing header and data
    """
    Evt_Obj = evt.EVT()
    stream = Evt_Obj.readFile(filename_or_object)
    return stream
