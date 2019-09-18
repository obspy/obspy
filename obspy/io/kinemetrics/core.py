# -*- coding: utf-8 -*-
"""
Evt (Kinemetrics files) bindings to ObsPy's core classes.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.core.util.decorator import file_format_check

from . import evt
from .evt_base import EvtBaseError


@file_format_check
def is_evt(filename_or_object, **kwargs):
    """
    Checks whether a file is Evt or not.

    :type filename_or_object: :class:`io.BytesIOBase`
    :param filename_or_object: Open file or file-like object to be checked
    :rtype: bool
    :return: ``True`` if a Evt file, ``False`` otherwise
    """
    fh = filename_or_object
    tag = evt.EvtTag()
    try:
        tag.read(fh)
        if tag.verify(verbose=False) is False:
            return False
        return True
    except EvtBaseError:
        return False


def read_evt(filename_or_object, **kwargs):
    """
    Reads a Evt file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead

    :type filename_or_object: str or file-like object
    :param filename_or_object: Evt file to be read
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: Stream object containing header and data
    """
    evt_obj = evt.Evt()
    stream = evt_obj.read_file(filename_or_object)
    return stream
