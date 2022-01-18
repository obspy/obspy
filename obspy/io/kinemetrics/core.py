# -*- coding: utf-8 -*-
"""
Evt (Kinemetrics files) bindings to ObsPy's core classes.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from . import evt
from .evt_base import EvtBaseError


def is_evt(filename_or_object):
    """
    Checks whether a file is Evt or not.

    :type filename_or_object: str or file-like object
    :param filename_or_object: Evt file to be checked
    :rtype: bool
    :return: ``True`` if a Evt file, ``False`` otherwise
    """
    if hasattr(filename_or_object, "seek") and \
            hasattr(filename_or_object, "tell") and \
            hasattr(filename_or_object, "read"):
        is_fileobject = True
        pos = filename_or_object.tell()
    else:
        is_fileobject = False

    tag = evt.EvtTag()

    if is_fileobject:
        try:
            tag.read(filename_or_object)
            if tag.verify(verbose=False) is False:
                return False
            return True
        except EvtBaseError:
            return False
        finally:
            filename_or_object.seek(pos, 0)
    else:
        with open(filename_or_object, "rb") as file_obj:
            try:
                tag.read(file_obj)
                if tag.verify(verbose=False) is False:
                    return False
                return True
            except (EvtBaseError, IOError):
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
    stream = evt_obj.read_file(filename_or_object, **kwargs)
    return stream
