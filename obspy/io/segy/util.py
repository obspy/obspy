# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import io
import os
from struct import unpack

from obspy.core.util.libnames import _load_cdll


# Import shared libsegy
clibsegy = _load_cdll("segy")


def unpack_header_value(endian, packed_value, length, special_format):
    """
    Unpacks a single value.
    """
    # Use special format if necessary.
    if special_format:
        fmt = ('%s%s' % (endian, special_format)).encode('ascii', 'strict')
        return unpack(fmt, packed_value)[0]
    # Unpack according to different lengths.
    elif length == 2:
        format = ('%sh' % endian).encode('ascii', 'strict')
        return unpack(format, packed_value)[0]
    # Update: Seems to be correct. Two's complement integers seem to be
    # the common way to store integer values.
    elif length == 4:
        format = ('%si' % endian).encode('ascii', 'strict')
        return unpack(format, packed_value)[0]
    # The unassigned field. Since it is unclear how this field is
    # encoded it will just be stored as a string.
    elif length == 8:
        return packed_value
    # Should not happen
    else:
        raise Exception


def get_filesize(file):
    """Get a size of provided file-like object."""
    try:
        fileno = file.fileno()
    except (AttributeError, io.UnsupportedOperation):
        pos = file.tell()
        file.seek(0, 2)  # go t end of file
        filesize = file.tell()
        file.seek(pos, 0)
    else:
        filesize = os.fstat(fileno)[6]

    return filesize
