# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from struct import pack, unpack

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


def _pack_attribute_nicer_exception(obj, name, format):
    """
    packs obj.name with the given format but raises a nicer error message.
    """
    x = getattr(obj, name)
    try:
        return pack(format, x)
    except Exception as e:
        msg = ("Failed to pack header value `%s` (%s) with format `%s` due "
               "to: `%s`")
        try:
            format = format.decode()
        except AttributeError:
            pass
        raise ValueError(msg % (name, str(x), format, e.args[0]))
