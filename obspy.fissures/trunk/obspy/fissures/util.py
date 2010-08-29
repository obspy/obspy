# -*- coding: utf-8 -*-
"""
Various additional utilities for obspy.fissures

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np
from obspy.core import AttribDict, UTCDateTime
import obspy.fissures.idl
from obspy.fissures.idl import Fissures
import warnings


class FissuresException(Exception):
    pass


class FissuresWarning(Warning):
    pass


def poleZeroFilter2PAZ(filter):
    """
    Converts a Fissures PoleZeroFilter to a PAZ dictionary.

    :type poleZeroFilter:
            :class:`~obspy.fissures.idl.Fissures.IfNetwork.PoleZeroFilter`
    :param poleZeroFilter: PoleZeroFilter to convert
    :returns: :class:`~obspy.core.util.AttribDict` containing PAZ information
    """
    # be nice and catch an easy to make mistake
    if isinstance(filter, obspy.fissures.idl.Fissures.IfNetwork.Filter):
        if str(filter._d) != "POLEZERO":
            raise TypeError("Filter is no PoleZeroFilter.")
        filter = filter._v
    paz = AttribDict()
    paz['poles'] = [complex(p.real, p.imaginary) for p in filter.poles]
    paz['zeros'] = [complex(z.real, z.imaginary) for z in filter.zeros]
    # check if errors are specified and show a warning
    errors = [p.real_error for p in filter.poles] + \
             [p.imaginary_error for p in filter.poles] + \
             [z.real_error for z in filter.zeros] + \
             [z.imaginary_error for z in filter.zeros]
    if any(errors):
        msg = "Filter contains error information that is lost during " + \
              "conversion."
        warnings.warn(msg)
    return paz


def utcdatetime2Fissures(utc_datetime):
    """
    Convert datetime instance to fissures time object
    
    :type utc_datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param utc_datetime: Time
    :return: Time as :class:`~obspy.fissures.idl.Fissures.Time` object
    """
    t = str(UTCDateTime(utc_datetime))[:-3] + 'Z'
    return Fissures.Time(t, -1)


def use_first_and_raise_or_warn(list, type_str):
    """
    Chooses first element of given list.
    If list is empty raises a FissuresException, if list has more than one
    element issues a FissuresWarning with the given type_str.
    """
    if not list:
        raise FissuresException("No data for %s." % type_str)
    elif len(list) > 1:
        msg = "Received more than one %s object from server. " % type_str + \
              "Using first."
        warnings.warn(msg, FissuresWarning)
    return list[0]


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
