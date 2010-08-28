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
from obspy.core import AttribDict
import obspy.fissures.idl
import warnings


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
            raise Exception("Filter is no PoleZeroFilter.")
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


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
