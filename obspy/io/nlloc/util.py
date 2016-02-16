#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NonLinLoc file format support for ObsPy

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import native_str

import numpy as np


def read_nlloc_scatter(filename, coordinate_converter=None):
    """
    Read a NonLinLoc Scatter file into a Numpy array.

    .. note::

        Coordinate conversion from coordinate frame of NonLinLoc model files /
        location run to WGS84 has to be specified explicitly by the user if
        necessary.

    :type filename: str
    :param filename: Filename with NonLinLoc scatter.
    :type coordinate_converter: func
    :param coordinate_converter: Function to convert (x, y, z)
        coordinates of NonLinLoc output to geographical coordinates and depth
        in meters (longitude, latitude, depth in kilometers).
        If left ``None``, NonLinLoc (x, y, z) output is left unchanged (e.g. if
        it is in geographical coordinates already like for NonLinLoc in
        global mode).
        The function should accept three arguments x, y, z (each of type
        :class:`numpy.ndarray`) and return a tuple of three
        :class:`numpy.ndarray` (lon, lat, depth in kilometers).
    :returns: NonLinLoc scatter information as structured numpy array (fields:
        "x", "y", "z", "pdf").
    """
    # omit the first 4 values (header information) and reshape
    dtype = np.dtype([
        (native_str("x"), native_str("<f4")),
        (native_str("y"), native_str("<f4")),
        (native_str("z"), native_str("<f4")),
        (native_str("pdf"), native_str("<f4"))])
    data = np.fromfile(filename, dtype=dtype)[4:]
    if coordinate_converter:
        data["x"], data["y"], data["z"] = coordinate_converter(
            data["x"], data["y"], data["z"])
    return data


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
