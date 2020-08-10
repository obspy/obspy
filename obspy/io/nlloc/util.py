# -*- coding: utf-8 -*-
"""
NonLinLoc file format support for ObsPy

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
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
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("pdf", "<f4")])
    data = np.fromfile(filename, dtype=dtype)[4:]
    if coordinate_converter:
        data["x"], data["y"], data["z"] = coordinate_converter(
            data["x"], data["y"], data["z"])
    return data
