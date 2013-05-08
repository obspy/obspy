#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: rotate.py
#  Purpose: Various Seismogram Rotation Functions
#   Author: Tobias Megies, Tom Richter, Lion Krischer
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2009-2013 Tobias Megies, Tom Richter, Lion Krischer
#---------------------------------------------------------------------
"""
Various Seismogram Rotation Functions

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from math import pi, sin, cos
import numpy as np


def rotate_NE_RT(n, e, ba):
    """
    Rotates horizontal components of a seismogram.

    The North- and East-Component of a seismogram will be rotated in Radial
    and Transversal Component. The angle is given as the back-azimuth, that is
    defined as the angle measured between the vector pointing from the station
    to the source and the vector pointing from the station to the north.

    :type n: :class:`~numpy.ndarray`
    :param n: Data of the North component of the seismogram.
    :type e: :class:`~numpy.ndarray`
    :param e: Data of the East component of the seismogram.
    :type ba: float
    :param ba: The back azimuth from station to source in degrees.
    :return: Radial and Transversal component of seismogram.
    """
    if len(n) != len(e):
        raise TypeError("North and East component have different length.")
    if ba < 0 or ba > 360:
        raise ValueError("Back Azimuth should be between 0 and 360 degrees.")
    r = e * sin((ba + 180) * 2 * pi / 360) + n * cos((ba + 180) * 2 * pi / 360)
    t = e * cos((ba + 180) * 2 * pi / 360) - n * sin((ba + 180) * 2 * pi / 360)
    return r, t


def rotate_RT_NE(n, e, ba):
    """
    Rotates horizontal components of a seismogram.

    Rotates from radial and tranversal components to north and east
    components.

    This is the inverse transformation of the transformation described
    in :func:`rotate_NE_RT`.
    """
    ba = 360.0 - ba
    return rotate_NE_RT(n, e, ba)


def rotate_ZNE_LQT(z, n, e, ba, inc):
    """
    Rotates all components of a seismogram.

    The components will be rotated from ZNE (Z, North, East, left-handed) to
    LQT (e.g. ray coordinate system, right-handed). The rotation angles are
    given as the back-azimuth and inclination.

    The transformation consists of 3 steps::

        1. mirroring of E-component at ZN plain: ZNE -> ZNW
        2. negative rotation of coordinate system around Z-axis with angle ba:
           ZNW -> ZRT
        3. negative rotation of coordinate system around T-axis with angle inc:
           ZRT -> LQT

    :type z: :class:`~numpy.ndarray`
    :param z: Data of the Z component of the seismogram.
    :type n: :class:`~numpy.ndarray`
    :param n: Data of the North component of the seismogram.
    :type e: :class:`~numpy.ndarray`
    :param e: Data of the East component of the seismogram.
    :type ba: float
    :param ba: The back azimuth from station to source in degrees.
    :type inc: float
    :param inc: The inclination of the ray at the station in degrees.
    :return: L-, Q- and T-component of seismogram.
    """
    if len(z) != len(n) or len(z) != len(e):
        raise TypeError("Z, North and East component have different length!?!")
    if ba < 0 or ba > 360:
        raise ValueError("Back Azimuth should be between 0 and 360 degrees!")
    if inc < 0 or inc > 360:
        raise ValueError("Inclination should be between 0 and 360 degrees!")
    ba *= 2 * pi / 360
    inc *= 2 * pi / 360
    l = z * cos(inc) - n * sin(inc) * cos(ba) - e * sin(inc) * sin(ba)
    q = z * sin(inc) + n * cos(inc) * cos(ba) + e * cos(inc) * sin(ba)
    t = n * sin(ba) - e * cos(ba)
    return l, q, t


def rotate_LQT_ZNE(l, q, t, ba, inc):
    """
    Rotates all components of a seismogram.

    The components will be rotated from LQT to ZNE.
    This is the inverse transformation of the transformation described
    in :func:`rotate_ZNE_LQT`.
    """
    if len(l) != len(q) or len(l) != len(t):
        raise TypeError("L, Q and T component have different length!?!")
    if ba < 0 or ba > 360:
        raise ValueError("Back Azimuth should be between 0 and 360 degrees!")
    if inc < 0 or inc > 360:
        raise ValueError("Inclination should be between 0 and 360 degrees!")
    ba *= 2 * pi / 360
    inc *= 2 * pi / 360
    z = l * cos(inc) + q * sin(inc)
    n = -l * sin(inc) * cos(ba) + q * cos(inc) * cos(ba) + t * sin(ba)
    e = -l * sin(inc) * sin(ba) + q * cos(inc) * sin(ba) - t * cos(ba)
    return z, n, e


def _dip_azimuth2ZSE_base_vector(dip, azimuth):
    """
    Helper function converting a vector described with azimuth and dip of unit
    length to a vector in the ZSE (vertical, south, east) base.

    The definition of azimuth and dip is according to the SEED reference
    manual, as are the following examples (they use rounding for small
    numerical inaccuracies - also positive and negative zero are treated as
    equal):

    >>> r = lambda x: np.array([_i if _i != -0.0 else 0.0\
        for _i in np.round(x, 10)])
    >>> r(_dip_azimuth2ZSE_base_vector(-90, 0)) #doctest: +NORMALIZE_WHITESPACE
    array([ 1., 0., 0.])
    >>> r(_dip_azimuth2ZSE_base_vector(90, 0)) #doctest: +NORMALIZE_WHITESPACE
    array([-1., 0., 0.])
    >>> r(_dip_azimuth2ZSE_base_vector(0, 0)) #doctest: +NORMALIZE_WHITESPACE
    array([ 0., -1., 0.])
    >>> r(_dip_azimuth2ZSE_base_vector(0, 180)) #doctest: +NORMALIZE_WHITESPACE
    array([ 0., 1., 0.])
    >>> r(_dip_azimuth2ZSE_base_vector(0, 90)) #doctest: +NORMALIZE_WHITESPACE
    array([ 0., 0., 1.])
    >>> r(_dip_azimuth2ZSE_base_vector(0, 270)) #doctest: +NORMALIZE_WHITESPACE
    array([ 0., 0., -1.])
    """
    # Convert both to radian.
    dip = np.deg2rad(dip)
    azimuth = np.deg2rad(azimuth)

    # Define the rotation axis for the dip.
    c1 = 0.0
    c2 = 0.0
    c3 = -1.0
    # Now the dip rotation matrix.
    dip_rotation_matrix = np.cos(dip) * \
        np.matrix(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))) + \
        (1 - np.cos(dip)) * np.matrix(((c1 * c1, c1 * c2, c1 * c3),
                                         (c2 * c1, c2 * c2, c2 * c3),
                                         (c3 * c1, c3 * c2, c3 * c3))) + \
        np.sin(dip) * np.matrix(((0, -c3, c2), (c3, 0, -c1), (-c2, c1, 0)))
    # Do the same for the azimuth.
    c1 = -1.0
    c2 = 0.0
    c3 = 0.0
    azimuth_rotation_matrix = np.cos(azimuth) * \
        np.matrix(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))) + \
        (1 - np.cos(azimuth)) * np.matrix(((c1 * c1, c1 * c2, c1 * c3),
                                         (c2 * c1, c2 * c2, c2 * c3),
                                         (c3 * c1, c3 * c2, c3 * c3))) + \
        np.sin(azimuth) * np.matrix(((0, -c3, c2), (c3, 0, -c1), (-c2, c1, 0)))

    # Now simply rotate a north pointing unit vector with both matrixes.
    temp = np.array([azimuth_rotation_matrix.dot([0.0, -1.0, 0.0])])[0][0]
    return np.array(dip_rotation_matrix.dot(temp))[0]


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
