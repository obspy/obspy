#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: rotate.py
#  Purpose: Various Seismogram Rotation Functions
#   Author: Tobias Megies, Tom Richter
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2009-2012 Tobias Megies, Tom Richter
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
