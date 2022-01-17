#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: rotate.py
#  Purpose: Various Seismogram Rotation Functions
#   Author: Tobias Megies, Tom Richter, Lion Krischer
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2009-2013 Tobias Megies, Tom Richter, Lion Krischer
# --------------------------------------------------------------------
"""
Various Seismogram Rotation Functions

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import warnings
from math import cos, sin, radians

import numpy as np


def rotate_ne_rt(n, e, ba):
    """
    Rotates horizontal components of a seismogram.

    The North- and East-Component of a seismogram will be rotated in Radial
    and Transversal Component. The angle is given as the back-azimuth, that is
    defined as the angle measured between the vector pointing from the station
    to the source and the vector pointing from the station to the North.

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
    ba = radians(ba)
    r = - e * sin(ba) - n * cos(ba)
    t = - e * cos(ba) + n * sin(ba)
    return r, t


def rotate_rt_ne(n, e, ba):
    """
    Rotates horizontal components of a seismogram.

    Rotates from radial and transversal components to North and East
    components.

    This is the inverse transformation of the transformation described
    in :func:`rotate_ne_rt`.
    """
    ba = 360.0 - ba
    return rotate_ne_rt(n, e, ba)


def rotate_zne_lqt(z, n, e, ba, inc):
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
    ba = radians(ba)
    inc = radians(inc)
    l = z * cos(inc) - n * sin(inc) * cos(ba) - e * sin(inc) * sin(ba)  # NOQA
    q = z * sin(inc) + n * cos(inc) * cos(ba) + e * cos(inc) * sin(ba)  # NOQA
    t = n * sin(ba) - e * cos(ba)  # NOQA
    return l, q, t


def rotate_lqt_zne(l, q, t, ba, inc):  # NOQA
    """
    Rotates all components of a seismogram.

    The components will be rotated from LQT to ZNE.
    This is the inverse transformation of the transformation described
    in :func:`rotate_zne_lqt`.
    """
    if len(l) != len(q) or len(l) != len(t):
        raise TypeError("L, Q and T component have different length!?!")
    if ba < 0 or ba > 360:
        raise ValueError("Back Azimuth should be between 0 and 360 degrees!")
    if inc < 0 or inc > 360:
        raise ValueError("Inclination should be between 0 and 360 degrees!")
    ba = radians(ba)
    inc = radians(inc)
    z = l * cos(inc) + q * sin(inc)
    n = -l * sin(inc) * cos(ba) + q * cos(inc) * cos(ba) + t * sin(ba)
    e = -l * sin(inc) * sin(ba) + q * cos(inc) * sin(ba) - t * cos(ba)
    return z, n, e


def _dip_azimuth2zne_base_vector(dip, azimuth):
    """
    Helper function converting a vector described with azimuth and dip of unit
    length to a vector in the ZNE (Vertical, North, East) base.

    The definition of azimuth and dip is according to the SEED reference
    manual.
    """
    dip = np.deg2rad(dip)
    azimuth = np.deg2rad(azimuth)

    return np.array([-np.sin(dip),
                     np.cos(azimuth) * np.cos(dip),
                     np.sin(azimuth) * np.cos(dip)])


def rotate2zne(data_1, azimuth_1, dip_1, data_2, azimuth_2, dip_2, data_3,
               azimuth_3, dip_3, inverse=False):
    """
    Rotates an arbitrarily oriented three-component vector to ZNE.

    Each components orientation is described with a azimuth and a dip. The
    azimuth is defined as the degrees from North, clockwise and the dip is the
    defined as the number of degrees, down from horizontal. Both definitions
    are according to the SEED standard.

    The three components need not be orthogonal to each other but the
    components have to be linearly independent. The function performs a full
    base change to orthogonal Vertical, North, and East orientations.

    :param data_1: Data component 1.
    :param azimuth_1: The azimuth of component 1.
    :param dip_1: The dip of component 1.
    :param data_2: Data component 2.
    :param azimuth_2: The azimuth of component 2.
    :param dip_2: The dip of component 2.
    :param data_3: Data component 3.
    :param azimuth_3: The azimuth of component 3.
    :param dip_3: The dip of component 3.
    :param inverse: If `True`, the data arrays will be converted from ZNE to
        whatever coordinate system the azimuths and dips specify. In that
        case data_1, data_2, data_3 have to be data arrays for Z, N,
        and E and the dips and azimuths specify where to transform to.
    :type inverse: bool

    :rtype: tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`,
        :class:`numpy.ndarray`)
    :returns: The three rotated components, oriented in Z, N, and E if
        `inverse` is `False`. Otherwise they will be oriented as specified
        by the dips and azimuths.

    An input of ZNE yields an output of ZNE

    >>> rotate2zne(np.arange(3), 0, -90, np.arange(3) * 2, 0, 0, \
            np.arange(3) * 3, 90, 0) # doctest: +NORMALIZE_WHITESPACE
    (array([ 0., 1., 2.]), array([ 0., 2., 4.]), array([ 0., 3., 6.]))

    An input of ZSE yields an output of ZNE

    >>> rotate2zne(np.arange(3), 0, -90, np.arange(3) * 2, 180, 0, \
            np.arange(3) * 3, 90, 0) # doctest: +NORMALIZE_WHITESPACE
    (array([ 0., 1., 2.]), array([ 0., -2., -4.]), array([ 0., 3., 6.]))

    Mixed up components should get rotated to ZNE.

    >>> rotate2zne(np.arange(3), 0, 0, np.arange(3) * 2, 90, 0, \
            np.arange(3) * 3, 0, -90) # doctest: +NORMALIZE_WHITESPACE
    (array([ 0., 3., 6.]), array([ 0., 1., 2.]), array([ 0., 2., 4.]))
    """
    if len(set(len(i_) for i_ in (data_1, data_2, data_3))) != 1:
        msg = "All three data arrays must be of same length."
        raise ValueError(msg)

    # Define the base vectors of the old base in terms of the new base vectors.
    base_vector_1 = _dip_azimuth2zne_base_vector(dip_1, azimuth_1)
    base_vector_2 = _dip_azimuth2zne_base_vector(dip_2, azimuth_2)
    base_vector_3 = _dip_azimuth2zne_base_vector(dip_3, azimuth_3)

    # Base change matrix.
    m = np.array([base_vector_1,
                  base_vector_2,
                  base_vector_3])

    # Determinant gives the volume change of a unit cube going from one
    # basis to the next. It should neither be too small nor to large. These
    # here are arbitrary limits.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                '.*invalid value encountered in det.*')
        det = np.linalg.det(m)
    if not (1E-6 < abs(det) < 1E6):
        raise ValueError("The given directions are not linearly independent, "
                         "at least within numerical precision. Determinant "
                         "of the base change matrix: %g" % det)

    if not inverse:
        m = np.linalg.inv(m)

    z, n, e = np.dot(m, [data_1, data_2, data_3])

    # Replace all negative zeros. These might confuse some further
    # processing programs.
    z = np.array(z).ravel()
    z[z == -0.0] = 0
    n = np.array(n).ravel()
    n[n == -0.0] = 0
    e = np.array(e).ravel()
    e[e == -0.0] = 0

    return z, n, e


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
