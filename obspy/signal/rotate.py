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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from math import cos, pi, sin

import numpy as np

from obspy.core.util.decorator import deprecated


@deprecated("rotate_NE_RT() has been renamed to rotate_ne_rt().")
def rotate_NE_RT(*args, **kwargs):  # noqa
    return rotate_ne_rt(*args, **kwargs)


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
    r = e * sin((ba + 180) * 2 * pi / 360) + n * cos((ba + 180) * 2 * pi / 360)
    t = e * cos((ba + 180) * 2 * pi / 360) - n * sin((ba + 180) * 2 * pi / 360)
    return r, t


@deprecated("rotate_RT_NE() has been renamed to rotate_rt_ne().")
def rotate_RT_NE(*args, **kwargs):  # noqa
    return rotate_rt_ne(*args, **kwargs)


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


@deprecated("rotate_ZNE_LQT() has been renamed to rotate_zne_lqt().")
def rotate_ZNE_LQT(*args, **kwargs):  # noqa
    return rotate_zne_lqt(*args, **kwargs)


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
    ba *= 2 * pi / 360
    inc *= 2 * pi / 360
    l = z * cos(inc) - n * sin(inc) * cos(ba) - e * sin(inc) * sin(ba)
    q = z * sin(inc) + n * cos(inc) * cos(ba) + e * cos(inc) * sin(ba)
    t = n * sin(ba) - e * cos(ba)
    return l, q, t


@deprecated("rotate_LQT_ZNE() has been renamed to rotate_lqt_zne().")
def rotate_LQT_ZNE(*args, **kwargs):  # noqa
    return rotate_lqt_zne(*args, **kwargs)


def rotate_lqt_zne(l, q, t, ba, inc):
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
    ba *= 2 * pi / 360
    inc *= 2 * pi / 360
    z = l * cos(inc) + q * sin(inc)
    n = -l * sin(inc) * cos(ba) + q * cos(inc) * cos(ba) + t * sin(ba)
    e = -l * sin(inc) * sin(ba) + q * cos(inc) * sin(ba) - t * cos(ba)
    return z, n, e


def _dip_azimuth2zse_base_vector(dip, azimuth):
    """
    Helper function converting a vector described with azimuth and dip of unit
    length to a vector in the ZSE (Vertical, South, East) base.

    The definition of azimuth and dip is according to the SEED reference
    manual, as are the following examples (they use rounding for small
    numerical inaccuracies - also positive and negative zero are treated as
    equal):

    >>> r = lambda x: np.array([_i if _i != -0.0 else 0.0\
        for _i in np.round(x, 10)])
    >>> r(_dip_azimuth2zse_base_vector(-90, 0)) #doctest: +NORMALIZE_WHITESPACE
    array([ 1., 0., 0.])
    >>> r(_dip_azimuth2zse_base_vector(90, 0)) #doctest: +NORMALIZE_WHITESPACE
    array([-1., 0., 0.])
    >>> r(_dip_azimuth2zse_base_vector(0, 0)) #doctest: +NORMALIZE_WHITESPACE
    array([ 0., -1., 0.])
    >>> r(_dip_azimuth2zse_base_vector(0, 180)) #doctest: +NORMALIZE_WHITESPACE
    array([ 0., 1., 0.])
    >>> r(_dip_azimuth2zse_base_vector(0, 90)) #doctest: +NORMALIZE_WHITESPACE
    array([ 0., 0., 1.])
    >>> r(_dip_azimuth2zse_base_vector(0, 270)) #doctest: +NORMALIZE_WHITESPACE
    array([ 0., 0., -1.])
    """
    dip = np.deg2rad(dip)
    azimuth = np.deg2rad(azimuth)

    return np.array([-np.sin(dip),
                     -np.cos(azimuth) * np.cos(dip),
                     np.sin(azimuth) * np.cos(dip)])


@deprecated("rotate2ZNE() has been renamed to rotate2zne().")
def rotate2ZNE(*args, **kwargs):  # noqa
    return rotate2zne(*args, **kwargs)


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

    :rtype: Tuple of three NumPy arrays.
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

    # Internally works in Vertical, South, and East components; a right handed
    # coordinate system.

    # Define the base vectors of the old base in terms of the new base vectors.
    base_vector_1 = _dip_azimuth2zse_base_vector(dip_1, azimuth_1)
    base_vector_2 = _dip_azimuth2zse_base_vector(dip_2, azimuth_2)
    base_vector_3 = _dip_azimuth2zse_base_vector(dip_3, azimuth_3)

    # Build transformation matrix.
    _t = np.matrix([base_vector_1, base_vector_2, base_vector_3]).transpose()

    if not inverse:
        x, y, z = np.dot(_t, [data_1, data_2, data_3])
        y *= -1
    else:
        x, y, z = np.dot(np.linalg.inv(_t), [data_1, -data_2, data_3])

    # Replace all negative zeros. These might confuse some further
    # processing programs.
    x = np.array(x).ravel()
    x[x == -0.0] = 0
    y = np.array(y).ravel()
    y[y == -0.0] = 0
    z = np.array(z).ravel()
    z[z == -0.0] = 0

    return x, y, z


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
