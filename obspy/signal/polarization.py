# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: polarization.py
#   Author: Conny Hammer
#    Email: conny.hammer@geo.uni-potsdam.de
#
# Copyright (C) 2008-2012 Conny Hammer
# ------------------------------------------------------------------
"""
Polarization Analysis

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from scipy import signal
import numpy as np


def eigval(datax, datay, dataz, fk, normf=1):
    """
    Polarization attributes of a signal.

    Computes the rectilinearity, the planarity and the eigenvalues of the given
    data which can be windowed or not.
    The time derivatives are calculated by central differences and the
    parameter ``fk`` describes the coefficients of the used polynomial. The
    values of ``fk`` depend on the order of the derivative you want to
    calculate. If you do not want to use derivatives you can simply
    use [1, 1, 1, 1, 1] for ``fk``.

    The algorithm is mainly based on the paper by [Jurkevics1988]_. The rest is
    just the numerical differentiation by central differences (carried out by
    the routine :func:`scipy.signal.lfilter(data, 1, fk)`).

    :type datax: :class:`~numpy.ndarray`
    :param datax: Data of x component. Note this is most usefull with
        windowed data, represented by a 2 dimensional array. First
        dimension window number, second dimension the actualy data.
    :type datay: :class:`~numpy.ndarray`
    :param datay: Data of y component. See also note in datax.
    :type dataz: :class:`~numpy.ndarray`
    :param dataz: Data of z component. See also note in datax.
    :type fk: list
    :param fk: Coefficients of polynomial used for calculating the time
        derivatives.
    :param normf: Factor for normalization.
    :return: **leigenv1, leigenv2, leigenv3, rect, plan, dleigenv, drect,
        dplan** - Smallest eigenvalue, Intermediate eigenvalue, Largest
        eigenvalue, Rectilinearity, Planarity, Time derivative of eigenvalues,
        time derivative of rectilinearity, Time derivative of planarity.
    """
    # function is made for windowed (two dimensional input).
    # However be nice and allow one dimensional input, see #919
    datax = np.atleast_2d(datax)
    datay = np.atleast_2d(datay)
    dataz = np.atleast_2d(dataz)
    covmat = np.zeros([3, 3])
    leigenv1 = np.zeros(datax.shape[0], dtype=np.float64)
    leigenv2 = np.zeros(datax.shape[0], dtype=np.float64)
    leigenv3 = np.zeros(datax.shape[0], dtype=np.float64)
    dleigenv = np.zeros([datax.shape[0], 3], dtype=np.float64)
    rect = np.zeros(datax.shape[0], dtype=np.float64)
    plan = np.zeros(datax.shape[0], dtype=np.float64)
    i = 0
    for i in range(datax.shape[0]):
        covmat[0][0] = np.cov(datax[i, :], rowvar=False)
        covmat[0][1] = covmat[1][0] = np.cov(datax[i, :], datay[i, :],
                                             rowvar=False)[0, 1]
        covmat[0][2] = covmat[2][0] = np.cov(datax[i, :], dataz[i, :],
                                             rowvar=False)[0, 1]
        covmat[1][1] = np.cov(datay[i, :], rowvar=False)
        covmat[1][2] = covmat[2][1] = np.cov(dataz[i, :], datay[i, :],
                                             rowvar=False)[0, 1]
        covmat[2][2] = np.cov(dataz[i, :], rowvar=False)
        _eigvec, eigenval, _v = (np.linalg.svd(covmat))
        eigenv = np.sort(eigenval)
        leigenv1[i] = eigenv[0]
        leigenv2[i] = eigenv[1]
        leigenv3[i] = eigenv[2]
        rect[i] = 1 - ((eigenv[1] + eigenv[0]) / (2 * eigenv[2]))
        plan[i] = 1 - ((2 * eigenv[0]) / (eigenv[1] + eigenv[2]))
    leigenv1 = leigenv1 / normf
    leigenv2 = leigenv2 / normf
    leigenv3 = leigenv3 / normf

    leigenv1_add = np.append(
        np.append([leigenv1[0]] * (np.size(fk) // 2), leigenv1),
        [leigenv1[np.size(leigenv1) - 1]] * (np.size(fk) // 2))
    dleigenv1 = signal.lfilter(fk, 1, leigenv1_add)
    dleigenv[:, 0] = dleigenv1[len(fk) - 1:]
    # dleigenv1 = dleigenv1[np.size(fk) // 2:(np.size(dleigenv1) -
    #        np.size(fk) / 2)]

    leigenv2_add = np.append(
        np.append(
            [leigenv2[0]] * (np.size(fk) // 2),
            leigenv2), [leigenv2[np.size(leigenv2) - 1]] * (np.size(fk) // 2))
    dleigenv2 = signal.lfilter(fk, 1, leigenv2_add)
    dleigenv[:, 1] = dleigenv2[len(fk) - 1:]
    # dleigenv2 = dleigenv2[np.size(fk) // 2:(np.size(dleigenv2) -
    #        np.size(fk) / 2)]

    leigenv3_add = np.append(
        np.append(
            [leigenv3[0]] * (np.size(fk) // 2), leigenv3),
        [leigenv3[np.size(leigenv3) - 1]] * (np.size(fk) // 2))
    dleigenv3 = signal.lfilter(fk, 1, leigenv3_add)
    dleigenv[:, 2] = dleigenv3[len(fk) - 1:]
    # dleigenv3 = dleigenv3[np.size(fk) // 2:(np.size(dleigenv3) -
    #        np.size(fk) / 2)]

    rect_add = np.append(
        np.append([rect[0]] * (np.size(fk) // 2), rect),
        [rect[np.size(rect) - 1]] * (np.size(fk) // 2))
    drect = signal.lfilter(fk, 1, rect_add)
    drect = drect[len(fk) - 1:]
    # drect = drect[np.size(fk) // 2:(np.size(drect3) - np.size(fk) // 2)]

    plan_add = np.append(
        np.append([plan[0]] * (np.size(fk) // 2), plan),
        [plan[np.size(plan) - 1]] * (np.size(fk) // 2))
    dplan = signal.lfilter(fk, 1, plan_add)
    dplan = dplan[len(fk) - 1:]
    # dplan = dplan[np.size(fk) // 2:(np.size(dplan) - np.size(fk) // 2)]

    return leigenv1, leigenv2, leigenv3, rect, plan, dleigenv, drect, dplan
