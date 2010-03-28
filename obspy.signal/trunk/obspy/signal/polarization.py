#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: polarization.py
#   Author: Conny Hammer
#    Email: conny@geo.uni-potsdam.de
#
# Copyright (C) 2008-2010 Conny Hammer
#-------------------------------------------------------------------
"""
Polarization Analysis

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np


def eigval(datax, datay, dataz, normf=1):
    """
    Polarization attributes of a signal:

    Computes the rectilinearity, the planarity and the eigenvalues of the given
    data which can be windowed or not.

    :param datax: Data of x component, type numpy.ndarray.
    :param datay: Data of y component, type numpy.ndarray.
    :param dataz: Data of z component, type numpy.ndarray.
    :param normf: Factor for normalization.
    :return leigenv1: Smallest eigenvalue.
    :return leigenv2: Intermediate eigenvalue.
    :return leigenv3: Largest eigenvalue.
    :return rect: Rectilinearity.
    :return plan: Planarity.
    """
    covmat = np.zeros([3, 3])
    leigenv1 = np.zeros(datax.shape[0], dtype='float64')
    leigenv2 = np.zeros(datax.shape[0], dtype='float64')
    leigenv3 = np.zeros(datax.shape[0], dtype='float64')
    rect = np.zeros(datax.shape[0], dtype='float64')
    plan = np.zeros(datax.shape[0], dtype='float64')
    i = 0
    for i in xrange(datax.shape[0]):
        covmat[0][0] = np.cov(datax[i, :], rowvar=False)
        covmat[0][1] = covmat[1][0] = np.cov(datax[i, :], datay[i, :],
                                             rowvar=False)[0, 1]
        covmat[0][2] = covmat[2][0] = np.cov(datax[i, :], dataz[i, :],
                                             rowvar=False)[0, 1]
        covmat[1][1] = np.cov(datay[i, :], rowvar=False)
        covmat[1][2] = covmat[2][1] = np.cov(dataz[i, :], datay[i, :],
                                             rowvar=False)[0, 1]
        covmat[2][2] = np.cov(dataz[i, :], rowvar=False)
        eigenv = np.sort(np.linalg.eigvals(covmat))
        leigenv1[i] = eigenv[0]
        leigenv2[i] = eigenv[1]
        leigenv3[i] = eigenv[2]
        rect[i] = 1 - ((eigenv[1] + eigenv[0]) / (2 * eigenv[2]))
        plan[i] = 1 - ((2 * eigenv[0]) / (eigenv[1] + eigenv[2]))
    leigenv1 = leigenv1 / normf
    leigenv2 = leigenv2 / normf
    leigenv3 = leigenv3 / normf
    return leigenv1, leigenv2, leigenv3, rect, plan
