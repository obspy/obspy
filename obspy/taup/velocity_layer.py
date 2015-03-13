#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functionality for dealing with a single velocity layer.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import numpy as np


DEFAULT_DENSITY = 2.6
DEFAULT_QP = 1000.0
DEFAULT_QS = 2000.0

#: The VelocityLayer dtype stores a single layer. An entire velocity model is
#: implemented as an array of layers. The elements are:
#:
#: * ``topDepth``: The top depth of the layer.
#: * ``botDepth``: The bottom depth of the layer.
#: * ``topPVelocity``: The compressional (P) wave velocity at the top.
#: * ``botPVelocity``: The compressional (P) wave velocity at the bottom.
#: * ``topSVelocity``: The shear (S) wave velocity at the top.
#: * ``botSVelocity``: The shear (S) wave velocity at the bottom.
#: * ``topDensity``: The density at the top.
#: * ``botDensity``: The density at the bottom.
#: * ``topQp``: The P wave attenuation at the top.
#: * ``botQp``: The P wave attenuation at the bottom.
#: * ``topQs``: The S wave attenuation at the top.
#: * ``botQs``: The S wave attenuation at the bottom.
VelocityLayer = np.dtype([
    (native_str('topDepth'), np.float_),
    (native_str('botDepth'), np.float_),
    (native_str('topPVelocity'), np.float_),
    (native_str('botPVelocity'), np.float_),
    (native_str('topSVelocity'), np.float_),
    (native_str('botSVelocity'), np.float_),
    (native_str('topDensity'), np.float_),
    (native_str('botDensity'), np.float_),
    (native_str('topQp'), np.float_),
    (native_str('botQp'), np.float_),
    (native_str('topQs'), np.float_),
    (native_str('botQs'), np.float_),
])


def evaluateVelocityAtBottom(layer, prop):
    """
    Evaluate material properties at bottom of a velocity layer.

    .. seealso:: :func:`evaluateVelocityAtTop`, :func:`evaluateVelocityAt`

    :param layer: The velocity layer to use for evaluation.
    :type layer: :class:`~numpy.ndarray`, dtype = :py:const:`.VelocityLayer`
    :param prop: The material property to evaluate. One of:

        * ``p``
            Compressional (P) velocity (km/s)
        * ``s``
            Shear (S) velocity (km/s)
        * ``r`` or ``d``
            Density (g/cm^3)
    :type prop: str

    :returns: The value of the material property requested.
    :rtype: :class:`~numpy.ndarray` (dtype = :class:`float`, shape equivalent
        to ``layer``)
    """
    prop = prop.lower()
    if prop == "p":
        return layer['botPVelocity']
    elif prop == "s":
        return layer['botSVelocity']
    elif prop in "rd":
        return layer['botDensity']
    raise ValueError("Unknown material property, use p, s, or d.")


def evaluateVelocityAtTop(layer, prop):
    """
    Evaluate material properties at top of a velocity layer.

    .. seealso:: :func:`evaluateVelocityAtBottom`, :func:`evaluateVelocityAt`

    :param layer: The velocity layer to use for evaluation.
    :type layer: :class:`~numpy.ndarray`, dtype = :py:const:`.VelocityLayer`
    :param prop: The material property to evaluate. One of:

        * ``p``
            Compressional (P) velocity (km/s)
        * ``s``
            Shear (S) velocity (km/s)
        * ``r`` or ``d``
            Density (g/cm^3)
    :type prop: str

    :returns: The value of the material property requested.
    :rtype: :class:`~numpy.ndarray` (dtype = :class:`float`, shape equivalent
        to ``layer``)
    """
    prop = prop.lower()
    if prop == "p":
        return layer['topPVelocity']
    elif prop == "s":
        return layer['topSVelocity']
    elif prop in "rd":
        return layer['topDensity']
    raise ValueError("Unknown material property, use p, s, or d.")


def evaluateVelocityAt(layer, depth, prop):
    """
    Evaluate material properties at some depth in a velocity layer.

    .. seealso:: :func:`evaluateVelocityAtTop`,
        :func:`evaluateVelocityAtBottom`

    :param layer: The velocity layer to use for evaluation.
    :type layer: :class:`~numpy.ndarray`, dtype = :py:const:`.VelocityLayer`
    :param depth: The depth at which the material property should be
        evaluated. Must be within the bounds of the layer or results will be
        undefined.
    :type depth: float
    :param prop: The material property to evaluate. One of:

        * ``p``
            Compressional (P) velocity (km/s)
        * ``s``
            Shear (S) velocity (km/s)
        * ``r`` or ``d``
            Density (g/cm^3)
    :type prop: str

    :returns: The value of the material property requested.
    :rtype: :class:`~numpy.ndarray` (dtype = :class:`float`, shape equivalent
        to ``layer``)
    """
    thick = layer['botDepth'] - layer['topDepth']
    prop = prop.lower()
    if prop == "p":
        slope = (layer['botPVelocity'] - layer['topPVelocity']) / thick
        return slope * (depth - layer['topDepth']) + layer['topPVelocity']
    elif prop == "s":
        slope = (layer['botSVelocity'] - layer['topSVelocity']) / thick
        return slope * (depth - layer['topDepth']) + layer['topSVelocity']
    elif prop in "rd":
        slope = (layer['botDensity'] - layer['topDensity']) / thick
        return slope * (depth - layer['topDepth']) + layer['topDensity']
    raise ValueError("Unknown material property, use p, s, or d.")
