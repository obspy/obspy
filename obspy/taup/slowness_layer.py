#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions acting on slowness layers.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import math

import numpy as np

from .c_wrappers import clibtau
from .helper_classes import SlownessLayer, SlownessModelError
from .velocity_layer import evaluateVelocityAtBottom, evaluateVelocityAtTop


def bullenRadialSlowness(layer, p, radiusOfEarth, check=True):
    """
    Calculate time and distance increments of a spherical ray.

    The time and distance (in radians) increments accumulated by a ray of
    spherical ray parameter p when passing through this layer. Note that this
    gives half of the true range and time increments since there will be both
    an upgoing and a downgoing path. Here we use the Mohorovicic or Bullen
    law: p=A*r^B

    The ``layer`` and ``p`` parameters must be either 0-D, or both of the same
    shape.

    :param layer: The layer(s) in which to calculate the increments.
    :type layer: :class:`~numpy.ndarray`, dtype = :const:`SlownessLayer`
    :param p: The spherical ray paramater to use for calculation, in s/km.
    :type p: :class:`~numpy.ndarray`, dtype = :class:`float`
    :param radiusOfEarth: The radius of the Earth to use, in km.
    :type radiusOfEarth: float
    :param check: Check that the calculated results are not invalid. This
        check may be disabled if the layers requested are expected not to
        include the specified ray.
    :type check: bool

    :returns: Time (in s) and distance (in rad) increments.
    :rtype: tuple of :class:`~numpy.ndarray`
    """
    ldim = np.ndim(layer)
    pdim = np.ndim(p)
    if ldim == 1 and pdim == 0:
        time = np.zeros(shape=layer.shape, dtype=np.float_)
        dist = np.zeros(shape=layer.shape, dtype=np.float_)
    elif ldim == 0 and pdim == 1:
        time = np.zeros(shape=p.shape, dtype=np.float_)
        dist = np.zeros(shape=p.shape, dtype=np.float_)
    elif ldim == pdim and (ldim == 0 or layer.shape == p.shape):
        time = np.zeros(shape=layer.shape, dtype=np.float_)
        dist = np.zeros(shape=layer.shape, dtype=np.float_)
    else:
        raise TypeError('Either layer or p must be 0D, or they must have '
                        'the same shape.')

    length = len(time)
    if isinstance(p, np.float_):
        p = np.array([p] * length, dtype=np.float_)

    clibtau.bullen_radial_slowness_inner_loop(
        layer, p, time, dist, radiusOfEarth, length)

    if check and (np.any(time < 0) or np.any(np.isnan(time)) or
                  np.any(dist < 0) or np.any(np.isnan(dist))):
        raise SlownessModelError("timedist.time or .dist < 0 or Nan")

    return time, dist


def bullenDepthFor(layer, ray_param, radiusOfEarth):
    """
    Finds the depth for a ray parameter within this layer.

    Uses a Bullen interpolant, Ar^B. Special case for ``botP == 0`` or
    ``botDepth == radiusOfEarth`` as these cause division by 0; use linear
    interpolation in this case.

    :param layer: The layer to check.
    :type layer: :class:`~numpy.ndarray` (shape = (1, ), dtype =
        :const:`SlownessLayer`)
    :param ray_param: The ray parameter to use for calculation, in s/km.
    :type ray_param: float
    :param radiusOfEarth: The radius (in km) of the Earth to use.
    :type radiusOfEarth: float

    :returns: The depth (in km) for the specified ray parameter.
    :rtype: float
    """
    if (layer['topP'] - ray_param) * (ray_param - layer['botP']) >= 0:
        # Easy cases for 0 thickness layer, or ray parameter found at
        # top or bottom.
        if layer['topDepth'] == layer['botDepth']:
            return layer['botDepth']
        if layer['topP'] == ray_param:
            return layer['topDepth']
        if layer['botP'] == ray_param:
            return layer['botDepth']
        if layer['botP'] != 0 and layer['botDepth'] != radiusOfEarth:
            B = np.divide(math.log(layer['topP'] / layer['botP']),
                          math.log((radiusOfEarth - layer['topDepth']) /
                          (radiusOfEarth - layer['botDepth'])))
            # This is a cludge but it's needed to mimic the Java behaviour.
            try:
                denom = math.pow((radiusOfEarth - layer['topDepth']), B)
            except OverflowError:
                denom = np.inf
            A = np.divide(layer['topP'], denom)
            if A != 0 and B != 0:
                tempDepth = radiusOfEarth - math.exp(
                    1.0 / B * math.log(np.divide(ray_param, A)))
                # or equivalent (maybe better stability?):
                # tempDepth = radiusOfEarth - math.pow(ray_param/A, 1/B)
            else:
                # Overflow. Use linear interpolation.
                tempDepth = ((layer['botDepth'] - layer['topDepth']) /
                             (layer['botP'] - layer['topP']) *
                             (ray_param - layer['topP'])) + layer['topDepth']
            # Check if slightly outside layer due to rounding or
            # numerical instability:
            if layer['topDepth'] > tempDepth > layer['topDepth'] - 0.000001:
                tempDepth = layer['topDepth']
            if layer['botDepth'] < tempDepth < layer['botDepth'] + 0.000001:
                tempDepth = layer['botDepth']
            if tempDepth < 0 \
                    or math.isnan(tempDepth) \
                    or math.isinf(tempDepth) \
                    or tempDepth < layer['topDepth'] \
                    or tempDepth > layer['botDepth']:
                # Numerical instability in power law calculation? Try a
                # linear interpolation if the layer is small (<5km).
                if layer['botDepth'] - layer['topDepth'] < 5:
                    linear = ((layer['botDepth'] - layer['topDepth']) /
                              (layer['botP'] - layer['topP']) *
                              (ray_param - layer['topP']) + layer['topDepth'])
                    if linear >= 0 \
                            and not math.isnan(linear) \
                            and not math.isinf(linear):
                        return linear
                raise SlownessModelError(
                    "Calculated depth is outside layer, negative, or NaN.")
            # Check for tempDepth just above topDepth or below bottomDepth.
            if tempDepth < layer['topDepth'] \
                    and layer['topDepth'] - tempDepth < 1e-10:
                return layer['topDepth']
            if tempDepth > layer['botDepth'] \
                    and tempDepth - layer['botDepth'] < 1e-10:
                return layer['botDepth']
            return tempDepth
        else:
            # Special case for the centre of the Earth, since Ar^B might
            #  blow up at r = 0.
            if layer['topP'] != layer['botP']:
                return (layer['botDepth'] + (ray_param - layer['botP']) *
                        (layer['topDepth'] - layer['botDepth']) /
                        (layer['topP'] - layer['botP']))
            else:
                # weird case, return botDepth??
                return layer['botDepth']
    else:
        raise SlownessModelError(
            "Ray parameter is not contained within this slowness layer.")


def evaluateAtBullen(layer, depth, radiusOfEarth):
    """
    Find the slowness at the given depth.

    Note that this method assumes a Bullen type of slowness interpolation,
    i.e., p(r) = a*r^b. This will produce results consistent with a tau model
    that uses this interpolant, but it may differ slightly from going directly
    to the velocity model. Also, if the tau model is generated using another
    interpolant, linear for instance, then the result may not be consistent
    with the tau model.

    :param layer: The layer to use for the calculation.
    :type layer: :class:`numpy.ndarray`, dtype = :const:`SlownessLayer`
    :param depth: The depth (in km) to use for the calculation. It must be
        contained within the provided ``layer`` or else results are undefined.
    :type depth: float
    :param radiusOfEarth: The radius of the Earth to use, in km.
    :type radiusOfEarth: float
    """
    topP = layer['topP']
    botP = layer['botP']
    topDepth = layer['topDepth']
    botDepth = layer['botDepth']
    # Could do some safeguard asserts...
    assert not botDepth > radiusOfEarth
    assert not (topDepth - depth) * (depth - botDepth) < 0
    if depth == topDepth:
        return topP
    elif depth == botDepth:
        return botP
    else:
        B = np.divide(math.log(np.divide(topP, botP)),
                      math.log(np.divide((radiusOfEarth - topDepth),
                                         (radiusOfEarth - botDepth))))
        ADenominator = pow((radiusOfEarth - topDepth), B)
        A = topP / ADenominator
        answer = A * pow((radiusOfEarth - depth), B)
        if answer < 0 or math.isnan(answer) or math.isinf(answer):
            # numerical instability in power law calculation???
            # try a linear interpolation if the layer is small ( <2 km)
            # or if denominator of A is infinity as we probably overflowed
            # the double in that case.
            if botDepth - topDepth < 2 \
                    or math.isinf(ADenominator) \
                    or botP == 0:
                linear = (botP - topP) / (botDepth - topDepth) * \
                         (depth - topDepth) + topP
                if linear < 0 \
                        or math.isinf(linear) \
                        or math.isnan(linear):
                    pass
                else:
                    return linear
            raise SlownessModelError(
                "Calculated Slowness is NaN or negative!")
    return answer


def create_from_vlayer(vLayer, isPWave, radiusOfEarth=6371,
                       isSpherical=True):
    """
    Compute the slowness layer from a velocity layer.

    :param vLayer: The velocity layer to convert.
    :type vLayer: :class:`numpy.ndarray`, dtype = :const:`VelocityLayer`
    :param isPWave: Whether this velocity layer is for compressional/P
         (``True``) or shear/S (``False``) waves.
    :type isPWave: bool
    :param radiusOfEarth: The radius of the Earth to use, in km.
    :type radiusOfEarth: float
    :param isSpherical: Whether the model is spherical. Non-spherical models
        are not currently supported.
    :type isSpherical: bool
    """
    ret = np.empty(shape=vLayer.shape, dtype=SlownessLayer)
    ret['topDepth'] = vLayer['topDepth']
    ret['botDepth'] = vLayer['botDepth']
    waveType = ('p' if isPWave else 's')
    if isSpherical:
        ret['topP'] = (radiusOfEarth - ret['topDepth']) / \
            evaluateVelocityAtTop(vLayer, waveType)
        ret['botP'] = (radiusOfEarth - ret['botDepth']) / \
            evaluateVelocityAtBottom(vLayer, waveType)
    else:
        raise NotImplementedError("no flat models yet")
    return ret
