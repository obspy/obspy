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


def bullenRadialSlowness(layer, p, radius_of_planet, check=True):
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
    :param radius_of_planet: The radius of the planet to use, in km.
    :type radius_of_planet: float
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
        p = p * np.ones(length, dtype=np.float_)

    clibtau.bullen_radial_slowness_inner_loop(
        layer, p, time, dist, radius_of_planet, length)

    if check and (np.any(time < 0) or np.any(np.isnan(time)) or
                  np.any(dist < 0) or np.any(np.isnan(dist))):
        raise SlownessModelError("timedist.time or .dist < 0 or Nan")

    return time, dist


def bullenDepthFor(layer, ray_param, radius_of_planet, check=True):
    """
    Finds the depth for a ray parameter within this layer.

    Uses a Bullen interpolant, Ar^B. Special case for ``botP == 0`` or
    ``botDepth == radius_of_planet`` as these cause division by 0; use linear
    interpolation in this case.

    The ``layer`` and ``ray_param`` parameters must be either 0-D, or both of
    the same shape.

    :param layer: The layer(s) to check.
    :type layer: :class:`~numpy.ndarray` (shape = (1, ), dtype =
        :const:`SlownessLayer`)
    :param ray_param: The ray parameter(s) to use for calculation, in s/km.
    :type ray_param: float
    :param radius_of_planet: The radius (in km) of the planet to use.
    :type radius_of_planet: float

    :returns: The depth (in km) for the specified ray parameter.
    :rtype: float
    """
    ldim = np.ndim(layer)
    pdim = np.ndim(ray_param)
    if ldim == 1 and pdim == 0:
        ray_param = ray_param * np.ones(layer.shape, dtype=np.float_)
        depth = np.zeros(shape=layer.shape, dtype=np.float_)
    elif ldim == 0 and pdim == 1:
        layer = layer * np.ones(ray_param.shape, dtype=SlownessLayer)
        depth = np.zeros(shape=ray_param.shape, dtype=np.float_)
    elif ldim == pdim and (ldim == 0 or layer.shape == ray_param.shape):
        if ldim == 0:
            # Make array-like to work with NumPy < 1.9.
            layer = np.array([layer], dtype=SlownessLayer)
            ray_param = np.array([ray_param])
        depth = np.zeros(shape=layer.shape, dtype=np.float_)
    else:
        raise TypeError('Either layer or ray_param must be 0D, or they must '
                        'have the same shape.')

    valid = (layer['topP'] - ray_param) * (ray_param - layer['botP']) >= 0
    if not check or np.all(valid):
        leftover = np.ones_like(depth, dtype=np.bool_)

        # Easy cases for 0 thickness layer, or ray parameter found at
        # top or bottom.
        mask = layer['topDepth'] == layer['botDepth']
        depth[mask] = layer['botDepth'][mask]
        leftover &= ~mask

        mask = leftover & (layer['topP'] == ray_param)
        depth[mask] = layer['topDepth'][mask]
        leftover &= ~mask

        mask = leftover & (layer['botP'] == ray_param)
        depth[mask] = layer['botDepth'][mask]
        leftover &= ~mask

        mask = leftover & (
            (layer['botP'] != 0) & (layer['botDepth'] != radius_of_planet))
        if np.any(mask):
            topP_mask = layer['topP'][mask]
            botP_mask = layer['botP'][mask]
            topDepth_mask = layer['topDepth'][mask]
            botDepth_mask = layer['botDepth'][mask]
            ray_param_mask = ray_param[mask]

            B = np.divide(np.log(topP_mask / botP_mask),
                          np.log((radius_of_planet - topDepth_mask) /
                                 (radius_of_planet - botDepth_mask)))
            with np.errstate(over='ignore'):
                denom = np.power(radius_of_planet - topDepth_mask, B)
            A = np.divide(topP_mask, denom)

            tempDepth = np.empty_like(A)
            mask2 = (A != 0) & (B != 0)
            tempDepth[mask2] = radius_of_planet - np.exp(
                1.0 / B[mask2] * np.log(np.divide(ray_param_mask[mask2],
                                                  A[mask2])))
            # or equivalent (maybe better stability?):
            # tempDepth = radius_of_planet - math.pow(ray_param_mask/A, 1/B)

            # Overflow. Use linear interpolation.
            tempDepth[~mask2] = (
                (botDepth_mask[~mask2] - topDepth_mask[~mask2]) /
                (botP_mask[~mask2] - topP_mask[~mask2]) *
                (ray_param_mask[~mask2] - topP_mask[~mask2])
            ) + topDepth_mask[~mask2]

            # Check if slightly outside layer due to rounding or
            # numerical instability:
            mask2 = ((topDepth_mask > tempDepth) &
                     (tempDepth > topDepth_mask - 0.000001))
            tempDepth[mask2] = topDepth_mask[mask2]
            mask2 = ((botDepth_mask < tempDepth) &
                     (tempDepth < botDepth_mask + 0.000001))
            tempDepth[mask2] = botDepth_mask[mask2]

            mask2 = ((tempDepth < 0) | np.isnan(tempDepth) |
                     np.isinf(tempDepth) |
                     (tempDepth < topDepth_mask) |
                     (tempDepth > botDepth_mask))
            # Numerical instability in power law calculation? Try a
            # linear interpolation if the layer is small (<5km).
            small_layer = botDepth_mask[mask2] - topDepth_mask[mask2] > 5
            if np.any(small_layer):
                if check:
                    raise SlownessModelError(
                        "Calculated depth is outside layer, negative, or NaN.")
                else:
                    tempDepth[mask2][small_layer] = np.nan

            linear = (
                (botDepth_mask[mask2] - topDepth_mask[mask2]) /
                (botP_mask[mask2] - topP_mask[mask2]) *
                (ray_param_mask[mask2] - topP_mask[mask2])
            ) + topDepth_mask[mask2]
            outside_layer = small_layer & (
                linear < 0 | np.isnan(linear) | np.isinf(linear))
            if np.any(outside_layer):
                if check:
                    raise SlownessModelError(
                        "Calculated depth is outside layer, negative, or NaN.")
                else:
                    tempDepth[mask2][outside_layer] = np.nan
            tempDepth[mask2] = linear

            # Check for tempDepth just above topDepth or below bottomDepth.
            mask2 = ((tempDepth < topDepth_mask) &
                     (topDepth_mask - tempDepth < 1e-10))
            tempDepth[mask2] = topDepth_mask[mask2]
            mask2 = ((tempDepth > botDepth_mask) &
                     (tempDepth - botDepth_mask < 1e-10))
            tempDepth[mask2] = botDepth_mask[mask2]

            depth[mask] = tempDepth
            leftover &= ~mask

        # Special case for the centre of the planet, since Ar^B might
        # blow up at r = 0.
        mask = leftover & (layer['topP'] != layer['botP'])
        depth[mask] = (layer['botDepth'][mask] +
                       (ray_param[mask] - layer['botP'][mask]) *
                       (layer['topDepth'][mask] - layer['botDepth'][mask]) /
                       (layer['topP'][mask] - layer['botP'][mask]))
        leftover &= ~mask

        # weird case, return botDepth??
        depth[leftover] = layer['botDepth'][leftover]

        # Make sure invalid cases are left out
        depth[~valid] = np.nan

        # NumPy < 1.9 compatibility.
        if ldim == 0 and pdim == 0:
            return depth[0]
        else:
            return depth
    else:
        raise SlownessModelError(
            "Ray parameter is not contained within this slowness layer.")


def evaluateAtBullen(layer, depth, radius_of_planet):
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
    :param radius_of_planet: The radius of the planet to use, in km.
    :type radius_of_planet: float
    """
    topP = layer['topP']
    botP = layer['botP']
    topDepth = layer['topDepth']
    botDepth = layer['botDepth']
    # Could do some safeguard asserts...
    assert not botDepth > radius_of_planet
    assert not (topDepth - depth) * (depth - botDepth) < 0
    if depth == topDepth:
        return topP
    elif depth == botDepth:
        return botP
    else:
        B = np.divide(math.log(np.divide(topP, botP)),
                      math.log(np.divide((radius_of_planet - topDepth),
                                         (radius_of_planet - botDepth))))
        ADenominator = pow((radius_of_planet - topDepth), B)
        A = topP / ADenominator
        answer = A * pow((radius_of_planet - depth), B)
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


def create_from_vlayer(vLayer, isPWave, radius_of_planet, isSpherical=True):
    """
    Compute the slowness layer from a velocity layer.

    :param vLayer: The velocity layer to convert.
    :type vLayer: :class:`numpy.ndarray`, dtype = :const:`VelocityLayer`
    :param isPWave: Whether this velocity layer is for compressional/P
         (``True``) or shear/S (``False``) waves.
    :type isPWave: bool
    :param radius_of_planet: The radius of the planet to use, in km.
    :type radius_of_planet: float
    :param isSpherical: Whether the model is spherical. Non-spherical models
        are not currently supported.
    :type isSpherical: bool
    """
    ret = np.empty(shape=vLayer.shape, dtype=SlownessLayer)
    ret['topDepth'] = vLayer['topDepth']
    ret['botDepth'] = vLayer['botDepth']
    waveType = ('p' if isPWave else 's')
    if isSpherical:
        ret['topP'] = (radius_of_planet - ret['topDepth']) / \
                       evaluateVelocityAtTop(vLayer, waveType)

        bot_depth = ret["botDepth"]
        bot_vel = evaluateVelocityAtBottom(vLayer, waveType)

        if bot_depth == radius_of_planet and bot_vel == 0.0:
            ret['botP'] = np.inf
        else:
            ret['botP'] = (radius_of_planet - bot_depth) / bot_vel
    else:
        raise NotImplementedError("no flat models yet")
    return ret
