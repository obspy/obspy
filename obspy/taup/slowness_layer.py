# -*- coding: utf-8 -*-
"""
Functions acting on slowness layers.
"""
import math

import numpy as np

from .c_wrappers import clibtau
from .helper_classes import SlownessLayer, SlownessModelError
from .velocity_layer import (evaluate_velocity_at_bottom,
                             evaluate_velocity_at_top)


def bullen_radial_slowness(layer, p, radius_of_planet, check=True):
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


def bullen_depth_for(layer, ray_param, radius_of_planet, check=True):
    """
    Finds the depth for a ray parameter within this layer.

    Uses a Bullen interpolant, Ar^B. Special case for ``bot_p == 0`` or
    ``bot_depth == radius_of_planet`` as these cause division by 0; use linear
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

    valid = (layer['top_p'] - ray_param) * (ray_param - layer['bot_p']) >= 0
    if not check or np.all(valid):
        leftover = np.ones_like(depth, dtype=np.bool_)

        # Easy cases for 0 thickness layer, or ray parameter found at
        # top or bottom.
        mask = layer['top_depth'] == layer['bot_depth']
        depth[mask] = layer['bot_depth'][mask]
        leftover &= ~mask

        mask = leftover & (layer['top_p'] == ray_param)
        depth[mask] = layer['top_depth'][mask]
        leftover &= ~mask

        mask = leftover & (layer['bot_p'] == ray_param)
        depth[mask] = layer['bot_depth'][mask]
        leftover &= ~mask

        mask = leftover & (
            (layer['bot_p'] != 0) & (layer['bot_depth'] != radius_of_planet))
        if np.any(mask):
            top_p_mask = layer['top_p'][mask]
            bot_p_mask = layer['bot_p'][mask]
            top_depth_mask = layer['top_depth'][mask]
            bot_depth_mask = layer['bot_depth'][mask]
            ray_param_mask = ray_param[mask]

            b = np.divide(np.log(top_p_mask / bot_p_mask),
                          np.log((radius_of_planet - top_depth_mask) /
                                 (radius_of_planet - bot_depth_mask)))
            with np.errstate(over='ignore'):
                denom = np.power(radius_of_planet - top_depth_mask, b)
            a = np.divide(top_p_mask, denom)

            temp_depth = np.empty_like(a)
            mask2 = (a != 0) & (b != 0)
            temp_depth[mask2] = radius_of_planet - np.exp(
                1.0 / b[mask2] * np.log(np.divide(ray_param_mask[mask2],
                                                  a[mask2])))
            # or equivalent (maybe better stability?):
            # tempDepth = radius_of_planet - math.pow(ray_param_mask/A, 1/B)

            # Overflow. Use linear interpolation.
            temp_depth[~mask2] = (
                (bot_depth_mask[~mask2] - top_depth_mask[~mask2]) /
                (bot_p_mask[~mask2] - top_p_mask[~mask2]) *
                (ray_param_mask[~mask2] - top_p_mask[~mask2])
            ) + top_depth_mask[~mask2]

            # Check if slightly outside layer due to rounding or
            # numerical instability:
            mask2 = ((top_depth_mask > temp_depth) &
                     (temp_depth > top_depth_mask - 0.000001))
            temp_depth[mask2] = top_depth_mask[mask2]
            mask2 = ((bot_depth_mask < temp_depth) &
                     (temp_depth < bot_depth_mask + 0.000001))
            temp_depth[mask2] = bot_depth_mask[mask2]

            mask2 = ((temp_depth < 0) | np.isnan(temp_depth) |
                     np.isinf(temp_depth) |
                     (temp_depth < top_depth_mask) |
                     (temp_depth > bot_depth_mask))
            # Numerical instability in power law calculation? Try a
            # linear interpolation if the layer is small (<5km).
            small_layer = bot_depth_mask[mask2] - top_depth_mask[mask2] > 5
            if np.any(small_layer):
                if check:
                    raise SlownessModelError(
                        "Calculated depth is outside layer, negative, or NaN.")
                else:
                    temp_depth[mask2][small_layer] = np.nan

            linear = (
                (bot_depth_mask[mask2] - top_depth_mask[mask2]) /
                (bot_p_mask[mask2] - top_p_mask[mask2]) *
                (ray_param_mask[mask2] - top_p_mask[mask2])
            ) + top_depth_mask[mask2]
            outside_layer = small_layer & (
                linear < 0 | np.isnan(linear) | np.isinf(linear))
            if np.any(outside_layer):
                if check:
                    raise SlownessModelError(
                        "Calculated depth is outside layer, negative, or NaN.")
                else:
                    temp_depth[mask2][outside_layer] = np.nan
            temp_depth[mask2] = linear

            # Check for tempDepth just above top_depth or below bottomDepth.
            mask2 = ((temp_depth < top_depth_mask) &
                     (top_depth_mask - temp_depth < 1e-10))
            temp_depth[mask2] = top_depth_mask[mask2]
            mask2 = ((temp_depth > bot_depth_mask) &
                     (temp_depth - bot_depth_mask < 1e-10))
            temp_depth[mask2] = bot_depth_mask[mask2]

            depth[mask] = temp_depth
            leftover &= ~mask

        # Special case for the centre of the planet, since Ar^B might
        # blow up at r = 0.
        mask = leftover & (layer['top_p'] != layer['bot_p'])
        depth[mask] = (layer['bot_depth'][mask] +
                       (ray_param[mask] - layer['bot_p'][mask]) *
                       (layer['top_depth'][mask] - layer['bot_depth'][mask]) /
                       (layer['top_p'][mask] - layer['bot_p'][mask]))
        leftover &= ~mask

        # weird case, return bot_depth??
        depth[leftover] = layer['bot_depth'][leftover]

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


def evaluate_at_bullen(layer, depth, radius_of_planet):
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
    top_p = layer['top_p']
    bot_p = layer['bot_p']
    top_depth = layer['top_depth']
    bot_depth = layer['bot_depth']
    # Could do some safeguard asserts...
    assert not bot_depth > radius_of_planet
    assert not (top_depth - depth) * (depth - bot_depth) < 0
    if depth == top_depth:
        return top_p
    elif depth == bot_depth:
        return bot_p
    else:
        try:
            # The power law calculation has some stability issues with very
            # small layers. Catch them here to trigger the fallback
            # computations later on.
            with np.errstate(all="raise"):
                b = math.log(top_p / bot_p) / \
                    math.log((radius_of_planet - top_depth) /
                             (radius_of_planet - bot_depth))
                a_denominator = pow((radius_of_planet - top_depth), b)
                a = top_p / a_denominator
                answer = a * pow((radius_of_planet - depth), b)
        except FloatingPointError:
            answer = np.nan
        if answer < 0 or math.isnan(answer) or math.isinf(answer):
            # numerical instability in power law calculation???
            # try a linear interpolation if the layer is small ( <2 km)
            # or if denominator of A is infinity as we probably overflowed
            # the double in that case.
            if bot_depth - top_depth < 2 \
                    or math.isinf(a_denominator) \
                    or bot_p == 0:
                linear = (bot_p - top_p) / (bot_depth - top_depth) * \
                         (depth - top_depth) + top_p
                if linear < 0 \
                        or math.isinf(linear) \
                        or math.isnan(linear):
                    pass
                else:
                    return linear
            raise SlownessModelError(
                "Calculated Slowness is NaN or negative!")
    return answer


def create_from_vlayer(v_layer, is_p_wave, radius_of_planet,
                       is_spherical=True):
    """
    Compute the slowness layer from a velocity layer.

    :param v_layer: The velocity layer to convert.
    :type v_layer: :class:`numpy.ndarray`, dtype = :const:`VelocityLayer`
    :param is_p_wave: Whether this velocity layer is for compressional/P
         (``True``) or shear/S (``False``) waves.
    :type is_p_wave: bool
    :param radius_of_planet: The radius of the planet to use, in km.
    :type radius_of_planet: float
    :param is_spherical: Whether the model is spherical. Non-spherical models
        are not currently supported.
    :type is_spherical: bool
    """
    ret = np.empty(shape=v_layer.shape, dtype=SlownessLayer)
    ret['top_depth'] = v_layer['top_depth']
    ret['bot_depth'] = v_layer['bot_depth']
    wave_type = ('p' if is_p_wave else 's')
    if is_spherical:
        ret['top_p'] = (radius_of_planet - ret['top_depth']) / \
            evaluate_velocity_at_top(v_layer, wave_type)

        bot_depth = ret["bot_depth"]
        bot_vel = evaluate_velocity_at_bottom(v_layer, wave_type)

        if bot_depth.shape:
            if bot_depth[-1] == radius_of_planet and bot_vel[-1] == 0.0:
                bot_depth[-1] = 1.0
        ret['bot_p'] = (radius_of_planet - bot_depth) / bot_vel
    else:
        raise NotImplementedError("no flat models yet")
    return ret
