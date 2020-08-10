# -*- coding: utf-8 -*-
"""
Functionality for dealing with a single velocity layer.
"""
import numpy as np


#: The VelocityLayer dtype stores a single layer. An entire velocity model is
#: implemented as an array of layers. The elements are:
#:
#: * ``top_depth``: The top depth of the layer.
#: * ``bot_depth``: The bottom depth of the layer.
#: * ``top_p_velocity``: The compressional (P) wave velocity at the top.
#: * ``bot_p_velocity``: The compressional (P) wave velocity at the bottom.
#: * ``top_s_velocity``: The shear (S) wave velocity at the top.
#: * ``bot_s_velocity``: The shear (S) wave velocity at the bottom.
#: * ``top_density``: The density at the top.
#: * ``bot_density``: The density at the bottom.
#: * ``top_qp``: The P wave attenuation at the top.
#: * ``bot_qp``: The P wave attenuation at the bottom.
#: * ``top_qs``: The S wave attenuation at the top.
#: * ``bot_qs``: The S wave attenuation at the bottom.
VelocityLayer = np.dtype([
    ('top_depth', np.float_),
    ('bot_depth', np.float_),
    ('top_p_velocity', np.float_),
    ('bot_p_velocity', np.float_),
    ('top_s_velocity', np.float_),
    ('bot_s_velocity', np.float_),
    ('top_density', np.float_),
    ('bot_density', np.float_),
    ('top_qp', np.float_),
    ('bot_qp', np.float_),
    ('top_qs', np.float_),
    ('bot_qs', np.float_),
])


def evaluate_velocity_at_bottom(layer, prop):
    """
    Evaluate material properties at bottom of a velocity layer.

    .. seealso:: :func:`evaluate_velocity_at_top`, :func:`evaluate_velocity_at`

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
        return layer['bot_p_velocity']
    elif prop == "s":
        return layer['bot_s_velocity']
    elif prop in "rd":
        return layer['bot_density']
    raise ValueError("Unknown material property, use p, s, or d.")


def evaluate_velocity_at_top(layer, prop):
    """
    Evaluate material properties at top of a velocity layer.

    .. seealso:: :func:`evaluate_velocity_at_bottom`,
        :func:`evaluate_velocity_at`

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
        return layer['top_p_velocity']
    elif prop == "s":
        return layer['top_s_velocity']
    elif prop in "rd":
        return layer['top_density']
    raise ValueError("Unknown material property, use p, s, or d.")


def evaluate_velocity_at(layer, depth, prop):
    """
    Evaluate material properties at some depth in a velocity layer.

    .. seealso:: :func:`evaluate_velocity_at_top`,
        :func:`evaluate_velocity_at_bottom`

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
    thick = layer['bot_depth'] - layer['top_depth']
    prop = prop.lower()
    if prop == "p":
        slope = (layer['bot_p_velocity'] - layer['top_p_velocity']) / thick
        return slope * (depth - layer['top_depth']) + layer['top_p_velocity']
    elif prop == "s":
        slope = (layer['bot_s_velocity'] - layer['top_s_velocity']) / thick
        return slope * (depth - layer['top_depth']) + layer['top_s_velocity']
    elif prop in "rd":
        slope = (layer['bot_density'] - layer['top_density']) / thick
        return slope * (depth - layer['top_depth']) + layer['top_density']
    raise ValueError("Unknown material property, use p, s, or d.")
