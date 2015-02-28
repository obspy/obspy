#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Package for storage and manipulation of seismic earth models.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import numpy as np


DEFAULT_DENSITY = 2.6
DEFAULT_QP = 1000.0
DEFAULT_QS = 2000.0

"""
The VelocityLayer dtype stores and manipulates a single layer. An
entire velocity model is implemented as an array of layers.
"""
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


def evaluateVelocityAtBottom(layer, materialProperty):
    materialProperty = materialProperty.lower()
    if materialProperty == "p":
        return layer['botPVelocity']
    elif materialProperty == "s":
        return layer['botSVelocity']
    elif materialProperty in "rd":
        return layer['botDensity']
    raise ValueError("Unknown material property, use p, s, or d.")


def evaluateVelocityAtTop(layer, materialProperty):
    materialProperty = materialProperty.lower()
    if materialProperty == "p":
        return layer['topPVelocity']
    elif materialProperty == "s":
        return layer['topSVelocity']
    elif materialProperty in "rd":
        return layer['topDensity']
    raise ValueError("Unknown material property, use p, s, or d.")


def evaluateVelocityAt(layer, depth, materialProperty):
    thick = layer['botDepth'] - layer['topDepth']
    materialProperty = materialProperty.lower()
    if materialProperty == "p":
        slope = (layer['botPVelocity'] - layer['topPVelocity']) / thick
        return slope * (depth - layer['topDepth']) + layer['topPVelocity']
    elif materialProperty == "s":
        slope = (layer['botSVelocity'] - layer['topSVelocity']) / thick
        return slope * (depth - layer['topDepth']) + layer['topSVelocity']
    elif materialProperty in "rd":
        slope = (layer['botDensity'] - layer['topDensity']) / thick
        return slope * (depth - layer['topDepth']) + layer['topDensity']
    raise ValueError("Unknown material property, use p, s, or d.")
