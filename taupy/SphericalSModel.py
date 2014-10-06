#!/usr/bin/env python
# -*- coding: utf-8 -*-

from taupy.SlownessModel import SlownessModel


class SphericalSModel(SlownessModel):
    """
    Dummy class, not needed atm. This class is meant to provide methods
    specific to spherical slowness models, as opposed to the methods in
    SlownessModel which apply to spherical and flat models. However,
    flat models are useless and not even implemented in the java code,
    so this class is not needed; any methods are given in SlownessModel.
    """
    pass
