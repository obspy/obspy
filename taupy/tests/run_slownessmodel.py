#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to run the model e.g. for a debugger.
"""

from taupy.SlownessModel import SlownessModel
from taupy.VelocityModel import VelocityModel

testmod = SlownessModel(VelocityModel.readVelocityFile('./data/iasp91.tvel'))
