# -*- coding: utf-8 -*-
"""
Signal processing routines for seismology. 

Capabilities include filtering, triggering, rotation, instrument
correction and coordinate transformations.


GNU General Public License (GPL)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301, USA.
"""

from filter import *
from rotate import *
from invsim import cosTaper
from invsim import pazToFreqResp
from invsim import pazToFreqResp2
from invsim import specInv
from invsim import seisSim
import seismometer
from trigger import *
