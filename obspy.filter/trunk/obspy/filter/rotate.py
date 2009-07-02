#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: rotate.py
#  Purpose: Various Seismogram Rotation Functions
#   Author: Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2009 Tobias Megies
#---------------------------------------------------------------------
"""
Various Seismogram Rotation Functions

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""

from numpy import array, sin, cos, pi

def rotate_NE_RT(n,e,ba):
  """Rotates horizontal components of a seismogram:

  The North- and East-Component of a seismogram will be rotated in Radial and Transversal
  Component. The angle is given as the back-azimuth, that is defined as the angle measured
  between the vector pointing from the station to the source and the vector pointing from
  the station to the north.
  
  @param n: Data of the North component of the seismogram.
  @param e: Data of the East component of the seismogram.
  @param ba: The back azimuth from station to source in degrees.
  @return: Radial and Transversal component of seismogram.
  """
  if n.__len__()!=e.__len__():
      raise TypeError("North and East component have different length!?!")
  if ba<0 or ba>360:
      raise ValueError("Back Azimuth should be between 0 and 360 degrees!")
  r=e*sin((ba+180)*2*pi/360)+n*cos((ba+180)*2*pi/360)
  t=e*cos((ba+180)*2*pi/360)-n*sin((ba+180)*2*pi/360)
  return r,t
