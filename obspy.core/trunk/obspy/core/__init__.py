# -*- coding: utf-8 -*-
"""
Core class of ObsPy, Python for Seismological Observatories

This class contains common methods and classes for ObsPy. It includes
a UTCDateTime, a Stats, the Stream and Trace classes and methods
for reading seismograms and checking which obspy modules are installed.


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

from core import *
from util import *
import util
import parser
from testing import runTests
