package obspy.gse2
==================

Copyright
---------
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

    Copyright (c) 2009 by:
        * Moritz Beyreuther
        * Stefan Stange


Overview
--------
    obspy.gse2 - Read & Write Seismograms, Format GSE2.

    This module contains Python wrappers for gse_functions - The GSE2 library
    of Stefan Stange (http://www.orfeus-eu.org/Software/softwarelib.html#gse).
    Currently CM6 compressed GSE2 files are supported, this should be 
    sufficient for most cases. Gse_functions are written in C and interfaced 
    via python-ctypes.

    For more information visit http://www.obspy.org.


Dependencies
------------
    * numpy
    * setuptools
    * obspy.core
