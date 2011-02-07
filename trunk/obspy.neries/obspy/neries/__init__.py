# -*- coding: utf-8 -*-
"""
obspy.neries - NERIES web service client for ObsPy
==================================================
The obspy.neries package contains a client for the Seismic Data Portal which was
developed under the European Commission-funded NERIES project. The Portal
provides a single point of access to diverse, distributed European earthquake
data provided in a unique joint initiative by observatories and research
institutes in and around Europe.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats and
seismological signal processing routines which allow the manipulation of
seismological time series (see  Beyreuther et. al. 2010). The goal of the ObsPy
project is to facilitate rapid application development for seismology. 

For more information visit http://www.obspy.org.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.util import _getVersionString
from client import Client


__version__ = _getVersionString("obspy.neries")
