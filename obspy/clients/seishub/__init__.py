# -*- coding: utf-8 -*-
"""
obspy.clients.seishub - SeisHub database client for ObsPy
=========================================================
The obspy.clients.seishub package contains a client for the seismological
database SeisHub (http://www.seishub.org).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from .client import Client


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
