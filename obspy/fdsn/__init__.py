# -*- coding: utf-8 -*-
"""
obspy.fdsn - FDSN Web service client for ObsPy
==============================================
The obspy.fdsn package contains a client to access web servers that implement
the FDSN web service definitions (http://www.fdsn.org/webservices/).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from client import Client


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
