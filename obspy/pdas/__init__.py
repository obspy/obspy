# -*- coding: utf-8 -*-
"""
obspy.pdas - PDAS file read support for ObsPy
=============================================

The obspy.pdas package contains methods in order to read files in the PDAS file
format.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

"""
from obspy.pdas.core import readPDAS, isPDAS


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
