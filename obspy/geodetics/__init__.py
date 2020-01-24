# -*- coding: utf-8 -*-
"""
obspy.geodetics - Various geodetic utilities for ObsPy
======================================================

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .base import (calc_vincenty_inverse, degrees2kilometers,
                   gps2dist_azimuth, inside_geobounds,
                   kilometer2degrees, kilometers2degrees, locations2degrees)
from .flinnengdahl import FlinnEngdahl


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
