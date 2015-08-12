# -*- coding: utf-8 -*-
"""
obspy.geodetics - Various geodetic utilities for ObsPy.
=======================================================

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import sys

from obspy.core.util.deprecation_helpers import \
    DynamicAttributeImportRerouteModule

from .base import (calc_vincenty_inverse, degrees2kilometers, gps2dist_azimuth,
                   kilometer2degrees, locations2degrees)
from .flinnengdahl import FlinnEngdahl


# Remove once 0.11 has been released.
sys.modules[__name__] = DynamicAttributeImportRerouteModule(
    name=__name__, doc=__doc__, locs=locals(),
    import_map={},
    function_map={
        "calcVincentyInverse": "obspy.geodetics.base.calc_vincenty_inverse",
        "gps2DistAzimuth": "obspy.geodetics.base.gps2dist_azimuth"
    })

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
