# -*- coding: utf-8 -*-
"""
obspy.taup - Travel time calculation tool
=========================================

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    Unknown
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

# Convenience imports.
from obspy.taup.taup import getTravelTimes, travelTimePlot

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
