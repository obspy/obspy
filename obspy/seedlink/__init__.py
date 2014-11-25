# -*- coding: utf-8 -*-
"""
obspy.seedlink - SeedLink client for ObsPy
==========================================

The obspy.seedlink module provides an implementation of the SeedLink client
protocol for ObsPy.

A higher level client is provided in the :mod:`obspy.seedlink.easyseedlink`
module.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

The obspy.seedlink development has been supported by the NERA project ["Network
of European Research Infrastructures for Earthquake Risk Assessment and
Mitigation" under the European Community's Seventh Framework Programme
(FP7/2007-2013) grant agreement n° 262330] and implemented within the
activities of the JRA2/WP12 "Tools for real-time seismology, acquisition and
mining".
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
