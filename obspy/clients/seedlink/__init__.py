# -*- coding: utf-8 -*-
"""
obspy.clients.seedlink - SeedLink client for ObsPy
==================================================

The obspy.clients.seedlink module provides an implementation of the SeedLink
client protocol for ObsPy.

For simple requests of finite time windows see
:class:`~obspy.clients.seedlink.basic_client.Client`. To work with continuous
data streams see
:class:`~obspy.clients.seedlink.easyseedlink.EasySeedLinkClient`, or for
lower-level packet handling see
:class:`~obspy.clients.seedlink.slclient.SLClient`.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

The obspy.clients.seedlink development has been supported by the NERA project
["Network of European Research Infrastructures for Earthquake Risk Assessment
and Mitigation" under the European Community's Seventh Framework Programme
(FP7/2007-2013) grant agreement n° 262330] and implemented within the
activities of the JRA2/WP12 "Tools for real-time seismology, acquisition and
mining".
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .basic_client import Client  # NOQA
from .slclient import SLClient  # NOQA
from .easyseedlink import EasySeedLinkClient  # NOQA

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
