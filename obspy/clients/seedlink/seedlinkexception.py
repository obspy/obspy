# -*- coding: utf-8 -*-
"""
SeedLinkException.

Part of Python implementation of libslink of Chad Trabant and
JSeedLink of Anthony Lomax

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""


class SeedLinkException(Exception):
    """
    SeedLink error.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
