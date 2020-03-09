# -*- coding: utf-8 -*-
"""
obspy.io.nordic - Nordic file format support for ObsPy
======================================================

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
class NordicParsingError(Exception):
    """
    Internal general error for IO operations in obspy.core.io.nordic.
    """
    def __init__(self, value):
        self.value = value
