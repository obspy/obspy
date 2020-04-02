# -*- coding: utf-8 -*-
"""
Library name handling for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""


class ObsPyDeprecationWarning(UserWarning):
    """
    Make a custom deprecation warning as deprecation warnings or hidden by
    default since Python 2.7 and 3.2 and we really want users to notice these.
    """
    pass
